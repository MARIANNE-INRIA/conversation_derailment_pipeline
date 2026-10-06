#!/usr/bin/env python3
"""Train the RoBERTa derailment classifier from BEST_RoBERTa.ipynb.

Input files must be torch-serialized dictionaries containing input_ids,
attention_mask, labels, conversation_ids, timesteps, and total_turns.
"""

import argparse
from hpo_common import report_epoch, threshold_grid
from training_resume import (TrainingResume, add_resume_arguments, epoch_loader,
                             install_checkpoint_signals, pause_if_requested)
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, get_linear_schedule_with_warmup


class CGATemporalDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.labels = data["labels"]
        self.conversation_ids = data["conversation_ids"]
        self.timesteps = data["timesteps"]
        self.total_turns = data["total_turns"]
        print(f"Loaded {len(self.labels)} samples from {path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
            "conversation_id": self.conversation_ids[index],
            "timestep": self.timesteps[index],
            "total_turns": self.total_turns[index],
        }


def collate_fn(batch):
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.ones((len(batch), max_length), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for row_index, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[row_index, :length] = item["input_ids"]
        attention_mask[row_index, :length] = item["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
        "conversation_ids": [item["conversation_id"] for item in batch],
        "timesteps": [item["timestep"] for item in batch],
        "total_turns": [item["total_turns"] for item in batch],
    }


class RoBERTaForDerailment(nn.Module):
    def __init__(self, model_name, dropout=0.2, class_weights=None):
        super().__init__()
        # roberta-base ships cached safetensors weights, so this avoids torch.load entirely
        # without needing torch>=2.6 or any online safetensors-conversion check.
        self.roberta = RobertaModel.from_pretrained(model_name, use_safetensors=True)
        self.roberta.gradient_checkpointing_enable()
        self.roberta.config.use_cache = False
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return loss, logits


def collect_predictions(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            pause_if_requested()
            _, logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            labels = batch["labels"].tolist()
            for index, conversation_id in enumerate(batch["conversation_ids"]):
                rows.append({
                    "conversation_id": conversation_id,
                    "timestep": batch["timesteps"][index],
                    "total_turns": batch["total_turns"][index],
                    "label": int(labels[index]),
                    "probability": probabilities[index],
                })
    return pd.DataFrame(rows).sort_values(["conversation_id", "timestep"])


def conversation_outputs(predictions):
    conversations = {}
    for conversation_id, group in predictions.groupby("conversation_id"):
        conversations[conversation_id] = {
            "label": int(group["label"].iloc[0]),
            "predictions": list(zip(group["timestep"], group["probability"])),
        }
    return conversations


def metrics_for_threshold(conversations, threshold):
    labels, predictions, horizons = [], [], []
    for conversation in conversations.values():
        pairs = sorted(conversation["predictions"], key=lambda pair: pair[0])
        triggered = [probability > threshold for _, probability in pairs]
        label = conversation["label"]
        prediction = int(any(triggered))
        labels.append(label)
        predictions.append(prediction)
        if prediction and label:
            first_trigger = next(index for index, value in enumerate(triggered) if value)
            horizons.append(len(pairs) - 1 - first_trigger)
    return {
        "conv_f1": f1_score(labels, predictions, zero_division=0),
        "conv_acc": accuracy_score(labels, predictions),
        "conv_precision": precision_score(labels, predictions, zero_division=0),
        "conv_recall": recall_score(labels, predictions, zero_division=0),
        "mean_H": float(np.mean(horizons)) if horizons else 0.0,
    }


def prefix_metrics(predictions, threshold):
    predicted = (predictions["probability"] > threshold).astype(int)
    return {
        "prefix_f1": f1_score(predictions["label"], predicted, zero_division=0),
        "prefix_accuracy": accuracy_score(predictions["label"], predicted),
        "prefix_precision": precision_score(predictions["label"], predicted, zero_division=0),
        "prefix_recall": recall_score(predictions["label"], predicted, zero_division=0),
    }


def evaluate(model, loader, device, threshold):
    predictions = collect_predictions(model, loader, device)
    conversations = conversation_outputs(predictions)
    return metrics_for_threshold(conversations, threshold), conversations, predictions


def threshold_sweep(model, loader, device, start, end, step):
    conversations = conversation_outputs(collect_predictions(model, loader, device))
    rows = []
    for threshold in np.arange(start, end + 1e-9, step):
        rows.append({"tau": float(threshold), **metrics_for_threshold(conversations, threshold)})
    return pd.DataFrame(rows)


def make_loader(dataset, batch_size, shuffle, workers):
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    install_checkpoint_signals()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.56)
    parser.add_argument("--tau_start", type=float, default=0.05)
    parser.add_argument("--tau_end", type=float, default=0.95)
    parser.add_argument("--tau_step", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    add_resume_arguments(parser)
    args = parser.parse_args()
    threshold_grid(args.tau_start, args.tau_end, args.tau_step)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    train_dataset = CGATemporalDataset(args.train_path)
    val_dataset = CGATemporalDataset(args.val_path)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.workers)

    labels = np.asarray(train_dataset.labels)
    negative_count = max(int((labels == 0).sum()), 1)
    positive_count = max(int((labels == 1).sum()), 1)
    total = negative_count + positive_count
    class_weights = torch.tensor(
        [total / (2 * negative_count), total / (2 * positive_count)],
        dtype=torch.float32,
        device=device,
    )

    model = RoBERTaForDerailment(
        args.model_name,
        args.dropout,
        class_weights=class_weights,
    ).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    parameter_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_path = os.path.join(args.output_dir, "best_model.pt")
    history_path = os.path.join(args.output_dir, "training_history.csv")
    resume = TrainingResume(os.path.join(args.output_dir, "training_resume.pt"), model, optimizer,
                            scheduler, scaler, args.checkpoint_interval_seconds)
    progress = (None if args.no_resume else resume.load()) or {
        "next_epoch": 1, "next_batch": 0, "losses": [], "best_metric": -1.0,
        "stale_epochs": 0, "history": [], "finished": False}
    start_epoch, best_metric = progress["next_epoch"], progress["best_metric"]
    stale_epochs, history = progress["stale_epochs"], progress["history"]
    if history:
        pd.DataFrame(history).to_csv(history_path, index=False)
    for row in history:
        report_epoch(row["epoch"], row["conv_f1"])
    if progress["finished"]:
        start_epoch = args.epochs + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_batch = progress["next_batch"] if epoch == progress["next_epoch"] else 0
        losses = list(progress["losses"]) if start_batch else []
        train_loader = epoch_loader(train_dataset, args.batch_size, collate_fn, args.workers,
                                    args.seed, epoch, start_batch)
        def checkpoint_batch(next_batch, force=False):
            resume.maybe_save({"next_epoch": epoch, "next_batch": next_batch, "losses": losses,
                               "best_metric": best_metric, "stale_epochs": stale_epochs,
                               "history": history, "finished": False}, force=force)
        for step, batch in enumerate(train_loader, start=start_batch + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss, _ = model(
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                    batch["labels"].to(device, non_blocking=True),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            checkpoint_batch(step)

        checkpoint_batch(start_batch + len(train_loader), force=True)
        _, conversations, _ = evaluate(model, val_loader, device, args.threshold)
        validation_metrics = max(
            ({"tau": tau, **metrics_for_threshold(conversations, tau)}
             for tau in threshold_grid(args.tau_start, args.tau_end, args.tau_step)),
            key=lambda row: row["conv_f1"],
        )
        monitor = validation_metrics["conv_f1"]
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, conv_f1={monitor:.4f}")
        history.append({"epoch": epoch, "train_loss": np.mean(losses), **validation_metrics})
        if monitor > best_metric:
            best_metric, stale_epochs = monitor, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": validation_metrics, "args": vars(args)}, best_path)
            print(f"Saved best model to {best_path}")
        else:
            stale_epochs += 1
        resume.maybe_save({"next_epoch": epoch + 1, "next_batch": 0, "losses": [],
                           "best_metric": best_metric, "stale_epochs": stale_epochs, "history": history,
                           "finished": stale_epochs >= args.patience or epoch == args.epochs}, force=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        report_epoch(epoch, monitor)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without improvement")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    sweep = threshold_sweep(model, val_loader, device, args.tau_start, args.tau_end, args.tau_step)
    best_row = sweep.loc[sweep["conv_f1"].idxmax()]
    best_threshold = float(best_row["tau"])
    sweep.to_csv(os.path.join(args.output_dir, "threshold_search.csv"), index=False)
    with open(os.path.join(args.output_dir, "best_tau.json"), "w") as handle:
        json.dump({"best_tau": best_threshold}, handle, indent=2)
    final_metrics, final_conversations, final_predictions = evaluate(
        model, val_loader, device, best_threshold
    )
    final_predictions["prediction"] = (
        final_predictions["probability"] > best_threshold
    ).astype(int)
    final_predictions["correct"] = (
        final_predictions["label"] == final_predictions["prediction"]
    )
    conversation_rows = []
    for conversation_id, group in final_predictions.groupby("conversation_id"):
        label = int(group["label"].iloc[0])
        prediction = int(group["prediction"].any())
        conversation_rows.append({
            "conversation_id": conversation_id,
            "label": label,
            "prediction": prediction,
            "correct": label == prediction,
        })
    conversation_table = pd.DataFrame(conversation_rows)
    conversation_table["error_type"] = np.select(
        [
            (conversation_table["label"] == 0) & (conversation_table["prediction"] == 1),
            (conversation_table["label"] == 1) & (conversation_table["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )
    final_metrics.update(prefix_metrics(final_predictions, best_threshold))
    final_metrics["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    final_predictions.to_csv(
        os.path.join(args.output_dir, "validation_prefix_predictions.csv"),
        index=False,
    )
    final_predictions.loc[~final_predictions["correct"]].to_csv(
        os.path.join(args.output_dir, "validation_prefix_errors.csv"),
        index=False,
    )
    conversation_table.to_csv(
        os.path.join(args.output_dir, "validation_conversation_predictions.csv"),
        index=False,
    )
    conversation_table.loc[~conversation_table["correct"]].to_csv(
        os.path.join(args.output_dir, "validation_conversation_errors.csv"),
        index=False,
    )
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(args.output_dir, "validation_metrics.csv"),
        index=False,
    )
    with open(os.path.join(args.output_dir, "comparison_metadata.json"), "w") as handle:
        json.dump(
            {
                "model_family": "roberta",
                "model_name": args.model_name,
                "seed": args.seed,
                "train_path": args.train_path,
                "val_path": args.val_path,
                "selection_metric": "conv_f1",
                "best_tau": best_threshold,
            },
            handle,
            indent=2,
        )
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(json.dumps(final_metrics, indent=2))
    print(f"Training complete. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
