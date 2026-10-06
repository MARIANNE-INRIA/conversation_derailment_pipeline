"""
Temporal Conversation Derailment Prediction — QLoRA Fine-tuning

Cluster-ready training script for parameter-efficient fine-tuning of
decoder-only large language models (e.g., Gemma, Mistral) for prefix-level
binary forecasting of conversation derailment.

Features
--------
- LoRA fine-tuning with native PyTorch reduced-precision weights and PEFT
- Dynamic padding and left-side truncation (preserving the most recent
conversation turns, following the recommendation of Tran et al. (2025))
- Early stopping and checkpoint resumption
- Prefix-level and conversation-level evaluation
- Automatic threshold search and Mean Horizon computation
- Reproducible training via saved configuration and fixed random seeds

Usage
-----
python train_llm_forecasting.py \
    --model_name /path/to/model \
    --train_path ./data/train_samplesLLMs.csv \
    --val_path ./data/val_samplesLLMs.csv \
    --output_dir ./checkpoints

Monitor training:
    tail -f checkpoints/train.log
"""

import os
import json
import random
import time
import logging
import argparse
import math
from hpo_common import report_epoch, threshold_grid
from training_resume import (TrainingResume, add_resume_arguments, epoch_loader,
                             install_checkpoint_signals, pause_if_requested)

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from online_alerts import (
    conversation_alert_metrics,
    replay_alerts,
    select_validation_threshold,
)

# 1. Configuration

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for conversation derailment prediction")

    # Model
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")

    # Data
    parser.add_argument("--train_path", type=str, default="./data/train_samplesLLMs.csv")
    parser.add_argument("--val_path", type=str, default="./data/val_samplesLLMs.csv")

    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of micro-batches to accumulate before optimizer step")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs with no F1 improvement)")

    # Input
    parser.add_argument("--max_length", type=int, default=1024)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.15)

    # Evaluation
    parser.add_argument("--threshold_objective", choices=["f1", "constrained"], default="constrained")
    parser.add_argument("--tau_start", type=float, default=0.05)
    parser.add_argument("--tau_end", type=float, default=0.95)
    parser.add_argument("--tau_step", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.56, help="Training threshold in constrained mode; F1 mode searches tau_start..tau_end")
    parser.add_argument(
        "--max_negative_alert_rate",
        type=float,
        default=0.10,
        help="Maximum validation proportion of negative conversations that may alert",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing for cross-entropy loss (0.0 disables it)")
    parser.add_argument(
        "--amp_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="CUDA autocast dtype. auto prefers BF16 when supported.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=100, help="Log training progress every N steps")
    parser.add_argument("--val_every", type=int, default=1, help="Run validation and best-model checks every N epochs")
    parser.add_argument("--save_best_every", type=int, default=1, help="Only save a new best checkpoint every N improvements; 1 means every improvement")
    parser.add_argument("--resume",type=str,default=None,help="Path to a batch-resumable training_resume.pt checkpoint")

    add_resume_arguments(parser)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps, args.patience) < 1:
        parser.error("epochs, batch size, accumulation and patience must be positive")
    if args.val_every != 1 or args.save_best_every != 1:
        parser.error("validation and best-checkpoint saving must run every epoch/improvement")
    threshold_grid(args.tau_start, args.tau_end, args.tau_step)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir):
    """
    Configure logging to write to BOTH the console and a file inside
    output_dir. This allows `tail -f output_dir/train.log`
    from another terminal/ssh session while the job runs unattended.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to {log_path}")

# 2. Prompt & Dataset

def build_prompt(utterances):
    """Convert a conversation prefix into the instruction prompt proposed by Tran et al. (2025)"""
    conversation = []
    for i, utt in enumerate(utterances, start=1):
        conversation.append(f"Turn {i}:")
        conversation.append(utt.strip())
        conversation.append("")

    conversation_text = "\n".join(conversation)

    prompt = (
    "Instruction:\n"
    "You are a moderator observing an ongoing conversation. "
    "Your goal is to determine whether the conversation will derail into a personal attack. "
    "Pay attention to conversational flow and speaker dynamics. "
    "Be careful—sensitive topics do not always lead to personal attacks.\n\n"
    "Conversation Transcript:\n\n"
    f"{conversation_text}\n\n"
    "Will the above conversation derail into a personal attack now or at any point in the future?\n"
    "Answer:")
    return prompt
class TemporalDataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        utterances = json.loads(row["utterances"])
        prompt = build_prompt(utterances)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding later
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int(row["label"]),
            "conversation_id": row["conversation_id"],
            "timestep": int(row["timestep"]),
            "total_turns": int(row["total_turns"]),
        }


class TemporalDataCollator:

    def __init__(self, tokenizer):
        self.padding_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, batch):
        conversation_ids = [x["conversation_id"] for x in batch]
        timesteps = [x["timestep"] for x in batch]
        total_turns = [x["total_turns"] for x in batch]

        features = [
            {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"],
                "labels": x["labels"],
            }
            for x in batch
        ]

        batch = self.padding_collator(features)
        batch["conversation_ids"] = conversation_ids
        batch["timesteps"] = timesteps
        batch["total_turns"] = total_turns
        return batch

# 3. Training / validation loops


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs, log_every, label_smoothing=0.1, gradient_accumulation_steps=1, amp_dtype=torch.float16, scaler=None, start_batch=0, loss_sum=0.0, checkpoint_callback=None):
    model.train()
    total_loss = loss_sum
    total_batches = start_batch + len(dataloader)
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    micro_step = 0

    for step, batch in enumerate(dataloader, start=start_batch + 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if device.type == "cuda" and amp_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.cross_entropy(outputs.logits.float(), labels, label_smoothing=label_smoothing)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch={epoch + 1}, step={step}. "
                "Try --amp_dtype bfloat16 or float32 and a lower --lr."
            )
        window_start = ((step - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
        window_size = min(gradient_accumulation_steps, total_batches - window_start)
        loss = loss / window_size
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        micro_step += 1
        total_loss += loss.item() * window_size

        if micro_step % gradient_accumulation_steps == 0 or step == total_batches:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            trainable_parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient at epoch={epoch + 1}, step={step}. "
                    "Use --amp_dtype float32 and a lower --lr."
                )
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not all(
                torch.isfinite(parameter).all()
                for parameter in trainable_parameters
            ):
                raise RuntimeError(
                    f"Optimizer produced non-finite LoRA weights at epoch={epoch + 1}, step={step}. "
                    "Use --amp_dtype float32 and a lower --lr."
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0
            if checkpoint_callback is not None:
                checkpoint_callback(step, total_loss)

        avg_loss = total_loss / step

        if step % log_every == 0 or step == total_batches:
            elapsed = time.time() - start_time
            avg_step_time = elapsed / step
            eta = avg_step_time * (total_batches - step)

            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f" | GPU mem {torch.cuda.memory_allocated() / 1e9:.2f}GB"

            logging.info(
                f"Epoch {epoch + 1}/{total_epochs} | Step {step}/{total_batches} | "
                f"Loss {avg_loss:.4f} | Elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m{gpu_mem}"
            )

    epoch_time = time.time() - start_time
    final_avg_loss = total_loss / total_batches
    logging.info(f"Training finished in {epoch_time / 60:.1f} minutes | Average Loss: {final_avg_loss:.4f}")
    return final_avg_loss


def validate(model, dataloader, device, amp_dtype=torch.float16):
    model.eval()
    total_loss = 0.0

    all_labels, all_probabilities = [], []
    all_conversation_ids, all_timesteps, all_total_turns = [], [], []
    all_logit_0, all_logit_1 = [], []


    with torch.no_grad():
        for batch in dataloader:
            pause_if_requested()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda" and amp_dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            loss = F.cross_entropy(logits.float(), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss: model weights are unstable.")
            total_loss += loss.item()

            probabilities = torch.softmax(logits, dim=-1)

            all_logit_0.extend(logits[:, 0].cpu().tolist())
            all_logit_1.extend(logits[:, 1].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.extend(probabilities[:, 1].cpu().tolist())
            all_conversation_ids.extend(batch["conversation_ids"])
            all_timesteps.extend(batch["timesteps"])
            all_total_turns.extend(batch["total_turns"])

           
    predictions_df = pd.DataFrame({
        "conversation_id": all_conversation_ids,
        "timestep": all_timesteps,
        "total_turns": all_total_turns,
        "label": all_labels,
        "probability": all_probabilities,
        "logit_0": all_logit_0,
        "logit_1": all_logit_1,
    }).sort_values(["conversation_id", "timestep"]).reset_index(drop=True)

    return total_loss / len(dataloader), predictions_df


def prefix_metrics(predictions_df, tau):
    preds = (predictions_df["probability"] > tau).astype(int)
    labels = predictions_df["label"]
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

# 4. Temporal forecasting evaluation (conversation-level aggregation)


def aggregate_predictions(predictions_df, tau):
    _, summary = replay_alerts(predictions_df, tau)
    return summary.rename(
        columns={
            "alerted": "prediction",
            "first_alert_timestep": "first_trigger",
        }
    )


def conversation_metrics(conversation_df):
    return {
        "accuracy": accuracy_score(conversation_df["label"], conversation_df["prediction"]),
        "precision": precision_score(conversation_df["label"], conversation_df["prediction"], zero_division=0),
        "recall": recall_score(conversation_df["label"], conversation_df["prediction"], zero_division=0),
        "f1": f1_score(conversation_df["label"], conversation_df["prediction"], zero_division=0),
    }


def find_best_f1_threshold(predictions_df, start=0.05, end=0.95, step=0.01):
    grouped = predictions_df.groupby("conversation_id").agg(
        label=("label", "first"), probability=("probability", "max")
    )
    rows = []
    for tau in threshold_grid(start, end, step):
        grouped["prediction"] = (grouped["probability"] > tau).astype(int)
        rows.append({"tau": tau, **conversation_metrics(grouped)})
    curve = pd.DataFrame(rows)
    return float(curve.loc[curve["f1"].idxmax(), "tau"]), curve


def find_best_threshold(predictions_df, max_negative_alert_rate):
    selected, curve, f1_threshold = select_validation_threshold(
        predictions_df,
        max_negative_alert_rate=max_negative_alert_rate,
    )
    return selected, curve.rename(columns={"threshold": "tau"}), f1_threshold


def mean_horizon(predictions_df, tau):
    """Mean Horizon â€” higher is better."""
    horizons = []
    predictions_df = predictions_df.sort_values(["conversation_id", "timestep"])

    for conversation_id, group in predictions_df.groupby("conversation_id"):
        label = int(group["label"].iloc[0])
        if label == 0:
            continue

        probs = group["probability"].values
        triggers = probs > tau
        if not np.any(triggers):
            continue

        first_trigger = group.loc[triggers, "timestep"].iloc[0]
        last_prefix = group["timestep"].max()
        horizons.append(last_prefix - first_trigger)

    return float(np.mean(horizons)) if horizons else 0.0


def save_best_model(model, tokenizer, output_dir, metrics=None):
    """Persist the best adapter/model checkpoint and any metadata for later reuse."""
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    model.save_pretrained(best_model_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(best_model_dir)

    if metrics is not None:
        with open(os.path.join(best_model_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    best_checkpoint_path = os.path.join(output_dir, "checkpoint_best.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_f1": metrics["conversation_f1"] if metrics is not None else None,
        },
        best_checkpoint_path,
    )

    return best_model_dir

# 5. Main

def main():
    install_checkpoint_signals()
    args = parse_args()
    cfg = vars(args)

    setup_logging(cfg["output_dir"])
    logging.info(f"Config: {json.dumps(cfg, indent=2)}")

    # Save the exact config used for this run, for reproducibility
    with open(os.path.join(cfg["output_dir"], "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if device.type != "cuda" or args.amp_dtype == "float32":
        amp_dtype = torch.float32
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logging.info(f"AMP dtype: {amp_dtype} | GradScaler: {use_scaler}")

    # Tokenizer & data
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Keep the most recent turns when conversations exceed max_length
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # temporary sanity check, remove after confirming
    print("truncation_side:", tokenizer.truncation_side)
    print(build_prompt(["test message 1", "test message 2"])[:200])


    collator = TemporalDataCollator(tokenizer)

    train_dataset = TemporalDataset(cfg["train_path"], tokenizer, cfg["max_length"])
    val_dataset = TemporalDataset(cfg["val_path"], tokenizer, cfg["max_length"])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collator
    )

    # Model
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        torch_dtype=model_dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    logging.info(
        "Package versions: torch=%s, transformers=%s, peft=%s",
        torch.__version__,
        __import__("transformers").__version__,
        __import__("peft").__version__,
    )
    lora_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    lora_target_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in lora_suffixes
        and "vision_tower" not in name
        and "vision_model" not in name
    ]
    if not lora_target_modules:
        raise RuntimeError(
            "No language-model projection layers were found for LoRA. "
            "Inspect the loaded model module names before training."
        )
    logging.info("LoRA target layers: %d language projections", len(lora_target_modules))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=lora_target_modules,
        modules_to_save=["score"],
    )

    # autocast_adapter_dtype=False: the fp32 cast crashes on this torch/H100 build
    # ("no kernel image is available"); keep adapters in the base model dtype instead.
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainable_params_str = model.print_trainable_parameters()  # peft prints internally too
    logging.info(f"Model loaded on: {next(model.parameters()).device}")

    # Optimizer / scheduler 
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    num_training_steps = math.ceil(len(train_loader) / cfg["gradient_accumulation_steps"]) * cfg["epochs"]
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    resume = TrainingResume(args.resume or os.path.join(cfg["output_dir"], "training_resume.pt"),
                            model, optimizer, scheduler, scaler, args.checkpoint_interval_seconds)
    progress = resume.load() or {"next_epoch": 0, "next_batch": 0, "loss_sum": 0.0,
                                "best_metric": -1.0, "stale_epochs": 0, "history": [], "finished": False}
    start_epoch = progress["next_epoch"]
    best_conversation_f1 = progress["best_metric"]
    patience_counter = progress["stale_epochs"]
    best_save_counter = 0
    history = progress["history"]
    if history:
        pd.DataFrame(history).to_csv(os.path.join(cfg["output_dir"], "training_history.csv"), index=False)
    for row in history:
        report_epoch(row["epoch"], row["conversation_f1"])
    if progress["finished"]:
        start_epoch = args.epochs

    for epoch in range(start_epoch, args.epochs):
        logging.info("=" * 80)
        logging.info(f"Epoch {epoch + 1}/{cfg['epochs']}")
        logging.info("=" * 80)

        start_batch = progress["next_batch"] if epoch == progress["next_epoch"] else 0
        loss_sum = progress["loss_sum"] if start_batch else 0.0
        train_loader = epoch_loader(train_dataset, args.batch_size, collator, 0, args.seed, epoch, start_batch)
        def checkpoint_batch(next_batch, total_loss):
            resume.maybe_save({"next_epoch": epoch, "next_batch": next_batch, "loss_sum": total_loss,
                               "best_metric": best_conversation_f1, "stale_epochs": patience_counter,
                               "history": history, "finished": False})
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch=epoch, total_epochs=cfg["epochs"], log_every=cfg["log_every"],
            label_smoothing=cfg["label_smoothing"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            amp_dtype=amp_dtype,
            scaler=scaler, start_batch=start_batch, loss_sum=loss_sum,
            checkpoint_callback=checkpoint_batch,
        )

        # Commit all optimizer steps before potentially long validation.
        resume.maybe_save({"next_epoch": epoch, "next_batch": start_batch + len(train_loader),
                           "loss_sum": train_loss * (start_batch + len(train_loader)),
                           "best_metric": best_conversation_f1, "stale_epochs": patience_counter,
                           "history": history, "finished": False}, force=True)
        if (epoch + 1) % cfg["val_every"] == 0 or epoch == args.epochs - 1:
            val_loss, val_predictions = validate(model, val_loader, device, amp_dtype=amp_dtype)

            epoch_tau = cfg["tau"]
            if cfg["threshold_objective"] == "f1":
                epoch_tau, _ = find_best_f1_threshold(val_predictions, args.tau_start, args.tau_end, args.tau_step)
            prefix_results = prefix_metrics(val_predictions, epoch_tau)
            conversation_predictions = aggregate_predictions(val_predictions, epoch_tau)
            conversation_results = conversation_metrics(conversation_predictions)
            mh = mean_horizon(val_predictions, epoch_tau)

            loss_gap = val_loss - train_loss

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_gap": loss_gap,
                "prefix_accuracy": prefix_results["accuracy"],
                "prefix_precision": prefix_results["precision"],
                "prefix_recall": prefix_results["recall"],
                "prefix_f1": prefix_results["f1"],
                "conversation_accuracy": conversation_results["accuracy"],
                "conversation_precision": conversation_results["precision"],
                "conversation_recall": conversation_results["recall"],
                "conversation_f1": conversation_results["f1"],
                "mean_H": mh,
                "tau": epoch_tau,
            })

            pd.DataFrame(history).to_csv(os.path.join(cfg["output_dir"], "training_history.csv"), index=False)

            improved = conversation_results["f1"] > best_conversation_f1
            if improved:
                best_conversation_f1 = conversation_results["f1"]
                patience_counter = 0
                best_save_counter += 1

                if best_save_counter % cfg["save_best_every"] == 0:
                    best_model_dir = save_best_model(
                        model,
                        tokenizer,
                        cfg["output_dir"],
                        metrics={
                            "epoch": epoch + 1,
                            "conversation_f1": conversation_results["f1"],
                            "conversation_accuracy": conversation_results["accuracy"],
                            "conversation_precision": conversation_results["precision"],
                            "conversation_recall": conversation_results["recall"],
                            "mean_H": mh,
                            "tau": epoch_tau,
                        },
                    )
                    val_predictions.to_csv(os.path.join(cfg["output_dir"], "best_validation_predictions.csv"), index=False)
                    status = f"Best model saved to {best_model_dir}"
                else:
                    status = "Validation improved but best checkpoint deferred by save_best_every"
            else:
                patience_counter += 1
                status = f"No improvement ({patience_counter}/{cfg['patience']})"
        else:
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": None,
                "loss_gap": None,
                "prefix_accuracy": None,
                "prefix_precision": None,
                "prefix_recall": None,
                "prefix_f1": None,
                "conversation_accuracy": None,
                "conversation_precision": None,
                "conversation_recall": None,
                "conversation_f1": None,
                "mean_H": None,
                "tau": epoch_tau,
            })
            pd.DataFrame(history).to_csv(os.path.join(cfg["output_dir"], "training_history.csv"), index=False)
            status = "Skipped validation for this epoch"
        resume.maybe_save({"next_epoch": epoch + 1, "next_batch": 0, "loss_sum": 0.0,
                           "best_metric": best_conversation_f1, "stale_epochs": patience_counter,
                           "history": history,
                           "finished": patience_counter >= args.patience or epoch + 1 == args.epochs}, force=True)

        logging.info("Epoch Summary")
        logging.info("-" * 80)
        logging.info(f"Train Loss           : {train_loss:.4f}")
        logging.info(f"Validation Loss      : {val_loss:.4f}")
        logging.info(f"Loss Gap (val-train)  : {loss_gap:+.4f}")
        logging.info(f"Conversation F1      : {conversation_results['f1']:.4f}")
        logging.info(f"Best Conversation F1 : {best_conversation_f1:.4f}")
        logging.info(f"Mean Horizon         : {mh:.2f}")
        logging.info(status)

        logging.info("Prefix Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in prefix_results.items()))
        logging.info("Conversation Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in conversation_results.items()))

        report_epoch(epoch + 1, conversation_results["f1"])
        if patience_counter >= cfg["patience"]:
            logging.info("Early stopping triggered.")
            break

    # Final evaluation with best model 
    logging.info("=" * 80)
    logging.info("Loading best model for final evaluation")
    logging.info("=" * 80)

    best_model_dir = os.path.join(cfg["output_dir"], "best_model")

    # Best-checkpoint predictions avoid loading a second base model on the GPU.
    val_predictions = pd.read_csv(os.path.join(cfg["output_dir"], "best_validation_predictions.csv"))
    if cfg["threshold_objective"] == "f1":
        best_tau, threshold_results = find_best_f1_threshold(val_predictions, args.tau_start, args.tau_end, args.tau_step)
        f1_tau = best_tau
    else:
        best_tau, threshold_results, f1_tau = find_best_threshold(val_predictions, cfg["max_negative_alert_rate"])
    with open(os.path.join(cfg["output_dir"], "best_tau.json"), "w") as f:
        json.dump(
            {
                "best_tau": float(best_tau),
                "selection_rule": "max_conversation_f1" if cfg["threshold_objective"] == "f1" else "max_positive_alert_recall_under_negative_alert_rate",
                "max_negative_alert_rate": cfg["max_negative_alert_rate"],
                "comparison_f1_tau": float(f1_tau),
            },
            f,
            indent=2,
        )
    sample_results = prefix_metrics(val_predictions, best_tau)

    conversation_predictions = aggregate_predictions(val_predictions, best_tau)
    conversation_results = conversation_metrics(conversation_predictions)
    mh = mean_horizon(val_predictions, best_tau)

    logging.info("Final Evaluation:")
    logging.info(f"Operating threshold (validation selection): {best_tau:.6f}")
    logging.info(f"Comparison threshold (validation conversation F1): {f1_tau:.6f}")
    logging.info(f"sample_f1          : {sample_results['f1']:.4f}")
    logging.info(f"sample_acc         : {sample_results['accuracy']:.4f}")
    logging.info(f"sample_precision   : {sample_results['precision']:.4f}")
    logging.info(f"sample_recall      : {sample_results['recall']:.4f}")

    logging.info(f"conv_f1            : {conversation_results['f1']:.4f}")
    logging.info(f"conv_acc           : {conversation_results['accuracy']:.4f}")
    logging.info(f"conv_precision     : {conversation_results['precision']:.4f}")
    logging.info(f"conv_recall        : {conversation_results['recall']:.4f}")
    logging.info(f"mean_H             : {mh:.4f}")

    conversation_predictions.to_csv(os.path.join(cfg["output_dir"], "conversation_predictions.csv"), index=False)
    threshold_results.to_csv(os.path.join(cfg["output_dir"], "threshold_search.csv"), index=False)

    pd.DataFrame([{**{"conv_" + k: v for k, v in conversation_results.items()},
                   **{"prefix_" + k: v for k, v in sample_results.items()},
                   "mean_H": mh, "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0}]).to_csv(os.path.join(cfg["output_dir"], "validation_metrics.csv"), index=False)
    logging.info("Done. All outputs saved to: " + cfg["output_dir"])


if __name__ == "__main__":
    main()
