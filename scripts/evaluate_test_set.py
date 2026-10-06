"""
Evaluate a fine-tuned causal language model for temporal conversation derailment prediction.

This script performs inference on a held-out test set using a fine-tuned PEFT/LoRA
sequence classification model (e.g., Gemma). Each conversation prefix is converted
into an instruction-style prompt and scored independently to estimate the probability
that the conversation will eventually derail into a personal attack.

The evaluation is performed at two levels:

1. Prefix level
   - Computes the predicted probability for every conversation prefix.
   - Applies a decision threshold (selected on the validation set) to obtain
     binary prefix predictions.
   - Reports prefix-level accuracy, precision, recall, and F1.

2. Conversation level
   - Aggregates prefix predictions for each conversation.
   - A conversation is classified as positive if any prefix exceeds the selected
     probability threshold.
   - Reports conversation-level accuracy, precision, recall, and F1.
   - Computes the mean prediction horizon, defined as the average number of
     remaining turns between the first positive prediction and the end of each
     positive conversation.

Additionally, the script evaluates model performance as a function of conversation
progress (10%-100% of the conversation) to analyze early detection performance.

Inputs
------
--model_name
    Base Hugging Face model identifier.

--best_model_dir
    Directory containing the fine-tuned PEFT adapter and tokenizer.

--data_path
    CSV file containing the temporal test dataset.

--threshold
    Classification threshold selected on the validation set.

Outputs
-------
evaluation_config.csv
    Stores the decision threshold used for evaluation.

prefix_predictions.csv
    Prefix-level probabilities and predictions.

conversation_predictions.csv
    Conversation-level aggregated predictions.

progress_metrics.csv
    Performance metrics at different stages of conversation progress.
"""




import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from online_alerts import conversation_alert_metrics, replay_alerts


# Dataset


def build_prompt(utterances):
    """Convert a conversation prefix into an instruction prompt."""
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
        "Answer:"
    )

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
        encoding = self.tokenizer(
            build_prompt(json.loads(row["utterances"])),
            truncation=True,
            max_length=self.max_length,
            padding=False,
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
        self.padding = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )

    def __call__(self, batch):

        meta = {
            "conversation_ids": [x["conversation_id"] for x in batch],
            "timesteps": [x["timestep"] for x in batch],
            "total_turns": [x["total_turns"] for x in batch],
        }

        features = [
            {
                "input_ids": x["input_ids"],
                "attention_mask": x["attention_mask"],
                "labels": x["labels"],
            }
            for x in batch
        ]

        batch = self.padding(features)

        batch.update(meta)

        return batch


# Evaluation

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    labels = []
    probs = []
    conv_ids = []
    timesteps = []
    total_turns = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            total_loss += F.cross_entropy(logits.float(), y).item()

            p = torch.softmax(logits, dim=-1)[:, 1]

            labels.extend(y.cpu().tolist())
            probs.extend(p.cpu().tolist())

            conv_ids.extend(batch["conversation_ids"])
            timesteps.extend(batch["timesteps"])
            total_turns.extend(batch["total_turns"])

    df = pd.DataFrame(
        {
            "conversation_id": conv_ids,
            "timestep": timesteps,
            "total_turns": total_turns,
            "label": labels,
            "probability": probs,
        }
    ).sort_values(["conversation_id", "timestep"])

    return total_loss / len(loader), df


def aggregate_predictions(df, tau):
    _, summary = replay_alerts(df, tau)
    return summary.rename(
        columns={
            "alerted": "prediction",
            "first_alert_timestep": "first_trigger",
        }
    )


def conversation_metrics(df):

    return {
        "accuracy": accuracy_score(df.label, df.prediction),
        "precision": precision_score(df.label, df.prediction, zero_division=0),
        "recall": recall_score(df.label, df.prediction, zero_division=0),
        "f1": f1_score(df.label, df.prediction, zero_division=0),
    }

def prefix_metrics(df):

    return {
        "accuracy": accuracy_score(
            df.label,
            df.prediction,
        ),
        "precision": precision_score(
            df.label,
            df.prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            df.label,
            df.prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            df.label,
            df.prediction,
            zero_division=0,
        ),
    }
def metrics_by_progress(df):

    rows = []

    tmp = df.copy()

    tmp["progress"] = (
        tmp["timestep"] /
        tmp["total_turns"]
    )

    bins = np.arange(0.1, 1.01, 0.1)

    for b in bins:

        subset = tmp[
            tmp["progress"] <= b
        ]

        if len(subset) == 0:
            continue

        rows.append(
            {
                "progress": b,
                "f1": f1_score(
                    subset.label,
                    subset.prediction,
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    subset.label,
                    subset.prediction,
                ),
            }
        )

    return pd.DataFrame(rows)   

def mean_horizon(df, tau):

    horizons = []

    for _, group in df.groupby("conversation_id"):

        if group["label"].iloc[0] == 0:
            continue

        trigger = group["probability"] > tau

        if not trigger.any():
            continue

        first = group.loc[trigger, "timestep"].iloc[0]
        last = group["timestep"].max()

        horizons.append(last - first)

    return np.mean(horizons) if horizons else 0

# Main

MODEL_CONFIGS = {
    "mistral": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "checkpoint_dir": "checkpoints/mistral_7b",
    },
    "gemma2": {
        "model_name": "google/gemma-2-9b-it",
        "checkpoint_dir": "checkpoints/gemma2_9b",
    },
    "gemma3": {
        "model_name": "google/gemma-3-4b-it",
        "checkpoint_dir": "checkpoints/gemma3_4b",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=MODEL_CONFIGS)
parser.add_argument("--model_name")
parser.add_argument("--checkpoint_dir")
parser.add_argument("--best_model_dir")
parser.add_argument("--data_path", required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--threshold", type=float)
parser.add_argument(
    "--output_dir",
    help="Directory for evaluation outputs. Defaults to <checkpoint_dir>/test_evaluation.",
)

args = parser.parse_args()

if args.model:
    selected = MODEL_CONFIGS[args.model]
    args.model_name = args.model_name or selected["model_name"]
    args.checkpoint_dir = args.checkpoint_dir or selected["checkpoint_dir"]

if not args.model_name or not (args.best_model_dir or args.checkpoint_dir):
    parser.error("provide --model, or provide --model_name and --best_model_dir/--checkpoint_dir")

if args.checkpoint_dir:
    args.best_model_dir = args.best_model_dir or os.path.join(args.checkpoint_dir, "best_model")

if args.threshold is None and args.checkpoint_dir:
    threshold_path = os.path.join(args.checkpoint_dir, "best_tau.json")
    if not os.path.isfile(threshold_path):
        parser.error(f"no validation threshold found at {threshold_path}; pass --threshold")
    with open(threshold_path) as f:
        args.threshold = float(json.load(f)["best_tau"])

if args.output_dir is None:
    args.output_dir = os.path.join(
        args.checkpoint_dir or os.path.dirname(args.best_model_dir),
        "test_evaluation",
    )
os.makedirs(args.output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.best_model_dir)
tokenizer.truncation_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = TemporalDataset(
    args.data_path,
    tokenizer,
    args.max_length,
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=TemporalDataCollator(tokenizer),
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto",
)

base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(
    base_model,
    args.best_model_dir,
)

model.eval()

_, predictions = evaluate(model, loader, device)
# The threshold was tuned on validation and is intentionally fixed on test.
best_tau = args.threshold
alert_replay, alert_summary = replay_alerts(predictions, best_tau)
predictions["prediction"] = alert_replay["alert"].to_numpy()
predictions["correct"] = (predictions["label"] == predictions["prediction"])
predictions["error_type"] = np.select(
    [
        (predictions["label"] == 0) & (predictions["prediction"] == 1),
        (predictions["label"] == 1) & (predictions["prediction"] == 0),
    ],
    ["false_positive", "false_negative"],
    default="correct",
)
pd.DataFrame(
    [{"threshold": best_tau}]
).to_csv(
    os.path.join(args.output_dir, "evaluation_config.csv"),
    index=False,
)
prefix_results = prefix_metrics(predictions)

progress_results = metrics_by_progress(predictions)
conversation_predictions = aggregate_predictions(
    predictions,
    best_tau,
)

metrics = conversation_metrics(conversation_predictions)
alert_metrics = conversation_alert_metrics(alert_summary)

mh = mean_horizon(
    predictions,
    best_tau,
)
progress_results.to_csv(
    os.path.join(args.output_dir, "progress_metrics.csv"),
    index=False,
)
print(f"\nEvaluation threshold fixed from validation: {best_tau:.6f}")
print()

for k, v in metrics.items():
    print(f"{k:12s}: {v:.4f}")
print("\nOnline alert metrics")
for key in (
    "conversation_count",
    "positive_conversations",
    "negative_conversations",
    "alert_precision",
    "alert_recall",
    "conversation_f1",
    "negative_alert_rate",
    "positive_withdrawal_rate",
    "negative_withdrawal_rate",
):
    value = alert_metrics[key]
    print(f"{key:28s}: {value if isinstance(value, int) else f'{value:.4f}'}")
positive_delays = alert_summary.loc[
    (alert_summary["label"] == 1) & alert_summary["delay_before_derailment"].notna(),
    "delay_before_derailment",
]
if len(positive_delays):
    print(
        "positive alert delay (n/min/median/max): "
        f"{len(positive_delays)}/{positive_delays.min():.0f}/"
        f"{positive_delays.median():.1f}/{positive_delays.max():.0f}"
    )
print("\nPrefix metrics")
for k, v in prefix_results.items():
    print(f"{k:12s}: {v:.4f}")
print(f"Mean Horizon: {mh:.3f}")

predictions.to_csv(
    os.path.join(args.output_dir, "prefix_predictions.csv"),
    index=False,
)
alert_replay.to_csv(
    os.path.join(args.output_dir, "online_alert_replay.csv"), index=False
)
alert_summary.to_csv(
    os.path.join(args.output_dir, "online_alert_conversation_summary.csv"), index=False
)
conversation_predictions["correct"] = (
    conversation_predictions["label"] == conversation_predictions["prediction"]
)
conversation_predictions["error_type"] = np.select(
    [
        (conversation_predictions["label"] == 0)
        & (conversation_predictions["prediction"] == 1),
        (conversation_predictions["label"] == 1)
        & (conversation_predictions["prediction"] == 0),
    ],
    ["false_positive", "false_negative"],
    default="correct",
)
conversation_predictions.to_csv(
    os.path.join(args.output_dir, "conversation_predictions.csv"),
    index=False,
)

predictions.loc[~predictions["correct"]].to_csv(
    os.path.join(args.output_dir, "prefix_errors.csv"),
    index=False,
)
conversation_predictions.loc[~conversation_predictions["correct"]].to_csv(
    os.path.join(args.output_dir, "conversation_errors.csv"),
    index=False,
)


print("\nSaved:")
print(f"  {args.output_dir}/prefix_predictions.csv")
print(f"  {args.output_dir}/conversation_predictions.csv")
print(f"  {args.output_dir}/prefix_errors.csv")
print(f"  {args.output_dir}/conversation_errors.csv")
