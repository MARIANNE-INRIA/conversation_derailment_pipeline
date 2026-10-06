#!/usr/bin/env python3
"""Create encoder .pt datasets from the exact LLM CSV splits.

The output format is shared by train_deberta.py and train_roberta.py. Each
encoder gets its own output directory because token IDs are tokenizer-specific.
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

try:
    import google.protobuf
except ImportError as error:
    raise ImportError(
        "DeBERTa tokenization requires protobuf. Install it with: "
        "python -m pip install protobuf"
    ) from error


def build_prompt(utterances):
    conversation = []
    for index, utterance in enumerate(utterances, start=1):
        conversation.append(f"Turn {index}:")
        conversation.append(utterance.strip())
        conversation.append("")
    conversation_text = "\n".join(conversation)
    return (
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


def prepare_split(csv_path, tokenizer, max_length):
    dataframe = pd.read_csv(csv_path)
    output = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "conversation_ids": [],
        "timesteps": [],
        "total_turns": [],
    }

    for row in dataframe.itertuples(index=False):
        utterances = json.loads(row.utterances)
        encoded = tokenizer(
            build_prompt(utterances),
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        output["input_ids"].append(torch.tensor(encoded["input_ids"], dtype=torch.long))
        output["attention_mask"].append(torch.tensor(encoded["attention_mask"], dtype=torch.long))
        output["labels"].append(int(row.label))
        output["conversation_ids"].append(str(row.conversation_id))
        output["timesteps"].append(int(row.timestep))
        output["total_turns"].append(int(row.total_turns))

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    if os.path.isabs(args.model_name) and not os.path.isdir(args.model_name):
        raise FileNotFoundError(
            f"Local model directory does not exist: {args.model_name}. "
            "Pass a valid directory or a Hugging Face model ID."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for split_name, csv_path in (
        ("train", args.train_csv),
        ("val", args.val_csv),
        ("test", args.test_csv),
    ):
        print(f"Preparing {split_name}: {csv_path}")
        data = prepare_split(csv_path, tokenizer, args.max_length)
        output_path = os.path.join(args.output_dir, f"{split_name}_samples.pt")
        torch.save(data, output_path)
        print(f"Saved {len(data['labels'])} samples to {output_path}")

    with open(os.path.join(args.output_dir, "preprocessing_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
