"""
CGA-WIKI Test Set Preprocessing for LLM Evaluation
=================================================

Preprocess the CGA-WIKI test set into temporal prefix samples for evaluation of
large language models (LLMs) for temporal conversation derailment prediction.

Creates temporal prefix samples for the CGA-WIKI test split only.

This script does not perform prompt construction or tokenization. Instead,
conversation prefixes are stored as JSON lists of utterances, allowing each
downstream LLM (e.g., Gemma, Llama, GPT, or Qwen) to apply its own prompt
template and tokenizer.

Output columns
--------------
conversation_id
source_file
timestep
prefix_length
remaining_turns
total_turns
label
utterances
message_info

message_info preserves the original annotations for later qualitative
analysis.

Temporal prefix rule
--------------------
TOXIC conversations:
    prefixes = turns 1...(L-1)

NOT_TOXIC conversations:
    prefixes = turns 1...L
"""

import argparse
import glob
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def label_from_filename(path):

    name = os.path.basename(path)

    if name.endswith("_NOT_TOXIC.csv"):
        return 0

    if name.endswith("_TOXIC.csv"):
        return 1

    raise ValueError(f"Cannot determine label from {name}")


def load_conversation(path):

    df = pd.read_csv(path, dtype=str)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    df["turn_index"] = pd.to_numeric(
        df["turn_index"],
        errors="coerce"
    )

    df = (
        df
        .sort_values("turn_index")
        .reset_index(drop=True)
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["y_i"] = label_from_filename(path)

    df["source_file"] = os.path.basename(path)

    return df


def load_split(folder):

    paths = sorted(
        glob.glob(
            os.path.join(folder, "*.csv")
        )
    )

    if len(paths) == 0:
        raise FileNotFoundError(folder)

    conversations = []

    for p in paths:

        conversations.append(
            load_conversation(p)
        )

    return conversations


def build_prefixes(conversations):

    rows = []

    for df in conversations:

        label = int(df["y_i"].iloc[0])

        conv_id = str(
            df["conversation_id"].iloc[0]
        )

        source_file = str(
            df["source_file"].iloc[0]
        )

        messages = (
            df["text"]
            .fillna("")
            .tolist()
        )

        message_info = (
            df[
                [
                    "turn_index",
                    "speaker",
                    "text",
                    "annot1",
                    "annot2",
                    "comment_has_personal_attack"
                ]
            ]
            .fillna("")
            .to_dict("records")
        )

        total_turns = len(messages)

        if total_turns == 0:
            continue

        max_prefix = (
            total_turns - 1
            if label == 1
            else total_turns
        )

        for t in range(1, max_prefix + 1):

            rows.append({

                "conversation_id":
                conv_id,

                "source_file":
                source_file,

                "label":
                label,

                "timestep":
                t,

                "prefix_length":
                t,

                "remaining_turns":
                total_turns - t,

                "total_turns":
                total_turns,

                "utterances":
                json.dumps(
                    messages[:t],
                    ensure_ascii=False
                ),

                "message_info":
                json.dumps(
                    message_info[:t],
                    ensure_ascii=False
                )

            })

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder containing the test CSV files."
    )

    parser.add_argument(
        "--output_csv",
        default="test_samples.csv"
    )

    args = parser.parse_args()

    print("Loading conversations...")

    conversations = load_split(args.data_dir)

    print(
        f"{len(conversations)} conversations loaded."
    )

    print("Building temporal prefixes...")

    df = build_prefixes(conversations)

    print(
        f"{len(df)} prefixes created."
    )

    df.to_csv(
        args.output_csv,
        index=False
    )

    print()
    print("Saved to:")
    print(args.output_csv)

    print()
    print(df.head())


if __name__ == "__main__":
    main()
