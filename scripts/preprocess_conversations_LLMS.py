"""
Preprocess the CGA-WIKI corpus into temporal prefix samples for training and
validation of large language models (LLMs) for temporal conversation derailment
prediction.

This script converts each conversation in the CGA-WIKI dataset into a collection of
conversation prefixes suitable for temporal derailment prediction.

For every conversation, one training example is generated for each valid conversation
prefix:

- Toxic conversations:
    Prefixes include turns 1,...,L-1, excluding the final toxic turn.

- Non-toxic conversations:
    Prefixes include turns 1,...,L.

Each generated sample contains:
    - conversation_id
    - timestep
    - total_turns
    - utterances (stored as a JSON list)
    - label

The script does not perform prompt construction or tokenization.
These steps are deferred to the downstream model pipeline.

Input directory structure:
    data/
        train/
        val/
        test/

Output:
    train_samples.csv
    val_samples.csv
    (optionally test_samples.csv)

Usage:
    python preprocess_cga.py \
        --data_dir data/ \
        --output_dir out/ """

import argparse
import glob
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Blank annotation columns to drop before anything else
BLANK_COLS = ["annot1", "annot1_comment", "annot2", "annot2_comment"]


# 1.  LOADING


def label_from_filename(path: str) -> int:
    """
    _NOT_TOXIC.csv → 0
    _TOXIC.csv     → 1
    """
    name = os.path.basename(path)
    if name.endswith("_NOT_TOXIC.csv"):
        return 0
    if name.endswith("_TOXIC.csv"):
        return 1
    raise ValueError(f"Cannot determine label from filename: {name}")


def load_conversation_file(path: str) -> pd.DataFrame:
    """ Load a single CGA-WIKI conversation and assign its conversation-level label
    based on the filename. """
    df = pd.read_csv(path, dtype=str)

    # Drop blank annotation columns silently
    df.drop(columns=[c for c in BLANK_COLS if c in df.columns], inplace=True)
    df.columns = df.columns.str.strip().str.lower()

    # Conversation-level label from filename
    df["y_i"] = label_from_filename(path)
    df["source_file"] = os.path.basename(path)

    # Sort by turn_index (already ordered, but ensure int sort)
    df["turn_index"] = pd.to_numeric(df["turn_index"], errors="coerce")
    df = df.sort_values("turn_index").reset_index(drop=True)

    # CGA-WIKI text is already clean
    df["text"] = df["text"].fillna("").astype(str)

    return df






def load_split(folder: str) -> list:
    """Load all CSV files from a split folder."""
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    convs, errors = [], []
    for p in paths:
        try:
            convs.append(load_conversation_file(p))
        except Exception as e:
            errors.append(f"  SKIP {os.path.basename(p)}: {e}")

    if errors:
        print(f"  [WARNING] {len(errors)} files skipped:")
        for msg in errors:
            print(msg)

    return convs

# 3.  TEMPORAL PREFIX GENERATION

def build_temporal_samples(convs: list, split_name: str = "") -> pd.DataFrame:
    """
    For every conversation, emit one row per valid time step t.

    Each row stores:
        conversation_id, timestep, total_turns,
        utterances (JSON list of strings), label

    Utterances are stored as a JSON list.
    Prompt formatting and tokenization are performed later by the
    target language model (e.g., Gemma, Llama, Qwen).
    """
    records = []
    skipped = 0

    for df in convs:
        y_i     = int(df["y_i"].iloc[0])
        conv_id = str(df["conversation_id"].iloc[0])
        source_file = str(df["source_file"].iloc[0])
        messages = df["text"].tolist()
        L_i      = len(messages)

        if L_i == 0:
            skipped += 1
            continue

        # TOXIC: exclude last (toxic) turn from all prefixes
        # NOT_TOXIC: all turns are valid
        max_t = L_i - 1 if y_i == 1 else L_i
        if max_t < 1:
            skipped += 1
            continue

        for t in range(1, max_t + 1):

            record = {
                "conversation_id": conv_id,
                "timestep":        t,
                "total_turns":     L_i,
                "utterances":      json.dumps(messages[:t]),
                "label":           y_i,
            }

            # Keep original filename only for test split
            if split_name == "test":
                record["source_file"] = source_file

            records.append(record)
    df_out = pd.DataFrame(records)
    n_pos  = (df_out["label"] == 1).sum()
    n_neg  = (df_out["label"] == 0).sum()
    n_conv = df_out["conversation_id"].nunique()
    print(f"  [{split_name}] conversations={n_conv}  samples={len(df_out)}  "
          f"positive={n_pos} ({100*n_pos/max(len(df_out),1):.1f}%)  "
          f"negative={n_neg} ({100*n_neg/max(len(df_out),1):.1f}%)"
          + (f"  skipped={skipped}" if skipped else ""))

    return df_out


# 6.  MAIN


def main():
    parser = argparse.ArgumentParser(description="CGA-WIKI temporal prefix generation")
    parser.add_argument("--data_dir",      required=True,
                        help="Root folder containing train/, val/, test/")
    parser.add_argument("--output_dir",    default="out/",
                        help="Output folder for *_samples.csv files")
 
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Which splits to process (default: train val)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = {
        name: os.path.join(args.data_dir, name)
        for name in args.splits
    }

    # Step 1: Load 
    print("\n[1/3] Loading conversations...")
    split_convs = {}
    for name, folder in splits.items():
        split_convs[name] = load_split(folder)
        print(f"  {name}: {len(split_convs[name])} conversations loaded")

    # Step 2: Temporal prefix samples 
    print("\n[2/3] Building temporal prefix samples...")
    split_dfs = {
        name: build_temporal_samples(convs, name)
        for name, convs in split_convs.items()
    }


    # Save
    print("\nSaving outputs...")
    for name, df in split_dfs.items():
        out_path = os.path.join(args.output_dir, f"{name}_samples.csv")
        df.to_csv(out_path, index=False)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  {name}_samples.csv  →  {len(df):>6} rows  ({size_mb:.2f} MB)")

    print("Pipeline complete")


if __name__ == "__main__":
    main()

