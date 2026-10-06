#!/usr/bin/env python3
"""Audit conversation-level split disjointness and temporal prefix bounds."""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

splits = {
    name: pd.read_csv(DATA / f"{name}_samplesLLMs.csv")
    for name in ("train", "val", "test")
}

id_sets = {
    name: set(frame["conversation_id"].astype(str))
    for name, frame in splits.items()
}

leakage = False
prefix_leakage = False
for left, right in combinations(id_sets, 2):
    overlap = id_sets[left] & id_sets[right]
    leakage = leakage or bool(overlap)
    print(f"{left} vs {right}: {len(overlap)} shared conversation_id")
    if overlap:
        print("  ", sorted(overlap)[:10])

for name, frame in splits.items():
    violations = []
    for conversation_id, group in frame.groupby("conversation_id"):
        labels = set(group["label"].astype(int))
        totals = set(group["total_turns"].astype(int))
        if len(labels) != 1 or len(totals) != 1:
            violations.append((conversation_id, "inconsistent label/total_turns"))
            continue
        label = next(iter(labels))
        total_turns = next(iter(totals))
        maximum = total_turns - (1 if label == 1 else 0)
        expected = list(range(1, maximum + 1))
        actual = sorted(group["timestep"].astype(int))
        if actual != expected:
            violations.append((conversation_id, f"expected 1..{maximum}, got {actual}"))
    print(f"{name}: {len(frame)} prefixes, {frame['conversation_id'].nunique()} conversations")
    print(f"{name}: {len(violations)} prefix-bound violations")
    prefix_leakage = prefix_leakage or bool(violations)
    if violations:
        print("  ", violations[:10])

# Exact repeated messages are not conversation leakage, but are reported so the
# experiment record distinguishes repeated short text from shared conversations.
message_sets = {}
for name, frame in splits.items():
    messages = set()
    for value in frame["utterances"]:
        messages.update(json.loads(value))
    message_sets[name] = messages
for left, right in combinations(message_sets, 2):
    overlap = message_sets[left] & message_sets[right]
    print(f"{left} vs {right}: {len(overlap)} shared exact message strings")

if leakage:
    raise SystemExit("DATA LEAKAGE: conversation_id appears in multiple splits")
if prefix_leakage:
    raise SystemExit("INVALID PREFIX DATA: temporal bounds are inconsistent")
print("No conversation-level leakage detected.")
