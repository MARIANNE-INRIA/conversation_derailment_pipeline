"""Online alert replay and validation-threshold selection utilities."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def validate_prefix_bounds(predictions: pd.DataFrame) -> None:
    """Reject malformed temporal samples before evaluating alerts."""
    required = {"conversation_id", "timestep", "total_turns", "label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for conversation_id, group in predictions.groupby("conversation_id"):
        labels = set(group["label"].astype(int))
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for conversation {conversation_id}")
        total_turns = set(group["total_turns"].astype(int))
        if len(total_turns) != 1:
            raise ValueError(f"Conflicting total_turns for conversation {conversation_id}")
        label = next(iter(labels))
        maximum = next(iter(total_turns)) - (1 if label == 1 else 0)
        timesteps = sorted(group["timestep"].astype(int))
        expected = list(range(1, maximum + 1))
        if timesteps != expected:
            raise ValueError(
                f"Invalid prefix bounds for conversation {conversation_id}: "
                f"got {timesteps[:3]}...{timesteps[-3:]}, expected 1..{maximum}"
            )


def replay_alerts(predictions: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay prefixes in time order and summarize alerts without erasing withdrawals."""
    validate_prefix_bounds(predictions)
    prefixes = predictions.sort_values(["conversation_id", "timestep"]).copy()
    prefixes["threshold"] = float(threshold)
    prefixes["alert"] = (prefixes["probability"] > threshold).astype(int)
    prefixes["alert_withdrawn"] = (
        prefixes.groupby("conversation_id")["alert"].shift(1).eq(1)
        & prefixes["alert"].eq(0)
    ).fillna(False).astype(int)

    rows = []
    for conversation_id, group in prefixes.groupby("conversation_id", sort=False):
        group = group.sort_values("timestep")
        alerts = group["alert"].to_numpy(dtype=bool)
        alert_times = group.loc[group["alert"].eq(1), "timestep"]
        label = int(group["label"].iloc[0])
        total_turns = int(group["total_turns"].iloc[0])
        first_alert = int(alert_times.iloc[0]) if not alert_times.empty else None
        rows.append(
            {
                "conversation_id": conversation_id,
                "label": label,
                "total_turns": total_turns,
                "alerted": int(alerts.any()),
                "first_alert_timestep": first_alert,
                "delay_before_derailment": (
                    total_turns - first_alert if label == 1 and first_alert is not None else np.nan
                ),
                "withdrawal_count": int(group["alert_withdrawn"].sum()),
                "has_withdrawal": int(group["alert_withdrawn"].any()),
            }
        )
    return prefixes, pd.DataFrame(rows)


def conversation_alert_metrics(summary: pd.DataFrame) -> dict[str, float | int]:
    """Report alert, withdrawal, and recovery metrics at conversation level.

    CR is a withdrawal in an observed-negative conversation; IR is a withdrawal
    in an observed-positive conversation before its final observed event.
    Both rates use the total number of conversations N as denominator.
    """
    labels = summary["label"].astype(int)
    alerts = summary["alerted"].astype(int)
    positives = summary[summary["label"] == 1]
    negatives = summary[summary["label"] == 0]
    correct_recovery_count = int(negatives["has_withdrawal"].sum())
    incorrect_recovery_count = int(positives["has_withdrawal"].sum())
    conversation_count = int(len(summary))
    correct_recovery_rate = correct_recovery_count / conversation_count if conversation_count else 0.0
    incorrect_recovery_rate = incorrect_recovery_count / conversation_count if conversation_count else 0.0
    return {
        "conversation_count": conversation_count,
        "positive_conversations": int(len(positives)),
        "negative_conversations": int(len(negatives)),
        "alert_precision": float(precision_score(labels, alerts, zero_division=0)),
        "alert_recall": float(recall_score(labels, alerts, zero_division=0)),
        "conversation_f1": float(f1_score(labels, alerts, zero_division=0)),
        "negative_alert_rate": float(negatives["alerted"].mean()) if len(negatives) else 0.0,
        "positive_alert_rate": float(positives["alerted"].mean()) if len(positives) else 0.0,
        "withdrawal_rate": float(summary["has_withdrawal"].mean()) if len(summary) else 0.0,
        "positive_withdrawal_rate": float(positives["has_withdrawal"].mean()) if len(positives) else 0.0,
        "negative_withdrawal_rate": float(negatives["has_withdrawal"].mean()) if len(negatives) else 0.0,
        "correct_recovery_count": correct_recovery_count,
        "incorrect_recovery_count": incorrect_recovery_count,
        "correct_recovery_rate": correct_recovery_rate,
        "incorrect_recovery_rate": incorrect_recovery_rate,
        "recovery": correct_recovery_rate - incorrect_recovery_rate,
    }


def threshold_candidates(predictions: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(predictions["probability"], dtype=float)
    return np.unique(np.concatenate(([0.0, 1.0], probabilities)))


def select_validation_threshold(
    validation_predictions: pd.DataFrame,
    max_negative_alert_rate: float = 0.10,
) -> tuple[float, pd.DataFrame, float]:
    """Select the lowest-risk threshold maximizing positive recall under a false-alert limit.

    The returned third value is the unconstrained conversation-F1 threshold, retained
    only as a validation comparison and never selected from test predictions.
    """
    if not 0.0 <= max_negative_alert_rate <= 1.0:
        raise ValueError("max_negative_alert_rate must be between 0 and 1")
    rows = []
    for threshold in threshold_candidates(validation_predictions):
        _, summary = replay_alerts(validation_predictions, float(threshold))
        metrics = conversation_alert_metrics(summary)
        rows.append({"threshold": float(threshold), **metrics})
    curve = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    allowed = curve[curve["negative_alert_rate"] <= max_negative_alert_rate]
    if allowed.empty:
        raise ValueError("No threshold satisfies max_negative_alert_rate")
    selected = allowed.sort_values(
        ["positive_alert_rate", "threshold"], ascending=[False, True]
    ).iloc[0]
    f1_selected = curve.sort_values(
        ["conversation_f1", "threshold"], ascending=[False, True]
    ).iloc[0]
    return float(selected["threshold"]), curve, float(f1_selected["threshold"])


def audit_split_dataframes(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Audit conversation-level split disjointness and prefix bounds."""
    ids = {name: set(frame["conversation_id"].astype(str)) for name, frame in splits.items()}
    overlaps = {}
    for left, right in itertools.combinations(ids, 2):
        overlaps[f"{left}__{right}"] = sorted(ids[left] & ids[right])
    for frame in splits.values():
        validate_prefix_bounds(frame)
    return {"conversation_id_overlaps": overlaps, "leakage": any(overlaps.values())}
