import unittest

import pandas as pd

from scripts.online_alerts import (
    audit_split_dataframes,
    conversation_alert_metrics,
    replay_alerts,
    select_validation_threshold,
    validate_prefix_bounds,
)


def make_predictions():
    return pd.DataFrame(
        [
            {"conversation_id": "pos", "timestep": 1, "total_turns": 3, "label": 1, "probability": 0.2},
            {"conversation_id": "pos", "timestep": 2, "total_turns": 3, "label": 1, "probability": 0.8},
            {"conversation_id": "pos_long", "timestep": 1, "total_turns": 4, "label": 1, "probability": 0.2},
            {"conversation_id": "pos_long", "timestep": 2, "total_turns": 4, "label": 1, "probability": 0.8},
            {"conversation_id": "pos_long", "timestep": 3, "total_turns": 4, "label": 1, "probability": 0.1},
            {"conversation_id": "neg", "timestep": 1, "total_turns": 2, "label": 0, "probability": 0.7},
            {"conversation_id": "neg", "timestep": 2, "total_turns": 2, "label": 0, "probability": 0.2},
        ]
    )


class OnlineAlertTests(unittest.TestCase):
    def test_prefix_bounds_and_split_disjointness(self):
        predictions = make_predictions()
        validate_prefix_bounds(predictions)
        audit = audit_split_dataframes(
            {
                "train": predictions[predictions["conversation_id"] != "neg"],
                "val": predictions[predictions["conversation_id"] == "neg"],
            }
        )
        self.assertFalse(audit["leakage"])

        invalid = predictions.copy()
        invalid.loc[invalid["conversation_id"] == "pos", "timestep"] = [1, 3]
        with self.assertRaises(ValueError):
            validate_prefix_bounds(invalid)

    def test_replay_preserves_first_alert_and_counts_withdrawal(self):
        prefixes = make_predictions()
        replay, summary = replay_alerts(prefixes, threshold=0.5)
        pos = summary.loc[summary["conversation_id"] == "pos_long"].iloc[0]
        neg = summary.loc[summary["conversation_id"] == "neg"].iloc[0]
        self.assertEqual(pos["alerted"], 1)
        self.assertEqual(pos["first_alert_timestep"], 2)
        self.assertEqual(pos["delay_before_derailment"], 2)
        self.assertEqual(pos["withdrawal_count"], 1)
        self.assertEqual(neg["alerted"], 1)
        self.assertEqual(neg["withdrawal_count"], 1)
        self.assertEqual(
            replay.set_index(["conversation_id", "timestep"])["alert"].to_dict(),
            {
                ("neg", 1): 1,
                ("neg", 2): 0,
                ("pos", 1): 0,
                ("pos", 2): 1,
                ("pos_long", 1): 0,
                ("pos_long", 2): 1,
                ("pos_long", 3): 0,
            },
        )
        metrics = conversation_alert_metrics(summary)
        self.assertEqual(metrics["correct_recovery_count"], 1)
        self.assertEqual(metrics["incorrect_recovery_count"], 1)
        self.assertAlmostEqual(metrics["correct_recovery_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["incorrect_recovery_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["recovery"], 0.0)

    def test_threshold_is_selected_from_validation_and_f1_is_comparison_only(self):
        selected, curve, f1_threshold = select_validation_threshold(
            make_predictions(), max_negative_alert_rate=0.0
        )
        self.assertEqual(selected, 0.7)
        self.assertEqual(f1_threshold, 0.7)
        self.assertTrue((curve["negative_alert_rate"] <= 1.0).all())
        self.assertNotIn("test", curve.columns)


if __name__ == "__main__":
    unittest.main()
