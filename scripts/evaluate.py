from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def compute_metrics(y_true, y_pred):
    # Ensure lengths match
    if len(y_true) != len(y_pred):
        raise ValueError("Lists must have the same length.")
    return {
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro'),
        "accuracy": accuracy_score(y_true, y_pred)
    }

results_for_metrics = results_df.sort_values(["conv_id", "turn_id"]).groupby("conv_id", group_keys=False).head(-1) 
# Compute metrics on filtered results

metrics = compute_metrics(df["gold"], df["predicted"])
print("Results:", metrics)
cls_report = classification_report(df["gold"], df["predicted"], digits=4)
print("Classification Report:\n", cls_report)
