"""
Evaluate One-Class SVM novelty detection against the injected incident labels.

We can compute honest precision/recall here because generate_data.py injects
known incidents — in production you would only get alert review feedback.
"""
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score,
)

WINDOW = 12


def main(report_path: str = "results/anomaly_report.csv"):
    rep = pd.read_csv(report_path, parse_dates=["timestamp"])
    y_true = rep["truth_incident"].values
    y_pred = 1 - rep["pred_normal"].values  # 1 = flagged novelty
    scores = rep["score"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, -scores)  # lower score = more anomalous

    print("Confusion matrix (rows=truth, cols=pred):")
    print(f"              normal  novel")
    print(f"   normal    {cm[0,0]:7d} {cm[0,1]:6d}")
    print(f"   novel     {cm[1,0]:7d} {cm[1,1]:6d}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  "
          f"F1: {f1:.4f}  ROC-AUC: {auc:.4f}")
    print()
    print(classification_report(
        y_true, y_pred, target_names=["normal", "novel"], zero_division=0))

    # per incident type breakdown (if generate_data produced the labels)
    return {"precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


if __name__ == "__main__":
    main()
