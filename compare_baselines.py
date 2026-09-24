"""
Baseline comparison + error analysis for the novelty detector.

All detectors are fitted on HEALTHY windows only and scored on all windows:
  1. Static threshold rules   - classic ops alerting (healthy mean + k*sigma on
                                CPU level/spread and memory slope)
  2. Isolation Forest         - shallow unsupervised ensemble baseline
  3. One-Class SVM (proposed) - RBF novelty detection

Because a detector's false-positive rate is set by its threshold/nu, all three
are swept over their own operating parameter and compared BOTH at default and
at their best-F1 operating point (fair comparison).

Outputs:
  results/baseline_comparison.csv   - default settings
  results/baseline_sweep.csv        - every swept setting
  results/baseline_tuned.csv        - best-F1 setting per detector
  results/per_type_recall.csv       - cpu_spike vs mem_leak recall
  results/false_positive_hours.csv  - FP distribution by hour
  results/baseline_comparison.png, baseline_sweep.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detect_anomalies import NU, build_features, fit_ocsvm

RESULTS = "results"

# operating parameter per detector (k-sigma / contamination / nu)
SWEEP = {
    "Static 3-sigma threshold": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "Isolation Forest": [0.005, 0.01, 0.02, 0.05, 0.10],
    "One-Class SVM (proposed)": [0.005, 0.01, 0.02, 0.05, 0.10],
}
DEFAULTS = {"Static 3-sigma threshold": 3.0, "Isolation Forest": 0.05,
            "One-Class SVM (proposed)": NU}


# --------------------------------------------------------------------------
# detectors: each returns (hard predictions, continuous anomaly score)
# --------------------------------------------------------------------------
def threshold_rules(X_train: pd.DataFrame, X: pd.DataFrame, k: float = 3.0):
    """Alert when a window leaves the healthy mean + k*sigma band."""
    excess = []
    bands = {}
    for feat in ("cpu_pct_mean", "cpu_pct_range", "mem_pct_slope"):
        mu, sd = X_train[feat].mean(), X_train[feat].std()
        bands[feat] = (mu + k * sd)
        excess.append((X[feat].values - (mu + k * sd)) / (k * sd))
    score = np.maximum.reduce(excess)          # >0 => outside healthy band
    return np.where(score > 0, 1, 0), score


def isolation_forest(X_train: np.ndarray, X: np.ndarray, contamination=0.05):
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(contamination=contamination, random_state=42))
    model.fit(X_train)
    return (model.predict(X) == -1).astype(int), -model.score_samples(X)


def ocsvm(X_train: np.ndarray, X: np.ndarray, nu: float = NU):
    model = fit_ocsvm(X_train, nu=nu)
    score = model.decision_function(X)
    return np.where(score < 0, 1, 0), -score


def metrics(y_true, y_pred, score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, score),
    }


def incident_blocks(meta: pd.DataFrame):
    """Contiguous incident blocks: (type, start_ts, end_ts), gaps > 30 min split."""
    inc = meta[meta["incident"] == 1].sort_index()
    blocks, cur = [], []
    for ts, row in inc.iterrows():
        if cur and (ts - cur[-1][0]) <= pd.Timedelta(minutes=30):
            cur.append((ts, row["incident_type"]))
        else:
            if cur:
                blocks.append((cur[0][1], cur[0][0], cur[-1][0]))
            cur = [(ts, row["incident_type"])]
    if cur:
        blocks.append((cur[0][1], cur[0][0], cur[-1][0]))
    return blocks


def detection_latency(meta: pd.DataFrame, pred: np.ndarray):
    """Samples from each incident's start until its first flagged window."""
    rows = []
    for inc_type, start, end in incident_blocks(meta):
        window_pred = pred[(meta.index >= start) & (meta.index <= end)]
        hits = np.flatnonzero(window_pred == 1)
        rows.append({
            "incident_type": inc_type,
            "start": start, "end": end,
            "duration_samples": int((meta.index <= end).sum() - (meta.index < start).sum()),
            "detected": bool(hits.size),
            "latency_samples": int(hits[0]) if hits.size else None,
            "latency_minutes": round(float(hits[0]) * 0.5, 1) if hits.size else None,
        })
    return pd.DataFrame(rows)


def label_artifact_analysis(X: pd.DataFrame, meta: pd.DataFrame,
                            y_true: np.ndarray, pred: np.ndarray):
    """Split false positives into post-leak elevated-memory vs genuine boundary errors.

    The simulator labels only the leak ramp as anomalous, but memory stays
    permanently elevated afterwards (the leak is never released). Windows in
    that aftermath are labelled 'normal' yet are physically abnormal, so they
    are counted as false positives.
    """
    blocks = incident_blocks(meta)
    leak_ends = [end for inc_type, _, end in blocks if inc_type == "mem_leak"]
    if not leak_ends:
        return None, None
    # memory is elevated after EVERY leak ramp, so the unreliable-label region
    # starts at the first leak end and covers all healthy-labelled windows after it
    first_leak_end = min(leak_ends)
    post = (meta.index > first_leak_end) & (y_true == 0)
    fp = (pred == 1) & (y_true == 0)
    n_post = int((fp & pd.Series(post, index=meta.index)).sum())

    keep = ~pd.Series(post, index=meta.index)          # drop the aftermath
    m = metrics(y_true[keep.values], pred[keep.values],
                np.zeros(keep.sum()))                   # dummy score for AUC
    adjusted = {k: v for k, v in m.items() if k != "ROC-AUC"}

    summary = pd.DataFrame([
        {"metric": "total_false_positives", "value": int(fp.sum())},
        {"metric": "false_positives_post_leak", "value": n_post},
        {"metric": "false_positives_other", "value": int(fp.sum()) - n_post},
        {"metric": "mean_mem_pct_post_leak_fp",
         "value": round(float(X.loc[fp & pd.Series(post, index=meta.index),
                                   "mem_pct_mean"].mean()), 2)},
        {"metric": "mean_mem_pct_healthy_windows",
         "value": round(float(X.loc[y_true == 0, "mem_pct_mean"].mean()), 2)},
        {"metric": "precision_after_correction",
         "value": round(adjusted["Precision"], 4)},
        {"metric": "recall_after_correction",
         "value": round(adjusted["Recall"], 4)},
        {"metric": "f1_after_correction", "value": round(adjusted["F1"], 4)},
        {"metric": "windows_with_unreliable_labels", "value": int(post.sum())},
        {"metric": "windows_evaluated_after_correction", "value": int(keep.sum())},
    ])
    return summary, adjusted


# --------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = pd.read_csv("data/metrics.csv", parse_dates=["timestamp"])
    X = build_features(df)
    meta = df.set_index("timestamp").reindex(X.index)
    y_true = meta["incident"].values
    healthy = y_true == 0
    X_train_np = X[healthy].values
    X_train_df = X[healthy]

    detectors = {
        "Static 3-sigma threshold": lambda p: threshold_rules(X_train_df, X, p),
        "Isolation Forest": lambda p: isolation_forest(X_train_np, X.values, p),
        "One-Class SVM (proposed)": lambda p: ocsvm(X_train_np, X.values, p),
    }

    # ---- default-setting comparison ---------------------------------------
    default_rows, sweep_rows, per_type_rows = [], [], []
    for name, detect in detectors.items():
        for param in SWEEP[name]:
            pred, score = detect(param)
            row = {"Model": name, "param": param, **metrics(y_true, pred, score)}
            sweep_rows.append(row)
            if param == DEFAULTS[name]:
                default_rows.append(row)

            inc_type = meta["incident_type"].values
            for t in ("cpu_spike", "mem_leak"):
                mask = inc_type == t
                if mask.sum():
                    per_type_rows.append({
                        "model": name, "param": param, "incident_type": t,
                        "windows": int(mask.sum()),
                        "recall": float(pred[mask].mean()),
                    })

    sweep = pd.DataFrame(sweep_rows).round(4)
    sweep.to_csv(os.path.join(RESULTS, "baseline_sweep.csv"), index=False)

    cmp_df = pd.DataFrame(default_rows).drop(columns=["param"])
    cmp_df = cmp_df[["Model", "Accuracy", "Precision", "Recall", "F1",
                     "ROC-AUC"]].round(4)
    cmp_df.to_csv(os.path.join(RESULTS, "baseline_comparison.csv"), index=False)
    print("=== Default settings (k=3.0 sigma, contamination=nu=0.05) ===")
    print(cmp_df.to_string(index=False))

    tuned = (sweep.sort_values("F1", ascending=False)
             .groupby("Model", as_index=False).first()
             .sort_values("F1", ascending=False))
    tuned = tuned[["Model", "param", "Accuracy", "Precision", "Recall", "F1",
                   "ROC-AUC"]].round(4)
    tuned.to_csv(os.path.join(RESULTS, "baseline_tuned.csv"), index=False)
    print("\n=== Best-F1 operating point per detector (fair comparison) ===")
    print(tuned.to_string(index=False))

    per_type = pd.DataFrame(per_type_rows).round(4)
    per_type.to_csv(os.path.join(RESULTS, "per_type_recall.csv"), index=False)
    print("\n=== Recall by incident type (default settings) ===")
    print(per_type[per_type["param"] == 0.05].round(4).to_string(index=False))

    # ---- error analysis for the proposed detector -------------------------
    pred, score = detectors["One-Class SVM (proposed)"](DEFAULTS["One-Class SVM (proposed)"])
    fp = (pred == 1) & (y_true == 0)
    fp_hours = pd.Series(meta.index[fp].hour).value_counts().sort_index()
    fp_hours.to_frame("false_positives").to_csv(
        os.path.join(RESULTS, "false_positive_hours.csv"))
    print(f"\n=== One-Class SVM error profile ===")
    print(f"false positives: {int(fp.sum())} / {int((y_true == 0).sum())} healthy windows "
          f"({fp.sum() / (y_true == 0).sum():.2%})")
    for feat in ("cpu_pct_mean", "mem_pct_mean"):
        print(f"  {feat}: false-positive mean {X.loc[fp, feat].mean():.2f} | "
              f"healthy-window mean {X.loc[~fp & (y_true == 0), feat].mean():.2f}")
    print(f"  busiest FP hours: {fp_hours.sort_values(ascending=False).head(4).to_dict()}")

    # detection latency per incident
    latency = detection_latency(meta, pred)
    latency.to_csv(os.path.join(RESULTS, "detection_latency.csv"), index=False)
    print("\n=== Detection latency (samples after incident start, 30 s each) ===")
    print(latency[["incident_type", "duration_samples", "detected",
                   "latency_samples", "latency_minutes"]].to_string(index=False))

    # false positives that are really unlabelled leak aftermath
    summary, adjusted = label_artifact_analysis(X, meta, y_true, pred)
    if summary is not None:
        summary.to_csv(os.path.join(RESULTS, "label_artifact_analysis.csv"), index=False)
        print("\n=== Label-artifact (post-leak) analysis of false positives ===")
        print(summary.to_string(index=False))
        print(f"  -> corrected: precision {adjusted['Precision']:.4f}, "
              f"recall {adjusted['Recall']:.4f}, F1 {adjusted['F1']:.4f}")

    # ---- figures ----------------------------------------------------------
    ax = cmp_df.set_index("Model")[["Precision", "Recall", "F1", "ROC-AUC"]].plot(
        kind="bar", figsize=(9, 4.5), ylim=(0, 1.05), rot=8,
        color=["#4e79a7", "#e15759", "#59a14f", "#b07aa1"])
    ax.set_ylabel("score")
    ax.set_title("Default settings: baseline vs. proposed detector")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "baseline_comparison.png"), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in SWEEP:
        sub = sweep[sweep["Model"] == name]
        ax.plot(sub["param"], sub["F1"], marker="o", label=name)
    ax.set_xlabel("operating parameter (k-sigma / contamination / nu)")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs. operating point (all detectors tuned on healthy data)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "baseline_sweep.png"), dpi=140)
    plt.close()
    print(f"\nsaved {RESULTS}/baseline_*.csv|png, per_type_recall.csv, "
          f"false_positive_hours.csv, detection_latency.csv, "
          f"label_artifact_analysis.csv")


if __name__ == "__main__":
    main()
