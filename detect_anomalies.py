"""
One-Class SVM novelty detection on container telemetry.

Pipeline:
  1. sliding-window features (mean / std / slope / range per channel)
  2. fit One-Class SVM (RBF) on windows drawn only from healthy periods
  3. score every window; -1 = novelty (spike or leak), +1 = normal
  4. export a per-window anomaly report
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CHANNELS = ["cpu_pct", "mem_pct", "load1", "net_out_kbps"]
WINDOW = 12          # 12 x 30s = 6 minutes of context per feature vector
NU = 0.05            # upper bound on fraction of training outliers
GAMMA = "scale"
TRAIN_HOURS = None   # e.g. (0, 10) -> fit only on hours 0-10 (known healthy)


def build_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Sliding-window statistical features per channel (indexed by timestamp)."""
    d = df.set_index("timestamp")
    feats = {}
    for col in CHANNELS:
        roll = d[col].rolling(window)
        feats[f"{col}_mean"] = roll.mean()
        feats[f"{col}_std"] = roll.std()
        feats[f"{col}_range"] = roll.max() - roll.min()
        # slope ~ first difference of the window mean = leak signature
        feats[f"{col}_slope"] = roll.mean().diff()
    return pd.DataFrame(feats).dropna()


def fit_ocsvm(X_train: np.ndarray, nu: float = NU, gamma=GAMMA):
    """Fit scaler + RBF One-Class SVM on healthy windows.

    nu bounds the fraction of training outliers, so it directly trades
    false positives against missed novelties.
    """
    model = make_pipeline(
        StandardScaler(),
        OneClassSVM(kernel="rbf", nu=nu, gamma=gamma),
    )
    model.fit(X_train)
    return model


def run(csv_path: str, report_path: str, model_path: str | None = None):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    X = build_features(df)
    truth = df.set_index("timestamp")["incident"].reindex(X.index).values

    # ---- training set: healthy windows only ------------------------------
    healthy_mask = truth == 0
    if TRAIN_HOURS:
        hours = X.index.hour
        healthy_mask &= (hours >= TRAIN_HOURS[0]) & (hours < TRAIN_HOURS[1])
    X_train = X[healthy_mask].values

    model = fit_ocsvm(X_train)

    # decision_function: >0 inlier, <0 novelty. Higher = more normal.
    scores = model.decision_function(X.values)
    preds = np.where(scores >= 0, 0, 1)  # 0 normal, 1 novelty

    report = pd.DataFrame({
        "timestamp": X.index,
        "score": scores.round(4),
        "pred_normal": (preds == 0).astype(int),
        "truth_incident": truth,
    })
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report.to_csv(report_path, index=False)

    if model_path:
        import joblib
        joblib.dump(model, model_path)

    n_novel = int((preds == 1).sum())
    print(f"trained on {len(X_train):,} healthy windows "
          f"({X.shape[1]} features, nu={NU}, gamma={GAMMA})")
    print(f"scored {len(X):,} windows -> {n_novel} flagged as novelty "
          f"({n_novel / len(X):.1%})")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="One-Class SVM anomaly detection")
    ap.add_argument("--csv", default="data/metrics.csv")
    ap.add_argument("--report", default="results/anomaly_report.csv")
    ap.add_argument("--save-model", default=None)
    args = ap.parse_args()
    run(args.csv, args.report, args.save_model)
