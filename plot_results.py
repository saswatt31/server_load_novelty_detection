"""
Plots for the server-load novelty detection project:
  1. timeline.png      - CPU/memory with detected anomalies highlighted
  2. decision_score.png - SVM decision score over time vs. the margin at 0
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_DIR = "results"


def main(csv_path="data/metrics.csv", report_path="results/anomaly_report.csv"):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    rep = pd.read_csv(report_path, parse_dates=["timestamp"])
    os.makedirs(RESULT_DIR, exist_ok=True)

    # map per-window predictions back onto raw samples for shading
    pred_anom_windows = rep[rep["pred_normal"] == 0]["timestamp"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(df["timestamp"], df["cpu_pct"], lw=0.7, color="tab:blue",
                 label="CPU %")
    axes[0].plot(df["timestamp"], df["mem_pct"], lw=0.9, color="tab:green",
                 label="Memory %")
    for ts in pred_anom_windows:
        axes[0].axvspan(ts, ts + pd.Timedelta(minutes=6), color="red", alpha=0.25)
    axes[0].set_ylabel("percent")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("One-Class SVM detections (red) vs. injected incidents")

    axes[1].plot(rep["timestamp"], -rep["score"], lw=0.8, color="tab:purple",
                 label="anomaly score (negated decision function)")
    axes[1].axhline(0, color="black", ls="--", lw=1, label="SVM margin")
    anom = rep[rep["pred_normal"] == 0]
    axes[1].scatter(anom["timestamp"], -anom["score"], s=6, color="red",
                    zorder=3, label="flagged")
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "timeline.png"), dpi=140)
    plt.close(fig)
    print(f"saved {RESULT_DIR}/timeline.png")


if __name__ == "__main__":
    main()
