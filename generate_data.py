"""
Synthetic server-load telemetry generator.

Produces a realistic multivariate time series for a Node.js automation
container with channels: CPU %, memory %, load1, net in/out (KB/s).

Two regimes are simulated:
  normal  - smooth daily cycle + noise
  novel   - CPU spikes (runaway jobs / event-loop saturation) and
            slow memory-leak ramps (heap growth never released)

The channels are physically consistent (CPU is jittery, memory is smooth),
which is the structure One-Class SVM picks up on.
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42


def simulate(n_steps: int = 7_200,   # 30-second samples -> 60 hours
             start: str = "2026-09-01 00:00:00",
             incidents: dict | None = None) -> pd.DataFrame:
    """Build the telemetry DataFrame with injected incidents.

    incidents maps a start step to a dict:
      {"type": "cpu_spike"|"mem_leak", "duration": steps}
    """
    rng = np.random.default_rng(SEED)
    incidents = incidents or {
        900:  {"type": "cpu_spike", "duration": 40},    # +12h
        2600: {"type": "mem_leak",  "duration": 360},   # ~+21h
        4300: {"type": "cpu_spike", "duration": 50},    # ~+36h
        6100: {"type": "mem_leak",  "duration": 300},   # ~+51h
    }

    t = np.arange(n_steps)
    ts = pd.date_range(start, periods=n_steps, freq="30s")

    # --- baseline: gentle daily cycle (busy 8:00-20:00) -------------------
    hour = np.asarray(ts.hour + ts.minute / 60)
    daily = 0.12 * np.sin((hour - 6) / 24 * 2 * np.pi) + 0.08

    cpu = 22 + 35 * daily + rng.normal(0, 2.5, n_steps)
    mem = 41 + 8 * daily + np.cumsum(rng.normal(0, 0.02, n_steps))  # smooth drift
    load1 = cpu / 100 * 4 * (1 + rng.normal(0, 0.05, n_steps))
    net_in = np.abs(rng.normal(120, 25, n_steps))
    net_out = net_in * (0.4 + 0.2 * daily) + np.abs(rng.normal(0, 15, n_steps))

    labels = np.zeros(n_steps, dtype=int)  # 0 = normal, 1 = injected anomaly

    # --- inject incidents ---------------------------------------------------
    for start_step, spec in incidents.items():
        dur = spec["duration"]
        sl = slice(start_step, min(start_step + dur, n_steps))
        n = sl.stop - sl.start
        if n <= 0:
            continue
        labels[sl] = 1
        if spec["type"] == "cpu_spike":
            # fast attack, jittery plateau, quick decay (runaway worker)
            k = max(n // 6, 1)
            envelope = np.r_[np.linspace(0, 1, k),
                             np.ones(max(n - 2 * k, 0)),
                             np.linspace(1, 0, k)][:n]
            cpu[sl] += envelope * rng.uniform(35, 55)
            load1[sl] *= 1 + 1.8 * envelope
            net_out[sl] += envelope * rng.uniform(200, 500, n)
        else:  # mem_leak
            # slow monotone ramp: heap grows ~20-30% and is never released
            ramp = np.linspace(0, rng.uniform(18, 28), n)
            mem[sl] += ramp
            mem[start_step + dur:] += ramp[-1]  # leaked memory never recovered
            cpu[sl] += rng.normal(3, 1, n)  # GC pressure

    cpu = np.clip(cpu, 1, 100)
    mem = np.clip(mem, 5, 100)
    df = pd.DataFrame({
        "timestamp": ts,
        "cpu_pct": cpu.round(2),
        "mem_pct": mem.round(2),
        "load1": load1.round(3),
        "net_in_kbps": net_in.round(2),
        "net_out_kbps": net_out.round(2),
        "incident": labels,
        "incident_type": [
            _type_at(i, incidents) if labels[i] else "normal"
            for i in range(n_steps)
        ],
    })
    return df


def _type_at(i, incidents):
    for s, spec in incidents.items():
        if s <= i < s + spec["duration"]:
            return spec["type"]
    return "normal"


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic container telemetry")
    ap.add_argument("--steps", type=int, default=7200, help="number of 30s samples")
    ap.add_argument("--out", default="data/metrics.csv")
    args = ap.parse_args()

    df = simulate(args.steps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    n_anom = int(df["incident"].sum())
    print(f"wrote {args.out}: {len(df):,} rows, {n_anom} anomaly samples "
          f"({n_anom / len(df):.1%})")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
