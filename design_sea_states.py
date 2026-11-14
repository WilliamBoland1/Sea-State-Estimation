"""Step 1: choose the sea states the simulator should run.

Writes data/runs.csv (one row per OSP run) and figures/design.png.

How the points are chosen
-------------------------
Latin hypercube sampling (LHS) spreads the points evenly. We sample in
(Tp, u), u in [0, 1], and set
    Hs = Hs_min + u * (Hs_upper(Tp) - Hs_min)
where Hs_upper(Tp) is the lower of Hs_max and the steepness limit.
This keeps every point physically realistic without throwing samples away.

The test set uses a different seed, so it never shares points with the
calibration set.
"""
import numpy as np
import pandas as pd
from scipy.stats import qmc

import config as C


def hs_upper(tp):
    """Largest allowed Hs for a given Tp (height range and steepness limit)."""
    hs_steep = C.MAX_STEEPNESS * C.G * tp**2 / (2 * np.pi)
    return np.minimum(C.HS_RANGE[1], hs_steep)


def lhs_design(n, seed):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    pts = sampler.random(n)
    tp = C.TP_RANGE[0] + pts[:, 0] * (C.TP_RANGE[1] - C.TP_RANGE[0])
    hs = C.HS_RANGE[0] + pts[:, 1] * (hs_upper(tp) - C.HS_RANGE[0])
    return hs, tp


def build_runs():
    rng = np.random.default_rng(C.DESIGN_SEED_CAL * 1000 + C.DESIGN_SEED_TEST)
    rows = []
    for split, n, seed in [("cal", C.N_CAL, C.DESIGN_SEED_CAL),
                           ("test", C.N_TEST, C.DESIGN_SEED_TEST)]:
        for heading in C.HEADINGS_DEG:
            hs, tp = lhs_design(n, seed)
            for h, t in zip(hs, tp):
                rows.append(dict(split=split, Hs=round(h, 3), Tp=round(t, 3),
                                 gamma=C.GAMMA, heading_deg=heading))
    runs = pd.DataFrame(rows)
    runs.insert(0, "run_id", [f"run_{i:04d}" for i in range(len(runs))])
    # Random-phase seed for the wave realisation in each run
    runs["wave_seed"] = rng.integers(0, 2**31 - 1, len(runs))
    runs["duration_s"] = C.DURATION_S
    runs["ramp_s"] = C.RAMP_S
    runs["dt_s"] = C.DT_S
    runs["output_file"] = runs["run_id"] + ".csv"
    return runs


def plot_design(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tp_line = np.linspace(*C.TP_RANGE, 200)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(tp_line, hs_upper(tp_line), "k--", lw=1, label="upper limit (Hs max / steepness)")
    for split, marker, color in [("cal", "o", "tab:blue"), ("test", "^", "tab:red")]:
        d = runs[runs.split == split]
        ax.scatter(d.Tp, d.Hs, s=18, marker=marker, color=color,
                   label=f"{split} ({len(d)})", alpha=0.8)
    ax.set_xlabel("Tp [s]")
    ax.set_ylabel("Hs [m]")
    ax.set_title("Sea states to simulate")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / "design.png", dpi=150)


if __name__ == "__main__":
    C.DATA_DIR.mkdir(exist_ok=True)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    runs = build_runs()
    runs.to_csv(C.RUNS_FILE, index=False)
    plot_design(runs)
    print(f"Wrote {len(runs)} runs to {C.RUNS_FILE}")
    print(runs.groupby("split")[["Hs", "Tp"]].describe().round(2).T)
