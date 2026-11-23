"""Step 7: the models of step 6 on the test runs, with an RAO error or a sensor they have not seen.

Writes data/robust/ (the changed .mat files and their OSP runs), data/robust.csv and
figures/robust.png.

    python robustness.py           # make the .mat files, run what is missing in OSP, score
    python robustness.py --check   # check the .mat files, the control and a known answer

Why
---
The hybrid of step 6 corrects one fixed RAO error: L35 (used to invert) against
gunnerus.mat (used to make the runs), and the cal and test runs share it. On a real vessel
the error changes with loading. Here everything of step 6 is frozen (L35, the models fitted
on the cal runs, the components in cv.csv, the assumed sensor noise C.SENSOR_NOISE_STD),
and only the true vessel or the sensor changes:

control   the test runs of step 6 as they are; must give data/test.csv again
gain      heave and pitch |RAO| of gunnerus.mat times g, rerun in OSP
shift     heave and pitch RAO moved to s times the frequency, amp(w) -> amp(w / s), rerun
          in OSP. The phase is kept: only |RAO| enters the spectra. It also scales the
          pitch RAO of long waves (the wave slope k) by 1 / s^2, so it is a stand-in for a
          loading change, not a hull model. A draft change of 6-10 % moves the natural
          periods about 3-5 %.
noise     the sensor noise std changed to every value in C.NOISE_LEVELS; no new runs

Only motionRAO changes in the .mat files, so the DP system sees exactly the same loads.
Every condition uses the same sea states, wave phases and noise draws (scaled), so a change
in error comes from the change alone. Nothing is tuned here, so the test runs can be used
again.

What data/robust.csv holds (one row per condition and method)
-------------------------------------------------------------
condition, kind, level      e.g. gain_1.10, gain, 1.10. The control is in every kind at its
                            nominal level: gain 1, shift 1, noise C.SENSOR_NOISE_STD
method
<target>_rms, _low, _high   RMS relative error [%] and its 95 % bootstrap interval
<target>_bias               mean relative error [%]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import config as C
from build_spectra import COLORS, SOFT, SURFACE, run_spectra, save, style
from cross_validate import HYBRID_BASE, LABEL, TARGETS, load, rms_error
from evaluate import N_BOOTSTRAP, chosen, predict
from invert_spectra import (METHODS, estimate, fine_grid, noise_level, rao_squared,
                            welch_kernel)

HEAVE, PITCH = 2, 4         # rows of motionRAO.amp
NOMINAL = {"gain": 1.0, "shift": 1.0, "noise": C.SENSOR_NOISE_STD["heave"]}
PRINTED = ["fit_pitch", HYBRID_BASE, "pls", "pcr", "hybrid"]
SHOWN = {"fit_pitch": (SOFT, (0, (4, 3))), HYBRID_BASE: (SOFT, "-"),
         "pls": (COLORS[0], "-"), "hybrid": (COLORS[2], "-")}
XLABEL = {"gain": "RAO gain [-]", "shift": "RAO frequency shift [-]",
          "noise": "Sensor noise std [m, deg]"}


def name(kind, level):
    return f"{kind}_{level:.2f}"


def perturbations():
    """(kind, level) of every condition that needs its own OSP runs."""
    return [("gain", g) for g in C.RAO_GAINS] + [("shift", s) for s in C.RAO_SHIFTS]


def change(amp, omega, kind, level):
    """One RAO amplitude cell (frequencies, headings, speeds) changed as kind says."""
    if kind == "gain":
        return amp * level
    return interp1d(omega, amp, axis=0, bounds_error=False,
                    fill_value=(amp[0], amp[-1]))(omega / level)


def make_mat(kind, level):
    """gunnerus.mat with heave and pitch RAO changed, in data/robust/. Returns its path."""
    path = C.ROBUST_DIR / f"{name(kind, level)}.mat"
    if not path.exists():
        raw = loadmat(C.TRUE_MAT)
        rao = raw["vessel"]["motionRAO"][0, 0]
        omega, cells = rao["w"][0, 0].ravel(), rao["amp"][0, 0]
        for dof in (HEAVE, PITCH):
            cells[0, dof] = change(cells[0, dof], omega, kind, level)
        C.ROBUST_DIR.mkdir(parents=True, exist_ok=True)
        savemat(path, {"vessel": raw["vessel"]}, do_compression=True)
    return path


def simulate(test):
    """Runs the test runs in OSP for every changed .mat file, unless they are all done."""
    for kind, level in perturbations():
        mat, folder = make_mat(kind, level), C.ROBUST_DIR / name(kind, level)
        if all((folder / f).exists() for f in test.output_file):
            continue
        import run_osp      # needs OSP; only imported when there is something to run
        print(f"{name(kind, level)}:", flush=True)
        if run_osp.run_batch(list(test.run_id), mat, folder):
            raise RuntimeError(f"{name(kind, level)}: runs failed, start again to retry them")


def conditions():
    """(condition, kind, level, folder, noise std per motion) of every condition."""
    nominal = (C.OUTPUT_DIR, C.SENSOR_NOISE_STD)
    out = [("control", kind, level, *nominal) for kind, level in NOMINAL.items()]
    out += [(name(k, v), k, v, C.ROBUST_DIR / name(k, v), C.SENSOR_NOISE_STD)
            for k, v in perturbations()]
    out += [(name("noise", n), "noise", n, C.OUTPUT_DIR, dict.fromkeys(C.SENSOR_NOISE_STD, n))
            for n in C.NOISE_LEVELS]
    return out


def estimates(test, folder, noise_std, components, cal):
    """{method: log estimate (runs, targets)} of every physics method and model for the test
    runs in folder, with sensor noise noise_std. The physics fit still assumes the nominal
    noise, and the models are the ones fitted on cal = load("cal")."""
    spectra = [run_spectra(row, folder, noise_std) for row in test.itertuples()]
    psd, freq = np.array([s for s, _ in spectra]), spectra[0][1]
    fine, dt = fine_grid(freq), test.dt_s.iloc[0]
    kernel, noise = welch_kernel(dt), noise_level(dt)
    raos = {h: rao_squared(fine, h) for h in test.heading_deg.unique()}
    est = pd.DataFrame([estimate(p, raos[h], fine, kernel, noise)
                        for p, h in zip(psd, test.heading_deg)])
    log_pred = {m: np.log(est[[f"{t}_{m}" for t in TARGETS]].to_numpy()) for m in METHODS}
    return log_pred | predict(components, cal, np.log10(psd).reshape(len(test), -1),
                              log_pred[HYBRID_BASE])


def score(log_pred, log_true, resample):
    """One row per method: RMS, its 95 % bootstrap interval over the runs, and the bias [%]."""
    rows = []
    for method, p in log_pred.items():
        rms = rms_error(p, log_true)
        low, high = np.percentile(rms_error(p[resample], log_true[resample]), [2.5, 97.5], axis=0)
        bias = 100.0 * np.mean(np.exp(p - log_true) - 1.0, axis=0)
        row = {"method": method}
        for t, target in enumerate(TARGETS):
            row |= {f"{target}_rms": rms[t], f"{target}_low": low[t], f"{target}_high": high[t],
                    f"{target}_bias": bias[t]}
        rows.append(row)
    return rows


def test_runs():
    runs = pd.read_csv(C.RUNS_FILE)
    return runs[runs.split == "test"].reset_index(drop=True)


def build():
    test = test_runs()
    simulate(test)
    components, cal = chosen(), load("cal")
    log_true = np.log(test[TARGETS].to_numpy())
    # The same resampled runs for every condition, so the intervals stay paired.
    resample = np.random.default_rng(0).integers(0, len(test), (N_BOOTSTRAP, len(test)))
    rows, done = [], {}
    for condition, kind, level, folder, noise_std in conditions():
        if condition not in done:
            log_pred = estimates(test, folder, noise_std, components, cal)
            done[condition] = score(log_pred, log_true, resample)
        rows += [{"condition": condition, "kind": kind, "level": level} | r
                 for r in done[condition]]
    table = pd.DataFrame(rows)
    table.to_csv(C.ROBUST_FILE, index=False, float_format="%.4f")
    plot(table)
    print(f"Wrote {C.ROBUST_FILE}: {len(done)} conditions on the {len(test)} test runs, "
          f"models fitted on the cal runs ({components}). RMS relative error [%]:")
    unique = table.drop_duplicates(["condition", "method"]).set_index(["condition", "method"])
    order = list(done)
    for target in TARGETS:
        print(f"\n{target}")
        print(unique[f"{target}_rms"].unstack().loc[order, PRINTED].round(2).to_string())


def plot(table):
    """Error against the level of every kind of change, with its bootstrap interval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(NOMINAL)
    # Not sharey: PLS with another noise level would flatten the other panels.
    fig, axes = plt.subplots(len(TARGETS), len(kinds), figsize=(10.0, 6.4),
                             facecolor=SURFACE, layout="constrained")
    for t, target in enumerate(TARGETS):
        for k, kind in enumerate(kinds):
            ax = axes[t, k]
            for method, (color, dash) in SHOWN.items():
                rows = table[(table.kind == kind) & (table.method == method)].sort_values("level")
                ax.plot(rows.level, rows[f"{target}_rms"], color=color, ls=dash, lw=1.4,
                        marker="o", ms=4, label=LABEL.get(method, method))
                ax.fill_between(rows.level, rows[f"{target}_low"], rows[f"{target}_high"],
                                color=color, alpha=0.12, lw=0)
            ax.axvline(NOMINAL[kind], color=SOFT, lw=0.8, ls=":")
            style(ax, f"{target}, {kind}", XLABEL[kind], "RMS relative error [%]" if k == 0 else "")
    save(fig, axes.ravel(), "Test-run error with a true vessel or sensor the models were not "
                            "fitted to; dotted line = step 6", "robust.png")


def check():
    """The .mat files, the control against data/test.csv, and the known answer of a gain."""
    from ospfmusim.hull.helpfunctions.Wamit import load_mat_vessel

    def tables(path):
        data = load_mat_vessel(path, frequency_count=None)
        return (data["frequency_grid"], data["motion_table"].values,
                [data[k].values for k in ("force_table", "drift_table")]
                + [data[k] for k in ("added_mass_by_frequency", "radiation_damping_by_frequency",
                                     "restoring_by_frequency")])

    omega, motion, rest = tables(C.TRUE_MAT)
    for kind, level in perturbations():
        _, changed, changed_rest = tables(make_mat(kind, level))
        others = [d for d in range(6) if d not in (HEAVE, PITCH)]
        assert np.array_equal(changed[others], motion[others]), f"{kind} {level}: other DOFs"
        assert all(np.array_equal(a, b) for a, b in zip(changed_rest, rest)), \
            f"{kind} {level}: loads or matrices changed"
        for dof in (HEAVE, PITCH):
            # The amplitudes in the file are >= 0, so |RAO| is the changed amplitude.
            expected = change(np.abs(motion[dof]), omega, kind, level)
            assert np.allclose(np.abs(changed[dof]), expected, rtol=1e-12, atol=1e-15), \
                f"{kind} {level}: DOF {dof}"
            keep = np.abs(motion[dof]) > 0
            assert np.allclose(np.angle(changed[dof])[keep], np.angle(motion[dof])[keep]), \
                f"{kind} {level}: phase of DOF {dof}"

    test = test_runs()
    components, cal = chosen(), load("cal")
    control = estimates(test, C.OUTPUT_DIR, C.SENSOR_NOISE_STD, components, cal)
    saved = pd.read_csv(C.TEST_FILE)
    for method, p in control.items():
        assert np.allclose(np.exp(p), saved[[f"{t}_{method}" for t in TARGETS]], rtol=1e-4), \
            f"control differs from test.csv for {method}"

    # Heave and pitch are linear in the RAO and the DP loads do not change, so a gain g must
    # give g times the Hs of the control and the same Tp. This proves OSP used the file.
    g = max(C.RAO_GAINS)
    gained = estimates(test, C.ROBUST_DIR / name("gain", g), C.SENSOR_NOISE_STD, components, cal)
    ratio = np.exp(gained["fit_pitch"] - control["fit_pitch"])
    assert np.allclose(ratio[:, 0], g, rtol=0.01), f"Hs ratio {ratio[:, 0].min():.4f}-" \
                                                  f"{ratio[:, 0].max():.4f}, expected {g}"
    assert np.allclose(ratio[:, 1], 1.0, rtol=0.01), "a gain must not change Tp"

    table = pd.read_csv(C.ROBUST_FILE)
    assert set(table.condition) == {c for c, *_ in conditions()}, "robust.csv is out of date"
    assert np.isfinite(table.drop(columns=["condition", "kind", "method"]).to_numpy()).all()
    print(f"No problems found: {len(perturbations())} .mat files, the control equals "
          f"test.csv, gain {g} gives Hs x {ratio[:, 0].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="check the .mat files, the control and the known answer")
    if parser.parse_args().check:
        check()
    else:
        build()
