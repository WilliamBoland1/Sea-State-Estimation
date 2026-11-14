"""Step 6: score the chosen models once on the hidden test runs, and plot the results.

Reads data/spectra.npz, data/physics.csv and data/cv.csv and writes data/test.csv,
figures/cv.png and figures/test.png.

    python evaluate.py           # fit on every cal run, score the test runs, plot
    python evaluate.py --check   # check the file, and that no test run leaks into the models

Every model of step 5 is fitted on all cal runs with the number of components that had
the lowest Hs + Tp error in data/cv.csv, and estimates the test runs. The physics
estimates of step 4 are not fitted to anything, so their test rows are scored as they are.
This is the one time the test runs are scored: changing a setting because of what it
shows would make them part of the tuning, and the errors would no longer be unbiased.

The table gives, per method and target:
cal     the CV error of step 5 for the models; the error on the cal runs for the physics
        methods, which are not fitted
test    RMS relative error on the test runs [%], with its 95 % bootstrap interval over the
        runs (with 50 runs it is about +-10 % of the error itself), and the mean (bias) [%]

What data/test.csv holds (one row per test run, in the order of runs.csv)
-------------------------------------------------------------------------
run_id, Hs, Tp              the true sea state [m], [s]
Hs_<method>, Tp_<method>    the estimate of every physics method and every model
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import numpy as np
import pandas as pd

import config as C
from build_spectra import COLORS, SOFT, SURFACE, save, style
from cross_validate import HYBRID_BASE, MODELS, TARGETS, base, fit_predict, load, rms_error

N_BOOTSTRAP = 2000
SHOWN = [HYBRID_BASE, "pls", "hybrid"]      # in figures/test.png: the hybrid, its start, PLS alone
COLOR = dict(zip(MODELS, COLORS)) | {HYBRID_BASE: SOFT}


def chosen():
    """{model: number of components} with the lowest Hs + Tp CV error, as step 5 chose it."""
    cv = pd.read_csv(C.CV_FILE)
    best = (cv.Hs_rms + cv.Tp_rms).groupby(cv.model).idxmin()
    return {model: int(cv.components[best[model]]) for model in MODELS}


def physics_estimates(split):
    """{method: log estimate (runs, targets)} of every step 4 method, for the runs in split."""
    physics = pd.read_csv(C.PHYSICS_FILE)
    rows = physics.split == split
    methods = [column[3:] for column in physics.columns if column.startswith("Hs_")]
    return {m: np.log(physics.loc[rows, [f"{t}_{m}" for t in TARGETS]].to_numpy()) for m in methods}


def predict(components, cal, X_new, physics_new):
    """{model: log estimate (runs, targets)} for X_new, every model fitted on cal = load("cal")."""
    X, log_y, physics = cal
    return {model: fit_predict(MODELS[model], n, X, log_y - base(model, physics), X_new)
                   + base(model, physics_new)
            for model, n in components.items()}


def estimates(components):
    """The log truth of the test runs (runs, targets) and {method: log estimate}."""
    X_test, log_true, physics_test = load("test")
    return log_true, physics_estimates("test") | predict(components, load("cal"), X_test,
                                                          physics_test)


def cal_errors(components):
    """{method: RMS relative error [%] per target} on the cal runs, CV for the models."""
    log_y = load("cal")[1]
    cv = pd.read_csv(C.CV_FILE).set_index(["model", "components"])
    return ({m: rms_error(p, log_y) for m, p in physics_estimates("cal").items()}
            | {model: cv.loc[(model, n), [f"{t}_rms" for t in TARGETS]].to_numpy()
               for model, n in components.items()})


def report(log_true, log_pred, components, cal_error):
    """Cal and test error of every method, the test error with its 95 % bootstrap interval."""
    resample = np.random.default_rng(0).integers(0, len(log_true), (N_BOOTSTRAP, len(log_true)))
    table = {}
    for method, p in log_pred.items():
        test = rms_error(p, log_true)
        low, high = np.percentile(rms_error(p[resample], log_true[resample]), [2.5, 97.5], axis=0)
        bias = 100.0 * np.mean(np.exp(p - log_true) - 1.0, axis=0)
        row = table[method] = {"components": components.get(method, "-")}
        for t, target in enumerate(TARGETS):
            row[f"{target} cal %"] = cal_error[method][t]
            row[f"{target} test %"] = test[t]
            row[f"{target} test 95 %"] = f"{low[t]:.2f}-{high[t]:.2f}"
            row[f"{target} bias %"] = bias[t]
    print(pd.DataFrame.from_dict(table, orient="index").round(2).to_string())


def plot_cv(components, cal_error):
    """CV error against the number of components, and the cal error of the physics estimate."""
    cv = pd.read_csv(C.CV_FILE)
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(8.5, 3.9), facecolor=SURFACE,
                             layout="constrained")
    for t, (ax, target) in enumerate(zip(axes, TARGETS)):
        for model, n in components.items():
            error = cv[cv.model == model].set_index("components")[f"{target}_rms"]
            ax.semilogy(error.index, error, color=COLOR[model], lw=1.4, label=model)
            ax.plot(n, error[n], "o", color=COLOR[model], ms=5)
        ax.axhline(cal_error[HYBRID_BASE][t], color=SOFT, lw=1.0, ls=(0, (4, 3)),
                   label=f"physics ({HYBRID_BASE})")
        ax.yaxis.set_major_formatter(LogFormatter())      # 60, not 6 x 10^1
        ax.yaxis.set_minor_formatter(LogFormatter())
        style(ax, target, "Components", "RMS relative error [%]")
    save(fig, axes, f"Cross-validation error on the {C.N_CAL} cal runs; dots mark the chosen "
                    "number of components", "cv.png")


def plot_test(log_true, log_pred):
    """Relative error of every test run against its true Tp."""
    tp = np.exp(log_true[:, TARGETS.index("Tp")])
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(8.5, 3.9), sharex=True, facecolor=SURFACE,
                             layout="constrained")
    for t, (ax, target) in enumerate(zip(axes, TARGETS)):
        ax.axhline(0.0, color=SOFT, lw=1.0)
        for method in SHOWN:
            error = 100.0 * (np.exp(log_pred[method][:, t] - log_true[:, t]) - 1.0)
            ax.plot(tp, error, "o", color=COLOR[method], ms=4, label=method)
        style(ax, target, "True Tp [s]", "Relative error [%]")
    save(fig, axes, f"Error of the estimates on the {len(tp)} test runs", "test.png")


def build():
    components = chosen()
    log_true, log_pred = estimates(components)
    runs = pd.read_csv(C.RUNS_FILE)
    out = pd.concat([pd.DataFrame({"run_id": runs.run_id[runs.split == "test"].to_numpy()}),
                     pd.DataFrame(np.exp(log_true), columns=TARGETS),
                     *[pd.DataFrame(np.exp(p), columns=[f"{t}_{m}" for t in TARGETS])
                       for m, p in log_pred.items()]], axis=1)
    out.to_csv(C.TEST_FILE, index=False, float_format="%.6g")
    cal_error = cal_errors(components)
    plot_cv(components, cal_error)
    plot_test(log_true, log_pred)
    print(f"Wrote {C.TEST_FILE}: {len(out)} test runs. Models fitted on the "
          f"{(runs.split == 'cal').sum()} cal runs:")
    report(log_true, log_pred, components, cal_error)


def check():
    """The file against runs.csv and a fresh estimate, then that the test runs do not leak in."""
    runs, out = pd.read_csv(C.RUNS_FILE), pd.read_csv(C.TEST_FILE)
    test = runs[runs.split == "test"].reset_index(drop=True)
    assert list(out.run_id) == list(test.run_id), "rows must be the test runs of runs.csv"
    assert np.allclose(out[TARGETS], test[TARGETS])
    assert np.isfinite(out.drop(columns="run_id").to_numpy()).all()
    components = chosen()
    _, log_pred = estimates(components)
    for m, p in log_pred.items():
        assert np.allclose(np.log(out[[f"{t}_{m}" for t in TARGETS]]), p, atol=1e-5), \
            f"test.csv is out of date ({m}), rerun this step"

    # Each test run estimated alone must get the same estimate as with the others: then no
    # statistic of the test runs (a mean, a scaling) enters the models.
    cal = load("cal")
    X_test, _, physics_test = load("test")
    alone = [predict(components, cal, X_test[[i]], physics_test[[i]]) for i in range(len(X_test))]
    for model in components:
        assert np.allclose(np.vstack([a[model] for a in alone]), log_pred[model]), \
            f"{model}: the estimate of a test run depends on the other test runs"
    print(f"No problems found: {len(out)} test runs, {len(log_pred)} methods")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="check the file, and that no test run leaks into the models")
    if parser.parse_args().check:
        check()
    else:
        build()
