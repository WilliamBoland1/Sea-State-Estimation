"""Step 5: PCA, PLSR and a hybrid model, compared by cross-validation on the cal runs.

Reads data/spectra.npz and data/physics.csv and writes data/cv.csv. The test runs are not
touched: step 6 fits the chosen models on all cal runs and scores them on the test runs,
with load() and fit_predict() from here.

    python cross_validate.py           # cross-validate every model, print the best settings
    python cross_validate.py --check   # check PLS and PCR against least squares, and the file

Models
------
X is the log10 spectra, the targets are log Hs and log Tp. Both are autoscaled on the
training runs only.
pls     partial least squares (PLS2, both targets at once) on the spectra
pcr     least squares on the first principal components of the spectra
hybrid  the physics estimate HYBRID_BASE from step 4, corrected by PLS on the spectra:
        log Y = log Y_physics + PLS(X). With 0 components only its mean bias is removed.
        Shown as "CIM hybrid": it follows Martens' CIM-ML (known physics first, a small
        PLS model for what it misses), but is trained once, not continuously.

Cross-validation
----------------
C.N_FOLDS folds, repeated for every seed in C.CV_SEEDS, with 0 to C.MAX_COMPONENTS
components. The error is the RMS relative error over the cal runs, averaged over the seeds.
The number of components with the lowest Hs + Tp error is chosen. That error is a little
optimistic, because it also chose the number; the test runs in step 6 are not.

What data/cv.csv holds
----------------------
model, components, Hs_rms, Tp_rms   [%], one row per model and number of components
"""
import argparse

import numpy as np
import pandas as pd

import config as C

TARGETS = ["Hs", "Tp"]
HYBRID_BASE = "fit_all"     # the physics estimate in physics.csv that the hybrid corrects
# Names shown in figures, the same as in the report.
LABEL = {"hybrid": "Hybrid", "pls": "PLS", "pcr": "PCR", HYBRID_BASE: "Physics fit"}


def autoscale(train, *others):
    """train and others scaled with the mean and standard deviation of train, and those two."""
    mean, std = train.mean(axis=0), train.std(axis=0, ddof=1)
    return [(a - mean) / std for a in (train, *others)], mean, std


def pls(X, Y, n):
    """PLS2 coefficients (features, targets) for autoscaled X and Y with n components."""
    W, P, Q = [], [], []
    for _ in range(n):
        # The weight NIPALS converges to: the first left singular vector of X'Y.
        w = np.linalg.svd(X.T @ Y, full_matrices=False)[0][:, 0]
        t = X @ w
        W.append(w)
        P.append(X.T @ t / (t @ t))
        Q.append(Y.T @ t / (t @ t))
        X = X - np.outer(t, P[-1])
    W, P, Q = np.array(W).T, np.array(P).T, np.array(Q).T
    return W @ np.linalg.pinv(P.T @ W) @ Q.T


def pcr(X, Y, n):
    """PCR coefficients (features, targets) for autoscaled X and Y with n principal components."""
    V = np.linalg.svd(X, full_matrices=False)[2][:n].T
    return V @ np.linalg.lstsq(X @ V, Y, rcond=None)[0]


MODELS = {"pls": pls, "pcr": pcr, "hybrid": pls}


def fit_predict(method, n, X_train, Y_train, X_new):
    """Train method with n components and predict Y for X_new, in the units of Y_train."""
    (X_train, X_new), _, _ = autoscale(X_train, X_new)
    (Y_train,), mean, std = autoscale(Y_train)
    B = method(X_train, Y_train, n) if n else np.zeros((X_train.shape[1], Y_train.shape[1]))
    return X_new @ B * std + mean


def load(split):
    """X, log Y and the log of the physics estimate of the runs in split ("cal" or "test")."""
    with np.load(C.SPECTRA_FILE) as data:
        rows = data["split"] == split
        X, Y, run_id = data["X"][rows], data["Y"][rows], data["run_id"][rows]
    physics = pd.read_csv(C.PHYSICS_FILE)
    assert list(physics.run_id[rows]) == list(run_id), "physics.csv is out of date, rerun step 4"
    return X, np.log(Y), np.log(physics[[f"{t}_{HYBRID_BASE}" for t in TARGETS]].to_numpy()[rows])


def base(model, physics):
    """What the model adds its prediction to, in log units."""
    return physics if model == "hybrid" else 0.0


def rms_error(log_pred, log_true):
    """RMS relative error [%] per target, over the runs (second last axis)."""
    return 100.0 * np.sqrt(np.mean((np.exp(log_pred - log_true) - 1.0) ** 2, axis=-2))


def cross_validate(method, X, log_y, offset):
    """RMS relative error [%], shape (seeds, components 0..MAX_COMPONENTS, targets)."""
    errors = []
    for seed in C.CV_SEEDS:
        folds = np.array_split(np.random.default_rng(seed).permutation(len(X)), C.N_FOLDS)
        pred = np.empty((C.MAX_COMPONENTS + 1, *log_y.shape))
        for test in folds:
            train = np.setdiff1d(np.arange(len(X)), test)
            for n in range(C.MAX_COMPONENTS + 1):
                pred[n, test] = fit_predict(method, n, X[train], (log_y - offset)[train], X[test])
        errors.append(rms_error(pred + offset, log_y))
    return np.array(errors)


def build():
    X, log_y, physics = load("cal")
    (scaled,), _, _ = autoscale(X)
    share = np.cumsum(np.linalg.svd(scaled, compute_uv=False) ** 2)
    share /= share[-1]
    print(f"PCA of the {len(X)} cal spectra ({X.shape[1]} features): " + ", ".join(
        f"{np.searchsorted(share, q) + 1} components hold {100 * q:.0f} %" for q in (0.9, 0.99)))

    rows, table = [], {}
    hs, tp = rms_error(physics, log_y)
    table[f"physics ({HYBRID_BASE})"] = {"components": "-", "Hs RMS %": hs, "Tp RMS %": tp}
    for model, method in MODELS.items():
        errors = cross_validate(method, X, log_y, base(model, physics))
        mean = errors.mean(axis=0)
        rows += [{"model": model, "components": n, "Hs_rms": hs, "Tp_rms": tp}
                 for n, (hs, tp) in enumerate(mean)]
        n = int(np.argmin(mean.sum(axis=1)))
        spread = np.ptp(errors[:, n], axis=0)
        table[model] = {"components": n, "Hs RMS %": mean[n, 0], "Tp RMS %": mean[n, 1],
                        "Hs seed spread": spread[0], "Tp seed spread": spread[1]}
    pd.DataFrame(rows).to_csv(C.CV_FILE, index=False, float_format="%.4f")
    print(f"Wrote {C.CV_FILE}. {C.N_FOLDS}-fold CV on the cal runs, seeds {C.CV_SEEDS}:")
    print(pd.DataFrame.from_dict(table, orient="index").round(2).to_string(na_rep=""))


def check():
    """With every component, PLS and PCR are plain least squares; then the file."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    (X, ), _, _ = autoscale(X)
    (Y, ), _, _ = autoscale(X @ rng.normal(size=(6, 2)) + 0.1 * rng.normal(size=(40, 2)))
    ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    for method in (pls, pcr):
        assert np.allclose(method(X, Y, 6), ols), f"{method.__name__} with all components"
    # One component of PLS points along X'y, the covariance with the target.
    w = np.linalg.svd(X.T @ Y[:, :1], full_matrices=False)[0][:, 0]
    t = X @ w
    assert np.allclose(pls(X, Y[:, :1], 1), np.outer(w, Y[:, 0] @ t / (t @ t))), "one component"

    cv = pd.read_csv(C.CV_FILE)
    assert sorted(cv.model.unique()) == sorted(MODELS), "cv.csv is out of date"
    assert (cv.groupby("model").components.apply(list)
            == [list(range(C.MAX_COMPONENTS + 1))] * len(MODELS)).all()
    assert np.isfinite(cv[["Hs_rms", "Tp_rms"]].to_numpy()).all()
    print(f"No problems found: {len(MODELS)} models, 0-{C.MAX_COMPONENTS} components")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="check PLS and PCR against least squares, and the file")
    if parser.parse_args().check:
        check()
    else:
        build()
