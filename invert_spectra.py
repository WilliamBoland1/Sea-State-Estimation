"""Step 4: estimate Hs and Tp from the motion spectra with the RAOs alone.

Reads data/spectra.npz and writes data/physics.csv. Nothing is fitted to the runs: the
only inputs are the spectra, the RAOs in C.INVERSION_MAT and the sensor noise level.

    python invert_spectra.py           # estimate every run, score the cal runs
    python invert_spectra.py --check   # check the units, the smoothing and the fit

Two methods
-----------
divide  S_wave = S_motion / |RAO|^2 where |RAO| > RAO_FLOOR * its maximum.
        Hs = 4 sqrt(area), Tp = 1 / peak frequency. The quick baseline.
fit     The JONSWAP spectrum (Hs, Tp, gamma) whose motion spectrum
        |RAO|^2 S_wave, smoothed by the Welch window, + sensor noise
        is closest to the measured one in log scale, over the same bins as divide.
        Without the smoothing, narrow peaks at long Tp come out about 3 % too long.

What data/physics.csv holds (one row per run, in the order of runs.csv)
-----------------------------------------------------------------------
run_id, split
Hs_<method>, Tp_<method>   [m], [s] for divide_<motion>, fit_<motion> and fit_all
gamma_<method>             [-] for the fit methods. fit_all uses every motion at once.

The test runs are estimated too, for the hybrid model in step 5, but only the cal runs
are scored here.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import get_window

from ospfmusim.hull.helpfunctions.Wamit import load_mat_vessel
from ospfmusim.waves.helpfunctions.spectra import jonswap

import config as C

TARGETS = ["Hs", "Tp"]
RAO_FLOOR = 0.2             # use only the bins where |RAO| > this share of its maximum
DOF = {"heave": 2, "roll": 3, "pitch": 4}
TO_DEG = {"heave": 1.0, "roll": np.degrees(1.0), "pitch": np.degrees(1.0)}  # RAO table is in rad/m
LOWER, UPPER = (0.05, 2.0, 1.0), (20.0, 30.0, 10.0)     # Hs [m], Tp [s], gamma [-]
GAMMA_START = 3.3           # DNV's default; the fit is free to move it
OVERSAMPLE = 8              # model points per Welch bin
REACH = 4                   # Welch bins on each side that the window spreads a wave over
FITS = [*C.MOTIONS, "all"]
METHODS = [f"divide_{m}" for m in C.MOTIONS] + [f"fit_{name}" for name in FITS]


def fine_grid(freq):
    """The Welch bins freq [Hz] with OVERSAMPLE points per bin and REACH bins more each side."""
    step = (freq[1] - freq[0]) / OVERSAMPLE
    return freq[0] + step * np.arange(-REACH * OVERSAMPLE,
                                      (len(freq) - 1 + REACH) * OVERSAMPLE + 1)


def on_bins(values):
    """The values of a fine-grid array (last axis) at the Welch bins."""
    return values[..., REACH * OVERSAMPLE:-REACH * OVERSAMPLE:OVERSAMPLE]


def welch_kernel(dt):
    """How the Hann window of one Welch segment spreads a single frequency over the fine grid."""
    n = round(C.WELCH_SEGMENT_S / dt)
    power = np.abs(np.fft.fft(get_window("hann", n), n * OVERSAMPLE)) ** 2
    kernel = np.concatenate([power[-REACH * OVERSAMPLE:], power[:REACH * OVERSAMPLE + 1]])
    return kernel / kernel.sum()


def smooth(spectra, kernel):
    """Spectra on the fine grid (rows, fine) as Welch would see them at its bins (rows, bins)."""
    return np.array([np.convolve(row, kernel, "valid")[::OVERSAMPLE] for row in spectra])


def rao_squared(freq, heading_deg, mat=C.INVERSION_MAT):
    """|RAO|^2 per motion at freq [Hz], shape (motions, frequencies): (m/m)^2 and (deg/m)^2."""
    table = load_mat_vessel(mat, frequency_count=None)["motion_table"]
    omega = 2.0 * np.pi * freq
    rao = table.at_headings(table.at_frequencies(omega),
                            np.full(len(omega), np.radians(heading_deg)))
    return np.array([(TO_DEG[m] * np.abs(rao[DOF[m]])) ** 2 for m in C.MOTIONS])


def noise_level(dt):
    """Spectral density of the sensor noise per motion, one-sided [unit^2/Hz]."""
    return np.array([2.0 * C.SENSOR_NOISE_STD[m] ** 2 * dt for m in C.MOTIONS])


def wave_spectrum(freq, hs, tp, gamma):
    """JONSWAP spectrum per Hz [m^2/Hz], the one the simulator generates the waves from."""
    return 2.0 * np.pi * jonswap(2.0 * np.pi * freq, hs, tp, gamma)


def log_model(params, rao2, fine, kernel, noise):
    """log of the expected Welch spectra (motions, bins) for params = log(Hs, Tp, gamma).

    rao2 is on the fine grid, because the window smooths the motion spectrum, not the RAO.
    """
    return np.log(smooth(rao2 * wave_spectrum(fine, *np.exp(params)), kernel) + noise[:, None])


def divide(psd, rao2, freq):
    """Hs and Tp from one motion spectrum, divided by |RAO|^2 where the RAO is large enough."""
    keep = rao2 > RAO_FLOOR**2 * rao2.max()
    wave = psd[keep] / rao2[keep]
    return 4.0 * np.sqrt(wave.sum() * (freq[1] - freq[0])), 1.0 / freq[keep][np.argmax(wave)]


def fit(psd, rao2, fine, kernel, noise, start):
    """Hs, Tp and gamma of the JONSWAP spectrum that best explains psd (motions, bins).

    Only where |RAO| > RAO_FLOOR * its maximum, as in divide(): where the RAO is small, a
    small error in it is a large error in log scale, and the fit would follow it.
    """
    keep = on_bins(rao2) > RAO_FLOOR**2 * rao2.max(axis=1, keepdims=True)
    result = least_squares(
        lambda p: (log_model(p, rao2, fine, kernel, noise) - np.log(psd))[keep],
        np.log(start), bounds=(np.log(LOWER), np.log(UPPER)))
    return np.exp(result.x)


def estimate(psd, rao2, fine, kernel, noise):
    """Every method on the spectra of one run (motions, bins), as {column: value}.

    rao2 is on the fine grid of fine_grid(), kernel from welch_kernel().
    """
    out = {}
    for m, motion in enumerate(C.MOTIONS):
        out[f"Hs_divide_{motion}"], out[f"Tp_divide_{motion}"] = divide(
            psd[m], on_bins(rao2[m]), on_bins(fine))
    for name in FITS:
        rows = list(range(len(C.MOTIONS))) if name == "all" else [C.MOTIONS.index(name)]
        first = C.MOTIONS[rows[0]]      # start from the division of the first motion used
        start = (out[f"Hs_divide_{first}"], out[f"Tp_divide_{first}"], GAMMA_START)
        hs, tp, gamma = fit(psd[rows], rao2[rows], fine, kernel, noise[rows],
                            np.clip(start, LOWER, UPPER))
        out[f"Hs_fit_{name}"], out[f"Tp_fit_{name}"], out[f"gamma_fit_{name}"] = hs, tp, gamma
    return out


def estimate_all(mat=C.INVERSION_MAT):
    """Every method on every run of spectra.npz, as a frame in the order of runs.csv."""
    runs = pd.read_csv(C.RUNS_FILE)
    with np.load(C.SPECTRA_FILE) as data:
        psd, freq = data["psd"], data["freq"]
        assert list(data["motions"]) == C.MOTIONS, "spectra.npz is out of date, rerun step 3"
        assert list(data["run_id"]) == list(runs.run_id), "rows must follow runs.csv"
    fine = fine_grid(freq)
    raos = {h: rao_squared(fine, h, mat) for h in runs.heading_deg.unique()}
    kernels = {dt: welch_kernel(dt) for dt in runs.dt_s.unique()}
    rows = [estimate(psd[i], raos[row.heading_deg], fine, kernels[row.dt_s],
                     noise_level(row.dt_s))
            for i, row in enumerate(runs.itertuples())]
    return runs, pd.concat([runs[["run_id", "split"]], pd.DataFrame(rows)], axis=1)


def report(runs, est):
    """RMS and mean (bias) relative error [%] of every method, on the cal runs only."""
    cal = runs.split == "cal"
    table = {}
    for method in METHODS:
        table[method] = {}
        for target in TARGETS:
            rel = est[f"{target}_{method}"][cal] / runs[target][cal] - 1.0
            table[method][f"{target} RMS %"] = 100.0 * np.sqrt(np.mean(rel**2))
            table[method][f"{target} bias %"] = 100.0 * np.mean(rel)
    print(pd.DataFrame(table).T.round(2).to_string())


def build():
    runs, est = estimate_all()
    est.to_csv(C.PHYSICS_FILE, index=False, float_format="%.6g")
    print(f"Wrote {C.PHYSICS_FILE}: {len(est)} runs. "
          f"Errors on the {(runs.split == 'cal').sum()} cal runs, RAOs from {C.INVERSION_MAT.name}:")
    report(runs, est)


def check():
    """Known answers for the units, the smoothing and the fit, then the file against runs.csv."""
    from build_spectra import in_band, spectrum

    # Long waves lift the ship as they are: heave RAO 1 m/m, pitch RAO the wave slope k.
    low = np.array([0.04])                # 25 s waves, inside the RAO table
    rao2 = rao_squared(low, 180.0)
    k = (2.0 * np.pi * low[0]) ** 2 / C.G
    assert np.isclose(np.sqrt(rao2[C.MOTIONS.index("heave")]), 1.0, rtol=0.05), "heave RAO"
    assert np.isclose(np.sqrt(rao2[C.MOTIONS.index("pitch")]), np.degrees(k), rtol=0.1), "pitch RAO"
    f = np.linspace(0.001, 2.0, 20000)
    assert np.isclose(np.trapezoid(wave_spectrum(f, 2.0, 9.0, 3.3), f), 2.0**2 / 16.0, rtol=0.02), \
        "area of the wave spectrum must be Hs^2 / 16"

    # A sine between two bins must be smeared over the bins as Welch itself does it.
    dt, amplitude = C.DT_S, 0.5
    freq = np.fft.rfftfreq(round(C.WELCH_SEGMENT_S / dt), dt)      # the bins welch() returns
    freq = freq[in_band(freq)]
    fine, kernel = fine_grid(freq), welch_kernel(dt)
    line = 20 * OVERSAMPLE + 3            # a fine point 3/8 of a bin above bin 16
    t = np.arange(0.0, C.DURATION_S, dt)
    welch_freq, welch_psd = spectrum(amplitude * np.sin(2.0 * np.pi * fine[line] * t), dt)
    spike = np.zeros((1, len(fine)))
    spike[0, line] = amplitude**2 / 2.0 / (fine[1] - fine[0])
    predicted = smooth(spike, kernel)[0]
    near = predicted > 0.01 * predicted.max()
    measured = welch_psd[in_band(welch_freq)][near]
    assert np.allclose(predicted[near], measured, rtol=0.02), "smoothing differs from Welch"

    # A spectrum made with the model must give back its own sea state, from every motion.
    truth = np.array([2.0, 9.0, 3.3])
    rao2, noise = rao_squared(fine, 180.0), noise_level(dt)
    psd = np.exp(log_model(np.log(truth), rao2, fine, kernel, noise))
    est = estimate(psd, rao2, fine, kernel, noise)
    for name in FITS:
        found = [est[f"Hs_fit_{name}"], est[f"Tp_fit_{name}"], est[f"gamma_fit_{name}"]]
        assert np.allclose(found, truth, rtol=1e-3), f"fit_{name} found {found}"

    runs, physics = pd.read_csv(C.RUNS_FILE), pd.read_csv(C.PHYSICS_FILE)
    assert list(physics.run_id) == list(runs.run_id), "rows must follow runs.csv"
    assert list(physics.split) == list(runs.split), "rows must follow runs.csv"
    columns = [f"{t}_{m}" for m in METHODS for t in TARGETS]
    assert np.isfinite(physics[columns].to_numpy()).all()
    print(f"No problems found: {len(physics)} runs, {len(METHODS)} methods")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="check the units, the smoothing, the fit and the file")
    if parser.parse_args().check:
        check()
    else:
        build()
