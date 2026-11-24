"""Step 3: load the runs, make Welch spectra and build X and Y.

Writes data/spectra.npz and figures/spectra.png. It takes seconds and does not
need OSP, so the settings in config.py can be changed and this run again.

    python build_spectra.py           # build
    python build_spectra.py --check   # check the spectrum calculation and the file

For every run: keep the analysed record (time >= ramp_s), add sensor noise, and
make one Welch spectrum per motion in C.MOTIONS.

What data/spectra.npz holds (n runs, m motions, k frequencies)
--------------------------------------------------------------
psd      (n, m, k)  the spectra: heave [m^2/Hz], pitch [deg^2/Hz]. For the physics inversion.
X        (n, m * k) log10(psd), one row per run, one motion after the other. For PCA and PLSR.
Y        (n, 2)     the targets, in the order of `targets`: Hs [m], Tp [s]
freq     (k,)       frequency [Hz]
motions, targets, run_id, split

X is not scaled. Fit any scaling on the cal runs only, so nothing from the test
runs leaks into the model. The figure only shows cal runs, for the same reason.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch

import config as C

TARGETS = ["Hs", "Tp"]
INK, SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"     # every figure's colours
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def spectrum(signal, dt):
    """Welch spectrum of a signal sampled every dt [s]: (frequency [Hz], one-sided density)."""
    nperseg = round(C.WELCH_SEGMENT_S / dt)
    return welch(signal, fs=1.0 / dt, nperseg=nperseg, noverlap=round(nperseg * C.WELCH_OVERLAP))


def pieces(signal, dt):
    """The spectrum of every Welch segment: (frequency [Hz], density (frequencies, segments)).

    Same settings as spectrum(), whose result is their mean.
    """
    nperseg = round(C.WELCH_SEGMENT_S / dt)
    freq, _, density = spectrogram(signal, fs=1.0 / dt, window="hann", nperseg=nperseg,
                                   noverlap=round(nperseg * C.WELCH_OVERLAP), mode="psd")
    return freq, density


def in_band(freq):
    return (freq >= C.FREQ_RANGE_HZ[0]) & (freq <= C.FREQ_RANGE_HZ[1])


def noisy_record(row, folder=C.OUTPUT_DIR, noise_std=C.SENSOR_NOISE_STD):
    """The analysed record of one row of runs.csv plus sensor noise, shape (motions, samples).

    folder holds the output file; noise_std is the sensor noise per motion. Another
    noise_std scales the same random draws, so runs stay comparable.
    """
    data = pd.read_csv(folder / row.output_file, usecols=["time", *C.MOTIONS])
    record = data[data.time >= row.ramp_s]
    if len(record) != round(row.duration_s / row.dt_s):
        raise ValueError(f"{row.run_id}: expected {row.duration_s} s of data, "
                         f"found {len(record)} samples")
    rng = np.random.default_rng([C.NOISE_SEED, int(row.wave_seed)])    # same noise every time
    return np.array([record[m].to_numpy() + rng.normal(0.0, noise_std[m], len(record))
                     for m in C.MOTIONS])


def run_spectra(row, folder=C.OUTPUT_DIR, noise_std=C.SENSOR_NOISE_STD):
    """The spectra of one row of runs.csv, shape (motions, frequencies), and their frequencies."""
    spectra = [spectrum(x, row.dt_s) for x in noisy_record(row, folder, noise_std)]
    keep = in_band(spectra[0][0])
    return np.array([density[keep] for _, density in spectra]), spectra[0][0][keep]


def build():
    runs = pd.read_csv(C.RUNS_FILE)
    spectra = [run_spectra(row) for row in runs.itertuples()]
    psd = np.array([s for s, _ in spectra])
    freq = spectra[0][1]
    np.savez(C.SPECTRA_FILE, psd=psd, X=np.log10(psd).reshape(len(runs), -1),
             Y=runs[TARGETS].to_numpy(), freq=freq, motions=C.MOTIONS, targets=TARGETS,
             run_id=runs.run_id.to_numpy(dtype=str), split=runs.split.to_numpy(dtype=str))
    plot_spectra(runs, psd, freq)
    print(f"Wrote {C.SPECTRA_FILE}: {len(runs)} runs, {len(C.MOTIONS)} motions, "
          f"{len(freq)} frequencies, {freq[0]:.3f}-{freq[-1]:.3f} Hz, "
          f"every {freq[1] - freq[0]:.4f} Hz")


def plot_spectra(runs, psd, freq):
    """Three cal runs of about the same Hs, from short to long waves, and the sensor noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = {"heave": "m²/Hz", "roll": "deg²/Hz", "pitch": "deg²/Hz"}
    cal = runs[runs.split == "cal"]
    chosen = [(abs(cal.Tp - tp) + abs(cal.Hs - 2.0)).idxmin() for tp in (5.5, 9.0, 13.0)]

    fig, axes = plt.subplots(1, len(C.MOTIONS), figsize=(8.5, 3.9), sharex=True,
                             facecolor=SURFACE, layout="constrained")
    for m, (ax, motion) in enumerate(zip(axes, C.MOTIONS)):
        for i, color in zip(chosen, COLORS):
            line = psd[i, m]
            peak = np.argmax(line)
            ax.semilogy(freq, line, color=color, lw=1.4, label=f"Tp {runs.Tp[i]:.1f} s")
            ax.annotate(f"{runs.Tp[i]:.1f} s", (freq[peak], line[peak]), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8, color=SOFT)
        floor = 2.0 * C.SENSOR_NOISE_STD[motion] ** 2 * runs.dt_s.iloc[0]
        ax.axhline(floor, color=SOFT, lw=1.0, ls=(0, (4, 3)), label="Sensor noise")
        ax.set_ylim(floor / 4.0, psd[chosen, m].max() * 6.0)
        ax.set_xlim(*C.FREQ_RANGE_HZ)
        style(ax, motion.capitalize(), "Frequency [Hz]", f"Spectral density [{units[motion]}]")
    save(fig, axes, "Motion spectra for three calibration runs, Hs about 2 m", "spectra.png")


def style(ax, title, xlabel, ylabel):
    """The look of every panel of the project's figures."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=SOFT)
    ax.set_ylabel(ylabel, fontsize=9, color=SOFT)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=SOFT, labelsize=8, length=0, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, axes, title, name):
    """The legend of the first panel under the figure, the title over it, then figures/<name>.

    title=None leaves the title out, for figures whose caption is in the report.
    """
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=8,
               labelcolor=SOFT)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / name, dpi=300 if title is None else 150)


def check():
    """Known answers for the calculation, then the file against runs.csv."""
    dt, f0, amplitude, std = 0.5, 16.0 / C.WELCH_SEGMENT_S, 2.0, 0.3
    t = np.arange(0.0, C.DURATION_S, dt)
    freq, density = spectrum(amplitude * np.sin(2.0 * np.pi * f0 * t), dt)
    assert np.isclose(freq[np.argmax(density)], f0), "a sine must peak at its own frequency"
    assert np.isclose(np.trapezoid(density, freq), amplitude**2 / 2.0, rtol=0.01), "sine area"
    noise = np.random.default_rng(0).normal(0.0, std, len(t))
    _, density = spectrum(noise, dt)
    assert np.isclose(density.mean(), 2.0 * std**2 * dt, rtol=0.05), "white noise level"
    assert np.allclose(pieces(noise, dt)[1].mean(axis=1), density), "pieces must average to Welch"

    runs = pd.read_csv(C.RUNS_FILE)
    with np.load(C.SPECTRA_FILE) as data:
        psd, X, Y, freq = data["psd"], data["X"], data["Y"], data["freq"]
        assert list(data["motions"]) == C.MOTIONS and list(data["targets"]) == TARGETS
        assert list(data["run_id"]) == list(runs.run_id), "rows must follow runs.csv"
        assert list(data["split"]) == list(runs.split), "rows must follow runs.csv"
    assert psd.shape == (len(runs), len(C.MOTIONS), len(freq))
    assert X.shape == (len(runs), len(C.MOTIONS) * len(freq)) and np.isfinite(X).all()
    assert np.array_equal(Y, runs[TARGETS].to_numpy())
    assert in_band(freq).all() and np.allclose(np.diff(freq), 1.0 / C.WELCH_SEGMENT_S)
    # X must be the spectra, motion after motion. Row 0 is enough to catch a wrong order.
    assert np.allclose(X[0, :len(freq)], np.log10(psd[0, 0]))
    assert np.allclose(X[0, -len(freq):], np.log10(psd[0, -1]))
    print(f"No problems found: {len(runs)} runs, {X.shape[1]} features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check the calculation and the file")
    if parser.parse_args().check:
        check()
    else:
        build()
