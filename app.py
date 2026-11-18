"""The project step by step, one page per step:  .venv\\Scripts\\streamlit run app.py

Reads only saved results (data/ and the two .mat files): nothing is simulated or fitted
again. Pick one test run in the sidebar to follow it through the pages.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.io import loadmat

import config as C
from build_spectra import COLORS, INK, SOFT
from cross_validate import HYBRID_BASE, TARGETS, rms_error
from design_sea_states import hs_upper
from evaluate import COLOR, cal_errors, chosen
from invert_spectra import RAO_FLOOR, noise_level, rao_squared, wave_spectrum

NAME = {HYBRID_BASE: "physics", "pls": "PLS", "pcr": "PCR", "hybrid": "hybrid"}
UNIT = {"heave": "m", "pitch": "deg", "Hs": "m", "Tp": "s"}
LIGHT = "#c9c8c3"   # the runs that are not the chosen one


@st.cache_data
def load():
    runs = pd.read_csv(C.RUNS_FILE)
    with np.load(C.SPECTRA_FILE) as data:
        psd, freq = data["psd"], data["freq"]
    components = chosen()
    return (runs, psd, freq, pd.read_csv(C.PHYSICS_FILE), pd.read_csv(C.TEST_FILE),
            pd.read_csv(C.CV_FILE), pd.read_csv(C.ROBUST_FILE), components, cal_errors(components))


@st.cache_data
def rao(mat, freq, heading_deg):
    """|RAO| per motion at freq [Hz]: m/m and deg/m."""
    return np.sqrt(rao_squared(freq, heading_deg, mat))


@st.cache_data
def ship_size():
    """Length and beam of the true ship [m]."""
    main = loadmat(str(C.TRUE_MAT), squeeze_me=True, struct_as_record=False)["vessel"].main
    return float(main.Lpp), float(main.B)


def test_error(method):
    """RMS relative error [%] of a method on the test runs, (Hs, Tp)."""
    return rms_error(np.log(test[[f"{t}_{method}" for t in TARGETS]].to_numpy()),
                     np.log(test[TARGETS].to_numpy()))


def panels(titles):
    return make_subplots(rows=1, cols=len(titles), subplot_titles=titles, horizontal_spacing=0.1)


def show(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0),
                      legend=dict(orientation="h", yanchor="top", y=-0.2))
    st.plotly_chart(fig)


def end(takeaway, detail):
    """The takeaway, the details folded away, and a link to the next page."""
    st.markdown(f"**{takeaway}**")
    with st.expander("More detail"):
        st.markdown(detail)
    k = [p.title for p in pages].index(page.title)
    last = k + 1 == len(pages)
    st.page_link(pages[0 if last else k + 1], icon=":material/arrow_forward:",
                 label="Back to the start" if last else f"Next: {pages[k + 1].title}")


# ---------------------------------------------------------------- pages
def start():
    st.title("Reading the sea from the ship")
    st.markdown(
        "A ship moves with the waves. This project turns that around: measure how the ship moves "
        "and work out the waves from it. Two numbers describe the waves:\n"
        "- **Hs**, the significant wave height: how high the waves are [m]\n"
        "- **Tp**, the peak period: the time between the biggest wave crests [s]\n\n"
        "**Waves** (Hs, Tp) → **ship** (its RAO: how much it moves per metre of wave) → "
        "**motions** (heave, pitch) → **spectrum** → **estimate** of Hs and Tp")
    with st.container(border=True):
        assumptions()
    st.subheader("The steps")
    for p, (_, _, line) in zip(pages[1:], STEPS):
        st.page_link(p, label=f"{p.title}: {line}")
    st.subheader(f"The answer, on {len(test)} sea states the methods never saw")
    hybrid, alone = test_error("hybrid"), test_error(HYBRID_BASE)
    for col, t, target in zip(st.columns(2), range(len(TARGETS)), TARGETS):
        col.metric(f"{target} error, hybrid", f"{hybrid[t]:.1f} %", delta_color="inverse",
                   delta=f"{hybrid[t] - alone[t]:+.1f} points against physics alone")
    st.caption("RMS relative error. Pick a sea state in the sidebar and follow it through the steps.")


def assumptions():
    """Where the waves come from and what they are, drawn for the chosen sea state."""
    st.subheader("What we assume")
    length, beam = ship_size()
    wavelength = C.G * row.Tp ** 2 / (2 * np.pi)       # deep water
    r = max(1.6 * wavelength, 1.2 * length)
    fig = panels(["Seen from above", "Seen from the side, at one point"])
    crests = np.arange(r - wavelength / 2, -r, -wavelength)
    fig.add_scatter(x=np.tile([-r, r, None], len(crests)), y=np.repeat(crests, 3),
                    mode="lines", line=dict(color=COLORS[0], width=2), showlegend=False,
                    hoverinfo="skip", row=1, col=1)
    fig.add_scatter(x=[0, beam / 2, beam / 2, -beam / 2, -beam / 2, 0],
                    y=[length / 2, length / 4, -length / 2, -length / 2, length / 4, length / 2],
                    fill="toself", mode="lines", line=dict(color=INK), showlegend=False,
                    hoverinfo="skip", row=1, col=1)
    fig.add_annotation(x=beam, y=0, xref="x", yref="y", text="ship", showarrow=False,
                       xanchor="left")
    fig.add_annotation(x=r, y=crests[0], xref="x", yref="y", text="wave crest",
                       showarrow=False, xanchor="right", yanchor="bottom")
    fig.add_annotation(x=0, y=length / 2 + 0.05 * r, ax=0, ay=0.92 * r, xref="x", yref="y",
                       axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=2,
                       arrowcolor=INK, text="waves come from here", bgcolor="white")
    fig.update_xaxes(range=[-r, r], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-r, r], visible=False, scaleanchor="x", row=1, col=1)

    # The same spectrum as the run, with our own random phases: not the run's exact waves.
    f = np.linspace(0.02, 0.5, 400)
    amp = np.sqrt(2 * wave_spectrum(f, row.Hs, row.Tp, row.gamma) * (f[1] - f[0]))
    phase = np.random.default_rng(0).uniform(0, 2 * np.pi, f.size)
    t = np.arange(0, 120, 0.25)
    fig.add_scatter(x=t, y=amp @ np.cos(2 * np.pi * np.outer(f, t) + phase[:, None]),
                    mode="lines", line=dict(color=COLORS[0], width=1.5), showlegend=False,
                    hovertemplate="%{x:.0f} s: %{y:+.2f} m<extra></extra>", row=1, col=2)
    for y in (-row.Hs / 2, row.Hs / 2):
        fig.add_hline(y=y, line=dict(color=SOFT, dash="dash", width=1), row=1, col=2)
    fig.update_xaxes(title="Time [s]", row=1, col=2)
    fig.update_yaxes(title="Water surface [m]", row=1, col=2)
    show(fig, 380)
    st.caption(f"Your sea state, {run_id}: Hs {row.Hs:.1f} m, Tp {row.Tp:.1f} s. Left, to scale: "
               f"crests one wavelength apart ({wavelength:.0f} m), ship {length:.0f} m long. "
               "Right: what this sea looks like passing one point; the dashed lines are Hs "
               "apart. Same spectrum as the run, not its exact waves.")
    st.markdown(
        "- **From straight ahead.** Every wave meets the bow (head seas, 180°) and all travel "
        "the same way (long-crested). The methods are given this; they do not work it out.\n"
        f"- **One simple sea.** Irregular waves from one JONSWAP spectrum (γ = {C.GAMMA}), set "
        f"by Hs and Tp alone and steady over the {C.DURATION_S / 60:.0f} min record. No swell "
        "on top, no wind, no current.\n"
        "- **The ship stands still.** Dynamic positioning (DP) holds it in place: no forward "
        "speed.\n"
        f"- **A good sensor.** Heave and pitch, with a little noise "
        f"({C.SENSOR_NOISE_STD['heave']} m, {C.SENSOR_NOISE_STD['pitch']}°) of known size.\n\n"
        "Other headings, mixed seas or a moving ship are not tested.")


def design():
    st.title("1. Design: which sea states?")
    st.markdown(
        f"To learn and to test we need many different seas. {len(runs)} sea states are spread "
        f"evenly over height Hs and period Tp: {C.N_CAL} to fit the methods on (*calibration*) and "
        f"{C.N_TEST} kept hidden until the very end (*test*). Short waves cannot be very high: "
        f"steeper than 1 in {1 / C.MAX_STEEPNESS:.0f} they break, so no sea state lies above the "
        "dashed line.")
    tp = np.linspace(*C.TP_RANGE, 200)
    fig = go.Figure()
    fig.add_scatter(x=tp, y=hs_upper(tp), mode="lines", name="highest possible Hs",
                    line=dict(color=SOFT, dash="dash", width=1.5), hoverinfo="skip")
    hover = "%{text}<br>Tp %{x:.1f} s<br>Hs %{y:.2f} m<extra></extra>"
    for split, label, color in (("cal", "calibration", LIGHT), ("test", "test", COLORS[0])):
        d = runs[runs.split == split]
        fig.add_scatter(x=d.Tp, y=d.Hs, mode="markers", name=f"{label} ({len(d)})", text=d.run_id,
                        marker=dict(color=color, size=8), hovertemplate=hover)
    fig.add_scatter(x=[row.Tp], y=[row.Hs], mode="markers+text", name="your sea state",
                    text=[run_id], textposition="top center", hovertemplate=hover,
                    marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=INK, width=2.5)))
    fig.update_xaxes(title="Tp, peak period [s]")
    fig.update_yaxes(title="Hs, wave height [m]")
    show(fig, 420)
    end("The sea states cover the whole range evenly, and the test ones are new points, "
        "not copies of the calibration ones.",
        "Latin hypercube sampling spreads the points: it draws Tp, and the share of the allowed "
        "height range for that Tp, so no point is thrown away. Every run has the waves straight "
        f"at the bow (head seas), a JONSWAP spectrum with γ = {C.GAMMA} and "
        f"{C.DURATION_S / 60:.0f} min of record.  \n"
        "Script: `design_sea_states.py` → `data/runs.csv`")


def simulate():
    st.title("2. Simulate: let the ship ride each sea")
    st.markdown(
        "OSP is a ship simulator. In it the research vessel Gunnerus holds its position with its "
        "thrusters (dynamic positioning, DP) while the waves come straight at the bow. Each sea "
        f"state runs {row.ramp_s:.0f} s to settle (shaded, thrown away), then "
        f"{row.duration_s / 60:.0f} min that we keep. We log **heave** (up and down) and "
        "**pitch** (bow up and down).")
    path = C.OUTPUT_DIR / row.output_file
    if not path.exists():
        st.info(f"The OSP output is not on this computer (it is not in git): "
                f"`python run_osp.py {run_id}` makes it.")
    else:
        motions = pd.read_csv(path, usecols=["time", *C.MOTIONS])
        fig = make_subplots(rows=len(C.MOTIONS), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                            subplot_titles=["Heave [m], positive down", "Pitch [deg]"])
        for r, motion in enumerate(C.MOTIONS, 1):
            fig.add_scatter(x=motions.time, y=motions[motion], mode="lines", showlegend=False,
                            name=motion, line=dict(color=COLORS[0], width=1), row=r, col=1)
        fig.add_vrect(x0=0, x1=row.ramp_s, fillcolor=LIGHT, opacity=0.5, line_width=0,
                      row="all", col=1)
        fig.update_xaxes(title="Time [s]", row=len(C.MOTIONS), col=1, range=[0, 300])
        show(fig, 440)
        st.caption(f"The first 5 min. Double-click the plot to see all "
                   f"{(row.ramp_s + row.duration_s) / 60:.0f} min.")
    end("The motions look random, but how big and how fast the ship swings holds the sea state.",
        "The waves are a sum of regular waves drawn from a JONSWAP spectrum with this run's Hs and "
        "Tp, with random phases (`wave_seed`). How the ship answers them comes from the RAO table "
        "in `gunnerus.mat`: from here on, *the true ship*. Output every "
        f"{row.dt_s} s. Roll is not used: in head seas the waves do not roll the ship.  \n"
        "Script: `run_osp.py` → `data/sim_output/`")


def spectra():
    st.title("3. Spectra: from wiggles to a fingerprint")
    st.markdown(
        "A spectrum sorts a signal by how fast it swings and shows how much motion there is at "
        "each frequency. Long waves (large Tp) give a peak to the left, short waves to the right, "
        "and higher waves lift the whole curve. A little sensor noise is added first, as a real "
        "motion sensor would have.")
    fig = panels([f"{m.capitalize()} [{UNIT[m]}²/Hz]" for m in C.MOTIONS])
    others = np.flatnonzero((runs.split == "test") & (runs.run_id != run_id))
    noise = noise_level(row.dt_s)
    for m in range(len(C.MOTIONS)):
        first = m == 0
        for n, j in enumerate(others):
            fig.add_scatter(x=freq, y=psd[j, m], mode="lines", name="other test runs",
                            legendgroup="others", showlegend=first and n == 0, hoverinfo="skip",
                            line=dict(color=LIGHT, width=1), row=1, col=m + 1)
        fig.add_scatter(x=freq[[0, -1]], y=[noise[m]] * 2, mode="lines", name="sensor noise",
                        legendgroup="noise", showlegend=first, hoverinfo="skip",
                        line=dict(color=SOFT, dash="dash", width=1.5), row=1, col=m + 1)
        fig.add_scatter(x=freq, y=psd[i, m], mode="lines", name=run_id, legendgroup="run",
                        showlegend=first, line=dict(color=COLORS[0], width=2.5), row=1, col=m + 1)
    fig.add_vline(x=1 / row.Tp, line=dict(color=INK, dash="dot", width=1), row="all", col="all",
                  annotation_text="1 / Tp")
    fig.update_yaxes(type="log", exponentformat="power")
    fig.update_xaxes(title="Frequency [Hz]")
    show(fig)
    end(f"Where the peak sits gives Tp, the area under it gives Hs. These {len(C.MOTIONS)} × "
        f"{len(freq)} = {psd[0].size} numbers are all that any method sees of a sea state.",
        f"Welch's method: the record is cut into {C.WELCH_SEGMENT_S:.0f} s pieces that overlap "
        f"by {100 * C.WELCH_OVERLAP:.0f} %, each gets a spectrum, and they are averaged, so the "
        f"frequency step is 1/{C.WELCH_SEGMENT_S:.0f} Hz. Only {C.FREQ_RANGE_HZ[0]}-"
        f"{C.FREQ_RANGE_HZ[1]} Hz is kept: outside it there is mostly noise. The sensor noise "
        f"(std {C.SENSOR_NOISE_STD['heave']} m and {C.SENSOR_NOISE_STD['pitch']} deg) is an "
        "assumed value, not from a data sheet.  \n"
        "Script: `build_spectra.py` → `data/spectra.npz`")


def physics():
    st.title("4. Physics: undo the ship")
    st.markdown(
        "The ship works like a filter. At each frequency it turns one metre of wave into a certain "
        "amount of motion: its **RAO** (response amplitude operator). So the wave spectrum is the "
        "motion spectrum divided by |RAO|²:")
    st.latex(r"S_\text{motion}(f) = |\text{RAO}(f)|^2 \, S_\text{wave}(f)")
    st.markdown(
        "The catch: nobody knows the RAO of a real ship exactly. The runs were made with one RAO "
        "table (*the true ship*); the method only gets another table of the same vessel "
        "(*our model*), as it would at sea.")
    fine = np.linspace(*C.FREQ_RANGE_HZ, 300)
    fig = panels([f"{m.capitalize()} RAO [{UNIT[m]}/m]" for m in C.MOTIONS])
    for mat, name, color in ((C.TRUE_MAT, "true ship (made the runs)", SOFT),
                             (C.INVERSION_MAT, "our model (used here)", COLORS[0])):
        amp = rao(str(mat), fine, row.heading_deg)
        for m in range(len(C.MOTIONS)):
            fig.add_scatter(x=fine, y=amp[m], mode="lines", name=name, legendgroup=name,
                            showlegend=m == 0, line=dict(color=color, width=2), row=1, col=m + 1)
    fig.update_xaxes(title="Frequency [Hz]")
    show(fig, 320)

    st.markdown(f"**Your sea state, {run_id}:** the true wave spectrum and what the method gets back.")
    rao2 = rao(str(C.INVERSION_MAT), freq, row.heading_deg) ** 2
    est = phys.iloc[i]
    fig = go.Figure()
    fig.add_scatter(x=fine, y=wave_spectrum(fine, row.Hs, row.Tp, row.gamma), mode="lines",
                    name="true waves", line=dict(color=SOFT, dash="dash", width=2))
    for m, motion in enumerate(C.MOTIONS):
        keep = rao2[m] > RAO_FLOOR**2 * rao2[m].max()
        fig.add_scatter(x=freq[keep], y=psd[i, m, keep] / rao2[m, keep], mode="markers",
                        name=f"{motion} / |RAO|²", marker=dict(color=COLORS[m], size=8))
    fig.add_scatter(x=fine, y=wave_spectrum(fine, *est[[f"{p}_{HYBRID_BASE}" for p in
                                                        ("Hs", "Tp", "gamma")]]),
                    mode="lines", name="best-fitting JONSWAP", line=dict(color=COLORS[2], width=2.5))
    fig.update_xaxes(title="Frequency [Hz]")
    fig.update_yaxes(title="Wave spectrum [m²/Hz]")
    show(fig)
    for col, target in zip(st.columns(len(TARGETS)), TARGETS):
        value, true = est[f"{target}_{HYBRID_BASE}"], row[target]
        col.metric(f"{target} from physics", f"{value:.2f} {UNIT[target]}", delta_color="off",
                   delta=f"{100 * (value / true - 1):+.1f} % (true {true:.2f} {UNIT[target]})")
    hs, tp = cal_err[HYBRID_BASE]
    end(f"Over the {C.N_CAL} calibration runs Tp comes out within {tp:.1f} %: the peak sits in the same "
        f"place whatever the RAO. Hs is {hs:.1f} % off: a slightly wrong RAO changes the area under "
        "the curve, and with it Hs.",
        "Two ways. **Divide** (the dots): the formula above, only where |RAO| is above "
        f"{100 * RAO_FLOOR:.0f} % of its peak, since dividing by almost zero blows up the noise; "
        "Hs = 4 √area and Tp = 1 / peak frequency. **Fit** (the green line): the JONSWAP spectrum "
        "(Hs, Tp, γ) whose motion spectra best match the measured heave and pitch spectra at once, "
        "with the smoothing of Welch's method and the noise floor included. The fit is *the "
        "physics estimate* from here on. Nothing is fitted to the runs: the only inputs are the "
        "spectra, our RAO model and the noise level. With the true ship's RAO the error almost "
        "disappears, so what is left comes from the RAO mismatch.  \n"
        "Script: `invert_spectra.py` → `data/physics.csv`")


def learn():
    st.title("5. Learn: let data correct the physics")
    st.markdown(
        "The physics error is not random: similar seas get similar mistakes, and data can learn "
        f"that. Three models, all fed the {psd[0].size} spectrum numbers:\n"
        "- **PLS** (partial least squares) finds the few combinations of the numbers that best "
        "predict Hs and Tp\n"
        "- **PCR** (principal component regression) first finds the main patterns in the "
        "spectra, then fits a straight line on them\n"
        "- **hybrid** starts from the physics estimate and lets PLS learn only the correction\n\n"
        "Each has one knob: how many patterns (*components*) to use. To set it fairly, a model is "
        f"fitted on 4/5 of the {C.N_CAL} calibration runs and scored on the last 5th, for each of the "
        f"{C.N_FOLDS} parts in turn (cross-validation). The test runs stay hidden.")
    fig = panels([f"{t} error [%]" for t in TARGETS])
    for model, n in components.items():
        d = cv[cv.model == model]
        for t, target in enumerate(TARGETS):
            fig.add_scatter(x=d.components, y=d[f"{target}_rms"], mode="lines", name=NAME[model],
                            legendgroup=model, showlegend=t == 0,
                            line=dict(color=COLOR[model], width=2), row=1, col=t + 1)
            fig.add_scatter(x=[n], y=[d[f"{target}_rms"].iloc[n]], mode="markers",
                            legendgroup=model, showlegend=False, marker=dict(color=COLOR[model], size=10),
                            hovertemplate=f"{NAME[model]}: {n} components<extra></extra>",
                            row=1, col=t + 1)
    for t in range(len(TARGETS)):
        fig.add_scatter(x=[0, C.MAX_COMPONENTS], y=[cal_err[HYBRID_BASE][t]] * 2, mode="lines",
                        name="physics alone", legendgroup="physics", showlegend=t == 0,
                        line=dict(color=SOFT, dash="dash", width=1.5), row=1, col=t + 1)
    fig.update_yaxes(type="log", tickvals=[0.5, 1, 2, 5, 10, 20, 50])
    fig.update_xaxes(title="Components")
    show(fig)
    hs, tp = cal_err["hybrid"]
    end("Too few components miss the pattern, too many start to learn noise; the dot marks the "
        f"lowest point. The hybrid needs only {components['hybrid']} components and ends far below "
        f"the others: Hs {hs:.1f} %, Tp {tp:.1f} %.",
        "The inputs are the log10 spectra, the targets log Hs and log Tp, both scaled to mean 0 "
        f"and std 1 on the training part only. {C.N_FOLDS} folds, repeated with "
        f"{len(C.CV_SEEDS)} different splits and averaged. The number of components with the "
        "lowest Hs + Tp error is chosen.  \n"
        "Script: `cross_validate.py` → `data/cv.csv`")


def exam():
    st.title("6. Test: the exam")
    st.markdown(
        f"Now the {len(test)} hidden sea states, scored once. Every model is fitted on all "
        f"{C.N_CAL} calibration runs with the components chosen in step 5. Nothing was changed after this.")
    shown = [HYBRID_BASE, "pcr", "pls", "hybrid"]
    errors = {m: test_error(m) for m in shown}
    fig = panels([f"{t} error [%]" for t in TARGETS])
    for t in range(len(TARGETS)):
        values = [errors[m][t] for m in shown]
        fig.add_bar(x=[NAME[m] for m in shown], y=values, marker_color=[COLOR[m] for m in shown],
                    text=[f"{v:.1f}" for v in values], textposition="outside", showlegend=False,
                    hovertemplate="%{x}: %{y:.2f} %<extra></extra>", row=1, col=t + 1)
    show(fig, 300)

    st.markdown("Where do the errors come from? Each dot is one test sea state.")
    fig = panels([f"{t} error [%]" for t in TARGETS])
    for m in (HYBRID_BASE, "pls", "hybrid"):
        for t, target in enumerate(TARGETS):
            fig.add_scatter(x=test.Tp, y=100 * (test[f"{target}_{m}"] / test[target] - 1),
                            mode="markers", name=NAME[m], legendgroup=m, showlegend=t == 0,
                            text=test.run_id, marker=dict(color=COLOR[m], size=8),
                            hovertemplate="%{text}: %{y:+.1f} %<extra></extra>", row=1, col=t + 1)
    fig.add_hline(y=0, line=dict(color=SOFT, width=1), row="all", col="all")
    fig.add_vline(x=row.Tp, line=dict(color=INK, dash="dot", width=1), row="all", col="all",
                  annotation_text=run_id)
    fig.update_xaxes(title="True Tp [s]")
    show(fig)

    truth = test.set_index("run_id").loc[run_id]
    st.markdown(f"**Your sea state, {run_id}:** true Hs {truth.Hs:.2f} m, Tp {truth.Tp:.2f} s")
    table = {NAME[m]: {f"{t} [{UNIT[t]}]": truth[f"{t}_{m}"] for t in TARGETS}
                      | {f"{t} error [%]": 100 * (truth[f"{t}_{m}"] / truth[t] - 1) for t in TARGETS}
             for m in shown}
    st.dataframe(pd.DataFrame.from_dict(table, orient="index").round(2))

    hs, tp = errors["hybrid"]
    end(f"The hybrid wins: Hs {hs:.1f} %, Tp {tp:.1f} %. The physics misses most for short waves "
        "(small Tp), and that is what the hybrid corrects. The test errors are close to the "
        "cross-validation ones, so nothing was overfitted.",
        "RMS relative error [%]: *calibration* is the cross-validation error of step 5 (for the physics, "
        "which is not fitted, its error on the calibration runs), *test* is this step.\n\n"
        "| | Hs calibration | Hs test | Tp calibration | Tp test |\n|---|---|---|---|---|\n"
        + "".join(f"| {NAME[m]} | {cal_err[m][0]:.2f} | {errors[m][0]:.2f} | {cal_err[m][1]:.2f} "
                  f"| {errors[m][1]:.2f} |\n" for m in shown)
        + "\nScript: `evaluate.py` → `data/test.csv`")


def robustness():
    st.title("7. Robustness: a different ship or sensor")
    st.markdown(
        "So far the true ship and the sensor were the same in training and in the test. At sea "
        "they are not: a ship's response changes with its loading, and sensors differ. Here the "
        f"frozen models of step 6 meet the same {len(test)} test sea states, but with:\n"
        "- **RAO gain**: the true ship moves g times as much\n"
        "- **RAO shift**: the true ship's response moved in frequency, as a change in loading does\n"
        "- **sensor noise**: from none to several times the assumed level")
    kinds = {"gain": "RAO gain g", "shift": "RAO shift s", "noise": "Sensor noise std [m, deg]"}
    fig = panels(list(kinds.values()))
    for c, kind in enumerate(kinds, 1):
        d = rob[rob.kind == kind].sort_values("level")
        for m in (HYBRID_BASE, "pls", "hybrid"):
            e = d[d.method == m]
            fig.add_scatter(x=e.level, y=e.Hs_rms, mode="lines+markers", name=NAME[m],
                            legendgroup=m, showlegend=c == 1, marker=dict(size=8),
                            line=dict(color=COLOR[m], width=2), row=1, col=c)
        fig.add_vline(x=d.level[d.condition == "control"].iloc[0], row=1, col=c,
                      line=dict(color=SOFT, dash="dot", width=1))
    fig.update_yaxes(title="Hs error [%]", row=1, col=1)
    show(fig)
    end("A gain error goes straight into every method's Hs: from the motions alone, a ship that "
        "moves 10 % more looks just like 10 % higher waves. A shift hurts PLS most, and the hybrid "
        "loses most of its edge over the physics. A noise level other than in training, even less "
        "noise, throws the learned models off (PLS badly), while the physics hardly moves.",
        "The dotted line is the setting of step 6. The gain and shift runs are new OSP runs with "
        "changed copies of `gunnerus.mat`; a new noise level needs no new runs. The Tp errors "
        "are in `data/robust.csv`. A possible fix for the noise problem (not done): train at the "
        "real noise level, or cut the spectra at the noise floor.  \n"
        "Script: `robustness.py` → `data/robust.csv`")


# ---------------------------------------------------------------- app
STEPS = [(design, "1. Design", "choose the sea states to test"),
         (simulate, "2. Simulate", "let the ship ride each one in OSP"),
         (spectra, "3. Spectra", "turn the motions into spectra"),
         (physics, "4. Physics", "undo the ship to get Hs and Tp"),
         (learn, "5. Learn", "let data correct the physics"),
         (exam, "6. Test", "score once on the hidden sea states"),
         (robustness, "7. Robustness", "change the ship or the sensor")]
pages = [st.Page(start, title="Start", icon=":material/home:", default=True),
         *[st.Page(f, title=title) for f, title, _ in STEPS]]
page = st.navigation(pages)

runs, psd, freq, phys, test, cv, rob, components, cal_err = load()
labels = [f"{r.run_id} · Hs {r.Hs:.1f} m · Tp {r.Tp:.1f} s"
          for r in runs[runs.split == "test"].sort_values("Tp").itertuples()]
run_id = st.sidebar.selectbox("Follow one sea state", labels, index=len(labels) // 2,
                              help="One of the test runs, sorted by Tp").split(" ")[0]
i = int(np.flatnonzero(runs.run_id == run_id)[0])
row = runs.iloc[i]
page.run()
