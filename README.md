# Sea state estimation from vessel motions (TMR06)

Can we tell how high the waves are (Hs) and how long they are (Tp) from how the ship moves?

## How it works
1. **Design** (`design_sea_states.py`): choose 250 sea states, 200 to fit on (*calibration*) and 50 hidden until step 6 (*test*). → `data/runs.csv`
2. **Simulate** (`run_osp.py`): let the ship ride each one in OSP and log heave and pitch. → `data/sim_output/`
3. **Spectra** (`build_spectra.py`): add sensor noise and turn the motions into spectra. → `data/spectra.npz`
4. **Physics** (`invert_spectra.py`): divide out the RAO of a similar ship (not the one the runs were made with) and fit a JONSWAP spectrum to get Hs and Tp. → `data/physics.csv`
5. **Learn** (`cross_validate.py`): cross-validate PLS, PCR and the hybrid (the physics estimate corrected by PLS) on the calibration runs. → `data/cv.csv`
6. **Test** (`evaluate.py`): score every method once on the test runs. Do not tune anything on what it shows. → `data/test.csv`
7. **Robustness** (`robustness.py`): score the frozen models again with a changed ship RAO (new OSP runs, about 5 min per vessel) or sensor noise. → `data/robust.csv`

Run a step with `python <script>.py`; `python <script>.py --check` checks its result. The docstring
at the top of each script says what it does in detail. All settings are in `config.py`.

**See it:** `.venv\Scripts\streamlit run app.py` opens one page per step in the browser, with plots,
and lets you follow one sea state through all of them.

## Setup
The simulations run in the DP simulator [ospdpsimpy](https://github.com/Olvegb/ospdpsimpy),
which this project installs as a package. Steps 2 and 7 run it, and step 4 reads the RAOs
with it.

1. Check out ospdpsimpy next to this folder (`../ospdpsimpy`, see `SIM_REPO` in `config.py`).
2. Put `gunnerus.mat` and `GunnerusL35_20201112.mat` in `../ospdpsimpy/data_examples/`. They are not in git.
3. From this folder, run `py -3.11 -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`.

`data/sim_output/` and `data/robust/` (the OSP output, about 124 MB) are not in git:
`run_osp.py` and `robustness.py` make them again. The results of steps 3 to 7 are in git.

## What the simulator (OSP) reads: `data/runs.csv`
One row per run:

| column | meaning |
|---|---|
| run_id | e.g. `run_0007` |
| split | `cal` (calibration, for fitting) or `test` (hidden until the end) |
| Hs | significant wave height [m] |
| Tp | peak period [s] |
| gamma | JONSWAP peak enhancement [-] |
| heading_deg | wave direction relative to ship, 180 = head seas |
| wave_seed | random seed for the wave phases (makes runs repeatable) |
| duration_s | analysed record length [s] |
| ramp_s | extra start-up time to simulate first, discarded later [s] |
| dt_s | output sample time [s] |
| output_file | file name to write the results to |

Total simulated time per run is `ramp_s + duration_s`.

## What the simulator writes: `data/sim_output/<output_file>`
One CSV per run, including the ramp-up period:

| column | unit |
|---|---|
| time | s, starting at 0 |
| heave | m, positive down (NED) |
| roll | deg |
| pitch | deg |

Extra columns are fine and will be ignored. If the units or sign
conventions in OSP differ, change them here and in `config.py`.

`run_osp.py` converts to these units and also writes `lf_yaw_deg`, `lf_north`,
`lf_east` and `utilisation`, to check the DP system. The file has
`(ramp_s + duration_s) / dt_s` rows, from time 0 to one step before the end, so
`time >= ramp_s` gives exactly the analysed record.

The wave grid of every run ends at pi rad/s (0.5 Hz), where the RAO table of
`gunnerus.mat` ends: the vessel does not respond to waves above it. At Tp near
4 s this leaves up to 4.5 % of the variance out of the waves, but not out of the
motions, so the `Hs` column stays the right target.
