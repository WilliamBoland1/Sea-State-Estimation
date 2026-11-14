# Sea state estimation from vessel motions (TMR06)

## Steps
1. `design_sea_states.py`: writes `data/runs.csv`, the list of sea states to simulate.
2. `run_osp.py`: runs every row in OSP and writes `data/sim_output/`. A batch that stops can be started again. `python run_osp.py --check` checks the result.
3. `build_spectra.py`: loads the runs, adds sensor noise, makes Welch spectra and writes `data/spectra.npz` (X and Y) and `figures/spectra.png`. What the file holds is at the top of the script. `python build_spectra.py --check` checks the result.
4. `invert_spectra.py`: estimates Hs and Tp from the spectra with the RAOs of `GunnerusL35_20201112.mat` alone (not the `gunnerus.mat` the runs were made with), by RAO division and by a JONSWAP fit, and writes `data/physics.csv`. Only cal runs are scored. `python invert_spectra.py --check` checks the units, the smoothing and the fit.
5. `cross_validate.py`: cross-validates PLS, PCR and a hybrid (the physics estimate of step 4 corrected by PLS) on the cal runs, and writes the error for every number of components to `data/cv.csv`. Step 6 uses its `load()` and `fit_predict()`. `python cross_validate.py --check` checks PLS and PCR against least squares.
6. `evaluate.py`: fits PLS, PCR and the hybrid on all cal runs with the number of components chosen in step 5, scores them and the physics estimates of step 4 once on the test runs, and writes `data/test.csv`, `figures/cv.png` and `figures/test.png`. Do not tune anything on what it shows. `python evaluate.py --check` checks the file, and that no test run leaks into the models.

7. `robustness.py`: scores the frozen models of step 6 on the test runs again, with a true vessel or a sensor they were not fitted to. It writes copies of `gunnerus.mat` with heave and pitch RAO scaled (`RAO_GAINS`) or moved in frequency (`RAO_SHIFTS`) to `data/robust/`, and reruns the test runs in OSP with each one (`run_osp.py --mat ... --out ...`, about 5 min per file). The sensor noise is also changed (`NOISE_LEVELS`), which needs no new runs. It writes `data/robust.csv` and `figures/robust.png`. `python robustness.py --check` checks the .mat files, that the control gives `data/test.csv` again, and that a gain g gives g times the Hs.

All settings are in `config.py`.

## What the simulator (OSP) reads: `data/runs.csv`
One row per run:

| column | meaning |
|---|---|
| run_id | e.g. `run_0007` |
| split | `cal` (fitting) or `test` (hidden until the end) |
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
