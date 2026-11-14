"""Step 2: run the sea states in data/runs.csv in OSP.

Writes data/sim_output/<output_file>, one CSV per run, as described in README.md.

Every run gets its own Python process, because an error in an FMU ends the whole
process. A run whose output file exists is skipped, so a stopped batch continues
where it stopped.

    python run_osp.py                     # every run that is not done
    python run_osp.py run_0000 run_0001   # only these runs
    python run_osp.py --check             # check every output file

How it differs from ospfmusim.run
---------------------------------
Only SIGNALS are saved, through a time series observer in memory, not every
variable to CSV. That is why the results app cannot read these runs. To look at
one run in full, use python -m ospfmusim.run.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from libcosimpy.CosimEnums import CosimVariableType
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimLogging import CosimLogLevel, log_output_level
from libcosimpy.CosimObserver import CosimObserver

from app.build.config_tools import make_paths_absolute, set_values
from ospfmusim.build import SYSTEM_STRUCTURE, build_fmus

import config as C

SYSTEM_FILE = C.DATA_DIR / "OspSystemStructure_batch.xml"
RUN_TIMEOUT_S = 600     # a run normally takes under a minute; a hung run must not stop the batch

# The only variables saved, per FMU. Angles are in radians here.
SIGNALS = {"vessel": ["down", "roll", "pitch", "lf_yaw", "lf_north", "lf_east"],
           "allocation": ["utilisation"]}


def write_system_structure(dt):
    """A copy of OspSystemStructure.xml with the step dt, using the FMUs in src/ospfmusim/fmus."""
    fmus = os.path.relpath(SYSTEM_STRUCTURE.parent / "fmus", C.DATA_DIR).replace(os.sep, "/")
    text = SYSTEM_STRUCTURE.read_text(encoding="utf-8")
    text = text.replace('source="fmus/', f'source="{fmus}/')
    text = re.sub(r"<BaseStepSize>.*?</BaseStepSize>", f"<BaseStepSize>{dt}</BaseStepSize>", text)
    SYSTEM_FILE.write_text(text, encoding="utf-8")


def run_config(row, mat=None):
    """The text of the simulation config for one row of runs.csv: long-crested waves only.

    mat: another hydrodynamic data file (.mat) for the vessel, instead of the config's.
    """
    base = C.OSP_CONFIG
    text = make_paths_absolute(base.read_text(encoding="utf-8-sig"), base.parent)
    heading_ref = tomllib.loads(text)["controller"]["heading_ref_deg"]
    vessel = {("vessel_fmu", "mat_file"): Path(mat).resolve().as_posix()} if mat else {}
    return set_values(text, vessel | {
        ("waves", "hs"): float(row.Hs),
        ("waves", "tp"): float(row.Tp),
        ("waves", "seed"): int(row.wave_seed),
        # heading_deg: relative to the bow, 180 = head seas.
        # direction_deg: where the waves come FROM, NED. With the bow north, 0 = head seas.
        ("waves", "direction_deg"): (float(row.heading_deg) - 180.0 + heading_ref) % 360.0,
        ("waves.spectrum", "gamma"): float(row.gamma),
        ("waves.spreading", "type"): "none",
        ("waves.discretization", "n_frequencies"): 100,     # all of them fit in N_WAVES
        # Top of the wave grid at pi rad/s for every Tp, where the RAO table ends. The default,
        # 3 * peak frequency, cuts the spectra at a place that gives away Tp.
        ("waves.discretization", "omega_max_factor"): float(row.Tp) / 2.0,
        ("wind", "enabled"): False,
        ("current", "enabled"): False,
    })


def simulate(config_path, steps):
    """Runs the system for steps steps and returns {name: samples} for SIGNALS, and "time" [s]."""
    log_output_level(CosimLogLevel.WARNING)
    execution = CosimExecution.from_osp_config_file(str(SYSTEM_FILE))
    slaves = {slave.name.decode(): slave.index for slave in execution.slave_infos()}
    refs = {}
    for name, index in slaves.items():
        for variable in execution.slave_variables(index):
            refs[name, variable.name.decode()] = variable.reference
            if variable.name == b"config_path":
                execution.string_initial_value(index, variable.reference, str(config_path))
    execution.real_initial_value(slaves["vessel"], refs["vessel", "max_internal_step_s"],
                                 C.MAX_INTERNAL_STEP_S)
    # Without buffer_size the observer only keeps the last 10 000 samples.
    observer = CosimObserver.create_time_series(buffer_size=steps + 1)
    execution.add_observer(observer=observer)
    pairs = [(name, variable) for name, variables in SIGNALS.items() for variable in variables]
    samples = {}
    try:
        for name, variable in pairs:
            observer.start_time_series(slaves[name], refs[name, variable], CosimVariableType.REAL)
        if not execution.step(steps):
            raise RuntimeError("The simulation failed, see the error message above.")
        for name, variable in pairs:
            # Asking for one more than steps gives time 0 ... (steps - 1) * dt, see README.md.
            time, _, values = observer.time_series_real_samples(
                slaves[name], refs[name, variable], from_step=0, sample_count=steps + 1)
            samples[variable] = np.array(values)
        samples["time"] = np.array(time) / 1e9      # libcosim measures time in nanoseconds
    finally:
        # A simulation that is still open when Python exits makes Python hang.
        del execution, observer
    return samples


def run_one(run_id, mat=None, out=C.OUTPUT_DIR):
    """Runs one row of runs.csv and writes its output file to the folder out."""
    runs = pd.read_csv(C.RUNS_FILE)
    row = next(runs[runs.run_id == run_id].itertuples())
    steps = round((row.ramp_s + row.duration_s) / row.dt_s)
    with tempfile.TemporaryDirectory(prefix="osp_run_") as temp:
        config_path = Path(temp) / "config.toml"
        config_path.write_text(run_config(row, mat), encoding="utf-8")
        s = simulate(config_path, steps)
    if len(s["time"]) != steps or not np.allclose(np.diff(s["time"]), row.dt_s):
        raise RuntimeError(f"Expected {steps} samples every {row.dt_s} s")
    frame = pd.DataFrame({"time": s["time"], "heave": s["down"],
                          "roll": np.degrees(s["roll"]), "pitch": np.degrees(s["pitch"]),
                          "lf_yaw_deg": np.degrees(s["lf_yaw"]),
                          "lf_north": s["lf_north"], "lf_east": s["lf_east"],
                          "utilisation": s["utilisation"]})
    if not np.isfinite(frame[["heave", "roll", "pitch"]].to_numpy()).all():
        raise RuntimeError("The motions are not finite")
    # Through another name, so a run that is stopped here never looks finished.
    output = Path(out) / row.output_file
    unfinished = output.with_suffix(".unfinished")
    frame.to_csv(unfinished, index=False, float_format="%.6g")
    os.replace(unfinished, output)


def start_worker(run_id, mat=None, out=C.OUTPUT_DIR):
    """Runs one row in its own process. Returns what went wrong, or "" if nothing did."""
    command = [sys.executable, __file__, "--worker", run_id, "--out", str(out)]
    if mat:
        command += ["--mat", str(mat)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"no result after {RUN_TIMEOUT_S} s"
    if result.returncode == 0:
        return ""
    return f"exit code {result.returncode}: {(result.stderr + result.stdout).strip()[-500:]}"


def run_batch(run_ids, mat=None, out=C.OUTPUT_DIR):
    """Runs the rows run_ids (all if empty) whose output file in out does not exist yet."""
    out = Path(out)
    runs = pd.read_csv(C.RUNS_FILE)
    if run_ids:
        runs = runs[runs.run_id.isin(run_ids)]
    if runs.dt_s.nunique() != 1:
        raise ValueError("Every run must have the same dt_s: they share one system structure")
    todo = [row.run_id for row in runs.itertuples() if not (out / row.output_file).exists()]
    print(f"{len(todo)} runs to do, {len(runs) - len(todo)} already done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    build_fmus()                # once here, so that the workers do not build at the same time
    write_system_structure(float(runs.dt_s.iloc[0]))
    failed = 0
    with ThreadPoolExecutor(C.N_WORKERS) as pool:
        results = pool.map(lambda run_id: start_worker(run_id, mat, out), todo)
        for done, (run_id, error) in enumerate(zip(todo, results), 1):
            failed += bool(error)
            result = f"FAILED {error}" if error else "ok"
            print(f"{done}/{len(todo)} {run_id} {result}", flush=True)
    print(f"Finished, {failed} failed. Runs that failed are tried again next time.")
    return failed


def check():
    """Checks every output file against runs.csv and shows how the DP system did."""
    problems, total_bytes, worst = [], 0, []
    for row in pd.read_csv(C.RUNS_FILE).itertuples():
        path = C.OUTPUT_DIR / row.output_file
        if not path.is_file():
            problems.append(f"{row.run_id}: missing")
            continue
        total_bytes += path.stat().st_size
        data = pd.read_csv(path)
        steps = round((row.ramp_s + row.duration_s) / row.dt_s)
        if (len(data) != steps or data.time.iloc[0] != 0.0
                or not np.allclose(np.diff(data.time), row.dt_s)):
            problems.append(f"{row.run_id}: wrong time vector")
        if data[["heave", "roll", "pitch"]].isna().any(axis=None):
            problems.append(f"{row.run_id}: missing motion values")
        record = data[data.time >= row.ramp_s]
        worst.append((record.utilisation.max(), record.lf_yaw_deg.abs().max(),
                      np.hypot(record.lf_north, record.lf_east).max()))
        if worst[-1][0] > 1.0:
            problems.append(f"{row.run_id}: thrusters saturated, utilisation {worst[-1][0]:.2f}")
    print(f"{len(worst)} output files, {total_bytes / 1e6:.0f} MB")
    if worst:
        utilisation, heading, offset = np.max(worst, axis=0)
        print(f"Worst in the analysed records: utilisation {utilisation:.2f}, "
              f"heading error {heading:.2f} deg, position offset {offset:.2f} m")
    print("\n".join(problems) if problems else "No problems found")
    return len(problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="*", help="only these runs (default: all of them)")
    parser.add_argument("--check", action="store_true", help="check the output files")
    parser.add_argument("--mat", type=Path, help="vessel .mat file (default: the config's)")
    parser.add_argument("--out", type=Path, default=C.OUTPUT_DIR,
                        help="folder for the output files (default: data/sim_output)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_one(args.run_ids[0], args.mat, args.out)
    elif args.check:
        sys.exit(1 if check() else 0)
    else:
        sys.exit(1 if run_batch(args.run_ids, args.mat, args.out) else 0)


if __name__ == "__main__":
    main()
