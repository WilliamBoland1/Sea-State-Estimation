"""Shared settings for the sea state estimation project (TMR06).

Every script imports from here, so a change in one place applies everywhere.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
SIM_REPO = ROOT.parent / "ospdpsimpy"   # the simulator repo, checked out next to this one
DATA_DIR = ROOT / "data"
RUNS_FILE = DATA_DIR / "runs.csv"          # sea states the simulator should run
OUTPUT_DIR = DATA_DIR / "sim_output"       # one time-series file per run goes here
FIG_DIR = ROOT / "figures"

# ---------------------------------------------------------------- physics
G = 9.81  # m/s^2

# ---------------------------------------------------------------- sea-state design
N_CAL = 200            # runs used for fitting and cross-validation
N_TEST = 50            # hidden runs, only used once at the very end
HS_RANGE = (0.5, 4.0)  # significant wave height [m]
TP_RANGE = (4.0, 14.0) # peak period [s]
MAX_STEEPNESS = 1 / 15 # 2*pi*Hs / (g*Tp^2); steeper seas are unrealistic
GAMMA = 3.3            # JONSWAP peak enhancement (fixed for now)
HEADINGS_DEG = [180.0] # 180 = head seas. Add more later, e.g. [180, 135, 90]
DESIGN_SEED_CAL = 1
DESIGN_SEED_TEST = 3   # 2 was used to choose the run settings, so those runs are not hidden

# ---------------------------------------------------------------- simulator run settings
DURATION_S = 1200.0    # length of the record that is analysed [s] (20 min)
RAMP_S = 50.0          # start-up transient, simulated but discarded [s]. 200 for oblique seas
DT_S = 0.5             # co-simulation step and output sample time [s] (2 Hz)
MAX_INTERNAL_STEP_S = 0.25                 # longest step of the vessel's low-frequency model [s]
OSP_CONFIG = SIM_REPO / "configs" / "gunnerus.toml"   # all other constants of the simulation
N_WORKERS = 6          # runs at the same time

# ---------------------------------------------------------------- the two vessel files
# Both are Gunnerus, from separate hydrodynamic calculations (README: "The two vessel files").
# They differ on purpose, so the physics method does not get the hull model for free.
TRUE_MAT = SIM_REPO / "data_examples" / "gunnerus.mat"   # makes the runs: same file as mat_file in OSP_CONFIG
INVERSION_MAT = SIM_REPO / "data_examples" / "GunnerusL35_20201112.mat"   # step 4 divides by its RAOs

# ---------------------------------------------------------------- motions
# Column names in each output file. Roll is left out: in head seas it has no wave signal.
# Add it back together with oblique headings.
MOTIONS = ["heave", "pitch"]

# ---------------------------------------------------------------- spectra
SPECTRA_FILE = DATA_DIR / "spectra.npz"    # X, Y and the spectra, from build_spectra.py
WELCH_SEGMENT_S = 128.0    # segment length [s]; the frequency resolution is 1 / this [Hz]
WELCH_OVERLAP = 0.75       # share of each segment shared with the next (Hann window)
FREQ_RANGE_HZ = (0.04, 0.4)    # kept band [Hz]; outside it the spectra are mostly sensor noise
# Sensor noise added to the motions before the spectra [m, deg, deg]. ASSUMED values:
# replace them with the data sheet of the sensor.
SENSOR_NOISE_STD = {"heave": 0.02, "roll": 0.02, "pitch": 0.02}
NOISE_SEED = 1

# ---------------------------------------------------------------- physics inversion
PHYSICS_FILE = DATA_DIR / "physics.csv"    # Hs and Tp per run, from invert_spectra.py

# ---------------------------------------------------------------- cross-validation
CV_FILE = DATA_DIR / "cv.csv"      # CV error per model and number of components, from cross_validate.py
N_FOLDS = 5
CV_SEEDS = [0, 1, 2]               # every seed splits the cal runs into folds in a new way
MAX_COMPONENTS = 40                # PCR still improves slowly up to 25

# ---------------------------------------------------------------- evaluation
TEST_FILE = DATA_DIR / "test.csv"  # every estimate on the test runs, from evaluate.py

# ---------------------------------------------------------------- robustness
# The test runs again, with a "true vessel" the models have not seen or a noisier sensor.
ROBUST_DIR = DATA_DIR / "robust"   # the changed .mat files and their runs, from robustness.py
ROBUST_FILE = DATA_DIR / "robust.csv"
RAO_GAINS = [0.90, 0.95, 1.05, 1.10]    # heave and pitch |RAO| times this
RAO_SHIFTS = [0.95, 0.97, 1.03, 1.05]   # heave and pitch RAO moved to this times the frequency
NOISE_LEVELS = [0.0, 0.05, 0.10]        # sensor noise std [m, deg]; 0.02 is the control
