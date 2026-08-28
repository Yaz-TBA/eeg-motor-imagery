"""The registered constants and shared helpers of permutation_design.py. Split out
2026-08-26. This module is imported FIRST by every other permdesign_* module: it
sets the BLAS and MNE environment before numpy loads, and it puts common.py's
directory on the path."""

import os
import sys
import time

# joblib spawns fresh processes that re-import mne at its DEFAULT log level, so
# mne.set_log_level() in this file never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
# One BLAS thread per worker. 16 loky workers each spawning their own BLAS pool
# oversubscribes the machine and makes the run SLOWER, not faster. CSP inverts
# 64x64 covariances, so the per-call BLAS work is far too small to parallelise.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import mne

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

from common import load_epochs

# --- registered constants. Changing any of these makes the run a smoke test. ---
SEED = 42
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
CROP = (1.0, 2.0)
L_FREQ, H_FREQ = 8.0, 30.0
N_SPLITS = 5

REG_N_ARM_A = 10_000          # per cell, arm A
REG_N_ARM_B = 2_000           # per cell, arm B
REG_N_REPRO = 300             # pilot reproduction, A269's own draw count
REG_N_GATE2 = 1_000           # sklearn-agreement gate

N_ARM_A = int(os.environ.get("PERMDESIGN_N_A", REG_N_ARM_A))
N_ARM_B = int(os.environ.get("PERMDESIGN_N_B", REG_N_ARM_B))
N_REPRO = int(os.environ.get("PERMDESIGN_N_REPRO", REG_N_REPRO))
N_GATE2 = int(os.environ.get("PERMDESIGN_N_GATE2", REG_N_GATE2))
IS_REGISTERED_RUN = (N_ARM_A == REG_N_ARM_A and N_ARM_B == REG_N_ARM_B
                     and N_REPRO == REG_N_REPRO and N_GATE2 == REG_N_GATE2)

N_JOBS = int(os.environ.get("PERMDESIGN_N_JOBS", "-1"))
CHUNK_A = 50
CHUNK_B = 10

# Subjects, by a rule fixed before any null ran. Subject 1 is the headline. The
# median of subjects 1..20 by published within-subject accuracy is 0.6333, and
# subjects 17 (0.6222) and 19 (0.6444) are EXACTLY equidistant from it, so both
# are included rather than inventing a tie-break.
SUBJECTS_A = [1, 17, 19]
SUBJECTS_B = list(range(1, 21))
WITHIN_CSV = "results/sweep_results.csv"

# A269, the pilot, quoted here as the PRIOR. This run confirms or contradicts it;
# it does not discover it independently. Order: null mean %, null sd %, null max %.
A269_PILOT = {
    "(i)   iid shuffle, re-stratified folds": (50.7, 8.9, 86.7),
    "(ii)  iid shuffle, FIXED folds        ": (47.7, 8.4, 71.1),
    "(iii) within-run cyclic shift         ": (49.0, 8.6, 73.3),
    "(iv)  within-run label permutation    ": (50.2, 8.2, 68.9),
}
A269_PILOT_P = 0.0033

# Registered materiality thresholds. Fixed before the run.
ONE_TRIAL = 1.0 / 45.0        # 2.2222 points. A sub-trial difference in a
                              # 45-trial null is not a difference anyone can act on.
MC_SIGMA = 3.0                # multiples of the Monte Carlo standard error
P_THRESHOLD = 0.05
CENTRE_LO, CENTRE_HI = 0.45, 0.55   # registered assert 9 / falsification 8
CENTRING = {}                 # every cell's null mean, recorded for section 6

# POST-REGISTRATION, added 2026-07-26 after an adversarial pass showed that the
# fixed-at-P0 cells are not exact tests. None of these edit the pre-registration.
#   C5 / C6: the EXACT version of the fixed-partition idea. The partition is built
#            WITHOUT the labels, so it is ancillary and freezing it preserves
#            exchangeability. This is what C2 / C4 were trying to be.
#   the dummy type-I control: zero information, H0 exactly true by construction.
N_EXACT = int(os.environ.get("PERMDESIGN_N_EXACT", "2000"))
# 200 x 199 rather than 2000 x 999: the effect being measured is the difference
# between a rejection rate near zero and one well above 0.5, so the MC se on a
# 0.05 rate at 200 draws (0.015) is an order of magnitude smaller than the effect.
# Spending 25x the compute to shrink an already-decisive interval is not a
# measurement, it is a way to make the script too slow to run.
N_TYPE1_OUTER = int(os.environ.get("PERMDESIGN_N_T1_OUTER", "200"))
N_TYPE1_INNER = int(os.environ.get("PERMDESIGN_N_T1_INNER", "199"))

SHUFFLE_MAX = 0.60            # the underived guard in cross_subject.py:146,
                              # quoted here as the thing being replaced.

TOL = 1e-9

def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    sys.stdout.flush()


def sub(title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    sys.stdout.flush()


def note(msg):
    """Progress to stderr, so stdout stays a clean provenance record."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# make_clf comes from common.py. CSP sits INSIDE the Pipeline, so it refits on the
# training fold only, inside every fold, in every replicate. Any other placement would
# leak the test fold into filter estimation and invalidate everything, and
# test_pipeline.py asserts the placement.

def fmt(acc, n):
    """Every accuracy prints with its k/n count. On n tested-once trials the
accuracy is a multiple of 1/n and no other value is attainable."""
    return f"{acc:.1%} ({int(round(acc * n))}/{n})"


# assert_lattice comes from common.py, at the same TOL = 1e-9. Equal folds mean the
# fold-mean IS the pooled count over n. Off-lattice means unequal folds or a scorer
# that is not accuracy, and then the fold-mean is not the accuracy at all. This is the
# check that caught two arithmetically impossible numbers (95.9%, 47.4%) in an earlier
# README, and test_pipeline.py now guards both of them by name.


def p_value(null, observed, n_draws):
    """p = (C + 1) / (N + 1) with >=, matching sklearn exactly. C is returned so
that 'both at the floor' stays distinguishable from 'both actually equal'."""
    c = int((np.asarray(null) >= observed - TOL).sum())
    return (c + 1) / (n_draws + 1), c


def p_str(p, c, n_draws):
    floor = 1.0 / (n_draws + 1)
    if c == 0:
        return f"<= {floor:.5g} (BOUND, the floor of {n_draws} draws, C = 0)"
    return f"=  {p:.5g} (C = {c} of {n_draws})"


def quant(x, q, n):
    """Order-statistic quantiles. method='inverted_cdf' returns an OBSERVED value,
so the percentile stays on the k/n lattice. Linear interpolation would print
percentiles that no replicate could have scored."""
    v = float(np.quantile(x, q, method="inverted_cdf"))
    return v, f"{v:.1%} ({int(round(v * n))}/{n})"


def describe_null(name, null, observed, n_trials, n_draws, extra_q=()):
    p, c = p_value(null, observed, n_draws)
    print(f"{name}")
    print(f"    null mean {null.mean():.2%}  sd {null.std(ddof=0):.2%}   "
      f"min {fmt(null.min(), n_trials)}   max {fmt(null.max(), n_trials)}")
    qs = sorted(set([0.50, 0.90, 0.95, 0.99, 0.999] + list(extra_q)))
    parts = []
    for q in qs:
        _, s = quant(null, q, n_trials)
        parts.append(f"p{q * 100:g}={s}")
    print("    " + "  ".join(parts))
    print(f"    observed {fmt(observed, n_trials)}   "
      f"standardised distance {(observed - null.mean()) / null.std(ddof=0):+.2f} sd "
      f"(descriptive only, the null is not normal)")
    print(f"    p {p_str(p, c, n_draws)}")
    sys.stdout.flush()
    return p, c


def wilson(k, n, z=1.959963985):
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d

# --------------------------------------------------------------------------- #
# loading. Identical to decode_csp.py / sweep_subjects.py / cross_subject.py.
# --------------------------------------------------------------------------- #
def load_subject(subject):
    """Runs 6/10/14, average reference over all 64 channels, 8-30 Hz FIR, epochs
-1.0 to 4.0 s, cropped to 1.0-2.0 s. Also returns the RUN INDEX per epoch. That
blocking variable has existed in this repo since the ablation rung and had never been
used in a null; it is the whole of the arm A block correction.

The body now lives in common.load_epochs. Verified bit-for-bit identical to the
version this file used to carry, on subjects 1 and 17: X, y and the run indices all
compare equal with np.array_equal."""
    return load_epochs(subject, runs=RUNS, l_freq=L_FREQ, h_freq=H_FREQ,
                       tmin=TMIN, tmax=TMAX, crop=CROP, return_runs=True)
