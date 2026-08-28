"""Shared pieces of inferential_stats.py: the estimators, the printing helpers, the
input loaders and every constant. Split out of that file on 2026-08-26; importing this
module loads the three persisted arrays and prints nothing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

# joblib/MNE workers are fresh processes that re-import mne at its default log
# level, so set_log_level() alone never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from scipy.optimize import brentq

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

from common import holm as common_holm, wilson_interval as common_wilson

ROOT = Path(__file__).resolve().parent.parent.parent  # the repo root: this file lives in src/checks/

ALPHA = 0.05
POWER = 0.80
Z_TWO_SIDED = 1.959963985  # the multiplier a NORMAL 95% interval uses
Z_POWER = 0.8416212336     # one-sided z at 80% power

# Subject 1, narrow regime -- identical constants to eegnet_compare.py, so the
# three measured figures below are comparable with experiment A there.
SUBJECT = 1
RUNS = [6, 10, 14]
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BN_EPS = 1e-3
NARROW = dict(l_freq=8.0, h_freq=30.0, crop=(1.0, 2.0))
N_SCALE_SEEDS = 5  # the scale diagnostic is seed-dependent; report its spread

# ---------------------------------------------------------------------------
# Estimators. Each returns the number AND the assumption it rests on.
# ---------------------------------------------------------------------------

def t_interval(x, alpha=ALPHA):
    """Student-t interval on the mean of x. df = n - 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * sem
    return m, m - half, m + half


def normal_interval(x, z=Z_TWO_SIDED):
    """Normal (z) interval on the mean of x. Wider df assumption than t."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - z * sem, m + z * sem


def wilson_interval(n_correct, n_total, z=Z_TWO_SIDED):
    """95% CI for a proportion, at this file's exact-normal multiplier.

    The formula now lives in common.py. It used to be reimplemented here, because
    evaluate_honestly.py defined it at module scope alongside a five-minute analysis and
    importing it would have run that analysis. Every script has a __main__ guard now, so
    importing is free and there is one definition instead of three.

    The z default stays Z_TWO_SIDED rather than common's 1.96, so this file's printed
    intervals are unchanged.
    """
    return common_wilson(n_correct, n_total, z)


def paired_power(delta, sd, n, alpha=ALPHA):
    """Two-sided power of a paired t-test, noncentral t reference."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = delta / sd * np.sqrt(n)
    return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)


def mde_noncentral(sd, n, alpha=ALPHA, power=POWER):
    """Smallest true difference a paired t-test detects with `power`.

    Noncentral-t solve. This is the correct reference distribution; the normal
    approximation below is ~5% smaller and is printed only so that a document
    quoting either can be identified.
    """
    hi = 3.0 * sd
    while paired_power(hi, sd, n, alpha) < power:
        hi *= 1.5
    return brentq(lambda d: paired_power(d, sd, n, alpha) - power, 1e-9, hi,
                  xtol=1e-9)


def mde_normal(sd, n, alpha=ALPHA, power=POWER):
    """The z-formula MDE. Anticonservative: it ignores the estimated variance."""
    return (stats.norm.ppf(1 - alpha / 2) + Z_POWER) * sd / np.sqrt(n)


# Holm-Bonferroni adjusted p-values, order preserved. Defined in common.py.
holm = common_holm

def head(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def sub(title):
    print(f"\n--- {title} ---")


def cannot(what, why, rerun):
    print(f"CANNOT RECOMPUTE: {what}")
    print(f"  reason:   {why}")
    print(f"  needs:    {rerun}")

# ---------------------------------------------------------------------------
# Input loading. Missing inputs are reported, never imputed.
# ---------------------------------------------------------------------------

def load_regime():
    path = ROOT / "results/regime_decomposition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_perfold():
    """Per-subject LOSO arrays for the five rung-9 pipelines.

    Repo-local only, and deliberately so. This used to fall back to an absolute
    path under ~/Documents/Projects/audits/, which is not a git repository and
    resolves to nothing in a clone, so the fallback could only ever fire on one
    machine. The file it named was byte-identical to the committed copy
    (md5 a7bc94bf7e8271e79cec718c0ea7d271, 2870 bytes), so dropping it loses no
    data.
    """
    path = ROOT / "results/riemannian_perfold.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_sweep():
    path = ROOT / "results/sweep_results.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        return list(csv.DictReader(fh))


REGIME = load_regime()
PERFOLD = load_perfold()
SWEEP = load_sweep()
