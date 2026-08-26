"""The registered constants, the loader and the channel-set derivation of
ablate_channels.py. Split out of that file 2026-08-26. Importing this module also
puts common.py's directory on the path, so every other ablation_* module imports
it FIRST."""

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings

import numpy as np
import mne
from mne.datasets import eegbci

# common.py lives one level up, beside the script groups; put its directory on the
# path so every ablation_* module can be imported from anywhere.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import SENSORIMOTOR

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- the registered constants. Module level so load_data() and derive_channel_sets()
# --- can be imported and called without running the analysis. Changing any of these
# --- makes the run something other than the registered one.
SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
SEED = 42
SEEDS = range(10)          # the sweep the hostile pass used, so this replicates it
N_PERMUTATIONS = 1000
TOL = 1e-9

def load_data():
    """Load, average-reference, filter and epoch subject 1. Identical to decode_csp.py,
    so the numbers here are comparable with the headline.

    Returns (cropped_epochs, labels, groups, ch_names, n, n_hands, n_feet, majority).
    `groups` is the run index per epoch, which condition (d) uses as its fold variable.
    """
    edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]

    # Record each run's length BEFORE concatenating, so every epoch can be traced
    # back to the run it came from. concatenate_raws consumes the list in place and
    # the run boundary is not recoverable from the result.
    run_edges = np.cumsum([r.n_times for r in raws])

    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    epochs = mne.Epochs(
        raw, events, dict(hands=2, feet=3),
        tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
    )
    labels = epochs.events[:, -1]
    cropped = epochs.copy().crop(tmin=1.0, tmax=2.0)

    # np.searchsorted maps each epoch's onset sample to the run whose span contains
    # it: 0 for samples before the first edge, 1 before the second, 2 after.
    onsets = cropped.events[:, 0] - raw.first_samp
    groups = np.searchsorted(run_edges, onsets, side="right")

    n = len(labels)
    n_hands, n_feet = int((labels == 2).sum()), int((labels == 3).sum())
    majority = max(n_hands, n_feet) / n
    return cropped, labels, groups, cropped.ch_names, n, n_hands, n_feet, majority


def derive_channel_sets(ch_names):
    """COMPLEMENT, WIDE and NOT_WIDE, derived from the montage rather than typed.

    A hand-typed 47-channel list can silently stop being the complement of SENSORIMOTOR
    the moment either one is edited, and then the arm keeps its name while measuring
    something else. Returns (COMPLEMENT, WIDE, NOT_WIDE).
    """
    complement = [c for c in ch_names if c not in SENSORIMOTOR]
    # The WIDE set adds FC5, FC6, CP5 and CP6 to the strip: the full FC/C/CP block.
    # Derived the same way, from the name, for the same reason.
    wide = [c for c in ch_names
            if c[:2] in ("FC", "CP") or (c[0] == "C" and c[1] != "P")]
    not_wide = [c for c in ch_names if c not in wide]
    return complement, wide, not_wide
