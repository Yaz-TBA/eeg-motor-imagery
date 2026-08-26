"""Epoch subject 1's motor-imagery runs into hands-vs-feet trials.

Epoching is just "cut the tape into clips, one per cue". Weird that the cue markers
live in the annotations, not the signal, so you have to translate before you can cut.

Runs 6/10/14 = Task 4 (imagine both fists vs. both feet). The EDF
annotations mark each cue: T1 = imagine fists (hands), T2 = imagine feet.
We concatenate the three runs, cut a fixed window around each cue, and
count how many labeled trials of each class we get.
"""

import mne
from mne.datasets import eegbci

SUBJECT = 1
RUNS = [6, 10, 14]  # imagined fists vs. feet
TMIN, TMAX = -1.0, 4.0  # seconds relative to each cue

# =============================================================================
# STEP 1: three runs into one continuous recording
# =============================================================================
# Load all three runs and stitch them into one continuous Raw.
# Worth knowing: concatenate_raws consumes `raws` in place, so the run boundaries
# are gone after this line. Nothing here needs them, but ablate_channels.py and
# permutation_design.py both do, and both record the lengths BEFORE stitching.
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raws = [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]
raw = mne.concatenate_raws(raws)

eegbci.standardize(raw)  # clean channel names ("Fc5." -> "FC5")
raw.set_montage("standard_1005")

# =============================================================================
# STEP 2: annotations -> events -> trials
# =============================================================================
# Turn annotations into an events array. Map the annotation labels to
# integer class ids: T1 -> 2 (hands), T2 -> 3 (feet). (T0/rest is dropped.)
#
# The events array is just [sample_index, 0, class_id] rows, one per cue. That
# middle column is a legacy slot almost nothing uses, ignore it.
#
# T1/T2 are POSITIONAL labels, not meanings ! In these runs T1 = both fists and
# T2 = both feet. In runs 4/8/12 the exact same T1/T2 mean LEFT fist and RIGHT
# fist. harder_contrast.py is built on those and calls this out as its trap #2.
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_id = dict(hands=2, feet=3)

# Slice the continuous signal into trials, one per cue.
# tmin=-1.0 grabs a second BEFORE each cue. We don't classify on it here, but
# having pre-cue data on hand is what lets regime_decomposition.py later run a
# pre-cue control window and check the decoder isn't reading the cue itself.
# baseline=None since CSP works on variance, which barely notices the DC shift
# that baseline correction removes. Nothing gained, compute spent.
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin=TMIN,
    tmax=TMAX,
    picks="eeg",
    baseline=None,
    preload=True,
)

# 21 hands / 24 feet. Note that imbalance, it is where the 53.3% chance rate comes
# from and it is NOT 50%. Every accuracy in this repo is measured against 53.3%.
print(epochs)
print("\nTrials per class:")
for label in event_id:
    print(f"  {label}: {len(epochs[label])}")
print(f"\nEach trial: {len(epochs.ch_names)} channels x "
      f"{len(epochs.times)} samples ({TMIN}s to {TMAX}s @ {raw.info['sfreq']:.0f} Hz)")
