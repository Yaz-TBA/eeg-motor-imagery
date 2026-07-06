"""Epoch subject 1's motor-imagery runs into hands-vs-feet trials.

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

# Load all three runs and stitch them into one continuous Raw.
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raws = [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]
raw = mne.concatenate_raws(raws)

eegbci.standardize(raw)  # clean channel names ("Fc5." -> "FC5")
raw.set_montage("standard_1005")

# Turn annotations into an events array. Map the annotation labels to
# integer class ids: T1 -> 2 (hands), T2 -> 3 (feet). (T0/rest is dropped.)
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_id = dict(hands=2, feet=3)

# Slice the continuous signal into trials, one per cue.
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

print(epochs)
print("\nTrials per class:")
for label in event_id:
    print(f"  {label}: {len(epochs[label])}")
print(f"\nEach trial: {len(epochs.ch_names)} channels x "
      f"{len(epochs.times)} samples ({TMIN}s to {TMAX}s @ {raw.info['sfreq']:.0f} Hz)")
