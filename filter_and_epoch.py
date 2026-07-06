"""Band-pass filter subject 1's motor-imagery data, then epoch.

Adds two preprocessing steps on top of epoch_trials.py:
  1. Band-pass to 8-30 Hz  -> keep only the mu/beta motor rhythms that
     carry the hands-vs-feet difference. Filter the CONTINUOUS raw first,
     so we avoid edge artifacts that arise from filtering short epochs.
  2. Average reference     -> re-reference each channel to the mean of all
     channels; the spatial baseline CSP expects.
Trial counts should be unchanged (21 hands, 24 feet) -- filtering changes
the signal, not the number of cues.
"""

import mne
from mne.datasets import eegbci

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0  # mu + beta band

edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in edf_paths])

eegbci.standardize(raw)
raw.set_montage("standard_1005")

# --- preprocessing (the new rung) ---
raw.set_eeg_reference("average", projection=False)
raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_id = dict(hands=2, feet=3)

epochs = mne.Epochs(
    raw, events, event_id,
    tmin=TMIN, tmax=TMAX,
    picks="eeg", baseline=None, preload=True,
)

print(epochs)
print("\nTrials per class (should be unchanged: 21 hands, 24 feet):")
for label in event_id:
    print(f"  {label}: {len(epochs[label])}")
print(f"\nBand-pass: {L_FREQ}-{H_FREQ} Hz | reference: average")
