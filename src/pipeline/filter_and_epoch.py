"""Band-pass filter subject 1's motor-imagery data, then epoch.

8 to 30 Hz keeps the frequencies where the motor imagery we're looking for lay
(mu & beta).

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

# =============================================================================
# The new rung: reference, then filter, both BEFORE epoching
# =============================================================================
# --- preprocessing (the new rung) ---
# Average reference first. Voltage is a difference between two points, so every
# channel needs a common thing to be measured against, and the mean of all 64 is
# the least-bad choice available. Side effect worth remembering: each channel now
# carries -1/64 of every other one, which is why later ablations can only BOUND an
# artifact rather than delete it.
raw.set_eeg_reference("average", projection=False)
# FIR filter ("firwin"), not IIR, since FIR is linear-phase: every frequency gets
# delayed by the same amount, so the timing of an event survives the filter. An IIR
# would smear different frequencies by different amounts, and we're about to cut
# fixed windows around cue times, so timing distortion would land directly in the
# features.
# skip_by_annotation="edge" stops it filtering ACROSS the joins between the three
# concatenated runs, which would otherwise invent signal at each seam.
raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_id = dict(hands=2, feet=3)

epochs = mne.Epochs(
    raw, events, event_id,
    tmin=TMIN, tmax=TMAX,
    picks="eeg", baseline=None, preload=True,
)

# The check that makes this rung worth running: filtering must not change how many
# trials exist. If these counts move, something ate cues and everything downstream
# is measuring a different dataset than it thinks it is.
print(epochs)
print("\nTrials per class (should be unchanged: 21 hands, 24 feet):")
for label in event_id:
    print(f"  {label}: {len(epochs[label])}")
print(f"\nBand-pass: {L_FREQ}-{H_FREQ} Hz | reference: average")
