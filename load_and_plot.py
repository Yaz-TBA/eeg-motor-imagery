"""Load one subject's motor-imagery EEG and save a raw-signal plot.

EEGBCI dataset: 109 subjects, 64-channel EEG. Runs 6/10/14 are Task 4 =
imagine both fists vs. both feet (our hands-vs-feet motor imagery).
Today: just look at the raw signal for subject 1, run 6.
"""

import matplotlib

matplotlib.use("Agg")  # save to file instead of opening a window

import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci

SUBJECT = 1
RUN = 6  # imagined both fists vs. both feet

# Download (cached after first run) and load the EDF file for this run.
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=[RUN], update_path=True)
raw = mne.io.read_raw_edf(edf_paths[0], preload=True)

# EEGBCI channel names have trailing dots ("Fc5.") — clean them so montages work.
eegbci.standardize(raw)
raw.set_montage("standard_1005")

print(raw.info)
print(f"\nSampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} s, Channels: {len(raw.ch_names)}")

# Plot the first 5 seconds of the first 10 channels.
fig = raw.plot(
    duration=5.0,
    n_channels=10,
    scalings="auto",
    show=False,
    title=f"EEGBCI subject {SUBJECT}, run {RUN} — raw EEG",
)
fig.savefig("raw_eeg.png", dpi=120, bbox_inches="tight")
print("\nSaved plot to raw_eeg.png")
