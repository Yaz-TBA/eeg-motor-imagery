"""Load one subject's motor-imagery EEG and save a raw-signal plot.

The initial rung, testing the beginning of the pipeline: load one subject, draw the
raw signal, look at it. You cannot debug a pipeline you haven't seen any input for !

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

# =============================================================================
# STEP 1: get the data onto disk and into memory
# =============================================================================
# Download (cached after first run) and load the EDF file for this run.
# EDF = European Data Format, the standard container for clinical/research
# biosignals. One file per run, holding all 64 channels plus the cue annotations.
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=[RUN], update_path=True)
raw = mne.io.read_raw_edf(edf_paths[0], preload=True)

# EEGBCI channel names have trailing dots ("Fc5.") so clean them for montages.
eegbci.standardize(raw)
# Attach 3-D scalp positions. Needed for anything spatial, including every topo
# plot in this repo.
raw.set_montage("standard_1005")

# =============================================================================
# STEP 2: look at it before trusting it
# =============================================================================
# Sanity numbers first. 160 Hz sampling, ~125 s per run, 64 channels. If any of
# these come back weird, stop here, since everything downstream inherits it.
print(raw.info)
print(f"\nSampling rate: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} s, Channels: {len(raw.ch_names)}")

# Plot the first 5 seconds of the first 10 channels.
# Raw EEG looks like noise, and that's the correct reaction to have. The signal we
# want is a change in rhythm POWER over a second or two, which the eye is genuinely
# bad at seeing. That's the whole reason the rest of the pipeline exists :)
fig = raw.plot(
    duration=5.0,
    n_channels=10,
    scalings="auto",
    show=False,
    title=f"EEGBCI subject {SUBJECT}, run {RUN}: raw EEG",
)
fig.savefig("figures/raw_eeg.png", dpi=120, bbox_inches="tight")
print("\nSaved plot to figures/raw_eeg.png")
