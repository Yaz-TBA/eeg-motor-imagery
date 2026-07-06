"""Decode hands vs. feet from subject 1 with a CSP + LDA baseline.

Pipeline:
  raw -> average ref -> 8-30 Hz band-pass -> epoch -> crop to imagery window
       -> CSP spatial filters -> log-variance features -> LDA -> cross-validate

CSP finds channel-weightings that make one class high-variance and the
other low-variance along a few axes; the log-variance along those axes is
the feature. LDA separates the two feature clouds with a hyperplane.
Cross-validation gives an honest accuracy instead of one lucky split.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0

# --- load + preprocess (rungs 1-2) ---
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in edf_paths])
eegbci.standardize(raw)
raw.set_montage("standard_1005")
raw.set_eeg_reference("average", projection=False)
raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
epochs = mne.Epochs(
    raw, events, dict(hands=2, feet=3),
    tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
)

# Labels (2=hands, 3=feet). Crop to the 1-2 s imagery window for features.
labels = epochs.events[:, -1]
train_data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False)

# --- CSP + LDA, cross-validated (rungs 3-4) ---
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP", csp), ("LDA", LinearDiscriminantAnalysis())])

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, train_data, labels, cv=cv)

chance = max(np.mean(labels == 2), np.mean(labels == 3))
print(f"\nCSP+LDA accuracy: {scores.mean():.1%}  (+/- {scores.std():.1%})")
print(f"Chance (majority class): {chance:.1%}")
print(f"Per-fold: {np.round(scores, 2)}")

# --- see the spatial filters: fit CSP on all trials and plot top patterns ---
csp.fit_transform(train_data, labels)
fig = csp.plot_patterns(epochs.info, components=range(4), ch_type="eeg", show=False)
fig.savefig("csp_patterns.png", dpi=120, bbox_inches="tight")
print("\nSaved CSP spatial patterns to csp_patterns.png")
