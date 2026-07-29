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

import os

# joblib spawns fresh processes that re-import mne at its DEFAULT log level, so
# mne.set_log_level() in this file never reaches them. Setting it in the
# environment does, because children inherit it. Without this, the parallel
# permutation test buries the results under megabytes of rank-estimation output.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
N_PERMUTATIONS = 1000

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

# Stratified k-fold, not ShuffleSplit: it tests every trial exactly once and
# keeps class balance steady across folds. Note what it did NOT do: over 100
# seeds ShuffleSplit averages 93.6% and StratifiedKFold 93.8%, so the switch is
# worth about +0.2 points in expectation and did not lower the headline. What
# the original 10x80/20 split overstated was the PRECISION (+/- 5.6% against a
# Wilson CI 17.2 points wide), plus it left 5 of 45 trials never tested. The
# 94.4 to 91.1 move is seed placement. See evaluate_honestly.py section 6.
#
# WITHDRAWN 2026-07-25, kept visible rather than deleted. This comment used to
# read "the original 10x80/20 split overstated both the accuracy and its
# precision." The precision half stands. The accuracy half is refuted by the
# script it points at: evaluate_honestly.py section 6 sweeps 100 random_state
# values and prints "The two estimators agree in expectation to 0.2 points",
# with stratified k-fold the HIGHER of the two. An estimator that raises the
# expectation cannot have lowered the headline. Seed 42 sits at the 49th
# percentile of the retracted estimator (94.4%) and the 3rd percentile of the
# published one (91.1%), which is where the 3.3 points went. The published
# 91.1% is a conservative draw from an 88.9-97.8% distribution, and the switch
# is still the right call, for coverage and stratification, not for integrity.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# error_score="raise", because sklearn's default is to CATCH a fold that throws,
# score it NaN, and hand back a mean that is quietly wrong. CSP inverts a
# covariance matrix, so a near-singular fold is a real failure mode here, not a
# hypothetical one. A pipeline that breaks should break loudly.
scores = cross_val_score(clf, train_data, labels, cv=cv, error_score="raise")

chance = max(np.mean(labels == 2), np.mean(labels == 3))

# POSITIVE CONTROL, and it runs BEFORE the permutation test on purpose. The
# permutation test is a negative control: it asks whether the score beats what
# SHUFFLED labels produce. This asks the other half -- is the model alive at all?
# A pipeline that silently degrades to predicting one class every time still
# returns a number, and that number is the majority-class rate. Checking it first
# means a dead model fails in a second instead of after 1000 permutations.
#
# Chance is 24/45 = 53.3%, NOT 50%. The classes are imbalanced 21 hands / 24
# feet, so "always guess feet" already scores 53.3%; beating 50% would prove
# nothing at all. This is the same imbalance the whole repo computes chance from.
#
# TOL, and it is NOT decoration. A majority-class DummyClassifier on these folds
# scores exactly 24/45, the same value as chance -- but the two are computed by
# different float paths (a mean of k/9 fold scores vs a single 24/45), and the
# dummy lands 1.1e-16 ABOVE. So a bare `> chance` passes a model that has learned
# nothing, which is precisely the failure this control exists to catch. Verified
# by substituting DummyClassifier for the pipeline. sweep_subjects.py hits the
# same trap and uses the same 1e-9 tolerance.
TOL = 1e-9
assert scores.mean() > chance + TOL, (
    f"Positive control failed: CSP+LDA scored {scores.mean():.1%}, which does not "
    f"beat the majority-class rate of {chance:.1%} "
    f"({int((labels == 2).sum())} hands / {int((labels == 3).sum())} feet). "
    "A constant predictor would do this well, so the model is not decoding "
    "anything and every number below would be meaningless."
)

# Shuffle the labels 1000x and re-run: does the real result stand outside chance?
# No error_score here because permutation_test_score does not take one -- it calls
# estimator.fit directly with nothing catching it, so a failing fold already
# propagates instead of being scored NaN. The knob is only needed on cross_val_score.
observed, null_scores, p_value = permutation_test_score(
    clf, train_data, labels, scoring="accuracy", cv=cv,
    n_permutations=N_PERMUTATIONS, random_state=42, n_jobs=-1,
)

print(f"\nCSP+LDA accuracy: {scores.mean():.1%}  (+/- {scores.std():.1%})")
print(f"Chance (majority class): {chance:.1%}")
print(f"Per-fold: {np.round(scores, 2)}")
# sklearn computes p = (C + 1)/(n + 1) where C counts permutations scoring >=
# observed, so 1/1001 is the FLOOR of a 1000-shuffle test. Printing "0.0010"
# invites reading the test's resolution limit as a measurement.
p_floor = 1.0 / (N_PERMUTATIONS + 1)
p_str = f"<= {p_floor:.3f}" if p_value <= p_floor + 1e-12 else f"=  {p_value:.4f}"
print(f"Permutation test: p {p_str} "
      f"(null {null_scores.mean():.1%} +/- {null_scores.std():.1%}, "
      f"max {null_scores.max():.1%})")

# --- see the spatial filters: fit CSP on all trials and plot top patterns ---
csp.fit_transform(train_data, labels)
fig = csp.plot_patterns(epochs.info, components=range(4), ch_type="eeg", show=False)
fig.savefig("csp_patterns.png", dpi=120, bbox_inches="tight")
print("\nSaved CSP spatial patterns to csp_patterns.png")
