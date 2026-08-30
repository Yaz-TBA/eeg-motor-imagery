"""Decode hands vs. feet from subject 1 with a CSP + LDA baseline.

START HERE ! If you only read one file in this repo, make it this one.

Pipeline:
  raw -> average ref -> 8-30 Hz band-pass -> epoch -> crop to imagery window
       -> CSP spatial filters -> log-variance features -> LDA -> cross-validate

CSP finds channel-weightings that make one class high-variance and the
other low-variance along a few axes; the log-variance along those axes is
the feature. LDA separates the two feature clouds with a hyperplane.
Cross-validation gives an honest accuracy instead of one lucky split.

What this is: the "hello world" of brain-computer interfaces. One subject, 45
trials, public data, the easiest contrast in the set. Not a novel result and I
don't present it as one. The interesting half of this repo is everything that
attacks this file's number afterwards, which is rungs 5 through 11.

Reading order if you're new: this file top to bottom, then evaluate_honestly.py
(where the headline moved 94.4% -> 91.1%), then ablate_channels.py (the artifact
control). EXPLAINER.md is the slow version of all of it.
"""

# Need this to make sure the plotting backend is applied, or the process runs as
# normal, completes, and just doesn't actually show anything. Confused me the
# first 3 times :(
import matplotlib

matplotlib.use("Agg")

import os

# joblib spawns fresh processes that re-import mne at its DEFAULT log level, so
# mne.set_log_level() in this file never reaches them. Setting it in the
# environment does, because children inherit it. Without this, the parallel
# permutation test buries the results under megabytes of rank-estimation output.
# Unironically lost an hour to this :/
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import numpy as np
import mne
from mne.datasets import eegbci          # the PhysioNet EEGBCI dataset loader
from mne.decoding import CSP             # Common Spatial Patterns, the whole trick
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

# --- the knobs, every one of them a choice I can defend ---
SUBJECT = 1              # Subject 1 is MNE's first & tutorial default. Seven subjects score
                         # higher actually, but sticking with default so no cherry-picking !
RUNS = [6, 10, 14]       # Task 4 = IMAGINED both fists vs. both feet. NOT 5/9/13, which are
                         # the same task actually PERFORMED. Decode those by accident and you
                         # get a great number attached to a false claim.
TMIN, TMAX = -1.0, 4.0   # epoch window around the cue, in seconds. Generous on purpose, we
                         # crop tighter later & you can't un-crop what you never cut.
L_FREQ, H_FREQ = 8.0, 30.0   # mu (8-13) + beta (13-30), THE band for motor imagery. Imagining
                             # a movement suppresses these rhythms over the matching bit of
                             # motor cortex. Everything outside is noise to us here.
N_PERMUTATIONS = 1000    # how many label shuffles build the null. 1000 puts the p-value floor
                         # at 1/1001, which matters at the print near the bottom.

# =============================================================================
# STEP 1: load and preprocess (rungs 1-2)
# From "EDF files on disk" to "clean, filtered, referenced signal".
# =============================================================================
# --- load + preprocess (rungs 1-2) ---
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
# Three recordings glued end to end. Careful: concatenate_raws EATS the list in
# place, so if you ever need the run boundaries, record them BEFORE this line.
# ablate_channels.py and permutation_design.py both do exactly that.
raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in edf_paths])
# PhysioNet ships names like "Fc5." with dots & odd casing. This renames them to
# standard 10-05 spellings so set_montage below can actually find them.
eegbci.standardize(raw)
# Attaches real 3-D scalp coordinates to each name. Without it MNE knows it has a
# channel called "C3" but has no idea WHERE C3 is on a head, so no topo plot at the
# bottom of this file and no spatial reasoning at all.
raw.set_montage("standard_1005")
# Average reference: subtract the mean of all 64 electrodes from each one.
# Remember from physics that voltage (EEG voltage here) is always the difference
# between two points, so voltage @ a location is meaningless until you say what
# it's relative to. Averaging every electrode is the least-bad answer available.
#
# The catch, and it matters later: every channel now carries -1/64 of every other
# channel. So when ablate_channels.py deletes the sensorimotor strip, the 47
# survivors still hold a whisper of it. Why it says "bounds" the artifact instead
# of "removes" it :P
raw.set_eeg_reference("average", projection=False)
# Band-pass the CONTINUOUS signal, not the epochs. Subtlest line in the file.
# Filtering short epochs makes edge artifacts at both ends of every trial, and
# those artifacts are exactly what a variance-based method like CSP would happily
# learn to classify. Filter long, cut short.
raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

# The EDF annotations mark when each cue fired. T1/T2 are the two imagery cues, T0
# is rest which we don't use. This turns the annotation stream into an events array
# of [sample_index, 0, event_code] rows.
#
# CAREFUL: T1 and T2 do NOT mean the same thing across the dataset. In runs 6/10/14
# (ours) T1 = both fists, T2 = both feet. In runs 4/8/12 they mean LEFT fist and
# RIGHT fist. Copy this block to another run set without renaming and every label
# downstream cheerfully lies to you.
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
# Slice the continuous signal into one array per trial.
# baseline=None on purpose: baseline correction subtracts a pre-cue mean, which is
# a DC shift, and CSP reasons about VARIANCE which is nearly blind to DC shifts.
# Correcting would cost compute and buy nothing.
epochs = mne.Epochs(
    raw, events, dict(hands=2, feet=3),
    tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
)

# Labels (2=hands, 3=feet). Crop to the 1-2 s imagery window for features.
# The first second after the cue is contaminated by the visual response to the cue
# itself, and by 2 s the imagery has usually faded. This window is where the signal
# actually lives.
# Got the cropping window from MNE's tutorial; neighboring windows swing by tens of
# points, so it's only defensible instead of properly tuned :3
labels = epochs.events[:, -1]
train_data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False)

# =============================================================================
# STEP 2: the model (rungs 3-4)
# 64 channels x 161 timepoints per trial -> one hands/feet decision.
# =============================================================================
# --- CSP + LDA, cross-validated (rungs 3-4) ---
# CSP = Common Spatial Patterns. It finds spatial filters (weighted combinations of
# the 64 electrodes) that MAXIMIZE variance for one class while MINIMIZING it for
# the other. Formally a generalized eigenvalue problem on the two classes'
# covariance matrices, C1 w = lambda C2 w, and the useful filters sit at the EXTREME
# ends of the eigenvalue spectrum since those are the maximally discriminative
# directions. n_components=4 keeps two from each end.
# log=True then takes the log of each component's variance over the trial: band
# power IS variance, and the log makes the distribution roughly normal, which is
# what LDA wants. So 64 x 161 numbers become 4 numbers per trial.
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
# LDA draws one straight boundary through those 4-D points. Deliberately boring:
# with 45 trials anything fancier overfits before it does anything useful.
#
# Pipeline wrapper is this way so the CSP can refit within each training fold.
# Stops it from cheating on the answers or smth LOL. Fit CSP once outside the loop
# and the spatial filters get to peek at the test trials, which inflates every
# number in this repo. test_pipeline.py asserts this placement on make_clf(), which this script
# never calls, so the construction on the next line is unguarded: move the CSP fit outside the
# loop here and the suite still reports 19/19. I checked, and that's a gap in the tests.
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

# Chance here is the MAJORITY-CLASS rate.
chance = max(np.mean(labels == 2), np.mean(labels == 3))

# =============================================================================
# STEP 3: is the number real? (the two controls)
# Positive control asks "is the model alive". Negative control asks "could chance
# have done this". You need both, in that order.
# =============================================================================
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
#
# Only the float gods know why a 1.1e-16 difference permits a dead model through :/
TOL = 1e-9
assert scores.mean() > chance + TOL, (
    f"Positive control failed: CSP+LDA scored {scores.mean():.1%}, which does not "
    f"beat the majority-class rate of {chance:.1%} "
    f"({int((labels == 2).sum())} hands / {int((labels == 3).sum())} feet). "
    "A constant predictor would do this well, so the model is not decoding "
    "anything and every number below would be meaningless."
)

# NEGATIVE CONTROL. Shuffle the labels so any real relationship is destroyed, re-run
# the ENTIRE pipeline (CSP refit and all), and see what accuracy pure chance gets.
# Do that 1000 times and you have a null distribution to compare the real score to.
# Shuffle the labels 1000x and re-run: does the real result stand outside chance?
# No error_score here because permutation_test_score does not take one -- it calls
# estimator.fit directly with nothing catching it, so a failing fold already
# propagates instead of being scored NaN. The knob is only needed on cross_val_score.
observed, null_scores, p_value = permutation_test_score(
    clf, train_data, labels, scoring="accuracy", cv=cv,
    n_permutations=N_PERMUTATIONS, random_state=42, n_jobs=-1,
)

# =============================================================================
# STEP 4: report it, every number carrying its own scope
# =============================================================================
print(f"\nCSP+LDA accuracy: {scores.mean():.1%}  (+/- {scores.std():.1%})")
print(f"Chance (majority class): {chance:.1%}")
print(f"Per-fold: {np.round(scores, 2)}")
# sklearn computes p = (C + 1)/(n + 1) where C counts permutations scoring >=
# observed, so 1/1001 is the FLOOR of a 1000-shuffle test. Printing "0.0010"
# invites reading the test's resolution limit as a measurement.
# The limits here: we can't say "p = 0.001"; only "p is at MOST 0.001, and with
# 1000 shuffles that's as fine as we can make it" !
p_floor = 1.0 / (N_PERMUTATIONS + 1)
p_str = f"<= {p_floor:.3f}" if p_value <= p_floor + 1e-12 else f"=  {p_value:.4f}"
print(f"Permutation test: p {p_str} "
      f"(null {null_scores.mean():.1%} +/- {null_scores.std():.1%}, "
      f"max {null_scores.max():.1%})")

# =============================================================================
# STEP 5: look at what CSP actually learned
# =============================================================================
# --- see the spatial filters: fit CSP on all trials and plot top patterns ---
# This fit is for LOOKING ONLY, on all 45 trials, and it deliberately feeds no
# number above. Refitting on everything would leak, but nothing downstream reads it,
# so the picture is honest and the accuracy stays clean.
#
# A scalp map is a picture of the MODEL'S WEIGHTS, not a picture of the brain. An
# earlier version of this repo pointed at these maps and claimed they proved the
# signal was motor. I retracted that.. the showcased component actually peaks
# parieto-occipital. The real artifact control is the channel ablation in
# ablate_channels.py. Cool looking pics are unfortunately not evidence :P
csp.fit_transform(train_data, labels)
fig = csp.plot_patterns(epochs.info, components=range(4), ch_type="eeg", show=False)
fig.savefig("figures/csp_patterns.png", dpi=120, bbox_inches="tight")
print("\nSaved CSP spatial patterns to figures/csp_patterns.png")
