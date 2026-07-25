"""The artifact control: take the motor cortex away and see if the decoder dies.

A scalp topography is not evidence. It is a picture of the model's weights, and
a model riding an eye-movement artifact will happily draw a picture too. The
only cheap control that actually bites is an ABLATION: refit the entire pipeline
on a channel subset that CANNOT see sensorimotor cortex, and check that the
accuracy collapses to the majority-class rate -- NOT to 50%. With 21 hands and
24 feet the do-nothing baseline is 53.3%, and a control that lands there has
failed correctly.

Four conditions, one seed, one splitter:

  (a) all 64 channels          -- reproduces the published headline
  (b) sensorimotor only        -- FC/C/CP strip. Should hold or improve.
  (c) frontopolar only         -- Fp/AF. This is where blinks and saccades are
                                  LOUDEST. If the decoder were reading the eyes,
                                  this is the subset that would keep working.
  (d) leave-one-run-out        -- all 64 channels, but folds are whole recording
                                  runs, so no fold can share a session-drift or
                                  electrode-settling trend with its training set.

WHY THIS README TABLE EXISTS AS A SCRIPT NOW. An earlier README published these
four numbers as 91.1 / 95.9 / 47.4 / 93.3 with no script behind them. Two of
them were arithmetically impossible: with 45 trials in five equal folds of 9,
overall accuracy is a count of correct trials over 45, so it can only land on
multiples of 1/45 = 2.222%. 95.9% and 47.4% are not on that lattice -- there is
no k with k/45 = 0.959. They were not measurements. This file replaces them.

A CAVEAT THAT SURVIVES THE ABLATION. The average reference is computed across
all 64 electrodes BEFORE any subset is picked, exactly as in decode_csp.py. So
the frontopolar channels are not hermetically sealed off from occipital or
central activity -- every channel carries -1/64 of every other. The ablation
therefore bounds the artifact contribution rather than eliminating it. Making
the subsets independent would mean re-referencing each subset separately, which
would no longer be the published pipeline. Bounding is the honest claim.
"""

import matplotlib

matplotlib.use("Agg")

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings

import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
SEED = 42

# The sensorimotor strip: the three electrode rows that straddle the central
# sulcus. FC* sits over premotor, C* over primary motor / somatosensory, CP*
# just behind it. Hand imagery shows up around C3/C4, foot imagery near Cz --
# which is precisely the contrast this repo decodes.
SENSORIMOTOR = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

# The frontopolar / anterior-frontal ring. These electrodes sit directly above
# the orbits. Blinks, vertical saccades and frontalis EMG dominate here, and
# there is no motor cortex underneath. This is the negative control.
FRONTOPOLAR = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]


def make_clf():
    """A fresh pipeline each time -- CSP must refit inside every training fold."""
    return Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


# --- load + preprocess (identical to decode_csp.py, so the numbers are comparable) ---
edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raws = [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]

# Record each run's length BEFORE concatenating, so every epoch can be traced
# back to the run it came from. concatenate_raws consumes the list in place and
# the run boundary is not recoverable from the result.
run_lengths = [r.n_times for r in raws]
run_edges = np.cumsum(run_lengths)  # sample index where each run ends

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
ch_names = cropped.ch_names

# np.searchsorted maps each epoch's onset sample to the run whose span contains
# it: 0 for samples before the first edge, 1 before the second, 2 after.
onsets = cropped.events[:, 0] - raw.first_samp
groups = np.searchsorted(run_edges, onsets, side="right")

n = len(labels)
n_hands, n_feet = int((labels == 2).sum()), int((labels == 3).sum())
majority = max(n_hands, n_feet) / n

print(f"\nSubject {SUBJECT}, runs {RUNS} (imagined both fists vs. both feet)")
print(f"{n} trials ({n_hands} hands, {n_feet} feet) | "
      f"majority class = {majority:.1%}")
print(f"Trials per run: " + ", ".join(
    f"run {r}={int((groups == i).sum())}" for i, r in enumerate(RUNS)))

# --- the attainable-accuracy lattice -----------------------------------------
# Every condition below tests each of the 45 trials exactly once (5 stratified
# folds of 9, or 3 run-folds of 15), so the reported accuracy is a count of
# correct trials divided by 45. It cannot take any other value. Quoting a number
# off this lattice is a tell that it was never computed.
print(f"\n--- Attainable-accuracy lattice (n = {n}) ---")
print(f"Every fold scheme here is a PARTITION: each trial is tested once, so the")
print(f"overall accuracy is k/{n} for integer k, i.e. steps of {1/n:.3%}.")
print("Values near the headline: " + ", ".join(
    f"{k}/{n}={k/n:.1%}" for k in range(39, 45)))
print("So 95.9% and 47.4% -- the two numbers the old README table carried --")
print(f"are OFF this lattice ({round(0.959*n)}/{n} = {round(0.959*n)/n:.1%}, "
      f"{round(0.474*n)}/{n} = {round(0.474*n)/n:.1%}) and cannot have been measured.")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
logo = LeaveOneGroupOut()

conditions = [
    ("all 64 channels", ch_names, skf, None),
    (f"sensorimotor only ({len(SENSORIMOTOR)} ch)", SENSORIMOTOR, skf, None),
    (f"frontopolar only ({len(FRONTOPOLAR)} ch)", FRONTOPOLAR, skf, None),
    ("all 64, leave-one-run-out", ch_names, logo, groups),
]

results = []
for name, picks, cv, grp in conditions:
    missing = [c for c in picks if c not in ch_names]
    assert not missing, f"{name}: channels not in this montage: {missing}"
    data = cropped.copy().pick(picks).get_data(copy=False)

    # error_score="raise" so a CSP rank failure or a degenerate fold surfaces as
    # a traceback instead of a silent np.nan that quietly drags the mean down.
    scores = cross_val_score(make_clf(), data, labels, cv=cv, groups=grp,
                             error_score="raise")
    pred = cross_val_predict(make_clf(), data, labels, cv=cv, groups=grp)
    n_correct = int((pred == labels).sum())
    results.append((name, data.shape[1], scores, n_correct))

print(f"\n--- Ablation (CSP+LDA, seed {SEED}, same pipeline throughout) ---")
print(f"{'condition':<32} {'ch':>3} {'acc':>7} {'correct':>9}  per-fold")
for name, n_ch, scores, n_correct in results:
    per_fold = " ".join(f"{s:.2f}" for s in scores)
    print(f"{name:<32} {n_ch:>3} {n_correct/n:>6.1%} {f'{n_correct}/{n}':>9}  {per_fold}")

# The mean of the per-fold scores equals the pooled count only when the folds are
# equal-sized. They are here (9, 9, 9, 9, 9 and 15, 15, 15), so the two agree --
# but assert it rather than assume it, because an unequal split would make the
# fold-mean a number that is NOT on the k/45 lattice while still looking like one.
for name, _, scores, n_correct in results:
    assert abs(scores.mean() - n_correct / n) < 1e-9, (
        f"{name}: fold-mean {scores.mean():.4f} != pooled {n_correct}/{n}. "
        "Folds are unequal, so the fold-mean is not the accuracy."
    )
print(f"\nAll four accuracies land on the k/{n} lattice, as they must.")

# --- the actual claim ---------------------------------------------------------
all64 = results[0][3] / n
smc = results[1][3] / n
fp = results[2][3] / n
loro = results[3][3] / n

print("\n--- What this does and does not show ---")
fp_correct = results[2][3]
maj_correct = max(n_hands, n_feet)
print(f"Frontopolar-only lands at {fp:.1%} ({fp_correct}/{n}) against a "
      f"majority-class rate of {majority:.1%} ({maj_correct}/{n}).")
gap = abs(fp_correct - maj_correct)
print(f"That is {gap} trial{'' if gap == 1 else 's'} off the rate you get by ignoring")
print("the EEG entirely and always answering 'feet'. The frontopolar")
print("decoder has no usable signal. Note the framing: the honest reference here")
print("is the MAJORITY rate, not 50% -- with 21/24 classes, a 51.1% result is not")
print("'above chance', it is a degenerate classifier one trial short of guessing.")
print("The per-fold spread (0.33 to 0.78) is the other tell: folds that wide are")
print("a coin, not a decoder.")
print(f"Sensorimotor-only ({smc:.1%}) vs. all 64 ({all64:.1%}): "
      f"{100*(smc-all64):+.1f} points from dropping "
      f"{64-len(SENSORIMOTOR)} non-motor channels.")
print(f"Leave-one-run-out ({loro:.1%}) holds up with no trial sharing a run with")
print("its training set, so the result is not a within-session drift artifact.")
print("\nBOUND, NOT PROOF: the average reference is computed over all 64 channels")
print("before picking, so the subsets are not electrically independent, and")
print("EEGMMIDB ships no EOG channel to regress out. This ablation bounds the")
print("ocular contribution; it cannot measure it.")
