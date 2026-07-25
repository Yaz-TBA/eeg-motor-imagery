"""Train on other people, test on someone the model has never seen.

Everything up to this rung is WITHIN-subject: the model trains and tests on the
same person's brain. That is a real result, but it is not the result a working
BCI needs. A deployed system meets a new user whose skull thickness, cortical
folding, and electrode placement are all different, and it has to work anyway.

So this rung pools trials across many subjects and evaluates leave-one-subject-out:
train on N-1 people, test on the held-out one, rotate. Naive CSP is known to
struggle here, because the spatial filters it learns are tuned to the training
population's anatomy and do not transfer cleanly.

EXPECT THE NUMBER TO FALL, AND DO NOT TUNE UNTIL IT STOPS FALLING. The size of
the within-to-cross gap is the deliverable. It is the honest measure of how far
this method is from something you could hand to a stranger, and it is what
motivates the Riemannian rung that follows.

Leakage safety: CSP lives inside the sklearn Pipeline, so it refits on only the
training subjects inside every fold. The held-out subject's data never touches
filter estimation. There is an explicit assertion below that checks this rather
than trusting it.
"""

import matplotlib

matplotlib.use("Agg")

import csv
import os
import warnings

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import matplotlib.pyplot as plt
import numpy as np
import mne
from joblib import Parallel, delayed
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# 20 subjects, stated up front. Full 109-subject LOSO is far more expensive and
# the conclusion does not change; an undocumented subset would be dishonest, a
# documented one is just a budget.
SUBJECTS = list(range(1, 21))
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
WITHIN_CSV = "sweep_results.csv"


def load_subject(subject):
    """Preprocess one subject exactly as every other rung does."""
    mne.set_log_level("ERROR")
    try:
        paths = eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.set_eeg_reference("average", projection=False)
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=TMIN, tmax=TMAX,
                            picks="eeg", baseline=None, preload=True)
        y = epochs.events[:, -1]
        X = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False)
        return subject, X, y
    except Exception as exc:  # noqa: BLE001
        print(f"  S{subject:03d} skipped: {type(exc).__name__}: {exc}")
        return subject, None, None


print(f"Loading {len(SUBJECTS)} subjects...")
loaded = Parallel(n_jobs=-1)(delayed(load_subject)(s) for s in SUBJECTS)
loaded = [(s, X, y) for s, X, y in loaded if X is not None]

n_samples = min(X.shape[-1] for _, X, _ in loaded)
X_all = np.concatenate([X[:, :, :n_samples] for _, X, _ in loaded], axis=0)
y_all = np.concatenate([y for _, _, y in loaded])
groups = np.concatenate([np.full(len(y), s) for s, _, y in loaded])

print(f"Pooled {X_all.shape[0]} trials from {len(loaded)} subjects "
      f"({X_all.shape[1]} channels x {n_samples} samples)")

clf = Pipeline([
    ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
    ("LDA", LinearDiscriminantAnalysis()),
])
logo = LeaveOneGroupOut()

# --- structural checks that can ACTUALLY FAIL --------------------------------
# The obvious assertion here -- "no group appears in both train and test" -- is
# definitionally true for LeaveOneGroupOut and can never fire. Asserting it and
# printing "leakage check passed" is theatre: it reads as evidence in a review
# while proving nothing. These three can genuinely fail.
folds = list(logo.split(X_all, y_all, groups))

# 1. Exactly one held-out subject per fold, and every subject held out once.
held_out = [set(groups[te]) for _, te in folds]
assert all(len(h) == 1 for h in held_out), "a fold held out more than one subject"
assert {next(iter(h)) for h in held_out} == set(groups), \
    "not every subject was held out exactly once"

# 2. The groups array is aligned with the data. A silent misalignment here would
#    corrupt every pairing downstream without raising anything.
assert len(groups) == len(y_all) == X_all.shape[0], "groups/labels/data length mismatch"

# 3. THE REAL LEAKAGE TEST, and the only empirical one: shuffle the labels and
#    the cross-subject score must collapse to chance. If a leak existed, the
#    shuffled score would stay elevated.
rng = np.random.default_rng(0)
shuffled = rng.permutation(y_all)
shuffled_score = cross_val_score(clf, X_all, shuffled, groups=groups,
                                 cv=logo, n_jobs=-1, error_score="raise").mean()
print(f"Structural checks passed ({len(folds)} folds, one subject each).")
print(f"Label-shuffled control: {shuffled_score:.1%} "
      f"(must sit near chance; elevated would mean a leak)")
assert shuffled_score < 0.60, (
    f"Label-shuffled cross-subject score is {shuffled_score:.1%}, too high. "
    "Something is leaking."
)

print("\nRunning leave-one-subject-out...")
# error_score="raise" on both CV calls above and here. The default would score a
# folding failure NaN and average it in, so a leak check that never actually ran
# could still print a comfortable number -- exactly the kind of quiet pass this
# rung is built to refuse.
scores = cross_val_score(clf, X_all, y_all, groups=groups, cv=logo, n_jobs=-1,
                         error_score="raise")
tested = [s for s, _, _ in loaded]

chance_all = max(np.mean(y_all == 2), np.mean(y_all == 3))
print(f"\n{'=' * 58}")
print(f"Cross-subject (LOSO, {len(scores)} subjects)")
print(f"{'=' * 58}")
print(f"mean      {scores.mean():.1%} +/- {scores.std():.1%}")
print(f"median    {np.median(scores):.1%}")
print(f"min / max {scores.min():.1%} / {scores.max():.1%}")
print(f"pooled chance {chance_all:.1%}")
print(f"subjects above pooled chance: {(scores > chance_all).sum()}/{len(scores)}")

# --- positive control, and the one rung where it MUST NOT be an assert ---------
# Every other classical rung asserts that the model beats the majority-class
# rate. Doing that here would be a category error. This rung's stated premise is
# "EXPECT THE NUMBER TO FALL, AND DO NOT TUNE UNTIL IT STOPS FALLING" -- a
# cross-subject score sitting at chance is a legitimate scientific finding about
# how badly naive CSP transfers, not a bug in the code. An assert would convert
# that finding into a crash, and the obvious way to make a crash go away is to
# tune until it passes. That is precisely the behaviour this file forbids.
#
# So it reports and warns. The negative control above (shuffled labels must
# collapse) is the one that gets to abort, because an elevated shuffled score
# has no honest interpretation.
#
# print, not warnings.warn -- filterwarnings("ignore") at the top would eat it.
# TOL because a score exactly equal to chance can land either side of a bare
# comparison on float noise alone -- see decode_csp.py, where a majority-class
# dummy tests 1.1e-16 above the line.
TOL = 1e-9
print(f"\nPositive control: LOSO mean {scores.mean():.1%} vs pooled majority-class "
      f"rate {chance_all:.1%} ({100 * (scores.mean() - chance_all):+.1f} points).")
if scores.mean() <= chance_all + TOL:
    print("  !! At or below chance. That is a REPORTABLE RESULT about transfer,")
    print("     not necessarily a defect -- but check the pipeline before you")
    print("     publish it, because a broken model looks identical from here.")
else:
    print("  Above chance, so the transfer gap below is a gap in a working model,")
    print("  not the distance between two kinds of noise.")

# --- compare against within-subject, if the sweep has been run ---------------
within = {}
if os.path.exists(WITHIN_CSV):
    with open(WITHIN_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["accuracy"]:
                within[int(row["subject"])] = float(row["accuracy"])

paired = [(s, within[s], sc) for s, sc in zip(tested, scores) if s in within]
if paired:
    w = np.array([p[1] for p in paired])
    c = np.array([p[2] for p in paired])
    print(f"\nWithin-subject (from {WITHIN_CSV}): {w.mean():.1%}")
    print(f"Cross-subject on the same people   : {c.mean():.1%}")
    print(f"THE GAP                            : {100 * (w.mean() - c.mean()):.1f} points")
    print(f"Cross beats within for {(c > w).sum()}/{len(paired)} subjects.")

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(paired))
    ax.bar(idx - 0.2, w, width=0.4, label="within-subject", color="#4a6fa5")
    ax.bar(idx + 0.2, c, width=0.4, label="cross-subject (LOSO)", color="#c0392b")
    ax.axhline(chance_all, color="#2c3e50", ls="--", lw=1.2, label="chance")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"S{p[0]:03d}" for p in paired], rotation=90, fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("The transfer gap: same people, trained on themselves vs. on others")
    ax.legend()
    fig.tight_layout()
    fig.savefig("cross_vs_within.png", dpi=120)
    print("\nSaved cross_vs_within.png")
else:
    print(f"\n({WITHIN_CSV} not found -- run sweep_subjects.py for the paired comparison.)")
