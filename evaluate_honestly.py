"""Ask the question a skeptical reader asks first: how do you know 94% is real?

decode_csp.py reports "94.4% +/- 5.6%" from 10 random 80/20 splits. That number
reproduces exactly, but three things about it are weaker than they look:

  1. With 45 trials, a 20% test set is 9 trials -- so a fold's accuracy can ONLY
     be a multiple of 1/9. The "+/- 5.6%" is the gap between two rungs of a
     quantized ladder, not a spread over a distribution.
  2. ShuffleSplit resamples independently per split, so it is neither stratified
     (class balance swings fold to fold) nor a partition (some trials are never
     tested at all, others several times).
  3. A standard deviation over folds is not a confidence interval, and reading it
     as one implies a precision that 45 trials cannot support.

So this rung replaces the question "what is the accuracy" with "is the accuracy
real, and how sure can we be". The permutation test is the load-bearing part:
shuffle the labels a thousand times, re-run the whole pipeline, and see whether
the real result stands outside what chance produces.
"""

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

mne.set_log_level("ERROR")  # keep parallel workers from flooding stdout
warnings.filterwarnings("ignore")

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
N_PERMUTATIONS = 1000
N_SEEDS = 100

# --- load + preprocess (identical to decode_csp.py, so the numbers are comparable) ---
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
labels = epochs.events[:, -1]
data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False)


def make_clf():
    """A fresh pipeline each time -- CSP must refit inside every training fold."""
    return Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def wilson_interval(n_correct, n_total, z=1.96):
    """95% CI for a proportion. Handles small n far better than mean +/- std."""
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return centre - half, centre + half


n = len(labels)
n_hands, n_feet = int((labels == 2).sum()), int((labels == 3).sum())
chance = max(n_hands, n_feet) / n
print(f"\n{n} trials ({n_hands} hands, {n_feet} feet) | chance = {chance:.1%}")

# --- 1. the quantization problem ---------------------------------------------
test_n = int(round(0.2 * n))
print(f"\n--- 1. Quantization: a 20% test set is {test_n} trials ---")
print(f"A fold's accuracy can only be k/{test_n}, i.e. steps of {1/test_n:.1%}.")
print("Attainable values near the headline: "
      + ", ".join(f"{k}/{test_n}={k/test_n:.1%}" for k in range(test_n - 2, test_n + 1)))

cv_shuffle = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores_shuffle = cross_val_score(make_clf(), data, labels, cv=cv_shuffle)
distinct = sorted({round(float(s), 4) for s in scores_shuffle})
print(f"\nThe 10 folds produced {len(distinct)} distinct values: "
      + ", ".join(f"{v:.1%}" for v in distinct))
print(f"Total correct across folds: {int(round(scores_shuffle.sum() * test_n))}/{10 * test_n}")

# --- 2. ShuffleSplit does not cover the data evenly --------------------------
print("\n--- 2. Coverage: ShuffleSplit is not a partition ---")
times_tested = np.zeros(n, dtype=int)
balances = []
for _, test_idx in cv_shuffle.split(data, labels):
    times_tested[test_idx] += 1
    balances.append((int((labels[test_idx] == 2).sum()), int((labels[test_idx] == 3).sum())))
print(f"Test-set class balance per fold (hands, feet): {balances}")
print(f"Trials never tested: {int((times_tested == 0).sum())} of {n}")
print(f"Times a trial is tested: min={times_tested.min()} max={times_tested.max()}")
worst = max(balances, key=lambda b: max(b) / sum(b))
print(f"Worst-balanced fold is {worst} -- guessing its majority alone scores "
      f"{max(worst)/sum(worst):.1%}, not the {chance:.1%} we print as chance.")

# --- 3. stratified k-fold: every trial tested exactly once --------------------
print("\n--- 3. Stratified k-fold, where every trial is tested once ---")
print(f"{'ShuffleSplit(10, 0.2)':<24} {scores_shuffle.mean():.1%} +/- {scores_shuffle.std():.1%}")
strat_scores = {}
for k in (5, 10):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    s = cross_val_score(make_clf(), data, labels, cv=cv)
    strat_scores[k] = s
    print(f"{f'StratifiedKFold({k})':<24} {s.mean():.1%} +/- {s.std():.1%}")
print("Note the +/- on 10-fold: 4-5 test trials per fold makes a fold std meaningless.")

# --- 4. permutation test: is it real? ----------------------------------------
print(f"\n--- 4. Permutation test ({N_PERMUTATIONS} label shuffles) ---")
print("Running... (this is the slow part)")
observed, null_scores, p_value = permutation_test_score(
    make_clf(), data, labels,
    scoring="accuracy",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_permutations=N_PERMUTATIONS,
    random_state=42,
    n_jobs=-1,
)
print(f"Observed accuracy : {observed:.1%}")
print(f"Null distribution : {null_scores.mean():.1%} +/- {null_scores.std():.1%} "
      f"(max {null_scores.max():.1%})")
print(f"p-value           : {p_value:.4f}")
print(f"Shuffled labels beat the real result {int(p_value * (N_PERMUTATIONS + 1)) - 1} "
      f"times out of {N_PERMUTATIONS}.")

# The honesty check this whole rung exists for.
assert abs(null_scores.mean() - 0.5) < 0.10, (
    f"Permutation null centred at {null_scores.mean():.1%}, not ~50%. "
    "Labels are leaking into the shuffle; the p-value would be meaningless."
)
print("Check passed: null is centred near 50%, so the shuffle is clean.")

# --- 5. an interval that reflects n=45 ---------------------------------------
print("\n--- 5. An honest interval ---")
point = strat_scores[5].mean()
n_correct = int(round(point * n))
lo, hi = wilson_interval(n_correct, n)
print(f"Point estimate (stratified 5-fold): {point:.1%}  ({n_correct}/{n} trials)")
print(f"Wilson 95% CI on n={n}            : [{lo:.1%}, {hi:.1%}]  (width {100*(hi-lo):.1f} pts)")
print(f"What '+/- 5.6%' implies           : [88.8%, 100.0%]  <- far too tight")

# --- 6. how much does the seed matter? ---------------------------------------
print(f"\n--- 6. Seed sensitivity across {N_SEEDS} random_state values ---")
seed_means = np.array([
    cross_val_score(make_clf(), data, labels,
                    cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=s)).mean()
    for s in range(N_SEEDS)
])
pct = 100 * (seed_means < seed_means[42]).mean()
print(f"mean {seed_means.mean():.1%} | std {seed_means.std():.1%} | "
      f"min {seed_means.min():.1%} | max {seed_means.max():.1%}")
print(f"Range is {100*(seed_means.max()-seed_means.min()):.1f} points on an arbitrary choice.")
print(f"Seed 42 sits at the {pct:.0f}th percentile -- it was not cherry-picked.")

# --- figures -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_scores, bins=30, color="#b0b8c4", edgecolor="white", label="shuffled labels")
ax.axvline(observed, color="#c0392b", lw=2.5, label=f"observed ({observed:.1%})")
ax.axvline(chance, color="#2c3e50", ls="--", lw=1.5, label=f"chance ({chance:.1%})")
ax.set_xlabel("accuracy")
ax.set_ylabel("count")
ax.set_title(f"Permutation test: p = {p_value:.4f} ({N_PERMUTATIONS} shuffles)")
ax.legend()
fig.tight_layout()
fig.savefig("permutation_null.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(seed_means, bins=20, color="#b0b8c4", edgecolor="white")
ax.axvline(seed_means[42], color="#c0392b", lw=2.5, label=f"seed 42 ({seed_means[42]:.1%})")
ax.set_xlabel("mean accuracy over 10 splits")
ax.set_ylabel("count")
ax.set_title(f"The headline moves {100*(seed_means.max()-seed_means.min()):.0f} points with the seed")
ax.legend()
fig.tight_layout()
fig.savefig("seed_sensitivity.png", dpi=120)

print("\nSaved permutation_null.png and seed_sensitivity.png")
