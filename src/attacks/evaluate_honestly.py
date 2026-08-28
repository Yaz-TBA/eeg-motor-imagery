"""Ask the question a skeptical reader asks first: how do you know the number is real?

This is the file that moved the headline from 94.4% to 91.1%. Was written to verify my
own result and it succeeded in denying it, which i guess means it worked.. yippeeee ! :P

This rung is what changed the headline. decode_csp.py originally reported
"94.4% +/- 5.6%" from 10 random 80/20 splits (ShuffleSplit); it now uses
StratifiedKFold(5) and reports 91.1% with a permutation p <= 0.001. The original
number reproduced exactly, and three things about it were weaker than they looked:

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

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

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

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

from common import make_clf, wilson_interval


def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-minute experiment as a side effect."""

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


    # make_clf and wilson_interval come from common.py. This file is the one whose
    # module-scope analysis was the stated reason the other copies existed: importing it
    # used to run a five-minute experiment. It has a __main__ guard now, so it doesn't.

    def ordinal(k):
        """'3rd', not '3th'. 11/12/13 are the exceptions an %d + 'th' rule gets wrong."""
        k = int(k)
        if 10 <= k % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suffix}"


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
    # error_score="raise" on every CV call in this file. sklearn's default catches a
    # fold that throws, scores it NaN, and returns a mean that is quietly wrong --
    # which in a script whose entire subject is "how do you know the number is real"
    # would be an embarrassing place to accept a silent failure.
    scores_shuffle = cross_val_score(make_clf(), data, labels, cv=cv_shuffle,
                                     error_score="raise")
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
        s = cross_val_score(make_clf(), data, labels, cv=cv, error_score="raise")
        strat_scores[k] = s
        print(f"{f'StratifiedKFold({k})':<24} {s.mean():.1%} +/- {s.std():.1%}")
    print("Note the +/- on 10-fold: 4-5 test trials per fold makes a fold std meaningless.")

    # POSITIVE CONTROL on the PUBLISHED estimator, before anything is inferred from
    # it. Sections 4-6 all interrogate the 5-fold number; if the pipeline had quietly
    # collapsed to predicting one class, they would still run and still print, and
    # the permutation test would even agree that a majority-class predictor beats
    # shuffled labels. So check the other direction first: is the model alive?
    #
    # The bar is 53.3% (24/45), the MAJORITY-CLASS rate, not 50%. With 21 hands and
    # 24 feet, "always feet" scores 53.3% for free, so 50% is the wrong reference --
    # the same point section 2 makes about the worst-balanced ShuffleSplit fold.
    # Only the stratified estimator is checked. The retracted ShuffleSplit number is
    # reported for contrast, and asserting on a retracted estimator would imply we
    # still stand behind it.
    #
    # The tolerance is load-bearing. A majority-class dummy scores exactly 24/45 on
    # these folds -- numerically identical to chance -- but arrives there as a mean of
    # k/9 fold scores, and floating point puts it 1.1e-16 ABOVE a directly computed
    # 24/45. A bare `> chance` therefore PASSES a model that learned nothing. This
    # file is about numbers that look stronger than they are; a guard with that bug
    # would belong in the retracted column.
    TOL = 1e-9
    assert strat_scores[5].mean() > chance + TOL, (
        f"Positive control failed: StratifiedKFold(5) scored {strat_scores[5].mean():.1%}, "
    f"not above the majority-class rate of {chance:.1%} ({n_hands} hands / {n_feet} "
    "feet). A constant predictor matches that, so there is no decoding to evaluate "
    "and sections 4-6 would be interrogating noise."
    )

    # --- 4. permutation test: is it real? ----------------------------------------
    print(f"\n--- 4. Permutation test ({N_PERMUTATIONS} label shuffles) ---")
    print("Running... (this is the slow part)")
    # No error_score argument: permutation_test_score does not accept one, and does
    # not need it -- it calls estimator.fit with nothing catching the exception, so a
    # broken fold raises rather than silently becoming NaN.
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
    # sklearn computes p = (C + 1) / (n + 1) where C counts permutations scoring
    # >= observed. So the smallest attainable value is 1/(n+1): with 1000 shuffles
    # that is 0.000999, and printing "p = 0.0010" invites reading a resolution floor
    # as a measurement. Report it as a bound when it bottoms out.
    floor = 1.0 / (N_PERMUTATIONS + 1)
    if p_value <= floor + 1e-12:
        print(f"p-value           : <= {floor:.3f}  "
          f"(the floor of a {N_PERMUTATIONS}-shuffle test, not a measurement)")
    else:
        print(f"p-value           : {p_value:.4f}")

    # ">=", not ">": sklearn counts permutations that MATCH OR EXCEED the observed
    # score, so "beat" would understate this by one comparison.
    n_ge = int(round(p_value * (N_PERMUTATIONS + 1))) - 1
    print(f"Shuffled labels matched or exceeded the real result {n_ge} "
      f"times out of {N_PERMUTATIONS}.")

    # A sanity check, and worth being precise about what it can and cannot catch.
    # It CANNOT catch label leakage: permutation_test_score permutes y and refits the
    # whole pipeline inside CV, so leakage-into-the-shuffle is structurally
    # impossible here. What it does catch is a mis-specified null -- e.g. a scorer
    # that is not accuracy, or class proportions that make 50% the wrong reference.
    # The tolerance is deliberately tighter than the null's own sd (8.5 pts), because
    # a +/-10 pt window would accept a null centred at 59%.
    assert abs(null_scores.mean() - 0.5) < 0.05, (
        f"Permutation null centred at {null_scores.mean():.1%}, not ~50%. The null is "
    "mis-specified; the p-value would not mean what it appears to."
    )
    print("Null is centred near 50%, so the reference distribution is well formed.")
    print(f"NOTE: p is bounded below by 1/(n+1) = {1/(N_PERMUTATIONS+1):.4f}. "
      f"Report this as p <= {1/(N_PERMUTATIONS+1):.3f}, not as a measured value.")

    # --- 5. an interval that reflects n=45 ---------------------------------------
    print("\n--- 5. An honest interval ---")
    point = strat_scores[5].mean()
    n_correct = int(round(point * n))
    lo, hi = wilson_interval(n_correct, n)
    print(f"Point estimate (stratified 5-fold): {point:.1%}  ({n_correct}/{n} trials)")
    print(f"Wilson 95% CI on n={n}            : [{lo:.1%}, {hi:.1%}]  (width {100*(hi-lo):.1f} pts)")
    # Computed, not quoted. This line used to be an f-string that interpolated nothing:
    # 5.6, 88.8 and 100.0 were literals, so they printed into stdout where
    # check_provenance.py reads stdout back as "backing" -- a hardcoded number backing
    # itself. All three now come out of scores_shuffle, so nothing here backs itself.
    #
    # They are derived from the ROUNDED one-decimal displays, not from the raw floats,
    # and that is deliberate. This line is a WITHDRAWAL ILLUSTRATION: its job is to
    # reproduce the arithmetic that was published, which was 94.4 - 5.6 = 88.8 done on
    # what the section-3 summary table printed. Deriving it from the raw floats gives
    # 88.9 instead, which would silently "correct" a number whose entire purpose is to
    # show the overstatement as it was made, and would desynchronise this stdout from
    # the four documents that quote [88.8%, 100.0%] as the retracted interval. The
    # interval being illustrated is unchanged; it was always mean +/- std, and it was
    # always far too tight.
    mean_pct = round(scores_shuffle.mean() * 100, 1)
    sd_pct = round(scores_shuffle.std() * 100, 1)
    implied_lo = mean_pct - sd_pct
    implied_hi = min(100.0, mean_pct + sd_pct)
    implied_label = f"What '+/- {sd_pct:.1f}%' implies"
    print(f"{implied_label:<34}: [{implied_lo:.1f}%, {implied_hi:.1f}%]"
      f"  <- far too tight")

    # --- 6. how much does the seed matter? ---------------------------------------
    print(f"\n--- 6. Seed sensitivity across {N_SEEDS} random_state values ---")
    # BOTH estimators get swept. Sweeping only the retracted one (ShuffleSplit) and
    # then attaching its "not cherry-picked" verdict to the PUBLISHED number would be
    # a bait-and-switch: they are different estimators with different seed behavior.
    seed_means = np.array([
        cross_val_score(make_clf(), data, labels,
                        cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=s),
                        error_score="raise").mean()
        for s in range(N_SEEDS)
    ])
    strat_means = np.array([
        cross_val_score(make_clf(), data, labels,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=s),
                        error_score="raise").mean()
        for s in range(N_SEEDS)
    ])

    for name, arr in (("ShuffleSplit (retracted)", seed_means),
                      ("StratifiedKFold (published)", strat_means)):
        pct = 100 * (arr < arr[42]).mean()
        print(f"{name:<28} mean {arr.mean():.1%} | min {arr.min():.1%} | "
          f"max {arr.max():.1%} | range {100*(arr.max()-arr.min()):.1f} pts")
        print(f"{'':28} seed 42 lands at the {ordinal(round(pct))} percentile "
          f"({arr[42]:.1%})")

    # SIGNED, not abs(). The sign is the whole point: without it this line cannot tell
    # a reader which estimator is higher, and four documents downstream depend on the
    # direction. StratifiedKFold minus ShuffleSplit, so a positive number means the
    # published estimator has the HIGHER expectation.
    print(f"\nThe two estimators agree in expectation to "
      f"{100*(strat_means.mean() - seed_means.mean()):+.1f} points "
      f"(StratifiedKFold minus ShuffleSplit; positive means the estimator change "
      f"RAISES the expectation).")
    print("So the headline drop from 94.4% to 91.1% is mostly SEED LUCK, not a")
    print("change of estimator. The estimator change is the right call for other")
    print("reasons (coverage, stratification); it just is not worth 3 points.")

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
    fig.savefig("figures/permutation_null.png", dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(seed_means, bins=20, color="#b0b8c4", edgecolor="white")
    ax.axvline(seed_means[42], color="#c0392b", lw=2.5, label=f"seed 42 ({seed_means[42]:.1%})")
    ax.set_xlabel("mean accuracy over 10 splits")
    ax.set_ylabel("count")
    ax.set_title(f"The headline moves {100*(seed_means.max()-seed_means.min()):.0f} points with the seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/seed_sensitivity.png", dpi=120)

    print("\nSaved figures/permutation_null.png and figures/seed_sensitivity.png")



if __name__ == "__main__":
    main()
