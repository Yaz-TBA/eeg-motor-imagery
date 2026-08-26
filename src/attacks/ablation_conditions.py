"""The six registered conditions of ablate_channels.py at seed 42, the
attainable-accuracy lattice, and the positive and negative controls. Split out
2026-08-26; the stage bodies are verbatim from that file."""

import ablation_data  # noqa: F401  -- installs the common.py path first

from mne.datasets import eegbci
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from common import FRONTOPOLAR, SENSORIMOTOR, make_clf
from ablation_data import RUNS, SEED, SUBJECT, TOL, derive_channel_sets, load_data


def load_and_partition():
    # --- load + preprocess (identical to decode_csp.py, so the numbers are comparable) ---
    cropped, labels, groups, ch_names, n, n_hands, n_feet, majority = load_data()
    # The SECONDARY arm below re-references from the raw EDFs, so it needs the paths.
    # This is a cached path lookup, not a second read.
    edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)

    print(f"\nSubject {SUBJECT}, runs {RUNS} (imagined both fists vs. both feet)")
    print(f"{n} trials ({n_hands} hands, {n_feet} feet) | "
          f"majority class = {majority:.1%}")
    print(f"Trials per run: " + ", ".join(
        f"run {r}={int((groups == i).sum())}" for i, r in enumerate(RUNS)))

    # --- the complement, as an exact set difference -------------------------------
    # All three are derived from the montage, never typed. The expression is the same
    # one the hostile pass used at its line 84, so the two implementations pick
    # identically or the disagreement is real rather than a transcription slip.
    COMPLEMENT, WIDE, NOT_WIDE = derive_channel_sets(ch_names)

    print("\n--- The deleted sets, as exact set differences ---")
    assert len(SENSORIMOTOR) == 17, f"SENSORIMOTOR is {len(SENSORIMOTOR)} channels, not 17"
    assert len(COMPLEMENT) == 47, f"COMPLEMENT is {len(COMPLEMENT)} channels, not 47"
    assert len(WIDE) == 21, f"WIDE is {len(WIDE)} channels, not 21"
    assert len(NOT_WIDE) == 43, f"NOT_WIDE is {len(NOT_WIDE)} channels, not 43"
    # A complement is a PARTITION or it is not a complement, and the name would then
    # be false. Both halves are checked: the union covers the montage and the
    # intersection is empty. Either one alone can pass on a set that overlaps.
    assert set(SENSORIMOTOR) | set(COMPLEMENT) == set(ch_names), (
        "SENSORIMOTOR + COMPLEMENT do not cover the montage")
    assert set(SENSORIMOTOR) & set(COMPLEMENT) == set(), (
        "SENSORIMOTOR and COMPLEMENT overlap, so COMPLEMENT is not a complement")
    assert set(WIDE) | set(NOT_WIDE) == set(ch_names) and not (set(WIDE) & set(NOT_WIDE))
    print(f"montage {len(ch_names)} ch | SENSORIMOTOR {len(SENSORIMOTOR)} + "
          f"COMPLEMENT {len(COMPLEMENT)} = {len(SENSORIMOTOR) + len(COMPLEMENT)}, "
          f"disjoint partition: asserted, not counted by hand")
    print(f"WIDE FC/C/CP {len(WIDE)} + NOT_WIDE {len(NOT_WIDE)} = "
          f"{len(WIDE) + len(NOT_WIDE)}, disjoint partition: asserted")
    print(f"COMPLEMENT ({len(COMPLEMENT)} ch, montage order): {' '.join(COMPLEMENT)}")
    # Printed so a disagreement with any other implementation of the same set
    # difference is diagnosable as a set difference AND as an ordering difference,
    # rather than argued about.
    print(f"COMPLEMENT keeps these four peri-Rolandic electrodes: "
          f"{[c for c in COMPLEMENT if c in ('FC5', 'FC6', 'CP5', 'CP6')]}")
    print(f"COMPLEMENT keeps this temporal ring (temporalis EMG territory, UNBOUNDED "
          f"in this repo): {[c for c in COMPLEMENT if c in ('T7', 'T8', 'T9', 'T10', 'TP7', 'TP8')]}")

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
    return (cropped, labels, groups, ch_names, n, n_hands, n_feet, majority,
            edf_paths, COMPLEMENT, WIDE, NOT_WIDE)


def run_conditions(cropped, labels, groups, ch_names, n, n_hands, n_feet, majority,
                   COMPLEMENT, NOT_WIDE):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logo = LeaveOneGroupOut()

    conditions = [
        ("all 64 channels", ch_names, skf, None),
        (f"sensorimotor only ({len(SENSORIMOTOR)} ch)", SENSORIMOTOR, skf, None),
        (f"frontopolar only ({len(FRONTOPOLAR)} ch)", FRONTOPOLAR, skf, None),
        (f"sensorimotor DELETED ({len(COMPLEMENT)} kept)", COMPLEMENT, skf, None),
        (f"wide FC/C/CP DELETED ({len(NOT_WIDE)} kept)", NOT_WIDE, skf, None),
        ("all 64, leave-one-run-out", ch_names, logo, groups),
    ]

    results = []
    per_trial = {}   # condition -> boolean "was this trial predicted correctly"
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
        per_trial[name] = (pred == labels)

    # Look results up BY NAME, not by index. The old code read results[0] through
    # results[3] and would have kept running, silently reading the wrong condition,
    # if anyone ever inserted a row above the one they wanted.
    by_name = {r[0]: r for r in results}
    ALL64 = "all 64 channels"
    SMC = f"sensorimotor only ({len(SENSORIMOTOR)} ch)"
    FP = f"frontopolar only ({len(FRONTOPOLAR)} ch)"
    COMP = f"sensorimotor DELETED ({len(COMPLEMENT)} kept)"
    NWIDE = f"wide FC/C/CP DELETED ({len(NOT_WIDE)} kept)"
    LORO = "all 64, leave-one-run-out"

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
        assert abs(n_correct - round(n_correct)) < 1e-9
    print(f"\nAll {len(results)} accuracies land on the k/{n} lattice, as they must.")

    # --- controls, before anything is read off the table --------------------------
    # Both of these run BEFORE the ten-seed sweep and the permutation test, because a
    # dead pipeline or a live negative control makes every number below unreadable
    # and there is no reason to spend three minutes discovering that.
    maj_correct = max(n_hands, n_feet)
    _all64_scores = by_name[ALL64][2]
    # TOL is NOT decoration. A majority-class DummyClassifier scores exactly 24/45,
    # and floating point puts it 1.1e-16 ABOVE a bare `> majority`, so a bare
    # comparison passes a model that has learned nothing. Same trap, same 1e-9
    # tolerance, as decode_csp.py lines 111-118 and sweep_subjects.py.
    assert _all64_scores.mean() > majority + TOL, (
        f"POSITIVE CONTROL FAILED: all-64 scored {_all64_scores.mean():.1%}, which "
        f"does not beat the majority-class rate {majority:.1%}. The pipeline is not "
        "decoding anything and no other row in this table means what it says.")
    _fp_off = abs(by_name[FP][3] - maj_correct)
    assert _fp_off <= 3, (
        f"NEGATIVE CONTROL FAILED: frontopolar-only is {_fp_off} trials off the "
        f"majority rate ({by_name[FP][3]}/{n} vs {maj_correct}/{n}). The control has "
        "come alive; this run is not interpretable.")
    print(f"Positive control: all 64 at {by_name[ALL64][3]}/{n} beats majority "
          f"{maj_correct}/{n} by more than TOL={TOL:g}. PASS.")
    print(f"Negative control: frontopolar-only is {_fp_off} trial"
          f"{'' if _fp_off == 1 else 's'} off the majority rate, within the "
          f"registered 3. PASS.")
    return (skf, conditions, results, per_trial, by_name,
            ALL64, SMC, FP, COMP, NWIDE, LORO, maj_correct, _fp_off)
