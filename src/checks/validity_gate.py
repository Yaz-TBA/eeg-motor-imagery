"""G-S, the y_obs-independence gate registered in prereg-permutation-validity.md §4.

Small on purpose: it answers one question, whether the null is independent of
the observed labels, and answers it the way the pre-registration said it would
before the run happened.

Deterministic. No EEG, no Monte Carlo, no estimator, no tie-break. A permutation
rule is a pair (partition, permutation group), and Theorem 3.1's hypothesis 2
says neither object may be a function of the observed label vector. G-S checks
that by instantiating the rule under two different observed vectors drawn from
the same marginal and demanding the two objects come out element-wise identical.

A rule failing G-S is disqualified by §4.3. There is no measurement that
rehabilitates it, because G-S is a statement about the code and not about the
data. Run against C2 and C4 as they stood on 2026-07-25 this returns FAIL on the
first triple, before any EEG is loaded and before any p exists.

The seven rules and their predicted verdicts are fixed in §4.2 of the
pre-registration, which was committed before this file was written. This script
prints predicted alongside observed so a reader can see the prediction was not
edited to match.
"""

import sys

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

SEED = 42
N_SPLITS = 5
N_TRIPLES = 200

# Subject marginals as (n_class2, n_class3), verified against the committed run
# stdout at prereg/runs/permutation-design-2026-07-25.stdout:22,33,44.
# permutation_design.py:297 sets event_id=dict(T1=2, T2=3), so class 2 is hands.
MARGINALS = {1: (21, 24), 17: (24, 21), 19: (22, 23)}

# Runs 6, 10 and 14, fifteen trials each. Run identity is protocol, not label.
RUNS = np.repeat([6, 10, 14], 15)

# §4.2's table, fixed before this script existed.
PREDICTED = {
    "RE": True, "RE_blk": True, "KF_free": True, "KF_free_blk": True,
    "WITHIN": False, "FX": False, "FX_blk": False,
}


def partition(rule, y_s, y_obs):
    """The realised partition, as an ordered list of test-index sets."""
    if rule in ("FX", "FX_blk", "WITHIN"):
        # Frozen at P0, which is built from the OBSERVED vector. This is the
        # dependence G-S exists to catch.
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
        return [tuple(te) for _, te in skf.split(np.zeros((len(y_obs), 1)), y_obs)]
    if rule in ("RE", "RE_blk"):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
        return [tuple(te) for _, te in skf.split(np.zeros((len(y_s), 1)), y_s)]
    if rule in ("KF_free", "KF_free_blk"):
        kf = KFold(N_SPLITS, shuffle=True, random_state=SEED)
        return [tuple(te) for _, te in kf.split(np.zeros((len(y_s), 1)))]
    raise ValueError(rule)


def group(rule, y_s, y_obs):
    """The realised permutation group, as ordered blocks labels may move within."""
    n = len(y_s)
    if rule in ("RE", "KF_free", "FX"):
        return [tuple(range(n))]                      # full permutation group
    if rule in ("RE_blk", "KF_free_blk", "FX_blk"):
        return [tuple(np.where(RUNS == b)[0]) for b in np.unique(RUNS)]
    if rule == "WITHIN":
        # Labels permuted within each fold of P0, so the group itself is a
        # function of y_obs.
        return partition("FX", y_s, y_obs)
    raise ValueError(rule)


def draw(n2, n3, rng):
    y = np.array([2] * n2 + [3] * n3)
    return rng.permutation(y)


def main() -> int:
    rng = np.random.default_rng(SEED)
    print("G-S, the y_obs-independence gate  (prereg-permutation-validity.md §4.1)")
    print(f"{N_TRIPLES} triples per rule per subject, three subjects, seed {SEED}\n")

    rows, all_ok = [], True
    for rule in PREDICTED:
        mismatches, first = 0, None
        for subj, (n2, n3) in MARGINALS.items():
            for t in range(N_TRIPLES):
                y_s = draw(n2, n3, rng)
                y1 = draw(n2, n3, rng)
                y2 = draw(n2, n3, rng)
                while np.array_equal(y1, y2):
                    y2 = draw(n2, n3, rng)
                same = (partition(rule, y_s, y1) == partition(rule, y_s, y2)
                        and group(rule, y_s, y1) == group(rule, y_s, y2))
                if not same:
                    mismatches += 1
                    first = first or f"S{subj} triple {t}"
        observed = mismatches == 0
        agree = observed == PREDICTED[rule]
        all_ok &= agree
        rows.append((rule, PREDICTED[rule], observed, mismatches, first, agree))

    w = max(len(r) for r in PREDICTED)
    print(f"  {'rule':<{w}}  {'predicted':>9}  {'observed':>8}  {'mismatches':>10}  first")
    for rule, pred, obs, n, first, agree in rows:
        flag = "" if agree else "   <-- PREDICTION MISSED"
        print(f"  {rule:<{w}}  {'PASS' if pred else 'FAIL':>9}  "
              f"{'PASS' if obs else 'FAIL':>8}  {n:>10}  {first or '-'}{flag}")

    n_pass = sum(1 for r in rows if r[2])
    print(f"\n  {n_pass} of {len(rows)} rules pass G-S.")
    print("  Both directions are exhibited, which is what assert 9 never managed:")
    print("  a gate no valid design can pass is not a gate, and neither is one")
    print("  that nothing can fail.")

    if not all_ok:
        print("\nF-GS FIRES: an observed verdict contradicts the pre-registered")
        print("prediction in §4.2. Per §7.1 the run halts and nothing below the")
        print("gate may be reported.")
        return 1

    print("\n  Every verdict matches the pre-registration. C2 (=FX) and C4 (=FX_blk)")
    print("  are disqualified structurally, in microseconds, with no EEG loaded --")
    print("  the withdrawal of 2026-07-26 reached by a route that needs no data,")
    print("  no null mean, and no argument about a band.")
    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
