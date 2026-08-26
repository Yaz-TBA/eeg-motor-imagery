"""Section 4 of permutation_design.py: the secondary mechanism probe, explanation
only, changes no number. Split out 2026-08-26; the body is verbatim."""

import permdesign_lib  # noqa: F401  -- environment and path setup runs first

import numpy as np

from permdesign_lib import N_ARM_A, N_SPLITS, SUBJECTS_A, hdr


def run_mechanism(D, ARM_A):
    P0_A = D.P0_A

    # --------------------------------------------------------------------------- #
    # SECTION 4  SECONDARY MECHANISM PROBE
    # --------------------------------------------------------------------------- #
    hdr("SECTION 4  SECONDARY MECHANISM PROBE (EXPLANATION ONLY, CHANGES NO NUMBER)")

    print("""Registered SEPARATELY and in advance, because this project's round-one
failure mode was inventing the mechanism story in the same breath as the number,
and because A269 already records one FAILED directional prediction (a heavier
right tail for the restricted null, which reversed).

THE MECHANISM UNDER TEST: under a FIXED partition a permuted-label test fold can be
strongly unbalanced, and the complementary training folds are then unbalanced the
OTHER way, so the classifier's induced prior is anti-correlated with the test
fold's majority and those folds score WORSE. Re-stratification forbids such folds
entirely, which would be why its null sits higher.

PREDICTION: in C2, per-fold test-set imbalance correlates NEGATIVELY with per-fold
accuracy. If the correlation is zero or positive the mechanism is WRONG and gets
withdrawn. mean(d) and every p above are UNAFFECTED either way, because the
measurement does not depend on the explanation.""")

    MECH = {}
    for s in SUBJECTS_A:
        d = ARM_A[s]
        n, P0 = d["n"], P0_A[s]
        imb, acc_f = [], []
        for j, yp in enumerate(d["perms_iid"]):
            for f, (_, te) in enumerate(P0):
                lab = yp[te]
                imb.append(abs(int((lab == 2).sum()) - int((lab == 3).sum())) / len(te))
                acc_f.append(d["pf2"][j, f])
        imb, acc_f = np.asarray(imb), np.asarray(acc_f)
        r = float(np.corrcoef(imb, acc_f)[0, 1])
        # The re-stratified cell's own imbalance, for contrast: under re-stratification
        # every test fold is forced to (4,5) or (5,4), so its imbalance is a CONSTANT
        # and no correlation is even definable there. That is the mechanism's premise.
        print(f"\n  Subject {s}, C2 (fixed partition), {len(imb)} folds "
          f"({N_ARM_A} replicates x {N_SPLITS}):")
        print(f"    distinct test-fold imbalances seen: "
              + ", ".join(f"{v:.4f}" for v in sorted(set(np.round(imb, 6)))[:9]))
        print(f"    Pearson r(imbalance, per-fold accuracy) = {r:+.4f}")
        print(f"    mean per-fold accuracy at the most balanced imbalance "
          f"{imb.min():.4f}: {acc_f[imb == imb.min()].mean():.2%}")
        print(f"    mean per-fold accuracy at the most extreme imbalance "
          f"{imb.max():.4f}: {acc_f[imb == imb.max()].mean():.2%}")
        print(f"    MECHANISM PREDICTION (negative r) "
          f"{'HOLDS' if r < 0 else 'FAILS, and is withdrawn'}")
        MECH[s] = r

    return MECH
