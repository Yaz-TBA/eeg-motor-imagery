"""Section 0 of permutation_design.py: the subjects, the exchangeable unit, the
observed values on P0, and falsification checks 4 and 6. Split out 2026-08-26;
the body is verbatim. build_setup() returns what every later section reads."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import csv
from types import SimpleNamespace

import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import assert_lattice, make_clf
from permdesign_lib import (
    IS_REGISTERED_RUN, N_ARM_A, N_ARM_B, N_GATE2, N_REPRO, N_SPLITS, RUNS, SEED,
    SUBJECTS_A, TOL, WITHIN_CSV, fmt, hdr, load_subject, note, sub,
)


def build_setup():
    # --------------------------------------------------------------------------- #
    # SECTION 0
    # --------------------------------------------------------------------------- #
    hdr("SECTION 0  SETUP, THE EXCHANGEABLE UNIT, AND WHAT THIS RUN IS")

    print(f"mne {mne.__version__}   numpy {np.__version__}   "
      f"sklearn {__import__('sklearn').__version__}")
    print(f"Draws per cell: arm A N = {N_ARM_A} (floor p = {1 / (N_ARM_A + 1):.5g}), "
      f"arm B N = {N_ARM_B} (floor p = {1 / (N_ARM_B + 1):.5g})")
    print(f"Pilot reproduction N = {N_REPRO}, sklearn-agreement gate N = {N_GATE2}")
    if IS_REGISTERED_RUN:
        print("This IS the registered run: every N matches the pre-registration.")
    else:
        print("*** NOT THE REGISTERED RUN. At least one N was overridden from the")
        print("*** environment. Nothing below may be reported as a result.")

    print("""
THIS RUN IS NOT BLIND, and saying otherwise would be the exact failure mode the
pre-registration exists to correct. A pilot (canon A269 / A270, marked
uncommitted) already ran both arms on 2026-07-25 and its numbers were read before
the pre-registration was written. They are quoted in SECTION 1 as the PRIOR.
Agreement below is a CONFIRMATION at higher resolution with a paired design, not
an independent discovery.
""")

    note("loading subjects for arm A")
    DATA_A = {}
    for s in SUBJECTS_A:
        X, y, runs = load_subject(s)
        DATA_A[s] = (X, y, runs)

    within_pub = {}
    with open(WITHIN_CSV) as fh:
        for row in csv.DictReader(fh):
            if row["accuracy"]:
                within_pub[int(row["subject"])] = float(row["accuracy"])

    sub("Arm A subjects: label structure, reference set sizes, and the published value")
    from math import comb

    OBS_A, P0_A = {}, {}
    for s in SUBJECTS_A:
        X, y, runs = DATA_A[s]
        n = len(y)
        n_h, n_f = int((y == 2).sum()), int((y == 3).sum())
        majority = max(n_h, n_f) / n
        P0 = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED).split(X, y))
        obs = cross_val_score(make_clf(), X, y, cv=P0, error_score="raise").mean()
        assert_lattice([obs], n, f"observed, subject {s}")
        OBS_A[s], P0_A[s] = obs, P0

        iid_size = comb(n, n_h)
        blk_size = 1
        for r in np.unique(runs):
            yy = y[runs == r]
            blk_size *= comb(len(yy), int((yy == 2).sum()))

        print(f"\nSubject {s}: n = {n}  ({n_h} hands / {n_f} feet)   "
          f"majority-class rate (CHANCE, never 50%) = {fmt(majority, n)}")
        print(f"  observed on P0 = {fmt(obs, n)}   against chance {fmt(majority, n)}   "
          f"({100 * (obs - majority):+.1f} points)")
        print(f"  sweep_results.csv says {within_pub[s]:.4f} = "
          f"{fmt(within_pub[s], n)}   match: {abs(obs - within_pub[s]) < 5e-5}")
        for i, r in enumerate(np.unique(runs)):
            yy = y[runs == r]
            seq = "".join("H" if v == 2 else "F" for v in yy)
            print(f"  run {RUNS[i]}: n = {len(yy)}  {int((yy == 2).sum())} hands / "
              f"{int((yy == 3).sum())} feet   {seq}")
        print(f"  reference set, i.i.d. exchangeable over all {n}: C({n},{n_h}) = {iid_size:.3e}")
        print(f"  reference set, exchangeable WITHIN RUN only     : {blk_size:.3e}  "
          f"(smaller by a factor of {iid_size / blk_size:.1f})")
        print(f"  The BLOCK null is the SMALLER reference set and therefore the WEAKER")
        print(f"  assumption: it does not assume trials are exchangeable across runs.")

    # Falsification 4 and 6: the baselines every comparison is measured against.
    assert abs(OBS_A[1] - 41 / 45) < TOL, (
        f"Subject 1's observed accuracy on P0 is {OBS_A[1]:.4%}, not 41/45 = 91.1%. P0 "
    "is not the published partition and every 'corrected against published' "
    "comparison below would be against the wrong baseline."
    )
    for s, k in ((17, 28), (19, 29)):
        assert abs(OBS_A[s] - k / 45) < TOL, (
            f"Subject {s} scores {OBS_A[s]:.4%} on this pipeline, not {k}/45 = "
        f"{k / 45:.1%} as sweep_results.csv records. The median-selection rule "
        "selected on numbers this pipeline does not produce; the subject choice "
        "must be redone from recomputed values before any null is run."
        )
    print("\nFALSIFICATION CHECKS 4 and 6 PASS: subject 1 is exactly 41/45 on P0, and "
      "subjects\n17 and 19 reproduce sweep_results.csv at 28/45 and 29/45.")

    return SimpleNamespace(DATA_A=DATA_A, OBS_A=OBS_A, P0_A=P0_A)
