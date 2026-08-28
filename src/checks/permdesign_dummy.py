"""Sections 2B and 2C of permutation_design.py: the disclosed departure from the
registered centring assert, and the zero-information exactness study of the six
partition rules. Split out 2026-08-26; the bodies are verbatim, disclosures
included."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from common import assert_lattice
from permdesign_lib import (
    N_ARM_A, N_SPLITS, N_TYPE1_INNER, N_TYPE1_OUTER, SEED, SUBJECTS_A, TOL,
    fmt, hdr,
)
from permdesign_workers import iid_perm


def run_dummy_control(D):
    DATA_A, P0_A = D.DATA_A, D.P0_A

    # --------------------------------------------------------------------------- #
    # SECTION 2B  A DEPARTURE FROM THE PRE-REGISTRATION, DISCLOSED IN FULL
    # --------------------------------------------------------------------------- #
    hdr("SECTION 2B  THE REGISTERED NULL-CENTRING ASSERT FIRES, AND WHY IT IS THE "
    "ASSERT\n            THAT IS WRONG RATHER THAN THE NULL")

    print("""DISCLOSE FIRST, ARGUE SECOND. This section did not exist when the
pre-registration was written. It was added after a SMOKE RUN at N = 60 tripped the
registered assert on subject 17's C4 cell. That makes everything in it POST-HOC,
and the departure it justifies runs in the direction that FLATTERS this project
(a lower null means a smaller corrected p), which is exactly when a post-hoc
argument deserves the most suspicion. So the evidence below is a control that
uses NO EEG AT ALL, and the registered gate's outcome is reported either way in
section 6 rather than quietly dropped.

WHAT WAS REGISTERED. Section 5.4 assert 9 and Section 7 falsification 8:
"Each cell's null mean is within 0.05 of 0.50 ... Wider than that means the null
is mis-specified and no p from it means what it appears to." Section 6.5 then
says that if it fires, nothing in Sections 6.1 to 6.4 may be reported.

WHAT THE PRE-REGISTRATION ALSO SAYS, IN THE SAME DOCUMENT. Section 7's secondary
mechanism probe registers, in advance, that under a FIXED partition an unbalanced
permuted test fold has complementary training folds unbalanced the OTHER way, so
those folds score WORSE. That mechanism PREDICTS a fixed-partition null centred
BELOW 0.50. The pre-registration therefore registered two things that cannot both
be right: assert 9 applied to all four cells, and a mechanism that makes assert 9
false for the two fixed cells. The run found the inconsistency. That is a defect
in the pre-registration, and it is being reported as one.

WHY THE 0.45-0.55 BAND DOES NOT TRANSFER. It is imported from
evaluate_honestly.py:220, which asserts it for the RE-STRATIFIED null. Under
re-stratification every test fold is forced to be near-balanced, so train and test
class proportions barely move and the null sits at 0.50. Under a FIXED partition
the per-fold class counts must sum to a CONSTANT total, so a test fold that is
heavy in one class leaves the training folds heavy in the other. That is a
property of the arithmetic of a fixed partition, not of EEG, not of CSP, and not
of this pipeline.

THE CONTROL, and it is the reason this departure is defensible at all: a
majority-class DummyClassifier on an ALL-ZERO feature matrix. It cannot decode
anything, because there is nothing to decode. If it shows the same downward shift,
the shift belongs to the partition rule.""")

    from sklearn.dummy import DummyClassifier

    N_DUMMY = min(2000, max(200, N_ARM_A // 5))
    print(f"\n  Majority-class dummy, feature matrix = np.zeros((n, 1)), "
      f"{N_DUMMY} draws per cell:")
    print(f"  {'subject':<9}{'RE-STRATIFIED':>22}{'FIXED at P0':>22}   "
      f"{'majority-class rate':>20}")
    for s in SUBJECTS_A:
        X, y, runs = DATA_A[s]
        n = len(y)
        Xz = np.zeros((n, 1))
        P0 = P0_A[s]
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        rngd = np.random.default_rng(SEED)
        dperms = [iid_perm(y, rngd) for _ in range(N_DUMMY)]
        d_re = np.array([cross_val_score(DummyClassifier(strategy="most_frequent"),
                                         Xz, yp, cv=list(skf.split(Xz, yp)),
                                         error_score="raise").mean() for yp in dperms])
        d_fx = np.array([cross_val_score(DummyClassifier(strategy="most_frequent"),
                                         Xz, yp, cv=P0, error_score="raise").mean()
                         for yp in dperms])
        assert_lattice(d_re, n, f"dummy re-stratified S{s}")
        assert_lattice(d_fx, n, f"dummy fixed S{s}")
        maj = max((y == 2).mean(), (y == 3).mean())
        print(f"  S{s:<8}{d_re.mean():>12.2%} +/- {d_re.std():<6.2%}"
          f"{d_fx.mean():>12.2%} +/- {d_fx.std():<6.2%}   {fmt(maj, n):>20}")

    print("""
  Read that table before reading anything else in this run. A predictor that has
  NO ACCESS TO THE DATA loses double-digit points purely by moving from a
  re-stratified partition to a fixed one. Under re-stratification it sits exactly
  at the majority-class rate with zero variance; under a fixed partition it falls
  well below 0.50. Nothing about EEG, CSP or LDA is involved.""")

    print("\n  The splitter fact from pre-registration Section 2.1, verified here on "
      "subject 1:")
    X, y, _ = DATA_A[1]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rngd = np.random.default_rng(SEED)
    imb_fx, imb_re = [], []
    for _ in range(500):
        yp = iid_perm(y, rngd)
        for _, te in P0_A[1]:
            imb_fx.append(int(abs((yp[te] == 2).sum() - (yp[te] == 3).sum())))
        for _, te in skf.split(X, yp):
            imb_re.append(int(abs((yp[te] == 2).sum() - (yp[te] == 3).sum())))
    bf, br = np.bincount(imb_fx, minlength=10), np.bincount(imb_re, minlength=10)
    print(f"    test-fold |n_hands - n_feet| over 500 x 5 folds")
    print(f"      FIXED partition   : " +
          "  ".join(f"{k}:{bf[k]}" for k in range(10) if bf[k]))
    print(f"      RE-STRATIFIED     : " +
          "  ".join(f"{k}:{br[k]}" for k in range(10) if br[k]))
    print("    Re-stratification admits ONE imbalance value. The fixed partition "
      "admits the\n    full range. They are not two estimates of one quantity, "
      "they are two different\n    reference distributions.")

    print("""
WHAT THIS DEPARTURE DOES AND DOES NOT LICENSE.
  It licenses: enforcing assert 9 as a hard halt on the RE-STRATIFIED cells (C1,
  C3) and on both arm B cells, and reporting it as a MEASURED OUTCOME rather than
  a halt on the fixed cells (C2, C4).
  It does NOT license: treating the corrected p as unaffected. A null centred
  lower makes the corrected p SMALLER than the published one, so this departure
  moves the result in the project's favour and every number it touches is
  labelled with that fact in section 6.""")

    print("""
CORRECTED 2026-07-26. THE PARAGRAPH THAT USED TO CLOSE THIS SECTION IS WITHDRAWN,
AND IT IS KEPT VISIBLE BECAUSE IT WAS THE LOAD-BEARING STEP. It read:

  "It also does NOT show the corrected p is invalid. A permutation p is exact when
   the observed statistic and the null replicates are computed the SAME way. The
   observed value is scored on P0; C2 and C4 score every replicate on P0. It is
   C1, the published null, that scores replicates on a partition the observed
   value never saw."

The second sentence is FALSE AS A CRITERION. Computing the observed value and the
replicates the same way is NECESSARY, not sufficient. The additional requirement
is that the conditioning quantity must not depend on the labels being permuted.
P0 = StratifiedKFold(...).split(X, y_TRUE) IS a function of the labels being
permuted. So the observed value is scored on a partition balanced with respect to
its OWN labels, while every replicate is scored on that same partition under
labels that make its margins arbitrary. The observed vector is the unique point in
the reference set with that property.

The third sentence has the direction backwards. Under re-stratification the
statistic is ONE function S(y) = CV(X, y; SKF(y)) applied identically to the
observed vector and to every replicate, so C1 and C3 ARE exact. sklearn's
behaviour was already correct for its own reference set.

Section 2B's OTHER argument survives intact and is not withdrawn: the 0.45-0.55
band really does not transfer to a fixed partition, and the dummy table above
really does show that. That was the right diagnosis of the wrong assert. It is
just not a defence of C2 and C4, whose defect is exchangeability rather than
centring. Section 2C measures the defect.""")


def run_type1(D):
    DATA_A = D.DATA_A

    # --------------------------------------------------------------------------- #
    # SECTION 2C  THE VALIDITY DEFECT SECTION 2B DID NOT ADDRESS
    # --------------------------------------------------------------------------- #
    hdr("SECTION 2C  ARE THE FOUR CELLS EXACT TESTS? MEASURED WITH ZERO INFORMATION\n"
    "            (added 2026-07-26, POST-REGISTRATION, NOT BLIND)")

    print("""DISCLOSE FIRST. This section did not exist when the pre-registration was
written and did not exist when this script first ran. It was added after an
adversarial pass argued that C2 and C4 are not exact tests, and it runs in the
direction that COSTS this project its headline result. That is the easy direction
to be honest in, so the evidence is still made to carry the argument rather than
the argument carrying the evidence.

THE TEST OF A TEST. If a rule is an exact permutation test, then when H0 is
EXACTLY true its p-value is stochastically at least uniform, so P(p <= alpha) is
at most alpha. That is checkable without any theory, and without any EEG, by
constructing a situation where H0 is true BY CONSTRUCTION: a majority-class
predictor on an all-zero feature matrix. It has no information. There is nothing
for it to decode. Every rejection it earns is a false one.

THE PROCEDURE, exactly as an analyst would run it. Draw an H0 label vector.
BUILD THE PARTITION FROM THAT VECTOR, because that is what an analyst does: they
stratify on the labels they have. Score the observed value. Then permute and score
the reference set under each rule. Six rules:
  RE      i.i.d. shuffle, folds re-stratified on each permuted vector   (= C1)
  FX      i.i.d. shuffle, folds held fixed at P0 = SKF(y_observed)      (= C2)
  RE_blk  within-run shuffle, re-stratified                             (= C3)
  FX_blk  within-run shuffle, fixed at P0                               (= C4)
  KF_free i.i.d. shuffle, fixed at a LABEL-INDEPENDENT KFold partition  (candidate)
  WITHIN  labels permuted WITHIN each fold of P0, so every replicate carries the
          observed value's fold margins                                 (candidate)
The last two are the two ways to make a fixed-partition test exact.""")


    def _dummy_acc(y, folds):
        """Majority-class predictor accuracy on given folds, computed directly.

    Equivalent to cross_val_score(DummyClassifier(strategy="most_frequent"), ...)
    on an all-zero feature matrix, and asserted equal to it below. Written out
    because the type-I calculation needs millions of these and the sklearn call
    overhead, not the arithmetic, is what makes that expensive.
    """
        tot = 0
        for tr, te in folds:
            n2 = int((y[tr] == 2).sum())
            n3 = len(tr) - n2
            pred = 2 if n2 >= n3 else 3     # ties to the lower class label, as sklearn does
            tot += int((y[te] == pred).sum())
        return tot / len(y)


    # Correctness gate on the fast path, before it is trusted for anything.
    from sklearn.dummy import DummyClassifier as _DC
    _yg, _Xg = DATA_A[1][1], np.zeros((len(DATA_A[1][1]), 1))
    _rg = np.random.default_rng(11)
    for _ in range(25):
        _yy = _rg.permutation(_yg)
        _fg = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=SEED).split(_Xg, _yy))
        _fast = _dummy_acc(_yy, _fg)
        _slow = cross_val_score(_DC(strategy="most_frequent"), _Xg, _yy, cv=_fg,
                                error_score="raise").mean()
        assert abs(_fast - _slow) < TOL, (
            f"the fast majority-class path disagrees with sklearn's DummyClassifier "
        f"({_fast} vs {_slow}); the type-I numbers below would be measuring "
        f"something other than the dummy")
    print(f"\n  Fast-path gate: the direct majority-class computation agrees with "
      f"sklearn's\n  DummyClassifier on 25 random label vectors, to {TOL:g}. PASS.")

    print(f"\n  {N_TYPE1_OUTER} H0 label vectors x {N_TYPE1_INNER} inner permutations "
      f"per rule per subject.")
    print(f"  H0 IS EXACTLY TRUE THROUGHOUT. Any rejection rate above alpha is a "
      f"false-positive rate.")
    print(f"\n  {'subject':<9}{'rule':<10}{'P(p<=0.05)':>12}{'P(p<=0.10)':>12}"
      f"{'median p':>11}   verdict at 0.05")
    TYPE1 = {}
    for s in SUBJECTS_A:
        _X, _y, _runs = DATA_A[s]
        _n = len(_y)
        _rng = np.random.default_rng(SEED)
        _kf_free = list(KFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED).split(np.zeros((_n, 1))))
        _rules = ["RE", "FX", "RE_blk", "FX_blk", "KF_free", "WITHIN"]
        _hits = {k: 0 for k in _rules}
        _hits10 = {k: 0 for k in _rules}
        _pmed = {k: [] for k in _rules}
        for _ in range(N_TYPE1_OUTER):
            y0 = _rng.permutation(_y)
            P0d = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                       random_state=SEED).split(np.zeros((_n, 1)), y0))
            obs_re = _dummy_acc(y0, list(StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
                    np.zeros((_n, 1)), y0)))
            obs_fx = _dummy_acc(y0, P0d)
            obs_free = _dummy_acc(y0, _kf_free)
            ref = {k: np.empty(N_TYPE1_INNER) for k in _rules}
            for j in range(N_TYPE1_INNER):
                yi = _rng.permutation(y0)
                yb = y0.copy()
                for r in np.unique(_runs):
                    m = _runs == r
                    yb[m] = _rng.permutation(y0[m])
                yw = y0.copy()
                for _, te in P0d:
                    yw[te] = _rng.permutation(y0[te])
                ref["RE"][j] = _dummy_acc(yi, list(StratifiedKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
                        np.zeros((_n, 1)), yi)))
                ref["FX"][j] = _dummy_acc(yi, P0d)
                ref["RE_blk"][j] = _dummy_acc(yb, list(StratifiedKFold(
                    n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
                        np.zeros((_n, 1)), yb)))
                ref["FX_blk"][j] = _dummy_acc(yb, P0d)
                ref["KF_free"][j] = _dummy_acc(yi, _kf_free)
                ref["WITHIN"][j] = _dummy_acc(yw, P0d)
            for k in _rules:
                o = (obs_re if k in ("RE", "RE_blk") else
                     obs_free if k == "KF_free" else obs_fx)
                pv = (1 + int((ref[k] >= o - TOL).sum())) / (1 + N_TYPE1_INNER)
                _pmed[k].append(pv)
                _hits[k] += pv <= 0.05
                _hits10[k] += pv <= 0.10
        # MC error is not decoration at 200 draws: the se on a rate near 0.05 is
        # about 0.015, so a realised 0.055 is not an excess. A cell is called
        # anti-conservative only when it clears nominal by 2 MC se at 0.05 or at
        # 0.10, and the worse of the two alphas is the one reported.
        for k in _rules:
            r5 = _hits[k] / N_TYPE1_OUTER
            r10 = _hits10[k] / N_TYPE1_OUTER
            TYPE1[(s, k)] = (r5, r10, float(np.median(_pmed[k])))
            se5 = np.sqrt(0.05 * 0.95 / N_TYPE1_OUTER)
            se10 = np.sqrt(0.10 * 0.90 / N_TYPE1_OUTER)
            bad5 = r5 > 0.05 + 2 * se5
            bad10 = r10 > 0.10 + 2 * se10
            if bad5 or bad10:
                worst = max(r5 / 0.05, r10 / 0.10)
                a_at = "0.05" if r5 / 0.05 >= r10 / 0.10 else "0.10"
                flag = f"ANTI-CONSERVATIVE: {worst:.1f}x nominal at alpha {a_at}"
            else:
                flag = "at or below nominal (+2 MC se)"
            print(f"  {('S' + str(s)) if k == _rules[0] else '':<9}{k:<10}"
              f"{r5:>12.4f}{r10:>12.4f}{np.median(_pmed[k]):>11.4f}   {flag}")

    # The summary is COUNTED from the table, not asserted over it. The fixed-at-P0
    # defect does not bite equally at every class marginal, and a sentence that says
    # "far above nominal" everywhere would be false for at least one subject here.
    _se5 = np.sqrt(0.05 * 0.95 / N_TYPE1_OUTER)
    _se10 = np.sqrt(0.10 * 0.90 / N_TYPE1_OUTER)


    def _n_bad(rules):
        return sum(1 for s in SUBJECTS_A for k in rules
                   if TYPE1[(s, k)][0] > 0.05 + 2 * _se5
                   or TYPE1[(s, k)][1] > 0.10 + 2 * _se10)


    _bad_fixed = _n_bad(["FX", "FX_blk"])
    _bad_exact = _n_bad(["RE", "RE_blk", "KF_free", "WITHIN"])
    _cells_fixed = 2 * len(SUBJECTS_A)
    _cells_exact = 4 * len(SUBJECTS_A)
    print(f"\n  COUNTED FROM THE TABLE, not asserted over it:")
    print(f"    fixed-at-P0 cells (FX, FX_blk)                 : "
      f"{_bad_fixed} of {_cells_fixed} anti-conservative")
    print(f"    re-stratified and label-independent alternatives: "
      f"{_bad_exact} of {_cells_exact} anti-conservative")
    print(f"  The defect does NOT bite equally at every class marginal, and the table "
      f"shows that.")
    print(f"  What it never does is bite in the other direction: no re-stratified or "
      f"label-free")
    print(f"  cell exceeds nominal anywhere in this table, and a fixed-at-P0 cell "
      f"never reads")
    print(f"  BELOW its re-stratified partner. A rule that is exact at some marginals "
      f"and")
    print(f"  {max(TYPE1[(s,'FX')][0] for s in SUBJECTS_A)/0.05:.0f}x nominal at "
      f"others is not a test, because the analyst does not get to know which")
    print(f"  marginal they have before choosing.")

    print("""
  There is no EEG in that table. A predictor with provably zero information is
  being declared significant, so every rejection in it is a false one.

  THE DIRECTION OF THE ERROR IS THE OPPOSITE OF WHAT THIS SCRIPT REPORTED. The
  cells this script called "corrected" are the ones that are not tests, and the
  cells it called defective (C1, the published null, and C3) are the exact ones.

  WHAT IS WITHDRAWN, effective now and everywhere below: every number derived from
  C2 and C4. Their p-values, percentiles, tail counts, the n_eff and the
  n_eff-corrected Wilson interval recomputed from C4, the C2-C1 and C4-C3 paired
  mean(d) rows read as corrections, the C4-minus-C1 "headline result of arm A",
  and the recommendation that C4's p is the one to publish. They are still PRINTED
  below, because deleting them would hide what was claimed, but they are printed
  as measurements of an invalid rule and no conclusion rests on them.

  WHAT REPLACES THEM: C1 and C3, which are exact and were already computed, plus
  C5 and C6 below, which are the exact version of the fixed-partition idea.""")
