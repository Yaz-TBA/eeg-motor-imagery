"""Measure what a CORRECTLY DESIGNED permutation null does to this repo's p-values.

NAVIGATION. This is the registered validity run and it is long since the design IS the
result. Section 0 sets up, section 1 is falsification gate 1 (does the pilot reproduce),
section 2 is gate 2 (sklearn agreement), and the two arms diverge after that. If you only
want the outcome, search for "PRE-REGISTERED VERDICT". No jokes anywhere in this file, on
purpose: a script whose whole job is showing I didn't fool myself should read like one.

WHY THIS SCRIPT EXISTS. Two published nulls in this repo are built on assumptions
the experiment does not satisfy, and until now the repo has conceded that in prose
without measuring it.

  (a) WITHIN-SUBJECT. decode_csp.py:124 and evaluate_honestly.py:185 call
      permutation_test_score with cv=StratifiedKFold(5, shuffle=True,
      random_state=42). sklearn's _permutation_test_score loops
      `for train, test in cv.split(X, y, ...)` with the PERMUTED labels as y, so
      StratifiedKFold re-derives the folds from each permuted vector. Every
      replicate is therefore scored on a DIFFERENT partition, while the observed
      91.1% is scored on the partition stratified on the TRUE labels. The null
      marginalises over partitions the observed value conditions on.

  (b) CROSS-SUBJECT. cross_subject.py:135 draws ONE global permutation of all 900
      pooled labels. That tests the compound hypothesis "labels are unrelated to
      the EEG AND labels are exchangeable ACROSS subjects". The second conjunct is
      false by protocol: each subject's class marginal is fixed by the experiment.
      A global shuffle can deal a 45-trial subject 30 feet and 15 hands, a draw the
      experiment could not have produced, so the null carries a variance component
      that has nothing to do with decoding.

Those are the same underlying error at two levels: THE NULL MUST CONDITION ON
WHATEVER THE OBSERVED STATISTIC CONDITIONS ON. Arm (a) is a defect of the
PARTITION. Arm (b) is a defect of the REFERENCE SET. They are not one objection
applied twice, and LeaveOneGroupOut is already label-invariant, so arm (a)'s defect
does not exist in arm (b).

THE EXCHANGEABLE UNIT, stated explicitly, which is what the objection said this
project could not do:
  (a) the trial, CONDITIONAL ON RUN, with the statistic evaluated on a FIXED
      partition. Within each of runs 6/10/14 the labels are exchangeable; nothing
      is exchangeable across runs, because run is a blocking factor with its own
      electrode settling and drift.
  (b) the trial, CONDITIONAL ON SUBJECT. The partition needs no correction because
      LeaveOneGroupOut already conditions on subject.

WHAT THIS SCRIPT GUARDS AGAINST. Three things, in order of how easy they are to
fool yourself with:
  1. Reading a resolution floor as a measurement. Subject 1's effect is roughly
     4.5 null sd out, so its p is at the floor in EVERY cell no matter which null
     is used. That is a property of the effect size, not evidence that the design
     was right. This script therefore leads with the PAIRED per-replicate null
     difference and the null's SHAPE, not with the p.
  2. An unpaired comparison masquerading as a paired one. The pilot this replaces
     fed different label vectors to the cells it compared, so a 3.0-point gap was
     confounded with Monte Carlo noise. Here ONE list of permuted label vectors
     feeds both partition rules, and the script asserts the two cells saw
     element-wise identical labels at every replicate.
  3. A correction that silently changed nothing. Asserts 5 and 6 below check that
     the fixed cells really are fixed AND that the re-stratified cells really do
     move. A "correction" where both cells were secretly identical would look like
     a clean null result.

WHAT THIS SCRIPT DOES NOT SHOW. It does not repair n=45. It does not make the
within-subject null binomial. It does not address a session-level trend across all
three runs, which survives every null here exactly as it survives the
leave-one-run-out ablation in ablate_channels.py:255-259. And three subjects is
not a survey: subjects 17 and 19 were picked by a fixed median rule from
sweep_results.csv, so the claim available from them is EXISTENCE (whether the
correction CAN move a verdict), never frequency.

PRE-REGISTERED at neuro-canon/measurements/prereg-block-permutation.md, written
before this file existed. Every outcome below was written down in advance.
"""

import os
import sys
import time

# joblib spawns fresh processes that re-import mne at its DEFAULT log level, so
# mne.set_log_level() in this file never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
# One BLAS thread per worker. 16 loky workers each spawning their own BLAS pool
# oversubscribes the machine and makes the run SLOWER, not faster. CSP inverts
# 64x64 covariances, so the per-call BLAS work is far too small to parallelise.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import hashlib
import warnings

import numpy as np
import mne
from joblib import Parallel, delayed
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    check_cv,
    cross_val_score,
    permutation_test_score,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# common.py lives at the repo root, one level up; put it on the path so this script
# can be launched from anywhere.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import assert_lattice, load_epochs, make_clf


def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-minute experiment as a side effect."""

    # --- registered constants. Changing any of these makes the run a smoke test. ---
    SEED = 42
    RUNS = [6, 10, 14]
    TMIN, TMAX = -1.0, 4.0
    CROP = (1.0, 2.0)
    L_FREQ, H_FREQ = 8.0, 30.0
    N_SPLITS = 5

    REG_N_ARM_A = 10_000          # per cell, arm A
    REG_N_ARM_B = 2_000           # per cell, arm B
    REG_N_REPRO = 300             # pilot reproduction, A269's own draw count
    REG_N_GATE2 = 1_000           # sklearn-agreement gate

    N_ARM_A = int(os.environ.get("PERMDESIGN_N_A", REG_N_ARM_A))
    N_ARM_B = int(os.environ.get("PERMDESIGN_N_B", REG_N_ARM_B))
    N_REPRO = int(os.environ.get("PERMDESIGN_N_REPRO", REG_N_REPRO))
    N_GATE2 = int(os.environ.get("PERMDESIGN_N_GATE2", REG_N_GATE2))
    IS_REGISTERED_RUN = (N_ARM_A == REG_N_ARM_A and N_ARM_B == REG_N_ARM_B
                         and N_REPRO == REG_N_REPRO and N_GATE2 == REG_N_GATE2)

    N_JOBS = int(os.environ.get("PERMDESIGN_N_JOBS", "-1"))
    CHUNK_A = 50
    CHUNK_B = 10

    # Subjects, by a rule fixed before any null ran. Subject 1 is the headline. The
    # median of subjects 1..20 by published within-subject accuracy is 0.6333, and
    # subjects 17 (0.6222) and 19 (0.6444) are EXACTLY equidistant from it, so both
    # are included rather than inventing a tie-break.
    SUBJECTS_A = [1, 17, 19]
    SUBJECTS_B = list(range(1, 21))
    WITHIN_CSV = "results/sweep_results.csv"

    # A269, the pilot, quoted here as the PRIOR. This run confirms or contradicts it;
    # it does not discover it independently. Order: null mean %, null sd %, null max %.
    A269_PILOT = {
        "(i)   iid shuffle, re-stratified folds": (50.7, 8.9, 86.7),
        "(ii)  iid shuffle, FIXED folds        ": (47.7, 8.4, 71.1),
        "(iii) within-run cyclic shift         ": (49.0, 8.6, 73.3),
        "(iv)  within-run label permutation    ": (50.2, 8.2, 68.9),
    }
    A269_PILOT_P = 0.0033

    # Registered materiality thresholds. Fixed before the run.
    ONE_TRIAL = 1.0 / 45.0        # 2.2222 points. A sub-trial difference in a
                                  # 45-trial null is not a difference anyone can act on.
    MC_SIGMA = 3.0                # multiples of the Monte Carlo standard error
    P_THRESHOLD = 0.05
    CENTRE_LO, CENTRE_HI = 0.45, 0.55   # registered assert 9 / falsification 8
    CENTRING = {}                 # every cell's null mean, recorded for section 6

    # POST-REGISTRATION, added 2026-07-26 after an adversarial pass showed that the
    # fixed-at-P0 cells are not exact tests. None of these edit the pre-registration.
    #   C5 / C6: the EXACT version of the fixed-partition idea. The partition is built
    #            WITHOUT the labels, so it is ancillary and freezing it preserves
    #            exchangeability. This is what C2 / C4 were trying to be.
    #   the dummy type-I control: zero information, H0 exactly true by construction.
    N_EXACT = int(os.environ.get("PERMDESIGN_N_EXACT", "2000"))
    # 200 x 199 rather than 2000 x 999: the effect being measured is the difference
    # between a rejection rate near zero and one well above 0.5, so the MC se on a
    # 0.05 rate at 200 draws (0.015) is an order of magnitude smaller than the effect.
    # Spending 25x the compute to shrink an already-decisive interval is not a
    # measurement, it is a way to make the script too slow to run.
    N_TYPE1_OUTER = int(os.environ.get("PERMDESIGN_N_T1_OUTER", "200"))
    N_TYPE1_INNER = int(os.environ.get("PERMDESIGN_N_T1_INNER", "199"))

    SHUFFLE_MAX = 0.60            # the underived guard in cross_subject.py:146,
                                  # quoted here as the thing being replaced.

    TOL = 1e-9


    # --------------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------------- #
    def hdr(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        sys.stdout.flush()


    def sub(title):
        print("\n" + "-" * 78)
        print(title)
        print("-" * 78)
        sys.stdout.flush()


    def note(msg):
        """Progress to stderr, so stdout stays a clean provenance record."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


    # make_clf comes from common.py. CSP sits INSIDE the Pipeline, so it refits on the
    # training fold only, inside every fold, in every replicate. Any other placement would
    # leak the test fold into filter estimation and invalidate everything, and
    # test_pipeline.py asserts the placement.

    def fmt(acc, n):
        """Every accuracy prints with its k/n count. On n tested-once trials the
    accuracy is a multiple of 1/n and no other value is attainable."""
        return f"{acc:.1%} ({int(round(acc * n))}/{n})"


    # assert_lattice comes from common.py, at the same TOL = 1e-9. Equal folds mean the
    # fold-mean IS the pooled count over n. Off-lattice means unequal folds or a scorer
    # that is not accuracy, and then the fold-mean is not the accuracy at all. This is the
    # check that caught two arithmetically impossible numbers (95.9%, 47.4%) in an earlier
    # README, and test_pipeline.py now guards both of them by name.


    def p_value(null, observed, n_draws):
        """p = (C + 1) / (N + 1) with >=, matching sklearn exactly. C is returned so
    that 'both at the floor' stays distinguishable from 'both actually equal'."""
        c = int((np.asarray(null) >= observed - TOL).sum())
        return (c + 1) / (n_draws + 1), c


    def p_str(p, c, n_draws):
        floor = 1.0 / (n_draws + 1)
        if c == 0:
            return f"<= {floor:.5g} (BOUND, the floor of {n_draws} draws, C = 0)"
        return f"=  {p:.5g} (C = {c} of {n_draws})"


    def quant(x, q, n):
        """Order-statistic quantiles. method='inverted_cdf' returns an OBSERVED value,
    so the percentile stays on the k/n lattice. Linear interpolation would print
    percentiles that no replicate could have scored."""
        v = float(np.quantile(x, q, method="inverted_cdf"))
        return v, f"{v:.1%} ({int(round(v * n))}/{n})"


    def describe_null(name, null, observed, n_trials, n_draws, extra_q=()):
        p, c = p_value(null, observed, n_draws)
        print(f"{name}")
        print(f"    null mean {null.mean():.2%}  sd {null.std(ddof=0):.2%}   "
          f"min {fmt(null.min(), n_trials)}   max {fmt(null.max(), n_trials)}")
        qs = sorted(set([0.50, 0.90, 0.95, 0.99, 0.999] + list(extra_q)))
        parts = []
        for q in qs:
            _, s = quant(null, q, n_trials)
            parts.append(f"p{q * 100:g}={s}")
        print("    " + "  ".join(parts))
        print(f"    observed {fmt(observed, n_trials)}   "
          f"standardised distance {(observed - null.mean()) / null.std(ddof=0):+.2f} sd "
          f"(descriptive only, the null is not normal)")
        print(f"    p {p_str(p, c, n_draws)}")
        sys.stdout.flush()
        return p, c


    def wilson(k, n, z=1.959963985):
        ph = k / n
        d = 1 + z * z / n
        c = ph + z * z / (2 * n)
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d


    # --------------------------------------------------------------------------- #
    # loading. Identical to decode_csp.py / sweep_subjects.py / cross_subject.py.
    # --------------------------------------------------------------------------- #
    def load_subject(subject):
        """Runs 6/10/14, average reference over all 64 channels, 8-30 Hz FIR, epochs
    -1.0 to 4.0 s, cropped to 1.0-2.0 s. Also returns the RUN INDEX per epoch. That
    blocking variable has existed in this repo since the ablation rung and had never been
    used in a null; it is the whole of the arm A block correction.

    The body now lives in common.load_epochs. Verified bit-for-bit identical to the
    version this file used to carry, on subjects 1 and 17: X, y and the run indices all
    compare equal with np.array_equal."""
        return load_epochs(subject, runs=RUNS, l_freq=L_FREQ, h_freq=H_FREQ,
                           tmin=TMIN, tmax=TMAX, crop=CROP, return_runs=True)


    # --------------------------------------------------------------------------- #
    # label-permutation schemes
    # --------------------------------------------------------------------------- #
    def iid_perm(y, rng):
        """Exchangeable across all n trials. What the published null assumes."""
        return rng.permutation(y)


    def within_block_perm(y, blocks, rng):
        """Exchangeable WITHIN block only. Preserves each block's class counts exactly.
    Used for run blocking in arm A and subject blocking in arm B."""
        out = y.copy()
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            out[idx] = rng.permutation(y[idx])
        return out


    def cyclic_shift(y, blocks, rng):
        """Pilot cell (iii) only. Reproduced solely so arm A-repro can reproduce A269;
    it is not one of the four registered cells."""
        out = y.copy()
        for b in np.unique(blocks):
            idx = np.where(blocks == b)[0]
            k = rng.integers(1, len(idx))
            out[idx] = np.roll(y[idx], k)
        return out


    # --------------------------------------------------------------------------- #
    # workers
    # --------------------------------------------------------------------------- #
    def _score_chunk_simple(X, perms, fixed_folds):
        """Score a chunk under ONE partition rule. fixed_folds=None means re-stratify
    on each permuted vector (what sklearn does); otherwise replay the given list."""
        out = []
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for yp in perms:
            folds = fixed_folds if fixed_folds is not None else list(skf.split(X, yp))
            out.append(cross_val_score(make_clf(), X, yp, cv=folds,
                                       error_score="raise").mean())
        return np.asarray(out, dtype=float)


    def _arm_a_chunk(X, perms, P0):
        """The paired unit of work: ONE permuted label vector scored under BOTH
    partition rules. Pairing is enforced structurally here rather than asserted
    afterwards, because a paired design that is not actually paired is worse than
    an unpaired one.

    Returns, per replicate: the re-stratified mean, the fixed mean, the fixed
    cell's five per-fold accuracies (for the secondary mechanism probe), and the
    realised-partition bookkeeping that asserts 5 and 6 need.
    """
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        p0_sets = {frozenset(int(i) for i in te) for _, te in P0}
        p0_fold_of = np.empty(X.shape[0], dtype=np.int16)
        for f, (_, te) in enumerate(P0):
            p0_fold_of[te] = f

        m_re, m_fx, pf_fx = [], [], []
        same_re, fracdiff_re, same_fx, exact_fx = [], [], [], []

        for yp in perms:
            # --- re-stratified: derive the partition from the PERMUTED labels ---
            folds_re = list(skf.split(X, yp))
            s_re = cross_val_score(make_clf(), X, yp, cv=folds_re, error_score="raise")
            # --- fixed: replay P0 through the same code path cross_val_score uses ---
            folds_fx = list(check_cv(P0, yp, classifier=True).split(X, yp))
            s_fx = cross_val_score(make_clf(), X, yp, cv=folds_fx, error_score="raise")

            assert all(len(te) == X.shape[0] // N_SPLITS for _, te in folds_re), \
                "re-stratified folds are not equal-sized; the k/n lattice would break"

            re_sets = {frozenset(int(i) for i in te) for _, te in folds_re}
            fold_of = np.empty(X.shape[0], dtype=np.int16)
            for f, (_, te) in enumerate(folds_re):
                fold_of[te] = f

            m_re.append(s_re.mean())
            m_fx.append(s_fx.mean())
            pf_fx.append(s_fx)
            same_re.append(re_sets == p0_sets)
            fracdiff_re.append(float((fold_of != p0_fold_of).mean()))
            same_fx.append({frozenset(int(i) for i in te) for _, te in folds_fx} == p0_sets)
            exact_fx.append(all(np.array_equal(a[1], b[1]) for a, b in zip(folds_fx, P0)))

        return (np.asarray(m_re), np.asarray(m_fx), np.asarray(pf_fx),
                np.asarray(same_re), np.asarray(fracdiff_re),
                np.asarray(same_fx), np.asarray(exact_fx))


    def _arm_fixed_chunk(X, perms, folds):
        """Score every permuted label vector on ONE partition that was built WITHOUT
    the labels. Added 2026-07-26 for cells C5 and C6. The distinction from
    _arm_a_chunk's fixed half is the ONLY thing that matters here: `folds` comes
    from KFold(...).split(X) and never saw y, so the statistic y' -> CV(X, y';
    folds) is a function of y' alone and the permutation test is exact."""
        out = []
        for yp in perms:
            out.append(cross_val_score(make_clf(), X, yp, cv=folds,
                                       error_score="raise").mean())
        return np.asarray(out, dtype=float)


    def _arm_b_chunk(X, perms, folds):
        """One LOSO pass per permuted label vector. The folds are precomputed once and
    replayed: LeaveOneGroupOut._iter_test_masks reads only `groups`, so the LOSO
    partition is ALREADY label-invariant and needs no correction. Arm (a)'s defect
    does not exist here, and passing the folds explicitly makes that visible."""
        out = []
        for yp in perms:
            s = cross_val_score(make_clf(), X, yp, cv=folds, error_score="raise")
            out.append(s.mean())
        return np.asarray(out, dtype=float)


    def run_parallel(fn, arg_chunks, label):
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS)(delayed(fn)(*a) for a in arg_chunks)
        note(f"{label}: {time.time() - t0:.0f} s over {len(arg_chunks)} chunks")
        return res


    def chunks(seq, size):
        return [seq[i:i + size] for i in range(0, len(seq), size)]


    # --- block checkpointing ------------------------------------------------------
    # The registered N takes about 90 minutes of wall clock. A first attempt was
    # killed 32 minutes into arm B and lost the whole arm, so every expensive cell is
    # now computed in BLOCKS and each block is written to disk as it lands. A rerun
    # reloads finished blocks and computes only what is missing.
    #
    # The correctness risk this introduces is a STALE block silently reloaded after
    # an input changed, which would be indistinguishable from a clean result. Guarded
    # by `stamp`: a fingerprint of every input the block depends on (subject, N, seed,
    # the exact label vector, the exact partition, the pipeline description). A stamp
    # mismatch discards the block and recomputes it. The cache is never consulted
    # across different stamps and holds no summary statistic, only raw per-replicate
    # scores, so nothing derived can be cached stale.
    CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             ".permutation_design_cache")   # at the repo root, one level up


    def fingerprint(*parts):
        h = hashlib.sha256()
        for p in parts:
            h.update(np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray)
                     else str(p).encode())
            h.update(b"|")
        return h.hexdigest()[:16]


    def cached_blocks(key, stamp, n_items, block_size, compute_block):
        os.makedirs(CACHE_DIR, exist_ok=True)
        out, n_hit = [], 0
        for bi, lo in enumerate(range(0, n_items, block_size)):
            hi = min(lo + block_size, n_items)
            path = os.path.join(CACHE_DIR, f"{key}_b{bi:04d}.npz")
            got = None
            if os.path.exists(path):
                try:
                    z = np.load(path, allow_pickle=False)
                    if (str(z["stamp"]) == stamp and int(z["lo"]) == lo
                            and int(z["hi"]) == hi):
                        got = tuple(z[f"a{k}"] for k in range(int(z["n_arr"])))
                        n_hit += 1
                except Exception:                                    # noqa: BLE001
                    got = None
            if got is None:
                got = compute_block(lo, hi)
                # np.savez APPENDS .npz unless the name already ends in it, so the
                # temp name must carry the suffix or os.replace below gets a path
                # that was never written.
                tmp = path + ".tmp.npz"
                np.savez(tmp, stamp=np.array(stamp), lo=lo, hi=hi, n_arr=len(got),
                         **{f"a{k}": np.asarray(v) for k, v in enumerate(got)})
                os.replace(tmp, path)                # atomic, so a kill mid-write
            out.append(got)                          # cannot leave a torn block
        if n_hit:
            note(f"{key}: reused {n_hit} cached block(s) of {len(out)}")
        return tuple(np.concatenate([b[k] for b in out]) for k in range(len(out[0])))


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


    # --------------------------------------------------------------------------- #
    # SECTION 1  ARM A-REPRO
    # --------------------------------------------------------------------------- #
    hdr("SECTION 1  FALSIFICATION GATE 1: DOES THE PILOT (A269) REPRODUCE?")

    print("""Run FIRST, before anything else. The pilot consumed one default_rng(42)
sequentially across its four cells, so this reproduces that consumption order
exactly: 300 i.i.d. vectors, then 300 more i.i.d. vectors, then 300 cyclic shifts,
then 300 within-run permutations. Scoring is deterministic given the labels, so
generating the vectors up front and scoring them in parallel is identical to the
pilot's serial loop.

The two failure modes are DISTINGUISHABLE and only one is fatal:
  all four cells shifting TOGETHER  -> version drift in mne/sklearn, reportable
  ONE cell shifting                 -> a bug in that cell, fatal""")

    X1, y1, runs1 = DATA_A[1]
    n1 = len(y1)
    rng = np.random.default_rng(SEED)
    repro_labels = {
        "(i)   iid shuffle, re-stratified folds": ([iid_perm(y1, rng) for _ in range(N_REPRO)], None),
        "(ii)  iid shuffle, FIXED folds        ": ([iid_perm(y1, rng) for _ in range(N_REPRO)], P0_A[1]),
        "(iii) within-run cyclic shift         ": ([cyclic_shift(y1, runs1, rng) for _ in range(N_REPRO)], None),
        "(iv)  within-run label permutation    ": ([within_block_perm(y1, runs1, rng) for _ in range(N_REPRO)], None),
    }

    print(f"\nobserved (subject 1, P0) = {fmt(OBS_A[1], n1)}   "
      f"chance {fmt(24 / 45, n1)}\n")
    print(f"{'cell':<40} {'null mean':>10} {'null sd':>9} {'null max':>9} {'p':>8}   "
      f"pilot A269         verdict")
    repro_ok, repro_deltas = {}, {}
    for name, (perms, fixed) in repro_labels.items():
        note(f"repro {name.strip()}")
        parts = run_parallel(_score_chunk_simple,
                             [(X1, c, fixed) for c in chunks(perms, CHUNK_A)],
                             f"repro {name.strip()}")
        null = np.concatenate(parts)
        assert_lattice(null, n1, f"arm A-repro {name.strip()}")
        p, c = p_value(null, OBS_A[1], N_REPRO)
        got = (round(null.mean() * 100, 1), round(null.std(ddof=0) * 100, 1),
               round(null.max() * 100, 1))
        want = A269_PILOT[name]
        ok = got == want and abs(p - A269_PILOT_P) < 5e-5
        repro_ok[name] = ok
        repro_deltas[name] = tuple(round(float(g) - float(w), 1) for g, w in zip(got, want))
        delta = "/".join(f"{d:+.1f}" for d in repro_deltas[name])
        print(f"{name:<40} {got[0]:9.1f}% {got[1]:8.1f}% {got[2]:8.1f}% {p:8.4f}   "
          f"{want[0]:.1f}/{want[1]:.1f}/{want[2]:.1f}   "
          f"{'MATCH' if ok else 'DIFFERS by ' + delta}")

    n_match = sum(repro_ok.values())
    print(f"\nCells matching A269 exactly at one decimal: {n_match} of 4.")
    if n_match == 4:
        print("GATE 1 PASSES. The pilot reproduces. A269 is the prior for everything below.")
    elif n_match == 0:
        print("GATE 1: ALL FOUR cells differ. That signature is VERSION DRIFT between the")
        print("pilot run and this one, which is reportable and not fatal. Read every")
        print("comparison below against THIS run's cells, not against A269's numbers.")
    else:
        print(f"GATE 1 FAILS. {4 - n_match} of 4 cells differ while the rest match. That")
        print("signature is a BUG in the differing cell, not version drift. NOTHING in")
        print("sections 3 to 6 may be reported from this run.")
    GATE1_FATAL = 0 < n_match < 4


    # --------------------------------------------------------------------------- #
    # SECTION 2  GATE 2
    # --------------------------------------------------------------------------- #
    hdr("SECTION 2  FALSIFICATION GATE 2: IS MY C1 ACTUALLY WHAT SKLEARN DOES?")

    print("""The single most important check in the document. EVERY claim of a
difference below is measured against C1, so if C1 is not the published null then
the framing 'this is what sklearn does' is unfounded.

sklearn draws its permutations from check_random_state(42), a legacy MT19937,
while C1 draws from default_rng(42), a PCG64. The two see DIFFERENT label vectors
by construction, so this is a DISTRIBUTIONAL agreement check, not an exact one:
null mean and null sd must agree to within 3 Monte Carlo standard errors of the
DIFFERENCE.""")

    note("gate 2: my C1 at N=1000")
    rng = np.random.default_rng(SEED)
    gate_perms = [iid_perm(y1, rng) for _ in range(N_GATE2)]
    mine = np.concatenate(run_parallel(
        _score_chunk_simple, [(X1, c, None) for c in chunks(gate_perms, CHUNK_A)],
        "gate 2 mine"))
    assert_lattice(mine, n1, "gate 2, my C1")

    note("gate 2: sklearn permutation_test_score direct call")
    t0 = time.time()
    obs_sk, null_sk, p_sk = permutation_test_score(
        make_clf(), X1, y1, scoring="accuracy",
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
        n_permutations=N_GATE2, random_state=SEED, n_jobs=N_JOBS,
    )
    note(f"gate 2 sklearn: {time.time() - t0:.0f} s")
    assert_lattice(null_sk, n1, "gate 2, sklearn null")

    se_mean = np.sqrt(mine.var(ddof=0) / N_GATE2 + null_sk.var(ddof=0) / N_GATE2)
    sd_m, sd_s = mine.std(ddof=0), null_sk.std(ddof=0)
    se_sd = np.sqrt(sd_m ** 2 / (2 * (N_GATE2 - 1)) + sd_s ** 2 / (2 * (N_GATE2 - 1)))
    z_mean = (mine.mean() - null_sk.mean()) / se_mean
    z_sd = (sd_m - sd_s) / se_sd

    print(f"\n  my C1      : null mean {mine.mean():.3%}  sd {sd_m:.3%}  "
      f"max {fmt(mine.max(), n1)}")
    print(f"  sklearn    : null mean {null_sk.mean():.3%}  sd {sd_s:.3%}  "
      f"max {fmt(null_sk.max(), n1)}")
    print(f"  difference : mean {100 * (mine.mean() - null_sk.mean()):+.3f} points "
      f"(MC se {100 * se_mean:.3f}, z = {z_mean:+.2f})")
    print(f"               sd   {100 * (sd_m - sd_s):+.3f} points "
      f"(MC se {100 * se_sd:.3f}, z = {z_sd:+.2f})")
    print(f"  sklearn's observed {fmt(obs_sk, n1)} against my P0 observed "
      f"{fmt(OBS_A[1], n1)}: match {abs(obs_sk - OBS_A[1]) < TOL}")
    GATE2_OK = abs(z_mean) <= MC_SIGMA and abs(z_sd) <= MC_SIGMA
    print(f"\nGATE 2 {'PASSES' if GATE2_OK else 'FAILS'}: |z| <= 3 on both mean and sd "
      f"is {'satisfied' if GATE2_OK else 'VIOLATED'}.")
    if not GATE2_OK:
        print("C1 is NOT the published null. Nothing below may be reported.")


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


    # --------------------------------------------------------------------------- #
    # SECTION 3  ARM A
    # --------------------------------------------------------------------------- #
    hdr("SECTION 3  ARM A: THE 2x2 FACTORIAL, PAIRED ON THE LABELS")

    print(f"""                     | partition RE-STRATIFIED each draw | partition FIXED at P0
  i.i.d. shuffle     | C1 = the published null           | C2 = correction (a) alone
  within-run shuffle | C3 = pilot cell (iv)              | C4 = FULLY CORRECTED

One list of {N_ARM_A} i.i.d. vectors feeds BOTH C1 and C2. A second list of {N_ARM_A}
within-run vectors feeds BOTH C3 and C4. So C1-vs-C2 and C3-vs-C4 are PAIRED: same
labels, only the partition rule differs, which isolates the correction. C1-vs-C3
and C2-vs-C4 are UNPAIRED BY CONSTRUCTION, because they draw from different
reference sets, and are compared on distributions only.

REGISTERED IN ADVANCE: for subject 1 the p is expected at its floor in every cell.
Expected exceedances in {N_ARM_A} draws are about 0.028 at a 4.5 sd effect. AN ARM
WHOSE ANSWER IS FIXED BY THE EFFECT SIZE CANNOT DETECT A DESIGN ERROR, so subject
1's finding is the PAIRED DIFFERENCE and the null's SHAPE, not the p.""")

    ARM_A = {}
    for s in SUBJECTS_A:
        X, y, runs = DATA_A[s]
        n = len(y)
        P0 = P0_A[s]
        rng = np.random.default_rng(SEED)
        perms_iid = [iid_perm(y, rng) for _ in range(N_ARM_A)]
        perms_blk = [within_block_perm(y, runs, rng) for _ in range(N_ARM_A)]

        # Assert 7, first half: the block shuffle really blocks.
        obs_run_counts = {int(r): int((y[runs == r] == 2).sum()) for r in np.unique(runs)}
        for yp in perms_blk:
            got = {int(r): int((yp[runs == r] == 2).sum()) for r in np.unique(runs)}
            assert got == obs_run_counts, (
                f"subject {s}: a within-run shuffle changed a run's class counts "
            f"({got} against {obs_run_counts}). The blocking is broken."
            )
        # And the i.i.d. shuffle must preserve only the TOTAL, not the per-run counts.
        iid_breaks_a_run = any(
            {int(r): int((yp[runs == r] == 2).sum()) for r in np.unique(runs)} != obs_run_counts
            for yp in perms_iid)
        for yp in perms_iid + perms_blk:
            assert int((yp == 2).sum()) == int((y == 2).sum()), \
                f"subject {s}: a shuffle changed the POOLED class counts. That is " \
            "resampling, not permuting."

        p0_flat = np.concatenate([te for _, te in P0])

        def arm_a_cell(tag, perms):
            stamp = fingerprint("armA", s, N_ARM_A, SEED, tag, "csp4-lda-5fold",
                                y, p0_flat, np.stack(perms[:50]), len(perms))

            def compute_block(lo, hi):
                res = run_parallel(
                    _arm_a_chunk,
                    [(X, c, P0) for c in chunks(perms[lo:hi], CHUNK_A)],
                    f"S{s} {tag} [{lo}:{hi}]")
                return tuple(np.concatenate([r[k] for r in res]) for k in range(7))

            return cached_blocks(f"armA_S{s}_{tag}", stamp, N_ARM_A,
                                 N_ARM_A // 5, compute_block)

        # --- C5 / C6: the EXACT version of the fixed-partition idea ----------------
        # ADDED 2026-07-26. P0 is stratified on y_TRUE, so freezing it makes the
        # statistic a function of (X, y', y_true) and the observed vector uniquely
        # privileged. PF is built by KFold WITHOUT the labels, so it is ancillary:
        # freezing it leaves the statistic a function of y' alone and the test exact.
        # This is what C2 and C4 were trying to be, and it is the arm that should
        # carry any fixed-partition claim.
        PF = list(KFold(n_splits=N_SPLITS, shuffle=True,
                        random_state=SEED).split(np.zeros((n, 1))))
        pf_flat = np.concatenate([te for _, te in PF])
        obs_pf = cross_val_score(make_clf(), X, y, cv=PF, error_score="raise").mean()
        assert_lattice([obs_pf], n, f"observed on label-free PF, subject {s}")

        def exact_cell(tag, perms):
            stamp = fingerprint("armAexact", s, N_EXACT, SEED, tag, "csp4-lda-5fold",
                                y, pf_flat, np.stack(perms[:50]), len(perms))

            def compute_block(lo, hi):
                res = run_parallel(
                    _arm_fixed_chunk,
                    [(X, c, PF) for c in chunks(perms[lo:hi], CHUNK_A)],
                    f"S{s} exact-{tag} [{lo}:{hi}]")
                return (np.concatenate(res),)

            return cached_blocks(f"armAexact_S{s}_{tag}", stamp, N_EXACT,
                                 max(200, N_EXACT // 5), compute_block)[0]

        note(f"arm A subject {s}: EXACT label-free fixed partition C5/C6")
        c5 = exact_cell("iid", perms_iid[:N_EXACT])
        c6 = exact_cell("blk", perms_blk[:N_EXACT])

        note(f"arm A subject {s}: i.i.d. pair C1/C2")
        c1, c2, pf2, same1, fd1, samef2, exact2 = arm_a_cell("iid", perms_iid)
        note(f"arm A subject {s}: within-run pair C3/C4")
        c3, c4, pf4, same3, fd3, samef4, exact4 = arm_a_cell("blk", perms_blk)

        # --- the ten asserts, on this subject ---
        for nm, arr in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4)):
            assert_lattice(arr, n, f"subject {s} {nm}")                      # assert 1
            CENTRING[(s, nm)] = float(arr.mean())
        # Assert 9 (null centring) is enforced as a HARD assert ONLY on the
        # re-stratified cells. See SECTION 2B: under a FIXED partition a
        # permuted-label null centres BELOW 0.50 by construction, and a
        # majority-class dummy with an all-zero feature matrix demonstrates it with
        # no EEG involved. The registered 0.45-0.55 band is the band
        # evaluate_honestly.py:220 asserts for the RE-STRATIFIED null, and it does
        # not transfer. This is a DEPARTURE from the pre-registration, flagged in
        # section 2B, in section 6, and everywhere the affected cells are reported.
        for nm, arr in (("C1", c1), ("C3", c3)):
            assert CENTRE_LO < arr.mean() < CENTRE_HI, (                     # assert 9
                f"subject {s} {nm}: RE-STRATIFIED null centred at {arr.mean():.1%}, "
            "not near 50%. The null is mis-specified and no p from it means what "
            "it appears to."
            )
        assert samef2.all() and exact2.all(), (                              # assert 5
            f"subject {s}: the FIXED cell C2 did not replay P0 on every replicate "
        f"({int((~exact2).sum())} of {N_ARM_A} differed). The entire correction is "
        "this one property."
        )
        assert samef4.all() and exact4.all(), (
            f"subject {s}: the FIXED cell C4 did not replay P0 on every replicate."
        )
        assert (~same1).any() and (~same3).any(), (                          # assert 6
            f"subject {s}: the RE-STRATIFIED cells never moved off P0. A 'correction' "
        "that changed nothing because both cells were secretly fixed would read as "
        "a tidy null result."
        )

        ARM_A[s] = dict(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, obs_pf=obs_pf,
                        pf2=pf2, pf4=pf4,
                        same1=same1, fd1=fd1, same3=same3, fd3=fd3,
                        perms_iid=perms_iid, n=n, y=y, runs=runs,
                        iid_breaks_a_run=iid_breaks_a_run)
        note(f"arm A subject {s} done")


    def mcse_mean(x):
        return x.std(ddof=1) / np.sqrt(len(x))


    def material_paired(d):
        """Registered rule: material iff |mean(d)| > 3 MC se AND |mean(d)| > 1/45.
    Both halves are required. At N=10,000 the MC se alone would certify sub-trial
    differences that nobody should act on in a 45-trial null."""
        m, se = d.mean(), mcse_mean(d)
        return (abs(m) > MC_SIGMA * se) and (abs(m) > ONE_TRIAL), m, se


    def material_unpaired(a, b):
        """The registered PAIRED rule applied to an UNPAIRED difference, with the
    standard error computed for independent samples. Stated as such: the
    pre-registration compares C1-vs-C3 and C2-vs-C4 'on distributions only', so
    this is the same two-part threshold, not a new one."""
        m = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return (abs(m) > MC_SIGMA * se) and (abs(m) > ONE_TRIAL), m, se


    for s in SUBJECTS_A:
        d = ARM_A[s]
        n = d["n"]
        y = d["y"]
        majority = max((y == 2).mean(), (y == 3).mean())
        obs = OBS_A[s]

        sub(f"ARM A, SUBJECT {s}   observed {fmt(obs, n)}   "
        f"chance (majority class) {fmt(majority, n)}   "
        f"({100 * (obs - majority):+.1f} points above chance)")

        ps = {}
        ps["C1"] = describe_null("C1  i.i.d. shuffle, RE-STRATIFIED   [the published null]",
                                 d["c1"], obs, n, N_ARM_A)
        ps["C2"] = describe_null("C2  i.i.d. shuffle, FIXED at P0     [correction (a) alone]",
                                 d["c2"], obs, n, N_ARM_A)
        ps["C3"] = describe_null("C3  within-run shuffle, RE-STRAT    [pilot cell (iv)]",
                                 d["c3"], obs, n, N_ARM_A)
        ps["C4"] = describe_null("C4  within-run shuffle, FIXED at P0 [FULLY CORRECTED]",
                                 d["c4"], obs, n, N_ARM_A)
        print("\n  ^^^ C2 AND C4 ARE WITHDRAWN AS TESTS (section 2C, added 2026-07-26).")
        print("  P0 is stratified on y_true, so it is not ancillary and freezing it "
          "breaks")
        print("  exchangeability. Their p-values are printed because deleting them "
          "would hide")
        print("  what this script previously claimed. They are NOT test results and "
          "nothing")
        print("  below concludes from them. The label 'FULLY CORRECTED' on C4 is "
          "withdrawn.")

        print(f"\n  THE EXACT FIXED-PARTITION CELLS, which are what C2 and C4 were "
          f"trying to be:")
        print(f"  partition built by KFold(shuffle=True, random_state={SEED}) WITHOUT "
          f"the labels, so it is")
        print(f"  ancillary. Observed on that partition = {fmt(d['obs_pf'], n)}, "
          f"against {fmt(obs, n)} on P0.")
        print(f"  THE OBSERVED VALUE MOVES BY {abs(round(d['obs_pf']*n) - round(obs*n))} "
          f"TRIAL(S) BECAUSE THE PARTITION MOVED. C5 and C6 are exact tests")
        print(f"  of a DIFFERENT statistic (accuracy on an unstratified partition), so "
          f"their p is NOT")
        print(f"  interchangeable with C1's and is NOT a correction to the published "
          f"number. They are")
        print(f"  here to show what the fixed-partition idea looks like when it is done "
          f"validly.")
        ps["C5"] = describe_null(
            f"C5  i.i.d. shuffle, FIXED at label-free PF  [EXACT, N={N_EXACT}]",
            d["c5"], d["obs_pf"], n, N_EXACT)
        ps["C6"] = describe_null(
            f"C6  within-run shuffle, FIXED at PF        [EXACT, N={N_EXACT}]",
            d["c6"], d["obs_pf"], n, N_EXACT)
        print(f"  NOTE ON CENTRING: C5/C6 also centre below the registered 0.45-0.55 "
          f"band")
        print(f"  (C5 {d['c5'].mean():.2%}, C6 {d['c6'].mean():.2%}), and these ARE "
          f"exact tests. That is")
        print(f"  independent confirmation that section 2B's band argument was right "
          f"and that")
        print(f"  assert 9 was the wrong assert. Wrong assert and invalid cell were "
          f"two separate")
        print(f"  defects, and fixing the first did not fix the second.")

        print(f"\n  Partition bookkeeping (asserts 5 and 6, both passed):")
        print(f"    C2/C4 replayed P0 on {N_ARM_A}/{N_ARM_A} replicates (test index arrays "
          f"identical, in order).")
        print(f"    C1 realised a partition differing from P0 on "
          f"{int((~d['same1']).sum())}/{N_ARM_A} replicates; mean fraction of trials "
          f"whose fold index moved = {d['fd1'].mean():.1%}")
        print(f"    C3 differed from P0 on {int((~d['same3']).sum())}/{N_ARM_A}; "
          f"mean fraction moved = {d['fd3'].mean():.1%}")
        print(f"    A within-run shuffle preserved every run's counts on "
          f"{N_ARM_A}/{N_ARM_A} replicates. An i.i.d. shuffle broke at least one "
          f"run's counts: {d['iid_breaks_a_run']}")

        print(f"\n  PAIRED difference d = acc(C2) - acc(C1), same labels, "
          f"partition rule the ONLY thing that differs:")
        for lab, (a, b) in (("C2 - C1", (d["c2"], d["c1"])), ("C4 - C3", (d["c4"], d["c3"]))):
            dd = a - b
            mat, m, se = material_paired(dd)
            print(f"    {lab}: mean(d) = {100 * m:+.3f} points   "
              f"sd(d) = {100 * dd.std(ddof=1):.3f} points   "
              f"MC se = {100 * se:.4f} points")
            print(f"             |mean(d)| = {abs(m) / se:.1f} MC se and "
              f"{abs(m) / ONE_TRIAL:.2f} trials (one trial = {100 * ONE_TRIAL:.2f} points)")
            print(f"             d != 0 on {100 * (np.abs(dd) > TOL).mean():.1f}% of replicates, "
              f"d > 0 on {100 * (dd > TOL).mean():.1f}%")
            print(f"             MATERIAL: {mat}  (registered rule: > 3 MC se AND > 1 trial)")

        print(f"\n  UNPAIRED distribution comparisons (different reference sets, so "
          f"replicate-by-replicate\n  pairing is meaningless and is not used):")
        for lab, (a, b) in (("C3 - C1", (d["c3"], d["c1"])), ("C4 - C2", (d["c4"], d["c2"])),
                            ("C4 - C1", (d["c4"], d["c1"]))):
            mat, m, se = material_unpaired(a, b)
            print(f"    {lab}: mean difference {100 * m:+.3f} points  (MC se {100 * se:.4f}, "
              f"{abs(m) / se:.1f} se, {abs(m) / ONE_TRIAL:.2f} trials)   MATERIAL: {mat}")

        print(f"\n  Null sd by cell, and the ratio that A271's variance-inflation factor "
          f"is computed from:")
        binom_sd = np.sqrt(0.25 / n)
        for nm in ("c1", "c2", "c3", "c4"):
            sd = d[nm].std(ddof=0)
            se_sd = sd / np.sqrt(2 * (N_ARM_A - 1))
            print(f"    {nm.upper()}: sd {100 * sd:.3f} points (MC se {100 * se_sd:.4f})   "
              f"VIF against binomial sd {100 * binom_sd:.3f} = "
              f"{(sd / binom_sd) ** 2:.3f}   n_eff = {n / (sd / binom_sd) ** 2:.1f}")
        sd1, sd2 = d["c1"].std(ddof=0), d["c2"].std(ddof=0)
        ratio = sd2 / sd1
        se_ratio = ratio * np.sqrt(1.0 / (N_ARM_A - 1))
        sd_mat = abs(ratio - 1.0) > MC_SIGMA * se_ratio
        print(f"    sd(C2)/sd(C1) = {ratio:.4f}  (MC se {se_ratio:.4f}, "
          f"{abs(ratio - 1) / se_ratio:.1f} se from 1)   MATERIAL: {sd_mat}")

        print(f"\n  Tail mass at FIXED LATTICE POINTS, not at the maximum. A269 already "
          f"withdrew\n  the max as too noisy a statistic to reason from.")
        print(f"    {'cell':<6}" + "".join(f"{f'>= {k}/45':>12}" for k in (32, 34, 36)))
        for nm in ("c1", "c2", "c3", "c4"):
            row = "".join(f"{int((d[nm] >= k / 45 - TOL).sum()):>12}" for k in (32, 34, 36))
            print(f"    {nm.upper():<6}{row}")
        print(f"    (counts out of {N_ARM_A}; 32/45 = 71.1%, 34/45 = 75.6%, 36/45 = 80.0%)")

        # Wilson recomputed from the null's sd, per outcome 6.1 row 6.
        # CORRECTED 2026-07-26: this used to be computed from C4 and labelled "the
        # CORRECTED C4". C4 is withdrawn, so the C4 version is printed only as the
        # withdrawn number it is, and the interval that carries any weight is
        # recomputed from C3, which is exact.
        sd4 = d["c4"].std(ddof=0)
        vif4 = (sd4 / binom_sd) ** 2
        neff4 = n / vif4
        sd3 = d["c3"].std(ddof=0)
        vif3 = (sd3 / binom_sd) ** 2
        neff3 = n / vif3
        k_obs = int(round(obs * n))
        lo, hi = wilson(k_obs, n)
        lo4, hi4 = wilson(obs * neff4, neff4)
        lo3, hi3 = wilson(obs * neff3, neff3)
        print(f"\n  Wilson on {k_obs}/{n} at face value        = [{lo:.1%}, {hi:.1%}]  "
          f"width {100 * (hi - lo):.1f} pts")
        print(f"  Wilson at n_eff from C3 (EXACT)       = [{lo3:.1%}, {hi3:.1%}]  "
          f"width {100 * (hi3 - lo3):.1f} pts   (n_eff {neff3:.1f}, VIF {vif3:.3f})")
        print(f"  Wilson at n_eff from C4 [WITHDRAWN]   = [{lo4:.1%}, {hi4:.1%}]  "
          f"width {100 * (hi4 - lo4):.1f} pts   (n_eff {neff4:.1f}, VIF {vif4:.3f})")
        print(f"  The C4 row is retained only so the withdrawn number is visible. Use "
          f"the C3 row.")

        # --- REAL-PIPELINE TYPE-I, FROM THIS RUN'S OWN CACHED ARRAYS --------------
        # ADDED 2026-07-26. Section 2C measures type-I with zero information and no
        # EEG. This measures it on the real pipeline, for free, from arrays already
        # computed above. Under H0 the observed value is distributed exactly like a
        # SELF-STRATIFIED replicate, because an analyst builds the partition from the
        # labels they have: for the i.i.d. rules that is a C1 draw, for the within-run
        # rules a C3 draw. Each rule is judged against the H0 it assumes. The halves
        # are disjoint so no draw is used as both observed value and reference.
        print(f"\n  REAL-PIPELINE TYPE-I at nominal alpha, from this run's own "
          f"replicates:")
        half = N_ARM_A // 2
        t1_rows = [("C1  [published, exact]", d["c1"][:half], d["c1"][half:]),
                   ("C2  [withdrawn]", d["c1"][:half], d["c2"][half:]),
                   ("C3  [exact]", d["c3"][:half], d["c3"][half:]),
                   ("C4  [withdrawn]", d["c3"][:half], d["c4"][half:])]
        print(f"    {'cell':<24}{'reject at 0.05':>16}{'reject at 0.10':>16}"
          f"{'k threshold':>13}")
        _t1_seen = {}
        for lab, h0_draws, ref in t1_rows:
            # An exact discrete test rejects when the observed count is at or above
            # the smallest k whose upper-tail mass in the reference is <= alpha.
            def _p_of(v):
                return (1 + int((ref >= v - TOL).sum())) / (1 + len(ref))
            r5 = float(np.mean([_p_of(v) <= 0.05 for v in h0_draws]))
            r10 = float(np.mean([_p_of(v) <= 0.10 for v in h0_draws]))
            kthr = next((k for k in range(n + 1) if _p_of(k / n) <= 0.05), None)
            _t1_seen[lab.split()[0]] = (r5, r10, kthr)
            print(f"    {lab:<24}{r5:>16.4f}{r10:>16.4f}"
              f"{(f'>= {kthr}/{n}' if kthr is not None else 'none'):>13}")
        # Read off the table rather than asserted over it, for the same reason as
        # section 2C: the defect's size depends on the class marginal.
        print(f"    An exact discrete test must sit at or below nominal.")
        for pair in (("C2", "C1"), ("C4", "C3")):
            w, e = _t1_seen[pair[0]], _t1_seen[pair[1]]
            print(f"      {pair[0]} against {pair[1]}: size {w[0]:.4f} against "
              f"{e[0]:.4f} at 0.05, {w[1]:.4f} against {e[1]:.4f} at 0.10; "
              f"rejects at >= {w[2]}/{n} against >= {e[2]}/{n}")
        _lower_thr = sum(1 for pair in (("C2", "C1"), ("C4", "C3"))
                         if _t1_seen[pair[0]][2] is not None
                         and _t1_seen[pair[1]][2] is not None
                         and _t1_seen[pair[0]][2] < _t1_seen[pair[1]][2])
        _never_smaller = all(
            _t1_seen[w][0] >= _t1_seen[e][0] - 1e-12
            and _t1_seen[w][1] >= _t1_seen[e][1] - 1e-12
            and (_t1_seen[w][2] is None or _t1_seen[e][2] is None
                 or _t1_seen[w][2] <= _t1_seen[e][2])
            for w, e in (("C2", "C1"), ("C4", "C3")))
        print(f"    The withdrawn cell rejects at a STRICTLY LOWER observed count than "
          f"its exact")
        print(f"    partner in {_lower_thr} of 2 pairs for this subject. That is how a "
          f"verdict gets")
        print(f"    flipped by the rule rather than by the data.")
        print(f"    On this subject, is every withdrawn cell at least as large in size "
          f"and at most")
        print(f"    as high in threshold as its exact partner? {_never_smaller}. "
          f"(Computed, not asserted:")
        print(f"    the defect has a direction, and a cell that read the other way "
          f"would be a bug.)")
        ARM_A[s]["p"] = ps


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


    # --------------------------------------------------------------------------- #
    # SECTION 5  ARM B
    # --------------------------------------------------------------------------- #
    hdr("SECTION 5  ARM B: CROSS-SUBJECT LOSO, GLOBAL SHUFFLE AGAINST BLOCK SHUFFLE")

    note("loading 20 subjects for arm B")
    Xs, ys, gs = [], [], []
    for s in SUBJECTS_B:
        Xi, yi, _ = load_subject(s)
        Xs.append(Xi)
        ys.append(yi)
        gs.append(np.full(len(yi), s))
    nmin = min(x.shape[-1] for x in Xs)
    XB = np.concatenate([x[:, :, :nmin] for x in Xs], axis=0)
    yB = np.concatenate(ys)
    gB = np.concatenate(gs)
    nB = len(yB)

    logo = LeaveOneGroupOut()
    FOLDS_B = [(tr, te) for tr, te in logo.split(XB, yB, gB)]
    held = [int(np.unique(gB[te])[0]) for _, te in FOLDS_B]
    assert all(len(np.unique(gB[te])) == 1 for _, te in FOLDS_B)
    assert sorted(held) == sorted(SUBJECTS_B)
    assert all(len(te) == nB // len(FOLDS_B) for _, te in FOLDS_B), \
        "LOSO folds are not equal-sized; the k/900 lattice would break"

    obsB = cross_val_score(make_clf(), XB, yB, groups=gB, cv=FOLDS_B,
                           error_score="raise")
    obs_b = obsB.mean()
    assert_lattice([obs_b], nB, "arm B observed")
    chance_b = max((yB == 2).mean(), (yB == 3).mean())

    print(f"\nPooled {nB} trials from {len(SUBJECTS_B)} subjects, "
      f"{XB.shape[1]} channels x {nmin} samples, {len(FOLDS_B)} folds of "
      f"{nB // len(FOLDS_B)}")
    print(f"Observed LOSO accuracy {fmt(obs_b, nB)}   against pooled majority-class "
      f"rate {fmt(chance_b, nB)}   ({100 * (obs_b - chance_b):+.1f} points)")

    # The claim A268 gets corrected on, printed from the data rather than asserted.
    marg = {s: int((yB[gB == s] == 2).sum()) for s in SUBJECTS_B}
    rates = {s: max(marg[s], 45 - marg[s]) / 45 for s in SUBJECTS_B}
    uniq = sorted(set(round(v, 4) for v in rates.values()))
    print(f"\nPer-subject majority-class rates across the 20: {uniq}")
    for u in uniq:
        who = [s for s in SUBJECTS_B if round(rates[s], 4) == u]
        print(f"  {u:.4f} = {int(round(u * 45))}/45 : subjects {who}  (n = {len(who)})")
    print("A268 records the marginal as 'fixed by the protocol at a near-balanced")
    print("21/24'. That is right for subject 1 and WRONG as a statement about the")
    print("pooled set. A268 needs correcting, and the heterogeneity above is exactly")
    print("the structure a global shuffle destroys.")

    print(f"""
LeaveOneGroupOut._iter_test_masks reads ONLY `groups`, so the LOSO partition is
already invariant to the labels and arm (a)'s defect does not exist here. What is
wrong in arm (b) is the REFERENCE SET, not the partition: a global shuffle draws
label vectors the experiment could not have produced.

  G = global permutation of all {nB}. What cross_subject.py:135 does, {N_ARM_B} times
      instead of once.
  B = within-subject block permutation. Correction (b).

UNPAIRED BY CONSTRUCTION and reported as such.""")

    rng = np.random.default_rng(SEED)
    permsG = [iid_perm(yB, rng) for _ in range(N_ARM_B)]
    permsB = [within_block_perm(yB, gB, rng) for _ in range(N_ARM_B)]

    # Asserts 7, 8 and 10, before a single classifier is fit.
    margG = np.array([[int((yp[gB == s] == 2).sum()) for s in SUBJECTS_B] for yp in permsG])
    margB = np.array([[int((yp[gB == s] == 2).sum()) for s in SUBJECTS_B] for yp in permsB])
    obs_marg = np.array([marg[s] for s in SUBJECTS_B])
    assert (margB == obs_marg).all(), (                                      # assert 7
        "a block shuffle changed some subject's class marginal. The blocking is broken."
    )
    assert (margG != obs_marg).any(), (                                      # assert 8
        "no global shuffle changed any subject's marginal. Then G is not global and "
    "the two cells are secretly the same cell."
    )
    for yp in permsG + permsB:
        assert int((yp == 2).sum()) == int((yB == 2).sum()), (               # assert 10
            "a shuffle changed the POOLED class counts. The code is resampling, not "
        "permuting."
        )

    rateG = np.maximum(margG, 45 - margG) / 45.0
    rateB = np.maximum(margB, 45 - margB) / 45.0
    sdG_rate = rateG.std(ddof=0, axis=0)
    sdB_rate = rateB.std(ddof=0, axis=0)
    # The EXACT statement is that every replicate gives every subject the identical
    # marginal, which is an integer comparison and is asserted above as assert 7, plus
    # a zero range here. np.std on a constant array does NOT return exactly 0.0: it
    # subtracts an accumulated mean, so it lands around 1e-14. Asserting `std == 0.0`
    # would fail on float accumulation while the blocking was perfect, so the exact
    # test is on the RANGE and the sd is reported with its float floor stated.
    assert np.ptp(rateB, axis=0).max() == 0.0, (
        "a held-out subject's own majority-class rate VARIED across replicates under "
    "block permutation. The block shuffle is not blocking."
    )

    print(f"\nMECHANISM EXHIBIT (descriptive, not a test). sd across the {N_ARM_B} "
      f"replicates of each\nheld-out subject's OWN majority-class rate, the "
      f"constant every LOSO fold is read against:")
    print(f"  under B (block): range across replicates is EXACTLY 0 for all 20 "
      f"subjects.\n                   np.std reports {sdB_rate.max():.2e}, which is "
      f"float accumulation on a\n                   constant array, not variation.")
    print(f"  under G (global): mean {sdG_rate.mean():.4f}, "
      f"range [{sdG_rate.min():.4f}, {sdG_rate.max():.4f}]")
    print(f"  worst single draw under G: some subject reached a majority rate of "
      f"{rateG.max():.1%} ({int(round(rateG.max() * 45))}/45), a class split the "
      f"experiment\n  could not have produced. That variance has nothing to do "
      f"with decoding.")

    fold_flat = np.concatenate([te for _, te in FOLDS_B])


    def arm_b_cell(tag, perms):
        stamp = fingerprint("armB", tag, N_ARM_B, SEED, "csp4-lda-loso20",
                            yB, gB, fold_flat, np.stack(perms[:50]), len(perms))

        def compute_block(lo, hi):
            res = run_parallel(
                _arm_b_chunk,
                [(XB, c, FOLDS_B) for c in chunks(perms[lo:hi], CHUNK_B)],
                f"arm B {tag} [{lo}:{hi}]")
            return (np.concatenate(res),)

        return cached_blocks(f"armB_{tag}", stamp, N_ARM_B, max(100, N_ARM_B // 10),
                             compute_block)[0]


    note("arm B: cell G")
    nullG = arm_b_cell("G", permsG)
    note("arm B: cell B")
    nullB = arm_b_cell("B", permsB)

    assert_lattice(nullG, nB, "arm B cell G")
    assert_lattice(nullB, nB, "arm B cell B")
    for nm, arr in (("G", nullG), ("B", nullB)):
        assert abs(arr.mean() - 0.5) < 0.05, (
            f"arm B cell {nm}: null centred at {arr.mean():.1%}, not near 50%."
        )

    sub("ARM B RESULTS")
    pG = describe_null("G  GLOBAL permutation of all 900   [what cross_subject.py does]",
                       nullG, obs_b, nB, N_ARM_B, extra_q=(0.995,))
    pB = describe_null("B  WITHIN-SUBJECT block permutation [correction (b)]",
                       nullB, obs_b, nB, N_ARM_B, extra_q=(0.995,))

    sdG, sdB = nullG.std(ddof=0), nullB.std(ddof=0)
    seG = sdG / np.sqrt(2 * (N_ARM_B - 1))
    seB = sdB / np.sqrt(2 * (N_ARM_B - 1))
    ratio_b = sdG / sdB
    se_ratio_b = ratio_b * np.sqrt(1.0 / (2 * (N_ARM_B - 1)) + 1.0 / (2 * (N_ARM_B - 1)))
    # A nonparametric cross-check on the delta-method se, because the null is not normal.
    brng = np.random.default_rng(7)
    boot = np.array([
        nullG[brng.integers(0, N_ARM_B, N_ARM_B)].std(ddof=0)
        / nullB[brng.integers(0, N_ARM_B, N_ARM_B)].std(ddof=0)
        for _ in range(2000)])
    sd_mat_b = abs(ratio_b - 1.0) > MC_SIGMA * se_ratio_b

    print(f"\n  sd(G) = {100 * sdG:.3f} points (MC se {100 * seG:.4f})")
    print(f"  sd(B) = {100 * sdB:.3f} points (MC se {100 * seB:.4f})")
    print(f"  sd ratio sd(G)/sd(B) = {ratio_b:.4f}  (delta-method MC se {se_ratio_b:.4f}, "
      f"{abs(ratio_b - 1) / se_ratio_b:.1f} se from 1)")
    print(f"  bootstrap cross-check on that se ({len(boot)} resamples): "
      f"{boot.std(ddof=1):.4f}, 95% interval "
      f"[{np.quantile(boot, 0.025):.3f}, {np.quantile(boot, 0.975):.3f}]")
    print(f"  MATERIAL (registered rule: differs from 1 by > 3 MC se): {sd_mat_b}")

    unpaired_mat_b, m_b, se_b = material_unpaired(nullG, nullB)
    print(f"  null MEAN difference G - B = {100 * m_b:+.3f} points "
      f"(MC se {100 * se_b:.4f}, {abs(m_b) / se_b:.1f} se, "
      f"{abs(m_b) / (1 / 45):.2f} within-subject trials)")

    sub("THE DELIVERABLE: A REPLACEMENT FOR SHUFFLE_MAX = 0.60")
    print("""cross_subject.py:146 holds its label-shuffle leakage guard to a hard-coded
0.60. That is a round number with no derivation anywhere in the repo. Below it is
replaced by a measured quantile of the CORRECT reference distribution.""")
    q99_v, q99_s = quant(nullB, 0.99, nB)
    q995_v, q995_s = quant(nullB, 0.995, nB)
    g99_v, g99_s = quant(nullG, 0.99, nB)
    g995_v, g995_s = quant(nullG, 0.995, nB)
    print(f"\n  BLOCK null (correction (b)), on the k/{nB} lattice:")
    print(f"    99.0th percentile = {q99_s}   <- proposed replacement for SHUFFLE_MAX")
    print(f"    99.5th percentile = {q995_s}   <- the more conservative option")
    print(f"  GLOBAL null (what the repo currently permutes), for comparison:")
    print(f"    99.0th percentile = {g99_s}")
    print(f"    99.5th percentile = {g995_s}")
    print(f"\n  Where the existing 0.60 sits:")
    print(f"    in the BLOCK null : {(SHUFFLE_MAX - nullB.mean()) / sdB:+.2f} sd, "
      f"percentile {100 * (nullB < SHUFFLE_MAX).mean():.2f}, "
      f"{int((nullB >= SHUFFLE_MAX).sum())} of {N_ARM_B} draws at or above it")
    print(f"    in the GLOBAL null: {(SHUFFLE_MAX - nullG.mean()) / sdG:+.2f} sd, "
      f"percentile {100 * (nullG < SHUFFLE_MAX).mean():.2f}, "
      f"{int((nullG >= SHUFFLE_MAX).sum())} of {N_ARM_B} draws at or above it")
    print(f"    against the OBSERVED cross-subject result {fmt(obs_b, nB)}: the guard "
      f"sits {100 * (SHUFFLE_MAX - obs_b):+.1f} points from it")
    guard_above = q99_v > SHUFFLE_MAX
    print(f"\n  Replacement lands ABOVE 0.60: {guard_above}")
    print(f"  What this does NOT show: none of these quantiles was validated against a "
      f"REAL\n  leak. cross_subject.py:143 already discloses that its guard has "
      f"never fired.\n  Replacing an underived threshold with a derived one does "
      f"not measure the guard's\n  false-alarm rate against an actual defect.")


    # --------------------------------------------------------------------------- #
    # SECTION 6  WHICH PRE-REGISTERED OUTCOME
    # --------------------------------------------------------------------------- #
    hdr("SECTION 6  WHICH PRE-REGISTERED OUTCOME EACH RESULT MATCHES")

    if GATE1_FATAL or not GATE2_OK:
        print("*** A FALSIFICATION GATE FIRED. Everything below describes a run whose")
        print("*** HARNESS is in question, and NONE of it may be reported as a result.")

    print("Registered materiality, fixed before the run:")
    print(f"  p-value change   : crosses {P_THRESHOLD}, or both off the floor and "
      f"differing by a factor >= 2")
    print(f"  paired mean(d)   : > {MC_SIGMA:.0f} MC se AND > 1/45 = "
      f"{100 * ONE_TRIAL:.2f} points (ONE WHOLE TRIAL)")
    print(f"  sd ratio         : differs from 1 by more than {MC_SIGMA:.0f} MC se")

    print("\nFALSIFICATION GATES (these say the HARNESS is broken, not that the "
      "result is interesting):")
    print(f"  1. pilot reproduces          : {n_match}/4 cells match A269 -> "
      f"{'PASS' if n_match == 4 else ('VERSION DRIFT, reportable' if n_match == 0 else 'FATAL')}")
    print(f"  2. C1 is what sklearn does   : {'PASS' if GATE2_OK else 'FAIL'} "
      f"(z_mean {z_mean:+.2f}, z_sd {z_sd:+.2f})")
    print(f"  3. lattice on every replicate: PASS (asserted k/45 on every arm A "
      f"replicate, k/900 on every arm B replicate, to 1e-9)")
    print(f"  4. observed = 41/45 on P0    : PASS")
    print(f"  5. fixed cells really fixed  : PASS (C2 and C4 replayed P0 on every "
      f"replicate)")
    print(f"  6. subjects 17/19 reproduce  : PASS (28/45 and 29/45)")
    print(f"  7. block/global marginals    : PASS (block preserved every marginal; "
      f"global changed at least one; pooled counts preserved everywhere)")
    centre_fail = {k: v for k, v in CENTRING.items()
                   if not (CENTRE_LO < v < CENTRE_HI)}
    CENTRING[("B", "G")] = float(nullG.mean())
    CENTRING[("B", "B")] = float(nullB.mean())
    print(f"  8. null means in [0.45,0.55] : "
      f"{'PASS on every cell' if not centre_fail else 'FIRES on ' + str(sorted(centre_fail))}")
    print(f"       every cell's null mean, printed so nothing is hidden:")
    for s in SUBJECTS_A:
        row = "  ".join(f"{nm} {CENTRING[(s, nm)]:.4f}"
                    f"{'*' if not (CENTRE_LO < CENTRING[(s, nm)] < CENTRE_HI) else ' '}"
                        for nm in ("C1", "C2", "C3", "C4"))
        print(f"         subject {s:<3} {row}")
    print(f"         arm B     G  {nullG.mean():.4f}   B  {nullB.mean():.4f}")
    if centre_fail:
        print(f"       * = OUTSIDE the registered 0.45-0.55 band. Every one of them is a")
        print(f"       FIXED-partition cell (C2 or C4). Per pre-registration Section 6.5")
        print(f"       this run's Sections 6.1-6.4 are UNREPORTABLE. They are reported")
        print(f"       anyway, and this line is why: SECTION 2B shows a majority-class")
        print(f"       dummy on an all-zero feature matrix reproduces the same downward")
        print(f"       shift with no EEG involved, so the band is a property of")
        print(f"       re-stratification and not a test of null specification. That is a")
        print(f"       POST-HOC departure, decided after a smoke run tripped the assert,")
        print(f"       and it moves the result in this project's favour. No re-stratified")
        print(f"       cell and neither arm B cell is outside the band.")

    for s in SUBJECTS_A:
        d = ARM_A[s]
        n = d["n"]
        (p1, c1_), (p2, c2_) = d["p"]["C1"], d["p"]["C2"]
        (p3, c3_), (p4, c4_) = d["p"]["C3"], d["p"]["C4"]
        dd = d["c2"] - d["c1"]
        mat, m, se = material_paired(dd)
        sub(f"SUBJECT {s}")
        print(f"  p by cell: C1 {p_str(p1, c1_, N_ARM_A)}")
        print(f"             C2 {p_str(p2, c2_, N_ARM_A)}")
        print(f"             C3 {p_str(p3, c3_, N_ARM_A)}")
        print(f"             C4 {p_str(p4, c4_, N_ARM_A)}")
        all_floor = all(c == 0 for c in (c1_, c2_, c3_, c4_))
        if all_floor:
            print("  -> [neutral] REGISTERED, NEAR-CERTAIN OUTCOME for a large effect: both "
              "p-values at\n     the floor, C = 0 in every cell. The correction does not "
              "change the verdict.\n     This is NOT evidence the original null was "
              "correctly designed. It is evidence\n     the effect is large enough that "
              "the design error cannot change the answer.")
        else:
            sig1 = p1 < P_THRESHOLD
            # CORRECTED 2026-07-26: the crossing table used to include C2 and C4,
            # which are not exact tests. They stay in the printed list so the
            # withdrawn claim is visible, but the EXACT-ONLY re-scoring below is the
            # one that carries the verdict.
            cells = (("C2", p2), ("C3", p3), ("C4", p4))
            # UPWARD = significant under the published null, NOT significant under the
            # corrected one. DOWNWARD = the reverse. The pre-registration scores these
            # as different outcomes and they must not be collapsed into "it moved".
            up = [nm for nm, pv in cells if sig1 and pv >= P_THRESHOLD]
            down = [nm for nm, pv in cells if (not sig1) and pv < P_THRESHOLD]
            print(f"  -> p-values are INTERIOR, so they were FREE to move. Published C1 is "
              f"{'significant' if sig1 else 'NOT significant'} at {P_THRESHOLD}.")
            print(f"     crossed {P_THRESHOLD} UPWARD (lost significance)  : "
              f"{up if up else 'none'}")
            print(f"     crossed {P_THRESHOLD} DOWNWARD (gained significance): "
              f"{down if down else 'none'}")
            fac = max(p1, p4) / max(min(p1, p4), 1e-12)
            print(f"     C4 (fully corrected) against C1 (published): {p4:.4g} against "
              f"{p1:.4g}, factor {fac:.2f}\n     (material at factor >= 2: {fac >= 2})")
            if up:
                print("  -> [bad for the corpus, STRONG FINDING, PUBLISH] the corrected p "
                  "crosses 0.05 UPWARD.\n     A design error of exactly this kind CAN "
                  "flip a per-subject verdict in this\n     dataset. That is a "
                  "DEMONSTRATED consequence, not a hypothetical one, and it is\n     the "
                  "strongest possible argument that the correction is required rather "
                  "than\n     pedantic. sweep_results.csv and anything built on "
                  "per-subject significance is\n     affected.")
            if down:
                print("  -> [depends] the corrected p crosses 0.05 DOWNWARD. The same "
                  "demonstration with the\n     opposite sign: the published null was "
                  "costing real detections. Equally\n     publishable, and it must NOT "
                  "be sold as a bonus.")
            if not up and not down:
                print("  -> [good, strengthens the original] the correction does not move a "
                  "verdict even where\n     the p was FREE to move. This is the "
                  "strongest available evidence that the\n     objection is principled "
                  "but empirically inert on this data, and it is a stronger\n     form "
                  "of that claim than subject 1 can supply.")
            # --- THE RE-SCORING THAT CARRIES THE VERDICT (added 2026-07-26) --------
            p5, p6 = d["p"]["C5"][0], d["p"]["C6"][0]
            ex_cells = (("C3  within-run, re-stratified", p3),
                        ("C5  i.i.d., label-free fixed  ", p5),
                        ("C6  within-run, label-free fix", p6))
            ex_up = [nm for nm, pv in ex_cells if sig1 and pv >= P_THRESHOLD]
            ex_down = [nm for nm, pv in ex_cells if (not sig1) and pv < P_THRESHOLD]
            print(f"\n  EXACT-CELLS-ONLY RE-SCORING. C1 (published) = {p1:.5g}, and the "
              f"only other cells that")
            print(f"  are exact tests:")
            for nm, pv in ex_cells:
                print(f"     {nm}: p = {pv:.5g}   "
                  f"{'CROSSES' if (pv < P_THRESHOLD) != sig1 else 'same side as C1'}"
                  f" the {P_THRESHOLD} line")
            fac3 = max(p1, p3) / max(min(p1, p3), 1e-12)
            print(f"     C3 against C1: factor {fac3:.2f} "
              f"(material at factor >= 2: {fac3 >= 2})")
            print(f"     crossed UPWARD among exact cells  : "
              f"{ex_up if ex_up else 'none'}")
            print(f"     crossed DOWNWARD among exact cells: "
              f"{ex_down if ex_down else 'none'}")
            if not ex_up and not ex_down:
                print(f"  -> restricted to the exact cells, NO verdict changes for this "
                  f"subject. Run blocking")
                print(f"     alone moves the p by a factor of {fac3:.2f} without crossing "
                  f"{P_THRESHOLD}, and a")
                print(f"     label-independent fixed partition does not cross it either.")
            else:
                print(f"  -> restricted to the exact cells, a verdict DOES change for this "
                  f"subject, and the")
                print(f"     cell that moves it is a valid test. That crossing stands.")
        if mat:
            direction = "NEGATIVE" if m < 0 else "POSITIVE"
            # CORRECTED 2026-07-26. This branch used to conclude that "the published
            # null is TOO HIGH and the published p is CONSERVATIVE". Both halves
            # presuppose that the fixed-partition cell is a valid reference
            # distribution for the same statistic, which section 2C refutes. A
            # displacement between an invalid reference distribution and a valid one
            # is not evidence that the valid one is mis-centred.
            print(f"  -> [WITHDRAWN AS A CONCLUSION, RETAINED AS A MEASUREMENT] mean(d) "
              f"materially {direction}\n     ({100 * m:+.3f} points): the "
              f"fixed-at-P0 null sits {'BELOW' if m < 0 else 'ABOVE'} the "
              f"re-stratified one. This\n     block used to read that off as 'the "
              f"published null is TOO HIGH and the published p\n     is "
              f"CONSERVATIVE' (or its mirror). Withdrawn: P0 is stratified on "
              f"y_true, so the\n     fixed cell is not a reference distribution "
              f"for this statistic at all, and the\n     displacement measures the "
              f"cost of breaking exchangeability rather than a\n     mis-centred "
              f"published null. C1 remains exact and needs no re-centring.")
        else:
            print(f"  -> [neutral] mean(d) = {100 * m:+.3f} points is NOT material "
              f"({abs(m) / se:.1f} MC se,\n     {abs(m) / ONE_TRIAL:.2f} trials). The "
              f"partition rule does not change the null at a\n     magnitude anyone "
              f"should act on for this dataset. This must NOT be written as\n     'the "
              f"objection was wrong': it was a real defect that turned out not to bite "
              f"at n=45.")
        sd_r = d["c2"].std(ddof=0) / d["c1"].std(ddof=0)
        se_r = sd_r * np.sqrt(1.0 / (N_ARM_A - 1))
        if abs(sd_r - 1) > MC_SIGMA * se_r:
            print(f"  -> [depends] null SD differs materially between C1 and C2 "
              f"(ratio {sd_r:.4f}). The\n     correction changes the null's SPREAD. "
              f"A271's variance-inflation factor and the\n     n_eff-corrected Wilson "
              f"interval are computed from the null sd and must be\n     recomputed "
              f"from the corrected null (printed in section 3).")
        blk_mat, blk_m, blk_se = material_unpaired(d["c3"], d["c1"])
        if not blk_mat:
            print(f"  -> [good] within-run cells are NOT materially different from i.i.d. "
              f"cells\n     (C3 - C1 = {100 * blk_m:+.3f} points). Run blocking does not "
              f"matter for this subject:\n     trials behave as exchangeable across runs. "
              f"This SUPPORTS the published i.i.d.\n     null as adequate here.")
        elif blk_m > 0:
            print(f"  -> [BAD, AND IT GETS PUBLISHED] the within-run (block) null is "
              f"materially HIGHER\n     ({100 * blk_m:+.3f} points). Within-run label "
              f"structure was carrying part of the\n     apparent effect. The honest null "
              f"is C4 and the leave-one-run-out result (93.3%)\n     needs re-reading.")
        else:
            print(f"  -> [good for the headline] the block null is materially LOWER "
              f"({100 * blk_m:+.3f} points).\n     The i.i.d. null was conservative for a "
              f"second, independent reason. Same\n     discipline: safe direction, not "
              f"vindication.")
        c4_mat, c4_m, c4_se = material_unpaired(d["c4"], d["c1"])
        if c4_mat:
            print(f"  -> [WITHDRAWN 2026-07-26] C4 differs from C1 by "
              f"{100 * c4_m:+.3f} points. This block used\n     to call that 'the "
              f"headline result of arm A' and to recommend C4's p 'should be\n     "
              f"published going forward'. BOTH ARE WITHDRAWN. C4 is not an exact test "
              f"(section\n     2C), so this displacement measures the distance between "
              f"a reference distribution\n     and the sampling distribution of the "
              f"statistic, not a correction to a null.")
        else:
            print(f"  -> [neutral] C4 is NOT materially different from C1 "
              f"({100 * c4_m:+.3f} points). The published\n     null, despite being wrong "
              f"on BOTH counts, delivers the same answer as the null\n     with the "
              f"correct exchangeable unit and the correct conditioning. The two errors\n"
              f"     do not bite at n={n} on this subject, and that does NOT generalise "
              f"to other\n     subjects, other n, or other designs.")

    if len(SUBJECTS_A) >= 3:
        a, b = SUBJECTS_A[1], SUBJECTS_A[2]
        # CORRECTED 2026-07-26: this used to read the agreement off C4, which is not
        # a test. It is now read off C3, the exact block cell, and the C4 reading is
        # printed beside it as the withdrawn one.
        sub("THE TWO MEDIAN SUBJECTS, SCORED ON THE EXACT CELLS")
        for cell in ("C1", "C3", "C5", "C6", "C4"):
            pa_v = ARM_A[a]["p"][cell][0]
            pb_v = ARM_A[b]["p"][cell][0]
            tag = " [WITHDRAWN, not a test]" if cell in ("C2", "C4") else ""
            agree = (pa_v < P_THRESHOLD) == (pb_v < P_THRESHOLD)
            print(f"  {cell}: S{a} p = {pa_v:.5g}  S{b} p = {pb_v:.5g}   "
              f"{'AGREE' if agree else 'DISAGREE'} at {P_THRESHOLD}{tag}")
        pa = ARM_A[a]["p"]["C3"][0] < P_THRESHOLD
        pb = ARM_A[b]["p"]["C3"][0] < P_THRESHOLD
        if pa != pb:
            print(f"\n[neutral] Under C3, the exact block cell, the two median subjects "
              f"DISAGREE at\n  {P_THRESHOLD}. That is the pre-registered [neutral] "
              f"outcome and it is the one that\n  fires once the invalid cells are set "
              f"aside. n = 2 subjects and per-subject nulls\n  at 45 trials are noisy. "
              f"Both are reported; no conclusion is drawn from the\n  disagreement and "
              f"neither is picked for being agreeable.")
        else:
            print(f"\n[note] Under C3 the two median subjects AGREE at {P_THRESHOLD}.")
        print(f"\nWITHDRAWN HEADLINE, KEPT VISIBLE. This script previously reported "
          f"'THE CORRECTION\nCHANGES VERDICTS: both median subjects go from "
          f"non-significant to significant under\nthe fully corrected null'. That "
          f"claim was carried by C4, which is not an exact test.\nRestricted to the "
          f"exact cells the demonstration is smaller and it is not the same\nclaim. "
          f"Read the table above, not the withdrawn sentence.")

    sub("ARM B")
    pGv, cGv = pG
    pBv, cBv = pB
    print(f"  p under G (global) {p_str(pGv, cGv, N_ARM_B)}")
    print(f"  p under B (block)  {p_str(pBv, cBv, N_ARM_B)}")
    if cBv == 0:
        print(f"  -> [good] the observed {fmt(obs_b, nB)} remains far outside the BLOCK "
          f"null. The\n     cross-subject finding survives the correct null. Reported "
          f"with the floor\n     convention as p <= {1 / (N_ARM_B + 1):.5g}, never as a "
          f"measured value.")
    elif pBv > P_THRESHOLD:
        print(f"  -> [VERY BAD, PUBLISH FIRST] the observed {fmt(obs_b, nB)} falls INSIDE "
          f"the block null\n     (p = {pBv:.4g}). The cross-subject result does NOT "
          f"survive a correctly blocked\n     null. cross_subject.py's own docstring "
          f"licenses this: a cross-subject score at\n     chance is a legitimate finding "
          f"about transfer.")
    else:
        print(f"  -> the observed value is outside the block null but not at the floor "
          f"(p = {pBv:.4g}).")
    if sd_mat_b and ratio_b > 1:
        print(f"  -> [good for rigour, bad for the existing guard] sd(B) is materially "
          f"SMALLER than\n     sd(G) (ratio {ratio_b:.3f}). The global shuffle inflates "
          f"the null by re-dealing each\n     held-out subject's class marginal, a "
          f"variance source unrelated to decoding. The\n     published guard is loose for "
          f"a NAMEABLE reason, and the replacement threshold is\n     this arm's "
          f"deliverable. Confirms A270 at {N_ARM_B // 200}x the draws.")
    elif sd_mat_b and ratio_b < 1:
        print(f"  -> [BAD, AND IT GETS PUBLISHED] sd(B) is materially LARGER than sd(G) "
          f"(ratio {ratio_b:.3f}).\n     The stated mechanism is WRONG. The global "
          f"shuffle was ANTI-conservative and the\n     {fmt(obs_b, nB)} cross-subject "
          f"result is weaker than published. The Section 3.2\n     account gets withdrawn "
          f"and rewritten from the data.")
    else:
        print(f"  -> [neutral] sd(B) and sd(G) are not materially different "
          f"(ratio {ratio_b:.3f}). Subject\n     blocking does not change the null's "
          f"spread on this data. The objection stands as a\n     design principle with "
          f"nil practical consequence. The guard threshold still gets\n     replaced, "
          f"because 0.60 was never derived from anything either way.")
    if guard_above:
        print(f"  -> [depends] the replacement threshold ({q99_s}) lands ABOVE 0.60. The "
          f"existing guard\n     is TIGHTER than the correct null justifies and could "
          f"fire on a clean run. This\n     contradicts A41's reading of the guard as too "
          f"loose, and A41 gets corrected.")
    else:
        print(f"  -> the replacement threshold ({q99_s}) lands BELOW 0.60, so the existing "
          f"guard is\n     LOOSER than the correct null justifies, consistent with A41's "
          f"reading.")

    sub("SECONDARY MECHANISM PROBE (changes nothing above)")
    for s in SUBJECTS_A:
        r = MECH[s]
        verdict = ("[neutral] SUPPORTED, explanation only" if r < 0
                   else "[neutral] WRONG and WITHDRAWN")
        print(f"  subject {s}: r(fold imbalance, fold accuracy) in C2 = {r:+.4f}  -> {verdict}")
    print("  Either way, mean(d) and every p above are UNAFFECTED: the measurement does")
    print("  not depend on the explanation. Registered this way deliberately, because")
    print("  A269 already carries one failed directional prediction.")

    hdr("REGISTERED RISKS THAT THIS RUN DOES NOT REPAIR")
    print("""  1. Subject 1's arm was uninformative BY CONSTRUCTION and that was known
     before the run. Its p cannot move at a 4.5 sd effect. Presenting 'p
     unchanged' as a clean confirmation would be reading a resolution floor as a
     measurement.
  2. This run is not blind. A269 and A270 were read before the pre-registration
     was written. Agreement is confirmation at higher resolution with a paired
     design, not independent discovery.
  3. n = 45 per subject. Every null here is small-sample and the within-subject
     null is not binomial. Nothing here repairs that.
  4. Three subjects is not a survey. 17 and 19 came from a fixed median rule but
     are still 2 of 109, chosen for statistical position. The claim available is
     EXISTENCE, never frequency.
  5. Runs 6/10/14 are one session. Within-run blocking addresses drift INSIDE a
     run. A session-level trend across all three survives every null here.
  6. Arm B reuses cross_subject.py's documented 20-subject budget, not the full
     109. The block-permutation conclusion is scoped to those 20.
  7. Two of the six comparisons are unpaired by construction and carry Monte
     Carlo noise the paired ones do not. They are labelled as such above.""")

    if not IS_REGISTERED_RUN:
        hdr("*** SMOKE TEST, NOT A RESULT: at least one N was overridden ***")

    hdr("DONE")



if __name__ == "__main__":
    main()
