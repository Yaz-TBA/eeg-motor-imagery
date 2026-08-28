"""The permutation schemes, the scoring workers and the block checkpointing of
permutation_design.py. Split out 2026-08-26; the bodies are verbatim, and the
block-cache stamps are unchanged, so existing .permutation_design_cache/ blocks
keep resolving."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import hashlib
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, check_cv, cross_val_score

from common import make_clf
from permdesign_lib import N_JOBS, N_SPLITS, SEED, note

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
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".permutation_design_cache")   # at the repo root, two levels up from src/checks/


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
