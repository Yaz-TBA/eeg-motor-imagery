"""Sections 1 and 2 of permutation_design.py: falsification gate 1 (does the
pilot reproduce) and gate 2 (is C1 what sklearn does). Split out 2026-08-26; the
bodies are verbatim from that file."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import time

import numpy as np
from sklearn.model_selection import StratifiedKFold, permutation_test_score

from common import assert_lattice, make_clf
from permdesign_lib import (
    A269_PILOT, A269_PILOT_P, CHUNK_A, MC_SIGMA, N_GATE2, N_JOBS, N_REPRO,
    N_SPLITS, SEED, TOL, fmt, hdr, note, p_value,
)
from permdesign_workers import (
    _score_chunk_simple, chunks, cyclic_shift, iid_perm, run_parallel,
    within_block_perm,
)


def run_gate1(D):
    DATA_A, OBS_A, P0_A = D.DATA_A, D.OBS_A, D.P0_A

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

    return n_match, GATE1_FATAL


def run_gate2(D):
    DATA_A, OBS_A = D.DATA_A, D.OBS_A
    X1, y1, runs1 = DATA_A[1]
    n1 = len(y1)

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

    return GATE2_OK, z_mean, z_sd
