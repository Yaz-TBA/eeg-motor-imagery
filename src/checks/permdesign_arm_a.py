"""Section 3 of permutation_design.py: arm A, the 2x2 factorial paired on the
labels, with the exact label-free cells C5 and C6, the ten asserts, and the full
per-subject report. Split out 2026-08-26; the body is verbatim, withdrawals
included."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from common import assert_lattice, make_clf
from permdesign_lib import (
    CENTRE_HI, CENTRE_LO, CENTRING, CHUNK_A, MC_SIGMA, N_ARM_A, N_EXACT,
    N_SPLITS, ONE_TRIAL, SEED, SUBJECTS_A, TOL, describe_null, fmt, hdr, note,
    sub, wilson,
)
from permdesign_workers import (
    _arm_a_chunk, _arm_fixed_chunk, cached_blocks, chunks, fingerprint,
    iid_perm, run_parallel, within_block_perm,
)


def run_arm_a(D):
    DATA_A, OBS_A, P0_A = D.DATA_A, D.OBS_A, D.P0_A

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

    return ARM_A, material_paired, material_unpaired
