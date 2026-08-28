"""Section 5 of permutation_design.py: arm B, the cross-subject LOSO nulls,
global shuffle against block shuffle, and the replacement for SHUFFLE_MAX. Split
out 2026-08-26; the body is verbatim from that file."""

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from common import assert_lattice, make_clf
from permdesign_lib import (
    CHUNK_B, MC_SIGMA, N_ARM_B, SEED, SHUFFLE_MAX, SUBJECTS_B, fmt, hdr,
    load_subject, note, p_str, quant, sub, describe_null,
)
from permdesign_workers import (
    _arm_b_chunk, cached_blocks, chunks, fingerprint, iid_perm, run_parallel,
    within_block_perm,
)


def run_arm_b(material_unpaired):
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

    return SimpleNamespace(
        nullG=nullG, nullB=nullB, pG=pG, pB=pB, obs_b=obs_b, nB=nB,
        sd_mat_b=sd_mat_b, ratio_b=ratio_b, guard_above=guard_above,
        q99_s=q99_s,
    )
