"""The secondary arms of ablate_channels.py: the complement re-referenced within
its own 47, the permutation test on the complement, the Wilson interval, and the
registered exact McNemar, paired per trial. Split out 2026-08-26; the stage bodies
are verbatim from that file."""

import ablation_data  # noqa: F401  -- installs the common.py path first

import numpy as np
import mne
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    permutation_test_score
from mne.datasets import eegbci

from common import make_clf, wilson_interval
from ablation_data import H_FREQ, L_FREQ, N_PERMUTATIONS, SEED, SEEDS, TMAX, TMIN
from ablation_design import ALPHA, forced_mcnemar_grid, most_lopsided_failing_split


def run_secondary(edf_paths, cropped, COMPLEMENT, labels, by_name, COMP, sweeps,
                  n, NOISE_BAND, G_THRESHOLD):
    # --- SECONDARY: the complement re-referenced inside its own 47 ----------------
    # WHAT THIS ARM PROVABLY IS, CORRECTED 2026-07-26. It used to be described here
    # and in the pre-registration's RESULTS as "the leak removed". It is not. Write
    # out the algebra: the primary is x_i - m64(x), the secondary is x_i - m47(x), so
    # secondary = primary - m47(primary), which is the primary MINUS ITS OWN
    # ACROSS-CHANNEL MEAN. That is a RANK-1 COMMON-MODE PROJECTION: it deletes one
    # spatial dimension, the uniform direction over the 47, and the rank of the data
    # drops from 47 to 46. Both facts are measured and asserted below rather than
    # argued.
    #
    # The time course it deletes is m64(x) - m47(x) = -(17/64) * (m47(x) - m17(x)),
    # which MIXES the average-referenced strip contribution -(17/64) m17 with
    # (17/64) of the complement's OWN global component m47. So the points this arm
    # costs cannot be assigned to the strip leak alone. A clean arm would project out
    # only an estimate of the strip's common mode and leave the complement's intact.
    # That arm is not built here and is not claimed.
    def build_reref_within(picks):
        """Rebuild from the EDFs with the average reference taken over PICKS alone."""
        _raw = mne.concatenate_raws(
            [mne.io.read_raw_edf(p, preload=True) for p in edf_paths])
        eegbci.standardize(_raw)
        _raw.set_montage("standard_1005")
        _raw.pick(picks)                    # pick BEFORE referencing: that is the point
        _raw.set_eeg_reference("average", projection=False)
        _raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")
        _ev, _ = mne.events_from_annotations(_raw, event_id=dict(T1=2, T2=3))
        _ep = mne.Epochs(_raw, _ev, dict(hands=2, feet=3), tmin=TMIN, tmax=TMAX,
                         picks="eeg", baseline=None, preload=True)
        return (_ep.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False),
                _ep.events[:, -1])


    print(f"\n--- SECONDARY: complement re-referenced within its own "
          f"{len(COMPLEMENT)} (ONE SPATIAL DIMENSION DELETED, not the leak removed) ---")
    X_ref, y_ref = build_reref_within(COMPLEMENT)
    assert np.array_equal(y_ref, labels), "re-referenced arm did not reproduce the labels"
    assert X_ref.shape[1] == len(COMPLEMENT)

    # WHAT THE MANIPULATION IS, MEASURED. Two facts, both checked here so neither has
    # to be taken on the author's word, and both printed so no reader meets the 6.0
    # points without meeting them.
    _X_comp_primary = cropped.copy().pick(COMPLEMENT).get_data(copy=False)
    _demeaned = _X_comp_primary - _X_comp_primary.mean(axis=1, keepdims=True)
    _reref_resid = float(np.abs(X_ref - _demeaned).max())
    _scale = float(np.abs(_X_comp_primary).max())
    _rank_primary = int(np.linalg.matrix_rank(
        _X_comp_primary.transpose(1, 0, 2).reshape(len(COMPLEMENT), -1)))
    _rank_ref = int(np.linalg.matrix_rank(
        X_ref.transpose(1, 0, 2).reshape(len(COMPLEMENT), -1)))
    print(f"IDENTITY CHECK: max |secondary - (primary - its own across-channel mean)| "
          f"= {_reref_resid:.2e}")
    print(f"      against a data scale of {_scale:.2e}, i.e. "
          f"{_reref_resid/_scale:.1e} relative. The secondary IS the primary")
    print(f"      with the uniform spatial direction removed. Nothing else changes.")
    print(f"RANK CHECK: rank drops {_rank_primary} -> {_rank_ref}. ONE spatial "
          f"dimension deleted, out of {len(COMPLEMENT)}.")
    assert _reref_resid / _scale < 1e-12, (
        "the secondary arm is NOT the primary minus its own across-channel mean, so "
        "the rank-1 description printed above is wrong and must be rewritten")
    assert _rank_ref == _rank_primary - 1, (
        f"expected a rank-1 projection ({_rank_primary} -> {_rank_primary - 1}), "
        f"got {_rank_primary} -> {_rank_ref}")
    print(f"WHAT THAT DIRECTION CARRIES: m64(x) - m47(x) = -(17/64)(m47(x) - m17(x)).")
    print(f"      It mixes the average-referenced strip contribution -(17/64)*m17 "
          f"WITH (17/64) of the")
    print(f"      complement's OWN global component m47. The cost below is therefore "
          f"NOT assignable")
    print(f"      to the strip leak alone, and this arm does not identify a causal "
          f"contribution.")
    ref_scores = cross_val_score(make_clf(), X_ref, labels,
                                 cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                    random_state=SEED),
                                 error_score="raise")
    ref_correct = int(round(ref_scores.mean() * n))
    assert abs(ref_scores.mean() - ref_correct / n) < 1e-9
    ref_vals = np.array([
        cross_val_score(make_clf(), X_ref, labels,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=s),
                        error_score="raise").mean()
        for s in SEEDS
    ])
    comp_seed42 = by_name[COMP][3]
    comp_sweep = sweeps[COMP]
    print(f"average ref over all 64, then pick 47 (PRIMARY): seed42 "
          f"{comp_seed42/n:.1%} ({comp_seed42}/{n})  10-seed {comp_sweep.mean():.1%} "
          f"[{comp_sweep.min():.1%}, {comp_sweep.max():.1%}]")
    print(f"average ref over the 47 alone   (SECONDARY): seed42 "
          f"{ref_correct/n:.1%} ({ref_correct}/{n})  10-seed {ref_vals.mean():.1%} "
          f"[{ref_vals.min():.1%}, {ref_vals.max():.1%}]")
    _ref_delta = 100 * (ref_vals.mean() - comp_sweep.mean())
    print(f"difference, secondary minus primary: "
          f"{_ref_delta:+.1f} points over ten seeds "
          f"(two trials = {NOISE_BAND:.3f} points, so anything inside that is nothing).")
    # THE PROJECT'S OWN EVIDENTIARY STANDARD, APPLIED TO THE ARM THAT WAS EXEMPT FROM
    # IT. Until 2026-07-26 this arm's 6.0 points was written up as established new
    # information ("removing the leak costs the complement 6.0 points, so the primary
    # complement score is inflated by the leak") while 14.7 points WITH a permutation
    # test behind it was written up as not established. That is two standards. The
    # rule this script already hard-codes for G is applied here instead.
    _ref_trials = abs(_ref_delta) * n / 100
    print(f"UNDER THE RULE THIS SCRIPT ALREADY HARD-CODES: {abs(_ref_delta):.1f} "
          f"points is {_ref_trials:.1f} trials,")
    print(f"      above the two-trial band ({NOISE_BAND:.3f}) and below the "
          f"{G_THRESHOLD:.1f}-point threshold, with NO")
    print(f"      confidence interval, NO significance test and NO registered "
          f"threshold of its own")
    print(f"      (the pre-registration registered this arm as a measurement and "
          f"registered no")
    print(f"      interpretation rule for it). So it is SUGGESTED, NOT ESTABLISHED, "
          f"on exactly the")
    print(f"      standard the {G_THRESHOLD:.1f}-point rule imposes on G. Any reading "
          f"stronger than that is")
    print(f"      post hoc. Any reading that assigns these points to the strip leak "
          f"specifically is")
    print(f"      also unidentified, per the decomposition printed above.")
    return comp_seed42, comp_sweep


def run_permutation(cropped, COMPLEMENT, labels, comp_seed42, n):
    # --- permutation test on the complement ---------------------------------------
    print(f"\n--- Permutation test on the complement ({len(COMPLEMENT)} ch), "
          f"{N_PERMUTATIONS} shuffles, seed {SEED} ---")
    X_comp = cropped.copy().pick(COMPLEMENT).get_data(copy=False)
    perm_obs, perm_null, perm_p = permutation_test_score(
        make_clf(), X_comp, labels, scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_permutations=N_PERMUTATIONS, random_state=SEED, n_jobs=-1,
    )
    assert abs(perm_obs * n - round(perm_obs * n)) < 1e-9
    assert abs(round(perm_obs * n) - comp_seed42) < 1e-9, (
        f"permutation_test_score's observed value ({round(perm_obs*n)}/{n}) disagrees "
        f"with cross_val_score's ({comp_seed42}/{n}) on identical folds")
    # sklearn computes p = (C + 1)/(n + 1), so 1/1001 is the RESOLUTION FLOOR of a
    # 1000-shuffle test and not a measurement. Printing "0.0010" invites reading the
    # limit of the instrument as a result. Same rule as decode_csp.py.
    p_floor = 1.0 / (N_PERMUTATIONS + 1)
    perm_p_str = (f"<= {p_floor:.3f}" if perm_p <= p_floor + 1e-12
                  else f"=  {perm_p:.4f}")
    print(f"observed {perm_obs:.1%} ({round(perm_obs*n)}/{n})   p {perm_p_str}")
    print(f"null mean {perm_null.mean():.1%} +/- {perm_null.std():.1%}, "
          f"max {perm_null.max():.1%}")
    null_off_50 = abs(100 * perm_null.mean() - 50.0)
    print(f"null centred {null_off_50:.1f} points off 50%, inside the registered 5. "
          f"{'PASS' if null_off_50 <= 5.0 else 'FIRED'}")
    assert null_off_50 <= 5.0, (
        f"permutation null centred at {perm_null.mean():.1%}, more than 5 points off "
        "50%. The null is mis-specified and the p does not mean what it looks like.")
    return perm_null, null_off_50


def run_wilson(comp_seed42, n, majority, maj_correct):
    # --- Wilson 95% CI on the complement's seed-42 count --------------------------
    w_lo, w_hi = wilson_interval(comp_seed42, n)
    print(f"\n--- Wilson 95% CI on the complement, z = 1.96 ---")
    print(f"{comp_seed42}/{n} = {comp_seed42/n:.1%}, Wilson [{w_lo:.1%}, {w_hi:.1%}], "
          f"width {100*(w_hi-w_lo):.1f} points, against the majority floor "
          f"{majority:.1%} ({maj_correct}/{n}).")
    print(f"An interval {100*(w_hi-w_lo):.1f} points wide is what n = {n} buys. It is "
          f"printed so the point estimate is never read alone.")


def run_paired_mcnemar(per_trial, ALL64, COMP, by_name, comp_seed42, n):
    # --- exact McNemar: all 64 against the complement, paired on the same folds ----
    # THE DECISION STATISTIC, and the blind half of it: nothing in the corpus carries
    # this comparison. Paired per-trial, because the two arms score the SAME 45
    # trials on the SAME seed-42 partition, and an unpaired comparison throws that
    # pairing away.
    print(f"\n--- Exact McNemar: all 64 vs. the complement, seed {SEED} folds, "
          f"paired per trial ---")
    c_all, c_comp = per_trial[ALL64], per_trial[COMP]
    both = int((c_all & c_comp).sum())
    only_all = int((c_all & ~c_comp).sum())          # b
    only_comp = int((~c_all & c_comp).sum())         # c
    neither = int((~c_all & ~c_comp).sum())
    n_disc = only_all + only_comp
    assert both + only_all + only_comp + neither == n
    assert both + only_all == by_name[ALL64][3] and both + only_comp == comp_seed42
    print(f"2x2: both correct {both}, all-64 only (b) {only_all}, "
          f"complement only (c) {only_comp}, neither {neither}. Sums to {n}.")
    if n_disc == 0:
        mcn_p = 1.0
        print("discordant pairs: 0. There is no test to run; p is 1.0 by definition.")
    else:
        mcn_p = float(stats.binomtest(only_all, n_disc, 0.5).pvalue)
        print(f"discordant pairs n_disc = {n_disc} ({only_all} vs {only_comp}), "
              f"exact two-sided p = {mcn_p:.4f}")
    # THE P IS QUOTED WITH ITS DISCORDANT COUNT OR NOT AT ALL. A56: at 10 discordant
    # pairs a one-trial shift moves the exact p from 0.109 to 0.754, a factor of about
    # seven. The line below is what this test could and could not have detected at the
    # n_disc it actually got, computed rather than asserted.
    worst = most_lopsided_failing_split(n_disc)
    if worst is None and n_disc > 0:
        print(f"At n_disc = {n_disc}, EVERY split reaches p < {ALPHA}. The test could "
              "not have returned a null result at this discordant count.")
    elif worst is not None:
        wb, wc, wp = worst
        print(f"POWER, at n_disc = {n_disc}: the most lopsided split that would STILL "
              f"have missed p < {ALPHA} is {wb} vs {wc} (p = {wp:.4f}). Anything less "
              f"lopsided than that is undetectable by this test at this n_disc.")
    else:
        print(f"At n_disc = 0 there is no split and no power to report.")

    # THE POWER LINE ABOVE CONDITIONS ON A RANDOM DRAW, AND THIS ONE DOES NOT.
    # n_disc = 8 is itself an outcome. The two MARGINALS, however, were known before
    # the pre-registration was written (they are printed in its section 0 and in its
    # arm table), and they fix b - c algebraically. So the set of 2x2 tables this
    # comparison could possibly have produced at seed 42 was enumerable in advance.
    print(f"MARGINAL-FORCED POWER, which conditions on nothing the run produced: at "
          f"seed {SEED} the two")
    print(f"      marginals are {by_name[ALL64][3]}/{n} and {comp_seed42}/{n}, so "
          f"b - c = {by_name[ALL64][3]} - {comp_seed42} = {by_name[ALL64][3]-comp_seed42} "
          f"is forced. Enumerating c:")
    _grid = forced_mcnemar_grid(by_name[ALL64][3], comp_seed42, n)
    for _b, _c, _nd, _p, _fires in _grid[:5]:
        print(f"        c = {_c}, b = {_b}, n_disc = {_nd}, p = {_p:.4f}"
              f"{'   FIRES' if _fires else ''}")
    _n_possible = sum(1 for r in _grid if r[4])
    print(f"      {_n_possible} of the {len(_grid)} attainable configurations reaches "
          f"p < {ALPHA}, and it is c = 0.")
    print(f"      The observed c = {only_comp}. So at these marginals the test was "
          f"near-predetermined to miss,")
    print(f"      and that was checkable with two lines of arithmetic at "
          f"prereg-writing time.")
    return both, only_all, only_comp, neither, n_disc, mcn_p, worst
