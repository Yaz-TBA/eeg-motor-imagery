"""Sections 6 and 7 of emg_proxy.py: arm (a), the univariate tests on log
high-band power, and the pre-cue diagnostic with its truncated-segment
contingency. Split out 2026-08-26; the bodies are verbatim from that file."""

import numpy as np
import mne
from scipy import stats

from common import TEMPORAL, holm
from emg_setup import ALPHA, PRECUE_CROP, hr, sub


def run_univariate(D):
    get_data, EPOCHS, labels = D.get_data, D.EPOCHS, D.labels
    N_HANDS, N_FEET, CHANNEL_SETS = D.N_HANDS, D.N_FEET, D.CHANNEL_SETS

    # =============================================================================
    # 6. Arm (a): univariate
    # =============================================================================
    hr("6. ARM (a), UNIVARIATE: DOES LOG HIGH-BAND POWER DIFFER BY CLASS?")

    print("\nPer trial per channel: mean band power over the feature window, then log.")
    print("Log because raw band power is right-skewed and heteroscedastic in a way")
    print("that scales with the class mean, which is the same argument this repo")
    print("already makes for log=True inside CSP.")
    print("\nTwo-sided, because there is no defensible directional prior. The cue is a")
    print("bar at the TOP of the screen for fists and the BOTTOM for feet, which gives")
    print("a class-dependent gaze direction but no prediction about which direction")
    print("produces more high-band temporal power.")
    print("\nBOTH tests must agree for a positive call. Disagreement is reported as")
    print("outlier-trial-driven, which is itself the realistic EMG failure mode: a few")
    print("trials with a clench, not a shifted distribution.")

    MDE_AGGREGATE = 0.837   # Cohen's d detectable at 80% power, 21 vs 24, alpha 0.05
    MDE_PERCHANNEL = 1.069  # same, at the Bonferroni floor alpha 0.05/8 = 0.00625
    print(f"\nPRE-COMPUTED POWER, so a null cannot be oversold. With {N_HANDS} vs "
          f"{N_FEET} trials at 80% power:")
    print(f"  minimum detectable Cohen's d = {MDE_AGGREGATE:.3f} for the aggregate test "
          f"at alpha {ALPHA}")
    print(f"  minimum detectable Cohen's d = {MDE_PERCHANNEL:.3f} for a single channel "
          f"at alpha {ALPHA / 8:.5f}")
    print("Both are LARGE effects. A null on arm (a) bounds large effects only.")


    def log_power(band, picks, crop=None):
        """(n_trials, n_ch) log mean-square power over the window."""
        if crop is None:
            arr = get_data(band, picks)
        else:
            ep = EPOCHS[band].copy().crop(tmin=crop[0], tmax=crop[1])
            idx = [ep.ch_names.index(c) for c in picks]
            arr = ep.get_data(copy=False)[:, idx, :]
        return np.log(np.mean(arr ** 2, axis=-1))


    def cohens_d(a, b):
        na, nb = len(a), len(b)
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        return (a.mean() - b.mean()) / sp


    def two_tests(vals):
        a, b = vals[labels == 2], vals[labels == 3]
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        return t_p, u_p, cohens_d(a, b)


    # holm comes from common.py. test_pipeline.py checks it against the textbook result.

    sub("Aggregate: mean log power across the 8 TEMPORAL channels, primary band")
    lp_temporal = log_power("PRIMARY", TEMPORAL)
    agg = lp_temporal.mean(axis=1)
    agg_t_p, agg_u_p, agg_d = two_tests(agg)
    raw_pow = np.mean(get_data("PRIMARY", TEMPORAL) ** 2, axis=-1).mean(axis=1)
    med_ratio = np.median(raw_pow[labels == 2]) / np.median(raw_pow[labels == 3])
    print(f"  Welch t two-sided p = {agg_t_p:.4f}")
    print(f"  Mann-Whitney U  p   = {agg_u_p:.4f}")
    print(f"  Cohen's d (hands minus feet, log power) = {agg_d:+.3f}   "
          f"(detectable floor {MDE_AGGREGATE:.3f})")
    print(f"  median raw power ratio hands/feet = {med_ratio:.3f}")
    ARM_A_POSITIVE = (agg_t_p < ALPHA) and (agg_u_p < ALPHA)
    ARM_A_DISAGREE = (agg_t_p < ALPHA) != (agg_u_p < ALPHA)
    if ARM_A_POSITIVE:
        print("  BOTH tests significant. Arm (a) aggregate is POSITIVE.")
    elif ARM_A_DISAGREE:
        print("  THE TWO TESTS DISAGREE. Reported as outlier-trial-driven, per the")
        print("  pre-registration. Not called a positive.")
    else:
        print("  Neither test significant. Arm (a) aggregate is NULL.")

    sub("Per channel, TEMPORAL ring, Holm-Bonferroni across 8 at family alpha 0.05")
    per_t = np.array([two_tests(lp_temporal[:, i])[0] for i in range(len(TEMPORAL))])
    per_u = np.array([two_tests(lp_temporal[:, i])[1] for i in range(len(TEMPORAL))])
    per_d = np.array([two_tests(lp_temporal[:, i])[2] for i in range(len(TEMPORAL))])
    holm_t, holm_u = holm(per_t), holm(per_u)
    print(f"  {'ch':<5} {'t p raw':>9} {'t p holm':>9} {'U p raw':>9} {'U p holm':>9} "
          f"{'d':>7}")
    for i, ch in enumerate(TEMPORAL):
        flag = "  *" if (holm_t[i] < ALPHA and holm_u[i] < ALPHA) else ""
        print(f"  {ch:<5} {per_t[i]:>9.4f} {holm_t[i]:>9.4f} {per_u[i]:>9.4f} "
              f"{holm_u[i]:>9.4f} {per_d[i]:>+7.3f}{flag}")
    n_sig = int(((holm_t < ALPHA) & (holm_u < ALPHA)).sum())
    print(f"  {n_sig} of {len(TEMPORAL)} channels significant on both tests after Holm.")

    sub("Same aggregate test on the other three channel sets, DESCRIPTIVE ONLY")
    print(f"  {'set':<14} {'t p':>9} {'U p':>9} {'d':>8}")
    for name, picks in CHANNEL_SETS:
        v = log_power("PRIMARY", picks).mean(axis=1)
        t_p, u_p, d = two_tests(v)
        print(f"  {name:<14} {t_p:>9.4f} {u_p:>9.4f} {d:>+8.3f}")

    return (agg_t_p, agg_u_p, agg_d, ARM_A_POSITIVE, MDE_AGGREGATE,
            MDE_PERCHANNEL, log_power, two_tests, cohens_d)


def run_precue(D, log_power, two_tests):
    raw_base, SFREQ, ALL_CH, EPOCHS = D.raw_base, D.SFREQ, D.ALL_CH, D.EPOCHS
    BANDS, N, CASCADE_HALF_S = D.BANDS, D.N, D.CASCADE_HALF_S

    # =============================================================================
    # 7. Pre-cue diagnostic
    # =============================================================================
    hr("7. PRE-CUE DIAGNOSTIC: IS THERE CLASS-CORRELATED HIGH-BAND POWER BEFORE THE CUE?")

    print(f"\nThe same aggregate test on the {PRECUE_CROP[0]:.1f} to {PRECUE_CROP[1]:.1f} s "
          f"window. Anything found here")
    print("cannot be imagery, because imagery has not started.")
    print(f"\nCAVEAT, RECOMPUTED FROM THE REALISED CASCADE HALF-LENGTH OF "
          f"{CASCADE_HALF_S:.3f} s:")
    print(f"  a sample at time t draws from [t - {CASCADE_HALF_S:.3f}, "
          f"t + {CASCADE_HALF_S:.3f}], so a pre-cue")
    print(f"  sample is filter-clean only if t <= -{CASCADE_HALF_S:.3f} s. The window "
          f"starts at {PRECUE_CROP[0]:.1f} s.")
    print(f"  NONE of this window is filter-clean in the primary band. The")
    print("  pre-registration predicted 0.175 s of clean window from a 0.825 s")
    print("  half-length; that figure describes a SINGLE filter, not the cascade.")
    print("  So a null here is uninformative and only a strong positive means anything,")
    print("  and either way the contingency re-run below is what actually arbitrates.")

    lp_pre = log_power("PRIMARY", TEMPORAL, crop=PRECUE_CROP)
    pre_agg = lp_pre.mean(axis=1)
    pre_t_p, pre_u_p, pre_d = two_tests(pre_agg)
    print(f"\n  Welch t two-sided p = {pre_t_p:.4f}")
    print(f"  Mann-Whitney U  p   = {pre_u_p:.4f}")
    print(f"  Cohen's d = {pre_d:+.3f}")
    PRECUE_SIG = (pre_t_p < ALPHA) and (pre_u_p < ALPHA)

    if PRECUE_SIG:
        print("\n  SIGNIFICANT. Triggering the pre-registered contingency: rebuild the")
        print("  window from raw segments that PHYSICALLY END at t = 0 before filtering,")
        print("  so no post-cue sample exists to leak backwards. This is the remedy")
        print("  regime_decomposition.py:174-177 already established for this problem.")
        b = BANDS["PRIMARY"]
        onsets = EPOCHS["PRIMARY"].events[:, 0]
        seg_len = int(4.0 * SFREQ)  # 4 s back from the cue, long enough for a 441-tap FIR
        win_len = int(round((PRECUE_CROP[1] - PRECUE_CROP[0]) * SFREQ)) + 1
        idx = [ALL_CH.index(c) for c in TEMPORAL]
        raw_arr = raw_base.get_data(picks=idx)
        segs = []
        for o in onsets:
            s0 = o - raw_base.first_samp - seg_len
            s1 = o - raw_base.first_samp
            segs.append(raw_arr[:, s0:s1])
        segs = np.array(segs)
        flat = segs.reshape(-1, seg_len)
        flat = mne.filter.filter_data(
            flat, SFREQ, b["l_freq"], b["h_freq"],
            l_trans_bandwidth=b["l_trans"], h_trans_bandwidth=b["h_trans"],
            method="fir", fir_design="firwin", verbose="error",
        )
        flat = mne.filter.notch_filter(
            flat, SFREQ, np.array([60.0]), notch_widths=2.0, trans_bandwidth=6.0,
            method="fir", fir_design="firwin", verbose="error",
        )
        segs = flat.reshape(N, len(TEMPORAL), seg_len)[:, :, -win_len:]
        trunc_agg = np.log(np.mean(segs ** 2, axis=-1)).mean(axis=1)
        tr_t_p, tr_u_p, tr_d = two_tests(trunc_agg)
        print(f"\n  Truncated-segment re-run ({seg_len / SFREQ:.1f} s ending exactly at "
              f"t = 0, last {win_len} samples used):")
        print(f"    Welch t two-sided p = {tr_t_p:.4f}")
        print(f"    Mann-Whitney U  p   = {tr_u_p:.4f}")
        print(f"    Cohen's d = {tr_d:+.3f}")
        PRECUE_SURVIVES = (tr_t_p < ALPHA) and (tr_u_p < ALPHA)
        if PRECUE_SURVIVES:
            print("\n    IT SURVIVES. The finding is not post-cue leakage. Per the")
            print("    pre-registration this is block structure, drift or a labelling")
            print("    artifact, and it damages considerably more than the EMG claim.")
        else:
            print("\n    It does NOT survive. The pre-cue result was filter smear from")
            print("    post-cue samples, which is exactly what the contingency exists to")
            print("    test. No pre-cue claim is made.")
    else:
        PRECUE_SURVIVES = False
        print("\n  Not significant. Per the pre-registration this null is UNINFORMATIVE,")
        print("  because the window is entirely filter-contaminated. It is not evidence")
        print("  of absence and the contingency is not triggered.")
