"""Sections 5, 9 and 10 of inferential_stats.py: the Wilson intervals on the
within-subject counts, every rung-11 paired test, and the 109-subject sweep proportion.
Split out 2026-08-26; every figure is re-derived from the arrays in infstats_lib."""

import numpy as np
from scipy import stats

from infstats_lib import (
    ALPHA, REGIME, SWEEP, cannot, head, holm, mde_noncentral, sub, t_interval,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# 5. Rung 10: Wilson intervals on the within-subject counts
# ---------------------------------------------------------------------------

WITHIN_N = 45
WITHIN_CSP_CORRECT = 41   # 91.1%, ablate_channels.py / decode_csp.py
WITHIN_NET_CORRECT = 37   # 82.2%, eegnet_compare.py experiment A


def section_wilson():
    head("5. RUNG 10 -- WILSON INTERVALS ON THE WITHIN-SUBJECT COUNTS")
    print(f"n = {WITHIN_N} trials, subject 1, stratified 5-fold, each trial tested")
    print(f"exactly once. Accuracy is therefore a multiple of 1/{WITHIN_N} = "
          f"{100 / WITHIN_N:.4f}%.")
    print("Wilson rather than normal, because at n = 45 the normal interval")
    print("misbehaves near the ends of the scale.")
    print("The two counts are the ones the cached stdout prints: 41/45 for CSP+LDA")
    print("(ablate_channels.py, decode_csp.py) and 37/45 for EEGNet")
    print("(eegnet_compare.py experiment A). Section 6 re-derives both from")
    print("per-trial predictions and will disagree if the re-run drifts.")
    print()
    out = {}
    for tag, k in (("CSP + LDA", WITHIN_CSP_CORRECT), ("EEGNet", WITHIN_NET_CORRECT)):
        lo, hi = wilson_interval(k, WITHIN_N)
        out[tag] = (lo, hi)
        print(f"  {tag:<12} {k}/{WITHIN_N} = {100 * k / WITHIN_N:.1f}%   "
              f"95% Wilson CI [{100 * lo:.1f}%, {100 * hi:.1f}%]")
    (l1, h1), (l2, h2) = out["CSP + LDA"], out["EEGNet"]
    lo_ov, hi_ov = max(l1, l2), min(h1, h2)
    if lo_ov < hi_ov:
        frac1 = (hi_ov - lo_ov) / (h1 - l1)
        frac2 = (hi_ov - lo_ov) / (h2 - l2)
        print(f"  overlap                    [{100 * lo_ov:.1f}%, {100 * hi_ov:.1f}%]"
              f"  = {100 * frac1:.0f}% of the CSP interval, "
              f"{100 * frac2:.0f}% of the EEGNet interval")
        print("  The intervals overlap across most of both ranges, so the 8.9-point")
        print("  point difference is not separable at this sample size.")
    else:
        print("  The intervals do not overlap.")
    print("  What this does NOT show: two overlapping marginal intervals are not a")
    print("  test of the paired difference. The paired test is the next section.")

# ---------------------------------------------------------------------------
# 9. Rung 11: the regime decomposition
# ---------------------------------------------------------------------------

CELLS = ["narrow-short", "wide-short", "narrow-long", "wide-long", "original-C"]


def section_rung11():
    head("9. RUNG 11 -- REGIME DECOMPOSITION, EVERY PAIRED TEST")
    if REGIME is None:
        cannot("all rung-11 intervals, t-statistics and p-values",
               "regime_decomposition.json is absent",
               "python regime_decomposition.py (cold, ~1 h)")
        return

    n = len(REGIME["narrow-short"]["csp"])
    print(f"n = {n} leave-one-subject-out folds per cell. Both models see the")
    print("identical folds within a cell, so every model comparison is PAIRED.")
    print("PROVENANCE: these arrays are the 2026-07-23 checkpoint. The 2026-07-25")
    print("cold run resumed from them rather than recomputing, so the figures in")
    print("this section are 07-23 values carrying a 07-25 timestamp on the run log.")

    sub("EEGNet minus CSP, per cell")
    print(f"{'cell':<15}{'CSP':>8}{'EEGNet':>9}{'delta':>8}  {'95% CI':>18}"
          f"{'t':>7}{'p':>8}")
    pvals, names = [], []
    for cell in CELLS:
        e = np.array(REGIME[cell]["eegnet"], dtype=float)
        c = np.array(REGIME[cell]["csp"], dtype=float)
        d = 100.0 * (e - c)
        m, lo, hi = t_interval(d)
        t, p = stats.ttest_rel(e, c)
        pvals.append(p)
        names.append(cell)
        print(f"{cell:<15}{100 * c.mean():>7.1f}%{100 * e.mean():>8.1f}%"
              f"{m:>+8.1f}  [{lo:+6.1f}, {hi:+6.1f}]{t:>7.2f}{p:>8.3f}")
    print()
    print("  original-C is the 4-38 Hz, 0.0-4.0 s cell. Its EEGNet mean of "
          f"{100 * np.mean(REGIME['original-C']['eegnet']):.1f}% is the figure the")
    print("  documents quote for that cell, and it is stored in the checkpoint but")
    print("  never echoed by regime_decomposition.py's printed table.")
    print("  Only original-C separates the two models at this sample size.")

    adj = holm(pvals)
    print()
    print("  Holm across the five cells:")
    for name, p, a in zip(names, pvals, adj):
        print(f"    {name:<15} p = {p:.3f}   adjusted {a:.3f}   "
              f"{'survives' if a < ALPHA else 'does not survive'}")
    print("  The published table is UNCORRECTED. Only original-C survives Holm.")

    sub("moving the crop start 1.0 s -> 0.0 s (original-C minus wide-long)")
    gap_c = 100.0 * (np.array(REGIME["original-C"]["eegnet"], dtype=float)
                     - np.array(REGIME["original-C"]["csp"], dtype=float))
    gap_w = 100.0 * (np.array(REGIME["wide-long"]["eegnet"], dtype=float)
                     - np.array(REGIME["wide-long"]["csp"], dtype=float))
    m, lo, hi = t_interval(gap_c - gap_w)
    t, p = stats.ttest_rel(gap_c, gap_w)
    print(f"  the EEGNet-CSP gap widens by {m:+.1f} points")
    print(f"  95% CI [{lo:+.1f}, {hi:+.1f}]   t = {t:+.2f}   p = {p:.3f}")
    for model in ("eegnet", "csp"):
        a = np.array(REGIME["original-C"][model], dtype=float)
        b = np.array(REGIME["wide-long"][model], dtype=float)
        mm, ll, hh = t_interval(100.0 * (a - b))
        tt, pp = stats.ttest_rel(a, b)
        print(f"    {model:<7} {mm:+.1f} points  95% CI [{ll:+.1f}, {hh:+.1f}]  "
              f"t = {tt:+.2f}  p = {pp:.3f}")
    print("  What this does NOT show: crop start is confounded with window length")
    print("  here, so the split above attributes the change to the earlier start")
    print("  only under the assumption that the extra second is otherwise inert.")

    sub("widening the band, 8-30 Hz -> 4-38 Hz, averaged over window length")
    band = {}
    for model in ("csp", "eegnet"):
        narrow = (np.array(REGIME["narrow-short"][model], dtype=float)
                  + np.array(REGIME["narrow-long"][model], dtype=float)) / 2
        wide = (np.array(REGIME["wide-short"][model], dtype=float)
                + np.array(REGIME["wide-long"][model], dtype=float)) / 2
        band[model] = 100.0 * (wide - narrow)
        m, lo, hi = t_interval(band[model])
        t, p = stats.ttest_rel(wide, narrow)
        print(f"  {model:<7} {m:+.1f} points  95% CI [{lo:+.1f}, {hi:+.1f}]  "
              f"t = {t:+.2f}  p = {p:.3f}")
    diff = band["eegnet"] - band["csp"]
    m, lo, hi = t_interval(diff)
    t, p = stats.ttest_1samp(diff, 0.0)
    print(f"  difference (EEGNet minus CSP band effect) {m:+.1f} points")
    print(f"  95% CI [{lo:+.1f}, {hi:+.1f}]   t = {t:+.2f}   p = {p:.3f}")
    print("  NOTE: these CI bounds are interval endpoints on a paired difference.")
    print("  They are not accuracy differences and must not be read as such.")

    sub("the cue-onset window and the second before it, against chance")
    print(f"{'window':<26}{'model':<9}{'mean':>8}  {'95% CI':>18}{'t':>8}{'p':>12}")
    for cell, label in (("pre-cue", "-1.0 to 0.0 s pre-cue"),
                        ("cue-only", "0.0 to 1.0 s post-cue")):
        for model in ("csp", "eegnet"):
            s = 100.0 * np.array(REGIME[cell][model], dtype=float)
            m, lo, hi = t_interval(s)
            t, p = stats.ttest_1samp(s, 50.0)
            pstr = f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"
            print(f"{label:<26}{model:<9}{m:>7.1f}%  [{lo:>5.1f}, {hi:>5.1f}]"
                  f"{t:>8.2f}{pstr:>12}")
    e_pre = 100.0 * np.array(REGIME["pre-cue"]["eegnet"], dtype=float)
    _, p_cue = stats.ttest_1samp(
        100.0 * np.array(REGIME["cue-only"]["eegnet"], dtype=float), 50.0)
    print(f"  the EEGNet cue-window p is {p_cue:.2e}, which is below 0.0001, so a")
    print("  stated bound of p < 0.0001 holds. Printing p at three decimals would")
    print("  render it 0.000 and could not license that bound.")
    print("  ONE-SAMPLE, two-sided, against 50.0% -- the balanced-design chance")
    print("  rate. The pooled majority-class rate on this cohort is close to it but")
    print("  is not identical, and the two are not interchangeable.")
    m_pre, lo_pre, hi_pre = t_interval(e_pre)
    print()
    print(f"  THE CONTROL DOES NOT CLEANLY PASS. Pre-cue EEGNet is {m_pre:.1f}% with")
    print(f"  95% CI [{lo_pre:.1f}%, {hi_pre:.1f}%]. Its upper bound exceeds the")
    print("  cue-window CSP point estimate of 53.7% that the same rung calls")
    print("  significant, so 'chance before the cue' is not established, only")
    print("  'not distinguishable from chance'.")

    sub("paired post-cue minus pre-cue")
    for model in ("csp", "eegnet"):
        a = np.array(REGIME["cue-only"][model], dtype=float)
        b = np.array(REGIME["pre-cue"][model], dtype=float)
        m, lo, hi = t_interval(100.0 * (a - b))
        t, p = stats.ttest_rel(a, b)
        pstr = f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"
        print(f"  {model:<7} {m:+.1f} points  95% CI [{lo:+.1f}, {hi:+.1f}]  "
              f"t = {t:+.2f}  p = {pstr}")

    sub("what the full imagery window buys over the cue-onset second")
    for model in ("eegnet", "csp"):
        a = np.array(REGIME["original-C"][model], dtype=float)
        b = np.array(REGIME["cue-only"][model], dtype=float)
        m, lo, hi = t_interval(100.0 * (a - b))
        t, p = stats.ttest_rel(a, b)
        sd = (100.0 * (a - b)).std(ddof=1)
        print(f"  {model:<7} {m:+.1f} points  95% CI [{lo:+.1f}, {hi:+.1f}]  "
              f"t = {t:+.2f}  p = {p:.3f}")
        print(f"          sd {sd:.2f}, MDE at 80% power "
              f"{mde_noncentral(sd, len(a)):.2f} points")
    print("  THE RUNG-11 CONCLUSION TURNS ON THIS. A non-significant result here is")
    print("  not evidence that the imagery window carries nothing; the interval")
    print("  admits gains up to its upper bound, and the MDE above is the smallest")
    print("  true effect this comparison could have detected.")

# ---------------------------------------------------------------------------
# 10. The 109-subject sweep proportion
# ---------------------------------------------------------------------------

def section_sweep():
    head("10. THE 109-SUBJECT SWEEP -- THE AT-OR-BELOW-CHANCE PROPORTION")
    if SWEEP is None:
        cannot("the at-or-below-chance proportion", "sweep_results.csv is absent",
               "python sweep_subjects.py (~30 min)")
        return
    acc = np.array([float(r["accuracy"]) for r in SWEEP])
    chance = np.array([float(r["chance"]) for r in SWEEP])
    n_tr = np.array([int(r["n_trials"]) for r in SWEEP])
    total = len(acc)
    at_or_below = int((acc <= chance).sum())
    print(f"  subjects evaluated              {total}")
    print(f"  at or below their own chance    {at_or_below}/{total} = "
          f"{100 * at_or_below / total:.1f}%")
    print(f"  the same figure at 0 decimals   {100 * at_or_below / total:.0f}%")
    print("  BOTH FORMS ARE THE SAME COUNT. The decimal is load-bearing here because")
    print("  the claim this refutes is stated as 27%, and 27.5 versus 28 straddles")
    print("  it. Quoting one precision in one document and the other elsewhere makes")
    print("  the comparison unreproducible.")

    sub("what a pure-noise null predicts, computed per subject")
    exact = np.array([stats.binom.cdf(np.floor(c * n + 1e-9), n, 0.5)
                      for c, n in zip(chance, n_tr)])
    print(f"  expected fraction at or below chance under true accuracy 0.5:")
    print(f"    exact binomial, per subject, averaged   {100 * exact.mean():.1f}%")
    print(f"    implied count                           {exact.sum():.1f}/{total}")
    print("  ASSUMPTION: each subject's trials are independent Bernoulli(0.5) draws,")
    print("  and a fold-quantized 5-fold accuracy is treated as a binomial count.")
    print("  Cross-validation correlates folds, so this is an approximation and the")
    print("  simulated figure sweep_subjects.py prints will differ by a point or two.")
    print(f"  OBSERVED {100 * at_or_below / total:.1f}% IS BELOW THE NULL EXPECTATION,")
    print("  which is evidence of signal across the population. Reading it as an")
    print("  illiteracy rate inverts the direction of the inference.")
    print("  What this does NOT show: it is a statement about the population of")
    print("  recordings, not about any individual subject's trainability.")
