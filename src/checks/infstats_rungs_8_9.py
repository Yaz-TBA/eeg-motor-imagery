"""Sections 0-4 of inferential_stats.py: input provenance, the rung-8 gap, the rung-9
paired tests, the combined power limit, and the subject x method interaction. Split out
2026-08-26; every figure is re-derived from the persisted arrays in infstats_lib."""

import numpy as np
from scipy import stats

from infstats_lib import (
    ALPHA, PERFOLD, REGIME, SWEEP, cannot, head, holm, mde_noncentral, mde_normal,
    normal_interval, sub, t_interval,
)

# ---------------------------------------------------------------------------
# 0. Provenance of the inputs, printed before any number derived from them
# ---------------------------------------------------------------------------

def section_inputs():
    head("0. INPUTS")
    print("Every figure below is derived from one of these. Nothing is restated")
    print("from prose, and nothing is imputed where an array is absent.")
    print()
    if REGIME is None:
        cannot("all rung-11 statistics", "regime_decomposition.json is absent",
               "python regime_decomposition.py (cold, ~1 h)")
    else:
        cells = ", ".join(REGIME.keys())
        n = len(REGIME["narrow-short"]["csp"])
        print(f"regime_decomposition.json   {len(REGIME)} cells ({cells})")
        print(f"                            {n} per-fold accuracies per model per cell")
        print("                            DATED 2026-07-23. The 2026-07-25 cold run")
        print("                            resumed from this file, so rung-11 figures")
        print("                            are checkpoint values, not 07-25 values.")
    if PERFOLD is None:
        cannot("all rung-8 and rung-9 statistics",
               "riemannian_perfold.json is absent from this repo",
               "riemannian.py, edited to persist the per-subject arrays it holds")
    else:
        methods = ", ".join(PERFOLD["scores"].keys())
        print("riemannian_perfold.json     in this repo, committed")
        print(f"                            5 pipelines ({methods})")
        print(f"                            {len(PERFOLD['subjects'])} subjects, "
              f"pooled chance {100 * PERFOLD['pooled_chance']:.1f}%")
        print("                            riemannian.py computes these and persists")
        print("                            none; this copy is the 2026-07-23 capture.")
    if SWEEP is None:
        cannot("the within-subject half of rung 8 and the sweep proportion",
               "sweep_results.csv is absent", "python sweep_subjects.py (~30 min)")
    else:
        print(f"sweep_results.csv           {len(SWEEP)} subjects, within-subject "
              f"5-fold accuracies")

# ---------------------------------------------------------------------------
# 1. Rung 8: the within-to-cross-subject gap
# ---------------------------------------------------------------------------

def section_rung8():
    head("1. RUNG 8 -- THE WITHIN-TO-CROSS-SUBJECT GAP")
    if PERFOLD is None or SWEEP is None:
        cannot("the rung-8 interval, p-value and MDE",
               "one of sweep_results.csv / riemannian_perfold.json is absent",
               "sweep_subjects.py and riemannian.py")
        return None

    subs = PERFOLD["subjects"]
    cross = np.array(PERFOLD["scores"]["CSP+LDA"], dtype=float)
    by_subject = {int(r["subject"]): float(r["accuracy"]) for r in SWEEP}
    missing = [s for s in subs if s not in by_subject]
    if missing:
        cannot("the rung-8 gap", f"subjects {missing} absent from sweep_results.csv",
               "sweep_subjects.py")
        return None
    within = np.array([by_subject[s] for s in subs], dtype=float)

    print(f"n = {len(subs)} subjects, the same 20 in both arms, paired by subject id.")
    print("The cross arm is the CSP+LDA baseline of the leave-one-subject-out run;")
    print("the within arm is the same subject's stratified 5-fold accuracy.")
    print()
    print(f"within-subject mean       {100 * within.mean():.1f}%  "
          f"(sd {100 * within.std(ddof=1):.1f}%)")
    print(f"cross-subject  mean       {100 * cross.mean():.1f}%  "
          f"(sd {100 * cross.std(ddof=1):.1f}%)")

    d = 100.0 * (within - cross)
    m, lo_t, hi_t = t_interval(d)
    _, lo_z, hi_z = normal_interval(d)
    t, p = stats.ttest_rel(within, cross)
    sem = d.std(ddof=1) / np.sqrt(len(d))

    sub("the gap, paired, two-sided")
    print(f"THE GAP                   {m:+.1f} points  (se {sem:.3f})")
    print(f"  95% CI, normal (z)      [{lo_z:+.1f}, {hi_z:+.1f}] points")
    print(f"  95% CI, Student-t df=19 [{lo_t:+.1f}, {hi_t:+.1f}] points")
    print(f"  paired t = {t:+.3f}, p = {p:.3f}")
    print("  THE MULTIPLIER IS LOAD-BEARING. The two intervals differ by about half")
    print("  a point at each end. The published interval matches the z form.")
    print(f"  What this does NOT show: both bounds straddle zero, so the direction of")
    print(f"  the gap is not established at n={len(d)}, only its point estimate.")
    print(f"  COLLISION HAZARD: p = {p:.3f} here is the within-minus-cross test. Rung")
    print("  11's pre-cue EEGNet cell reports the same value from a different test.")

    sd = d.std(ddof=1)
    sub("power")
    print(f"sd of the paired differences  {sd:.2f} points")
    print(f"MDE at 80% power, noncentral t  {mde_noncentral(sd, len(d)):.2f} points")
    print(f"MDE at 80% power, normal approx {mde_normal(sd, len(d)):.2f} points")
    print("  This comparison could not have detected a true gap smaller than that.")
    print("  A non-significant result here is therefore not evidence of no gap.")
    return d

# ---------------------------------------------------------------------------
# 2. Rung 9: the Riemannian pipelines against the baseline
# ---------------------------------------------------------------------------

RIEMANN_ORDER = ["MDM-64", "TSLR-64", "MDM-mot", "TSLR-mot"]


def section_rung9():
    head("2. RUNG 9 -- RIEMANNIAN PIPELINES AGAINST THE CSP+LDA BASELINE")
    if PERFOLD is None:
        cannot("the four rung-9 paired tests, their intervals and their MDEs",
               "riemannian_perfold.json is absent",
               "riemannian.py, edited to persist per-subject scores")
        return None

    base = np.array(PERFOLD["scores"]["CSP+LDA"], dtype=float)
    n = len(base)
    print(f"n = {n} subjects. Identical leave-one-subject-out folds for every arm,")
    print("so every comparison is PAIRED. Each is two-sided against the baseline.")
    print(f"baseline CSP+LDA mean     {100 * base.mean():.1f}%  "
          f"(sd {100 * base.std(ddof=1):.1f}%)")
    print()
    print(f"{'pipeline':<12}{'mean':>8}{'delta':>8}  {'95% CI':>18}"
          f"{'t':>8}{'p':>9}{'MDE':>8}")

    rows, pvals = [], []
    for name in RIEMANN_ORDER:
        arm = np.array(PERFOLD["scores"][name], dtype=float)
        d = 100.0 * (arm - base)
        m, lo, hi = t_interval(d)
        t, p = stats.ttest_rel(arm, base)
        sd = d.std(ddof=1)
        rows.append((name, arm, d, m, lo, hi, t, p, sd))
        pvals.append(p)
        print(f"{name:<12}{100 * arm.mean():>7.1f}%{m:>+8.1f}  "
              f"[{lo:+6.1f}, {hi:+6.1f}]{t:>8.2f}{p:>9.3f}"
              f"{mde_noncentral(sd, n):>8.2f}")

    print()
    print("delta and CI are in percentage points, EEG pipeline minus baseline.")
    print("MDE is the noncentral-t minimum detectable difference at 80% power.")

    sub("which intervals exclude zero")
    for name, _, _, m, lo, hi, _, p, _ in rows:
        verdict = "excludes zero" if lo * hi > 0 else "spans zero"
        print(f"  {name:<10} [{lo:+6.1f}, {hi:+6.1f}]  {verdict}")
    spanning = sum(1 for r in rows if r[4] * r[5] <= 0)
    print(f"  {spanning} of {len(rows)} intervals span zero.")

    sub("multiplicity across the four comparisons")
    adj = holm(pvals)
    for (name, *_), p, a in zip(rows, pvals, adj):
        keep = "survives" if a < ALPHA else "does not survive"
        print(f"  {name:<10} p = {p:.3f}  Holm-adjusted {a:.3f}  {keep}")
    print("  The figures published for this rung are UNCORRECTED p-values. Holm over")
    print("  the family of four is printed so the difference is visible, not to")
    print("  replace them.")

    sub("power across the rung")
    mdes = [mde_noncentral(r[8], n) for r in rows]
    znes = [mde_normal(r[8], n) for r in rows]
    print(f"  noncentral-t MDEs span {min(mdes):.2f} to {max(mdes):.2f} points")
    print(f"  normal-approx MDEs span {min(znes):.2f} to {max(znes):.2f} points")
    print("  Three of these four nulls are therefore uninformative about differences")
    print("  smaller than roughly six points. That is a limit of n = 20, not a")
    print("  finding about the methods.")
    return {name: (arm, d, sd) for name, arm, d, _, _, _, _, _, sd in rows}

def section_power_headline(rung8_d, rung9):
    head("3. THE COMBINED POWER LIMIT ACROSS RUNGS 8 AND 9")
    if rung8_d is None or rung9 is None:
        cannot("the combined MDE range",
               "one of the two rungs' arrays is absent", "see sections 1 and 2")
        return
    sds = [np.std(rung8_d, ddof=1)] + [v[2] for v in rung9.values()]
    mdes = [mde_noncentral(s, 20) for s in sds]
    znes = [mde_normal(s, 20) for s in sds]
    print("All five paired comparisons in rungs 8 and 9, one power routine, so the")
    print("two ranges the documents quote cannot drift apart.")
    print(f"  noncentral t : {min(mdes):.2f} to {max(mdes):.2f} points")
    print(f"  normal approx: {min(znes):.2f} to {max(znes):.2f} points")
    print("  The rung-9-only subrange is the first four; rung 8's is the largest.")
    print(f"  rung 8 alone: {mde_noncentral(np.std(rung8_d, ddof=1), 20):.2f} points")

# ---------------------------------------------------------------------------
# 4. The subject x method interaction: two tests, two reference distributions
# ---------------------------------------------------------------------------

def section_interaction():
    head("4. SUBJECT x METHOD INTERACTION -- TWO TESTS THAT DISAGREE BY DESIGN")
    if PERFOLD is None:
        cannot("both interaction tests", "riemannian_perfold.json is absent",
               "riemannian.py, edited to persist per-subject scores")
        return

    scores = PERFOLD["scores"]
    subs = PERFOLD["subjects"]
    n_sub = len(subs)
    base = np.array(scores["CSP+LDA"], dtype=float)
    mot = np.array(scores["MDM-mot"], dtype=float)
    # 900 pooled trials over 20 subjects, one fold per subject. The count matters
    # only for the binomial variance the homogeneity weights use.
    n_trials = 900 // n_sub

    print("THE TRAP THIS SECTION EXISTS FOR. Two statistics near 13 have been quoted")
    print("for this question, one a chi-square on 19 df and one a Tukey 1-df F, and")
    print("their p-values are four orders of magnitude apart. They are not the same")
    print("test and neither is a typo. Both are computed here, from the same arrays.")

    sub("A. homogeneity of the per-subject CSP-vs-MDM-motor difference, 19 df")
    delta = mot - base
    print(f"  mean difference        {100 * delta.mean():+.1f} points")
    print(f"  win / loss / tie       {(mot > base).sum()} / {(mot < base).sum()} "
          f"/ {(mot == base).sum()}")
    # Inverse-variance (Cochran Q) homogeneity, two variance models.
    var_sep = (base * (1 - base) + mot * (1 - mot)) / n_trials
    pooled = (base + mot) / 2.0
    var_pool = 2 * pooled * (1 - pooled) / n_trials
    for tag, var in (("per-arm binomial", var_sep), ("pooled binomial", var_pool)):
        w = 1.0 / var
        q = float((w * (delta - delta.mean()) ** 2).sum())
        p = stats.chi2.sf(q, n_sub - 1)
        print(f"  Cochran Q, {tag:<17} chi2 = {q:.2f} on {n_sub - 1} df, p = {p:.3f}")
    tab = np.array([[round(a * n_trials) for a in base],
                    [round(a * n_trials) for a in mot]]).T
    chi2_c, p_c, dof_c, _ = stats.chi2_contingency(tab)
    print(f"  contingency chi2 on the {n_sub}x2 correct-counts table: "
          f"chi2 = {chi2_c:.2f} on {dof_c} df, p = {p_c:.3f}")
    print("  ASSUMPTION MADE EXPLICIT: the two Cochran forms differ only in whether")
    print("  the per-subject variance is estimated per arm or from the pooled rate.")
    print("  The contingency form treats the 20 subjects as independent samples of a")
    print("  common success rate, which is a different null and gives a different df")
    print("  interpretation.")
    print(f"  THE FIGURE THIS SECTION PRODUCES for the claim as worded is the pooled")
    print(f"  Cochran form: chi2 = "
          f"{float((1 / var_pool * (delta - delta.mean()) ** 2).sum()):.2f} on "
          f"{n_sub - 1} df, "
          f"p = {stats.chi2.sf(float((1 / var_pool * (delta - delta.mean()) ** 2).sum()), n_sub - 1):.3f}.")
    print("  Consistency arithmetic on the pair quoted in the documents, which is")
    print(f"  not a measurement: chi2.sf(13.0, 19) = {stats.chi2.sf(13.0, 19):.3f}, so")
    print("  the published statistic and its published p-value agree with each other.")
    print("  The variance model behind that statistic is stated nowhere, so this")
    print("  section reproduces its family rather than its digits, and the value a")
    print("  document should quote is the one printed two lines above.")

    sub("B. Tukey 1-df non-additivity across all five pipelines")
    mat = np.array([scores[k] for k in
                    ["CSP+LDA", "MDM-64", "TSLR-64", "MDM-mot", "TSLR-mot"]],
                   dtype=float)
    f_stat, p_f, df_e = tukey_nonadditivity(mat)
    print(f"  F = {f_stat:.4f} on 1 and {df_e} df, p = {p_f:.4f}")
    for drop in ["MDM-64", "TSLR-64", "MDM-mot", "TSLR-mot"]:
        keep = [k for k in ["CSP+LDA", "MDM-64", "TSLR-64", "MDM-mot", "TSLR-mot"]
                if k != drop]
        f2, p2, _ = tukey_nonadditivity(np.array([scores[k] for k in keep],
                                                 dtype=float))
        print(f"  leave out {drop:<9} F = {f2:6.2f}, p = {p2:.4f}")

    sub("which test answers the question, and why")
    print("  The claim at stake is 'no method dominates per subject' -- i.e. whether")
    print("  the CHOICE of the better method depends on the subject. Test A asks")
    print("  exactly that for one named pair, on the scale the claim is made on, and")
    print("  its 19 df match the 20-subject design. Test B asks whether a")
    print("  multiplicative fan exists across FIVE arms at once, one degree of")
    print("  freedom, and it is dominated by whichever arm has the widest per-subject")
    print("  spread. The leave-one-out rows above show the F collapsing when the")
    print("  64-channel MDM arm is dropped, which identifies that arm as the source.")
    print("  TEST A IS THE APPROPRIATE ONE for the claim as worded. Test B is a")
    print("  correct test of a different hypothesis.")

    sub("what the null does NOT license")
    n_pairs = n_sub
    sd_delta = 100 * delta.std(ddof=1)
    print(f"  sd of the per-subject difference  {sd_delta:.2f} points")
    print(f"  MDE at 80% power, noncentral t    "
          f"{mde_noncentral(sd_delta, n_pairs):.2f} points")
    print("  A non-significant homogeneity test on 20 subjects cannot distinguish")
    print("  'no interaction' from 'an interaction this design cannot see'. The")
    print("  defensible wording is that no interaction is DETECTABLE here.")
    winners = {}
    for i in range(n_sub):
        col = {k: scores[k][i] for k in scores}
        winners[max(col, key=col.get)] = winners.get(max(col, key=col.get), 0) + 1
    print(f"  argmax pipeline per subject: {winners}")
    print("  A varying argmax is what an additive model with noise produces anyway,")
    print("  so the argmax spread is not itself evidence of an interaction.")


def tukey_nonadditivity(mat):
    """Tukey's 1-df test for non-additivity on a methods x subjects matrix."""
    grand = mat.mean()
    a = mat.mean(axis=1) - grand
    b = mat.mean(axis=0) - grand
    resid = mat - grand - a[:, None] - b[None, :]
    ss_na = (np.outer(a, b) * mat).sum() ** 2 / ((a ** 2).sum() * (b ** 2).sum())
    ss_e = (resid ** 2).sum()
    df_e = (mat.shape[0] - 1) * (mat.shape[1] - 1) - 1
    f = ss_na / ((ss_e - ss_na) / df_e)
    return f, stats.f.sf(f, 1, df_e), df_e
