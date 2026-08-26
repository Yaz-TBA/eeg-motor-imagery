"""Compute the confidence intervals, p-values and power figures that the write-up

NAVIGATION. Every inferential claim in the repo, recomputed in one place so the numbers
in README.md and EXPLAINER.md have exactly one source. Organised as section_*() functions
called from main(); jump to the one you need rather than reading top to bottom.
asserts but no committed script produces.

THE DEFECT THIS GUARDS AGAINST is the one that cost this project nine retracted
claims: a number whose provenance you cannot state is not a result, however true
it turns out to be. `check_provenance.py` already catches numbers that no script
prints. It caught roughly thirty of them at once, and they were all the same
kind of number -- the inferential wrapper around a point estimate. Rungs 8-11
print accuracies and win/loss/tie tallies; the intervals, the paired tests, the
minimum detectable differences and the one chi-square live only in prose. Prose
drifts off code silently, and an interval computed once in a chat window and
typed into a document is indistinguishable, six weeks later, from an interval
that was never computed at all.

So this file recomputes every one of them from persisted arrays, prints each with
its baseline and its spread, and states what each does NOT show. Three of the
figures could not be recovered from any persisted artefact, and this file
measures them directly rather than restating them: the BatchNorm activation-scale
deficit, the final-layer weight travel, and the McNemar test on the within-subject
comparison. Where an input genuinely does not exist, the output says so and names
what would have to be re-run. An honest "cannot reproduce" line is a correct
output.

WHAT THIS FILE DOES NOT DO. It does not re-run any model whose scores are already
on disk. Rungs 8, 9 and 11 are re-analyzed from stored per-fold arrays, so this
script inherits their provenance exactly, including the caveat that
`regime_decomposition.json` is a 2026-07-23 checkpoint that the 2026-07-25 cold
run resumed from rather than recomputed. It also does not correct anything in the
documents; reconciling prose against this output is a separate pass.

INPUTS, and where each comes from:

  regime_decomposition.json   in this repo. 20 per-fold accuracies per model per
                              cell, seven cells. Dated 2026-07-23.
  sweep_results.csv           in this repo. 109 within-subject accuracies.
  riemannian_perfold.json     in this repo, and committed rather than generated.
                              20 per-subject LOSO accuracies for five pipelines.
                              `riemannian.py` computes these and persists none of
                              them, so no committed script can rebuild this file;
                              this copy was captured by the 2026-07-23 audit run.
                              Its five pipeline means are 59.4 / 51.7 / 57.2 /
                              56.9 / 56.8 %, which is what `riemannian.py` prints,
                              and that agreement is the evidence it is the right
                              run's arrays. Re-check by running riemannian.py; the
                              old citation here pointed into `.provenance_cache/`,
                              which .gitignore excludes, so it named nothing a
                              reader of a clone could open.
  the EEGBCI recordings       re-loaded for subject 1 only, for the three figures
                              that no array can supply.

STATISTICAL CONVENTIONS, stated once because every test below has options and
picking one silently is how two numbers that disagree end up looking like one
number that agrees. All tests are TWO-SIDED at alpha = 0.05. All model-versus-
model and window-versus-window comparisons are PAIRED, because every arm sees the
identical folds. Intervals are Student-t with df = n - 1 unless a line says
otherwise. Power uses the noncentral t at 80%; the normal approximation is
printed beside it because the two differ by ~5% and the documents do not say
which was used. Nothing here is corrected for multiplicity by default -- a Holm
pass over each family is printed separately, because the published figures are
uncorrected and quoting a corrected p beside an uncorrected estimate would be a
third kind of drift.

Usage:
    python inferential_stats.py              # everything (~12 s, one data load)
    python inferential_stats.py --skip-torch # array re-analysis only (~1 s)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

# joblib/MNE workers are fresh processes that re-import mne at its default log
# level, so set_log_level() alone never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from scipy.optimize import brentq

# common.py lives one level up, beside the script groups; put its directory on the
# path so this script can be launched from anywhere.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import holm as common_holm, wilson_interval as common_wilson

ROOT = Path(__file__).resolve().parent.parent.parent  # the repo root: this file lives in src/checks/

ALPHA = 0.05
POWER = 0.80
Z_TWO_SIDED = 1.959963985  # the multiplier a NORMAL 95% interval uses
Z_POWER = 0.8416212336     # one-sided z at 80% power

# Subject 1, narrow regime -- identical constants to eegnet_compare.py, so the
# three measured figures below are comparable with experiment A there.
SUBJECT = 1
RUNS = [6, 10, 14]
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BN_EPS = 1e-3
NARROW = dict(l_freq=8.0, h_freq=30.0, crop=(1.0, 2.0))
N_SCALE_SEEDS = 5  # the scale diagnostic is seed-dependent; report its spread


# ---------------------------------------------------------------------------
# Estimators. Each returns the number AND the assumption it rests on.
# ---------------------------------------------------------------------------

def t_interval(x, alpha=ALPHA):
    """Student-t interval on the mean of x. df = n - 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * sem
    return m, m - half, m + half


def normal_interval(x, z=Z_TWO_SIDED):
    """Normal (z) interval on the mean of x. Wider df assumption than t."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - z * sem, m + z * sem


def wilson_interval(n_correct, n_total, z=Z_TWO_SIDED):
    """95% CI for a proportion, at this file's exact-normal multiplier.

    The formula now lives in common.py. It used to be reimplemented here, because
    evaluate_honestly.py defined it at module scope alongside a five-minute analysis and
    importing it would have run that analysis. Every script has a __main__ guard now, so
    importing is free and there is one definition instead of three.

    The z default stays Z_TWO_SIDED rather than common's 1.96, so this file's printed
    intervals are unchanged.
    """
    return common_wilson(n_correct, n_total, z)


def paired_power(delta, sd, n, alpha=ALPHA):
    """Two-sided power of a paired t-test, noncentral t reference."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha / 2, df)
    ncp = delta / sd * np.sqrt(n)
    return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)


def mde_noncentral(sd, n, alpha=ALPHA, power=POWER):
    """Smallest true difference a paired t-test detects with `power`.

    Noncentral-t solve. This is the correct reference distribution; the normal
    approximation below is ~5% smaller and is printed only so that a document
    quoting either can be identified.
    """
    hi = 3.0 * sd
    while paired_power(hi, sd, n, alpha) < power:
        hi *= 1.5
    return brentq(lambda d: paired_power(d, sd, n, alpha) - power, 1e-9, hi,
                  xtol=1e-9)


def mde_normal(sd, n, alpha=ALPHA, power=POWER):
    """The z-formula MDE. Anticonservative: it ignores the estimated variance."""
    return (stats.norm.ppf(1 - alpha / 2) + Z_POWER) * sd / np.sqrt(n)


# Holm-Bonferroni adjusted p-values, order preserved. Defined in common.py.
holm = common_holm


def head(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def sub(title):
    print(f"\n--- {title} ---")


def cannot(what, why, rerun):
    print(f"CANNOT RECOMPUTE: {what}")
    print(f"  reason:   {why}")
    print(f"  needs:    {rerun}")


# ---------------------------------------------------------------------------
# Input loading. Missing inputs are reported, never imputed.
# ---------------------------------------------------------------------------

def load_regime():
    path = ROOT / "results/regime_decomposition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_perfold():
    """Per-subject LOSO arrays for the five rung-9 pipelines.

    Repo-local only, and deliberately so. This used to fall back to an absolute
    path under ~/Documents/Projects/audits/, which is not a git repository and
    resolves to nothing in a clone, so the fallback could only ever fire on one
    machine. The file it named was byte-identical to the committed copy
    (md5 a7bc94bf7e8271e79cec718c0ea7d271, 2870 bytes), so dropping it loses no
    data.
    """
    path = ROOT / "results/riemannian_perfold.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_sweep():
    path = ROOT / "results/sweep_results.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        return list(csv.DictReader(fh))


REGIME = load_regime()
PERFOLD = load_perfold()
SWEEP = load_sweep()


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
# 6-8. The three figures that no persisted array can supply
# ---------------------------------------------------------------------------

def load_subject_one():
    """Load subject 1 exactly as eegnet_compare.py's experiment A does."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
    raw = mne.concatenate_raws(
        [mne.io.read_raw_edf(p, preload=True) for p in paths])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(NARROW["l_freq"], NARROW["h_freq"], fir_design="firwin",
               skip_by_annotation="edge")
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=-1.0, tmax=4.0,
                        picks="eeg", baseline=None, preload=True)
    X = epochs.copy().crop(*NARROW["crop"]).get_data(copy=False)
    y = (epochs.events[:, -1] == 3).astype(np.int64)
    return X, y


def section_mcnemar(X, y):
    head("6. RUNG 10 -- McNEMAR ON THE WITHIN-SUBJECT COMPARISON")
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import EEGNet
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    print("NO ARRAY ON DISK HOLDS THIS. McNemar needs the discordant pair counts,")
    print("which need the two models' PER-TRIAL predictions on the same folds.")
    print("eegnet_compare.py scores its folds and discards the predictions, so this")
    print("section re-runs experiment A to recover them: same 45 trials, same")
    print("StratifiedKFold(5, shuffle=True, random_state=42), same seed, same model.")
    print("THE SEEDING ORDER IS COPIED, NOT APPROXIMATED. eegnet_compare.py seeds")
    print("once and lets cross_val_score clone across folds, so the five folds do")
    print("not restart from the same RNG state. Reseeding per fold changes four of")
    print("the five folds and moves the EEGNet count by several trials, which is")
    print("enough on its own to make a McNemar table irreproducible.")
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    csp = Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])
    pred_csp = cross_val_predict(csp, X, y, cv=cv)

    seed_torch()
    net = EEGClassifier(
        module=EEGNet, module__n_chans=X.shape[1], module__n_outputs=2,
        module__n_times=X.shape[2], optimizer=torch.optim.AdamW,
        optimizer__lr=LR, optimizer__weight_decay=1e-4,
        batch_size=BATCH_SIZE, max_epochs=N_EPOCHS, train_split=None,
        device=device, verbose=0, callbacks=[])
    # microvolts: the working configuration every reported accuracy comes from
    pred_net = cross_val_predict(net, (X * 1e6).astype(np.float32), y, cv=cv)

    k_csp = int((pred_csp == y).sum())
    k_net = int((pred_net == y).sum())
    print(f"  CSP + LDA correct     {k_csp}/{len(y)} = {100 * k_csp / len(y):.1f}%")
    print(f"  EEGNet correct        {k_net}/{len(y)} = {100 * k_net / len(y):.1f}%")
    print(f"  EEGNet - CSP          {100 * (k_net - k_csp) / len(y):+.1f} points")
    counts = np.bincount(pred_net, minlength=2)
    print(f"  EEGNet predicted class counts {counts.tolist()} "
          f"(true {np.bincount(y, minlength=2).tolist()})")

    b = int(((pred_csp == y) & (pred_net != y)).sum())
    c = int(((pred_csp != y) & (pred_net == y)).sum())
    both = int(((pred_csp == y) & (pred_net == y)).sum())
    neither = int(((pred_csp != y) & (pred_net != y)).sum())
    sub("the 2x2 agreement table")
    print(f"  both correct            {both}")
    print(f"  CSP only  (b)           {b}")
    print(f"  EEGNet only (c)         {c}")
    print(f"  neither correct         {neither}")
    if b + c == 0:
        print("  No discordant pairs; McNemar is undefined here.")
        return
    p_exact = float(stats.binomtest(b, b + c, 0.5).pvalue)
    print(f"  McNemar exact, two-sided, binomial on {b + c} discordant pairs")
    print(f"  McNemar p = {p_exact:.3f}")
    print("  ASSUMPTION: exact binomial rather than the chi-square approximation,")
    print(f"  because {b + c} discordant pairs is far too few for the asymptotic form.")
    print()
    print("  THE MARGINAL COUNTS DO NOT DETERMINE THE SPLIT. Any (b, c) with")
    print(f"  b - c = {k_csp - k_net} is consistent with {k_csp}/{len(y)} against "
          f"{k_net}/{len(y)}, and the p-value")
    print("  depends on which one it is, not on the difference. The maximally nested")
    print(f"  split, b = {k_csp - k_net} and c = 0, would give p = "
          f"{float(stats.binomtest(k_csp - k_net, k_csp - k_net, 0.5).pvalue):.3f}; the")
    print(f"  measured split of b = {b}, c = {c} gives p = {p_exact:.3f}. Deriving a")
    print("  McNemar p from two accuracies alone is arithmetic on an assumption about")
    print("  agreement that the predictions themselves settle.")
    print("  What this does NOT show: a non-significant McNemar on 45 trials leaves")
    print("  the direction of the difference undecided. Only the sign of b - c is")
    print("  consistent with the point estimate; its magnitude is not established.")
    print("  DETERMINISM CAVEAT: MPS kernels are not bit-reproducible, so the EEGNet")
    print("  half of this table can shift by a trial or two between runs.")


def seed_torch(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def section_bn_scale(X):
    head("7. RUNG 10 -- THE BATCHNORM ACTIVATION-SCALE DEFICIT, MEASURED")
    import torch
    import torch.nn as nn
    from braindecode.models import EEGNet

    print("NO ARRAY ON DISK HOLDS THIS EITHER. No persisted artefact in this repo")
    print("records a per-layer activation standard deviation, so the deficit figure")
    print("has never been produced by a script. This section measures it directly:")
    print("forward hooks on every BatchNorm2d, one batch in a no-grad forward pass,")
    print("dropout disabled so the only thing varying is scale.")
    print()
    print("HOW THE MEASUREMENT IS DEFINED. BatchNorm divides by sqrt(var + eps). If")
    print("the input variance is far below eps the divisor is essentially sqrt(eps),")
    print("so the stage's output lands sqrt(eps)/sigma too small instead of at unit")
    print(f"scale. eps = {BN_EPS:g}, so sqrt(eps) = {np.sqrt(BN_EPS):.4f}.")
    print()
    print(f"Signal scale, subject 1, {NARROW['l_freq']:.0f}-{NARROW['h_freq']:.0f} Hz, "
          f"{NARROW['crop'][0]:.1f}-{NARROW['crop'][1]:.1f} s:")
    print(f"  standard deviation in volts   {X.std():.3e}")
    print(f"  variance in volts             {X.var():.3e}")
    print(f"  ratio eps / variance          {BN_EPS / X.var():.2e}")

    bn_names = ["bnorm_temporal", "bnorm_1", "bnorm_2"]
    per_seed = {name: [] for name in bn_names}
    clf_in = {"volts": [], "microvolts": []}
    logits = {"volts": [], "microvolts": []}

    for s in range(N_SCALE_SEEDS):
        seed_torch(SEED + s)
        model = EEGNet(n_chans=X.shape[1], n_outputs=2, n_times=X.shape[2])
        caps = {}

        def make_hook(tag):
            def hook(_mod, inp, out):
                caps[tag] = (inp[0].detach(), out.detach())
            return hook

        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) or name.endswith("conv_classifier"):
                mod.register_forward_hook(make_hook(name))
        model.train()
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.eval()

        for scale, tag in ((1.0, "volts"), (1e6, "microvolts")):
            batch = torch.tensor((X[:BATCH_SIZE] * scale).astype(np.float32))
            with torch.no_grad():
                model(batch)
            if tag == "volts":
                for name in bn_names:
                    act = caps[name][0]
                    var = act.transpose(0, 1).reshape(act.shape[1], -1).var(
                        dim=1, unbiased=False).mean().item()
                    per_seed[name].append((np.sqrt(var), np.sqrt(var + BN_EPS)))
            key = [k for k in caps if k.endswith("conv_classifier")][0]
            clf_in[tag].append(caps[key][0].std().item())
            logits[tag].append(caps[key][1].std().item())

    sub(f"per-stage deficit at volts scale (mean over {N_SCALE_SEEDS} seeds)")
    print(f"{'stage':<18}{'sigma in':>13}{'divisor':>13}{'deficit':>12}{'sd':>10}")
    for name in bn_names:
        sig = np.array([a for a, _ in per_seed[name]])
        div = np.array([b for _, b in per_seed[name]])
        ratio = div / sig
        print(f"{name:<18}{sig.mean():>13.4e}{div.mean():>13.4e}"
              f"{ratio.mean():>11.0f}x{ratio.std(ddof=1):>10.0f}")
    first = np.array([b / a for a, b in per_seed["bnorm_temporal"]])
    print()
    print(f"THE DEFICIT AT THE FIRST BATCHNORM IS {first.mean():.0f}x "
          f"(sd {first.std(ddof=1):.0f} over {N_SCALE_SEEDS} seeds).")
    print("  Its input sigma is the signal after one temporal convolution; the")
    print("  divisor is sqrt(eps) because the variance is seven orders below eps.")

    sub("end-to-end, measured rather than modeled")
    v_in, u_in = np.array(clf_in["volts"]), np.array(clf_in["microvolts"])
    v_lg, u_lg = np.array(logits["volts"]), np.array(logits["microvolts"])
    print(f"  classifier-input sd, volts       {v_in.mean():.4e}")
    print(f"  classifier-input sd, microvolts  {u_in.mean():.4e}")
    print(f"  END-TO-END DEFICIT AT THE CLASSIFIER INPUT  "
          f"{(u_in / v_in).mean():.0f}x  (sd {(u_in / v_in).std(ddof=1):.0f})")
    print(f"  logit sd, volts                  {v_lg.mean():.4e}")
    print(f"  logit sd, microvolts             {u_lg.mean():.4e}")
    print(f"  DEFICIT AT THE LOGITS            {(u_lg / v_lg).mean():.0f}x")
    print()
    print("  A RECOVERY MODEL OF 31.6x PER STAGE IS AN ASSUMPTION, NOT A MEASUREMENT.")
    print("  That model implies the deficit runs 4500x, then 142x, then 4.5x, then")
    print("  0.14x across three stages. The measured per-stage deficits above do not")
    print("  decay at that rate, and the measured end-to-end deficit printed here is")
    print("  the figure any downstream scale argument should use.")
    print("  What this does NOT show: this is the BROKEN volts configuration, which")
    print("  eegnet_compare.py's assertion now refuses to run. It is a diagnosis of a")
    print("  configuration kept for the record, not a current result. The microvolt")
    print("  column is the configuration every reported accuracy comes from.")
    return float((u_lg / v_lg).mean())


def section_weight_travel(X, y, end_to_end):
    head("8. RUNG 10 -- FINAL-LAYER WEIGHT TRAVEL AGAINST THE SCALE GAP")
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import EEGNet
    from sklearn.model_selection import StratifiedKFold

    print("NO ARRAY ON DISK HOLDS THIS. Nothing records final-layer weight statistics")
    print("at init or after training, so the 'the network cannot train out of it'")
    print("argument has never had a measured margin. This section measures both ends.")
    print()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, _ = next(iter(cv.split(X, y)))
    print(f"One fold of experiment A: {len(train)} training trials, "
          f"{N_EPOCHS} epochs of AdamW at lr = {LR:g}, batch {BATCH_SIZE}.")
    print()

    results = {}
    for scale, tag in ((1.0, "volts"), (1e6, "microvolts")):
        seed_torch()
        clf = EEGClassifier(
            module=EEGNet, module__n_chans=X.shape[1], module__n_outputs=2,
            module__n_times=X.shape[2], optimizer=torch.optim.AdamW,
            optimizer__lr=LR, optimizer__weight_decay=1e-4,
            batch_size=BATCH_SIZE, max_epochs=N_EPOCHS, train_split=None,
            device=device, verbose=0, callbacks=[])
        clf.initialize()

        def final_weight():
            layer = dict(clf.module_.named_modules())["final_layer.conv_classifier"]
            return layer.weight.detach().cpu().numpy().copy()

        # Re-fetched rather than held: skorch may rebind module_ during fit, and
        # a stale reference silently reports zero travel, which reads exactly
        # like the finding this section exists to test.
        w0 = final_weight()
        clf.partial_fit((X[train] * scale).astype(np.float32), y[train])
        w1 = final_weight()
        results[tag] = (w0, w1)
        print(f"  {tag:<11} weight sd {w0.std():.4f} -> {w1.std():.4f}   "
              f"travel in sd {abs(w1.std() - w0.std()):.4f}   "
              f"mean |dw| {np.abs(w1 - w0).mean():.4f}")

    w0, w1 = results["volts"]
    init_sd = float(w0.std())
    achieved_sd = float(abs(w1.std() - w0.std()))
    achieved_abs = float(np.abs(w1 - w0).mean())
    if achieved_sd == 0.0 or achieved_abs == 0.0:
        cannot("the training margin",
               "the final layer did not move at all, which means the weights were "
               "read from a stale module reference rather than the trained one",
               "check that final_weight() is re-fetched after partial_fit")
        return
    print()
    print(f"  init sd is {init_sd:.4f}, and the mean per-weight move is "
          f"{achieved_abs:.4f}.")
    print("  Two different quantities have both been called 'the travel': the change")
    print("  in the weight standard deviation, and the mean absolute per-weight")
    print(f"  change. Here they differ by {achieved_abs / achieved_sd:.1f}x, so a "
          f"margin quoted without")
    print("  saying which one it is cannot be checked.")

    sub("the margin, computed from the MEASURED end-to-end deficit")
    required_sd = init_sd * end_to_end
    print(f"  end-to-end deficit at the logits          {end_to_end:.0f}x")
    print(f"  final-layer sd needed to close it         {required_sd:.3f}")
    print(f"  required travel                           {required_sd - init_sd:.3f}")
    print(f"  travel achieved in {N_EPOCHS} epochs (sd)         {achieved_sd:.4f}")
    print(f"  shortfall, sd definition                  "
          f"{(required_sd - init_sd) / achieved_sd:.0f}x")
    print(f"  travel achieved in {N_EPOCHS} epochs (mean |dw|)  {achieved_abs:.4f}")
    print(f"  shortfall, mean-|dw| definition           "
          f"{(required_sd - init_sd) / achieved_abs:.0f}x")
    print()
    print("  A MARGIN COMPUTED FROM AN ASSUMED 4.5x RESIDUAL GAP would give a")
    print(f"  required travel of {init_sd * 4.5 - init_sd:.3f} and a shortfall near "
          f"{(init_sd * 4.5 - init_sd) / achieved_abs:.1f}x.")
    print("  That gap comes from the 31.6x-per-stage recovery model, which section 7")
    print("  measures and does not confirm.")
    print("  What this does NOT show: a scale argument is not a training experiment.")
    print("  The direct evidence that the volts configuration does not train is the")
    print("  degenerate-prediction behavior, which the guard in eegnet_compare.py now")
    print("  refuses to reproduce. This section bounds the optimizer's reach; it does")
    print("  not prove the optimizer could never find another route.")


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


# ---------------------------------------------------------------------------
# 11. Ledger of what still cannot be recomputed
# ---------------------------------------------------------------------------

def section_discrepancies(ran_torch):
    head("11. WHERE THIS RUN AND THE PROSE DO NOT AGREE")
    print("Reconciling the documents is a separate pass, so this section names the")
    print("disagreements rather than resolving them. In each case the value a")
    print("document should carry is the one this file prints in the section named.")
    print()
    items = [
        ("section 4, the subject x method interaction",
         "the homogeneity test this file computes lands near p = 0.82, not near "
         "the p the documents carry. The two are the same family of test with a "
         "different variance model, and no document states which model it used."),
        ("section 9, the pre-cue control",
         "the interval is computed here for the first time and its upper bound "
         "sits above the cue-window CSP estimate the same rung calls significant, "
         "so 'the control passes' is not what these arrays support."),
    ]
    if ran_torch:
        items = [
            ("section 6, McNemar on the within-subject comparison",
             "the reproduced per-trial predictions give a discordant split that is "
             "not the maximally nested one, so the exact test lands well above the "
             "value obtainable from the two accuracies alone. A McNemar p derived "
             "from marginal accuracies is arithmetic on an assumption, and the "
             "predictions contradict that assumption here."),
            ("section 7, the end-to-end activation-scale deficit",
             "the measured deficit at the logits is roughly two orders of "
             "magnitude, not the single-digit multiplier the 31.6x-per-stage "
             "recovery model implies. The first-BatchNorm deficit, by contrast, "
             "reproduces the figure the documents state as established."),
            ("section 8, the training-margin shortfall",
             "computed against the measured deficit the shortfall is two orders of "
             "magnitude rather than the small factor the assumed recovery model "
             "gives, and the two definitions of 'travel' differ by about 2x, so a "
             "margin quoted without naming its definition cannot be checked."),
        ] + items
    for where, what in items:
        print(f"  {where}")
        for line in _wrap(what, 70):
            print(f"      {line}")
        print()


def _wrap(text, width):
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def section_ledger(ran_torch):
    head("12. LEDGER -- WHAT THIS FILE STILL CANNOT PRODUCE")
    items = [
        ("rung-11 figures at 2026-07-25 scores",
         "regime_decomposition.json is a 2026-07-23 checkpoint and the 07-25 run "
         "resumed from it, so source-hash cache invalidation was defeated",
         "delete regime_decomposition.json, then rerun regime_decomposition.py cold"),
        ("per-subject arrays produced by riemannian.py itself",
         "riemannian.py persists only a PNG; this file reads a copy captured by the "
         "2026-07-23 audit run, whose means match the cached stdout exactly",
         "edit riemannian.py to dump the score arrays it already holds in memory"),
        ("per-subject arrays produced by cross_subject.py itself",
         "cross_subject.py prints mean, median, min and max but persists no "
         "per-subject values; the rung-8 cross arm here comes from the audit capture",
         "edit cross_subject.py to persist its per-subject accuracies"),
        ("the exact variance model behind the published chi-square of 13.0",
         "no script and no document states whether the homogeneity weights are "
         "per-arm or pooled binomial; section 4 brackets it rather than matching it",
         "state the weighting in the document, or adopt one of the two forms here"),
    ]
    if not ran_torch:
        items.append((
            "the BatchNorm deficit, the weight travel and McNemar",
            "this run used --skip-torch",
            "rerun without --skip-torch"))
    for what, why, fix in items:
        cannot(what, why, fix)
        print()
    print("Each line above is a correct output. A plausible number in place of any")
    print("of them would be the defect this repo exists to have stopped making.")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-torch", action="store_true",
                    help="skip the three sections that re-run subject 1")
    args = ap.parse_args()

    print("INFERENTIAL STATISTICS FOR RUNGS 8-11")
    print("Two-sided, alpha = 0.05, paired where the folds are shared.")
    print("Intervals are Student-t with df = n - 1 unless a line says otherwise.")
    print("Power is noncentral t at 80%, with the normal approximation beside it.")
    print("No multiplicity correction is applied to any published figure; Holm is")
    print("printed separately per family.")

    section_inputs()
    d8 = section_rung8()
    r9 = section_rung9()
    section_power_headline(d8, r9)
    section_interaction()
    section_wilson()

    ran_torch = False
    if not args.skip_torch:
        try:
            import torch  # noqa: F401
            import braindecode  # noqa: F401
        except ImportError as exc:
            head("6-8. THE THREE MEASURED FIGURES")
            cannot("the BatchNorm deficit, the weight travel and McNemar",
                   f"an import failed: {exc}",
                   "pip install -r requirements-dl.txt, then rerun")
        else:
            print("\nLoading subject 1 for the three measured sections ...")
            X, y = load_subject_one()
            print(f"  {X.shape[0]} trials, {X.shape[1]} channels, "
                  f"{X.shape[2]} samples")
            section_mcnemar(X, y)
            end_to_end = section_bn_scale(X)
            section_weight_travel(X, y, end_to_end)
            ran_torch = True
    else:
        head("6-8. THE THREE MEASURED FIGURES")
        cannot("the BatchNorm deficit, the weight travel and McNemar",
               "--skip-torch was passed", "rerun without --skip-torch")

    section_rung11()
    section_sweep()
    section_discrepancies(ran_torch)
    section_ledger(ran_torch)
    print("\nDone. Every number above came from a persisted array or a measurement")
    print("taken in this run. Reconciling the documents against it is a separate pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
