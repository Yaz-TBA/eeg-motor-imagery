"""Sections 11 and 12 of inferential_stats.py: where this run and the prose disagree,
and the ledger of what still cannot be recomputed. Split out 2026-08-26. An honest
"cannot reproduce" line is a correct output."""

from infstats_lib import cannot, head

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
