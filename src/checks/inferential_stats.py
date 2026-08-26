"""Compute the confidence intervals, p-values and power figures that the write-up

NAVIGATION. Every inferential claim in the repo, recomputed in one place so the numbers
in README.md and EXPLAINER.md have exactly one source. Organised as section_*() functions
in the infstats_* modules beside this file; jump to the one you need.
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
    python src/checks/inferential_stats.py              # everything (~12 s, one data load)
    python src/checks/inferential_stats.py --skip-torch # array re-analysis only (~1 s)
"""

from __future__ import annotations

import argparse
import sys

# common.py lives one level up, beside the script groups; put its directory on the
# path so this script can be launched from anywhere.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infstats_lib import cannot, head
from infstats_ledger import section_discrepancies, section_ledger
from infstats_measured import (
    load_subject_one, section_bn_scale, section_mcnemar, section_weight_travel,
)
from infstats_rungs_8_9 import (
    section_inputs, section_interaction, section_power_headline, section_rung8,
    section_rung9,
)
from infstats_rungs_10_11 import section_rung11, section_sweep, section_wilson

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

