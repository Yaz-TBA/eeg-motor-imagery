"""Compute the confidence intervals, p-values and power figures the write-up asserts.

Every inferential claim in the repo, recomputed in one place so the numbers in
README.md and EXPLAINER.md have exactly one source. Organised as section_*()
functions in the infstats_* modules beside this file; jump to the one you need.

The defect this guards against:

The one that cost this project nine retracted claims: a number whose provenance
you cannot state is not a result, however true it turns out to be.
check_provenance.py caught roughly thirty at once, all the same kind, the
inferential wrapper around a point estimate. Rungs 8-11 print accuracies and
tallies; the intervals, paired tests, minimum detectable differences and the one
chi-square lived only in prose, and an interval computed once in a chat window
is indistinguishable, six weeks later, from one that was never computed at all.
So this file recomputes each from persisted arrays, prints it with its baseline
and spread, and states what it does not show. Three figures no artefact can
supply are measured directly: the BatchNorm activation-scale deficit, the
final-layer weight travel, and the McNemar on the within-subject comparison.
Where an input does not exist, the output says so and names the re-run; an
honest "cannot reproduce" line is a correct output.

What this file does not do:

It re-runs no model whose scores are on disk, so rungs 8, 9 and 11 inherit their
provenance exactly, including that regime_decomposition.json is a 2026-07-23
checkpoint the 2026-07-25 cold run resumed from. It also corrects nothing in the
documents; reconciling prose against this output is a separate pass.

Inputs:

  regime_decomposition.json  in this repo; 20 per-fold accuracies per model per
                             cell, seven cells, dated 2026-07-23
  sweep_results.csv          in this repo; 109 within-subject accuracies
  riemannian_perfold.json    in this repo, committed rather than generated:
                             riemannian.py persists none of its 20 per-subject
                             LOSO arrays, so this copy is the 2026-07-23 audit
                             capture. Its five pipeline means, 59.4 / 51.7 /
                             57.2 / 56.9 / 56.8 %, match what riemannian.py
                             prints, which is the evidence it is the right run
  the EEGBCI recordings      re-loaded for subject 1 only, for the three
                             measured figures

Conventions, stated once because picking silently is how two numbers that
disagree end up looking like one that agrees: all tests two-sided at alpha =
0.05; every model-vs-model and window-vs-window comparison paired, since every
arm sees identical folds; intervals Student-t with df = n - 1 unless a line says
otherwise; power from the noncentral t at 80%, the normal approximation printed
beside it because they differ by ~5% and the documents never said which was
used. No multiplicity correction by default; Holm prints separately per family,
because the published figures are uncorrected.

Usage:
    python src/checks/inferential_stats.py              # everything (~12 s)
    python src/checks/inferential_stats.py --skip-torch # array re-analysis (~1 s)
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

