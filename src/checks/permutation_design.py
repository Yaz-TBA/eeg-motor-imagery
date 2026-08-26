"""Measure what a CORRECTLY DESIGNED permutation null does to this repo's p-values.

NAVIGATION. This is the registered validity run and it is long since the design IS the
result. Section 0 sets up, section 1 is falsification gate 1 (does the pilot reproduce),
section 2 is gate 2 (sklearn agreement), and the two arms diverge after that. If you only
want the outcome, search for "PRE-REGISTERED VERDICT". No jokes anywhere in this file, on
purpose: a script whose whole job is showing I didn't fool myself should read like one.

WHY THIS SCRIPT EXISTS. Two published nulls in this repo are built on assumptions
the experiment does not satisfy, and until now the repo has conceded that in prose
without measuring it.

  (a) WITHIN-SUBJECT. decode_csp.py:124 and evaluate_honestly.py:185 call
      permutation_test_score with cv=StratifiedKFold(5, shuffle=True,
      random_state=42). sklearn's _permutation_test_score loops
      `for train, test in cv.split(X, y, ...)` with the PERMUTED labels as y, so
      StratifiedKFold re-derives the folds from each permuted vector. Every
      replicate is therefore scored on a DIFFERENT partition, while the observed
      91.1% is scored on the partition stratified on the TRUE labels. The null
      marginalises over partitions the observed value conditions on.

  (b) CROSS-SUBJECT. cross_subject.py:135 draws ONE global permutation of all 900
      pooled labels. That tests the compound hypothesis "labels are unrelated to
      the EEG AND labels are exchangeable ACROSS subjects". The second conjunct is
      false by protocol: each subject's class marginal is fixed by the experiment.
      A global shuffle can deal a 45-trial subject 30 feet and 15 hands, a draw the
      experiment could not have produced, so the null carries a variance component
      that has nothing to do with decoding.

Those are the same underlying error at two levels: THE NULL MUST CONDITION ON
WHATEVER THE OBSERVED STATISTIC CONDITIONS ON. Arm (a) is a defect of the
PARTITION. Arm (b) is a defect of the REFERENCE SET. They are not one objection
applied twice, and LeaveOneGroupOut is already label-invariant, so arm (a)'s defect
does not exist in arm (b).

THE EXCHANGEABLE UNIT, stated explicitly, which is what the objection said this
project could not do:
  (a) the trial, CONDITIONAL ON RUN, with the statistic evaluated on a FIXED
      partition. Within each of runs 6/10/14 the labels are exchangeable; nothing
      is exchangeable across runs, because run is a blocking factor with its own
      electrode settling and drift.
  (b) the trial, CONDITIONAL ON SUBJECT. The partition needs no correction because
      LeaveOneGroupOut already conditions on subject.

WHAT THIS SCRIPT GUARDS AGAINST. Three things, in order of how easy they are to
fool yourself with:
  1. Reading a resolution floor as a measurement. Subject 1's effect is roughly
     4.5 null sd out, so its p is at the floor in EVERY cell no matter which null
     is used. That is a property of the effect size, not evidence that the design
     was right. This script therefore leads with the PAIRED per-replicate null
     difference and the null's SHAPE, not with the p.
  2. An unpaired comparison masquerading as a paired one. The pilot this replaces
     fed different label vectors to the cells it compared, so a 3.0-point gap was
     confounded with Monte Carlo noise. Here ONE list of permuted label vectors
     feeds both partition rules, and the script asserts the two cells saw
     element-wise identical labels at every replicate.
  3. A correction that silently changed nothing. Asserts 5 and 6 below check that
     the fixed cells really are fixed AND that the re-stratified cells really do
     move. A "correction" where both cells were secretly identical would look like
     a clean null result.

WHAT THIS SCRIPT DOES NOT SHOW. It does not repair n=45. It does not make the
within-subject null binomial. It does not address a session-level trend across all
three runs, which survives every null here exactly as it survives the
leave-one-run-out ablation in ablate_channels.py:255-259. And three subjects is
not a survey: subjects 17 and 19 were picked by a fixed median rule from
sweep_results.csv, so the claim available from them is EXISTENCE (whether the
correction CAN move a verdict), never frequency.

PRE-REGISTERED at neuro-canon/measurements/prereg-block-permutation.md, written
before this file existed. Every outcome below was written down in advance.
"""

from permdesign_setup import build_setup
from permdesign_gates import run_gate1, run_gate2
from permdesign_dummy import run_dummy_control, run_type1
from permdesign_arm_a import run_arm_a
from permdesign_mechanism import run_mechanism
from permdesign_arm_b import run_arm_b
from permdesign_verdict import run_verdict


def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-hour experiment as a side effect. The sections
    live in the permdesign_* modules beside this file, split 2026-08-26; their
    bodies are verbatim from the single-file version, so the stdout is unchanged
    line for line and the block-cache stamps still resolve."""

    D = build_setup()
    n_match, GATE1_FATAL = run_gate1(D)
    GATE2_OK, z_mean, z_sd = run_gate2(D)
    run_dummy_control(D)
    run_type1(D)
    ARM_A, material_paired, material_unpaired = run_arm_a(D)
    MECH = run_mechanism(D, ARM_A)
    B = run_arm_b(material_unpaired)
    run_verdict(D, ARM_A, MECH, B, n_match, GATE1_FATAL, GATE2_OK, z_mean,
                z_sd, material_paired, material_unpaired)


if __name__ == "__main__":
    main()
