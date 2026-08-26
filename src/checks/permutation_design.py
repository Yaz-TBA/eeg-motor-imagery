"""Measure what a correctly designed permutation null does to this repo's p-values.

The registered validity run, and it is long because the design is the result.
Section 0 sets up, sections 1 and 2 are the falsification gates (pilot
reproduction, sklearn agreement), and the two arms diverge after that. If you
only want the outcome, search for "PRE-REGISTERED VERDICT". No jokes anywhere in
this file, on purpose: a script whose whole job is showing I didn't fool myself
should read like one.

Why this exists:

Two published nulls rest on assumptions the experiment does not satisfy, conceded
until now in prose without measurement.

  (a) Within-subject. decode_csp.py and evaluate_honestly.py call
      permutation_test_score with cv=StratifiedKFold(5, shuffle=True,
      random_state=42), and sklearn re-derives the folds from each permuted
      vector. Every replicate is scored on a different partition while the
      observed 91.1% is scored on the partition stratified on the true labels:
      the null marginalises over partitions the observed value conditions on.
  (b) Cross-subject. cross_subject.py draws one global permutation of all 900
      pooled labels, which also assumes labels are exchangeable across subjects.
      That is false by protocol: each subject's class marginal is fixed by the
      experiment, and a global shuffle can deal a 45-trial subject 30 feet and
      15 hands, a draw the experiment could not produce. The null carries a
      variance component that has nothing to do with decoding.

Same underlying error at two levels: the null must condition on whatever the
observed statistic conditions on. (a) is a defect of the partition, (b) of the
reference set, and LeaveOneGroupOut is already label-invariant, so (a)'s defect
does not exist in arm (b).

The exchangeable unit, stated explicitly:

  (a) the trial, conditional on run, scored on a fixed partition. Labels are
      exchangeable within each of runs 6/10/14 and nothing is exchangeable
      across runs, since run is a blocking factor with its own settling and drift.
  (b) the trial, conditional on subject; LeaveOneGroupOut already conditions on
      subject, so the partition needs no correction.

What this guards against, in order of how easy each is to fool yourself with:

  1. Reading a resolution floor as a measurement. Subject 1 sits roughly 4.5
     null sd out, so its p is at the floor in every cell no matter the null.
     The finding for subject 1 is the paired per-replicate difference and the
     null's shape, never the p.
  2. An unpaired comparison masquerading as paired. The pilot fed different
     label vectors to the cells it compared. Here one list of permuted vectors
     feeds both partition rules, asserted element-wise identical per replicate.
  3. A correction that silently changed nothing. Asserts 5 and 6 check the fixed
     cells really are fixed and the re-stratified cells really move.

What it does not show:

It does not repair n=45, does not make the within-subject null binomial, and
does not address a session-level trend across the three runs, which survives
every null here as it survives the leave-one-run-out ablation. Three subjects is
not a survey: 17 and 19 came from a fixed median rule on sweep_results.csv, so
the available claim is existence (the correction can move a verdict), never
frequency.

Pre-registered at neuro-canon/measurements/prereg-block-permutation.md, written
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
