"""The artifact control: refit on the channels where the eyes are loudest and see
whether the result survives there.

Genuinely didn't expect this one.. deleting the motor strip was supposed to break the
decoder and it didn't ?? 77.8% on the remaining 47 channels. Lost this prediction i guess ?
Read the caveats below before taking anything from it though, since a channel deletion
can't falsify a source hypothesis in either direction.

A scalp topography is a picture of the model's weights, and a model riding an
eye-movement artifact will happily draw one too. The control that bites is an
ablation: refit the whole pipeline on a subset that cannot see sensorimotor
cortex and check that accuracy collapses to the majority-class rate, which with
21 hands and 24 feet is 53.3%, not 50%.

The six conditions:

  (a) all 64 channels       reproduces the published headline
  (b) sensorimotor only     FC/C/CP strip; should hold or improve
  (c) frontopolar only      Fp/AF, where blinks and saccades are loudest; the
                            subset that keeps working if the decoder reads eyes
  (d) leave-one-run-out     all 64, folds are whole runs, so no fold shares a
                            drift or settling trend with its training set
  (e) sensorimotor deleted  the 17-channel strip removed, the other 47 kept;
                            the necessity arm, added 2026-07-25
  (f) wide FC/C/CP deleted  the 21-channel block removed, 43 kept; bounds the
                            FC5/FC6/CP5/CP6 that (e) retains

(a)-(c), (e), (f) use stratified 5-fold; (d) uses leave-one-run-out, which is why
it prints three per-fold values against the others' five. Seed 42 is primary, and
every seeded condition also carries a ten-seed sweep, because one seed is one
quantized draw. The decision statistic for (e) is a ten-seed mean gap AND an exact
McNemar, both fixed in advance in prereg/prereg-complement-ablation.md, which also
records that (e)'s accuracy half had already been run once and was not blind.

Why this table is a script:

An earlier README published 91.1 / 95.9 / 47.4 / 93.3 with no script behind them.
On 45 trials tested once each, accuracy is k/45, steps of 2.222%, and 95.9% and
47.4% are not on that lattice. They were never measurements. This file replaces them.

Two caveats that survive every outcome:

The average reference is computed over all 64 electrodes before any subset is
picked, so every channel carries -1/64 of every other and no subset is
electrically sealed off. The ablation bounds the artifact contribution; it cannot
eliminate it. And (c) is not "sensorimotor cortex deleted": it keeps 8 of 64 and
drops the other 56, so its collapse is confounded with an 8x cut in feature
dimension. What (c) does test is the ocular hypothesis.

The necessity arm, built 2026-07-25:

(e) and (f) are the falsifiable form: "if the decoder reads sensorimotor cortex,
deleting sensorimotor cortex must break it." Read what (e) is before its number.
It deletes the 17-channel strip, not the cortex: it keeps FC5/FC6/CP5/CP6 (four
peri-Rolandic electrodes, which (f) bounds), T8/T10/TP8 (temporalis territory,
unbounded in this repo), and POz/PO4/Oz (the peak of the strongest retained CSP
pattern). The instrument limit comes before the number: deleting the electrodes
nearest a source does not delete the source from the rest of the montage, so no
channel-deletion result licenses a source claim in either direction.

Withdrawn, kept visible (2026-07-25):

  - "Four conditions, one seed, one splitter." False: the conditions block builds
    a StratifiedKFold and a LeaveOneGroupOut, and (d)'s three per-fold values
    were the visible tell. "One seed" was true.
  - The opening "take the motor cortex away and see if the decoder dies"
    described an arm this script did not then build.
  - "So the result is not a within-session drift artifact" overclaimed: runs
    6/10/14 are one session, so (d) removes within-run drift only, never a
    session-level trend.
  - "No condition here deletes the strip and retains the rest of the montage."
    True until 2026-07-25, false once (e) and (f) landed. Kept because the
    corpus quoted it as evidence the arm was missing, and a reader who meets it
    there has to see it was retired by building the arm, not by rewording.

Withdrawn, kept visible (2026-07-26), all four from an adversarial pass, all four
defects in what the run said rather than what it computed:

  - "The complement re-referenced within its own 47 (leak removed)." False: the
    secondary arm is provably the primary minus its own across-channel mean, a
    rank-1 common-mode projection (rank 47 -> 46) whose deleted direction mixes
    the strip contribution with the complement's own global component. Its cost
    is not assignable to the leak. Both facts are now measured and asserted.
  - The McNemar power line conditioned on the observed n_disc, a random draw.
    The binding constraint is the design: b - c is identically the trial gap, so
    the smallest gap that can reach p < 0.05 is 6 trials = 13.3 points, while
    the registered G threshold is 10.0 points = 4.5 trials. Between them the
    conjunctive rule cannot fire at any discordant count. That refutes the
    pre-registration's own justification and was checkable when it was written;
    the prereg is not edited, the refutation is reported.
  - The McNemar ran at seed 42 only while G is a mean over range(10), so one
    conjunctive rule was evaluated on disjoint seed sets. It now runs on every
    sweep seed; the registered verdict stays at seed 42 with the spread beside it.
  - "No channel-count control", declared and never run. Now run as arm 10, and
    retired for the 47-vs-64 comparison only.
"""

from ablation_conditions import load_and_partition, run_conditions
from ablation_design import print_detection_floor
from ablation_secondary import (
    run_paired_mcnemar,
    run_permutation,
    run_secondary,
    run_wilson,
)
from ablation_sweep import mcnemar_per_seed, run_random_deletion, run_sweep
from ablation_verdict import run_closing, run_falsifiers, run_verdict

def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-minute experiment as a side effect."""

    # Pre-registered decision thresholds. Fixed in
    # prereg/prereg-complement-ablation.md BEFORE this script ran.
    # One trial on n = 45 is 2.222 points, two trials are 4.444. A48 already refused
    # to call two trials a difference in the direction that flattered the project;
    # the same refusal is hard-coded here in the direction that does not.
    NOISE_BAND = 100 * 2 / 45   # 4.444 points = two trials
    G_THRESHOLD = 10.0          # more than twice the noise band, more than 4.5 trials

    # POST-REGISTRATION arm 10 (added 2026-07-26): the channel-count control that the
    # pre-registration declared at 2.4(d) and risk 6 and then did not run. 50 draws
    # puts the empirical-p resolution floor at 1/51 = 0.0196, which is below alpha;
    # 30 draws would put it at 1/31 = 0.0323, also below alpha but with less headroom
    # and a coarser null. The RNG seed is fixed here so the draws are reproducible.
    N_RANDOM_DRAWS = 50
    RANDOM_DELETION_SEED = 20260726

    # SENSORIMOTOR (the FC/C/CP strip straddling the central sulcus) and FRONTOPOLAR
    # (the Fp/AF ring over the orbits, the negative control) are defined in common.py
    # and imported at the top of this file, so the ablation conditions mean the same
    # thing here, in emg_proxy.py and in test_pipeline.py.
    #
    # make_clf and wilson_interval come from there too. wilson_interval used to be
    # copied in with a note that it was copied "byte-for-byte ... because an interval
    # computed two different ways is two intervals". That reasoning was right, and one
    # definition enforces it rather than asking each copy to stay in step.

    # The stages live in the ablation_* modules beside this file, one module per
    # part of the registered analysis, split 2026-08-26. Each takes what it reads
    # and returns what later stages need; the bodies are verbatim from the
    # single-file version, so the stdout is unchanged line for line.
    (cropped, labels, groups, ch_names, n, n_hands, n_feet, majority,
     edf_paths, COMPLEMENT, WIDE, NOT_WIDE) = load_and_partition()

    _min_gap, _mg_ndisc, _mg_b, _mg_c, _mg_p = print_detection_floor(n, G_THRESHOLD)

    (skf, conditions, results, per_trial, by_name,
     ALL64, SMC, FP, COMP, NWIDE, LORO, maj_correct, _fp_off) = run_conditions(
        cropped, labels, groups, ch_names, n, n_hands, n_feet, majority,
        COMPLEMENT, NOT_WIDE)

    sweeps = run_sweep(cropped, labels, conditions, by_name, n)

    sweep_mcn, _sweep_ps, _n_fire = mcnemar_per_seed(
        cropped, labels, ch_names, COMPLEMENT, n)

    comp_seed42, comp_sweep = run_secondary(
        edf_paths, cropped, COMPLEMENT, labels, by_name, COMP, sweeps,
        n, NOISE_BAND, G_THRESHOLD)

    perm_null, null_off_50 = run_permutation(cropped, COMPLEMENT, labels,
                                             comp_seed42, n)

    run_wilson(comp_seed42, n, majority, maj_correct)

    both, only_all, only_comp, neither, n_disc, mcn_p, worst = run_paired_mcnemar(
        per_trial, ALL64, COMP, by_name, comp_seed42, n)

    _null_means, _null_G, _G_obs = run_random_deletion(
        cropped, labels, ch_names, sweeps, ALL64, comp_sweep, n,
        N_RANDOM_DRAWS, RANDOM_DELETION_SEED, COMPLEMENT)

    G, PRIOR_COMPLEMENT_SWEEP = run_falsifiers(
        by_name, results, ALL64, SMC, FP, n, sweeps, comp_sweep, comp_seed42,
        COMPLEMENT, ch_names, null_off_50, perm_null, _fp_off, maj_correct,
        mcn_p, n_disc, NOISE_BAND, skf, labels)

    run_verdict(G, NOISE_BAND, G_THRESHOLD, mcn_p, n_disc, only_all, only_comp,
                both, neither, worst, _min_gap, _sweep_ps, _n_fire, sweeps, SMC,
                ALL64, comp_sweep, comp_seed42, n, COMPLEMENT)

    run_closing(ch_names, COMPLEMENT, by_name, ALL64, SMC, FP, LORO, NWIDE, n,
                majority, maj_correct, n_hands, n_feet, sweeps, WIDE,
                _null_means, _null_G, _G_obs, N_RANDOM_DRAWS, n_disc,
                PRIOR_COMPLEMENT_SWEEP)


if __name__ == "__main__":
    main()
