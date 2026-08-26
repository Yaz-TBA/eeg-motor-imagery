"""The artifact control: refit on the channels where the EYES are loudest and see
whether the result survives there.

Genuinely didn't expect this one.. deleting the motor strip was supposed to break the
decoder and it didn't ?? 77.8% on the remaining 47 channels. Lost this prediction i guess ?
Read the caveats below before taking anything from it though, since a channel deletion
can't falsify a SOURCE hypothesis in either direction.

A scalp topography is not evidence. It is a picture of the model's weights, and
a model riding an eye-movement artifact will happily draw a picture too. The
only cheap control that actually bites is an ABLATION: refit the entire pipeline
on a channel subset that CANNOT see sensorimotor cortex, and check that the
accuracy collapses to the majority-class rate -- NOT to 50%. With 21 hands and
24 feet the do-nothing baseline is 53.3%, and a control that lands there has
failed correctly.

Six conditions, TWO splitters -- stratified 5-fold for (a)-(c), (e) and (f),
leave-one-run-out for (d), which is why (d) reports three per-fold values and
the others report five:

  (a) all 64 channels          -- reproduces the published headline
  (b) sensorimotor only        -- FC/C/CP strip. Should hold or improve.
  (c) frontopolar only         -- Fp/AF. This is where blinks and saccades are
                                  LOUDEST. If the decoder were reading the eyes,
                                  this is the subset that would keep working.
  (d) leave-one-run-out        -- all 64 channels, but folds are whole recording
                                  runs, so no fold can share a session-drift or
                                  electrode-settling trend with its training set.
  (e) sensorimotor DELETED     -- the 17-channel strip removed, the OTHER 47
                                  electrodes kept. THE NECESSITY ARM, added
                                  2026-07-25. See the second caveat below.
  (f) wide FC/C/CP DELETED     -- the 21-channel FC/C/CP block removed, 43 kept.
                                  Bounds the FC5/FC6/CP5/CP6 that (e) retains.

Seed 42 is primary, and (a)-(c), (e) and (f) also carry a ten-seed sweep over
range(10), because a seed-42 point difference is ONE quantized draw -- this file
already annotates its own +4.4 point row as "one draw, not an effect size" and
the same rule now binds the arm that does not flatter. The decision statistic
for (e) is a ten-seed mean gap AND an exact McNemar, both fixed in advance in
prereg/prereg-complement-ablation.md, which also records that
the accuracy half of (e) had already been run once and was NOT blind.

WHY THIS README TABLE EXISTS AS A SCRIPT NOW. An earlier README published these
four numbers as 91.1 / 95.9 / 47.4 / 93.3 with no script behind them. Two of
them were arithmetically impossible: with 45 trials in five equal folds of 9,
overall accuracy is a count of correct trials over 45, so it can only land on
multiples of 1/45 = 2.222%. 95.9% and 47.4% are not on that lattice -- there is
no k with k/45 = 0.959. They were not measurements. This file replaces them.

A CAVEAT THAT SURVIVES THE ABLATION. The average reference is computed across
all 64 electrodes BEFORE any subset is picked, exactly as in decode_csp.py. So
the frontopolar channels are not hermetically sealed off from occipital or
central activity -- every channel carries -1/64 of every other. The ablation
therefore bounds the artifact contribution rather than eliminating it. Making
the subsets independent would mean re-referencing each subset separately, which
would no longer be the published pipeline. Bounding is the honest claim.

A SECOND CAVEAT, ON WHAT (c) IS AND IS NOT. Frontopolar-only is not "sensorimotor
cortex deleted." It KEEPS 8 of 64 electrodes and deletes the other 56 -- occipital,
parietal and temporal along with the central strip -- so its collapse is confounded
with an 8x cut in channel count and feature dimension. What (c) does test is the
OCULAR hypothesis: the subset where blinks and saccades are loudest does not carry
the result.

THE ARM THIS FILE USED TO DECLARE ABSENT, NOW BUILT (2026-07-25). Conditions (e)
and (f) delete the strip and keep the rest, which is the falsifiable form: "if the
decoder reads sensorimotor cortex then deleting sensorimotor cortex must break it."
Read what (e) IS before reading its number. It is "the 17-channel strip deleted,"
not "sensorimotor cortex deleted": SENSORIMOTOR does not contain FC5, FC6, CP5 or
CP6, so (e) keeps four peri-Rolandic electrodes, and (f) exists to bound exactly
that leak. (e) also keeps T8/T10/TP8, which is temporalis muscle territory that
NOTHING in this repo yet bounds, and POz/PO4/Oz, the peak of the strongest
retained CSP pattern. And the instrument limit comes BEFORE the number, not after
it: no channel-deletion experiment on a 64-channel scalp montage can falsify a
SOURCE hypothesis in either direction, because deleting the electrodes nearest a
source does not delete the source from the remaining ones. These are sensor-space
measurements and they license only sensor-space claims.

WHAT THIS SCRIPT WITHDREW, KEPT VISIBLE (2026-07-25):
  - "Four conditions, one seed, one splitter." False: the conditions block below
    constructs a StratifiedKFold AND a LeaveOneGroupOut, and hands the second to
    condition (d). "One seed" is true, and (d)'s three per-fold values against
    the others' five were the visible tell all along.
  - The opening line used to read "take the motor cortex away and see if the
    decoder dies," which describes an arm this script does not build. See the
    second caveat above.
  - The printed line "so the result is not a within-session drift artifact"
    claimed more than the design supports. Runs 6, 10 and 14 are three
    recordings from ONE session, so holding out a run removes drift shared
    inside a run but not a session-level trend running across all three.
  - "No condition in this script deletes the sensorimotor strip and retains the
    rest of the montage, so the falsifiable form 'if the decoder reads
    sensorimotor cortex then deleting sensorimotor cortex must break it' is NOT
    tested here and must not be attributed to this file." True from the day it
    was written until 2026-07-25, false from the edit that added conditions (e)
    and (f). Kept visible rather than deleted because that sentence was quoted
    elsewhere in the corpus as the evidence that the arm was missing, and a
    reader who meets it there has to be able to see that it was retired by
    BUILDING the arm rather than by rewording the disclosure.

WHAT THIS SCRIPT WITHDREW, KEPT VISIBLE (2026-07-26). All four came from an
adversarial pass, all four are defects in what the run SAID rather than in what
it computed, and no measured value below changes because of them.
  - "The complement re-referenced within its own 47 (LEAK REMOVED)." False. The
    secondary arm is provably the primary minus its own across-channel mean, a
    RANK-1 COMMON-MODE PROJECTION that drops the rank 47 to 46. The direction it
    deletes carries the average-referenced strip contribution AND the
    complement's own global component, so its cost is not assignable to the leak.
    Both the identity and the rank drop are now measured and asserted in the run.
  - The McNemar power line conditioned only on the observed n_disc, which is a
    random draw. The binding constraint is the DESIGN: in a paired 2x2 on the
    same 45 trials, b - c is identically the trial gap, so the smallest gap that
    can ever reach p < 0.05 is 6 trials = 13.3 points, while the registered G
    threshold is 10.0 points = 4.5 trials. The conjunctive rule cannot fire in
    between, at any discordant count. The pre-registration's stated reason for
    the rule ("at a gap of 10 or more points the McNemar should fire
    comfortably") is refuted by that arithmetic and was checkable when it was
    written. The pre-registration is NOT edited; the refutation is reported.
  - The McNemar was computed at seed 42 only, while G is a mean over range(10),
    so the two halves of one conjunctive rule were evaluated on DISJOINT seed
    sets. The test now runs on every sweep seed. The registered verdict stays at
    seed 42, where it was registered, and the spread is printed beside it.
  - "NO CHANNEL-COUNT CONTROL", declared as a live confound in this file and in
    the pre-registration and never run. It is now run, as arm 10, and it is
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
