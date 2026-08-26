"""The registered analysis-falsifiers of ablate_channels.py, the pre-registered
verdict with its bands, and the caveats printed in every band. Split out 2026-08-26;
the stage bodies are verbatim, and the band edges stay exactly as registered."""

import ablation_data  # noqa: F401  -- installs the common.py path first

import numpy as np

from common import FRONTOPOLAR, SENSORIMOTOR
from ablation_data import RUNS, SEED, SEEDS
from ablation_design import ALPHA


def run_falsifiers(by_name, results, ALL64, SMC, FP, n, sweeps, comp_sweep,
                   comp_seed42, COMPLEMENT, ch_names, null_off_50, perm_null,
                   _fp_off, maj_correct, mcn_p, n_disc, NOISE_BAND, skf, labels):
    # --- the registered analysis-falsifiers, all ten, printed --------------------
    # These fire when the MEASUREMENT is broken, not when the hypothesis moved. If any
    # fires, no number above is quotable until it is resolved. Printed rather than
    # asserted where the registered response is "report both and resolve", because an
    # assert would prevent the reporting the pre-registration requires.
    print("\n--- Registered analysis-falsifiers (prereg section 7) ---")
    PRIOR_COMPLEMENT_SWEEP = 0.793     # hostile_verify_A.stdout line 17, uncommitted
    ONE_TRIAL = 100 / n                # 2.222 points
    checks = []


    def check(idx, label, ok, detail):
        checks.append((idx, label, ok, detail))
        print(f"({idx}) {'PASS ' if ok else 'FIRED'} {label}: {detail}")


    check(1, "all-64 reproduces 41/45 at seed 42",
          by_name[ALL64][3] == 41,
          f"got {by_name[ALL64][3]}/{n} = {by_name[ALL64][3]/n:.1%}, "
          f"expected 41/{n} = {41/n:.1%}")
    check(2, "sensorimotor 43/45 and frontopolar 23/45 at seed 42",
          by_name[SMC][3] == 43 and by_name[FP][3] == 23,
          f"got {by_name[SMC][3]}/{n} and {by_name[FP][3]}/{n}, expected 43/45 and 23/45")
    check(3, "COMPLEMENT is 47 and partitions the montage with SENSORIMOTOR",
          len(COMPLEMENT) == 47
          and set(SENSORIMOTOR) | set(COMPLEMENT) == set(ch_names)
          and not (set(SENSORIMOTOR) & set(COMPLEMENT)),
          f"{len(SENSORIMOTOR)} + {len(COMPLEMENT)} = {len(ch_names)}, disjoint")
    _lattice_ok = all(abs(r[2].mean() - r[3] / n) < 1e-9 for r in results)
    check(4, "every accuracy on the k/45 lattice, fold-mean == pooled count",
          _lattice_ok,
          f"all {len(results)} conditions checked, plus every one of the "
          f"{len(SEEDS)} per-seed values in the sweep")
    _delta_prior = 100 * abs(comp_sweep.mean() - PRIOR_COMPLEMENT_SWEEP)
    check(5, "complement ten-seed mean within one trial of the prior uncommitted run",
          _delta_prior <= ONE_TRIAL + 1e-9,
          f"this run {comp_sweep.mean():.1%}, prior (uncommitted) "
          f"{PRIOR_COMPLEMENT_SWEEP:.1%}, difference {_delta_prior:.1f} points, "
          f"one trial = {ONE_TRIAL:.3f}")
    check(6, "complement is above the frontopolar-8 floor",
          comp_sweep.mean() > sweeps[FP].mean() and comp_seed42 > by_name[FP][3],
          f"complement {comp_sweep.mean():.1%} ten-seed / {comp_seed42}/{n} seed42 vs "
          f"frontopolar {sweeps[FP].mean():.1%} / {by_name[FP][3]}/{n}")
    check(7, "permutation null centred within 5 points of 50%",
          null_off_50 <= 5.0, f"null at {perm_null.mean():.1%}, {null_off_50:.1f} points off")
    check(8, "negative control stays dead (within 3 trials of majority)",
          _fp_off <= 3, f"frontopolar is {_fp_off} trial(s) off {maj_correct}/{n}")
    G = 100 * (sweeps[ALL64].mean() - comp_sweep.mean())
    check(9, "band E did not land (complement does not beat all 64 by > two trials)",
          G >= -NOISE_BAND, f"G = {G:+.1f} points, band-E edge is {-NOISE_BAND:.3f}")
    # SCOPE, and it is the registered scope, not a convenient one. Falsifier (10)
    # fires only in bands A and B: "if G lands in A or B and the McNemar does NOT
    # reach p < 0.05, that is a red flag about the pairing." Band C is NOT in that
    # scope, because band C already registers its own handling of exactly this
    # configuration: "if McNemar gives p >= 0.05, the verdict downgrades to band C2's
    # wording even though G cleared." A failed McNemar in band C is a REGISTERED
    # OUTCOME, not a broken measurement.
    #
    # WRITTEN DOWN BECAUSE IT WAS GOT WRONG ONCE. The first version of this check
    # fired whenever G >= 10.0 and p >= 0.05, which is wider than the pre-registration
    # and would have declared a registered band-C downgrade to be an analysis failure.
    # It was corrected against the prereg text AFTER the first run, which is a change
    # made with the answer visible, so it is recorded here rather than quietly fixed.
    # It changes no measured value: the 2x2, the p, G and every accuracy are identical
    # either way. What it changes is whether this run is reportable at all.
    _band_AB = G > 22.9
    check(10, "in band A or B, the McNemar also reaches p < 0.05 (scoped to A/B)",
          not (_band_AB and mcn_p >= ALPHA),
          f"G = {G:+.1f} is {'in' if _band_AB else 'NOT in'} band A/B; McNemar "
          f"p = {mcn_p:.4f} on n_disc = {n_disc}"
          + ("" if _band_AB else ", so this check does not apply"))
    # The pairing diagnostic falsifier (10) tells you to run. Cheap, so it runs
    # unconditionally rather than only when the falsifier fires.
    _folds_all = [te.tolist() for _, te in skf.split(np.zeros((n, 1)), labels)]
    _folds_again = [te.tolist() for _, te in skf.split(np.zeros((n, 1)), labels)]
    print(f"      pairing diagnostic: StratifiedKFold(random_state={SEED}) returns "
          f"identical test folds on repeated calls: {_folds_all == _folds_again}. "
          f"Both arms were scored on it, and for each arm cross_val_predict's pooled "
          f"count equals cross_val_score's fold-mean (asserted above), so the 2x2 is "
          f"a genuine per-trial pairing.")
    n_fired = sum(1 for c in checks if not c[2])
    print(f"{len(checks) - n_fired} of {len(checks)} passed. "
          + ("No analysis-falsifier fired; the numbers above are quotable."
             if n_fired == 0 else
             f"{n_fired} FIRED. Nothing above is quotable until they are resolved."))
    return G, PRIOR_COMPLEMENT_SWEEP


def run_verdict(G, NOISE_BAND, G_THRESHOLD, mcn_p, n_disc, only_all, only_comp,
                both, neither, worst, _min_gap, _sweep_ps, _n_fire, sweeps, SMC,
                ALL64, comp_sweep, comp_seed42, n, COMPLEMENT):
    # --- THE PRE-REGISTERED VERDICT ------------------------------------------------
    # G and the bands were fixed in writing before this script existed. The band is
    # selected by arithmetic here, not by reading, so the edge cannot be resolved in
    # the project's favour after the fact.
    print("\n--- THE PRE-REGISTERED VERDICT ---")
    print(f"G = (all-64 ten-seed mean) - (complement ten-seed mean) = "
          f"{100*sweeps[ALL64].mean():.1f} - {100*comp_sweep.mean():.1f} = "
          f"{G:+.1f} points.")
    print(f"Two trials on n = {n} is {NOISE_BAND:.3f} points (the prereg rounds this "
          f"to 4.4). The pre-committed threshold for a real loss is G > "
          f"{G_THRESHOLD:.1f} points AND McNemar p < {ALPHA}.")

    # Edges resolve DOWNWARD in G, to the band that claims LESS about the strip being
    # necessary. Fixed in advance so a boundary cannot be argued afterwards.
    if G > 38.4:
        band, band_name = "A", "COLLAPSE"
    elif G > 22.9:
        band, band_name = "B", "SEVERE LOSS"
    elif G > G_THRESHOLD:
        band, band_name = "C", "SUBSTANTIAL LOSS, NOT NECESSARY"
    elif G > NOISE_BAND:
        band, band_name = "C2", "SUGGESTIVE, NOT ESTABLISHED"
    elif G >= -NOISE_BAND:
        band, band_name = "D", "MATCH"
    else:
        band, band_name = "E", "EXCEEDS"

    mcnemar_fired = mcn_p < ALPHA
    print(f"BAND {band}: {band_name}. McNemar p = {mcn_p:.4f} on n_disc = {n_disc}, "
          f"which {'DOES' if mcnemar_fired else 'does NOT'} reach p < {ALPHA}.")

    if band in ("A", "B", "C") and mcnemar_fired:
        if band == "A":
            print("VERDICT: the 47 non-strip electrodes carry nothing. The sensorimotor")
            print("strip is NECESSARY as well as sufficient, and the falsifiable form")
            print("survives its test. Registered as SUSPECT-FIRST all the same: a")
            print("47-channel montage including the whole posterior and temporal rings")
            print("falling to the floor is not a plausible neurophysiological result.")
        elif band == "B":
            print("VERDICT: the complement decodes above the majority floor but is")
            print("crippled. The signal is strongly concentrated over the strip, and")
            print("there is a REAL RESIDUAL elsewhere that is disclosed, not rounded to")
            print("zero. The residual cannot be attributed to posterior cortex until an")
            print("EMG bound exists, because the complement retains T8, T10 and TP8.")
        else:
            print("VERDICT: the sensorimotor strip is SUFFICIENT BUT NOT NECESSARY.")
            print("The falsifiable form 'if it reads sensorimotor cortex, deleting")
            print("sensorimotor cortex must break it' is FALSE AT THIS INSTRUMENT.")
            print(f"{len(COMPLEMENT)} electrodes that exclude the strip still reach "
                  f"{comp_sweep.mean():.1%} over ten seeds, {comp_seed42}/{n} at seed "
                  f"{SEED}.")
            print("The framing this project leans on gets WEAKER, and that is what is")
            print("written. The one permitted positive sentence is the DENSITY one, and")
            print("it is a sensor-space claim, not a source claim:")
            print(f"  {len(SENSORIMOTOR)} channels reach {sweeps[SMC].mean():.1%} and "
                  f"the {len(COMPLEMENT)} that exclude them reach "
                  f"{comp_sweep.mean():.1%}, so per-channel")
            print(f"  discriminative density is roughly "
                  f"{(sweeps[SMC].mean()/len(SENSORIMOTOR))/(comp_sweep.mean()/len(COMPLEMENT)):.1f}x "
                  f"higher over the strip.")
    elif band in ("A", "B", "C") and not mcnemar_fired:
        print(f"VERDICT DOWNGRADED, by the rule and not by preference. G = {G:+.1f} "
              f"cleared the")
        print(f"{G_THRESHOLD:.1f}-point threshold; the McNemar did not reach "
              f"p < {ALPHA}. The registered rule")
        print("requires BOTH halves and forbids upgrading on whichever reads better.")
        print(f"REGISTERED VERDICT: a loss is SUGGESTED and NOT ESTABLISHED at n = {n}.")
        print("Reported as undecided, leaned neither way.")
        print(f"NOT ESTABLISHED BY THIS RUN, therefore not written: that the strip is")
        print("'sufficient but not necessary'. G points that way and the test that was")
        print("registered to confirm it did not. Both halves get said, or neither does.")
        if band == "C":
            # Band C registers this path explicitly, so it is a registered outcome and
            # not an analysis failure. Say why the McNemar failed, in the terms the
            # prereg fixed in advance: the count, then the power at that count.
            print(f"WHY IT FAILED, at the discordant count it actually got: n_disc = "
                  f"{n_disc} ({only_all} vs {only_comp}).")
            if worst is not None and worst[0] == only_all:
                print(f"      The observed split IS the most lopsided split that misses "
                      f"p < {ALPHA} at")
                print(f"      n_disc = {n_disc}. Only a {n_disc}-0 sweep would have "
                      f"reached it. This design had")
                print("      essentially no power here, and that is a property of n = 45")
                print("      with a strongly agreeing pair, not evidence for either side.")
            print(f"      {both} of {n} trials were called correctly by BOTH arms and "
                  f"{neither} by neither,")
            print("      which is what leaves the discordant count small. The pairing is")
            print("      working as designed; it is the sample that is small.")
            # ADDED 2026-07-26. The four sentences above are true and they are not
            # the whole truth. They describe the discordant count as if it were the
            # binding constraint. It is not: the rule itself cannot fire here.
            print(f"WHAT THE FOUR LINES ABOVE LEAVE OUT, and it is the larger part:")
            print(f"      (1) THE RULE CANNOT FIRE AT ITS OWN THRESHOLD. The McNemar "
                  f"half needs a")
            print(f"          {_min_gap}-trial = {100*_min_gap/n:.1f}-point gap at "
                  f"minimum, at any n_disc. The registered G half fires at")
            print(f"          {G_THRESHOLD:.1f} points = {G_THRESHOLD*n/100:.2f} "
                  f"trials. Between them the conjunction is unreachable by")
            print(f"          construction. This is a defect in the rule, and the "
                  f"pre-registration's stated")
            print(f"          justification for the rule ('at a gap of "
                  f"{G_THRESHOLD:.0f} or more points the McNemar should")
            print(f"          fire comfortably', section 6.2) is REFUTED by this "
                  f"arithmetic.")
            print(f"      (2) THE VERDICT IS NOT ROBUST TO SEED CHOICE. Seed {SEED} "
                  f"gives p = {mcn_p:.4f}. The same")
            print(f"          exact test on the {len(SEEDS)} seeds of this script's "
                  f"OWN registered sweep gives a")
            print(f"          median of {np.median(_sweep_ps):.4f}, with "
                  f"{_n_fire} of {len(SEEDS)} reaching p < {ALPHA}. Seed {SEED} is worse")
            print(f"          for the loss than {int((_sweep_ps < mcn_p).sum())} of "
                  f"those {len(SEEDS)} seeds and worse than their median.")
            print(f"          So 'not established' is substantially a fact about "
                  f"which integer was")
            print(f"          typed as random_state, not about the recording.")
            print(f"      (3) THE CONJUNCTION IS EVALUATED ON DISJOINT SEED SETS. G "
                  f"is a mean over")
            print(f"          {list(SEEDS)}; the McNemar is at seed {SEED}, which is "
                  f"not in that list.")
            print(f"      NONE OF THIS FLIPS THE VERDICT. Ten non-independent "
                  f"re-splits of the same {n}")
            print(f"      trials are not ten samples, so 'ESTABLISHED' is not "
                  f"licensed either. What is")
            print(f"      established is that the registered rule, as written, "
                  f"CANNOT certify band C at")
            print(f"      n = {n}, and that the reported outcome is a property of "
                  f"the rule at least as much")
            print(f"      as a reading of the data.")
        if band in ("A", "B"):
            print("At a gap this size the McNemar should have fired comfortably, so this")
            print("is a RED FLAG about the pairing rather than a licence to report the")
            print("gap anyway. See falsifier (10).")
    elif band == "C2":
        print(f"VERDICT: the complement is below all-64 by more than two trials but by")
        print(f"less than the pre-committed {G_THRESHOLD:.1f}-point threshold. A loss is")
        print(f"SUGGESTED and NOT ESTABLISHED at n = {n}. It is NOT upgraded with the")
        print("McNemar, because the rule requires both halves and this band fails the G")
        print("half by construction. Reported as undecided, leaned neither way.")
    elif band == "D":
        print("VERDICT: the complement is NOT DISTINGUISHABLE from the full montage.")
        print("The sensorimotor strip contributes nothing the rest of the montage does")
        print("not already carry. Every sensorimotor framing in this corpus weakens, and")
        print("the sufficiency arm is reduced to 'these 17 channels are one of several")
        print("sets that work'. A48 binds here: this corpus already refuses to call two")
        print("trials a difference in the direction that flattered it, so it refuses in")
        print("this direction too. 'No difference DETECTED' is not 'no difference'.")
    else:
        print("VERDICT WITHHELD. Band E: the complement BEATS the full montage by more")
        print("than two trials. Registered as SUSPECT FIRST, not as a result. Check the")
        print("channel picking, check that CSP is inside the fold, check the crop, and")
        print("re-run before reporting anything. Only reportable once those pass, and")
        print("then it is bad news for the framing.")


def run_closing(ch_names, COMPLEMENT, by_name, ALL64, SMC, FP, LORO, NWIDE, n,
                majority, maj_correct, n_hands, n_feet, sweeps, WIDE,
                _null_means, _null_G, _G_obs, N_RANDOM_DRAWS, n_disc,
                PRIOR_COMPLEMENT_SWEEP):
    # --- printed in EVERY band, without exception ---------------------------------
    print("\nTRUE IN EVERY BAND (1): THE AVERAGE-REFERENCE LEAK. The reference is")
    print(f"computed over all {len(ch_names)} electrodes BEFORE the "
          f"{len(COMPLEMENT)} are picked, so every")
    print(f"complement channel carries -1/{len(ch_names)} of every sensorimotor "
          f"channel. The complement is")
    print("NOT electrically independent of the strip. This measurement BOUNDS the")
    print("strip's necessity; it cannot establish it. A high complement score is")
    print("partly what volume conduction plus a 64-channel average reference PREDICTS.")
    print("TRUE IN EVERY BAND (2): THE INSTRUMENT LIMIT. No band licenses a SOURCE")
    print("claim in either direction. Forward-is-not-inverse refutes a negative source")
    print("claim exactly as hard as a positive one, so 'the strip is not necessary'")
    print("is a statement about 64 electrodes, not about cortex.")

    print("\n--- What this does and does not show ---")
    all64 = by_name[ALL64][3] / n
    smc = by_name[SMC][3] / n
    fp = by_name[FP][3] / n
    loro = by_name[LORO][3] / n
    fp_correct = by_name[FP][3]
    print(f"Frontopolar-only lands at {fp:.1%} ({fp_correct}/{n}) against a "
          f"majority-class rate of {majority:.1%} ({maj_correct}/{n}).")
    gap = abs(fp_correct - maj_correct)
    # EVERY NUMBER IN THIS BLOCK IS INTERPOLATED, and that is the point of the
    # rewrite. Until 2026-07-25 the lines below were bare print() calls with no
    # f-prefix, carrying the literals "21/24", "51.1%", "one trial" and "0.33 to
    # 0.78". They agreed with the table above by construction rather than by
    # measurement, and would have kept printing the old values if the ablation ever
    # moved -- which also means a provenance check that looks for a number in some
    # script's stdout would have found these no matter what the pipeline did.
    fp_folds = by_name[FP][2]
    print(f"That is {gap} trial{'' if gap == 1 else 's'} off the rate you get by ignoring")
    print("the EEG entirely and always answering 'feet'. The frontopolar")
    print("decoder has no usable signal. Note the framing: the honest reference here")
    print(f"is the MAJORITY rate, not 50% -- with {n_hands}/{n_feet} classes, a "
          f"{fp:.1%} result is not")
    print(f"'above chance', it is a degenerate classifier {gap} "
          f"trial{'' if gap == 1 else 's'} short of guessing.")
    print(f"The per-fold spread ({fp_folds.min():.2f} to {fp_folds.max():.2f}) is "
          f"the other tell: folds that wide are")
    print("a coin, not a decoder.")
    # SINGLE-SEED DRAW, not an effect size. The gain printed on the next line is one
    # seed-42 partition. A 20-seed reference on the same two conditions puts the
    # expected difference near +1 point, so this row is roughly 4x the expectation
    # and its DIRECTION is what the write-up treats as inside noise. Read it as one
    # draw, not as "dropping non-motor channels buys you 4 points."
    print(f"Sensorimotor-only ({smc:.1%}) vs. all 64 ({all64:.1%}): "
          f"{100*(smc-all64):+.1f} points from dropping "
          f"{64-len(SENSORIMOTOR)} non-motor channels (one seed, one partition).")
    print(f"Leave-one-run-out ({loro:.1%}) holds up with no trial sharing a run with")
    print("its training set, so the result is not an artifact of drift shared INSIDE")
    print(f"a run. It does not rule out a session-level trend: runs {RUNS} are three")
    print("recordings from one session, and a drift monotonic across all three")
    print("survives this control. EEGMMIDB has no second session to test against.")
    print("\nBOUND, NOT PROOF: the average reference is computed over all 64 channels")
    print("before picking, so the subsets are not electrically independent, and")
    print("EEGMMIDB ships no EOG channel to regress out. This ablation bounds the")
    print("ocular contribution; it cannot measure it.")

    # --- WHAT THE COMPLEMENT ARM DOES NOT SHOW ------------------------------------
    # Registered in advance, printed unconditionally, so none of it can be discovered
    # later and presented as a caveat the author thought of afterwards.
    print("\n--- WHAT THE COMPLEMENT ARM DOES NOT SHOW ---")
    print(f"(i)   THE AVERAGE-REFERENCE LEAK. Reference over all {len(ch_names)}, "
          f"then pick {len(COMPLEMENT)}. Every")
    print(f"      complement channel carries -1/{len(ch_names)} of every deleted "
          f"channel. BOUNDS, not proves.")
    print("(ii)  THE INSTRUMENT LIMIT. Deleting the electrodes nearest a source does")
    print("      not delete the source from the remaining electrodes. No")
    print("      channel-deletion result on a 64-channel scalp montage falsifies a")
    print("      source hypothesis, in EITHER direction. Sensor-space only.")
    print(f"(iii) THIS IS NOT 'SENSORIMOTOR CORTEX DELETED'. The complement keeps "
          f"{[c for c in COMPLEMENT if c in ('FC5', 'FC6', 'CP5', 'CP6')]},")
    print(f"      four peri-Rolandic electrodes. The wide-{len(WIDE)} arm above deletes "
          f"them too and lands")
    print(f"      at {by_name[NWIDE][3]}/{n} = {by_name[NWIDE][3]/n:.1%} at seed "
          f"{SEED}, {sweeps[NWIDE].mean():.1%} over ten seeds. That is the bound.")
    print(f"(iv)  UNBOUNDED EMG. The complement keeps "
          f"{[c for c in COMPLEMENT if c in ('T8', 'T10', 'TP8')]}, temporalis muscle")
    print("      territory, and NOTHING in this repo bounds a myogenic contribution.")
    print("      The permitted sentence is 'the non-strip electrodes decode above the")
    print("      floor, and how much of that is muscle is NOT yet bounded'. The")
    print("      sentence 'posterior cortex also decodes' is BLOCKED until it is.")
    print(f"(v)   CHANNEL-COUNT CONTROL: MEASURED 2026-07-26, PARTIALLY RETIRED. This "
          f"line used to")
    print(f"      read 'NO CHANNEL-COUNT CONTROL' and declare {len(COMPLEMENT)} vs "
          f"{len(SENSORIMOTOR)} vs {len(ch_names)} channels an")
    print("      unmeasured confound. It is now measured for the "
          f"{len(COMPLEMENT)}-vs-{len(ch_names)} comparison: arm 10")
    print(f"      deletes {len(SENSORIMOTOR)} channels at random {N_RANDOM_DRAWS} "
          f"times and the ten-seed mean barely moves")
    print(f"      ({_null_means.mean():.1%}, G null mean {_null_G.mean():+.1f} "
          f"points), while deleting the strip costs {_G_obs:+.1f}.")
    print("      So channel count does not explain the complement arm's deficit. The")
    print(f"      confound STANDS UNMEASURED for the {len(SENSORIMOTOR)}-channel and "
          f"{len(FRONTOPOLAR)}-channel arms, which keep")
    print("      far fewer channels than arm 10 ever does. Channel counts still print")
    print("      beside every row.")
    print(f"(vi)  n = {n}, ONE SUBJECT, ONE SESSION, three runs. Ten seeds is a small")
    print("      sample of splits, which is why the range prints beside every mean,")
    print("      and the McNemar sits on a discordant count of "
          f"{n_disc}, which is why that count")
    print("      travels with its p everywhere it is quoted.")
    print("(vii) THE ACCURACY HALF OF THIS ARM WAS NOT BLIND. It had been run once,")
    print("      uncommitted, before the pre-registration was written, and the prior")
    print(f"      value ({PRIOR_COMPLEMENT_SWEEP:.1%} over ten seeds) is checked "
          f"against this run at falsifier (5).")
    print("      The permutation test, the Wilson interval and the McNemar are blind.")
