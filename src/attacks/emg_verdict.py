"""Section 9 of emg_proxy.py: the pre-registered reading of the primary cell, the
spatial profile, arm (a) against arm (b), the limitations that survive every
outcome, and the word this run is allowed to use. Split out 2026-08-26; the body
is verbatim from that file."""

import time

from emg_setup import ALPHA, INTERMITTENT_FRACTION, hr, sub


def run_verdict(D, K_PRIMARY, PRIMARY_CELL, primary_results, null_stats,
                p_perm_str, agg_t_p, agg_u_p, agg_d, ARM_A_POSITIVE,
                MDE_AGGREGATE, MDE_PERCHANNEL, LADDER_FAILED, thr_by_topo,
                CONT_SHAPE_BOUND, ALL_TOPOS, T_START):
    acc_str, FLOOR, MAJ_CORRECT, N = D.acc_str, D.FLOOR, D.MAJ_CORRECT, D.N
    BANDS, NYQ = D.BANDS, D.NYQ

    # =============================================================================
    # 9. What this does and does not show
    # =============================================================================
    hr("9. WHAT THIS DOES AND DOES NOT SHOW")

    sub("The primary cell, declared primary before the run")
    print("TEMPORAL ring, 40-75 Hz with 60 Hz notched, arm (b), imagery window.")
    print("Everything else in this script is descriptive. This is stated so that four")
    print("channel sets by four bands by two arms cannot be mined for whichever cell")
    print("reads best.")
    print(f"\n  PRIMARY CELL = {acc_str(K_PRIMARY)}   against the majority floor "
          f"{FLOOR:.1%} ({MAJ_CORRECT}/{N})")
    print(f"  permutation p {p_perm_str(PRIMARY_CELL['p_perm'])}, "
          f"binomial p = {PRIMARY_CELL['p_binom']:.4f}")

    perm_sig = PRIMARY_CELL["p_perm"] is not None and PRIMARY_CELL["p_perm"] <= ALPHA
    if K_PRIMARY <= 20:
        verdict = ("<= 20/45. Clearly below the majority floor. No usable signal, the "
                   "same reading ablate_channels.py gives its 23/45 frontopolar row. "
                   "NOT evidence of anti-information: at n = 45 a value this low is a "
                   "coin, and the correct description is a degenerate classifier, not "
                   "an inverted one.")
    elif K_PRIMARY <= 24:
        verdict = ("<= 24/45, at or below the majority floor. NO decodable class "
                   "information in muscle-band power at muscle-territory electrodes.")
    elif K_PRIMARY <= 29 and not perm_sig:
        verdict = ("25/45 to 29/45 with permutation p > 0.05. Above the floor but not "
                   "distinguishable from noise at n = 45. AMBIGUOUS, and reported as "
                   "ambiguous, not as a null. The exposure is NARROWED, not closed, and "
                   "the honest label stays 'partially bounded' rather than 'controlled'.")
    elif K_PRIMARY <= 32 and perm_sig:
        verdict = ("30/45 to 32/45 with permutation p <= 0.05. A real but modest "
                   "class-correlated high-band temporal signal exists. 30/45 is "
                   "MARGINAL, not detection. This does not invalidate 91.1%, but the "
                   "honest statement changes from 'nothing bounds an EMG contribution' "
                   "to 'a muscle-band, muscle-location decoder reaches this accuracy'.")
    elif K_PRIMARY <= 39 and perm_sig:
        verdict = ("33/45 to 39/45 with permutation p <= 0.05. A SERIOUS CONFOUND. A "
                   "decoder with no access to mu or beta carries a large share of the "
                   "information the project attributes to motor imagery. The framing "
                   "that the frontopolar ablation addresses artifact contamination must "
                   "be WITHDRAWN as insufficient.")
    elif K_PRIMARY >= 40:
        verdict = (">= 40/45. THE WORST CASE. The probe matches or beats the "
                   "headline while blind to the band the headline claims to use. Treat "
                   "the headline as UNSUPPORTED pending a dataset with EOG and EMG "
                   "reference channels.")
    else:
        verdict = (f"{K_PRIMARY}/{N} with permutation p "
                   f"{p_perm_str(PRIMARY_CELL['p_perm'])}. Above the floor without "
                   "permutation significance. AMBIGUOUS, and reported as ambiguous.")
    print(f"\n  PRE-REGISTERED READING: {verdict}")

    sub("Spatial profile, primary band, arm (b)")
    # CORRECTED 2026-07-26. `above` compares to the MAJORITY FLOOR, which is what a
    # constant predictor scores, not what this pipeline scores under H0. The floor
    # comparison is kept because the pre-registration's branch table is written in
    # terms of it, but each row now also carries its position in ITS OWN null, which
    # is the comparison that answers "did this probe underperform chance".
    above = {name: r["k"] > MAJ_CORRECT for name, r in primary_results}
    for name, r in primary_results:
        ns = null_stats[name]
        print(f"  {name:<14} {acc_str(r['k'])}  perm p {p_perm_str(r['p_perm'])}  "
              f"{'ABOVE floor' if above[name] else 'at or below floor'}  "
              f"| own null median {ns['med']:.1f}/{N}, observed at the "
              f"{ns['pct_lt']:.1f} to {ns['pct_le']:.1f} percentile")
    if not any(above.values()):
        print("\n  All four sets at or below the MAJORITY FLOOR.")
        print("  Read against the FLOOR that phrasing invites 'the probe underperformed")
        print("  chance'. Read against each set's OWN null it says something different,")
        print("  and the null is the correct reference for this pipeline:")
        _at_null = [nm for nm in null_stats
                    if 25.0 <= null_stats[nm]["pct_le"] and null_stats[nm]["pct_lt"] <= 75.0]
        _low_tail = [nm for nm in null_stats if null_stats[nm]["pct_le"] < 25.0]
        print(f"    at chance for this pipeline (25th to 75th percentile of own null): "
              f"{_at_null if _at_null else 'none'}")
        print(f"    in the LOWER tail of own null (below the 25th percentile): "
              f"{_low_tail if _low_tail else 'none'}")
        if _low_tail:
            print("\n  THIS IS NOT THE 'NO INFORMATION ANYWHERE' OUTCOME AS WRITTEN.")
            print("  No-information predicts performance AT the null. What is measured is")
            print(f"  performance systematically BELOW it at {len(_low_tail)} of "
                  f"{len(null_stats)} channel sets. That is a")
            print("  different phenomenon and this run does NOT diagnose it. The")
            print("  one-sided permutation test above is structurally incapable of seeing")
            print("  it, because sklearn scores P(null >= observed) only.")
            print("  NO MECHANISM IS OFFERED HERE. Naming a cause in the same breath as")
            print("  the number is this project's round-one failure mode. It is recorded")
            print("  as an unexplained systematic property of the instrument and it needs")
            print("  its own pre-registration. See limitation 12.")
        print("\n  What survives regardless: no channel set shows high-band class")
        print("  information ABOVE its own null, so the bound is set entirely by the")
        print("  injection ladder rather than by the observed accuracy.")
    elif above["TEMPORAL"] and not above["FRONTOPOLAR"] and not above["SENSORIMOTOR"]:
        print("\n  Localised to temporal territory. Consistent with temporalis EMG, and")
        print("  ALSO consistent with any right-lateralized source this design cannot")
        print("  distinguish from it.")
    elif above["TEMPORAL"] and above["FRONTOPOLAR"] and not above["SENSORIMOTOR"]:
        print("\n  Anterior and lateral, not central. The leading candidate is the")
        print("  SACCADIC SPIKE POTENTIAL, plausible here because the cue is")
        print("  position-confounded with the label. This must NOT be reported as EMG.")
    elif all(above.values()):
        print("\n  A GLOBAL broadband class difference. Candidates: global muscle tone,")
        print("  arousal, or amplifier gain drift correlated with block structure. The")
        print("  least specific and the most alarming outcome.")
    elif above["SENSORIMOTOR"] and not above["TEMPORAL"]:
        print("\n  Possibly genuine sensorimotor high-gamma, which is a FINDING and not")
        print("  a confound, and must not be reported as EMG. It would still need its")
        print("  own artifact control, and this design provides none for it.")
    elif above["FRONTOPOLAR"] and not above["TEMPORAL"]:
        print("\n  Ocular, not muscular. It would mean the existing frontopolar")
        print("  ablation's null at 8-30 Hz was band-limited rather than conclusive.")
    else:
        print("\n  A mixed profile not matched by a single pre-registered pattern.")
        print("  Reported as-is rather than assigned to the nearest one.")

    sub("Arm (a) against arm (b)")
    b_above = K_PRIMARY > MAJ_CORRECT
    print(f"  arm (a) aggregate: {'POSITIVE' if ARM_A_POSITIVE else 'null'} "
          f"(t p = {agg_t_p:.4f}, U p = {agg_u_p:.4f}, d = {agg_d:+.3f})")
    print(f"  arm (b) primary  : {acc_str(K_PRIMARY)}, "
          f"{'above floor' if b_above else 'at or below floor'}")
    if not ARM_A_POSITIVE and not b_above:
        print("\n  Consistent across both instruments. The bound is reported at Cohen's")
        print(f"  d = {MDE_AGGREGATE:.3f} for the aggregate test and d = "
              f"{MDE_PERCHANNEL:.3f} per channel at the")
        print("  Bonferroni floor, PLUS the ladder's amplitude bound from arm (b).")
        print("  Both effect-size floors are LARGE effects, and that is the honest")
        print("  limit of what a null here can say.")
    elif ARM_A_POSITIVE and not b_above:
        print("\n  High-band temporal power differs by class but is not linearly")
        print("  separable at this n. A PARTIAL exposure, reported as partial. The")
        print("  presence of a power difference is the more conservative reading and it")
        print("  governs the write-up.")
    elif not ARM_A_POSITIVE and b_above:
        print("\n  A multivariate effect with no univariate marginal, which is exactly")
        print("  what CSP is built to find. ARM (a)'s NULL DOES NOT RESCUE ARM (b).")
        print("  Arm (b) is the sharper test and it governs. Reporting the (a) null as")
        print("  the headline here would be the precise failure the pre-registration")
        print("  was written to prevent.")
    else:
        print("\n  Confound confirmed on both instruments. No ambiguity left to report")
        print("  and no framing available that softens it.")

    sub("The limitations that survive every possible outcome")
    print("  1. SPECTRAL SCOPE, corrected 2026-07-26. This limitation used to read")
    print("     'even a perfect null bounds only the RECORDED part of the spectrum',")
    print(f"     which names the wrong boundary in the direction that overstates")
    print(f"     coverage: the recorded spectrum is 0 to {NYQ:.0f} Hz and this probe "
          f"covers")
    print(f"     {BANDS['PRIMARY']['l_freq']:.0f} to {BANDS['PRIMARY']['h_freq']:.0f} "
          f"Hz minus 56 to 64 Hz. Two truncations, not one:")
    print(f"     160 Hz sampling discards everything above the {NYQ:.0f} Hz Nyquist, "
          f"where surface")
    print("     temporalis EMG has substantial power, AND this probe declines the")
    print(f"     {BANDS['PRIMARY']['l_freq']:.0f} Hz below its own lower edge. Nothing "
          f"at all is bounded inside the")
    print("     decoder's own 8 to 30 Hz passband, which is the only band the headline")
    print("     can be contaminated in.")
    print("  2. The average reference is computed over all 64 channels BEFORE any")
    print("     subset is picked, so every channel carries -1/64 of every other and the")
    print("     temporal ring is NOT electrically sealed off. This BOUNDS a")
    print("     contribution; it does not isolate one.")
    print("  3. EEGMMIDB ships NO EOG and NO EMG reference channel. There is no ground")
    print("     truth for 'this is muscle'. This probe measures high-band power at")
    print("     muscle-adjacent scalp sites. It does not measure muscle.")
    print("  4. A positive cannot distinguish temporalis EMG from a saccadic spike")
    print("     potential, and the position-confounded cue makes the latter plausible.")
    print("  5. n = 45. Every accuracy here has a Wilson interval well over 20 points")
    print("     wide, and a four-trial difference between conditions is noise.")
    print("  6. Single subject, single session. Runs 6, 10 and 14 are three recordings")
    print("     from ONE session, and a session-level trend across all three survives")
    print("     every control in this design.")
    print("  7. A null does NOT license 'EMG is not a risk for this project'. Imagined")
    print("     fists versus feet is a low-EMG task. Imagined SPEECH, the corpus's own")
    print("     worked example, is a task where sub-vocalisation makes EMG")
    print("     class-correlated by construction.")
    print("  8. The injection topography is STIPULATED, not measured. Section 8B now")
    print("     measures how much that costs: over four shapes the threshold spans a")
    print("     multiple of itself in the registered units, so the registered figure is")
    print("     not a bound over topographies.")
    print("  9. The probe shares the headline's filter-before-fold structure. The")
    print("     band-pass is fitted on all data. Label-blind and fixed a priori, so not")
    print("     leakage, but not inside the fold either.")
    print(" 10. THE CORPUS'S CLOSURE CONDITION IS A TWO-ARM CONJUNCTION AND ONLY ONE")
    print("     ARM EXISTS. The corpus asks for (i) a temporal-channel-DELETED ablation")
    print("     row inside 8-30 Hz that should not hurt appreciably, AND (ii) a")
    print("     high-band EMG-proxy decoder that should land at or below the majority")
    print("     floor. This script is arm (ii). Arm (i) was deliberately not run (see")
    print("     'OUT OF SCOPE ON PURPOSE' in the docstring). A conjunction with one arm")
    print("     run is not satisfied, and no sentence here may be written as though the")
    print("     corpus's condition has been met.")
    print(" 11. THE TEMPORAL PROFILE IS STIPULATED TOO, and until 2026-07-26 that was")
    print("     not disclosed anywhere. The registered ladder injects a constant")
    print("     amplitude into every trial of a class. Section 8B adds the bursty arm")
    print("     the script's own text calls realistic, and the bound is now the worse of")
    print("     the two.")
    print(" 12. THE PERMUTATION NULL IS NOT THE MAJORITY FLOOR, and three of the four")
    print("     channel sets land in the LOWER tail of their own nulls. That is an")
    print("     unexplained systematic property of this instrument. It is named here and")
    print("     NOT explained here: see section 4's null table. Assigning it a mechanism")
    print("     in the same breath as measuring it is the exact failure this")
    print("     pre-registration exists to prevent, so it gets its own registration or")
    print("     it gets nothing.")

    sub("The word this run is allowed to use")
    if LADDER_FAILED:
        print("  NEITHER 'bounds' NOR 'eliminates'. The ladder failed its own primary")
        print("  falsifier, so this run produced an instrument that cannot detect what")
        print("  it was built to detect, and no claim about EMG follows from it.")
    elif not b_above and not ARM_A_POSITIVE:
        thr = max(v for v in thr_by_topo.values() if v != "NEVER")
        print(f"  BOUNDS. Not ELIMINATES.")
        print(f"  WITHDRAWN 2026-07-26, kept visible: this block used to print the")
        print(f"  unqualified sentence 'this recording contains no class-correlated")
        print(f"  broadband temporal source as large as a = {thr:.3f} times T8's own "
              f"high-band SD',")
        print(f"  and 'the corpus line \"nothing in the repo bounds an EMG "
              f"contribution\" is now FALSE'.")
        print(f"  Both overstated. Three separate scope failures, all of them in the")
        print(f"  direction that overstates coverage. The scoped versions:")
        print(f"\n  (1) SPECTRAL SCOPE. This probe covers "
              f"{BANDS['PRIMARY']['l_freq']:.0f} to {BANDS['PRIMARY']['h_freq']:.0f} Hz "
              f"minus a 60 Hz notch, out of a")
        print(f"      recorded 0 to {NYQ:.0f} Hz. The headline decoder lives at 8 to "
              f"30 Hz, which is where")
        print(f"      this probe is blind BY CONSTRUCTION. The corpus's own position is "
              f"that")
        print(f"      temporalis EMG is broadband and not excluded by any plausible "
              f"band-pass, so")
        print(f"      EMG INSIDE THE DECODER'S OWN PASSBAND REMAINS ENTIRELY UNBOUNDED. "
              f"The PSD table")
        print(f"      makes that worse rather than better: the recorded high band is "
              f"steeply")
        print(f"      attenuated, so any EMG present shows up preferentially at the low")
        print(f"      frequencies this probe excludes.")
        print(f"  (2) TOPOGRAPHY SCOPE. a = {thr:.3f} is the max over the TWO "
              f"registered shapes, both")
        print(f"      diffuse. Over {len(ALL_TOPOS)} shapes including the canonical "
              f"focal one-electrode source,")
        print(f"      the worst continuous threshold is a = {CONT_SHAPE_BOUND:.3f}. "
              f"That is the honest figure in")
        print(f"      the registered units, and a focal source at a = 0.500 sits INSIDE "
              f"this")
        print(f"      recording's tolerance while sitting OUTSIDE the registered bound.")
        print(f"  (3) TEMPORAL-STRUCTURE SCOPE. The registered ladder injects a "
              f"constant amplitude")
        print(f"      into EVERY trial of a class, which is the shifted distribution "
              f"this script")
        print(f"      itself calls UNrealistic. Section 8B's bursty arm shows the "
              f"registered")
        print(f"      detection criterion cannot adjudicate a "
              f"{INTERMITTENT_FRACTION:.0%}-duty-cycle source at all, so the")
        print(f"      bursty exposure is OPEN. It is not bounded at a larger number; it "
              f"is unbounded.")
        print(f"\n  WHAT IS ACTUALLY BOUNDED, and it is worth having: inside "
              f"{BANDS['PRIMARY']['l_freq']:.0f} to "
              f"{BANDS['PRIMARY']['h_freq']:.0f} Hz")
        print(f"  minus the notch, at the temporal ring, this recording contains no")
        print(f"  class-correlated broadband source large enough for this probe to "
              f"detect at the")
        print(f"  worst measured threshold, and no aggregate log-power difference as "
              f"large as")
        print(f"  Cohen's d = {MDE_AGGREGATE:.3f}.")
        print(f"  THE SCOPED CORPUS CLAIM: 'nothing in the repo bounds an EMG "
              f"contribution' is now")
        print(f"  false FOR THE {BANDS['PRIMARY']['l_freq']:.0f} TO "
              f"{BANDS['PRIMARY']['h_freq']:.0f} Hz BAND ONLY. It remains TRUE inside "
              f"the decoder's own 8 to 30 Hz")
        print(f"  passband, and the temporal-channel-deleted arm the corpus asks for")
        print(f"  alongside this one has not been run.")
    else:
        print("  MEASURED, not bounded. A class-correlated high-band signal is present")
        print("  at the temporal ring, so the honest statement is no longer about the")
        print("  size of an absent effect but about the size of a present one.")

    print(f"\nMEASURED RUNTIME: {time.time() - T_START:.1f} s wall on this machine.")
