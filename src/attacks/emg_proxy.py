"""The EMG probe: refit the pipeline on muscle-band frequencies at muscle-territory
electrodes and see whether hands vs. feet is still decodable there.

Checks if the jaw muscle could be affecting the result, with it only bounding the answer
(not closing it) since this probe is blindest to the decoder's own 8-30 Hz band.

Why this exists:

The repo's only artifact control was ablate_channels.py, whose frontopolar row
(23/45 = 51.1% against the 24/45 = 53.3% majority floor) addresses ocular
contamination. It structurally cannot address muscle: the published pipeline
band-passes to 8-30 Hz before any covariance is computed, and surface EMG lives
mostly above 30 Hz, so an EMG probe inside 8-30 Hz cannot see what it is probing
for. Meanwhile the fourth retained CSP component peaks at T8, with T10 and TP8 in
its top five, which is temporalis territory. This script closes that exposure in
the only direction the data permits: it bounds an EMG contribution, never
eliminates one.

Pre-registered:

Every band, channel set, test, threshold and outcome-meaning was fixed in
prereg/prereg-emg-proxy.md before any of it ran, so no number here can be
narrated after the fact. Measuring and explaining are separate steps, and the
explanation was written first on purpose.

What it runs:

  positive control  8-30 Hz, all 64 ch; must reproduce 41/45 = 91.1% or nothing
                    below is comparable to the ablation table
  arm (a)           univariate: does log high-band power differ by class?
                    Welch t plus Mann-Whitney U, Holm across the 8 temporal ch
  arm (b)           the sharp test, and the one that governs: the unmodified
                    CSP+LDA pipeline, same splitter, same seed 42, refit on
                    40-75 Hz (60 Hz notched) at the temporal ring, plus three
                    comparison channel sets so the answer has a spatial profile
  robustness        R1 40-55 (below line), R2 65-75 (above), R3 32-75 (greedy);
                    fixed role: they can qualify a positive or expose line
                    contamination, never promote a null primary
  ladder            the part that turns a null into a bound: inject a synthetic
                    class-correlated broadband source at known amplitude and
                    find the smallest one this probe can see. Without it a probe
                    at floor supports only "we looked and found nothing"

What it cannot show, in any outcome:

  1. 160 Hz sampling truncates the EMG spectrum; temporalis EMG has substantial
     power above the 80 Hz Nyquist and none of it was recorded.
  2. The average reference spans all 64 channels before any pick, so every
     channel carries -1/64 of every other; the ring is not sealed off.
  3. EEGMMIDB ships no EOG and no EMG channel. This measures high-band power at
     muscle-adjacent scalp sites, not muscle.
  4. A positive cannot separate temporalis EMG from a saccadic spike potential,
     and the cue is position-confounded with the label (bar top for fists,
     bottom for feet), so the ocular candidate is genuinely plausible.
  5. n = 45, one subject, one session; arm (a) detects only large effects.

Out of scope on purpose:

The temporal-channel-deleted row inside 8-30 Hz is the corpus's other requested
arm, and it answers a different question (does the headline need those
channels). Pre-registering it loosely alongside this probe would let a null on
one be read as covering the other, so it gets its own pre-registration.
"""

import time

from emg_setup import build
from emg_psd import run_psd
from emg_sharp import run_positive_control, run_sharp
from emg_univariate import run_precue, run_univariate
from emg_ladder import run_ladder
from emg_ladder_ext import run_ladder_extension
from emg_verdict import Primary, run_verdict


def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-minute experiment as a side effect. The sections
    live in the emg_* modules beside this file, split 2026-08-26; their bodies are
    verbatim from the single-file version, so the stdout is unchanged line for
    line."""

    T_START = time.time()

    D = build()
    R2_INFORMATIVE = run_psd(D)
    run_positive_control(D)
    (primary_results, PRIMARY_CELL, K_PRIMARY,
     null_stats, p_perm_str) = run_sharp(D, R2_INFORMATIVE)
    (agg_t_p, agg_u_p, agg_d, ARM_A_POSITIVE, MDE_AGGREGATE, MDE_PERCHANNEL,
     log_power, two_tests, cohens_d) = run_univariate(D)
    run_precue(D, log_power, two_tests)
    L = run_ladder(D, cohens_d, MDE_AGGREGATE)
    CONT_SHAPE_BOUND = run_ladder_extension(D, L)
    P = Primary(
        K_PRIMARY=K_PRIMARY, PRIMARY_CELL=PRIMARY_CELL,
        primary_results=primary_results, null_stats=null_stats,
        p_perm_str=p_perm_str, agg_t_p=agg_t_p, agg_u_p=agg_u_p, agg_d=agg_d,
        ARM_A_POSITIVE=ARM_A_POSITIVE, MDE_AGGREGATE=MDE_AGGREGATE,
        MDE_PERCHANNEL=MDE_PERCHANNEL, CONT_SHAPE_BOUND=CONT_SHAPE_BOUND,
        T_START=T_START)

    run_verdict(D, P, L.LADDER_FAILED, L.thr_by_topo, L.ALL_TOPOS)


if __name__ == "__main__":
    main()
