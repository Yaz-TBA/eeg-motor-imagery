"""The EMG probe: refit the pipeline on MUSCLE-BAND frequencies at MUSCLE-TERRITORY
electrodes and see whether hands vs. feet is still decodable there.

Checks if the jaw muscle could be affecting the result, with it only bounding the answer
(not closing it) since this probe is blindest to the decoder's own 8-30 Hz band.

WHY THIS SCRIPT EXISTS. The repo's only artifact control is ablate_channels.py,
whose frontopolar-only row (23/45, 51.1%, against a 24/45 = 53.3% majority floor)
addresses OCULAR contamination. It says nothing about MUSCLE, and it structurally
cannot: the published pipeline band-passes to 8-30 Hz, so everything above the
passband is discarded before any covariance is computed, and the surface-EMG
signature lives mostly above 30 Hz. The filter, not the feature, decides what is
findable. An EMG probe inside 8-30 Hz cannot see the thing it is probing for.

Meanwhile the fourth retained CSP component peaks at T8, with T10 and TP8 in its
top five, which is temporalis territory. The corpus names that as an open
exposure at canon level and then does not close it. This script closes it, in the
only direction the data permits: it BOUNDS an EMG contribution. It cannot
eliminate one.

PRE-REGISTERED. Every band, channel set, test, threshold and outcome-meaning in
this file was fixed in writing before any of it was executed, in
prereg/prereg-emg-proxy.md. The point of that document is that
no number produced here can be narrated after the fact. This project's round-one
failure mode was inventing the mechanism story in the same breath as the number.
Measuring and explaining are separate steps, and the explanation was written
first, on purpose, so it could not be fitted to the result.

WHAT IT RUNS.
  positive control  8-30 Hz, all 64 ch. Must reproduce 41/45 = 91.1%, or the
                    harness is not the published pipeline and nothing below is
                    comparable to the existing ablation table.
  arm (a)           univariate. Does log high-band power differ by class, per
                    channel and in aggregate? Welch t plus Mann-Whitney U,
                    Holm-Bonferroni across the 8 temporal channels.
  arm (b)           THE SHARP TEST, and the one that governs. The unmodified
                    CSP+LDA pipeline, same splitter, same seed 42, refit on
                    40-75 Hz (60 Hz notched) at the temporal ring. Plus three
                    comparison channel sets, so the answer has a spatial profile
                    rather than being a single number that cannot distinguish a
                    local source from a global one.
  robustness        R1 40-55 (below line), R2 65-75 (above line), R3 32-75
                    (greedy). Fixed role: they cannot promote a null primary to a
                    positive. They can only qualify a positive or expose line
                    contamination.
  ladder            THE PART THAT TURNS A NULL INTO A BOUND. Inject a synthetic
                    class-correlated broadband source with a fixed topography at
                    known amplitude and find the smallest one this probe can see.
                    Without it, a probe at floor supports only "we looked and
                    found nothing", which is another disclosure, not a
                    measurement.

WHAT IT DOES NOT SHOW, IN EVERY POSSIBLE OUTCOME.
  1. 160 Hz sampling truncates the EMG spectrum. Surface temporalis EMG has
     substantial power well above the 80 Hz Nyquist and none of it was recorded.
     Even a perfect null bounds only the recorded part of the spectrum.
  2. The average reference is computed over all 64 channels BEFORE any subset is
     picked, exactly as in decode_csp.py and ablate_channels.py. Every channel
     carries -1/64 of every other, so the temporal ring is not electrically
     sealed off from the rest of the head.
  3. EEGMMIDB ships no EOG and no EMG channel. There is no ground truth for
     "this is muscle". This probe measures high-band power at muscle-adjacent
     scalp sites. It does not measure muscle.
  4. A positive here cannot distinguish temporalis EMG from a saccadic spike
     potential, and the cue is position-confounded with the label (bar at the
     top of the screen for fists, bottom for feet), which makes the ocular
     candidate genuinely plausible. Different confounds, different remedies.
  5. n = 45, one subject, one session. Arm (a) can only detect large effects.

OUT OF SCOPE ON PURPOSE. The temporal-channel-DELETED row (all 64 minus the
temporal ring, inside the decoder's own 8-30 Hz band) is the corpus's other
requested arm and it is cheap. It is NOT run here, because it answers a different
question (does the headline NEED those channels) and pre-registering it loosely
alongside this probe would let a null on one be read as covering the other. It
gets its own pre-registration.
"""

import time

from emg_setup import build
from emg_psd import run_psd
from emg_sharp import run_positive_control, run_sharp
from emg_univariate import run_precue, run_univariate
from emg_ladder import run_ladder
from emg_ladder_ext import run_ladder_extension
from emg_verdict import run_verdict


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
    run_verdict(D, K_PRIMARY, PRIMARY_CELL, primary_results, null_stats,
                p_perm_str, agg_t_p, agg_u_p, agg_d, ARM_A_POSITIVE,
                MDE_AGGREGATE, MDE_PERCHANNEL, L.LADDER_FAILED, L.thr_by_topo,
                CONT_SHAPE_BOUND, L.ALL_TOPOS, T_START)


if __name__ == "__main__":
    main()
