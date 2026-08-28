"""Sections 0-2 of emg_proxy.py: the registered constants, the data held identical
to decode_csp.py, the bands with their realised filter lengths, and the epochs.
Split out 2026-08-26. Importing this module also puts common.py's directory on the
path, so the other emg_* modules import it FIRST. build() returns everything the
later sections read."""

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings
from types import SimpleNamespace

import numpy as np
import mne
from mne.datasets import eegbci

import _bootstrap  # noqa: F401  -- puts src/ on the path; must come first

from common import FRONTOPOLAR, SENSORIMOTOR, TEMPORAL

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
CROP = (1.0, 2.0)
PRECUE_CROP = (-1.0, 0.0)
SEED = 42
N_PERMUTATIONS = 1000
ALPHA = 0.05

# TEMPORAL (the probe: temporalis territory, where the fourth retained CSP component
# peaks at T8/T10/TP8), FRONTOPOLAR and SENSORIMOTOR are defined once in common.py
# and imported at the top of this file.
#
# The size match between TEMPORAL and FRONTOPOLAR is load-bearing and is asserted in
# test_pipeline.py: ablate_channels.py's own second caveat is that its frontopolar row
# confounds region with an 8x cut in channel count, and matching the counts removes
# that confound from this one comparison.

# The stipulated injection topography: right-lateralized, mimicking the shape of
# the observed component. STIPULATED, NOT MEASURED. It is a plausible right
# temporalis projection, not this subject's. A spatially flat topography is run
# alongside it precisely because the sensitivity figure should not rest on an
# assumed shape nobody measured.
TOPO_STIPULATED = dict(T8=1.00, T10=0.80, TP8=0.80, FT8=0.50,
                       T7=0.20, T9=0.15, TP7=0.15, FT7=0.10)
TOPO_FLAT = {ch: 1.0 for ch in TEMPORAL}

# POST-REGISTRATION, 2026-07-26. Two FOCAL shapes, added because the registered
# pair spans only the diffuse half of the space. a is defined by the source's
# contribution to T8 alone, while detectability depends on the TOTAL power the
# source puts on the ring, so at fixed a a flat source injects far more total
# power than a focal one and is correspondingly easier to see. The registered
# "max over topographies" is therefore a max over two shapes, not a bound over
# shapes. A generator sitting directly under one electrode is the canonical
# superficial-artifact geometry, not an exotic case, so the family has to
# include it.
TOPO_T8_ONLY = {ch: (1.0 if ch == "T8" else 0.0) for ch in TEMPORAL}
TOPO_T8_T10 = {ch: (1.0 if ch == "T8" else 0.6 if ch == "T10" else 0.0)
               for ch in TEMPORAL}

# a = the injected source's contribution to T8 as a fraction of T8's own measured
# high-band SD. a = 0.000 is the real data and is the reference row.
LADDER = [0.000, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.400, 0.600, 0.800]
N_INJECT_SEEDS = 10
INJECT_SEED_BASE = 9000  # distinct from SEED, which stays 42 for the CV throughout
DETECT_K = 31  # median accuracy across seeds must reach 31/45 = 68.9% to count

# POST-REGISTRATION additions, all dated 2026-07-26, all disclosed at the point
# of use. The pre-registration is NOT edited to accommodate any of them.
N_SEED_SWEEP = 100        # CV seeds for the primary cell, matching evaluate_honestly.py
LADDER_SEED_CHECK = [42, 0, 1, 2, 3]   # CV seeds the ladder threshold is re-derived at
INTERMITTENT_FRACTION = 0.25           # inside the 20% to 30% band asked for
INTERMITTENT_SEED_BASE = 77000         # distinct from INJECT_SEED_BASE and SEED

# Pinned in the pre-registration, checked here rather than assumed. This is the
# one quantity in the pipeline that silently rescales with sfreq.
PINNED_BANDPASS_TAPS = 265
PINNED_NOTCH_TAPS = 89
PINNED_CASCADE_HALF_S = 1.100


def hr(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title):
    print(f"\n--- {title} ---")


def build():
    # =============================================================================
    # 0. Data, held identical to decode_csp.py and ablate_channels.py
    # =============================================================================
    hr("0. DATA, HELD IDENTICAL TO decode_csp.py AND ablate_channels.py")

    edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
    raw_base = mne.concatenate_raws(
        [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]
    )
    eegbci.standardize(raw_base)
    raw_base.set_montage("standard_1005")
    raw_base.set_eeg_reference("average", projection=False)

    SFREQ = float(raw_base.info["sfreq"])
    NYQ = SFREQ / 2.0
    ALL_CH = list(raw_base.ch_names)

    print(f"\nSubject {SUBJECT}, runs {RUNS} (imagined both fists vs. imagined both feet)")
    print(f"{len(ALL_CH)} channels, sfreq {SFREQ:.1f} Hz, Nyquist {NYQ:.1f} Hz")
    print("Average reference computed over all 64 channels BEFORE any pick, so the")
    print("channel subsets below are NOT electrically independent of each other.")

    for name, picks in [("TEMPORAL", TEMPORAL), ("FRONTOPOLAR", FRONTOPOLAR),
                        ("SENSORIMOTOR", SENSORIMOTOR)]:
        missing = [c for c in picks if c not in ALL_CH]
        assert not missing, f"{name}: channels absent from this montage: {missing}"
    print(f"\nChannel sets verified present in the standardised montage:")
    print(f"  TEMPORAL     ({len(TEMPORAL):2d} ch) {' '.join(TEMPORAL)}")
    print(f"  FRONTOPOLAR  ({len(FRONTOPOLAR):2d} ch) {' '.join(FRONTOPOLAR)}")
    print(f"  SENSORIMOTOR ({len(SENSORIMOTOR):2d} ch) {' '.join(SENSORIMOTOR)}")
    print(f"  ALL64        ({len(ALL_CH):2d} ch)")

    CHANNEL_SETS = [
        ("TEMPORAL", TEMPORAL),
        ("FRONTOPOLAR", FRONTOPOLAR),
        ("SENSORIMOTOR", SENSORIMOTOR),
        ("ALL64", ALL_CH),
    ]

    # =============================================================================
    # 1. Bands, and the filter lengths that set the smear budget
    # =============================================================================
    hr("1. BANDS, AND THE REALISED FILTER LENGTHS THAT SET THE SMEAR BUDGET")

    BANDS = {
        # The decoder's own band. Default transition bandwidths, because reproducing
        # the published pipeline means reproducing its defaults.
        "DECODER": dict(l_freq=8.0, h_freq=30.0, l_trans=None, h_trans=None,
                        notch=False,
                        why="the published pipeline, for the positive control"),
        # PRIMARY. l_trans pinned at 2.0 so the probe's lower stopband edge lands at
        # 38 Hz, ABOVE the decoder's 37.5 Hz stopband edge. MNE's default l_trans
        # here would be 10 Hz, which would put the probe's stopband edge at 30 Hz and
        # overlap the decoder's passband, destroying the claim that the probe looks
        # where the decoder cannot.
        "PRIMARY": dict(l_freq=40.0, h_freq=75.0, l_trans=2.0, h_trans=2.0,
                        notch=True,
                        why="THE PROBE. 40-75 Hz with 60 Hz notched out"),
        "R1": dict(l_freq=40.0, h_freq=55.0, l_trans=2.0, h_trans=2.0, notch=False,
                   why="entirely BELOW line. Arbiter for a positive primary"),
        "R2": dict(l_freq=65.0, h_freq=75.0, l_trans=2.0, h_trans=2.0, notch=False,
                   why="entirely ABOVE line. Second arbiter"),
        "R3": dict(l_freq=32.0, h_freq=75.0, l_trans=2.0, h_trans=2.0, notch=True,
                   why="the greedy band, dipping into the decoder's 30-37.5 Hz "
                       "transition region"),
    }


    def filter_taps(l_freq, h_freq, l_trans, h_trans):
        h = mne.filter.create_filter(
            None, SFREQ, l_freq, h_freq,
            l_trans_bandwidth="auto" if l_trans is None else l_trans,
            h_trans_bandwidth="auto" if h_trans is None else h_trans,
            method="fir", fir_design="firwin", verbose="error",
        )
        return len(h)


    # MNE's notch is a band-stop built from the notch width PLUS half the transition
    # bandwidth on each side, and the per-side transition it then uses is
    # trans_bandwidth/2, not trans_bandwidth. Measure it rather than predict it.
    NOTCH_TAPS = len(mne.filter.create_filter(
        None, SFREQ, 60.0 + 1.0 + 3.0, 60.0 - 1.0 - 3.0,
        l_trans_bandwidth=3.0, h_trans_bandwidth=3.0,
        method="fir", fir_design="firwin", verbose="error",
    ))

    DECODER_TAPS = filter_taps(8.0, 30.0, None, None)
    PRIMARY_TAPS = filter_taps(40.0, 75.0, 2.0, 2.0)

    half_s = lambda taps: (taps - 1) / 2.0 / SFREQ
    CASCADE_TAPS = PRIMARY_TAPS + NOTCH_TAPS - 1
    CASCADE_HALF_S = half_s(CASCADE_TAPS)

    sub("Bands, with the role of each fixed in advance")
    for name, b in BANDS.items():
        lt = "auto" if b["l_trans"] is None else f"{b['l_trans']:.1f}"
        ht = "auto" if b["h_trans"] is None else f"{b['h_trans']:.1f}"
        taps = filter_taps(b["l_freq"], b["h_freq"], b["l_trans"], b["h_trans"])
        print(f"  {name:<8} {b['l_freq']:>5.1f} to {b['h_freq']:>5.1f} Hz  "
              f"trans {lt}/{ht}  notch={str(b['notch']):<5} "
              f"{taps} taps, half {half_s(taps):.3f} s")
        print(f"           {b['why']}")

    sub("Realised filter lengths versus the values pinned in the pre-registration")
    print(f"  band-pass 8-30 Hz  (decoder): {DECODER_TAPS} taps, "
          f"half-length {half_s(DECODER_TAPS):.3f} s")
    print(f"  band-pass 40-75 Hz (probe)  : {PRIMARY_TAPS} taps, "
          f"half-length {half_s(PRIMARY_TAPS):.3f} s   PINNED {PINNED_BANDPASS_TAPS}")
    print(f"  notch 60 Hz                 : {NOTCH_TAPS} taps, "
          f"half-length {half_s(NOTCH_TAPS):.3f} s   PINNED {PINNED_NOTCH_TAPS}")
    print(f"  cascade (probe + notch)     : {CASCADE_TAPS} taps, "
          f"half-length {CASCADE_HALF_S:.3f} s   PINNED about {PINNED_CASCADE_HALF_S:.3f} s")

    assert PRIMARY_TAPS == PINNED_BANDPASS_TAPS, (
        f"band-pass realised at {PRIMARY_TAPS} taps, pinned at {PINNED_BANDPASS_TAPS}. "
        "The smear budget is wrong and every window-based statement must be recomputed."
    )
    assert DECODER_TAPS == PINNED_BANDPASS_TAPS, (
        f"decoder band-pass realised at {DECODER_TAPS} taps, pinned at "
        f"{PINNED_BANDPASS_TAPS}. Probe and decoder no longer share a smear budget."
    )

    NOTCH_PINNED_OK = NOTCH_TAPS == PINNED_NOTCH_TAPS
    if not NOTCH_PINNED_OK:
        print(f"\n  *** DEVIATION FROM THE PRE-REGISTRATION, FALSIFIER 7, NOTCH TERM ***")
        print(f"  The pre-registration predicted an {PINNED_NOTCH_TAPS}-tap notch "
              f"(half-length {PINNED_NOTCH_TAPS // 2 / SFREQ:.3f} s) from")
        print(f"  trans_bandwidth=6.0. MNE splits that transition, using 3.0 Hz per")
        print(f"  side, so the realised notch is {NOTCH_TAPS} taps, half-length "
              f"{half_s(NOTCH_TAPS):.3f} s.")
        print(f"  The FREQUENCY design is exactly as pinned: band-stop 56 to 64 Hz,")
        print(f"  -6 dB at 54.5 and 65.5 Hz. Only the TIME-domain prediction was wrong.")
        print(f"  RECOMPUTED SMEAR BUDGET, which is what falsifier 7 requires:")
        print(f"    cascade half-length {CASCADE_HALF_S:.3f} s, not "
              f"{PINNED_CASCADE_HALF_S:.3f} s.")
        print(f"    the {CROP[0]:.1f} to {CROP[1]:.1f} s feature window can therefore draw "
              f"energy from as early")
        print(f"    as {CROP[0] - CASCADE_HALF_S:+.3f} s, not "
              f"{CROP[0] - PINNED_CASCADE_HALF_S:+.3f} s. Both are pre-cue.")
        print(f"    the pre-cue diagnostic window {PRECUE_CROP[0]:.1f} to "
              f"{PRECUE_CROP[1]:.1f} s is now ENTIRELY")
        print(f"    filter-contaminated in the primary band (clean portion requires")
        print(f"    t <= -{CASCADE_HALF_S:.3f} s, and the window starts at "
              f"{PRECUE_CROP[0]:.1f} s), so its")
        print(f"    contingency re-run is mandatory for any reading, not optional.")
        print(f"  DIRECTION IS UNCHANGED, which is why the primary result survives this:")
        print(f"    smear can only push a score UP, never down, so a null in the")
        print(f"    primary band remains conservative. R1 and R2 are single-filter")
        print(f"    (half-length {half_s(PRIMARY_TAPS):.3f} s) and their windows are "
              f"unaffected.")

    sub("Smear budget, computed from the realised lengths")
    print(f"  probe band-pass alone reaches {half_s(PRIMARY_TAPS):.3f} s either side of "
          f"any sample.")
    print(f"  probe + notch cascade reaches {CASCADE_HALF_S:.3f} s either side.")
    print(f"  the {CROP[0]:.1f} to {CROP[1]:.1f} s feature window can draw from "
          f"{CROP[0] - CASCADE_HALF_S:+.3f} s onward.")
    print(f"  R1 and R2 are single-filter, so their windows are clean from "
          f"{CROP[0] - half_s(PRIMARY_TAPS):+.3f} s onward,")
    print(f"  which is strictly post-cue. That is why they can arbitrate a positive")
    print(f"  primary and the primary cannot arbitrate itself.")

    # =============================================================================
    # 2. Build the epochs, one set per band
    # =============================================================================
    hr("2. EPOCHS, AND THE TRIAL COUNTS EVERY NUMBER BELOW RESTS ON")


    def build_epochs(band):
        b = BANDS[band]
        raw = raw_base.copy()
        kwargs = dict(fir_design="firwin", skip_by_annotation="edge")
        if b["l_trans"] is not None:
            kwargs["l_trans_bandwidth"] = b["l_trans"]
        if b["h_trans"] is not None:
            kwargs["h_trans_bandwidth"] = b["h_trans"]
        raw.filter(b["l_freq"], b["h_freq"], **kwargs)
        if b["notch"]:
            raw.notch_filter(freqs=np.array([60.0]), notch_widths=2.0,
                             trans_bandwidth=6.0, method="fir", fir_design="firwin")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        return mne.Epochs(
            raw, events, dict(hands=2, feet=3),
            tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
        )


    EPOCHS = {name: build_epochs(name) for name in BANDS}

    labels = EPOCHS["PRIMARY"].events[:, -1]
    N = len(labels)
    N_HANDS = int((labels == 2).sum())
    N_FEET = int((labels == 3).sum())
    MAJ_CORRECT = max(N_HANDS, N_FEET)
    FLOOR = MAJ_CORRECT / N

    assert N == 45, f"expected 45 trials, got {N}"
    assert N_HANDS == 21, f"expected 21 hands, got {N_HANDS}"
    assert N_FEET == 24, f"expected 24 feet, got {N_FEET}"
    for name in BANDS:
        assert np.array_equal(EPOCHS[name].events[:, -1], labels), (
            f"band {name} produced a different label vector; the bands are not "
            "epoching the same trials."
        )

    print(f"\n{N} trials ({N_HANDS} hands, {N_FEET} feet)")
    print(f"MAJORITY-CLASS FLOOR = {FLOOR:.1%} ({MAJ_CORRECT}/{N}). Chance is THIS, not 50%.")
    print("Always answering 'feet' scores 53.3% without looking at the EEG at all, so")
    print("'above 50%' is not a claim about anything.")

    sub(f"Attainable-accuracy lattice (n = {N})")
    print(f"Every CV here is a PARTITION: each trial is tested exactly once, so the")
    print(f"accuracy is k/{N} for integer k, i.e. steps of {1 / N:.3%}.")
    print("Values that matter for the pre-registered decision boundaries:")
    print("  " + "  ".join(f"{k}/{N}={k / N:.1%}" for k in [20, 24, 25, 29, 30]))
    print("  " + "  ".join(f"{k}/{N}={k / N:.1%}" for k in [31, 32, 33, 39, 40, 41]))
    print("Any value off this lattice was not measured.")


    def assert_lattice(k, tag):
        assert isinstance(k, (int, np.integer)), f"{tag}: k is not an integer"
        assert 0 <= k <= N, f"{tag}: k={k} outside 0..{N}"


    def acc_str(k):
        """The house format: percentage, then the count it came from."""
        assert_lattice(k, "acc_str")
        return f"{k / N:>6.1%} ({k}/{N})"


    BAND_CROPPED = {}
    for name in BANDS:
        BAND_CROPPED[name] = EPOCHS[name].copy().crop(tmin=CROP[0], tmax=CROP[1])
    CH_NAMES = BAND_CROPPED["PRIMARY"].ch_names
    N_TIMES = BAND_CROPPED["PRIMARY"].get_data(copy=False).shape[-1]
    print(f"\nFeature window {CROP[0]:.1f} to {CROP[1]:.1f} s = {N_TIMES} samples per trial.")


    def get_data(band, picks):
        idx = [CH_NAMES.index(c) for c in picks]
        return BAND_CROPPED[band].get_data(copy=False)[:, idx, :]

    return SimpleNamespace(
        raw_base=raw_base, SFREQ=SFREQ, NYQ=NYQ, ALL_CH=ALL_CH,
        CHANNEL_SETS=CHANNEL_SETS, BANDS=BANDS, CASCADE_HALF_S=CASCADE_HALF_S,
        EPOCHS=EPOCHS, labels=labels, N=N, N_HANDS=N_HANDS, N_FEET=N_FEET,
        MAJ_CORRECT=MAJ_CORRECT, FLOOR=FLOOR, BAND_CROPPED=BAND_CROPPED,
        CH_NAMES=CH_NAMES, N_TIMES=N_TIMES, assert_lattice=assert_lattice,
        acc_str=acc_str, get_data=get_data,
    )
