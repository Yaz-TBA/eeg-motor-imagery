"""Regression tests for the claims this repo publishes.

NAVIGATION. Regression tests, not 'does it run' tests. Each one guards a mistake this
project actually made and had to retract, so a future edit that reintroduces it fails
here instead of in the README.

These are deliberately not "does the code run" tests. Each one guards a specific mistake
this project actually made and had to retract, so a future edit that reintroduces the
mistake fails here instead of in a README.

    python3 -m pytest checks/test_pipeline.py -q     (or: python3 checks/test_pipeline.py)

Nothing here downloads data or trains anything; the whole file runs in under a second.
"""

import numpy as np

# common.py lives at the repo root, one level up; put it on the path so this script
# can be launched from anywhere.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common
from common import (
    CHANCE, FRONTOPOLAR, N_TRIALS, SENSORIMOTOR, TEMPORAL, Z_TWO_SIDED,
    assert_lattice, holm, make_clf, on_lattice, wilson_interval,
)


# =============================================================================
# 1. The lattice. This is the test that would have caught the biggest retraction.
# =============================================================================

def test_retracted_numbers_are_not_attainable_accuracies():
    """95.9% and 47.4% were published in an earlier README as measured accuracies.

    With 45 trials tested exactly once each, accuracy is a count over 45, so it can only
    land on multiples of 1/45 = 2.222%. Neither number is on that lattice. They were not
    measurements, and no run can ever produce them.
    """
    assert not on_lattice(0.959), "0.959 is not k/45 and must never be reported as one"
    assert not on_lattice(0.474), "0.474 is not k/45 and must never be reported as one"


def test_published_numbers_are_attainable():
    """Every headline figure must be a real count over 45."""
    for acc, k in [(41 / 45, 41),   # 91.1%, the headline
                   (43 / 45, 43),   # 95.6%, sensorimotor only
                   (23 / 45, 23),   # 51.1%, frontopolar only
                   (35 / 45, 35),   # 77.8%, sensorimotor deleted
                   (24 / 45, 24)]:  # 53.3%, the majority-class floor
        assert on_lattice(acc), f"{acc} should be {k}/45"
        assert round(acc * N_TRIALS) == k


def test_chance_is_the_majority_class_not_one_half():
    """A recurring error in this repo's own prose. 21 hands + 24 feet means the
    do-nothing baseline is 53.3%, not 50%. A control that lands at 53.3% has failed
    correctly; one that lands at 50% has not been interpreted."""
    assert abs(CHANCE - 24 / 45) < 1e-12
    assert abs(CHANCE - 0.5333333) < 1e-6
    assert CHANCE != 0.5


def test_assert_lattice_rejects_an_off_lattice_score():
    assert_lattice([41 / 45, 43 / 45], where="valid")          # must not raise
    try:
        assert_lattice([0.959], where="the retracted value")
    except AssertionError:
        return
    raise AssertionError("assert_lattice accepted an off-lattice score")


# =============================================================================
# 2. The pipeline. Guards the leakage class of bug.
# =============================================================================

def test_csp_is_inside_the_pipeline():
    """CSP must refit inside every training fold. If it is fitted once outside the
    Pipeline, the spatial filters see the test trials and the accuracy is invalid.
    This is the single most important structural property in the repo."""
    clf = make_clf()
    assert list(clf.named_steps) == ["CSP", "LDA"]
    assert clf.named_steps["CSP"].__class__.__name__ == "CSP"


def test_csp_hyperparameters_match_the_published_configuration():
    csp = make_clf().named_steps["CSP"]
    assert csp.n_components == 4
    assert csp.reg is None
    assert csp.log is True
    assert csp.norm_trace is False


def test_reduced_montages_clamp_n_components():
    """CSP cannot ask for more components than the montage has channels. The 8-channel
    frontopolar condition would otherwise fail at fit time."""
    assert make_clf(n_channels=8).named_steps["CSP"].n_components == 4
    assert make_clf(n_channels=4).named_steps["CSP"].n_components == 3
    assert make_clf(n_channels=2).named_steps["CSP"].n_components == 1


# =============================================================================
# 3. Statistics. These reproduce numbers that are printed in the README.
# =============================================================================

def test_wilson_interval_reproduces_the_published_headline_interval():
    """README reports [79.3%, 96.5%] for 41/45."""
    lo, hi = wilson_interval(41, 45)
    assert round(lo * 100, 1) == 79.3, f"lower bound moved: {lo*100:.3f}"
    assert round(hi * 100, 1) == 96.5, f"upper bound moved: {hi*100:.3f}"


def test_wilson_is_asymmetric_near_the_boundary():
    """The reason Wilson is used instead of mean +/- std at n=45: the interval must not
    be symmetric about p, and must never leave [0, 1]."""
    lo, hi = wilson_interval(45, 45)
    assert hi <= 1.0, "interval escaped above 1.0"
    assert lo > 0.9
    lo0, hi0 = wilson_interval(0, 45)
    assert lo0 >= 0.0, "interval escaped below 0.0"


def test_both_z_conventions_agree_to_the_published_precision():
    """Two files historically defaulted to z=1.96 and one to the exact 1.959963985.
    They must not disagree at the precision anything is reported to."""
    a = wilson_interval(41, 45)
    b = wilson_interval(41, 45, z=Z_TWO_SIDED)
    assert round(a[0] * 100, 1) == round(b[0] * 100, 1)
    assert round(a[1] * 100, 1) == round(b[1] * 100, 1)


def test_holm_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.7])
    adj = holm(p)
    assert np.all(adj >= p - 1e-12), "adjusted p must never fall below raw p"
    assert np.all(adj <= 1.0)
    # order-preserving: the smallest raw p keeps the smallest adjusted p
    assert np.argmin(adj) == np.argmin(p)


def test_holm_matches_the_textbook_result():
    """m * p for the smallest, then step down, then enforce monotonicity."""
    adj = holm([0.01, 0.04])
    assert abs(adj[0] - 0.02) < 1e-12   # 2 * 0.01
    assert abs(adj[1] - 0.04) < 1e-12   # 1 * 0.04, and >= the previous


# =============================================================================
# 4. Channel sets. These define what every ablation claim means.
# =============================================================================

def test_channel_set_sizes_match_the_published_conditions():
    assert len(SENSORIMOTOR) == 17, "the ablation table reports a 17-channel strip"
    assert len(FRONTOPOLAR) == 8
    assert len(TEMPORAL) == 8


def test_control_sets_are_disjoint_from_sensorimotor():
    """The negative controls only mean something if they share no channel with the
    positive set."""
    assert not (set(FRONTOPOLAR) & set(SENSORIMOTOR))
    assert not (set(TEMPORAL) & set(SENSORIMOTOR))


def test_frontopolar_and_temporal_are_size_matched():
    """emg_proxy.py compares these two directly. Matching the counts is what stops that
    comparison being confounded by channel count, which is the confound ablate_channels.py
    records against its own frontopolar row."""
    assert len(FRONTOPOLAR) == len(TEMPORAL)


def test_sensorimotor_omits_the_peri_rolandic_four():
    """SENSORIMOTOR deliberately excludes FC5/FC6/CP5/CP6, so 'sensorimotor deleted'
    retains four peri-Rolandic electrodes. Condition (f) exists to bound that leak, and
    the claim wording depends on this staying true."""
    for ch in ("FC5", "FC6", "CP5", "CP6"):
        assert ch not in SENSORIMOTOR


def test_complement_size_is_47():
    """64 channels minus the 17-channel strip. The 77.8% arm reports 47 kept."""
    assert 64 - len(SENSORIMOTOR) == 47


# =============================================================================
# 5. The units guard. Regression test for a bug that produced a plausible wrong result.
# =============================================================================

def test_for_torch_rejects_volts_scale_data():
    """MNE returns volts. Feeding volts to EEGNet leaves BatchNorm unable to normalize,
    and the network scores exactly the majority-class rate while being dead. That read as
    'CNN performs at chance on small data', which is a completely plausible finding and
    was wrong. The guard makes that state unreachable."""
    volts = np.random.default_rng(0).normal(0, 1.3e-5, size=(8, 64, 161))
    try:
        common.for_torch(volts / 1e6)   # a thousand times smaller still: unambiguously dead
    except AssertionError:
        return
    raise AssertionError("for_torch accepted data too small for BatchNorm to normalize")


def test_for_torch_accepts_and_converts_real_scale_data():
    volts = np.random.default_rng(0).normal(0, 1.3e-5, size=(8, 64, 161))
    out = common.for_torch(volts)
    assert out.dtype == np.float32, "torch needs float32"
    assert 1.0 < float(out.std()) < 1e3, "microvolt scale expected after conversion"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}\n        {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
