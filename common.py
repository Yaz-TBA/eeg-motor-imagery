"""Shared pipeline definitions, so "the published pipeline" is one object and not five copies.

WHY THIS FILE EXISTS. Until now every analysis script defined its own `make_clf`,
`wilson_interval`, `holm` and `load_subject`. The copies agreed, and two of them carried
docstrings explaining that they were copied deliberately rather than imported, because
importing a script that defines its helpers at module scope also runs that script's
five-minute analysis. That reasoning was correct and the workaround was the right call at
the time. It also meant the repo had five definitions of what "the classifier" is, and
nothing enforced that they stayed in step.

They are here now, once. Every script keeps its own `if __name__ == "__main__"` guard, so
importing this module runs nothing.

WHAT IS DELIBERATELY *NOT* UNIFIED. The five old `load_subject` functions were not the same
function wearing five hats. They differ on purpose:

  - cross_subject.py / riemannian.py   8-30 Hz, 1.0-2.0 s crop, the published settings
  - eegnet_compare.py                  4-38 Hz, 0-4 s, per-regime
  - regime_decomposition.py            five band/crop cells
  - permutation_design.py              also needs the RUN INDEX per epoch, for block permutation
  - harder_contrast.py                 runs 4/8/12 and a 0.5-5.0 Hz ablation band

`load_epochs` below takes every one of those as an argument rather than flattening them to
a default. If you find yourself adding a branch to it, add a parameter instead.
"""

import numpy as np

# --- the published experiment ------------------------------------------------
SUBJECT = 1
RUNS = [6, 10, 14]          # Task 4: IMAGINED both fists vs. both feet
SEED = 42
L_FREQ, H_FREQ = 8.0, 30.0  # mu + beta
TMIN, TMAX = -1.0, 4.0      # epoch window around the cue
CROP = (1.0, 2.0)           # the imagery window features are taken from
N_TRIALS = 45               # subject 1: 21 hands + 24 feet
CHANCE = 24 / 45            # majority class (feet), 53.3%. NOT 0.5.

ALPHA = 0.05
Z_95 = 1.96                 # the multiplier the published intervals were computed with
Z_TWO_SIDED = 1.959963985   # the exact normal 95% multiplier
BN_EPS = 1e-3               # braindecode EEGNet's BatchNorm eps

# --- channel sets ------------------------------------------------------------
# The FC/C/CP strip over sensorimotor cortex. NOTE it does not contain FC5, FC6,
# CP5 or CP6; ablate_channels.py's condition (f) exists to bound exactly that.
SENSORIMOTOR = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
]

# Frontopolar / anterior-frontal ring: directly above the orbits, where blinks,
# vertical saccades and frontalis EMG dominate and no motor cortex sits beneath.
# The negative control. Deliberately the same size as TEMPORAL, so the one
# comparison that uses both is not confounded by channel count.
FRONTOPOLAR = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]

# Temporalis muscle territory: the EMG that a mu/beta decoder cannot rule out.
TEMPORAL = ["T7", "T8", "T9", "T10", "TP7", "TP8", "FT7", "FT8"]


def make_clf(n_channels=None):
    """The committed pipeline: CSP then LDA.

    CSP sits INSIDE the Pipeline so it refits on the training fold only, inside every
    fold, in every replicate. Any other placement leaks the test fold into filter
    estimation and invalidates the accuracy. test_pipeline.py asserts this.

    `n_channels` clamps n_components for the reduced montages: CSP cannot ask for more
    components than the montage has channels. Pass None (the default) for full-montage
    runs to get the published n_components=4 exactly.
    """
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    n_comp = 4 if n_channels is None else min(4, n_channels - 1)
    return Pipeline([
        ("CSP", CSP(n_components=n_comp, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def wilson_interval(n_correct, n_total, z=Z_95):
    """95% CI for a proportion. Handles small n far better than mean +/- std.

    Default z=1.96 reproduces the intervals already published in the README. Callers
    wanting the exact normal multiplier pass z=Z_TWO_SIDED.

    The result is clamped to [0, 1]. In exact arithmetic Wilson cannot leave the unit
    interval, but at k == n the upper bound lands one ULP above 1.0 (2.2e-16), and this
    repo does hit k == n: subject 70 scores 45/45 in the 109-subject sweep. The clamp is
    below any precision anything is reported to and stops a "100.00000000000002%" ever
    reaching a page. test_pipeline.py asserts the boundary behaviour.
    """
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, input order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def on_lattice(acc, n=N_TRIALS, tol=1e-9):
    """Is `acc` an attainable accuracy for n trials tested exactly once each?

    With n trials in equal folds, accuracy is a count over n, so it can only land on
    multiples of 1/n. This is the check that caught 95.9% and 47.4% in an earlier README:
    there is no integer k with k/45 = 0.959, so those were never measurements.
    """
    return abs(np.asarray(acc, dtype=float) * n - np.round(np.asarray(acc, dtype=float) * n)) < tol


def assert_lattice(scores, n=N_TRIALS, where="", tol=1e-9):
    """Raise if any score is off the k/n lattice. Off-lattice means unequal folds or a
    scorer that is not accuracy, and then the fold-mean is not an accuracy at all."""
    s = np.asarray(scores, dtype=float)
    off = np.abs(s * n - np.round(s * n))
    assert off.max() < tol, (
        f"{where}: {int((off >= tol).sum())} of {s.size} scores are OFF the k/{n} "
        f"lattice (worst deviation {off.max():.3e}). Folds are not equal or the "
        f"scorer is not accuracy; the fold-mean is not an accuracy."
    )


def seed_everything(seed=SEED):
    """Seed python, numpy and torch. torch is imported lazily so that importing this
    module costs nothing for the classical-only scripts."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def for_torch(X, bn_eps=BN_EPS):
    """Volts -> microvolts -> float32, with the guard that catches the units bug.

    MNE returns VOLTS. braindecode's EEGNet normalises with BatchNorm2d(eps=1e-3), and a
    signal variance orders of magnitude below eps means the batch-norm denominator is
    essentially just eps, so normalisation never engages and the network trains from a
    dead start while still scoring the majority-class rate. The assert makes that
    unreachable. CSP is unaffected because it works on variance RATIOS, which are
    scale-invariant.
    """
    Xs = (X * 1e6).astype(np.float32)
    var = float(Xs.var())
    assert var > 1e3 * bn_eps, (
        f"Signal variance {var:.2e} is not comfortably above BatchNorm eps "
        f"{bn_eps:.0e}. BatchNorm will not normalise and the network will not train. "
        f"Check the units: MNE returns volts, torch models want microvolts."
    )
    return Xs


def load_epochs(subject, runs=RUNS, l_freq=L_FREQ, h_freq=H_FREQ,
                tmin=TMIN, tmax=TMAX, crop=CROP,
                return_runs=False, return_ch_names=False):
    """Load and preprocess one subject: concatenate runs, average reference over ALL
    channels, FIR band-pass, epoch around the cue, crop to the feature window.

    The average reference is computed across all 64 electrodes BEFORE any channel subset
    is picked, exactly as decode_csp.py does. Subsets are therefore not electrically
    independent of each other; every channel carries -1/64 of every other. Ablations
    bound an artifact contribution rather than eliminating it.

    Returns (X, y), plus the per-epoch run index if return_runs, plus the channel names
    if return_ch_names, in that order.
    """
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    paths = eegbci.load_data(subjects=subject, runs=runs, update_path=True)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in paths]
    # Run lengths must be recorded BEFORE concatenating: concatenate_raws consumes the
    # list and the run boundary is not recoverable from the result.
    run_edges = np.cumsum([r.n_times for r in raws])
    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=tmin, tmax=tmax,
                        picks="eeg", baseline=None, preload=True)
    y = epochs.events[:, -1]
    X = epochs.copy().crop(tmin=crop[0], tmax=crop[1]).get_data(copy=False)

    out = [X, y]
    if return_runs:
        onsets = epochs.events[:, 0] - raw.first_samp
        out.append(np.searchsorted(run_edges, onsets, side="right"))
    if return_ch_names:
        out.append(epochs.ch_names)
    return tuple(out)
