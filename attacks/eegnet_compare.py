"""Does a compact CNN beat a well-understood classical baseline?

The rung where a units bug meant the CNN was never actually training, and it still
produced a number that looked like a real finding. Everything downstream of that is why
the assert in `for_torch` exists.

EEGNet learns its temporal and spatial filters end to end instead of having them
hand-designed. Structurally it is doing what CSP does, but learned: a temporal
convolution discovers frequency filters, then a depthwise spatial convolution
learns a spatial filter per temporal filter. The interesting question is not
"is deep learning better" but "at what sample size does learning the filters
start to beat designing them".

So this rung runs three experiments, and the third one exists to keep the
comparison fair:

  A. WITHIN-SUBJECT, subject 1, 45 trials.
     The data-starvation case. CSP+LDA gets 91.1% here.

  B. CROSS-SUBJECT LOSO, 20 subjects, ~900 trials, on the IDENTICAL 1-second
     8-30 Hz data every other rung uses. Apples to apples: same folds, same
     preprocessing, same everything but the model.

  C. CROSS-SUBJECT LOSO on a LONGER window (0-4 s) and a WIDER band (4-38 Hz).
     Experiment B is a slightly rigged test. Handing EEGNet one second of
     narrowly band-passed signal removes most of what it is for -- it cannot
     learn useful temporal filters inside a band that has already been filtered
     down to 8-30 Hz. Regime C gives it room, and runs CSP+LDA on the same wider
     data so the comparison stays honest in both directions.

This mirrors the treatment riemannian.py got: run the naive configuration, then
run a fair one, and report both. A negative result that has been diagnosed is
worth far more than one that has not.

NOTE ON DETERMINISM: seeds are fixed for torch, numpy and python, but MPS (Apple
GPU) kernels are not guaranteed bit-reproducible. Expect small run-to-run drift
in the CNN numbers. The classical baselines are exactly reproducible.
"""

import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import mne
import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from joblib import Parallel, delayed
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_score,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECTS = list(range(1, 21))  # same 20 as cross_subject.py / riemannian.py
RUNS = [6, 10, 14]
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BN_EPS = 1e-3  # braindecode EEGNet's BatchNorm2d eps; see for_torch()

# Two preprocessing regimes. "narrow" is what every other rung uses.
NARROW = dict(l_freq=8.0, h_freq=30.0, crop=(1.0, 2.0))
WIDE = dict(l_freq=4.0, h_freq=38.0, crop=(0.0, 4.0))

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_subject(subject, regime):
    mne.set_log_level("ERROR")
    try:
        paths = eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.set_eeg_reference("average", projection=False)
        raw.filter(regime["l_freq"], regime["h_freq"], fir_design="firwin",
                   skip_by_annotation="edge")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=-1.0, tmax=4.0,
                            picks="eeg", baseline=None, preload=True)
        # float64 deliberately. See the dtype note on pool() below.
        X = epochs.copy().crop(*regime["crop"]).get_data(copy=False)
        return subject, X, epochs.events[:, -1]
    except Exception as exc:  # noqa: BLE001
        print(f"  S{subject:03d} skipped: {type(exc).__name__}")
        return subject, None, None


def pool(regime, subjects=SUBJECTS):
    """Pool subjects. Returns float64 X -- see the dtype warning below.

    DTYPE TRAP, found the hard way. Torch wants float32, so the obvious move is
    to cast here once and use the same array for both models. That silently
    breaks CSP.

    Average referencing makes 64-channel data rank 63, not 64, so the covariance
    matrix is singular. In float64 the generalized eigenvalue solver copes (MNE
    reduces the rank internally). In float32 the extra numerical noise makes the
    Cholesky factorization fail outright with "leading minor of order 64 of B is
    not positive definite".

    Worse, sklearn's cross_val_score defaults to error_score=np.nan, so the
    failure does not raise -- it returns nan and the run looks like it worked.
    Hence float64 here, an explicit .astype(np.float32) only where torch needs
    it, and error_score="raise" on every classical call below.
    """
    loaded = [(s, X, y) for s, X, y in
              Parallel(n_jobs=-1)(delayed(load_subject)(s, regime) for s in subjects)
              if X is not None]
    n_times = min(X.shape[-1] for _, X, _ in loaded)
    X_all = np.concatenate([X[:, :, :n_times] for _, X, _ in loaded], axis=0)
    y_all = np.concatenate([y for _, _, y in loaded])
    groups = np.concatenate([np.full(len(y), s) for s, _, y in loaded])
    # torch wants 0-indexed int64 classes, not the 2/3 event codes
    y_all = (y_all == 3).astype(np.int64)
    return X_all, y_all, groups


def for_torch(X):
    """Volts -> microvolts, then float32. BOTH conversions are load-bearing.

    UNITS BUG, caught by adversarial review after the first version of this rung
    was already written up. MNE returns data in VOLTS: X.std() is about 1.3e-5,
    so the variance is about 1.6e-10. braindecode's EEGNet normalizes with
    BatchNorm2d(eps=1e-3), and a variance SEVEN ORDERS OF MAGNITUDE below eps
    means the batch-norm denominator is essentially just eps: it divides by
    sqrt(1e-3) = 0.0316 where the signal's own sigma is ~7e-6, so the first
    BatchNorm's output lands about 4500x too small instead of at unit scale.
    Normalization never engages.

    WITHDRAWN 2026-07-25, AND KEPT VISIBLE. This docstring used to continue:
    "activations stay near 1e-8, and the network cannot train: reaching useful
    logits would need final-layer weights around 1e8, which 100 AdamW steps at
    lr=1e-3 cannot travel to." Both figures are wrong and EXPLAINER.md's rung-10
    section withdraws them. Nothing in the network is near 1e-8; the smallest
    activation standard deviation is ~7e-6, whose VARIANCE is ~4.9e-11, and it
    was that variance being read as a magnitude that produced 1e-8, after which
    1e-8 x 1e8 = 1 produced the weight figure. One error, printed twice.

    WHAT SURVIVES, AND WHERE IT STOPS. The established number is the deficit at
    the FIRST BatchNorm: ~4500x. What reaches the logits is not established --
    each BN stage renormalises, so the deficit decays down the stack, and no
    script in this repo measures the end-to-end scale. Do not restate any
    end-to-end multiplier from this file. "The network cannot train out of it"
    is therefore supported by the OBSERVED behavior below rather than by that
    arithmetic, and downstream prose should say plausible, not established.

    THE OBSERVED BEHAVIOUR, recorded as history rather than as a live result.
    Before the rescale the network emitted class 1 for all 45 trials of subject
    1 and scored 53.3%, exactly subject 1's majority-class rate -- so it read as
    "CNN performs at chance on small data", a completely plausible finding. It
    was a dead network. That configuration is NO LONGER REACHABLE from this
    file: the assert below refuses to run at volts scale, which is the whole
    point of it, so the 53.3% and the all-one-class prediction cannot be
    reproduced here and must not be quoted as if a current script produces
    them. What the file does produce is the rescaled run: experiment A prints
    82.2% for EEGNet on the same code and seed, with predicted class counts
    [21, 24] matching the true counts exactly.

    Every EEG deep-learning recipe scales to microvolts for this reason. CSP is
    unaffected because it works on variance RATIOS, which are scale-invariant.
    """
    Xs = (X * 1e6).astype(np.float32)
    # The guard that would have caught this in the first place. BatchNorm can
    # only normalize if the signal variance is well above its eps.
    var = float(Xs.var())
    assert var > 1e3 * BN_EPS, (
        f"Signal variance {var:.2e} is not comfortably above BatchNorm eps "
        f"{BN_EPS:.0e}. BatchNorm will not normalize and the network will not "
        f"train. Check the units: MNE returns volts, torch models want microvolts."
    )
    return Xs


def assert_not_degenerate(pred, tag):
    """A model that predicts one class for everything is broken, not accurate.

    This is the OTHER guard that would have caught the units bug. A dead network
    scores exactly the majority-class rate, which is indistinguishable from
    "performs at chance" unless you look at what it actually predicted.
    """
    counts = np.bincount(pred, minlength=2)
    assert counts.min() > 0, (
        f"{tag}: model predicted class {int(counts.argmax())} for all "
        f"{counts.sum()} trials. That is a degenerate classifier scoring the "
        f"majority-class rate, not a model performing at chance."
    )


def _guarded_accuracy(y_true, y_pred):
    """Accuracy, but refuses to score a model that predicted one class."""
    assert_not_degenerate(np.asarray(y_pred), "fold")
    return accuracy_score(y_true, y_pred)


# Passed as scoring= to every EEGNet call, so the guard runs on EVERY fold at
# zero extra compute. Defining a guard and never calling it is worse than no
# guard, because it reads as protection in a review.
GUARDED = make_scorer(_guarded_accuracy)


def make_csp():
    return Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def make_eegnet(n_chans, n_times):
    seed_everything()
    return EEGClassifier(
        module=EEGNet,
        module__n_chans=n_chans,
        module__n_outputs=2,
        module__n_times=n_times,
        optimizer=torch.optim.AdamW,
        optimizer__lr=LR,
        optimizer__weight_decay=1e-4,
        batch_size=BATCH_SIZE,
        max_epochs=N_EPOCHS,
        train_split=None,          # no internal validation split; the CV fold is the test
        device=DEVICE,
        verbose=0,
        callbacks=[],
    )


def param_count(n_chans, n_times):
    return sum(p.numel() for p in
               EEGNet(n_chans=n_chans, n_outputs=2, n_times=n_times).parameters())


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


print(f"torch {torch.__version__} | device: {DEVICE} | seed {SEED} | "
      f"{N_EPOCHS} epochs, batch {BATCH_SIZE}")

# ============================================================================
# A. WITHIN-SUBJECT: 45 trials is not enough data for a CNN
# ============================================================================
print(f"\n{'=' * 68}\nA. Within-subject (subject 1, 45 trials, narrow regime)\n{'=' * 68}")
X1, y1, _ = pool(NARROW, subjects=[1])
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

csp_a, t_csp_a = timed(lambda: cross_val_score(
    make_csp(), X1, y1, cv=cv5, error_score="raise"))
seed_everything()
net_a, t_net_a = timed(lambda: cross_val_score(
    make_eegnet(X1.shape[1], X1.shape[2]), for_torch(X1), y1,
    cv=cv5, scoring=GUARDED, error_score="raise"))

chance_a = max(np.mean(y1 == 0), np.mean(y1 == 1))
print(f"chance                {chance_a:.1%}")
print(f"CSP + LDA             {csp_a.mean():.1%} +/- {csp_a.std():.1%}   ({t_csp_a:.1f}s)")
print(f"EEGNet                {net_a.mean():.1%} +/- {net_a.std():.1%}   ({t_net_a:.1f}s)")
print(f"EEGNet - CSP          {100 * (net_a.mean() - csp_a.mean()):+.1f} points")
print(f"EEGNet parameters     {param_count(X1.shape[1], X1.shape[2]):,} "
      f"for {len(y1)} trials")

# ============================================================================
# B. CROSS-SUBJECT, identical data to every other rung
# ============================================================================
print(f"\n{'=' * 68}\nB. Cross-subject LOSO (20 subjects, narrow regime, identical folds)"
      f"\n{'=' * 68}")
Xn, yn, gn = pool(NARROW)
logo = LeaveOneGroupOut()
print(f"pooled {Xn.shape[0]} trials, {Xn.shape[1]} channels x {Xn.shape[2]} samples")

csp_b, t_csp_b = timed(lambda: cross_val_score(
    make_csp(), Xn, yn, groups=gn, cv=logo, n_jobs=-1, error_score="raise"))
seed_everything()
net_b, t_net_b = timed(lambda: cross_val_score(
    make_eegnet(Xn.shape[1], Xn.shape[2]), for_torch(Xn), yn,
    groups=gn, cv=logo, scoring=GUARDED, error_score="raise"))

chance_b = max(np.mean(yn == 0), np.mean(yn == 1))
print(f"chance                {chance_b:.1%}")
print(f"CSP + LDA             {csp_b.mean():.1%} +/- {csp_b.std():.1%}   ({t_csp_b:.1f}s)")
print(f"EEGNet                {net_b.mean():.1%} +/- {net_b.std():.1%}   ({t_net_b:.1f}s)")
print(f"EEGNet - CSP          {100 * (net_b.mean() - csp_b.mean()):+.1f} points")
d = net_b - csp_b
print(f"EEGNet beats baseline on {(d > 0).sum()}/{len(d)} subjects, "
      f"loses on {(d < 0).sum()}, ties on {(d == 0).sum()}")
print(f"EEGNet parameters     {param_count(Xn.shape[1], Xn.shape[2]):,} "
      f"for ~{int(0.95 * len(yn))} training trials")
print(f"compute ratio         EEGNet took {t_net_b / max(t_csp_b, 1e-9):.0f}x "
      f"the baseline's time")

# ============================================================================
# C. THE FAIR TEST: longer window, wider band, both models
# ============================================================================
print(f"\n{'=' * 68}\nC. Cross-subject LOSO (wide regime: 0-4 s, 4-38 Hz)\n{'=' * 68}")
Xw, yw, gw = pool(WIDE)
print(f"pooled {Xw.shape[0]} trials, {Xw.shape[1]} channels x {Xw.shape[2]} samples "
      f"({Xw.shape[2] / Xn.shape[2]:.1f}x longer than regime B)")

csp_c, t_csp_c = timed(lambda: cross_val_score(
    make_csp(), Xw, yw, groups=gw, cv=logo, n_jobs=-1, error_score="raise"))
seed_everything()
net_c, t_net_c = timed(lambda: cross_val_score(
    make_eegnet(Xw.shape[1], Xw.shape[2]), for_torch(Xw), yw,
    groups=gw, cv=logo, scoring=GUARDED, error_score="raise"))

print(f"chance                {max(np.mean(yw == 0), np.mean(yw == 1)):.1%}")
print(f"CSP + LDA (wide)      {csp_c.mean():.1%} +/- {csp_c.std():.1%}   ({t_csp_c:.1f}s)")
print(f"EEGNet (wide)         {net_c.mean():.1%} +/- {net_c.std():.1%}   ({t_net_c:.1f}s)")
print(f"EEGNet - CSP          {100 * (net_c.mean() - csp_c.mean()):+.1f} points")
d2 = net_c - csp_c
print(f"EEGNet beats baseline on {(d2 > 0).sum()}/{len(d2)} subjects, "
      f"loses on {(d2 < 0).sum()}, ties on {(d2 == 0).sum()}")
print(f"EEGNet parameters     {param_count(Xw.shape[1], Xw.shape[2]):,}")
print(f"\nEEGNet narrow -> wide: {100 * (net_c.mean() - net_b.mean()):+.1f} points")
print(f"CSP    narrow -> wide: {100 * (csp_c.mean() - csp_b.mean()):+.1f} points")

# ============================================================================
print(f"\n{'=' * 68}\nSUMMARY\n{'=' * 68}")
rows = [
    ("A within-subject (n=45)", csp_a.mean(), net_a.mean()),
    ("B cross-subject narrow", csp_b.mean(), net_b.mean()),
    ("C cross-subject wide", csp_c.mean(), net_c.mean()),
]
print(f"{'experiment':<26}{'CSP+LDA':>10}{'EEGNet':>10}{'delta':>10}")
for name, c, n in rows:
    print(f"{name:<26}{c:>9.1%}{n:>10.1%}{100 * (n - c):>+9.1f}")
print(f"\ntotal EEGNet training time: "
      f"{(t_net_a + t_net_b + t_net_c) / 60:.1f} min on {DEVICE}")

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(rows))
ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="CSP + LDA", color="#4a6fa5")
ax.bar(x + 0.2, [r[2] for r in rows], width=0.4, label="EEGNet", color="#8e44ad")
ax.axhline(0.5, color="#c0392b", ls="--", lw=1.2, label="chance")
ax.set_xticks(x)
ax.set_xticklabels([r[0].replace(" (", "\n(").replace("subject ", "subject\n")
                    for r in rows], fontsize=9)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1)
ax.set_title("Learned filters vs. designed filters, at three sample sizes")
ax.legend()
fig.tight_layout()
fig.savefig("figures/eegnet_vs_csp.png", dpi=120)
print("\nSaved figures/eegnet_vs_csp.png")
