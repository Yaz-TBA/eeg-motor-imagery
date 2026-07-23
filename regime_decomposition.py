"""What did regime C actually measure? A 2x2 that separates band from window.

eegnet_compare.py runs three regimes, and the third one is not interpretable.
Regime B is 8-30 Hz on a 1-2 s crop. Regime C is 4-38 Hz on a 0-4 s crop. That
changes THREE things at once:

  1. the band       8-30 Hz  ->  4-38 Hz
  2. the window     1 s      ->  4 s long
  3. the START      1.0 s    ->  0.0 s

Change three factors and measure one difference and you have measured nothing.
The write-up nonetheless told a mechanism story about it ("CSP wins in B only
because the band was pre-selected for it"), which adversarial review refuted.
Item 3 is the worst of the three because it was never mentioned at all: moving
the crop start to 0.0 s pulls in the cue-evoked response, so regime C decodes a
different COGNITIVE window, not merely a longer one. A model can score on the
visual evoked potential to the cue and never touch motor imagery.

So this rung re-runs it as a factorial. Crop start is pinned at 1.0 s for the
four cells of the 2x2, so "longer window" means "more imagery", not "now with
added cue response":

                     8-30 Hz          4-38 Hz
    1.0-2.0 s      narrow-short      wide-short
    1.0-4.0 s      narrow-long       wide-long

From those four cells the band effect, the window effect, and the interaction
between them are each identified separately. A fifth cell reproduces the
original regime C exactly (4-38 Hz, 0.0-4.0 s) so the cue-onset contribution --
the change nobody documented -- gets its own number instead of hiding inside
the others.

Both models run in every cell on identical LOSO folds. Reporting the CNN across
a grid while holding the classical baseline fixed would smuggle the confound
back in through the comparison.

CHECKPOINTING, because this is the run that died. The first attempt at regime C
was killed mid-run and left nothing behind, so a multi-hour job produced zero
recoverable results. Every cell here appends to regime_decomposition.json the
moment it finishes. Kill this script at any point and every completed cell
survives; re-running skips what is already on disk.
"""

import matplotlib

matplotlib.use("Agg")

import os

os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import json
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
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECTS = list(range(1, 21))  # same 20 as cross_subject.py / eegnet_compare.py
RUNS = [6, 10, 14]
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BN_EPS = 1e-3
CHECKPOINT = "regime_decomposition.json"

# The 2x2, plus the original regime C as a fifth reference cell. Crop start is
# 1.0 s everywhere except that fifth cell, which exists precisely to price the
# 0.0 s start separately.
CELLS = {
    "narrow-short": dict(l_freq=8.0, h_freq=30.0, crop=(1.0, 2.0), band="narrow", window="short"),
    "wide-short":   dict(l_freq=4.0, h_freq=38.0, crop=(1.0, 2.0), band="wide",   window="short"),
    "narrow-long":  dict(l_freq=8.0, h_freq=30.0, crop=(1.0, 4.0), band="narrow", window="long"),
    "wide-long":    dict(l_freq=4.0, h_freq=38.0, crop=(1.0, 4.0), band="wide",   window="long"),
    "original-C":   dict(l_freq=4.0, h_freq=38.0, crop=(0.0, 4.0), band="wide",   window="long+cue"),
    # The confirmatory cell. The 2x2 shows the whole regime-C effect comes from
    # admitting the 0-1 s cue period, and the obvious explanation is that a CNN
    # with temporal convolutions can read a phase-locked cue-evoked response
    # while CSP's log-variance band power is close to blind to it. That is an
    # interpretation, not a measurement, so this cell measures it: decode the
    # CUE WINDOW ALONE, containing no imagery at all. If the explanation is
    # right, EEGNet should score above chance here and CSP should not.
    "cue-only":     dict(l_freq=4.0, h_freq=38.0, crop=(0.0, 1.0), band="wide",   window="cue only"),
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_subject(subject, cell):
    mne.set_log_level("ERROR")
    try:
        paths = eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.set_eeg_reference("average", projection=False)
        raw.filter(cell["l_freq"], cell["h_freq"], fir_design="firwin",
                   skip_by_annotation="edge")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=-1.0, tmax=4.0,
                            picks="eeg", baseline=None, preload=True)
        X = epochs.copy().crop(*cell["crop"]).get_data(copy=False)
        return subject, X, epochs.events[:, -1]
    except Exception as exc:  # noqa: BLE001
        print(f"  S{subject:03d} skipped: {type(exc).__name__}", flush=True)
        return subject, None, None


def pool(cell):
    """float64 out. Casting to float32 here silently breaks CSP -- average
    referencing makes the covariance singular and float32 noise tips the
    Cholesky factorization over. See the longer note in eegnet_compare.py."""
    loaded = [(s, X, y) for s, X, y in
              Parallel(n_jobs=-1)(delayed(load_subject)(s, cell) for s in SUBJECTS)
              if X is not None]
    n_times = min(X.shape[-1] for _, X, _ in loaded)
    X_all = np.concatenate([X[:, :, :n_times] for _, X, _ in loaded], axis=0)
    y_all = np.concatenate([y for _, _, y in loaded])
    groups = np.concatenate([np.full(len(y), s) for s, _, y in loaded])
    return X_all, (y_all == 3).astype(np.int64), groups


def for_torch(X):
    """Volts -> microvolts -> float32, with the guard that would have caught the
    units bug. MNE returns volts; a variance seven orders of magnitude below
    BatchNorm's eps means normalisation never engages and the network is dead
    while still scoring the majority-class rate."""
    Xs = (X * 1e6).astype(np.float32)
    var = float(Xs.var())
    assert var > 1e3 * BN_EPS, (
        f"Signal variance {var:.2e} is not comfortably above BatchNorm eps "
        f"{BN_EPS:.0e}. Check units: MNE returns volts, torch wants microvolts."
    )
    return Xs


def _guarded_accuracy(y_true, y_pred):
    """A model predicting one class for everything is broken, not accurate."""
    counts = np.bincount(np.asarray(y_pred), minlength=2)
    assert counts.min() > 0, (
        f"Model predicted class {int(counts.argmax())} for all {counts.sum()} "
        f"trials: a degenerate classifier scoring the majority-class rate."
    )
    return accuracy_score(y_true, y_pred)


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
        train_split=None,
        device=DEVICE,
        verbose=0,
        callbacks=[],
    )


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as fh:
            return json.load(fh)
    return {}


def save_checkpoint(results):
    with open(CHECKPOINT, "w") as fh:
        json.dump(results, fh, indent=2)


def run_cell(name, cell):
    X, y, g = pool(cell)
    logo = LeaveOneGroupOut()
    print(f"  pooled {X.shape[0]} trials, {X.shape[1]} ch x {X.shape[2]} samples",
          flush=True)

    t0 = time.perf_counter()
    csp = cross_val_score(make_csp(), X, y, groups=g, cv=logo, n_jobs=-1,
                          error_score="raise")
    t_csp = time.perf_counter() - t0

    seed_everything()
    t0 = time.perf_counter()
    net = cross_val_score(make_eegnet(X.shape[1], X.shape[2]), for_torch(X), y,
                          groups=g, cv=logo, scoring=GUARDED, error_score="raise")
    t_net = time.perf_counter() - t0

    print(f"  CSP+LDA {csp.mean():.1%} +/- {csp.std():.1%} ({t_csp:.0f}s) | "
          f"EEGNet {net.mean():.1%} +/- {net.std():.1%} ({t_net:.0f}s) | "
          f"delta {100 * (net.mean() - csp.mean()):+.1f}", flush=True)

    return dict(
        band=cell["band"], window=cell["window"], crop=list(cell["crop"]),
        l_freq=cell["l_freq"], h_freq=cell["h_freq"],
        n_trials=int(X.shape[0]), n_times=int(X.shape[2]),
        csp=csp.tolist(), eegnet=net.tolist(),
        csp_mean=float(csp.mean()), eegnet_mean=float(net.mean()),
        t_csp=t_csp, t_net=t_net,
    )


print(f"torch {torch.__version__} | device: {DEVICE} | seed {SEED} | "
      f"{N_EPOCHS} epochs, batch {BATCH_SIZE}", flush=True)

results = load_checkpoint()
if results:
    print(f"Resuming: {len(results)} cell(s) already on disk "
          f"({', '.join(results)})", flush=True)

for name, cell in CELLS.items():
    if name in results:
        print(f"\n{name}: cached, skipping", flush=True)
        continue
    print(f"\n{'=' * 70}\n{name}: {cell['l_freq']}-{cell['h_freq']} Hz, "
          f"crop {cell['crop'][0]}-{cell['crop'][1]} s\n{'=' * 70}", flush=True)
    results[name] = run_cell(name, cell)
    save_checkpoint(results)          # survive a kill
    print(f"  checkpointed -> {CHECKPOINT}", flush=True)

# ============================================================================
# The decomposition the original regime C could not support
# ============================================================================
print(f"\n{'=' * 70}\nTHE 2x2\n{'=' * 70}")
print(f"{'':<16}{'8-30 Hz':>22}{'4-38 Hz':>22}")
for window, short_long in (("1.0-2.0 s", "short"), ("1.0-4.0 s", "long")):
    n = results["narrow-" + short_long]
    w = results["wide-" + short_long]
    cell_n = "CSP {:.1%} / NET {:.1%}".format(n["csp_mean"], n["eegnet_mean"])
    cell_w = "CSP {:.1%} / NET {:.1%}".format(w["csp_mean"], w["eegnet_mean"])
    print(f"{window:<16}{cell_n:>22}{cell_w:>22}")

print(f"\n{'=' * 70}\nMAIN EFFECTS AND INTERACTION (percentage points)\n{'=' * 70}")
print(f"{'model':<10}{'band':>10}{'window':>10}{'interaction':>14}{'cue onset':>12}")
for model in ("csp", "eegnet"):
    k = f"{model}_mean"
    ns, ws = results["narrow-short"][k], results["wide-short"][k]
    nl, wl = results["narrow-long"][k], results["wide-long"][k]
    orig = results["original-C"][k]
    band = 100 * (((ws - ns) + (wl - nl)) / 2)
    window = 100 * (((nl - ns) + (wl - ws)) / 2)
    interaction = 100 * ((wl - ws) - (nl - ns))
    cue = 100 * (orig - wl)
    print(f"{model:<10}{band:>+10.1f}{window:>+10.1f}{interaction:>+14.1f}{cue:>+12.1f}")

print("\nband        = wide minus narrow, averaged over both window lengths")
print("window      = long minus short, averaged over both bands")
print("interaction = how much the band effect changes when the window lengthens")
print("cue onset   = moving the crop start 1.0 s -> 0.0 s, the undocumented third change")

# The confirmatory test: the cue window on its own contains NO imagery.
if "cue-only" in results:
    cue = results["cue-only"]
    print(f"\n{'=' * 70}\nCUE WINDOW ALONE (0-1 s, no imagery in it at all)\n{'=' * 70}")
    print(f"CSP + LDA  {cue['csp_mean']:.1%}")
    print(f"EEGNet     {cue['eegnet_mean']:.1%}")
    print("\nIf EEGNet scores above chance on a window containing no imagery, then")
    print("regime C's 'the ranking flips' is the CNN reading the cue-evoked")
    print("response, not learning motor imagery better than CSP does.")

# --- figure ------------------------------------------------------------------
order = ["narrow-short", "wide-short", "narrow-long", "wide-long", "original-C",
         "cue-only"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(order))
ax.bar(x - 0.2, [results[c]["csp_mean"] for c in order], width=0.4,
       label="CSP + LDA", color="#4a6fa5")
ax.bar(x + 0.2, [results[c]["eegnet_mean"] for c in order], width=0.4,
       label="EEGNet", color="#8e44ad")
ax.axhline(0.5, color="#c0392b", ls="--", lw=1.2, label="chance")
ax.axvline(3.5, color="#7f8c8d", ls=":", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("-", "\n") for c in order], fontsize=9)
ax.set_ylabel("accuracy (LOSO, 20 subjects)")
ax.set_ylim(0, 1)
ax.set_title("Regime C decomposed: band and window varied independently\n"
             "(rightmost cell adds the undocumented 0 s crop start)", fontsize=11)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("regime_decomposition.png", dpi=120)
print("\nSaved regime_decomposition.png")
