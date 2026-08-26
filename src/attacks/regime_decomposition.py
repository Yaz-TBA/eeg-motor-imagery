# SIZE-WAIVER: one pre-registered design. Seven cells of one factorial are read
# together, and the pre-cue control the conclusion rests on cannot be separated from
# the cells it controls. Ninety-five lines over the ceiling, all of them one run.
"""What did regime C actually measure? A 2x2 that separates band from window.

The 2x2 that worked out what regime C had actually measured. TL,DR: the CNN was decoding
the cue, not the imagery.

eegnet_compare.py's regime C is not interpretable: against regime B (8-30 Hz,
1-2 s) it changes three things at once,

  1. the band     8-30 Hz  ->  4-38 Hz
  2. the window   1 s      ->  4 s long
  3. the start    1.0 s    ->  0.0 s

and change three factors while measuring one difference and you have measured
nothing. The write-up still told a mechanism story ("CSP wins in B only because
the band was pre-selected for it"), which adversarial review refuted. Item 3 is
the worst because it was never mentioned: starting at 0.0 s pulls in the
cue-evoked response, a different cognitive window, and a model can score on the
visual evoked potential without touching motor imagery.

The factorial:

Crop start pinned at 1.0 s for the four 2x2 cells, so "longer window" means
"more imagery", never "added cue response":

                     8-30 Hz          4-38 Hz
    1.0-2.0 s      narrow-short      wide-short
    1.0-4.0 s      narrow-long       wide-long

Those four identify the band effect, the window effect and their interaction. A
fifth cell reproduces regime C exactly (4-38 Hz, 0.0-4.0 s), so the cue-onset
contribution gets its own number. A sixth decodes 0-1 s alone, and the seventh
is its control: 0-1 s holds the evoked response AND the first second of imagery
(the subject starts imagining at the cue), so the honest control is -1.0 to
0.0 s, which holds neither. That second sits inside the T0 rest, onsets 8.3 s
apart with 4.2 s rest before each. Chance is the correct answer for the pre-cue
cell, which makes it the only cell that can fail: above-chance decoding there is
a defect (identity leakage, drift, ringing, a bug), not a finding. Both models
run in every cell on identical LOSO folds; a grid for the CNN with a fixed
classical baseline would smuggle the confound back in.

Checkpointing, because this is the run that died:

The first regime-C attempt was killed mid-run and left nothing. Every cell here
appends to regime_decomposition.json the moment it finishes, so a kill loses
nothing and a re-run skips finished cells. What that costs is not small: a
resumed run can print a complete report while computing nothing. The 2026-07-25
provenance run did exactly that, its stdout opens "Resuming: 7 cell(s) already
on disk" and regime_decomposition.json is dated 2026-07-23, so rung 11's
published figures are a 07-23 checkpoint that run did not freshly reproduce, and
nothing downstream may say otherwise. check_provenance.py cannot detect this: a
cached cell prints the same stdout a computed one would. To reproduce cold,
delete regime_decomposition.json first; a resumed run says so in its own output.
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
from scipy import stats

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECTS = list(range(1, 21))  # same 20 as cross_subject.py / eegnet_compare.py
RUNS = [6, 10, 14]
SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
BN_EPS = 1e-3
CHECKPOINT = "results/regime_decomposition.json"

# Epoch bounds. tmin is 1 s BEFORE the cue, which is what makes the pre-cue
# control cell possible without re-epoching; every crop below must fit inside
# these bounds and there is an assert at the bottom of CELLS that checks it.
EPOCH_TMIN, EPOCH_TMAX = -1.0, 4.0

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
    # 0-1 s window ALONE. The prediction, kept here verbatim as it was written:
    # "EEGNet should score above chance here and CSP should not."
    #
    # OUTCOME, RECORDED 2026-07-25: HALF OF THAT PREDICTION FAILED, and the
    # failure went unreported for two days. Across the 20 LOSO folds CSP scores
    # 53.7% with p = 0.023 against 0.5. CSP IS above chance in the cue window.
    # EEGNet scores 61.1%. Paired against pre-cue on the same folds, CSP gains
    # +6.0 points (p = 0.0103) and EEGNet +9.3 (p = 0.0000): CSP follows the cue
    # effect LESS than EEGNet does, which is a difference of degree, not the
    # presence-versus-absence this comment predicted. The rung's conclusion, that
    # regime C's ranking flip is cue-locked rather than better imagery decoding,
    # rests on the pre-cue control and the paired deltas and is untouched. The
    # MECHANISM sentence above -- that CSP's log-variance band power is "close
    # to blind" to a phase-locked response -- is not supported by this grid and
    # must not be repeated as if it were. The failure is now printed by the script itself,
    # under the cue/pre-cue table, so it cannot be lost again.
    #
    # ALSO WITHDRAWN, kept visible: this comment used to describe the 0-1 s
    # window as "containing no imagery at all." It does not. Imagery begins AT
    # the cue, as the docstring above and the pre-cue cell below both say, so
    # 0-1 s holds the evoked response AND the first second of imagery. That was
    # the mechanism story invented in the same breath as the number, in the one
    # comment that defines what the cell is for.
    "cue-only":     dict(l_freq=4.0, h_freq=38.0, crop=(0.0, 1.0), band="wide",   window="cue only"),
    # The control for the cell above, and the only cell here where chance is the
    # right answer. "cue-only" is misnamed if taken literally: imagery starts AT
    # the cue, so 0-1 s contains the evoked response AND one second of imagery.
    # This cell takes the second BEFORE the cue -- no flash, no imagery, pure
    # rest -- with the band and window length matched to "cue-only" so the two
    # differ in nothing but their position relative to the cue. At chance here
    # and above chance there, the effect is post-cue. Above chance HERE and the
    # whole rung is measuring a leak, not a brain.
    "pre-cue":      dict(l_freq=4.0, h_freq=38.0, crop=(-1.0, 0.0), band="wide",  window="pre-cue"),
}

# CAVEAT ON THE CONTROL ITSELF. "Contains no post-cue signal" is true of the
# window but not quite true of the filtered samples in it. load_subject filters
# the CONTINUOUS recording and crops afterwards, and MNE's zero-phase firwin at
# 4-38 Hz is a 265-tap symmetric FIR: half-length 132 samples = 0.825 s at
# 160 Hz. Energy at t=0 therefore spreads backwards to t=-0.825 s, so only the
# first 0.175 s of this window is strictly filter-clean. The smear is real and
# measurable -- a linear ERP-style decoder (which, unlike CSP log-variance, can
# see a phase-locked evoked response) scores 53.7% here as-scripted, p=0.029,
# and falls to 52.0%, p=0.16, when the window is rebuilt from a segment that
# physically ENDS at t=0.0 s so no post-cue sample exists to leak. Neither model
# in this grid is affected -- CSP lands at 47.7% and EEGNet at 51.8%, both at
# chance -- but the direction matters: smear can only push a pre-cue score UP,
# so a null here is conservative, while an above-chance pre-cue result would
# have to be re-run with truncated filtering before it meant anything.

# LIMITATION, and this control does not remove it. A pre-cue null localises the
# effect to AFTER the cue; it cannot split "cue flash" from "imagery onset",
# because in EEGBCI those two begin at the same instant. Separating them needs a
# contrast in which the cue looks IDENTICAL across classes, and EEGBCI is not
# that: the target is a bar at the TOP of the screen for fists and the BOTTOM
# for feet. The stimulus is position-confounded with the label, so a
# class-discriminative visual evoked response -- different retinotopic locus,
# different eye movement to fixate it -- necessarily exists post-cue whatever
# the subject imagines. Deciding how much of the 0-1 s score is visual would
# take a dataset with a class-neutral cue (e.g. an identical central symbol, or
# an auditory cue), or a delayed-response design that puts a gap between cue and
# imagery onset. This grid can bound WHEN the effect starts, not WHAT it is.

# Every crop must fit the epoch. A cell asking for samples the epoching never
# produced would otherwise fail as a silently short array in pool()'s min().
for _name, _c in CELLS.items():
    assert EPOCH_TMIN <= _c["crop"][0] < _c["crop"][1] <= EPOCH_TMAX, (
        f"cell {_name!r} crops {_c['crop']} outside epoch bounds "
        f"({EPOCH_TMIN}, {EPOCH_TMAX}); widen the epoching in load_subject()."
    )

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
        epochs = mne.Epochs(raw, events, dict(hands=2, feet=3),
                            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
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
    BatchNorm's eps means normalization never engages and the network is dead
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
    # Say it in the stdout, because the stdout is what gets captured and cited.
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(CHECKPOINT)))
    print(f"  THIS IS NOT A COLD RUN. Every cached cell below is READ FROM "
          f"{CHECKPOINT}\n  (last written {mtime}), not recomputed, and the "
          f"tables at the end are that file's\n  numbers. Delete "
          f"{CHECKPOINT} to reproduce from the data.", flush=True)

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

# The confirmatory test and its control, read as a pair. Same band, same window
# length, same folds; the only difference is which side of the cue they sit on.
if "cue-only" in results and "pre-cue" in results:
    cue, pre = results["cue-only"], results["pre-cue"]
    print(f"\n{'=' * 70}\nTHE CUE, AND THE SECOND BEFORE IT\n{'=' * 70}")
    # "At chance" is a claim about a distribution, so it gets a test rather than
    # an eyeball. One-sample t across the 20 LOSO folds against 0.5, and a
    # paired t between the two windows on the SAME folds.
    #
    # TWO CAVEATS THAT APPLY TO EVERY p PRINTED BELOW, and they both push the
    # same way.
    # (1) LOSO folds are NOT independent observations. Any two training sets
    #     share 19 of their 20 subjects, so fold-to-fold variance underestimates
    #     sampling variance and every p here is anti-conservative. README.md and
    #     EXPLAINER.md flag exactly this hazard for the Wilson interval on n=45;
    #     it applies with more force to a 20-fold t-test.
    # (2) The null is 0.5, not the pooled majority-class rate. On the same 20
    #     subjects, cross_subject.py and eegnet_compare.py both print a pooled
    #     chance of 50.1%, so a test against 0.5 is testing against a floor
    #     slightly below the one the write-up quotes.
    #     The gap is small here and it is not zero; the tests are left against
    #     0.5 so the printed numbers keep matching the checkpoint they came
    #     from, and the discrepancy is named instead of quietly closed.
    def _vs_chance(scores):
        s = np.asarray(scores)
        t, p = stats.ttest_1samp(s, 0.5)
        return f"{s.mean():>7.1%} (p={p:.3f})"

    print(f"{'window':<26}{'CSP+LDA':>20}{'EEGNet':>20}")
    print(f"{'-1.0 to 0.0 s  (pre-cue)':<26}"
          f"{_vs_chance(pre['csp']):>20}{_vs_chance(pre['eegnet']):>20}"
          f"   <- must be chance")
    print(f"{'0.0 to 1.0 s   (post-cue)':<26}"
          f"{_vs_chance(cue['csp']):>20}{_vs_chance(cue['eegnet']):>20}")
    for model, label in (("csp", "CSP+LDA"), ("eegnet", "EEGNet")):
        a, b = np.asarray(pre[model]), np.asarray(cue[model])
        t, p = stats.ttest_rel(b, a)
        print(f"  paired post-cue minus pre-cue, {label:<8} "
              f"{100 * (b - a).mean():+.1f} pts (p={p:.4f})")
    print("\nThe post-cue row on its own proves nothing -- a model can look")
    print("above chance there because of subject leakage or drift that has")
    print("nothing to do with the cue. The pre-cue row is the control that")
    print("rules those out: it is matched in band, length and folds, and it")
    print("contains neither the cue flash nor any imagery. Chance pre-cue plus")
    print("above-chance post-cue localises the effect to cue onset, so regime")
    print("C's 'the ranking flips' is the CNN reading something time-locked to")
    print("the cue rather than learning motor imagery better than CSP does.")
    print("It does NOT say which post-cue thing: see the LIMITATION note next")
    print("to the cell definitions -- EEGBCI's cue is position-confounded with")
    print("the class, so cue flash and imagery onset cannot be separated here.")

    # HALF THE STATED PREDICTION FAILED, and this block exists so the script
    # says so itself instead of leaving it to a reader who recomputes the JSON.
    _csp_cue = np.asarray(cue["csp"])
    _t_cue, _p_cue = stats.ttest_1samp(_csp_cue, 0.5)
    if _csp_cue.mean() > 0.5 and _p_cue < 0.05:
        print("\nAND HALF THE PREDICTION THIS CELL WAS BUILT ON FAILED.")
        print("The cue-only cell was written to predict that EEGNet would score")
        print("above chance in the cue window and that CSP would NOT. CSP scores")
        print(f"{_csp_cue.mean():.1%} there (p={_p_cue:.3f}), which is above chance.")
        print("So CSP is not blind to whatever the cue window carries; it simply")
        print("follows it less than EEGNet does. The cue-locking conclusion above")
        print("stands, because it rests on the pre-cue control and the paired")
        print("deltas. The MECHANISM story -- 'CSP's log-variance features cannot")
        print("see a phase-locked evoked response' -- does not, and should not be")
        print("repeated anywhere downstream.")

    print("\nEvery p in this block is a t-test across 20 leave-one-subject-out")
    print("folds treated as independent draws. They overlap in 19/20 of their")
    print("training data, so these p-values are anti-conservative. Read them as")
    print("ordering evidence, not as calibrated tail probabilities.")

# --- figure ------------------------------------------------------------------
order = ["narrow-short", "wide-short", "narrow-long", "wide-long", "original-C",
         "cue-only", "pre-cue"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(order))
ax.bar(x - 0.2, [results[c]["csp_mean"] for c in order], width=0.4,
       label="CSP + LDA", color="#4a6fa5")
ax.bar(x + 0.2, [results[c]["eegnet_mean"] for c in order], width=0.4,
       label="EEGNet", color="#8e44ad")
ax.axhline(0.5, color="#c0392b", ls="--", lw=1.2, label="chance")
ax.axvline(3.5, color="#7f8c8d", ls=":", lw=1.2)   # 2x2 | cue-window cells
ax.axvline(5.5, color="#7f8c8d", ls=":", lw=1.2)   # post-cue | the control
ax.set_xticks(x)
ax.set_xticklabels([c.replace("-", "\n") for c in order], fontsize=9)
ax.set_ylabel("accuracy (LOSO, 20 subjects)")
ax.set_ylim(0, 1)
ax.set_title("Regime C decomposed: band and window varied independently\n"
             "(cells 5-6 add the undocumented 0 s crop start; cell 7 is the "
             "pre-cue control, which must land on chance)", fontsize=11)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("figures/regime_decomposition.png", dpi=120)
print("\nSaved figures/regime_decomposition.png")
