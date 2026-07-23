"""Classify covariance matrices on their own curved geometry, not as flat vectors.

This rung exists to answer the failure the previous one measured. cross_subject.py
shows CSP+LDA losing accuracy when it has to generalise to an unseen person. The
reason is geometric: CSP learns spatial filters tuned to the training population's
anatomy, and a new skull shifts everything.

Riemannian methods attack that directly. A trial's spatial covariance matrix is a
symmetric positive definite (SPD) matrix, and SPD matrices do not live in ordinary
flat space -- they live on a curved manifold. Treating them as flat feature vectors
(which is effectively what log-variance does) distorts the distances between them.
Measuring distance along the manifold instead respects the actual geometry, and it
turns out to be far more robust to the between-subject shift.

Two pipelines, both classical and both interpretable:

  Covariances -> MDM
      Minimum Distance to Riemannian Mean. Compute each class's mean covariance
      matrix on the manifold, then assign a trial to whichever mean is closer.
      Almost embarrassingly simple, and a strong baseline.

  Covariances -> TangentSpace -> LogisticRegression
      Project every covariance matrix onto the flat tangent plane touching the
      manifold at the geometric mean of the data. Near that point the curved space
      is locally flat, so ordinary linear classifiers work properly again. Usually
      the stronger of the two.

Compared against CSP+LDA on THE IDENTICAL leave-one-subject-out folds -- a paired
comparison on different splits is not a comparison at all. Where Riemannian loses,
this script says so.
"""

import matplotlib

matplotlib.use("Agg")

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import mne
from joblib import Parallel, delayed
from mne.datasets import eegbci
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECTS = list(range(1, 21))  # same 20 as cross_subject.py, so the folds match
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0


def load_subject(subject):
    mne.set_log_level("ERROR")
    try:
        paths = eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.set_eeg_reference("average", projection=False)
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")
        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=TMIN, tmax=TMAX,
                            picks="eeg", baseline=None, preload=True)
        return (subject, epochs.copy().crop(1.0, 2.0).get_data(copy=False),
                epochs.events[:, -1], epochs.ch_names)
    except Exception as exc:  # noqa: BLE001
        print(f"  S{subject:03d} skipped: {type(exc).__name__}")
        return subject, None, None, None


print(f"Loading {len(SUBJECTS)} subjects...")
loaded = [(s, X, y, ch) for s, X, y, ch in
          Parallel(n_jobs=-1)(delayed(load_subject)(s) for s in SUBJECTS)
          if X is not None]

n_samples = min(X.shape[-1] for _, X, _, _ in loaded)
X_all = np.concatenate([X[:, :, :n_samples] for _, X, _, _ in loaded], axis=0)
y_all = np.concatenate([y for _, _, y, _ in loaded])
groups = np.concatenate([np.full(len(y), s) for s, _, y, _ in loaded])
ch_names = loaded[0][3]
print(f"Pooled {X_all.shape[0]} trials from {len(loaded)} subjects")

# --- give the manifold methods a fair test -----------------------------------
# A 64x64 covariance matrix has 64*65/2 = 2080 free parameters, estimated from
# 161 samples per trial, and its tangent space has 2080 features for ~855
# training trials. That is hopelessly under-determined, and it is the standard
# trap when Riemannian methods are dropped onto a full montage. The literature
# applies them to a sensorimotor subset. So run BOTH and let the comparison show
# whether a loss is the method's fault or the montage's.
MOTOR_CHANNELS = ["FC3", "FC1", "FCz", "FC2", "FC4",
                  "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP3", "CP1", "CPz", "CP2", "CP4"]
motor_idx = [ch_names.index(c) for c in MOTOR_CHANNELS if c in ch_names]
X_motor = X_all[:, motor_idx, :]
print(f"Sensorimotor subset: {len(motor_idx)} channels "
      f"({len(motor_idx) * (len(motor_idx) + 1) // 2} covariance parameters "
      f"vs {X_all.shape[1] * (X_all.shape[1] + 1) // 2} for the full montage)")

PIPELINES = {
    "CSP + LDA (baseline)": (Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ]), X_all),
    "Cov + MDM (64 ch)": (Pipeline([
        ("Cov", Covariances(estimator="oas")),
        ("MDM", MDM()),
    ]), X_all),
    "Cov + TS + LR (64 ch)": (Pipeline([
        ("Cov", Covariances(estimator="oas")),
        ("TS", TangentSpace()),
        ("LR", LogisticRegression(max_iter=1000)),
    ]), X_all),
    "Cov + MDM (motor)": (Pipeline([
        ("Cov", Covariances(estimator="oas")),
        ("MDM", MDM()),
    ]), X_motor),
    "Cov + TS + LR (motor)": (Pipeline([
        ("Cov", Covariances(estimator="oas")),
        ("TS", TangentSpace()),
        ("LR", LogisticRegression(max_iter=1000)),
    ]), X_motor),
}

logo = LeaveOneGroupOut()
chance = max(np.mean(y_all == 2), np.mean(y_all == 3))

print(f"\nLeave-one-subject-out, identical folds for every pipeline.")
print(f"Pooled chance: {chance:.1%}\n")

results = {}
for name, (pipe, X) in PIPELINES.items():
    scores = cross_val_score(pipe, X, y_all, groups=groups, cv=logo, n_jobs=-1)
    results[name] = scores
    print(f"{name:<24} {scores.mean():.1%} +/- {scores.std():.1%}  "
          f"(median {np.median(scores):.1%}, "
          f"{(scores > chance).sum()}/{len(scores)} above chance)")

# --- the paired comparison, including where Riemannian loses ------------------
base = results["CSP + LDA (baseline)"]
print(f"\n{'=' * 62}")
print("Paired against the CSP+LDA baseline, subject by subject")
print(f"{'=' * 62}")
for name, scores in results.items():
    if name.startswith("CSP"):
        continue
    delta = scores - base
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    print(f"{name}:")
    print(f"  mean change {100 * delta.mean():+.1f} points "
          f"(best {100 * delta.max():+.1f}, worst {100 * delta.min():+.1f})")
    print(f"  beats baseline on {wins}/{len(delta)} subjects, "
          f"loses on {losses}, ties on {len(delta) - wins - losses}")

best_name = max(results, key=lambda k: results[k].mean())
print(f"\nBest on this data: {best_name} ({results[best_name].mean():.1%})")

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
names = list(results)
data = [results[n] for n in names]
bp = ax.boxplot(data, tick_labels=[n.replace(" (", "\n(") for n in names],
                patch_artist=True, widths=0.55)
for patch, color in zip(bp["boxes"],
                        ["#4a6fa5", "#e67e22", "#27ae60", "#e67e22", "#27ae60"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for i, scores in enumerate(data, 1):
    ax.scatter(np.random.default_rng(42).normal(i, 0.05, len(scores)), scores,
               s=14, color="#2c3e50", alpha=0.6, zorder=3)
ax.axhline(chance, color="#c0392b", ls="--", lw=1.3, label=f"chance ({chance:.1%})")
ax.set_ylabel("cross-subject accuracy (LOSO)")
ax.set_title("Does manifold geometry survive the jump to a new person?")
ax.legend()
fig.tight_layout()
fig.savefig("riemannian_comparison.png", dpi=120)
print("\nSaved riemannian_comparison.png")
