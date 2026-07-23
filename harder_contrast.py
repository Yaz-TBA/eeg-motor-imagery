"""Left fist vs. right fist: the same pipeline on a genuinely harder problem.

Fists-vs-feet is close to the easiest motor-imagery contrast there is. Hands sit
laterally on the motor homunculus and feet sit up near the midline, so the two
classes light up parts of cortex that are centimetres apart and their scalp
patterns barely overlap.

Left-hand vs. right-hand is harder on purpose. Both live on the SAME sensorimotor
strip, just in opposite hemispheres, so the model has to separate mirror-image
patterns rather than distant ones. Accuracy should drop. That drop is the point:
it tells you what the method costs when the classes move closer together, and a
left/right decision is what actually drives a cursor in a real BCI.

TWO TRAPS, both silent, both worth understanding:

  1. RUN NUMBERS. This uses runs 4/8/12, NOT 3/7/11. On PhysioNet's EEGBCI,
     runs 3/7/11 are motor EXECUTION (really moving the hand) and runs 4/8/12
     are motor IMAGERY. An earlier version of this repo's EXPLAINER said 3/7/11
     were the imagined ones. Building this rung on those runs would have decoded
     real movement -- which is a stronger, easier signal -- while the writeup
     claimed imagined movement. The code would have run fine and produced a
     flattering number attached to a false claim.

  2. LABEL MEANING. T1 and T2 do not mean fixed things across the dataset. In
     runs 5/6/9/10/13/14, T1 = both fists and T2 = both feet. In runs
     3/4/7/8/11/12, T1 = LEFT fist and T2 = RIGHT fist. Copy the epoching block
     from decode_csp.py without renaming and every label downstream says "hands"
     while holding left-fist trials.
"""

import matplotlib

matplotlib.use("Agg")

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings

import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECT = 1
RUNS = [4, 8, 12]  # imagined LEFT vs RIGHT fist. Not 3/7/11 -- those are executed.
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
N_PERMUTATIONS = 1000

edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in edf_paths])
eegbci.standardize(raw)
raw.set_montage("standard_1005")
raw.set_eeg_reference("average", projection=False)
raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

# T1 = left fist, T2 = right fist in THESE runs. The integer codes are the same
# as decode_csp.py but the classes they name are completely different.
events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
event_id = dict(left=2, right=3)
epochs = mne.Epochs(
    raw, events, event_id,
    tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
)

labels = epochs.events[:, -1]
data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False)

n_left, n_right = int((labels == 2).sum()), int((labels == 3).sum())
chance = max(n_left, n_right) / len(labels)
print(f"\nRuns {RUNS} (imagined left vs. right fist), subject {SUBJECT}")
print(f"{len(labels)} trials ({n_left} left, {n_right} right) | chance = {chance:.1%}")

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP", csp), ("LDA", LinearDiscriminantAnalysis())])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(clf, data, labels, cv=cv)
observed, null_scores, p_value = permutation_test_score(
    clf, data, labels, scoring="accuracy", cv=cv,
    n_permutations=N_PERMUTATIONS, random_state=42, n_jobs=-1,
)

print(f"\nCSP+LDA accuracy: {scores.mean():.1%}  (+/- {scores.std():.1%})")
print(f"Chance (majority class): {chance:.1%}")
print(f"Per-fold: {np.round(scores, 2)}")
print(f"Permutation test: p = {p_value:.4f} "
      f"(null {null_scores.mean():.1%} +/- {null_scores.std():.1%})")
print(f"\nCompare: fists-vs-feet on this subject was 91.1% (p = 0.0010).")
print(f"Difference here: {100 * (scores.mean() - 0.911):+.1f} points.")

# --- the honesty check, and it is a strong one -------------------------------
# Left and right hand imagery differ by HEMISPHERE. So the CSP patterns must come
# out laterally asymmetric, weighted over C3 vs C4, rather than the central-vs-
# lateral pattern that fists-vs-feet produced. If they are not lateralised,
# either the labels are wrong or the model is riding an artifact.
csp.fit_transform(data, labels)
fig = csp.plot_patterns(epochs.info, components=range(4), ch_type="eeg", show=False)
fig.savefig("csp_patterns_lr.png", dpi=120, bbox_inches="tight")

print("\nSaved csp_patterns_lr.png")
print("Sanity check: these patterns should be LEFT/RIGHT asymmetric (C3 vs C4),")
print("not the central-vs-lateral pattern in csp_patterns.png. Compare them.")
