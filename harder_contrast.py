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

WHAT THIS RUNG RETRACTED, AND WHY IT IS STILL HERE

The first version of this file printed one sentence: "fists-vs-feet on this
subject was 91.1%, so a harder contrast costs 17.8 points." Both halves of that
were hardcoded string constants, and the framing around them was wrong:

  - It is n = 1. Rung 6 had just swept all 109 subjects and this rung silently
    reverted to the single cleanest one, then reported the gap as if it were a
    property of the method rather than of subject 1.
  - The two conditions come from DIFFERENT RECORDING RUNS (4/8/12 vs 6/10/14),
    so "harder contrast" cannot be separated from "different session."
  - The 1-second crop was the joint maximum. The window sweep printed below
    shows adjacent windows swinging by tens of points; the published one is the
    peak of that noise.
  - The comparison number was frozen in a print statement. If the baseline moved
    -- and it did, from 94.4% to 91.1% when rung 5 fixed the estimator -- this
    file would have kept quoting the stale one. So it is now RECOMPUTED at
    runtime, from the same pipeline, in this same process.

And then the real problem, which is the reason this rung is worth keeping. The
PhysioNet protocol puts the target on the LEFT or RIGHT of the screen and leaves
it there for the whole trial, so a lateralised visual stimulus is present for the
entire decoding window. A left/right decoder can ride the subject's EYES instead
of their motor cortex. EEGMMIDB ships no EOG channels and this pipeline has no
ICA, so that confound can be bounded by ablation but never removed. The ablation
is run below, at MATCHED settings, because an ablation run at different settings
than the headline is not a control -- it is a second experiment.
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
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECT = 1
RUNS = [4, 8, 12]  # imagined LEFT vs RIGHT fist. Not 3/7/11 -- those are executed.
REFERENCE_RUNS = [6, 10, 14]  # imagined both fists vs both feet -- the rung-4 baseline.
TMIN, TMAX = -1.0, 4.0
CROP = (1.0, 2.0)  # the imagery window used by every rung in this repo
L_FREQ, H_FREQ = 8.0, 30.0
LOW_FREQ = (0.5, 5.0)  # the band ocular/gaze drift lives in, not the mu/beta band
N_PERMUTATIONS = 1000
SEED = 42

# The 8 most anterior electrodes in the 64-channel montage. If a "motor imagery"
# decoder works using ONLY these, it is not reading motor cortex -- they sit above
# the eyes, which is exactly where a lateralised gaze artifact would show up.
FRONTOPOLAR = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]


def load_raw(runs):
    """Load + standardise + average-reference one subject's runs. No filtering yet:
    the band is a variable in the ablation below, so it is applied per-analysis."""
    paths = eegbci.load_data(subjects=SUBJECT, runs=runs, update_path=True)
    raw = mne.concatenate_raws([mne.io.read_raw_edf(p, preload=True) for p in paths])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    return raw


def make_epochs(raw, l_freq, h_freq, event_id):
    """Band-pass a COPY of the raw and epoch it. Filtering a copy matters: filtering
    in place would make each band in the sweep inherit the previous band's filter."""
    r = raw.copy().filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")
    events, _ = mne.events_from_annotations(r, event_id=dict(T1=2, T2=3))
    return mne.Epochs(
        r, events, event_id,
        tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
    )


def make_clf(n_channels):
    """A FRESH pipeline per call -- CSP must refit inside every training fold.
    CSP cannot ask for more components than the montage has channels."""
    return Pipeline([
        ("CSP", CSP(n_components=min(4, n_channels - 1), reg=None,
                    log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])


def cv_split():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def features(epochs, picks=None, crop=CROP):
    e = epochs.copy().pick(picks) if picks else epochs
    y = e.events[:, -1]
    X = e.copy().crop(tmin=crop[0], tmax=crop[1]).get_data(copy=False)
    return X, y


def csp_accuracy(epochs, picks=None, crop=CROP):
    X, y = features(epochs, picks, crop)
    # error_score="raise": a fold that throws (rank-deficient covariance, a NaN
    # channel) would otherwise be scored as nan and silently drag the mean down,
    # or vanish into a warning. On 45 trials a single swallowed fold is 20% of
    # the estimate. Fail loudly instead.
    return cross_val_score(make_clf(X.shape[1]), X, y, cv=cv_split(),
                           error_score="raise")


def p_as_bound(p_value, n_permutations=N_PERMUTATIONS):
    """sklearn computes p = (C + 1)/(n + 1) where C counts permutations scoring >=
    observed, so 1/1001 is the FLOOR of a 1000-shuffle test. Printing "0.0010"
    invites reading the test's resolution limit as a measurement. Same convention
    as decode_csp.py and evaluate_honestly.py."""
    floor = 1.0 / (n_permutations + 1)
    return f"<= {floor:.3f}" if p_value <= floor + 1e-12 else f"=  {p_value:.4f}"


def permutation(epochs, picks=None, crop=CROP):
    X, y = features(epochs, picks, crop)
    # permutation_test_score takes no error_score argument; it fits directly and
    # lets exceptions propagate, which is the behaviour we want anyway.
    observed, null_scores, p_value = permutation_test_score(
        make_clf(X.shape[1]), X, y, scoring="accuracy", cv=cv_split(),
        n_permutations=N_PERMUTATIONS, random_state=SEED, n_jobs=-1,
    )
    return observed, null_scores, p_value


# --- 1. the harder contrast --------------------------------------------------
raw_lr = load_raw(RUNS)
epochs = make_epochs(raw_lr, L_FREQ, H_FREQ, dict(left=2, right=3))
labels = epochs.events[:, -1]
data = epochs.copy().crop(tmin=CROP[0], tmax=CROP[1]).get_data(copy=False)

n_left, n_right = int((labels == 2).sum()), int((labels == 3).sum())
chance = max(n_left, n_right) / len(labels)
print(f"\nRuns {RUNS} (imagined left vs. right fist), subject {SUBJECT}")
print(f"{len(labels)} trials ({n_left} left, {n_right} right) | chance = {chance:.1%}")

scores = csp_accuracy(epochs)
_, null_lr, p_lr = permutation(epochs)

print(f"\nCSP+LDA accuracy: {scores.mean():.1%}  (+/- {scores.std():.1%})")
print(f"Chance (majority class): {chance:.1%}")
print(f"Per-fold: {np.round(scores, 2)}")
print(f"Permutation test: p {p_as_bound(p_lr)} "
      f"(null {null_lr.mean():.1%} +/- {null_lr.std():.1%}, max {null_lr.max():.1%})")

# --- 2. the comparison, RECOMPUTED here rather than quoted from memory --------
print(f"\n--- Reference contrast, runs {REFERENCE_RUNS} (both fists vs. both feet) ---")
print("Recomputed in this process with the identical pipeline, so the two numbers")
print("cannot drift apart the way a hardcoded constant does.")
raw_ff = load_raw(REFERENCE_RUNS)
epochs_ff = make_epochs(raw_ff, L_FREQ, H_FREQ, dict(hands=2, feet=3))
scores_ff = csp_accuracy(epochs_ff)
_, null_ff, p_ff = permutation(epochs_ff)
print(f"Fists-vs-feet: {scores_ff.mean():.1%} (+/- {scores_ff.std():.1%}), "
      f"p {p_as_bound(p_ff)}")

delta = 100 * (scores.mean() - scores_ff.mean())
print(f"Left/right minus fists/feet: {delta:+.1f} points, on subject {SUBJECT} only.")
print("What that gap is NOT: the cost of a harder contrast. It is one subject, two")
print("different recording sessions, 45 trials each, and the crop window below is")
print("the best of five. Treat it as a direction, not a quantity.")

# --- 3. window sensitivity: is the published crop special? --------------------
print("\n--- Crop-window sweep (left vs. right, all 64 channels, 8-30 Hz) ---")
windows = [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0), (1.5, 2.5), (2.0, 3.0)]
sweep = [(w, csp_accuracy(epochs, crop=w).mean()) for w in windows]
for (t0, t1), acc in sweep:
    mark = "  <- published" if (t0, t1) == CROP else ""
    print(f"  {t0:.1f}-{t1:.1f} s: {acc:.1%}{mark}")
spread = 100 * (max(a for _, a in sweep) - min(a for _, a in sweep))
print(f"Range across five overlapping 1-second windows: {spread:.1f} points.")
print("Overlapping windows of the same trials should not disagree by that much.")
print("That spread is the honest error bar on this rung, and it is larger than")
print("the gap in section 2.")

# --- 4. the gaze ablation, at MATCHED settings -------------------------------
print("\n--- Gaze confound: can 8 frontopolar channels do this on their own? ---")
print(f"Frontopolar set: {', '.join(FRONTOPOLAR)}")
print(f"All rows use the SAME trials, folds and {CROP[0]}-{CROP[1]} s window as the")
print("headline above. Only the channels and the band change.")

epochs_low = make_epochs(raw_lr, LOW_FREQ[0], LOW_FREQ[1], dict(left=2, right=3))
rows = [
    (f"all 64 ch, {L_FREQ:.0f}-{H_FREQ:.0f} Hz (headline)", epochs, None),
    (f"frontopolar 8 ch, {L_FREQ:.0f}-{H_FREQ:.0f} Hz", epochs, FRONTOPOLAR),
    (f"all 64 ch, {LOW_FREQ[0]}-{LOW_FREQ[1]} Hz", epochs_low, None),
    (f"frontopolar 8 ch, {LOW_FREQ[0]}-{LOW_FREQ[1]} Hz", epochs_low, FRONTOPOLAR),
]
ablation = {}
for name, eps, picks in rows:
    acc = csp_accuracy(eps, picks=picks).mean()
    ablation[name] = acc
    print(f"  {name:<40} {acc:.1%}")

head = ablation[f"all 64 ch, {L_FREQ:.0f}-{H_FREQ:.0f} Hz (headline)"]
fp_low = ablation[f"frontopolar 8 ch, {LOW_FREQ[0]}-{LOW_FREQ[1]} Hz"]
print(f"\nAt matched settings the frontopolar-only decoder is {100*(head - fp_low):.1f} "
      f"points BELOW the\n64-channel headline, not level with it. An earlier version of the "
      "writeup said\nthe frontopolar decoder 'matches the 64-channel result'. It does not, at "
      "matched\nsettings. Here is where that claim came from:")

print(f"\n--- Same frontopolar decoder, {LOW_FREQ[0]}-{LOW_FREQ[1]} Hz, across windows ---")
fp_windows = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 4.0)]
for t0, t1 in fp_windows:
    acc = csp_accuracy(epochs_low, picks=FRONTOPOLAR, crop=(t0, t1)).mean()
    note = "  <- matched to the headline" if (t0, t1) == CROP else ""
    if (t1 - t0) > 1.5:
        note = "  <- 4x longer than the headline window; NOT a matched comparison"
    print(f"  {t0:.1f}-{t1:.1f} s: {acc:.1%}{note}")
print("The impressive frontopolar number came from the whole-epoch window, while the")
print("headline came from a 1-second crop. Comparing those two is comparing two")
print("different experiments. Matched, the ablation is much less dramatic.")

# --- 5. but the confound is still real: look at the raw amplitude -------------
# CSP builds log-VARIANCE features, which are close to blind to a steady DC offset.
# A lateralised gaze deviation is exactly a steady offset, so CSP is the wrong
# instrument for finding it, and section 4's near-chance frontopolar row is NOT
# evidence that the eyes are quiet. Mean amplitude is the right instrument.
print("\n--- The confound is real anyway: frontopolar MEAN AMPLITUDE, not variance ---")
print("CSP features are log-variance, which is nearly blind to a steady DC shift --")
print("and a sustained gaze deviation IS a steady DC shift. So section 4 undertests")
print("the confound. Swapping the feature, not the channels, is what finds it.")

amp_clf = Pipeline([("scale", StandardScaler()),
                    ("LDA", LinearDiscriminantAnalysis())])
for t0, t1 in [(0.0, 1.0), CROP]:
    Xa, ya = features(epochs_low, picks=FRONTOPOLAR, crop=(t0, t1))
    Xa = Xa.mean(axis=2)  # per-channel mean voltage over the window
    amp_scores = cross_val_score(amp_clf, Xa, ya, cv=cv_split(), error_score="raise")
    _, _, amp_p = permutation_test_score(
        amp_clf, Xa, ya, scoring="accuracy", cv=cv_split(),
        n_permutations=N_PERMUTATIONS, random_state=SEED, n_jobs=-1,
    )
    tag = "cue period" if (t0, t1) == (0.0, 1.0) else "matched to headline"
    print(f"  {t0:.1f}-{t1:.1f} s ({tag:19s}): {amp_scores.mean():.1%} "
          f"(+/- {amp_scores.std():.1%}), p {p_as_bound(amp_p)}")

# State the effect in microvolts, not as an accuracy. The summary statistic has to
# be LEFT-minus-RIGHT: a gaze deviation is antisymmetric across the midline, so
# averaging all eight channels cancels it (that average gives t = 0.23, p = 0.82 --
# a null produced entirely by choosing the wrong statistic).
LEFT_FP = ["Fp1", "AF7", "AF3"]
RIGHT_FP = ["Fp2", "AF8", "AF4"]
print(f"\nFrontopolar asymmetry, mean({'+'.join(LEFT_FP)}) - "
      f"mean({'+'.join(RIGHT_FP)}), in microvolts:")
for t0, t1 in [(0.0, 1.0), CROP, (0.0, 4.0)]:
    e = epochs_low.copy().pick(FRONTOPOLAR).crop(tmin=t0, tmax=t1)
    y = e.events[:, -1]
    volts = e.get_data(copy=False).mean(axis=2) * 1e6
    li = [e.ch_names.index(c) for c in LEFT_FP]
    ri = [e.ch_names.index(c) for c in RIGHT_FP]
    asym = volts[:, li].mean(axis=1) - volts[:, ri].mean(axis=1)
    t_stat, t_p = stats.ttest_ind(asym[y == 2], asym[y == 3], equal_var=False)
    print(f"  {t0:.1f}-{t1:.1f} s: left cues {asym[y == 2].mean():+6.2f} uV, "
          f"right cues {asym[y == 3].mean():+6.2f} uV  "
          f"(Welch t = {t_stat:+.2f}, p = {t_p:.2g})")

print("\nThe sign flips with the cue side and the effect is largest in the CUE window,")
print("on AF7/AF3 vs AF4/AF8 -- the electrodes nearest the eyes. That is the")
print("signature of the eyes moving to the target, not of sensorimotor cortex.")
print("Note the 1-2 s row reverses polarity: that row is not significant (p = 0.14),")
print("so it is noise, and quoting it as a real reversal would be reading a coin flip.")
print("EEGMMIDB has no EOG channels and this pipeline has no ICA, so this bounds the")
print("confound; it does not remove it. n = 1 subject -- this is a flag, not a rate.")

# --- 6. the honesty check on the spatial patterns ----------------------------
# Left and right hand imagery differ by HEMISPHERE. So the CSP patterns must come
# out laterally asymmetric, weighted over C3 vs C4, rather than the central-vs-
# lateral pattern that fists-vs-feet produced. If they are not lateralised,
# either the labels are wrong or the model is riding an artifact. This is a weak
# check: sections 4-5 above are the real ones, because a topography read by eye
# cannot tell a motor pattern from an ocular one.
csp_full = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp_full.fit_transform(data, labels)
fig = csp_full.plot_patterns(epochs.info, components=range(4), ch_type="eeg", show=False)
fig.savefig("csp_patterns_lr.png", dpi=120, bbox_inches="tight")

print("\nSaved csp_patterns_lr.png")
print("Sanity check: these patterns should be LEFT/RIGHT asymmetric (C3 vs C4),")
print("not the central-vs-lateral pattern in csp_patterns.png. Compare them --")
print("but treat the ablation above, not the picture, as the evidence.")
