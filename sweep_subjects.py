"""Run the whole pipeline on all 109 subjects, not just the lucky one.

This is the rung that turns a demo into a result. A single subject's accuracy
says almost nothing about the METHOD -- subject 1 is a clean recording, and
per-subject quality in this dataset varies enormously.

So we run the identical pipeline across every subject and look at the whole
distribution: median, spread, and how many people land below their own chance
line.

READ THAT LAST COUNT CAREFULLY, BECAUSE AN EARLIER VERSION OF THIS SCRIPT GOT IT
BACKWARDS. It called the below-chance fraction a "BCI illiteracy rate" and noted
it fell inside the literature's familiar 15-30%, which made it feel like
replication. The inference was inverted. This pipeline's own permutation null is
50.7% +/- 8.5%, so under a GLOBAL NULL in which nobody has any signal you would
expect roughly 55% of subjects to land at or below their own chance line. The
observed figure is 28%. Seeing half the noise-only rate is evidence of signal
ACROSS THE POPULATION, not a measure of failure. The pure-noise expectation is
now printed directly beneath the counts so the number cannot be misread that way
a second time.

The literature comparison was wrong too: 15-30% describes users who cannot
achieve control AFTER training with online feedback. These are naive,
single-session, offline subjects. By that literature's own operational criterion
(~70% for usable binary control), this sweep says 65% fall short, not 27%.

Two rules this script follows:
  1. Chance is computed PER SUBJECT. Class balance differs between people, so
     borrowing subject 1's 53.3% to judge subject 47 would be its own small lie.
  2. Nothing is dropped silently. Any subject that fails to process is recorded
     with its reason and reported in the exclusion list.

First run downloads ~840 MB of EDF files (108 new subjects x 3 runs); MNE caches
them in ~/mne_data, so later runs are offline and fast.
"""

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import mne
from joblib import Parallel, delayed
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SUBJECTS = range(1, 110)
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
CROP = (1.0, 2.0)
N_SPLITS = 5
EXPECTED_SFREQ = 160.0


def evaluate_subject(subject):
    """Return one subject's row. Never raises -- failures come back as data."""
    mne.set_log_level("ERROR")  # workers get a fresh process, so set it again
    try:
        paths = eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        raw = mne.concatenate_raws(
            [mne.io.read_raw_edf(p, preload=True) for p in paths]
        )
        sfreq = float(raw.info["sfreq"])

        eegbci.standardize(raw)
        raw.set_montage("standard_1005")
        raw.set_eeg_reference("average", projection=False)
        raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

        events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
        epochs = mne.Epochs(
            raw, events, dict(hands=2, feet=3),
            tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
        )
        y = epochs.events[:, -1]
        X = epochs.copy().crop(tmin=CROP[0], tmax=CROP[1]).get_data(copy=False)

        n_hands, n_feet = int((y == 2).sum()), int((y == 3).sum())
        if min(n_hands, n_feet) < N_SPLITS:
            return dict(subject=subject, status=f"too few trials in a class "
                                                f"({n_hands}/{n_feet})",
                        accuracy=np.nan, chance=np.nan, n_trials=len(y), sfreq=sfreq)

        clf = Pipeline([
            ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ("LDA", LinearDiscriminantAnalysis()),
        ])
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv)

        return dict(
            subject=subject,
            status="ok" if sfreq == EXPECTED_SFREQ else f"ok (sfreq={sfreq:g} Hz)",
            accuracy=float(scores.mean()),
            chance=max(n_hands, n_feet) / len(y),
            n_trials=len(y),
            sfreq=sfreq,
        )
    except Exception as exc:  # noqa: BLE001 -- we want every failure as a row
        return dict(subject=subject, status=f"FAILED: {type(exc).__name__}: {exc}",
                    accuracy=np.nan, chance=np.nan, n_trials=0, sfreq=np.nan)


# --- phase 1: fetch before computing, so the CPU workers never race the cache -
# Threaded rather than sequential: this is network-bound, and each subject writes
# to its own S0NN/ directory, so there is nothing to collide over. Downloads are
# hash-verified, so an interrupted run just re-fetches the incomplete file.
def fetch(subject):
    try:
        eegbci.load_data(subjects=subject, runs=RUNS, update_path=True)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"  S{subject:03d} fetch failed: {type(exc).__name__}: {exc}"


n_subjects = len(list(SUBJECTS))
print(f"Fetching data for {n_subjects} subjects (~840 MB on first run, then cached)...",
      flush=True)
problems = Parallel(n_jobs=8, prefer="threads", verbose=0)(
    delayed(fetch)(s) for s in SUBJECTS
)
for msg in filter(None, problems):
    print(msg, flush=True)
print("Fetch complete.\n", flush=True)

# --- phase 2: evaluate in parallel ------------------------------------------
print("Evaluating (CSP+LDA, stratified 5-fold, per-subject chance)...")
rows = Parallel(n_jobs=-1, verbose=0)(
    delayed(evaluate_subject)(s) for s in SUBJECTS
)
rows.sort(key=lambda r: r["subject"])

# --- report ------------------------------------------------------------------
ok = [r for r in rows if not np.isnan(r["accuracy"])]
bad = [r for r in rows if np.isnan(r["accuracy"])]
acc = np.array([r["accuracy"] for r in ok])
chance = np.array([r["chance"] for r in ok])

# TIES ARE REAL AND MUST BE NAMED. `acc > chance` looks obvious and is a trap:
# accuracy is mean(k_i/9) and chance is m/45, and for several subjects those are
# mathematically EQUAL. Whether the float mean lands one ULP (1e-16) above or
# below then decides which bucket the subject falls in, so the headline count
# changes with fold ordering. Report three buckets with an explicit tolerance.
TOL = 1e-9
beat = acc > chance + TOL
tied = np.abs(acc - chance) <= TOL
below = acc < chance - TOL

print(f"\n{'=' * 62}")
print(f"{len(ok)} of {len(rows)} subjects evaluated successfully")
print(f"{'=' * 62}")
print(f"mean       {acc.mean():.1%}")
print(f"median     {np.median(acc):.1%}")
print(f"IQR        {np.percentile(acc, 25):.1%} to {np.percentile(acc, 75):.1%}")
print(f"min / max  {acc.min():.1%} (S{ok[int(acc.argmin())]['subject']:03d}) "
      f"/ {acc.max():.1%} (S{ok[int(acc.argmax())]['subject']:03d})")
print(f"mean per-subject chance  {chance.mean():.1%}")

print(f"\nAbove their own chance : {beat.sum()}/{len(ok)}")
print(f"Exactly AT chance      : {tied.sum()}/{len(ok)}  (mathematical ties, not noise)")
print(f"Below their own chance : {below.sum()}/{len(ok)}")
for thresh in (0.60, 0.70, 0.80, 0.90):
    print(f"  above {thresh:.0%}: {(acc > thresh).sum():3d}/{len(ok)}")

# --- what would PURE NOISE produce? The counts above are meaningless without it.
# Our own permutation test (evaluate_honestly.py) measured the null at
# 50.7% +/- 8.5% for this pipeline at n=45. Under a GLOBAL null where nobody has
# any signal, a subject still falls at-or-below its own chance line with
# probability Phi((chance - 50.7%) / 8.5%).
NULL_MEAN, NULL_SD = 0.507, 0.085
from math import erf, sqrt  # noqa: E402

phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))  # noqa: E731
p_below_null = np.array([phi((c - NULL_MEAN) / NULL_SD) for c in chance])
print(f"\nUnder a pure-noise null ({NULL_MEAN:.1%} +/- {NULL_SD:.1%}), you would "
      f"expect\n  {p_below_null.mean():.0%} ({p_below_null.sum():.0f}/{len(ok)}) "
      f"at-or-below their own chance line.")
print(f"Observed: {(~beat).sum()}/{len(ok)} ({(~beat).mean():.0%}).")
if (~beat).mean() < p_below_null.mean():
    print("  -> FEWER below chance than noise alone predicts. That is evidence of")
    print("     signal across the population, NOT a measure of BCI illiteracy.")
    print("     Calling this figure an illiteracy rate would be backwards.")

sub1 = next((r for r in ok if r["subject"] == 1), None)
if sub1:
    pct = int(round(100 * (acc < sub1["accuracy"]).mean()))
    # 91st, not "91th". 11/12/13 always take "th"; otherwise 1/2/3 -> st/nd/rd.
    suffix = ("th" if 11 <= pct % 100 <= 13
              else {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th"))
    print(f"\nSubject 1 (the original headline): {sub1['accuracy']:.1%}, "
          f"the {pct}{suffix} percentile of subjects.")

if bad:
    print(f"\nExcluded ({len(bad)}), with reasons:")
    for r in bad:
        print(f"  S{r['subject']:03d}: {r['status']}")
else:
    print("\nNo subjects excluded.")

odd = [r for r in ok if r["sfreq"] != EXPECTED_SFREQ]
if odd:
    print(f"\nNon-standard sampling rate ({len(odd)}):")
    for r in odd:
        print(f"  S{r['subject']:03d}: {r['sfreq']:g} Hz")

# --- csv + figure -------------------------------------------------------------
with open("sweep_results.csv", "w") as fh:
    fh.write("subject,accuracy,chance,n_trials,sfreq,status\n")
    for r in rows:
        a = "" if np.isnan(r["accuracy"]) else f"{r['accuracy']:.4f}"
        c = "" if np.isnan(r["chance"]) else f"{r['chance']:.4f}"
        sf = "" if np.isnan(r["sfreq"]) else f"{r['sfreq']:g}"
        fh.write(f"{r['subject']},{a},{c},{r['n_trials']},{sf},\"{r['status']}\"\n")

order = np.argsort(acc)
fig, ax = plt.subplots(figsize=(11, 5))
colors = ["#c0392b" if not b else "#4a6fa5" for b in beat[order]]
ax.bar(range(len(acc)), acc[order], color=colors, width=0.9)
ax.plot(range(len(acc)), chance[order], color="#2c3e50", ls="--", lw=1.2,
        label="that subject's chance")
ax.axhline(np.median(acc), color="#e67e22", lw=1.5,
           label=f"median {np.median(acc):.1%}")
ax.set_xlabel(f"subject (sorted by accuracy, n={len(acc)})")
ax.set_ylabel("accuracy (stratified 5-fold)")
ax.set_title("CSP+LDA across all subjects: the method, not one recording")
ax.set_ylim(0, 1)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("subject_distribution.png", dpi=120)

print("\nSaved sweep_results.csv and subject_distribution.png")
