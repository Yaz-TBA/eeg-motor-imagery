# EEG Motor-Imagery Decoding — The Complete Explainer

A deep, self-contained walkthrough of this repository: what it does, the science and math
behind every step, why each decision was made, how to run it, and how to take it further.
By the end you should be able to defend every line to a mentor and know exactly which knobs
to turn next.

---

## 1. The one-paragraph summary

When you *imagine* moving a body part — without actually moving it — the part of your brain
that controls that body part changes its electrical rhythm. This project records that
electrical activity from the scalp (EEG), isolates the relevant rhythms, and trains a simple
machine-learning model to guess **which** movement a person imagined: *both fists* or *both
feet*. On one clean subject it reaches **91% accuracy** (chance is 53%, permutation p <= 0.001),
and it establishes that the signal is motor by **ablation**: restricted to 17 sensorimotor channels
the decoder holds at 95.6%, restricted to 8 frontopolar channels it falls to 51.1% — *below* the
53.3% majority-class floor. This is the canonical "hello world" of brain-computer interfaces
(BCIs), done honestly and reproducibly.

---

## 2. Why this problem matters (the motivation)

A **brain–computer interface** turns brain activity into commands — moving a cursor, a
prosthetic, a wheelchair — *without* muscles. The dream user is someone with paralysis or ALS
who can still *think* about moving but can't send the signal to their limbs. **Motor imagery**
(imagining movement) produces a brain signature very similar to real movement, so if a
computer can read that signature, it can act on the person's intent.

The core scientific bet this project rests on:

> Imagining a movement activates roughly the same sensorimotor cortex as performing it, and
> that activation is **spatially specific** — imagining your hands lights up a different patch
> of cortex than imagining your feet. EEG can pick up the difference from the scalp.

This project is the smallest honest demonstration that the bet pays off.

---

## 3. The neuroscience you need (and why each fact drives the code)

You don't need a neuro degree, but four facts explain *every* preprocessing choice in the code.

### 3.1 The motor homunculus — *why hands-vs-feet is separable*
The strip of cortex that controls movement (the primary motor cortex, plus the sensory strip
behind it) is laid out like a distorted map of the body — the "homunculus." Crucially:
- **Hands/fists** map to a region on the **side** of this strip (lateral), roughly under EEG
  electrodes C3 (right hand) and C4 (left hand).
- **Feet** map to the **top-center**, down in the midline crevice between the hemispheres,
  under electrode Cz.

Because hands and feet sit in physically different places, their scalp signatures *differ in
where on the head they appear*. That spatial difference is exactly what the model exploits.
**This is why the README calls it an "easy contrast"** — the two classes are far apart on the
homunculus, so they're easy to tell apart. Left-hand-vs-right-hand would be much harder because
both live in the same lateral strip, just mirrored across hemispheres.

### 3.2 Mu and beta rhythms — *why we band-pass to 8–30 Hz*
When a body region is **idle**, its patch of motor cortex idles in a synchronized oscillation:
- **Mu rhythm:** ~8–12 Hz
- **Beta rhythm:** ~13–30 Hz

### 3.3 Event-Related Desynchronization (ERD) — *why imagined movement is detectable at all*
When you *engage* (or imagine engaging) that body part, the local neurons stop firing in
lockstep. The synchronized rhythm **breaks down**, so the power in the mu/beta band **drops**
over the active region. This drop is called **Event-Related Desynchronization (ERD)**.

So "imagine your fists" → mu/beta power drops over the *lateral* electrodes (C3/C4).
"Imagine your feet" → mu/beta power drops over the *central* electrode (Cz).

The model's entire job reduces to: **find where in the 8–30 Hz band the power dropped, and
map that location to a class.** Everything upstream in the pipeline exists to make this
signal cleaner and easier to read.

### 3.4 EEG is a spatial mixture — *why we need CSP*
Each scalp electrode doesn't see one brain source; it sees a blurry sum of *all* sources
(the skull smears everything). So the "power dropped over C3" story is never clean at a single
electrode. You need a method that **re-combines all 64 electrodes** into a few virtual channels
that maximize the class difference. That method is **CSP** (Section 8). Hold that thought.

---

## 4. The dataset: PhysioNet EEGBCI

- **Source:** the [EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/)
  on PhysioNet, loaded automatically through MNE's `mne.datasets.eegbci` helper.
- **Size:** 109 subjects, **64-channel** EEG, sampled at **160 Hz** (160 samples per second per channel).
- **Format:** EDF files (European Data Format — the standard container for clinical
  physiological recordings). One file per subject per run.
- **Structure:** each subject did 14 runs, and different runs are different *tasks*. The
  distinction that matters most is **executed** versus **imagined** movement, because the two
  sit next to each other in the numbering and are trivially easy to confuse:

| Task | Runs | What the subject did |
|---|---|---|
| baseline | 1, 2 | eyes open, eyes closed |
| 1 | 3, 7, 11 | **executed** left vs. right fist |
| 2 | **4, 8, 12** | **imagined** left vs. right fist |
| 3 | 5, 9, 13 | **executed** both fists vs. both feet |
| 4 | **6, 10, 14** | **imagined** both fists vs. both feet |

**This project uses runs 6/10/14** — Task 4, imagined fists vs. feet. Using all three gives more
trials than one run alone. The harder left/right contrast in rung 7 uses **4/8/12**.

> An earlier version of this document twice told the reader to use **3/7/11** for the harder
> left/right contrast. Those are *executed* movement. Building an imagery result on them and
> writing it up as imagery is a silent over-claim, and it is precisely the kind a reviewer
> catches in one minute. The code always used 4/8/12; only this document was wrong.

### 4.1 Annotations: how the data knows what the subject was doing
Inside each EDF file are **annotations** — timestamped markers the experimenters recorded when
they showed the subject a cue. Three labels appear:
- **T0** = rest (do nothing). *We drop these.*
- **T1** = the cue to imagine **fists** (in these runs). We call this class **"hands"**.
- **T2** = the cue to imagine **feet**. Class **"feet"**.

These annotations are the *ground-truth labels*. Without them we'd have EEG but no idea what
the person was told to imagine, and supervised learning would be impossible.

> ⚠️ Gotcha worth knowing for a mentor conversation: T1/T2 mean *different things in different
> runs*. In the fist runs — 3/7/11 executed, 4/8/12 imagined — T1 = left fist and T2 = right
> fist. In *these* runs (6/10/14) T1 = both fists and T2 = both feet. The code hard-codes the
> runs so this mapping is correct, but if you ever swap runs the labels silently change meaning
> and **nothing raises an error**. This is a classic EEGBCI footgun, and this project tripped
> over the neighbouring version of it (see the run table above).

---

## 5. The software stack (`requirements.txt`)

The pinned versions matter for reproducibility, but conceptually there are only four libraries
doing real work:

| Library | Role in this project |
|---|---|
| **MNE** (`mne==1.12.1`) | The EEG/MEG workhorse. Downloads the data, reads EDF, holds the signal in `Raw`/`Epochs` objects, does filtering, referencing, epoching, and even ships the CSP implementation and the scalp-map plotting. **~90% of the domain logic is MNE.** |
| **scikit-learn** (`scikit-learn==1.9.0`) | The generic ML layer: `LinearDiscriminantAnalysis` (the classifier), `Pipeline` (chain CSP→LDA), and `cross_val_score` / `StratifiedKFold` / `LeaveOneGroupOut` / `permutation_test_score` (honest evaluation). |
| **NumPy** (`numpy==2.5.1`) | Array math under everything; used directly for the chance-level calculation, the seed sweeps, and the effect decompositions. |
| **matplotlib** (`matplotlib==3.11.0`) | Renders every PNG in the repo. |

Three more libraries arrive with the later rungs and are only needed for those:

| Library | Used by |
|---|---|
| **pyriemann** (`pyriemann==0.12`) | `riemannian.py` — covariance classification on the SPD manifold. |
| **PyTorch** (`torch==2.13.0`) | `eegnet_compare.py`, `regime_decomposition.py` — the CNN. Runs on Apple GPU via the `mps` backend when available. |
| **braindecode** (`braindecode==1.6.1`, with `skorch` and `einops`) | The EEGNet implementation and its scikit-learn-compatible `EEGClassifier` wrapper, which is what lets a CNN drop into `cross_val_score` alongside CSP+LDA. |

`joblib` is used directly too — `Parallel` fans the per-subject data loading across cores in the
sweep and the cross-subject rungs. The rest of `requirements.txt` (certifi, scipy, pooch, tqdm,
pillow, …) is transitive. `pooch` is worth a mention: it is the downloader MNE uses to fetch and
cache the dataset.

`.gitignore` keeps the virtual environment (`.venv/`) and Python bytecode caches out of git —
standard hygiene so the repo stays just source + results.

---

## 6. The architecture: a ladder of eleven rungs

The defining design choice of this repo is that it's built **rung by rung**. Each script is a
complete, runnable checkpoint that adds exactly one new idea on top of the previous one. The
git history mirrors this — one commit per rung. This is deliberate and worth articulating to a
mentor: *it makes the pipeline debuggable and teachable, because every stage can be run and
inspected in isolation before the next stage is added.*

```
BUILD IT
 1  load_and_plot.py          Can I load the data and see a signal?
 2  epoch_trials.py           Can I cut it into labeled trials?
 3  filter_and_epoch.py       Can I isolate the motor rhythms first?
 4  decode_csp.py             Can I classify it?

ATTACK IT
 5  evaluate_honestly.py      Is the number real, or an artifact of how I measured it?
 6  sweep_subjects.py         Does it hold across 109 people, or just the lucky one?
 7  harder_contrast.py        What happens on a genuinely harder contrast?
 8  cross_subject.py          Does it transfer to a person the model has never seen?
 9  riemannian.py             Does a stronger classical method beat it?
10  eegnet_compare.py         Does a CNN beat it, and at what sample size?
11  regime_decomposition.py   What did rung 10's third experiment actually measure?

GUARD IT  (controls on the repo, not rungs of the ladder)
    ablate_channels.py       Produces the artifact-ablation table, and asserts the k/45 lattice
    check_provenance.py      Fails if any number in README/EXPLAINER is printed by no script
```

`decode_csp.py` is the *only* script you need to run to reproduce the headline result; it
re-does the work of rungs 1–3 internally. Every file is standalone — nothing imports anything
else — so any rung can be run and read on its own.

**The shape of that list is the point.** Four rungs build the result and seven try to break it.
Three of them succeeded: rung 7 found a gaze confound in this project's own data, rung 10 was
measuring a network that turned out to be dead, and rung 6's headline inference was backwards.
A project that only climbs is a demo. The rungs that found something wrong are the ones worth
talking about, and §12 reports what each one actually returned.

The next sections walk each rung in depth.

---

## 7. Rung by rung

### Rung 1 — `load_and_plot.py`: get the data, look at it

**Goal:** prove the data loads and eyeball the raw signal before doing anything clever.

What it does, step by step:
1. `matplotlib.use("Agg")` **before** importing pyplot. "Agg" is a non-interactive backend
   that renders straight to a file. This is what lets the script save a PNG on a headless
   machine (a server, CI) with no display attached. Order matters — you must set the backend
   before pyplot initializes.
2. `eegbci.load_data(subjects=1, runs=[6], update_path=True)` downloads run 6 for subject 1
   (or reads it from the `~/mne_data` cache on later runs) and returns the local EDF path.
3. `mne.io.read_raw_edf(path, preload=True)` loads it into a **`Raw`** object — MNE's container
   for a continuous recording (channels × time). `preload=True` pulls the samples into RAM now
   rather than lazily, which is required for the operations that follow.
4. **`eegbci.standardize(raw)`** — EEGBCI stores channel names with trailing dots, like `"Fc5."`.
   This renames them to the standard form (`"FC5"`) so MNE can match them to electrode positions.
5. **`raw.set_montage("standard_1005")`** — attaches real 3-D scalp coordinates to each channel
   using the standard 10-05 electrode layout. Without this, MNE knows the *numbers* but not
   *where on the head* each electrode sits — and you couldn't draw a scalp map later.
6. Prints metadata (sampling rate, duration, channel count) and saves the first 5 seconds of
   the first 10 channels to **`raw_eeg.png`**.

**Why it exists:** sanity. If the download, channel naming, or montage is broken, you find out
here — before you've built a classifier on top of a silent bug.

### Rung 2 — `epoch_trials.py`: cut the stream into labeled trials

**Goal:** turn one long continuous recording into a stack of short, labeled **trials** (called
**epochs** in EEG-speak), one per cue.

New ideas introduced:
1. **Concatenation.** Loads all three runs (6/10/14) and `mne.concatenate_raws(...)` stitches
   them end-to-end into one continuous `Raw`. More runs → more trials → a more trustworthy
   accuracy estimate.
2. **Events from annotations.** `mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))`
   converts the text annotations into an **events array** — a table of `[sample_index, 0, class_id]`
   rows. Here T1→2 and T2→3. **T0 (rest) isn't in the mapping, so it's dropped entirely.** The
   integers 2 and 3 are arbitrary class IDs; the code then names them: `event_id = dict(hands=2, feet=3)`.
3. **Epoching.** `mne.Epochs(...)` cuts a fixed window around every event:
   - `tmin=-1.0, tmax=4.0` → each trial spans from **1 second before** the cue to **4 seconds
     after** it (a 5-second window). The pre-cue second is a baseline/reference period; the
     post-cue seconds contain the actual imagined movement.
   - `baseline=None` → no baseline correction is applied here (kept deliberately simple).
   - `picks="eeg"` → keep only EEG channels.
   - `preload=True` → materialize into memory.

**Result it prints:** 21 hands trials, 24 feet trials (45 total). Each trial is 64 channels ×
801 samples (5 seconds × 160 Hz + 1). This 21/24 split is the number to remember — it reappears
as the *chance level* later (a dumb model that always guesses "feet" would be right 24/45 = 53%).

**Why it exists:** classifiers need labeled examples, not a continuous stream. This is where raw
signal becomes a supervised-learning dataset.

### Rung 3 — `filter_and_epoch.py`: isolate the motor rhythms

**Goal:** clean the signal so the mu/beta ERD story is what's left, *before* epoching.

The two new preprocessing steps (added between load and epoch):

1. **Average reference** — `raw.set_eeg_reference("average", projection=False)`.
   EEG voltages are always *relative* to some reference point; the raw recording's reference is
   somewhat arbitrary. Re-referencing every channel to the **average of all channels** gives a
   neutral, spatially balanced baseline. **CSP assumes this** — it reasons about how variance
   is distributed *across* channels, and that logic is cleanest when no single channel is the
   privileged reference. `projection=False` applies the reference directly to the data rather
   than storing it as a lazy projection.

2. **Band-pass filter 8–30 Hz** — `raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")`.
   This throws away everything *outside* 8–30 Hz. Why that band? Because (Section 3.2) that's
   exactly where the mu (8–12) and beta (13–30) motor rhythms live. Below 8 Hz you get slow
   drifts and eye movements; above 30 Hz you get muscle artifacts and line noise. Keeping only
   8–30 Hz means the model sees mostly *motor* signal.

**The most important subtlety in the whole repo** — *why filter the continuous signal, not the
epochs?* Digital filters produce garbage at the very start and end of whatever they're applied
to ("edge artifacts" / filter ringing). If you filtered each short 5-second epoch, those
artifacts would land *inside every trial*. By filtering the long continuous recording first,
the artifacts are confined to the very beginning and end of the whole recording — far from any
trial. `skip_by_annotation="edge"` additionally avoids filtering across the seams where the
three runs were concatenated. **This is a genuinely load-bearing decision, and a great thing to
be able to explain to a mentor** because it separates people who understand DSP from people who
copy pipelines.

`fir_design="firwin"` just specifies a well-behaved, linear-phase FIR filter (doesn't distort
the timing of the rhythms).

**Sanity check built in:** it re-prints the trial counts and asserts (in a comment) they should
still be 21/24 — because *filtering changes the signal values, not the number of cues.* If the
count changed, something upstream broke.

### Rung 4 — `decode_csp.py`: classify it, and prove it's real

This is the payoff. It repeats rungs 1–3, then adds the actual decoding. Two conceptual halves:

**(a) Feature extraction — crop to the imagery window.**
```python
labels = epochs.events[:, -1]                                   # 2=hands, 3=feet
train_data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data()  # 1–2 s after cue
```
Of the 5-second epoch, only the slice from **1 to 2 seconds after the cue** is used for
features. Why? The ERD (the power drop) takes a beat to develop after the cue and is most
stable a second or so in; the pre-cue and immediate-post-cue periods are noisier. Cropping to
a clean 1-second imagery window sharpens the class difference. (This is a tunable knob — see §11.)

`get_data()` produces a plain NumPy array of shape `(45 trials, 64 channels, 160 samples)`.

**(b) The classifier: CSP → LDA, cross-validated.**
```python
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
clf = Pipeline([("CSP", csp), ("LDA", LinearDiscriminantAnalysis())])

# Stratified k-fold, not ShuffleSplit: it tests every trial exactly once and
# keeps class balance steady across folds.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, train_data, labels, cv=cv)
```
CSP and LDA are explained in full in Sections 8 and 9; cross-validation in Section 10.

**(c) The significance test.** Shuffle the labels a thousand times, re-run the whole pipeline
each time, and ask how often chance alone matches the real result:
```python
observed, null_scores, p_value = permutation_test_score(
    clf, train_data, labels, scoring="accuracy", cv=cv,
    n_permutations=1000, random_state=42, n_jobs=-1,
)
```
This is the difference between "91% sounds high" and "91% is outside what this pipeline
produces on data with no signal in it." §10.3 covers what the resulting p-value can and cannot
say.

Then it prints the scoreboard against the majority-class baseline:
```python
chance = max(np.mean(labels == 2), np.mean(labels == 3))   # majority-class baseline
```
and finally **visualizes what CSP learned** by fitting it on all trials and drawing the top 4
spatial patterns as scalp maps → **`csp_patterns.png`**. That plot is *interesting, and it is
not the credibility check* — §8.3 explains why, and it is the single most important correction
in this document.

### Rung 5 — `evaluate_honestly.py`: is the number real, or an artifact of how I measured it?

**Goal:** attack the headline before anyone else does.

Rung 4 originally reported **94.4% ± 5.6%** from `ShuffleSplit(n_splits=10, test_size=0.2)`.
Three things were wrong with that, and this rung finds all three.

1. **The ± was quantization, not spread.** A 20% test set of 45 trials is **9 trials**, so a
   fold's accuracy can only be a multiple of 1/9 ≈ 11.1%. The ten folds landed on just **two
   distinct values**. "± 5.6%" is the gap between two rungs of a ladder, not a standard
   deviation over a distribution.
2. **ShuffleSplit is not a partition.** It resamples independently per split, so some trials are
   **never tested at all** while others are tested several times, and class balance swings fold
   to fold. It is also not stratified.
3. **A fold standard deviation is not a confidence interval**, and reading it as one implies a
   precision that 45 trials cannot support.

Switching to `StratifiedKFold(n_splits=5, shuffle=True)` — every trial tested exactly once,
class balance held steady — gives the number this repo now publishes: **91.1%**, with a
1000-shuffle permutation test at **p ≤ 0.001**.

**The uncomfortable part, which is the actually interesting finding.** It would be tidy to say
the estimator change corrected an inflated number. It did not. Sweeping **100 cross-validation
seeds** through both estimators:

| estimator | mean | range | where seed 42 falls |
|---|---|---|---|
| ShuffleSplit (retracted) | 93.6% | 87.8–98.9% | 49th percentile |
| **StratifiedKFold (published)** | **93.8%** | 88.9–97.8% | **3rd percentile** |

The two estimators **agree in expectation to about 0.2 points**. So the 94.4 → 91.1 drop is
roughly **2.7 points of seed luck and 0.6 points of real estimator change**. The switch is still
correct, for coverage and stratification reasons, but presenting it as an integrity correction
would be its own small dishonesty. The published 91.1% is a **conservative draw** from an
88.9–97.8% distribution, and that is how it should be described out loud.

> **One caveat on the percentile column, which is this rung demonstrating its own thesis.** Those
> ranks count seeds *strictly below* seed 42. Because both estimators are quantized — the
> stratified one to 1/45 — the 100 seeds land on only a handful of distinct values and many tie
> exactly on the headline. Strictly-below is the most flattering of the tie conventions available,
> and counting ties as at-or-below moves the published estimator's "3rd" materially higher (and
> the retracted estimator's "49th" by more). Nothing substantive turns on it: the 2.7-point gap
> between 91.1% and the 93.8% mean is a comparison of means, which ties do not touch. But a rung
> whose entire thesis is *"quantization makes fold-level statistics misleading"* reintroducing
> quantization into its own headline rank is worth naming rather than leaving for a reader to find.

There is a methodological trap here worth naming, because this project fell into it: the
"seed 42 is not cherry-picked" credential was originally computed for **ShuffleSplit** and then
silently carried onto the **StratifiedKFold** number. Diagnostics do not transfer across
estimators. The script now sweeps both so neither one's verdict can be attached to the other's
number.

### Rung 6 — `sweep_subjects.py`: does it hold across 109 people?

**Goal:** turn a claim about *this subject* into a claim about *the method*.

The identical pipeline runs on all 109 subjects, computing **chance per subject** — class
balance differs between people, so borrowing subject 1's 53.3% to judge subject 47 would be its
own small lie.

| | |
|---|---|
| median | **60.0%** |
| IQR | 52.8–75.6% |
| *numerically* above their own chance line | 79 / 109 |
| exactly at chance | 6 / 109 |
| below their own chance | 24 / 109 |

> **Read that 79 as a descriptive bucket, not as "79 people have decodable signal."** The script
> compares each subject's accuracy to their own majority-class rate with a bare inequality; it runs
> no per-subject significance test, and roughly half of the 79 sit within noise of their own chance
> line. Nothing in this rung's conclusion rests on the count — the conclusion below is a
> *population*-level comparison against the pure-noise expectation, which is unaffected. The row is
> relabelled rather than removed so the gap between "counted" and "tested" is visible.

**Subject 1's 91.1% is the 91st percentile.** Say that out loud before anyone has to ask.

**The inference this rung originally drew was backwards, which is worth more than the numbers.**
The first write-up read "27% of subjects at or below chance" as a **BCI illiteracy rate**, and
the coincidence that 27% sits inside the literature's familiar 15–30% band made it feel like
replication. Both halves were wrong:

- **The direction.** This pipeline's own permutation null is 50.7% ± 8.5%. Under a **global null
  in which nobody has any signal**, the expected fraction landing at or below their own chance
  line is **~55% (59/109)**. Observed: **30/109 (28%)**. Seeing *half* the noise-only rate is
  **evidence of signal across the population**, not a measure of failure.
- **The comparison.** The literature's 15–30% describes users who cannot achieve control *after
  training with online feedback*. These are naive, single-session, offline subjects. By the
  literature's own operational criterion (~70% for usable binary control), this sweep says
  **65% fall short, not 27%**.

The script now prints the pure-noise expectation directly beneath the counts, so the number
cannot be misread that way twice.

**A second trap it caught.** The original bucketing used `beat = acc > chance`, and for **six
subjects** those two quantities are *mathematically equal* (accuracy is `mean(k/9)`, chance is
`m/45`). Whether the float landed one ULP above or below decided the bucket, so the headline
count moved with fold ordering. Three explicit buckets with a tolerance replaced it, which is
why the table above reports ties as their own row.

### Rung 7 — `harder_contrast.py`: how I found a gaze confound in my own result

**Goal as stated:** measure what the method costs when the classes move closer together. Left
fist versus right fist share the same sensorimotor strip, mirrored, instead of sitting
centimetres apart the way fists and feet do.

**Runs 4/8/12** — imagined left vs. right fist. Not 3/7/11, which are *executed* movement.

Subject 1 scores **73.3%** (p = 0.0020), against a majority-class floor of 51.1%. The original
write-up reported the gap to fists-vs-feet as "what a harder contrast costs: 17.8 points."
Nearly everything about that sentence was wrong.

- **It is n=1.** Rung 6 had just swept 109 subjects and this rung silently reverted to one.
- **The two conditions come from different recording runs**, so "harder contrast" cannot be
  separated from "different session."
- **The window was the joint maximum, and the script now proves it.** Sliding the 1-second crop
  across the same trials gives **66.7 / 55.6 / 73.3 (used) / 64.4 / 46.7%** for windows starting
  at 0.0 / 0.5 / 1.0 / 1.5 / 2.0 s. That is a **26.7-point range** across *overlapping* windows
  of the same data — larger than the 17.8-point gap the original sentence advertised as a
  finding. The spread is the honest error bar on this rung, and the published window is the peak
  of it.

The −17.8 is still printed, because on subject 1 it is numerically true and deleting a real
number to make a retraction tidier is its own dishonesty. It is printed with the window sweep
directly beneath it, so nobody can read it as a quantity again.

> **Two figures withdrawn from this passage for lack of provenance.** It used to justify the
> retraction with a **16-subject group mean of 57.5% (median 53.3%)**, giving "the real cost is
> about 7 points, not 17.8," and with a **95% CI of [2.4, 33.1]** on the difference. No script in
> this repo computes either, and the unexplained *n* = 16 matched no other multi-subject rung
> (8–11 all use 20). Both are removed rather than re-derived. The retraction does **not** depend
> on them: the window sweep above refutes the 17.8 on its own, from code, and more decisively.

**And then the real problem.** The PhysioNet protocol places the target on the **left or right
of the screen and leaves it there** until the subject relaxes, so a lateralised visual stimulus
is present for the entire decoding window. On subject 1 the frontopolar asymmetry
`mean(Fp1,AF7,AF3) − mean(Fp2,AF8,AF4)` is:

| window | left cues | right cues | Welch t | p |
|---|---|---|---|---|
| **0.0–1.0 s (the cue)** | **+11.89 µV** | **−12.99 µV** | **+7.71** | **3.7e-09** |
| 1.0–2.0 s (the decoding window) | −2.99 µV | +2.43 µV | −1.51 | 0.14 — n.s. |
| 0.0–4.0 s (whole epoch) | +2.83 µV | −2.58 µV | +5.10 | 7.6e-06 |

The sign flips with the cue side, the effect is largest in the **cue** window, and it is driven by
AF7/AF3 against AF4/AF8 — the electrodes nearest the eyes. That is the signature of eyes moving
to a target. (The 1–2 s row reverses polarity, but it is not significant; quoting that reversal as
real would be reading a coin flip.)

**A methodological trap worth naming, because the obvious statistic hides this effect.** Averaging
all eight frontopolar channels gives a total null. A gaze deviation is *antisymmetric* across the
midline, so pooling both hemispheres cancels it exactly. The confound is only visible in the
lateral difference.

**A second trap, and it is the reason the channel ablation undertests this.** CSP features are
log-variance, and a sustained gaze deviation is a steady DC shift — which variance is nearly blind
to. Swapping the *feature* rather than the channels is what finds it. A **frontopolar
mean-amplitude** decoder scores:

| window | accuracy | p |
|---|---|---|
| 0.0–1.0 s (cue period) | **86.7%** (± 8.3%) | **≤ 0.001** |
| 1.0–2.0 s (matched to headline) | 62.2% (± 20.6%) | 0.0939 |

> **The evidence originally offered for this confound was itself window-shopped, and that
> correction matters more than the confound.** This section used to claim that "a decoder using
> only 8 frontopolar channels at 0.5–5 Hz reaches **73.3%** — numerically identical to the
> 64-channel headline." At **matched** settings — same trials, same folds, same 1.0–2.0 s window —
> the real comparison is:
>
> | | accuracy |
> |---|---|
> | all 64 ch, 8–30 Hz (the headline) | 73.3% |
> | frontopolar 8 ch, 8–30 Hz | 62.2% |
> | all 64 ch, 0.5–5 Hz | 51.1% |
> | frontopolar 8 ch, 0.5–5 Hz | **53.3%** |
>
> Frontopolar-only is **20 points below** the headline, not level with it. The old 73.3% came from
> running the frontopolar decoder on the whole **0–4 s epoch** — four times longer than the
> headline's 1-second crop — so the "numerically identical" match was an artifact of comparing two
> different experiments. Across windows that same decoder gives 66.7 / 53.3 / 66.7 / 73.3%, and
> the published figure was its maximum.
>
> Also withdrawn from this passage: **"+4.41 µV on left cues and −3.69 µV on right cues
> (t = 5.12)"** — close to the whole-epoch row above (t = +5.10) but with different microvolt
> values and, more importantly, **no stated window**; the correct headline statistic is the cue
> window, t = +7.71. **"significant in 11 of 16 subjects and sign-consistent in 15
> (p = 0.0005)"** and **"group-wide the ocular decoder averages 53.9% against the pipeline's
> 57.5%"** — no script computes either. And **"mu alone 73.3%, beta 64.4%, combined 73.3%, so this
> is an alpha-band decoder"** — those were full-64-channel numbers presented as if they supported
> the frontopolar claim, and no script splits the band.
>
> The confound survives all of this and is now *better* evidenced than it was, on 3.7e-09 rather
> than on a maximum over unstated windows. What does not survive is the claim that gaze alone
> reproduces the headline, and the n=1 scope: this is a **flag, not a rate**.

EEGMMIDB has **no EOG channels** and this pipeline has no ICA, so the confound can be bounded but
neither removed nor monitored.

**Why this rung is kept.** As "the cost of a harder contrast" it is a bad measurement. As "I
built a rung, believed it, and then found the confound in my own data" it is the most useful
thing in the repository.

### Rung 8 — `cross_subject.py`: does it transfer to a person the model has never seen?

**Goal:** the result a deployed BCI actually needs. Everything up to here is *within*-subject —
the model trains and tests on the same brain. A real system meets a new user whose skull
thickness, cortical folding and electrode placement are all different, and it has to work anyway.

Trials from 20 subjects are pooled and evaluated **leave-one-subject-out**: train on 19, test on
the held-out person, rotate, with subject as the group in `LeaveOneGroupOut`.

The within-to-cross gap is the deliverable, and the honest form of it is a **confidence interval
rather than a point estimate**: cross-subject sits at near-parity, 95% CI **[−1.9, +11.2]
points, p = 0.181**. An 11-point drop is fully consistent with this data, and so is no drop at
all. The experiment cannot separate them.

**Retracted from the original write-up:** "the barrier is sample size, not anatomy." That came
from a single uncontrolled comparison that varied two factors at once, with no learning curve
behind it. It may well be true. This rung does not show it, and the way to show it is to hold
the subject fixed and sweep training-set size, which needs more trials per subject than
EEGMMIDB has.

**A correction about a check rather than a result.** This script asserted that no subject
appears on both sides of a `LeaveOneGroupOut` split, and reported passing it as evidence of no
leakage. That assertion is **definitionally true and can never fail** — it restates the
definition of the splitter. It now carries an honest comment about what it can and cannot catch.
A guard that cannot fail is worse than no guard, because it reads as protection in a review.

### Rung 9 — `riemannian.py`: does a stronger classical method beat it?

**Goal:** answer the failure rung 8 measured. CSP learns spatial filters tuned to the training
population's anatomy, and a new skull shifts everything. Riemannian methods attack that
directly. A trial's spatial covariance matrix is **symmetric positive definite**, SPD matrices
live on a curved manifold rather than in flat space, and treating them as flat feature vectors
distorts the distances between them. Measuring distance *along* the manifold respects the actual
geometry, and it is the current state of the art for classical BCI.

Four pipelines — MDM and Tangent Space, each on all 64 channels and on the sensorimotor subset —
run against the CSP+LDA baseline on **identical LOSO folds**.

**It lost.** The honest reading of *how* it lost is much narrower than what was first written:

| comparison | paired p |
|---|---|
| MDM-64 | **0.005** |
| MDM-motor | 0.200 |
| TS-64 | 0.349 |
| TS-motor | 0.330 |

Only MDM-64 is significant; the other three confidence intervals span zero. The minimum
detectable difference at 80% power with n=20 is roughly **6 to 8 points** depending on the
comparison, and three of the four deltas are smaller than that.

> Two caveats on that MDE. It was previously quoted as "about 5–6 points," which is optimistic —
> only one of the four comparisons is that tight, and rung 8's is nearly 10. And like the four
> paired p-values above, it is **computed in this prose and by no committed script**: `riemannian.py`
> persists nothing but a PNG and prints no test, so reproducing this table currently requires
> editing the source. That is a real provenance gap, listed in §12.2.

**Retracted:** "no method dominates, and the best method is subject-specific." Per-subject
optimality requires a **subject × method interaction**, and none is **detectable** here
(χ²₁₉ = 13.0, **p = 0.84**) — a test with low power against the effect sizes in play, so "no
interaction" and "cannot tell" are again indistinguishable, and only the second is claimed. The
9-8-3 win/loss/tie split is consistent with a *uniform* −2.6-point difference plus 45-trial noise. "No method dominates" is indistinguishable here from "this
experiment cannot tell these methods apart," and only the second is supported.

Two further caveats surfaced on review. The script selects its best pipeline by **max mean over
the same test folds it reports from**, which is selection on the test set. And framing the
comparison as "2080 parameters versus a classical baseline" ignored that
`Covariances(estimator="oas")` is a shrinkage estimator *built* for exactly the small-sample
regime, while the CSP baseline it loses to runs with `reg=None`.

**That asymmetry runs the opposite way to charity, and it is worth stating plainly.** Shrinkage
sits on the **Riemannian** side only (`riemannian.py:121/126/131/135`); the CSP baseline is
`reg=None` (`riemannian.py:117`). Regularizing only one arm favours **that** arm. So the baseline
won a comparison that was tilted against it. Shrinkage is not a nicety on the Riemannian side
either — `raw.set_eeg_reference("average")` costs one degree of freedom, which makes every 64×64
covariance rank-63 and singular by construction, so the two 64-channel Riemannian pipelines cannot
run on a plain sample covariance at all. Some regularization is **required** for them to exist;
OAS is one of several ways to supply it.

### Rung 10 — `eegnet_compare.py`: does a CNN beat designed filters?

**Goal:** the question is not "is deep learning better" but **at what sample size does
*learning* the filters start to beat *designing* them**. EEGNet is structurally doing what CSP
does — a temporal convolution discovers frequency filters, then a depthwise spatial convolution
learns a spatial filter per temporal filter — except end to end.

| regime | data | CSP + LDA | EEGNet |
|---|---|---|---|
| **A** within-subject, subject 1 | 45 trials | **91.1%** | 82.2% |
| **B** cross-subject LOSO, narrow band | ~900 trials | 59.4% | **60.1%** |

At n=45 the CNN loses by **8.9 points**. Pooled across 20 subjects the two are level.

**Two things keep that from being the "learned filters need volume" result it was written up as.**

First, 8.9 points on 45 trials is **41/45 against 37/45 — four trials**, and this rung runs no
significance test on it. Four discordant trials out of 45 is not a distinguishable difference, and
the Wilson intervals overlap across most of their range. The defensible statement is directional:
*CSP scores higher, by an amount this experiment cannot resolve.*

Second, **sample size and optimisation budget are confounded across the two regimes.** `N_EPOCHS`
is a fixed 100 with `BATCH_SIZE = 32`, so regime A's 36-trial training fold yields one batch per
epoch — about 100 gradient steps — while regime B's ~855-trial fold yields 26 batches per epoch,
roughly 2600 steps. The regimes differ in how much data the network saw *and* in how long it was
allowed to train, and the write-up credited only the first. Whatever the −8.9 measures, it is not
cleanly "what happens at n=45."

Third, and unaddressed here: the EEGNet numbers are **single-seed**. There is no seed sweep, and
the printed "± 11.3%" is a spread across five 9-trial folds — another rung of the same
quantization ladder §10.2 disowns — not an interval on the estimate.

**This rung was measuring a dead network, and it took adversarial review to catch it.**

MNE returns data in **volts**. The signal standard deviation is about 1.3e-5, so the variance is
about **1.6e-10**. braindecode's EEGNet normalises with `BatchNorm2d(eps=1e-3)` — a variance
**seven orders of magnitude below eps**. So batch norm divides by `sqrt(var + eps)` ≈
`sqrt(1e-3)` = **0.0316** instead of by the signal's own sigma of ~7e-6: its output comes out
about **4500× too small** where it should come out at 1. Normalisation never engages.

Batch norm does not do *nothing*, though, and getting this right matters for saying it out loud.
Each of the three BN stages multiplies by ~31.6×, so the deficit decays geometrically down the
stack and the **end-to-end logits are only about 53× too small**. The network still cannot train
out of it: closing a 53× gap means growing the final layer's weights from an init standard
deviation of ~0.065 to ~3.4 — a travel of about 3.3 — while 100 epochs of AdamW at lr=1e-3 on a
36-trial fold moves them by roughly 0.1. The budget is set by the optimizer, not by the data
scale.

> **Correction to the mechanism, not to the conclusion.** This paragraph used to say activations
> "stay near 1e-8" and that recovery "would require final-layer weights around 1e8." Both are
> wrong by about six orders of magnitude — nothing in the network is anywhere near 1e-8; the
> smallest activation standard deviation is ~7e-6. The 1e-8 was the *variance* of the first BN's
> output being mistaken for an activation magnitude, and `1e-8 × 1e8 = 1` then generated the bogus
> weight figure. One error, stated twice. The **"seven orders of magnitude"** headline was always
> right, and so was the conclusion; the arithmetic in between was invented. Since this is the part
> a reviewer would hear spoken aloud, it is corrected in place rather than softened.

The failure was **silent, and it looked like a result**:

| | accuracy | predicted class counts |
|---|---|---|
| as originally committed (volts) | 53.3% | **[0, 45]** |
| rescaled to microvolts | **82.2%** | [21, 24] — matches truth exactly |

The dead model **predicted a single class for all 45 trials**. Its 53.3% was the *majority-class
rate*, not chance performance — and "a CNN performs at chance on small data" is an entirely
plausible finding, which is exactly why it was written up as a headline result and recommended
for memorisation. The gap originally reported was **−37.8 points**. The real one is **−8.9**.

**The two guards now in the file are the actual lesson.** Every check in this project up to that
point was a *null* check: permutation tests, chance baselines, leakage assertions. Those catch a
model that is too good and **can never catch one that is dead**. So the file now carries

1. a **variance-versus-eps assertion** that fails loudly when the units are wrong, and
2. a **degenerate-prediction check** wired into the scorer itself, so it runs on *every fold* at
   no extra compute and refuses to score a model that emitted one class for everything.

CSP is unaffected by the units, because it works on variance *ratios*, which are scale
invariant. That asymmetry is precisely why the bug hid: the baseline was healthy, so the
comparison looked healthy.

### Rung 11 — `regime_decomposition.py`: what did rung 10's third experiment actually measure?

Rung 10 has a third regime this document has not mentioned yet, and the reason is that **it was
not interpretable**. Regime C reported that EEGNet *beat* CSP once both were given a wider band
and a longer window, and the write-up explained it: "CSP wins in regime B only because the band
was pre-selected for it."

Regime C differs from regime B in **three** ways at once:

1. the **band**, 8–30 Hz → 4–38 Hz
2. the **window length**, 1 s → 4 s
3. the **crop start**, 1.0 s → 0.0 s

Change three factors, measure one difference, and you have measured nothing. The third change is
the worst of them because it was never mentioned at all: starting at 0.0 s admits the **cue
period** into the decoding window, so regime C decodes a different *cognitive* window rather than
simply a longer one.

This rung re-runs it as a factorial with the crop start **pinned at 1.0 s**, so "longer window"
means "more imagery" rather than "now with added cue":

| | 8–30 Hz | 4–38 Hz |
|---|---|---|
| **1.0–2.0 s** | CSP 59.4% / EEGNet 60.1% | CSP 57.9% / EEGNet 56.1% |
| **1.0–4.0 s** | CSP 58.9% / EEGNet 60.7% | CSP 55.3% / EEGNet 57.7% |

with a fifth cell reproducing the original regime C exactly (4–38 Hz, **0.0**–4.0 s):
**CSP 51.4% / EEGNet 63.0%**.

**First, a control.** The `narrow-short` cell is regime B's configuration, and it reproduces
regime B's numbers exactly — CSP 59.4%, EEGNet 60.1%. That confirms the rewired harness — the new
cell config, the refactored pooling and scorer, the checkpointing wrapper — still measures what
rung 10 measured, which is what licenses reading the cross-cell deltas below as statements about
rung 10's claim.

> **Correction to how that control was described.** It used to read "an independent
> reimplementation landing on the same values is the evidence that this harness measures what the
> rung it audits measured." It is **not** an independent reimplementation: `load_subject`,
> `make_eegnet` and `seed_everything` are shared with `eegnet_compare.py`, with the same subjects,
> runs, seed, epoch count, batch size, CSP and LDA settings. Agreement therefore demonstrates that
> the plumbing did not perturb the measurement — a regression check — and **not** validity. A
> shared bug would reproduce perfectly.

**The result.** Paired across the same 20 subjects, EEGNet does not significantly beat CSP
anywhere in the 2×2. It only does so in the cell with the undocumented crop start:

| cell | EEGNet − CSP | 95% CI | p |
|---|---|---|---|
| narrow-short | +0.7 | [−5.2, +6.5] | 0.815 |
| wide-short | −1.8 | [−8.3, +4.8] | 0.578 |
| narrow-long | +1.8 | [−5.3, +8.8] | 0.603 |
| wide-long | +2.3 | [−5.1, +9.8] | 0.522 |
| **original-C** | **+11.6** | **[+5.0, +18.1]** | **0.002** |

Isolating that one change: the gap between the two models **widens by 9.2 points** when the crop
start moves from 1.0 s to 0.0 s — 95% CI [+2.0, +16.4], **p = 0.015**. It accounts for essentially
all of the 11.6. Split by model, the cue period **helps EEGNet (+5.3, p = 0.044)** and does not
significantly move CSP (−3.9, p = 0.134).

**And the stated mechanism is refuted outright.** "The band was pre-selected for CSP" predicts
that widening it should hurt CSP more than EEGNet. The opposite is closer to true:

| | effect of widening the band | p |
|---|---|---|
| CSP | −2.6 | 0.089 |
| EEGNet | **−3.5** | **0.017** |
| difference | −0.9, 95% CI [−5.2, +3.3] | 0.645 |

Widening the band hurts *both* models by about the same amount, and the only band effect that
reaches significance is the one on **EEGNet**. Nothing in the data supports the explanation that
was published.

**So what is regime C?** The measured claim is narrow and solid: *the entire "ranking flips"
result is produced by admitting the cue period, not by the band or the window that the write-up
credited.* The obvious explanation — a CNN's temporal convolutions can exploit a phase-locked
cue-evoked response, while CSP's log-variance band power is close to blind to one — is an
**interpretation, and this project's recurring failure has been inventing the mechanism in the
same breath as the number.** So rung 11 tests it instead of asserting it, with a sixth cell that
decodes the **cue window alone (0–1 s), which contains no imagery at all.**

**The confirmation.** Decoding the cue window on its own — 0–1 s, containing no motor imagery at
all — against the pooled majority-class floor of **50.1%** (pooling 20 subjects balances the
classes, unlike subject 1's 53.3%):

| model | cue window alone | against chance |
|---|---|---|
| CSP + LDA | 53.7% | 95% CI [50.6%, 56.8%], p = 0.027 |
| **EEGNet** | **61.1%** | 95% CI [57.3%, 64.9%], **p < 0.0001** |

EEGNet decodes a window with no imagery in it at **61.1%**. And the decisive comparison: adding
the entire three-second imagery window on top of the cue buys it **+1.9 points, 95% CI
[−2.8, +6.6], p = 0.41** — indistinguishable from nothing. CSP is not helped either (−2.2 points,
p = 0.20).

So EEGNet's 63.0% in regime C is, within measurement error, **entirely available from the cue
period alone**.

**And the control that check itself needed.** "EEGNet decodes 61.1% on the cue window" is only
about the cue if the same models score *nothing* on a window with no cue in it. A seventh cell
decodes the **second before the cue** (−1.0 to 0.0 s), band- and length-matched to the cue cell so
the two differ only in which side of the cue they sit on. That second is task-free: task onsets
are 8.3 s apart with a 4.2 s rest before each.

| cell | CSP + LDA | EEGNet |
|---|---|---|
| **pre-cue** (−1.0–0.0 s) | 47.7% (t = −1.69, p = 0.108 — at chance) | 51.8% (t = +1.39, p = 0.181 — at chance) |
| **cue** (0.0–1.0 s) | 53.7% (t = +2.48, p = 0.023) | 61.1% (t = +6.18, p < 0.0001) |
| paired difference, same 20 folds | **+6.0 pts** (t = +2.85, p = 0.0103) | **+9.3 pts** (t = +5.43, p < 0.0001) |

Both models sit at chance before the cue and above it after, so the effect is genuinely
**post-cue** rather than subject leakage, drift, or a harness bug. The control passes.

> Two limits on what it establishes, both of which the code now carries as comments. **(a)** A
> pre-cue null localises the effect to *after* the cue; it cannot split "cue flash" from "imagery
> onset," because in EEGBCI those begin at the same instant. Separating them needs a cue that
> looks identical across classes, and EEGBCI is position-confounded — target at screen top for
> fists, bottom for feet — so a class-discriminative visual evoked response necessarily exists
> post-cue whatever the subject imagines. This grid bounds **when** the effect starts, not **what**
> it is. **(b)** The filtering is applied to continuous data before cropping, and MNE's zero-phase
> 4–38 Hz firwin is a 265-tap symmetric FIR with a half-length of 0.825 s, so post-cue energy
> smears *backwards* into the pre-cue window. The direction is favourable — smear can only push a
> pre-cue score up, so a null is conservative — but an above-chance pre-cue result would have
> needed a truncated-filter re-run before it meant anything. And p = 0.108 is a failure to reject,
> not proof of a true null.

The interpretation is now a measurement: regime C's "the ranking flips" is the
CNN reading the **cue-evoked response**, not learning motor imagery better than CSP does. The
reason CSP cannot follow it there is structural — a phase-locked evoked deflection is a
*temporal* pattern, and CSP's log-variance features throw timing away by construction, keeping
only band power.

This is the same shape of finding as rung 7's gaze confound, and it earns the same framing: build
a rung, believe it, then find the confound in your own result. **Two of the eleven rungs turned
out to be decoding the stimulus rather than the intention.** For a project about reading
*intent*, that is the most useful thing in the repository, and it is why regime C is kept and
re-analysed rather than deleted.

> **The general lesson, worth carrying to any BCI result.** Both confounds were invisible to
> accuracy, to cross-validation, and to permutation testing, because in every case the model was
> finding real, reproducible structure. It was simply the wrong structure. The only checks that
> caught either one were **ablations that removed the thing the claim depends on** — frontopolar
> channels in rung 7, the imagery window here. If a result cannot survive deleting what it
> claims to be reading, the claim is about something else.

---

## 8. CSP — Common Spatial Patterns (the heart of the method)

CSP is the one genuinely EEG-specific algorithm here, and the thing most worth understanding
deeply. If you can explain CSP, you understand this project.

### 8.1 The problem CSP solves
Recall (§3.4): each electrode sees a smeared mixture of all brain sources. You can't just read
"C3 power" off one electrode. You want to *recombine* all 64 electrodes into a few **virtual
channels** (spatial filters) chosen so that the mu/beta **variance** is as different as possible
between the two classes.

Key link: for a band-passed signal, **variance = band power**. Since ERD *is* a change in
band power, a virtual channel whose variance separates the classes is literally reading out the
ERD difference. CSP finds those channels automatically.

### 8.2 What CSP actually computes (the intuition, minimal math)
CSP finds a set of channel-weightings (spatial filters) `w` such that when you mix the 64
channels with weights `w`, the resulting signal has:
- **high variance for class A (hands) and low variance for class B (feet)**, or vice versa.

Mechanically it does this by simultaneously diagonalizing the two classes' covariance matrices —
it solves a generalized eigenvalue problem on `Cov(hands)` and `Cov(feet)`. The filters at the
extreme ends of the eigenvalue spectrum are the ones where the variance ratio between classes
is most lopsided — the most discriminative spatial patterns.

`n_components=4` keeps the **4 most discriminative** filters (typically the 2 most "hands-favoring"
and 2 most "feet-favoring"). This is dimensionality reduction: 64 channels → 4 numbers per trial.

**The feature fed to the classifier** (`log=True`) is the **log-variance** of each of the 4
virtual channels over the trial:
- The log compresses the range and makes the feature distribution roughly Gaussian, which is
  exactly what LDA (next section) assumes. This is a standard, well-motivated pairing.
- `reg=None` → no covariance regularization (fine here; with fewer trials or more channels you'd
  often add shrinkage regularization for stability). `norm_trace=False` → don't normalize the
  covariance trace (a minor scaling choice).

So each trial becomes a 4-dimensional feature vector: `[logvar₁, logvar₂, logvar₃, logvar₄]`.

### 8.3 The `csp_patterns.png` plot — *interesting, but NOT the honesty check*
`csp.plot_patterns(...)` draws each spatial filter as a **scalp map** (a top-down head with a
color heatmap).

> [!warning] **An earlier version of this section was wrong and it is worth understanding why.**
> It claimed the learned patterns were "focal over central / sensorimotor cortex" and offered that
> as proof the model found motor sources. Checking the actual channel weights: component 0 *is*
> sensorimotor (FC3/C3/FC1), component 1 mixes sensorimotor weights (FC4/FC2/C4) with occipital
> ones, and the component this document showcased peaks at **POz, PO4, Oz** — parieto-occipital,
> and alpha-like. In MNE topomaps posterior is at the **bottom**, and a lower-central blob was
> misread as the vertex. **Reading a topography by eye is not a control.**
>
> **A later layer on this same correction.** The sentence above used to end "…and correlates
> **r = 0.57** with the subject's own eyes-closed alpha map." That figure is withdrawn: no script
> in this repo computes any correlation, and it did not reproduce under the obvious definitions of
> one. It was load-bearing prose *inside a retraction*, which makes it the worst possible place for
> an unproduced number — the honest-self-correction section was itself resting on one. The
> qualitative claim survives on the channel weights alone; the figure does not survive at all.

**Why this matters enormously:** a model can hit 91% by cheating — locking onto an eye-blink
artifact, a neck-muscle tension that happens to correlate with the cue, or a per-run drift. The
control that actually catches this is an **ablation**: remove the cortex that should carry the
signal and see whether the decoder dies. Here it does — frontopolar-only lands at **51.1%
(23/45)**, one trial *below* the 53.3% majority-class floor — while keeping only the 17
sensorimotor channels holds accuracy at **95.6% (43/45)**.
The intuition people reach for is: if CSP had learned an artifact, the scalp maps would light up at
the *edges* of the head (eyes, temples, neck) rather than the center, so centered patterns prove
motor sources. **That intuition is too weak to rely on, and this project is a worked example of why.**
Occipital alpha sits at the back of the head, not the edge; it is large, it is inside the 8-30 Hz
band, and at a glance in a topomap it is easy to mistake for something central.

**Accuracy alone never proves the signal is neural, and neither does a scalp map.** What proves it is
an ablation, because it makes a falsifiable prediction: if the decoder is reading sensorimotor cortex,
then deleting sensorimotor cortex must break it. Here that prediction holds, sharply. Being able to
say *"here is the control I ran, and here is what would have falsified it"* is what separates a
credible BCI result from a lucky number.

---

## 9. LDA — Linear Discriminant Analysis (the classifier)

After CSP, each trial is just 4 numbers. LDA is a simple, fast linear classifier:
- It models each class as a Gaussian blob in the 4-D feature space and finds the straight
  boundary (a hyperplane) that best separates the two blobs, assuming they share a covariance
  shape.
- **Why LDA and not something fancier?** With only 45 trials, a complex model would overfit
  instantly. LDA has almost no free parameters, is the textbook partner to CSP log-variance
  features (which were designed to be Gaussian-ish for exactly this), and is the canonical
  baseline in the BCI literature. It's the *right* amount of model for the data size.

The `Pipeline([("CSP", csp), ("LDA", ...)])` chains them so that — critically — **CSP is
re-fit on only the training data inside each cross-validation fold.** If you fit CSP on all data
once and then cross-validated only the LDA, CSP would have "seen" the test trials and your
accuracy would be inflated. Wrapping both in a Pipeline is what makes the evaluation honest.

---

## 10. Evaluation — cross-validation and the scoreboard

### 10.1 Why not just train/test once?
With 45 trials, a single random split could be lucky or unlucky by chance. One number would be
meaningless.

### 10.2 StratifiedKFold ×5
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:
- **Partition** the 45 trials into 5 folds. Train on 4, test on 1, rotate.
- **Stratified**: each fold preserves the 21/24 class balance, so no fold is scored against a
  wildly different floor than the others.
- Every trial is tested **exactly once**. Nothing is skipped and nothing is double-counted.
- `random_state=42` fixes the fold assignment so the run is reproducible.

This replaced `ShuffleSplit(n_splits=10, test_size=0.2)`, which was neither a partition nor
stratified: it never tested 5 of the 45 trials while testing others up to 6 times, and let class
balance swing from 2:7 to 7:2 across folds. §7 rung 5 walks through the whole comparison,
including the finding that most of the resulting 94.4 → 91.1 change was **seed luck rather than
a correction**.

> **The ± is still not a spread, and this document should not pretend otherwise.** The five folds
> score `[8/9, 8/9, 8/9, 8/9, 9/9]`. A 9-trial test set can only produce multiples of 1/9, so
> "± 4.4%" is one rung of exactly the same quantization ladder that made "± 5.6%" meaningless. It
> is reported because the script prints it, not because it is a confidence interval. For an
> actual interval, a Wilson bound on 45 trials gives roughly **[79%, 97%]** — and even that is
> optimistic, because it treats 45 cross-validated predictions as independent draws from one
> model when they come from five different ones.

### 10.3 The chance baseline
```python
chance = max(mean(labels==2), mean(labels==3))   # = 24/45 = 53.3%
```
A model that always guesses the majority class ("feet") would be right 53.3% of the time. So the
number that matters is never "91%" in a vacuum but **"91% against a 53.3% floor."** *Always
report accuracy against chance;* a raw accuracy with no baseline is a red flag a mentor will
catch immediately.

Note this is the **majority-class rate**, not 50%. That distinction did real damage in this
project: a broken network in rung 10 scored exactly 53.3% by predicting one class for every
trial, and because 53.3% reads as "chance," it was mistaken for a finding rather than a bug.

### 10.4 The permutation test — is 91% outside what noise produces?

Cross-validation tells you the estimate is stable. It does not tell you the signal is real. For
that, `permutation_test_score` shuffles the labels 1000 times and re-runs the **entire pipeline**
on each shuffle, building the distribution of accuracies this exact method produces on data whose
labels mean nothing.

The null lands at **50.7% ± 8.5%**, with a maximum of 82.2% across 1000 shuffles. The observed
91.1% sits outside all of it.

Two honesty notes about the reported p-value:

- With 1000 permutations the smallest reportable value is **1/1001**, so a printed `p = 0.0010`
  would be the **resolution floor of the test, not a measurement**. Both `decode_csp.py` and
  `evaluate_honestly.py` detect that case and print **`p <= 0.001`** instead.
- scikit-learn counts permutations scoring **≥** the observed value, so the correct phrasing is
  "no shuffle matched or exceeded the real result," not "none beat it."

### 10.5 The ablation — the control that actually rules out artifacts

A permutation test proves the model found *structure*. It cannot prove that structure is
**motor**. A decoder riding an eye-movement artifact that correlates with the cue would pass a
permutation test comfortably.

The control that discriminates is an **ablation**, because it makes a falsifiable prediction: if
the decoder is reading sensorimotor cortex, then removing sensorimotor cortex must break it.

Every row below is printed by **`ablate_channels.py`**, which reuses `decode_csp.py`'s
preprocessing verbatim — one seed (42), one splitter, one pipeline, the same 45 trials. The
all-64 row reproducing the published 91.1% exactly is the fidelity check that the pipeline was
*reused* rather than reimplemented.

| channels used | ch | accuracy | trials | per-fold |
|---|---|---|---|---|
| sensorimotor only — FC3/1/z/2/4, C5…C6, CP3/1/z/2/4 | 17 | **95.6%** | 43/45 | 1.00 .89 1.00 .89 1.00 |
| all 64 | 64 | 91.1% | 41/45 | .89 .89 .89 .89 1.00 |
| frontopolar only — Fp1/z/2, AF7/3/z/4/8 | 8 | **51.1%** | 23/45 | .56 .78 .33 .56 .33 |
| leave-one-run-out, all 64 | 64 | 93.3% | 42/45 | .87 .93 1.00 |

Delete the cortex that should carry the signal and the decoder falls to the **majority-class
floor**: 51.1% is 23/45 against the 24/45 you get by ignoring the EEG entirely and always
answering "feet." The per-fold spread — 0.33 to 0.78 — is the other tell; folds that wide are a
coin, not a decoder. Leave-one-run-out at 93.3% holds with no trial sharing a run with its
training set, so the headline is not a within-session drift artifact. That is a control. Reading
a scalp map by eye — which is what this document used to offer here — is not, and §8.3 is the
full account of why.

**The other direction is much weaker than this document claimed, and the distinction matters.**
Sensorimotor-only is 43/45 against all-64's 41/45. That is a **two-trial** difference on n=45,
well inside noise, and it is mostly a dimensionality effect anyway — a 64×64 covariance estimated
from 45 trials overfits, so almost any well-chosen small subset avoids that. The claim the
artifact control actually needs is *"removing 47 non-motor channels does not hurt,"* which the
data fully supports. *"The sensorimotor subset is better"* is not supported and is no longer
claimed.

> ### The correction that produced this script
>
> Until this revision the table above read **95.9%** for sensorimotor and **47.4%, i.e. chance**
> for frontopolar, and **no script in the repository produced any of the four rows**.
> `ablate_channels.py` did not exist. This was the repo's designated artifact control — the thing
> §8.3 says "replaced" the retracted scalp-map defence — so it was simultaneously the most
> load-bearing and the least sourced number here.
>
> Both bolded values were also *arithmetically unreachable*. With 45 trials and a partition that
> tests each exactly once, accuracy is k/45 — steps of 2.222%. Attainable values near the
> headline: 39/45 = 86.7%, 40/45 = 88.9%, 41/45 = 91.1%, 42/45 = 93.3%, 43/45 = 95.6%,
> 44/45 = 97.8%. **No k gives 0.959 or 0.474.** The script now both prints this lattice and
> *asserts* it, and cross-checks `cross_val_score`'s fold mean against a pooled
> `cross_val_predict` count, so an unequal-fold scheme cannot silently emit an off-lattice number
> that still looks like one.
>
> | published | actual | reason |
> |---|---|---|
> | 95.9% | **95.6%** (43/45) | off-lattice; almost certainly a transcription slip of 95.6 |
> | 47.4%, "i.e. chance" | **51.1%** (23/45) | off-lattice, *and* the wrong reference — chance here is the 53.3% majority-class rate, not 50% |
> | 91.1% | 91.1% (41/45) | already correct |
> | 93.3% | 93.3% (42/45) | already correct |
>
> This is the same failure mode §12.1 names five times over — *the number and the story arrived
> together* — except here the number arrived without even a run behind it. It is why
> `check_provenance.py` now exists: it extracts every figure from this document and the README and
> fails if no script's stdout contains it.
>
> **One limit on what any of this bounds.** The average reference is computed over all 64 channels
> *before* a subset is picked, which is what `decode_csp.py` does, so the subsets are not
> electrically independent — every channel carries −1/64 of every other. Re-referencing each
> subset separately would give a cleaner number but would no longer be the published pipeline. The
> ablation therefore **bounds** the ocular contribution rather than eliminating it. It is also
> **band-conditional**: at 8–30 Hz it says nothing about low-frequency ocular contamination, which
> is where rung 7's gaze confound lives.

### 10.6 How to read the final printout
```
CSP+LDA accuracy: 91.1%  (+/- 4.4%)
Chance (majority class): 53.3%
Per-fold: [0.89 0.89 0.89 0.89 1.  ]
Permutation test: p <= 0.001 (null 50.7% +/- 8.5%, max 82.2%)
```
Four folds at 8/9 and one at 9/9. Read that per-fold line as the quantization warning it is: the
folds are not sampling a smooth distribution, they are landing on the only values a 9-trial test
set can produce.

---

## 11. The tunable knobs (your iteration surface)

Every constant near the top of the scripts is a lever. Here's what each does and what happens
if you turn it:

| Knob | Current | What it controls | Turning it |
|---|---|---|---|
| `SUBJECT` | 1 | Which person | **Already swept** — rung 6 runs all 109. Median 60.0%, and subject 1 is the 91st percentile. |
| `RUNS` | 6,10,14 | Which task | **4,8,12** = *imagined* left vs. right fist, a much harder contrast on the same homunculus strip. **Not 3,7,11 — those are executed movement** (see the run table in §4). **Already built** as rung 7, where it found a gaze confound. |
| `L_FREQ,H_FREQ` | 8,30 | The band kept | Split into mu (8–12) and beta (13–30) and combine (filter-bank CSP) — often a real gain, and still unbuilt here. Splitting them on the left/right contrast is what exposed it as an alpha-band decoder. |
| CSP `n_components` | 4 | # spatial filters | 6 or 8 — more filters can help or overfit; cross-validate to decide. Untested here. |
| CSP `reg` | None | Covariance shrinkage | `'ledoit_wolf'` — stabilizes CSP when trials are few or channels many. Untested on this baseline. Worth noting that rung 9's Riemannian arm ran with OAS shrinkage while its CSP baseline ran `reg=None` — an asymmetry that favours the **Riemannian** arm, not the baseline. The baseline won anyway. |
| crop `1.0–2.0 s` | 1 s window | Which slice becomes features | Sliding it on the left/right contrast gave **66.7 / 55.6 / 73.3 / 64.4 / 46.7%** for starts at 0.0 / 0.5 / 1.0 / 1.5 / 2.0 s — a **26.7-point** range across *overlapping* windows of the same trials. Treat this knob as a **noise source**, not a tuning surface. |
| `cv` | `StratifiedKFold(5, shuffle=True)` | Evaluation rigor | **Leave-one-subject-out** once you go multi-subject (rungs 8–10 do). Leave-one-**run**-out is the cheaper session-level check: it holds at 93.3%. |
| `random_state` | 42 | Reproducibility seed | Rung 5 already swept 100 seeds: 88.9–97.8%, mean **93.8%**, with 42 at the **3rd percentile** — so the published number understates its own estimator by ~2.7 points. That "3rd" is a *strictly-below* rank; the estimator is quantized to 1/45, many seeds tie exactly on 91.1%, and a tie-aware rank puts 42 higher. The 2.7-point gap does not depend on the convention; the word "3rd" does. Worth knowing that 42 was **inherited from MNE's CSP tutorial, not chosen** — which makes "the seed wasn't cherry-picked" true but vacuous. |

> **On that last row.** This baseline is MNE's CSP tutorial almost verbatim: the runs, the
> subject, `tmin`/`tmax`, `firwin`, the 1–2 s crop, `CSP(n_components=4, log=True,
> norm_trace=False)`, and the seed all match. That is entirely legitimate for a baseline, and it
> is much better to say so first than to have someone else point it out.

---

## 12. The scoreboard: what every rung actually returned

An earlier version of this section was a **wish list of six next projects, five of which were
already built**. Here is the real state instead.

| Rung | Question | Answer |
|---|---|---|
| 4 | Can I decode imagined fists vs. feet? | **91.1%** vs. 53.3% chance, permutation **p ≤ 0.001** |
| 5 | Is that number an artifact of the estimator? | Partly. The estimator change was worth ~0.6 points; ~2.7 points were **seed luck** |
| 6 | Does it hold across 109 people? | Median **60.0%**, IQR 52.8–75.6%. Subject 1 is the **91st percentile** |
| 7 | What does a harder contrast cost? | **Unanswerable as run** — the left/right rung is **gaze-confounded**, and overlapping crop windows of the same trials disagree by 26.7 points |
| 8 | Does it transfer to an unseen person? | Near-parity, 95% CI **[−1.9, +11.2]**, p = 0.181. Cannot distinguish a drop from no drop |
| 9 | Does a Riemannian method beat it? | No, but only **MDM-64 is significant** (p = 0.005); the rest is underpowered — and the comparison was tilted *toward* Riemannian by shrinkage |
| 10 | Does a CNN beat it? | Scores lower by **4 of 45 trials** at n=45 (reported as 8.9 points, untested), level at ~900 trials |
| 11 | What did rung 10's regime C measure? | **The cue, not the imagery.** EEGNet scores 61.1% on the cue window alone and **51.8% on the second before it**; adding the whole imagery window buys +1.9 (p = 0.41) |
| — | Is the artifact ablation real? | **It is now.** `ablate_channels.py` produces it; two of its four published values were unreachable and are corrected below |

### 12.1 What is retracted, and why that list matters

Claims this project published that did not survive adversarial review. They are listed here
rather than quietly deleted, because the list is more informative than the results. **This list
only grows** — when a correction is itself later found wanting, a layer is added rather than the
record rewritten.

**Round one — the six found by the first review:**

- **"27% BCI illiteracy."** Inverted inference. A pure-noise null predicts ~55% below chance;
  28% was observed. It is evidence *of* signal.
- **"EEGNet loses by 37.8 points."** A units bug. The network was never training. Real gap: 8.9.
- **"The ranking flips once both models get a wider band and a longer window."** Three factors
  moved at once, and the effect turns out to belong to none of the two that were credited. Rung
  11 traces it to an undocumented crop-start change, shows the stated band mechanism is
  backwards, and confirms by measurement that EEGNet scores **61.1% on a window containing no
  imagery at all**.
- **"73.3% is what a harder contrast costs."** Group value is ~7 points, and the rung is
  gaze-confounded.
- **"No method dominates / the best method is subject-specific."** No subject × method
  interaction is **detectable** (p = 0.84), and per-subject optimality requires one. (An earlier
  version of this line said no interaction *exists*, which asserts a null this experiment has too
  little power to establish — the same move §14 item 6 warns against.)
- **"CSP patterns are focal over sensorimotor cortex, which is my evidence against artifacts."**
  False for the showcased component, which is parieto-occipital (§8.3).

The pattern is worth naming, because it is the same mistake five times: **the mechanism story was
invented in the same breath as the number**. Measuring and explaining are separate steps, and
doing them together is how a plausible narrative gets attached to noise.

**Round two — found by writing the scripts the round-one corrections had promised.** Every entry
below is a number that survived the first review because nobody checked whether any code produced
it. This is a *different* failure mode from round one, and arguably a worse one: round one
invented mechanisms for real measurements; round two published measurements that were never made.

- **"Sensorimotor only 95.9%, frontopolar only 47.4%, i.e. chance."** The repo's headline artifact
  control, quoted in the README, in §1, in §8.3, in §10.5 and in the one-minute talking points —
  and produced by **no script**. Both values are also off the k/45 lattice that 45 trials and a
  partition CV force. Real: **95.6% (43/45)** and **51.1% (23/45)**, the latter *below* the 53.3%
  majority-class floor rather than at "chance." `ablate_channels.py` now produces the table and
  asserts the lattice (§10.5).
- **"Keep only that cortex and it improves."** 43/45 versus 41/45 is a two-trial difference on
  n=45 and is not distinguishable from no change. The supported claim is that dropping 47
  non-motor channels *does not hurt*. Only the collapse half of the ablation was ever load-bearing.
- **"A frontopolar-only decoder at 0.5–5 Hz matches the 64-channel result."** The evidence for
  rung 7's gaze confound, and it was window-shopped: 73.3% frontopolar came from the whole 0–4 s
  epoch while the 64-channel headline came from a 1-second crop. Matched, it is **53.3% against
  73.3%** (§7 rung 7). The confound is real and is now better evidenced — a cue-window frontopolar
  asymmetry at **t = +7.71, p = 3.7e-09** — but by a different measurement than the one published.
- **"r = 0.57 between the showcased CSP component and the subject's eyes-closed alpha map."**
  Stated three times across the two flagship documents, computed nowhere, and not reproducible
  under the obvious definitions. Withdrawn. It sat *inside* the §8.3 retraction, so the
  self-correction was itself resting on an unproduced number.
- **"Ocular checks come back clean: HEOG p = 0.27, VEOG p = 0.44."** EEGMMIDB ships **zero EOG
  channels**, which the same page says two paragraphs earlier. These were undisclosed frontal-EEG
  surrogates, computed by no script, and an underpowered null presented as a clean bill of health.
  Withdrawn and replaced by the scripted frontopolar ablation (§14 item 4).
- **"Rung 9's shrinkage flattered the comparison in the baseline's favour."** Backwards. OAS sits
  on the **Riemannian** arm; the CSP baseline runs `reg=None`. The asymmetry favoured Riemannian,
  and the baseline won anyway. §7 rung 9 had it right and §11 had it wrong — the document
  contradicted itself about its own most-cited negative result.
- **"An independent reimplementation landing on the same values."** Rung 11's control shares
  `load_subject`, `make_eegnet` and `seed_everything` with the rung it audits. It is a regression
  check on the plumbing, not independent validation; a shared bug would reproduce perfectly.
- **"EEGNet loses by 8.9 points at n=45 — learned filters need volume."** The 8.9 points are
  **four trials out of 45**, no significance test was run on them, and the two regimes differ in
  optimisation budget (~100 gradient steps versus ~2600) as well as in sample size. The direction
  is fine; "learned filters need volume" is not what was measured.
- **"79 / 109 subjects above their own chance."** A bare inequality reported in a table that reads
  as a count of people with decodable signal. No per-subject test is run. Relabelled, not removed —
  rung 6's actual conclusion is a population-level comparison that does not use it.

**The round-two pattern has its own name, and it is the one worth carrying forward:** *a number
whose provenance you cannot state is not a result, however true it turns out to be.* Two of the
figures above (91.1%, 93.3%) were correct all along, which is precisely why the defect survived
review — a table that is half right reads as sourced. `check_provenance.py` now extracts every
figure from this document and the README and fails if no script's stdout contains it.

### 12.2 What is genuinely next

1. **Filter-bank CSP (FBCSP).** Run CSP separately in sub-bands and let the classifier combine
   them. A well-known, reliable gain, still classical and interpretable. Genuinely not built.
2. **More trials per subject.** Almost every limitation above traces back to **45 trials**: the
   quantized folds, the underpowered comparisons, the untestable learning curve. Public corpora
   exist with 2,000–5,000 trials per subject, reachable through one library. This is the single
   highest-value change available, and it unblocks items 3 and 4.
3. **The learning curve that settles rung 8's retracted claim.** Hold the subject fixed and sweep
   training-set size from 45 upward. That converts "the barrier is sample size, not anatomy" from
   an assertion into a measurement. It needs item 2 first.
4. **EEGNet where it can actually win.** The CNN lost at n=45 and tied at ~900. The comparison
   only becomes interesting with an order of magnitude more data, and item 3's curve predicts
   where the crossover should be — so it becomes a **test of a prediction** rather than another
   isolated data point.
5. **Artifact rejection.** ICA-based ocular cleaning, and a paradigm with EOG channels. Rung 7
   demonstrated this project cannot currently monitor the confound it found, let alone remove it.
6. **Trial-count QC, and a live bug inside it.** 12 of 109 subjects reach the sweep with
   non-standard trial counts (36–57 instead of 45), and three record at 128 Hz rather than 160.
   The sweep reports the sampling-rate anomaly and not the timing one. More importantly, **most
   of those non-standard counts are this repo's own fault, not the dataset's**: `sweep_subjects.py`
   epochs at `tmin=-1.0, tmax=4.0` — inherited verbatim from MNE's CSP tutorial — but only ever
   uses the 1–2 s crop, so on subjects whose runs end early the 4-second tail runs off the end of
   the recording and MNE silently drops those trials. The script's own stated rule is "nothing is
   dropped silently," and at the trial level it does not hold. Right-sizing the window to
   `tmin=0.0, tmax=2.0` would recover them and would move the published per-subject numbers.
   Untouched here deliberately: the sweep's figures in this document are what the current script
   prints, and changing the script and the prose in the same pass is how a repo loses track of
   which is which.
7. **Statistics that live only in prose.** Several inferential figures in rungs 7–9 — the
   cross-subject CI, rung 9's paired p-values and its subject × method interaction — are correct
   but are computed by no committed script, and rungs 8 and 9 persist no per-fold arrays, so a
   reader must edit the source to re-derive them. `check_provenance.py` flags this class; closing
   it means printing the tests from the scripts that own them.

---

## 13. Reproduce it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python decode_csp.py        # full pipeline + writes csp_patterns.png
```
- The dataset downloads automatically on first run and is cached in `~/mne_data`, so later runs
  are fast and offline-capable. The first run of rung 6 pulls ~840 MB (all 109 subjects).
- `decode_csp.py` alone reproduces the headline number and the scalp-map figure. Rungs 1–3 are
  optional checkpoints you can run individually to inspect each stage; rungs 5–11 are the
  attacks on the result and each runs standalone.
- Because `random_state=42` is fixed, `decode_csp.py` prints **91.1% (+/- 4.4%)** and
  **p <= 0.001** exactly. The classical rungs are all deterministic in this way.
- **The CNN rungs report a single torch seed.** Seeds are fixed for torch, numpy and python, and
  in practice repeated runs on this machine have come back identical — but run-to-run stability is
  **not** robustness to seed *choice*, and no seed sweep is run. Treat every EEGNet number as one
  draw. (An earlier version of this bullet asserted the CNN rungs were "not bit-reproducible" and
  blamed MPS nondeterminism. That reassured the reader out of noticing the real issue, which is
  that a single initialisation is being reported as an estimate.)
- Rungs 10 and 11 are the slow ones (LOSO with a CNN at every fold). `regime_decomposition.py`
  checkpoints to `regime_decomposition.json` after each cell, so it can be killed and resumed
  without losing completed work.

---

## 14. The honest limitations (know these before a mentor asks)

The README states these plainly, which is itself a strength — over-claiming is the cardinal sin
in BCI. Be ready to volunteer them:

1. **Within-subject, small-n.** One subject, 45 trials. The headline says nothing about whether
   this works on a *new* person. And the estimate is genuinely imprecise: an honest interval is
   roughly **[79%, 97%]**, not the ± the script prints, which is a quantization step (§10.2).
2. **Easy contrast.** Fists-vs-feet are far apart on the homunculus, so their scalp patterns
   differ a lot. This is close to the easiest possible motor-imagery discrimination.
3. **Clean subject, and I know exactly how clean.** Subject 1 is the **91st percentile** of the
   109; the median subject gets 60.0%. Picking it is fair for a baseline demo, and quoting it
   without the distribution would not be.
4. **No artifact rejection, and no way to monitor it.** There is no ICA and no removal of
   eye-blinks or muscle activity. **This document used to claim the "centered" CSP scalp maps
   gave confidence the model was not riding artifacts. That claim was false** — the showcased
   component is parieto-occipital and alpha-like (§8.3). The real defence is the ablation in
   §10.5: on fists-vs-feet, a frontopolar-only decoder lands at **51.1%**, one trial below the
   majority-class floor. For the left/right rung the same kind of check emphatically fails.

   > **Correction, and it is about vocabulary as much as about numbers.** This item used to read
   > "for fists-vs-feet the ocular checks come back clean (HEOG p = 0.27, VEOG p = 0.44)." Both
   > figures are withdrawn. **EEGMMIDB ships no EOG channels** — which this same page says two
   > bullets down — so nothing here is an HEOG or a VEOG; they were undisclosed *frontal-EEG
   > surrogates* (a lateral AF7−AF8 difference and a mean over Fp1/Fp2), computed by no script,
   > and described with labels that imply electrodes the recording does not have. They were also
   > an underpowered null sold as a clean bill of health: at n=45 that test could not have detected
   > an ocular effect a third the size of the one rung 7 found. The scripted frontopolar ablation
   > above says the same thing honestly and with provenance. Any ocular statement in this repo is
   > a **frontal-EEG surrogate** and is named as one.
5. **Two confirmed confounds, in rungs 7 and 11.** Rung 7's left/right decoding rides a
   lateralised gaze artifact, and EEGMMIDB ships no EOG channels, so it can be bounded but
   neither removed nor monitored. Rung 11's wide-window CNN comparison turned out to be reading
   the cue-evoked response: EEGNet scores **61.1% on a window containing no imagery at all**.
   Neither touches the fists-vs-feet headline, which passes both its ocular checks and its
   channel ablation — but together they are why this project treats "the accuracy is real" and
   "the accuracy is measuring what I said it measures" as two separate claims.
6. **Underpowered comparisons.** With n=20 subjects the minimum detectable difference is roughly
   6–10 points depending on the comparison. Several "no difference" results in rungs 8 and 9 are
   really "this experiment cannot tell," which is a different statement. That includes the
   subject × method interaction rung 9 leans on: **"no interaction exists"** overstates it, and
   **"no interaction is detectable here"** is what the test supports. The retraction it backs — that
   the best method is subject-specific — stands either way, because a positive claim is withdrawn
   for lack of support regardless of the test's power.
7. **Not a real-time system.** This decodes pre-recorded, pre-cut trials offline. A live BCI adds
   continuous decoding, latency, and no clean trial boundaries.

None of these are bugs; they are the honest scope of a baseline. The project's credibility comes
precisely from *stating* them, and from the fact that item 4 is written against this document's
own earlier claim rather than quietly edited out.

---

## 15. Talking points — how to discuss this in one minute

If a mentor says *"walk me through it,"* this is the spine:

> "It decodes imagined movement — fists vs. feet — from scalp EEG. When you imagine moving,
> the motor rhythm (8–30 Hz mu/beta) drops in power over the specific patch of motor cortex for
> that body part — that's event-related desynchronization. Fists and feet map to different
> places on the motor homunculus, so the *spatial pattern* of that power drop differs. I
> band-pass to the motor band on the continuous signal — before epoching, to keep filter edge
> artifacts out of the trials — then use CSP to recombine all 64 electrodes into a few virtual
> channels whose variance best separates the classes, feed the log-variance into an LDA, and
> cross-validate. 91% versus a 53% chance baseline on one clean subject, with a thousand-shuffle
> permutation test at p under 0.001. My evidence that it's motor and not artifact is an ablation:
> on the 17 sensorimotor channels alone it holds at 95.6%, and on 8 frontopolar channels alone it
> drops to 51.1% — which is *below* the 53.3% majority-class floor, so that decoder is one trial
> worse than always guessing feet. The honest caveats are that it's one subject and an easy
> contrast, and that subject sits in the top decile of the 109 I swept, where the median is 60%."

And when they ask the follow-up that actually matters — *"what did you get wrong?"* — this is the
stronger half of the answer:

> "Four things, and two of them are the same mistake. I built a left-versus-right rung and
> reported what a harder contrast costs. It was decoding gaze: the cue sits on one side of the
> screen for the whole trial, and in the cue window the frontopolar left-minus-right asymmetry is
> plus-twelve microvolts on left cues and minus-thirteen on right, t of seven-point-seven. A
> decoder on frontopolar mean amplitude alone gets 87% there. Then I compared against a CNN and
> reported that it wins once
> you give it a wider band and a longer window. It wasn't the band or the window — I had also
> moved the crop start without noting it, which let the cue-evoked response into the window. The
> CNN scores 61% on the cue period alone, with no imagery in it at all, and adding the entire
> imagery window on top buys it two points I can't distinguish from zero. Both times the model
> was finding real, reproducible structure that survived permutation testing. It was just the
> wrong structure. The third was a units bug: MNE returns volts, and the network's batch-norm
> epsilon is **seven orders of magnitude** larger than the signal variance, so normalisation never
> engaged, it never trained, and it scored exactly the majority-class rate — which reads as 'CNN
> performs at chance on small data,' a completely plausible finding I wrote up as a headline.
>
> "The fourth is the one I'd lead with now, because it isn't about a mechanism. My headline
> artifact control — the channel ablation — was a table in the README that **no script in the repo
> produced**. Two of its four numbers weren't even reachable: with 45 trials tested once each,
> accuracy has to be a multiple of one-forty-fifth, and 95.9% and 47.4% aren't on that grid. The
> real numbers are 95.6% and 51.1%, and 51.1% is *below* the majority-class floor, so the control
> is actually stronger than the version I'd published. I wrote the script, corrected the table, and
> then wrote a second script that extracts every number in my docs and fails if no script prints
> it — because the same defect turned out to be sitting in three other places, including inside one
> of my own retractions."

That answer is worth more than the 91%. Anyone can report an accuracy; the thing a lab is
actually screening for is whether you find your own confounds before a reviewer does.

If you can say both, you own this repo.

---

## 16. Mini-glossary

- **BCI** — Brain–Computer Interface. Turning brain signals into commands.
- **EEG** — Electroencephalography. Recording brain electrical activity from scalp electrodes.
- **Motor imagery** — Imagining a movement without performing it.
- **Homunculus** — The body-map layout of motor/sensory cortex; hands lateral, feet central.
- **Mu / beta rhythms** — ~8–12 Hz / ~13–30 Hz oscillations over idling motor cortex.
- **ERD (Event-Related Desynchronization)** — The drop in mu/beta power when a region engages.
- **Raw / Epochs** — MNE objects for a continuous recording vs. a stack of cut trials.
- **Annotation / event** — A timestamped cue marker; the ground-truth label source.
- **Montage** — The 3-D scalp positions of the electrodes.
- **Average reference** — Re-referencing each channel to the mean of all channels.
- **Band-pass filter** — Keep only frequencies in a chosen band (here 8–30 Hz).
- **Edge artifact** — Filter distortion at the start/end of the filtered segment.
- **CSP (Common Spatial Patterns)** — Learns channel-mixings that maximize the between-class
  variance ratio; the core spatial-filtering step.
- **Log-variance feature** — Log of a virtual channel's variance = its band power; the feature CSP feeds LDA.
- **LDA (Linear Discriminant Analysis)** — Simple linear classifier separating two Gaussian feature clouds.
- **Cross-validation** — Repeatedly train and test on different splits to get an honest accuracy.
- **StratifiedKFold** — A cross-validation *partition* that tests every trial exactly once and
  preserves class balance in each fold. What this project uses.
- **ShuffleSplit** — Independently resampled train/test splits. Neither a partition nor
  stratified; the estimator this project moved away from.
- **Chance level** — Accuracy of always guessing the majority class (here 53.3%), *not* 50%.
- **Permutation test** — Shuffle the labels many times, re-run the whole pipeline, and see where
  the real result falls in the resulting null distribution.
- **Ablation** — Removing a part of the input to test whether the model depends on it. The
  artifact control this project relies on.
- **LOSO (leave-one-subject-out)** — Train on N−1 people, test on the held-out one, rotate.
- **Degenerate classifier** — A model predicting one class for every trial. It scores the
  majority-class rate, which is easily mistaken for chance performance.
- **EEGNet** — A compact CNN for EEG; the deep-learning comparison in rungs 10 and 11.
- **MDM / Tangent Space** — Riemannian classifiers operating on covariance matrices: nearest
  class mean along the manifold, and a projection to a flat space where ordinary classifiers work.
