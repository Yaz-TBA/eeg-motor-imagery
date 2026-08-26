# EEG Motor-Imagery Decoding: The Complete Explainer

This is the long version of the repository: what it does, the science and the math behind every
step, why I made each decision, how to run it, and where I would take it next.

I wrote it for the version of me who was starting this in June and didn't know any of it yet.
So it goes slowly, and it explains things a textbook would assume you already know. If you want
the short version, the README covers the result in about a page.

Everything I have gotten wrong so far is in here too, with the wrong number sitting next to the
one that replaced it. That is the part I would read first if I were you.

---

## 1. What this is

When you imagine moving your hands, the part of your brain that controls your hands changes
its electrical rhythm, even though nothing actually moves. You can pick that change up with
electrodes sitting on the scalp. This project reads that signal and guesses which movement you
were imagining: both fists, or both feet.

On one subject it gets **91.1%** right, against a **53.3%** floor you would hit by always
guessing "feet". A 1000-shuffle permutation test puts that at p <= 0.001.

I want to be as authentic as I can here with the limitations of the number here, because I've
already been mistaken on it once. It is one subject, 45 trials, a public dataset, and the
easiest contrast in the set. This is the "hello world" of brain-computer interfaces. It is not
a novel result and I am not presenting it as one.

The part I think is actually worth your time is the second half of this document: the suite I
built to attack my own result, and the four rounds of corrections it forced. The headline used
to say 94.4%. It doesn't anymore, and §10 is where that happened. Every number I have withdrawn
should still be in here, sitting right next to the retraction :3

### What the artifact controls do and don't show

The decoder **bounds** the ocular contribution by **ablation**. Restricted to 17 sensorimotor
channels it holds at 95.6%; restricted to 8 frontopolar channels it falls to 51.1%, *below* the
53.3% majority-class floor. That bounds the artifact rather than proving the signal is motor,
which is a weaker claim than I originally made and the right one.

**And the sharpest test of that story has now been run, and it came back the wrong way.** This
paragraph used to end with the concession *"no condition deletes sensorimotor cortex while
keeping the rest of the montage,"* which is kept here as the record of what was owed. That
condition now exists. Deleting the 17-channel sensorimotor strip and refitting on the remaining
47 electrodes leaves the decoder at **77.8% (35/45)**, far above the 53.3% floor and significant
against its own 1000-shuffle null at p <= 0.001. The falsifiable form of the artifact defence,
*if it reads sensorimotor cortex then deleting sensorimotor cortex must break it*, did not
hold. The pre-registered verdict is *a loss is suggested and not established at
n = 45*, because the paired McNemar that had to confirm it came in at p = 0.0703 on eight
discordant trials. Read §10.5 before quoting the 95.6% anywhere.

---

## 2. Why this problem matters (the motivation)

A **brain–computer interface** turns brain activity into commands, moving a cursor, a
prosthetic, a wheelchair, *without* muscles. The dream user is someone with paralysis or ALS
who can still *think* about moving but can't send the signal to their limbs. **Motor imagery**
(imagining movement) produces a brain signature very similar to real movement, so if a
computer can read that signature, it can act on the person's intent.

The core scientific bet this project rests on:

> Imagining a movement activates roughly the same sensorimotor cortex as performing it, and
> that activation is **spatially specific**: imagining your hands lights up a different patch
> of cortex than imagining your feet. EEG can pick up the difference from the scalp.

This project is the smallest honest demonstration that the bet pays off.

---

## 3. The neuroscience you need (and why each fact drives the code)

You don't need a neuro degree, but four facts explain *every* preprocessing choice in the code.

### 3.1 The motor homunculus: *why hands-vs-feet is separable*
The strip of cortex that controls movement (the primary motor cortex, plus the sensory strip
behind it) is laid out like a distorted map of the body. The "homunculus." Crucially:
- **Hands/fists** map to a region on the **side** of this strip (lateral), roughly under EEG
  electrodes C3 (right hand) and C4 (left hand).
- **Feet** map to the **top-center**, down in the midline crevice between the hemispheres,
  under electrode Cz.

Because hands and feet sit in physically different places, their scalp signatures *differ in
where on the head they appear*. That spatial difference is exactly what the model exploits.
**This is why the README calls it an "easy contrast"**: the two classes are far apart on the
homunculus, so they're easy to tell apart. Left-hand-vs-right-hand would be much harder because
both live in the same lateral strip, just mirrored across hemispheres.

### 3.2 Mu and beta rhythms: *why we band-pass to 8–30 Hz*
When a body region is **idle**, its patch of motor cortex idles in a synchronized oscillation:
- **Mu rhythm:** ~8–12 Hz
- **Beta rhythm:** ~13–30 Hz

### 3.3 Event-Related Desynchronization (ERD): *why imagined movement is detectable at all*
When you *engage* (or imagine engaging) that body part, the local neurons stop firing in
lockstep. The synchronized rhythm **breaks down**, so the power in the mu/beta band **drops**
over the active region. This drop is called **Event-Related Desynchronization (ERD)**.

So "imagine your fists" → mu/beta power drops over the *lateral* electrodes (C3/C4).
"Imagine your feet" → mu/beta power drops over the *central* electrode (Cz).

The counterintuitive bit: imagining a movement makes the rhythm go DOWN, not up. The idle
motor cortex hums, and getting ready to move interrupts the hum. We are detecting an
interruption in the hum, like a fan randomly stopping for a moment !

The model's entire job reduces to: **find where in the 8–30 Hz band the power dropped, and
map that location to a class.** Everything upstream in the pipeline exists to make this
signal cleaner and easier to read.

### 3.4 EEG is a spatial mixture: *why we need CSP*
Each scalp electrode doesn't see one brain source; it sees a blurry sum of *all* sources
(the skull smears everything). So the "power dropped over C3" story is never clean at a single
electrode. You need a method that **re-combines all 64 electrodes** into a few virtual channels
that maximize the class difference. That method is **CSP** (Section 8). Hold that thought.

---

## 4. The dataset: PhysioNet EEGBCI

- **Source:** the [EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/)
  on PhysioNet, loaded automatically through MNE's `mne.datasets.eegbci` helper.
- **Size:** 109 subjects, **64-channel** EEG, sampled at **160 Hz** (160 samples per second per channel).
- **Format:** EDF files (European Data Format. The standard container for clinical
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

**This project uses runs 6/10/14**: Task 4, imagined fists vs. feet. Using all three gives more
trials than one run alone. The harder left/right contrast in rung 7 uses **4/8/12**.

> An earlier version of this document twice told the reader to use **3/7/11** for the harder
> left/right contrast. Those are *executed* movement. Building an imagery result on them and
> writing it up as imagery is a silent over-claim, and it is precisely the kind a reviewer
> catches in one minute. The code always used 4/8/12; only this document was wrong.

### 4.1 Annotations: how the data knows what the subject was doing
Inside each EDF file are **annotations**: timestamped markers the experimenters recorded when
they showed the subject a cue. Three labels appear:
- **T0** = rest (do nothing). *We drop these.*
- **T1** = the cue to imagine **fists** (in these runs). We call this class **"hands"**.
- **T2** = the cue to imagine **feet**. Class **"feet"**.

These annotations are the *ground-truth labels*. Without them we'd have EEG but no idea what
the person was told to imagine, and supervised learning would be impossible.

> ⚠️ Gotcha worth knowing for a mentor conversation: T1/T2 mean *different things in different
> runs*. In the fist runs, 3/7/11 executed, 4/8/12 imagined, T1 = left fist and T2 = right
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
| **pyriemann** (`pyriemann==0.12`) | `riemannian.py`: covariance classification on the SPD manifold. |
| **PyTorch** (`torch==2.13.0`) | `eegnet_compare.py`, `regime_decomposition.py`: the CNN. Runs on Apple GPU via the `mps` backend when available. |
| **braindecode** (`braindecode==1.6.1`, with `skorch` and `einops`) | The EEGNet implementation and its scikit-learn-compatible `EEGClassifier` wrapper, which is what lets a CNN drop into `cross_val_score` alongside CSP+LDA. |

`joblib` is used directly too, `Parallel` fans the per-subject data loading across cores in the
sweep and the cross-subject rungs. The rest of `requirements.txt` (certifi, scipy, pooch, tqdm,
pillow, …) is transitive. `pooch` is worth a mention: it is the downloader MNE uses to fetch and
cache the dataset.

`.gitignore` keeps the virtual environment (`.venv/`) and Python bytecode caches out of git,
standard hygiene so the repo stays just source + results.

---

## 6. The architecture: a ladder of eleven rungs

The defining design choice of this repo is that it's built **rung by rung**. Each script is a
complete, runnable checkpoint that adds exactly one new idea on top of the previous one. This is
deliberate and worth articulating to a mentor: *it makes the pipeline debuggable and teachable,
because every stage can be run and inspected in isolation before the next stage is added.*

**The git history does not mirror it, and this document used to say it did.** That claim
("one commit per rung") is withdrawn: `git log --oneline extensions | wc -l` returns **26**
commits for eleven rungs, and it is wrong in both directions. Commit `d3edb50` adds
`sweep_subjects.py`, `harder_contrast.py`, `cross_subject.py` and `riemannian.py` in one go,
**four rungs in one commit**, while its message names only the left-vs-right contrast. Rung 11
spans **three** (`e07f209`, `57f4d03`, `c8e3326`), and both guard scripts land together in
`7b6fe9a`. Only rungs 1–3 were introduced in one commit and never touched again. The *scripts*
are one-idea-per-rung; the *history* is not, and since this section invites you to verify the
claim in the history, it had to be the history that gave way.

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
    ablate_channels.py       Produces the artifact-ablation table, and asserts the k/45 lattice.
                             Now includes the sensorimotor-DELETED arm the docs owed
    emg_proxy.py             Refits the pipeline at 40-75 Hz on the temporal ring, and turns
                             the null into a numeric sensitivity bound with an injection ladder
    permutation_design.py    Tests the tests. Measures each permutation null's false-positive
                             rate on data with provably zero information
    check_provenance.py      Fails on a percentage, p-value, r, or result-bearing count in
                             README/EXPLAINER that no script's stdout prints. Multipliers,
                             point-differences, µV and t-statistics are invisible to it.
```

The bottom block grew by three on 2026-07-25 and 2026-07-26, and the reason is worth stating
plainly because it is the criticism that produced them. A reviewer read the finished corpus and
said: *this project is trained to concede exposures with great precision and is not trained to
measure them.* Nine of ten findings named a hazard, named the exact test that would close it,
and then did not run the test. The three scripts above are three of those tests. Two came back
against the framing they were built to defend. That is what the section is for.

`decode_csp.py` is the *only* script you need to run to reproduce the headline result; it
re-does the work of rungs 1–3 internally. Every file is standalone. Nothing imports anything
else, so any rung can be run and read on its own.

**The shape of that list is the point.** Four rungs build the result and seven try to break it,
and **at least five of the seven succeeded**: rung 6's headline inference was backwards (the "27%
BCI illiteracy" reading), rung 7 found a gaze confound in this project's own data, rung 9 showed
no method dominates, rung 10 was measuring a network that turned out to be dead, and rung 11
found that rung 10's regime C was reading cue onset rather than imagery. A sixth retraction came
from rung 4's own topography claim, which is not an attack rung at all. A project that only
climbs is a demo. The rungs that found something wrong are the ones worth talking about, and §12
reports what each one actually returned.

> **This paragraph used to say "three of them succeeded."** That was an undercount, and it
> undersold the part of the project worth showing. Cross-checked against §12.1's round-one list,
> which contains six entries traceable to five distinct attack rungs plus the §8.3 topography
> retraction. Round two added nine more and round three twelve, but those came from auditing the
> documents rather than from running a rung, so they are not counted here.

The next sections walk each rung in depth.

---

## 7. Rung by rung

### Rung 1: `load_and_plot.py`: get the data, look at it

**Goal:** prove the data loads and eyeball the raw signal before doing anything clever.

What it does, step by step:
1. `matplotlib.use("Agg")` **before** importing pyplot. "Agg" is a non-interactive backend
   that renders straight to a file. This is what lets the script save a PNG on a headless
   machine (a server, CI) with no display attached. Order matters. You must set the backend
   before pyplot initializes.
2. `eegbci.load_data(subjects=1, runs=[6], update_path=True)` downloads run 6 for subject 1
   (or reads it from the `~/mne_data` cache on later runs) and returns the local EDF path.
3. `mne.io.read_raw_edf(path, preload=True)` loads it into a **`Raw`** object, MNE's container
   for a continuous recording (channels × time). `preload=True` pulls the samples into RAM now
   rather than lazily, which is required for the operations that follow.
4. **`eegbci.standardize(raw)`**: EEGBCI stores channel names with trailing dots, like `"Fc5."`.
   This renames them to the standard form (`"FC5"`) so MNE can match them to electrode positions.
5. **`raw.set_montage("standard_1005")`**: attaches real 3-D scalp coordinates to each channel
   using the standard 10-05 electrode layout. Without this, MNE knows the *numbers* but not
   *where on the head* each electrode sits, and you couldn't draw a scalp map later.
6. Prints metadata (sampling rate, duration, channel count) and saves the first 5 seconds of
   the first 10 channels to **`figures/raw_eeg.png`**.

**Why it exists:** sanity. If the download, channel naming, or montage is broken, you find out
here, before you've built a classifier on top of a silent bug.

### Rung 2: `epoch_trials.py`: cut the stream into labeled trials

**Goal:** turn one long continuous recording into a stack of short, labeled **trials** (called
**epochs** in EEG-speak), one per cue.

New ideas introduced:
1. **Concatenation.** Loads all three runs (6/10/14) and `mne.concatenate_raws(...)` stitches
   them end-to-end into one continuous `Raw`. More runs → more trials → a more trustworthy
   accuracy estimate.
2. **Events from annotations.** `mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))`
   converts the text annotations into an **events array**: a table of `[sample_index, 0, class_id]`
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
801 samples (5 seconds × 160 Hz + 1). This 21/24 split is the number to remember. It reappears
as the *chance level* later (a dumb model that always guesses "feet" would be right 24/45 = 53%).

**Why it exists:** classifiers need labeled examples, not a continuous stream. This is where raw
signal becomes a supervised-learning dataset.

### Rung 3: `filter_and_epoch.py`: isolate the motor rhythms

**Goal:** clean the signal so the mu/beta ERD story is what's left, *before* epoching.

The two new preprocessing steps (added between load and epoch):

1. **Average reference**: `raw.set_eeg_reference("average", projection=False)`.
   EEG voltages are always *relative* to some reference point; the raw recording's reference is
   somewhat arbitrary. Re-referencing every channel to the **average of all channels** gives a
   neutral, spatially balanced baseline. **CSP assumes this**: it reasons about how variance
   is distributed *across* channels, and that logic is cleanest when no single channel is the
   privileged reference. `projection=False` applies the reference directly to the data rather
   than storing it as a lazy projection.

2. **Band-pass filter 8–30 Hz**: `raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")`.
   This throws away everything *outside* 8–30 Hz. Why that band? Because (Section 3.2) that's
   exactly where the mu (8–12) and beta (13–30) motor rhythms live. Below 8 Hz you get slow
   drifts and eye movements; above 30 Hz you get muscle artifacts and line noise. Keeping only
   8–30 Hz means the model sees mostly *motor* signal.

**The most important subtlety in the whole repo**: *why filter the continuous signal, not the
epochs?* Digital filters produce garbage at the very start and end of whatever they're applied
to ("edge artifacts" / filter ringing). If you filtered each short 5-second epoch, those
artifacts would land *inside every trial*. By filtering the long continuous recording first,
the artifacts are confined to the very beginning and end of the whole recording, far from any
trial. `skip_by_annotation="edge"` additionally avoids filtering across the seams where the
three runs were concatenated. **This is a genuinely load-bearing decision, and a great thing to
be able to explain to a mentor** because it separates people who understand DSP from people who
copy pipelines.

`fir_design="firwin"` just specifies a well-behaved, linear-phase FIR filter (doesn't distort
the timing of the rhythms).

**Sanity check built in:** it re-prints the trial counts and asserts (in a comment) they should
still be 21/24, because *filtering changes the signal values, not the number of cues.* If the
count changed, something upstream broke.

### Rung 4: `decode_csp.py`: classify it, and prove it's real

This is the payoff. It repeats rungs 1–3, then adds the actual decoding. Two conceptual halves:

**(a) Feature extraction, crop to the imagery window.**
```python
labels = epochs.events[:, -1]                                   # 2=hands, 3=feet
train_data = epochs.copy().crop(tmin=1.0, tmax=2.0).get_data()  # 1–2 s after cue
```
Of the 5-second epoch, only the slice from **1 to 2 seconds after the cue** is used for
features. Why? The ERD (the power drop) takes a beat to develop after the cue and is most
stable a second or so in; the pre-cue and immediate-post-cue periods are noisier. Cropping to
a clean 1-second imagery window sharpens the class difference. (This is a tunable knob, see §11.)

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
spatial patterns as scalp maps → **`figures/csp_patterns.png`**. That plot is *interesting, and it is
not the credibility check*: §8.3 explains why, and it is the single most important correction
in this document.

### Rung 5: `evaluate_honestly.py`: is the number real, or an artifact of how I measured it?

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

Switching to `StratifiedKFold(n_splits=5, shuffle=True)`: every trial tested exactly once,
class balance held steady, gives the number this repo now publishes: **91.1%**, with a
1000-shuffle permutation test at **p ≤ 0.001**.

**The uncomfortable part, which is the actually interesting finding.** It would be tidy to say
the estimator change corrected an inflated number. It did not. Sweeping **100 cross-validation
seeds** through both estimators:

| estimator | mean | range | where seed 42 falls |
|---|---|---|---|
| ShuffleSplit (retracted) | 93.6% | 87.8–98.9% | 49th percentile |
| **StratifiedKFold (published)** | **93.8%** | 88.9–97.8% | **3rd percentile** |

The two estimators **agree in expectation to about 0.2 points**, and stratified k-fold is the
*higher* of the two: over 100 seeds ShuffleSplit averages 93.6% and StratifiedKFold averages
93.8%. So the switch of estimator did not lower the headline. In expectation it raises it
slightly. The 94.4 to 91.1 move is **seed placement**: seed 42 lands at the 49th percentile of
the retracted estimator (94.4%) and at the 3rd percentile of the published one (91.1%), 2.7
points below that estimator's own 93.8% mean. Seed placement accounts for 0.8 + 2.7 = 3.5
points; the estimator offsets 0.2 in the opposite direction, leaving the 3.3-point drop on the
page. The switch is still the right call, for coverage and stratification reasons, but
presenting it as an integrity correction would be its own small dishonesty. The published 91.1%
is a **conservative draw** from an 88.9–97.8% distribution, and that is how it should be
described out loud.

> **Withdrawn, 2026-07-25.** This paragraph used to read *"So the 94.4 → 91.1 drop is roughly
> **2.7 points of seed luck and 0.6 points of real estimator change**."* That decomposition is
> wrong and is withdrawn. The 0.6 came from `94.4 − 93.8`, which differences seed 42's
> **ShuffleSplit draw** against StratifiedKFold's **mean**. That silently moves 0.8 points of
> old-estimator seed luck onto the estimator's account and flips the estimator term's sign: the
> estimator is worth −0.2 points, i.e. it *raises* the expectation, and cannot have contributed a
> positive share of a drop. It survived review because `0.6 + 2.7 = 3.3` sums to the observed
> total. The total was right; the split was wrong. The corrected decomposition is seed placement
> 3.5 points, estimator −0.2 points. Backing: `evaluate_honestly.py` section 6, which prints the
> two estimator means, seed 42's percentile under each, and the signed estimator term. Run the
> script to regenerate it; the earlier pointer here named a path under `.provenance_cache/`, which
> `.gitignore` excludes, so it resolved to nothing in a clone.

> **One caveat on the percentile column, which is this rung demonstrating its own thesis.** Those
> ranks count seeds *strictly below* seed 42. Because both estimators are quantized, the
> stratified one to 1/45, the 100 seeds land on only a handful of distinct values and many tie
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

### Rung 6: `sweep_subjects.py`: does it hold across 109 people?

**Goal:** turn a claim about *this subject* into a claim about *the method*.

The identical pipeline runs on all 109 subjects, computing **chance per subject**: class
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
> line. Nothing in this rung's conclusion rests on the count. The conclusion below is a
> *population*-level comparison against the pure-noise expectation, which is unaffected. The row is
> relabelled rather than removed so the gap between "counted" and "tested" is visible.

**Subject 1's 91.1% is the 91st percentile.** Say that out loud before anyone has to ask.

**The inference this rung originally drew was backwards, which is worth more than the numbers.**
The first write-up read "27% of subjects at or below chance" as a **BCI illiteracy rate**, and
the coincidence that 27% sits inside the literature's familiar 15–30% band made it feel like
replication. Both halves were wrong:

- **The direction.** This pipeline's own permutation null is 50.7% ± 8.5%. Under a **global null
  in which nobody has any signal**, the expected fraction landing at or below their own chance
  line is **~55% (59/109)**. Observed: **30/109 = 27.5%**, which `sweep_subjects.py` prints at
  zero decimals as **28%**: the same count, not a second measurement, and worth writing out
  because the claim being refuted is stated as "27%" and 27.5 straddles the two roundings.
  (`inferential_stats.py` §10 recomputes the null per subject as an exact binomial rather than a
  simulation and gets **54.0%, 58.8/109**, a point below the simulated figure quoted here; the
  approximation is stated there, since 5-fold folds are not independent Bernoulli draws.)
  Seeing *half* the noise-only rate is
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

### Rung 7: `harder_contrast.py`: how I found a gaze confound in my own result

**Goal as stated:** measure what the method costs when the classes move closer together. Left
fist versus right fist share the same sensorimotor strip, mirrored, instead of sitting
centimetres apart the way fists and feet do.

**Runs 4/8/12**: imagined left vs. right fist. Not 3/7/11, which are *executed* movement.

Subject 1 scores **73.3%** (p = 0.0020), against a majority-class floor of 51.1%. The original
write-up reported the gap to fists-vs-feet as "what a harder contrast costs: 17.8 points."
Nearly everything about that sentence was wrong.

- **It is n=1.** Rung 6 had just swept 109 subjects and this rung silently reverted to one.
- **The two conditions come from different recording runs**, so "harder contrast" cannot be
  separated from "different session."
- **The window was the joint maximum, and the script now proves it.** Sliding the 1-second crop
  across the same trials gives **66.7 / 55.6 / 73.3 (used) / 64.4 / 46.7%** for windows starting
  at 0.0 / 0.5 / 1.0 / 1.5 / 2.0 s. That is a **26.7-point range** across *overlapping* windows
  of the same data, larger than the 17.8-point gap the original sentence advertised as a
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
of the screen and leaves it there** until the subject relaxes, so a lateralized visual stimulus
is present for the entire decoding window. On subject 1 the frontopolar asymmetry
`mean(Fp1,AF7,AF3) − mean(Fp2,AF8,AF4)` is:

| window | left cues | right cues | Welch t | p |
|---|---|---|---|---|
| **0.0–1.0 s (the cue)** | **+11.89 µV** | **−12.99 µV** | **+7.71** | **3.7e-09** |
| 1.0–2.0 s (the decoding window) | −2.99 µV | +2.43 µV | −1.51 | 0.14, n.s. |
| 0.0–4.0 s (whole epoch) | +2.83 µV | −2.58 µV | +5.10 | 7.6e-06 |

The sign flips with the cue side, the effect is largest in the **cue** window, and it is driven by
AF7/AF3 against AF4/AF8, the electrodes nearest the eyes. That is the signature of eyes moving
to a target. (The 1–2 s row reverses polarity, but it is not significant; quoting that reversal as
real would be reading a coin flip.)

**A methodological trap worth naming, because the obvious statistic hides this effect.** Averaging
all eight frontopolar channels gives a total null. A gaze deviation is *antisymmetric* across the
midline, so pooling both hemispheres cancels it exactly. The confound is only visible in the
lateral difference.

**A second trap, and it is the reason the channel ablation undertests this.** CSP features are
log-variance, and a sustained gaze deviation is a steady DC shift, which variance is nearly blind
to. Swapping the *feature* rather than the channels is what finds it. A **frontopolar
mean-amplitude** decoder scores:

| window | accuracy | p |
|---|---|---|
| 0.0–1.0 s (cue period) | **86.7%** (± 8.3%) | **≤ 0.001** |
| 1.0–2.0 s (matched to headline) | 62.2% (± 20.6%) | 0.0939 |

> **The evidence originally offered for this confound was itself window-shopped, and that
> correction matters more than the confound.** This section used to claim that "a decoder using
> only 8 frontopolar channels at 0.5–5 Hz reaches **73.3%**: numerically identical to the
> 64-channel headline." At **matched** settings, same trials, same folds, same 1.0–2.0 s window,
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
> running the frontopolar decoder on the whole **0–4 s epoch**: four times longer than the
> headline's 1-second crop, so the "numerically identical" match was an artifact of comparing two
> different experiments. Across windows that same decoder gives 66.7 / 53.3 / 66.7 / 73.3%, and
> the published figure was its maximum.
>
> Also withdrawn from this passage: **"+4.41 µV on left cues and −3.69 µV on right cues
> (t = 5.12)"**: close to the whole-epoch row above (t = +5.10) but with different microvolt
> values and, more importantly, **no stated window**; the correct headline statistic is the cue
> window, t = +7.71. **"significant in 11 of 16 subjects and sign-consistent in 15
> (p = 0.0005)"** and **"group-wide the ocular decoder averages 53.9% against the pipeline's
> 57.5%"**: no script computes either. And **"mu alone 73.3%, beta 64.4%, combined 73.3%, so this
> is an alpha-band decoder"**: those were full-64-channel numbers presented as if they supported
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

### Rung 8: `cross_subject.py`: does it transfer to a person the model has never seen?

**Goal:** the result a deployed BCI actually needs. Everything up to here is *within*-subject,
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
leakage. That assertion is **definitionally true and can never fail**: it restates the
definition of the splitter. It now carries an honest comment about what it can and cannot catch.
A guard that cannot fail is worse than no guard, because it reads as protection in a review.

### Rung 9: `riemannian.py`: does a stronger classical method beat it?

**Goal:** answer the failure rung 8 measured. CSP learns spatial filters tuned to the training
population's anatomy, and a new skull shifts everything. Riemannian methods attack that
directly. A trial's spatial covariance matrix is **symmetric positive definite**, SPD matrices
live on a curved manifold rather than in flat space, and treating them as flat feature vectors
distorts the distances between them. Measuring distance *along* the manifold respects the actual
geometry, and it is the current state of the art for classical BCI.

Four pipelines, MDM and Tangent Space, each on all 64 channels and on the sensorimotor subset,
run against the CSP+LDA baseline on **identical LOSO folds**.

**It lost.** The honest reading of *how* it lost is much narrower than what was first written:

| comparison | paired p |
|---|---|
| MDM-64 | **0.005** |
| MDM-motor | 0.200 |
| TS-64 | 0.349 |
| TS-motor | 0.330 |

Only MDM-64 is significant; the other three confidence intervals span zero. The minimum
detectable difference at 80% power with n=20 runs **5.7 to 7.9 points** depending on the
comparison (MDM-mot 5.68, TSLR-64 6.83, MDM-64 7.17, TSLR-mot 7.88), and three of the four deltas
are smaller than that. Only MDM-64 survives Holm across the family of four (0.005 → 0.019); the
p-values above are the uncorrected ones this rung publishes.

> Two caveats on that MDE. It was previously quoted as "about 5–6 points," which is optimistic,
> only one of the four comparisons is that tight, and rung 8's is nearly 10 (9.92). A later
> revision quoted "roughly **6 to 8** points," which rounds the low end up: the tightest is 5.68.
> And like the four paired p-values above, this was for two rounds **computed in prose and by no
> committed script**: `riemannian.py` persists nothing but a PNG and prints no test.
> `inferential_stats.py` §2 now computes all of it from `riemannian_perfold.json`, a captured copy
> of the per-fold scores; the remaining gap is that `riemannian.py` itself still persists nothing,
> so the copy cannot be regenerated without editing the source (§12.2 item 7).

**Retracted:** "no method dominates, and the best method is subject-specific." Per-subject
optimality requires a **crossover** subject × method interaction. The *ranking* of methods has to
change from subject to subject. Non-additivity in this 20 × 5 layout **is** detectable
(Tukey 1-df **F = 13.4627 on 1 and 75 df, p = 0.0005**), but it is a *fan* rather than a crossover,
and it is produced by one arm: leave MDM-64 out and it collapses to **F = 1.17, p = 0.2847**, while
leaving out any of the other three barely moves it (11.47 to 13.81). A fan magnifies an ordering;
it does not reverse one. The homogeneity test that *can* see a crossover, on the
CSP-vs-MDM-motor difference the "no method dominates" line is actually about, is null and
underpowered: **χ² = 13.33 on 19 df, p = 0.821**, MDE **5.68 points**. Neither result supports
per-subject optimality, so the positive claim stays withdrawn.

> **Correction to that χ², 2026-07-25 (second pass).** This line used to read
> *"none is **detectable** here (χ²₁₉ = **13.0**, **p = 0.84**)."* Both figures are withdrawn.
> `inferential_stats.py` §4 now computes the family: the pooled-binomial Cochran form gives
> **13.33**, the per-arm form **13.58**, and neither is 13.0. The published pair was at least
> self-consistent, `chi2.sf(13.0, 19) = 0.839`: which is exactly how an unproduced statistic
> survives review. More importantly the *sentence* was wrong, not just its digits: the
> design-appropriate omnibus test for an unreplicated 20 × 5 layout is Tukey's 1-df
> non-additivity, with one observation per cell the full 76-df interaction is confounded with
> error and cannot be tested at all, and it **rejects** additivity, p = 0.0005. "No interaction
> is detectable" was false for the five-arm design and true only for the one pair the 19-df test
> looks at.
> **The retraction is not in play either way.** "The best method is subject-specific" was a
> positive claim withdrawn for want of support, and a significant *fan* is not that support: what
> the leave-one-out rows locate is a single arm, MDM-64, whose per-subject scores do not track the
> others, so method differences widen on the subjects where anything works at all. That is a floor
> effect on one pipeline, not per-subject method selection. A reader who wants the stronger
> statement, that the fan is not merely a link-function artifact, should note that no committed
> script yet runs the test on transformed accuracies, so it is not asserted here.

> **A flag on that χ², added 2026-07-25 and RESOLVED the same day by `inferential_stats.py`.**
> Its **df = 19 = n − 1** over the 20 subjects. A subject × method interaction across **five**
> pipelines and 20 subjects
> would carry **df = (20 − 1)(5 − 1) = 76**. So whatever this statistic tests, the degrees of
> freedom say it is not a five-way interaction, and it therefore cannot license the sentence
> "no subject × method interaction is detectable" **across all five methods**: at most across a
> two-way contrast, which test produced it cannot be determined, because no committed script
> computes it and `riemannian.py` persists no per-fold array. The retraction it supports stands on
> other grounds, a positive claim ("the best method is subject-specific") is withdrawn for want
> of evidence, which does not require this test to be right. But the statistic itself should not
> be quoted anywhere until a script produces it. See §12.2 item 7.
>
> **The flag was right and is now discharged.** `inferential_stats.py` §4 computes both families
> from `riemannian_perfold.json` and prints them side by side. The df objection holds exactly as
> stated: the 19-df statistic is a two-arm homogeneity test, and the five-way question needs the
> 1-df Tukey test, which is what an unreplicated 20 × 5 layout affords. What the flag did not
> anticipate is that running the right test would *reject* rather than fail to reject. The flag
> assumed the conclusion would survive a better statistic and only the citation needed fixing.

The
9-8-3 win/loss/tie split is consistent with a *uniform* −2.6-point difference plus 45-trial noise. "No method dominates" is indistinguishable here from "this
experiment cannot tell these methods apart," and only the second is supported.

Two further caveats surfaced on review. The script selects its best pipeline by **max mean over
the same test folds it reports from**, which is selection on the test set. And framing the
comparison as "2080 parameters versus a classical baseline" ignored that
`Covariances(estimator="oas")` is a shrinkage estimator *built* for exactly the small-sample
regime, while the CSP baseline it loses to runs with `reg=None`.

**That asymmetry runs the opposite way to charity, and it is worth stating plainly.** Shrinkage
sits on the **Riemannian** side only (`riemannian.py:131/135/140/144`); the CSP baseline is
`reg=None` (`riemannian.py:127`). *(Locators re-derived 2026-07-25 18:0x by re-running
`grep -n 'Covariances(' riemannian.py`. This line previously read `121/125/130/134` and `:117`,
which were correct until `riemannian.py` gained a docstring at 17:36 and everything below it
shifted by ten lines. The set before that was `121/126/131/135`, withdrawn earlier. The fact has
not changed once; the locator has changed three times.)* Regularizing only one arm favors **that** arm. So the baseline
won a comparison that was tilted against it. Shrinkage is not a nicety on the Riemannian side
either, `raw.set_eeg_reference("average")` costs one degree of freedom, which makes every 64×64
covariance rank-63 and singular by construction, so the two 64-channel Riemannian pipelines cannot
run on a plain sample covariance at all. Some regularization is **required** for them to exist;
OAS is one of several ways to supply it.

### Rung 10: `eegnet_compare.py`: does a CNN beat designed filters?

**Goal:** the question is not "is deep learning better" but **at what sample size does
*learning* the filters start to beat *designing* them**. EEGNet is structurally doing what CSP
does, a temporal convolution discovers frequency filters, then a depthwise spatial convolution
learns a spatial filter per temporal filter, except end to end.

| regime | data | CSP + LDA | EEGNet |
|---|---|---|---|
| **A** within-subject, subject 1 | 45 trials | **91.1%** | 82.2% |
| **B** cross-subject LOSO, narrow band | ~900 trials | 59.4% | **60.1%** |

At n=45 the CNN loses by **8.9 points**. Pooled across 20 subjects the two are level.

**Three things keep that from being the "learned filters need volume" result it was written up as.**

First, 8.9 points on 45 trials is **41/45 against 37/45, four trials of net difference**, and
this rung runs no significance test on it. `inferential_stats.py` §6 re-runs experiment A to
recover the per-trial predictions and does run one: the two models disagree on **10** trials, not
four, CSP-only correct on **7**, EEGNet-only on **3**: and exact two-sided McNemar on that split
gives **p = 0.344**. The Wilson intervals also overlap across most of their range
(**[79.3%, 96.5%]** against **[68.7%, 90.7%]**). The defensible statement is directional:
*CSP scores higher, by an amount this experiment cannot resolve.*

> **Correction, 2026-07-25.** This paragraph used to read *"**Four discordant trials** out of 45
> is not a distinguishable difference."* Four is the **net** difference, `b − c`; the discordant
> count is `b + c` and it is 10. The two are not interchangeable and the p-value depends on the
> split rather than on the difference: any `(b, c)` with `b − c = 4` is consistent with 41/45
> against 37/45, and the maximally nested split `b = 4, c = 0` would give **p = 0.125** while the
> measured 7/3 split gives **0.344**. Deriving a McNemar p from two accuracies is arithmetic on an
> assumption about agreement that only the predictions can settle, which is why no such p was
> quoted here before one existed. The conclusion is unchanged and slightly strengthened: the
> difference is even less resolvable than the wording implied. One caveat on the table itself,
> which `inferential_stats.py` states in its own output: MPS kernels are not bit-reproducible, so
> the EEGNet half of the 2×2 can shift by a trial or two between runs. The CSP half cannot.

Second, **sample size and optimisation budget are confounded across the two regimes.** `N_EPOCHS`
is a fixed 100 with `BATCH_SIZE = 32`, so regime A's 36-trial training fold yields one batch per
epoch, about 100 gradient steps, while regime B's ~855-trial fold yields 26 batches per epoch,
roughly 2600 steps. The regimes differ in how much data the network saw *and* in how long it was
allowed to train, and the write-up credited only the first. Whatever the −8.9 measures, it is not
cleanly "what happens at n=45."

Third, and unaddressed here: the EEGNet numbers are **single-seed**. There is no seed sweep, and
the printed "± 11.3%" is a spread across five 9-trial folds, another rung of the same
quantization ladder §10.2 disowns, not an interval on the estimate.

**This rung was measuring a dead network, and it took adversarial review to catch it.**

MNE returns data in **volts**. The signal standard deviation is about 1.3e-5, so the variance is
about **1.6e-10**. braindecode's EEGNet normalizes with `BatchNorm2d(eps=1e-3)`: a variance
**seven orders of magnitude below eps**. So batch norm divides by `sqrt(var + eps)` ≈
`sqrt(1e-3)` = **0.0316** instead of by the signal's own sigma of ~7e-6: its output comes out
about **4500× too small** where it should come out at 1. Normalization never engages.

Batch norm does not do *nothing*, though, and getting this right matters for saying it out loud.
Each BN stage renormalises, so the deficit decays down the stack rather than compounding. **The
figure that is actually established is the deficit at the first BN: ~4500×.** Everything past
that point depends on a recovery model this repo never measured, and the model it used does not
produce the number it reported. Under that model, with each of the three BN stages recovering
~31.6×, the deficit runs 4500× at the input to BN1, **142×** after one stage, **4.5×** after two, and
**0.14×** after three, which would mean the logits end up about 7× too *large*. There is no
integer number of stages at which the chain lands on the "53×" this section used to claim
(solving `4500 / 31.6ˣ = 53` needs x = 1.29 stages). So the defensible statement stops at the
first BN, and the end-to-end figure is withdrawn.

**The stage count is no longer a guess. It is measured, and the recovery model was wrong.**
`inferential_stats.py` §7 puts forward hooks on every `BatchNorm2d` and reads the scale off a
real no-grad forward pass, five seeds, dropout disabled. The per-stage deficits are **4744×**
(sd 378) at `bnorm_temporal`, **459×** at `bnorm_1` and **59×** at `bnorm_2`, and end to end the
classifier input is **84×** (sd 11) too small with the **logits 102×** too small. So recovery per
stage is roughly 10× and then 8×, not 31.6×, and the chain does not keep decaying to nothing: it
flattens around two orders of magnitude. The first-BN figure survives the measurement, ~4500× by
hand against 4744× measured, inside one seed-to-seed sd, and the end-to-end figure is now a
measurement rather than a model.

Against the measured gap the training argument is much stronger than the version below it.
`inferential_stats.py` §8 trains one fold of experiment A and reads the final layer at both ends:
init sd **0.0645**, and after 100 epochs of AdamW at lr=1e-3 the weight sd has moved **0.0468**
while the mean absolute per-weight change is **0.1020**: two quantities that have both been
called "the travel" and that differ by **2.2×**, so a margin quoted without naming its definition
cannot be checked. Closing the measured deficit at the logits needs a final-layer sd of **6.568**,
a required travel of **6.504**: a **64×** shortfall on the mean-|dw| definition, **139×** on the
sd definition.

> **Correction, 2026-07-25 (third layer on this passage).** This paragraph used to read
> *"Closing a **4.5×** gap means growing the final layer's weights from an init standard deviation
> of ~0.065 to ~**0.29**, a travel of about **0.23** … a **~2.3× shortfall**, which makes 'the
> network cannot train out of it' **plausible but not established**."* Every figure in that
> sentence is withdrawn. The arithmetic was right, 0.0645 × 4.5 = 0.290, travel 0.226, and
> 0.226 / 0.102 = **2.2** (the published 2.3 came from rounding the inputs first), but it was
> arithmetic on the **31.6×-per-stage recovery model**, and that model is now measured and does
> not hold. The residual gap is ~102×, not 4.5×, so the shortfall is 64× rather than 2.3×.
> Note which way this cuts: the previous layer of correction *downgraded* "the network cannot
> train out of it" from established to plausible on the strength of an assumed 4.5×, and the
> measurement puts the margin back into the same order as the withdrawn 53× chain implied. The
> conclusion is not thereby restored, because the reason it was downgraded stands independently:
> **a scale argument is not a training experiment.** This bounds the optimizer's reach over 100
> epochs; it does not prove no other route exists, and the direct evidence. The degenerate
> single-class prediction, is what the guard at `eegnet_compare.py:190` now refuses to reproduce.

> **Correction to the mechanism, layer two, 2026-07-25.** This paragraph used to say activations
> "stay near 1e-8" and that recovery "would require final-layer weights around 1e8." Both figures
> are wrong and both are withdrawn: nothing in the network is anywhere near 1e-8, and the smallest
> activation standard deviation is ~7e-6. The 1e-8 was the *variance* of the first BN's output
> being mistaken for an activation magnitude, and `1e-8 × 1e8 = 1` then generated the bogus weight
> figure. One error, stated twice.
>
> **The correction block that replaced them was itself wrong twice, and that is withdrawn too.**
> It said the two figures were *"wrong by about six orders of magnitude."* Neither is: the
> activation error is `log10(7e-6 / 1e-8)` = **2.85 orders**, and the weight error is
> `log10(1e8 / 3.4)` = **7.47 orders**. One number, "six", was asserted for two different errors
> that are four and a half orders apart. And the block's own explanation cannot hold together with
> its own premise: if the activation standard deviation is 7e-6 then that layer's variance is
> **4.9e-11**, not 1e-8, so the 1e-8 cannot be "the variance of the first BN's output" either.
>
> The block also said *"so was the conclusion."* That was premature, for the reason set out
> above: the conclusion was computed from the 53× that is now withdrawn, and at the model's own
> 4.5× it is a 2.3× shortfall rather than a 33× one. The **"seven orders of magnitude"** headline
> (`1e-3 / 1.6e-10 ≈ 6e6`) and the **4500×** deficit at the first BN (`0.0316 / 7e-6 ≈ 4517`) are
> the two figures in this section that check out, and they are the only two stated as findings.
>
> **None of these numbers was produced by any script, and that is what let this section be wrong
> for two rounds.** `grep` for `4500`, `31.6`, `1.6e-10`, `1.3e-5` and `7e-6` across all 12 files
> in `.provenance_cache/` returns zero hits, and `check_provenance.py` cannot see multipliers at
> all (see the blind-spot table at the end of §12.1), so this whole section was hand-arithmetic
> that the repo's own guard is structurally unable to check.
>
> **Closed 2026-07-25 by `inferential_stats.py` §§7–8**, which measures the scale chain instead of
> modelling it: signal sd **1.265e-05**, variance **1.599e-10**, eps/variance **6.25e+06**, and
> per-stage deficits **4744× / 459× / 59×** with **84×** at the classifier input and **102×** at
> the logits. **Both figures this block certified as checking out survive**: the seven orders of
> magnitude (measured 6.25e+06) and the ~4500× at the first BN (measured 4744×, sd 378 over five
> seeds), and the 31.6×-per-stage recovery model that everything downstream rested on does not.
> The remaining hand-arithmetic here is now checkable against printed output rather than only by
> redoing it.

The failure was **silent, and it looked like a result**:

| | accuracy | predicted class counts |
|---|---|---|
| as originally committed (volts) | 53.3% | **[0, 45]** |
| rescaled to microvolts | **82.2%** | [21, 24], matches truth exactly |

> **The first row is a historical record, not a reproducible measurement.** That configuration is
> **no longer reachable**: `eegnet_compare.py:190` now asserts `var > 1e3 * BN_EPS` before
> training and halts if the units are wrong, which is the guard described two paragraphs down. No
> script on disk can produce the 53.3% / [0, 45] row today, and `check_provenance.py` cannot
> distinguish it from a live claim because 53.3% is *also* subject 1's majority-class rate and so
> matches other stdout for an unrelated reason. It is kept because deleting the evidence of a
> failure to make a provenance check pass would be exactly the move this repo exists to argue
> against. Read it as: *this is what the rung printed before the bug was found.*

The dead model **predicted a single class for all 45 trials**. Its 53.3% was the *majority-class
rate*, not chance performance, and "a CNN performs at chance on small data" is an entirely
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

### Rung 11: `regime_decomposition.py`: what did rung 10's third experiment actually measure?

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
regime B's numbers exactly, CSP 59.4%, EEGNet 60.1%. That confirms the rewired harness, meaning
the new cell config, the refactored pooling and scorer, and the checkpointing wrapper still measure what
rung 10 measured, which is what licenses reading the cross-cell deltas below as statements about
rung 10's claim.

> **Correction to how that control was described.** It used to read "an independent
> reimplementation landing on the same values is the evidence that this harness measures what the
> rung it audits measured." It is **not** an independent reimplementation: `load_subject`,
> `make_eegnet` and `seed_everything` are shared with `eegnet_compare.py`, with the same subjects,
> runs, seed, epoch count, batch size, CSP and LDA settings. Agreement therefore demonstrates that
> the plumbing did not perturb the measurement, which is a regression check and **not** validity. A
> shared bug would reproduce perfectly.
>
> **A second layer, 2026-07-25: the control cell reproduces and the audited cell does not.**
> `narrow-short` matches regime B exactly, as stated. But the `original-C` cell is the one this
> whole rung exists to audit, and it does **not** match rung 10: this table's **63.0%** against
> the **63.3%** that `eegnet_compare.py` prints for the identical configuration (section C of its
> stdout, the `EEGNet (wide)` row), a gap of **0.3 points**: 3 trials of 900, since the mean runs
> over 20 leave-one-subject-out folds of 45 trials each. CSP agrees to the digit (51.4% in both),
> so the disagreement is the CNN's, not the data's or the folds'. Do not expect to reproduce
> either figure exactly: that a byte-identical configuration moves between runs is the finding
> here, not a fault in the capture. Nothing
> in the rung's *conclusion* turns on 0.3 points, regime C is the cue-window effect either way,
> but the sentence above licenses reading the cross-cell deltas on the strength of a control that
> holds in the cell nobody was worried about and fails in the cell under audit, and that ordering
> is worth stating rather than leaving for a reader to find.
>
> **Where these numbers came from, which is not the 2026-07-25 cold run.** `original-C` and every
> other cell in the table were read from `regime_decomposition.json`, dated **2026-07-23**. The
> 07-25 run of this rung printed *"Resuming: 7 cell(s) already on disk"* and skipped every cell as
> cached, so **rung 11's figures throughout this section are 2026-07-23 checkpoint values and were
> not freshly reproduced.** The checkpointing that makes this rung killable and resumable (§13)
> also lets it defeat `check_provenance.py`'s source-hash cache invalidation. Rerunning it cold
> means deleting `regime_decomposition.json` first.

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
start moves from 1.0 s to 0.0 s, 95% CI [+2.0, +16.4], **p = 0.015**. It accounts for essentially
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
credited.* The obvious explanation, a CNN's temporal convolutions can exploit a phase-locked
cue-evoked response, while CSP's log-variance band power is close to blind to one, is an
**interpretation, and this project's recurring failure has been inventing the mechanism in the
same breath as the number.** So rung 11 tests it instead of asserting it, with a sixth cell that
decodes the **cue window alone (0–1 s)**.

> **What that window does and does not contain, corrected 2026-07-25.** This document used to call
> 0–1 s "the cue window, **which contains no imagery at all**," three times over. That is an
> assumption, not a measurement, and the script that produced the cell says so in its own docstring
> (`regime_decomposition.py:35-39`): *"Calling 0-1 s 'the cue window, no imagery in it' is an
> assumption, not a measurement: the subject begins imagining AT the cue, so 0-1 s holds the visual
> evoked response AND the first second of imagery, and either one could be carrying the score."*
> The honest description is **cue-onset window**: it contains the visual evoked response *and* the
> first second of imagery, and this design cannot separate them, because in EEGBCI both begin at
> the same instant. The false version is what reached the prose while the true one sat in the code
> the prose was describing.

**The confirmation.** Decoding the cue-onset window on its own (0–1 s). Pooling 20 subjects
balances the classes, so the reference here is near 50% rather than subject 1's 53.3%:

| model | cue-onset window alone | against chance |
|---|---|---|
| CSP + LDA | 53.7% | 95% CI [50.6%, 56.8%], p = 0.023 |
| **EEGNet** | **61.1%** | 95% CI [57.3%, 64.9%], **p < 0.0001** |

> **Two caveats on every p-value and interval in this rung, both of which push the same way.**
> This document used to state neither, while the script that computes them stated both in its own
> comments. That is backwards, so they are lifted into the prose here.
>
> **(a) The null is 0.5, not 50.1%.** `regime_decomposition.py:423` is `ttest_1samp(s, 0.5)`. An
> earlier version of this section told the reader the tests ran against the pooled majority-class
> floor of **50.1%**. They do not. The gap is a tenth of a point and it changes no conclusion in
> this rung, but a document that says which reference a test used should be right about it, and
> `check_provenance.py` allowlists `pct:50.0` by *value*, so the exemption applies regardless of
> whether 50% was the correct floor in context. The script names the discrepancy rather than
> quietly closing it, and so does this note.
>
> **(b) Twenty LOSO folds are not twenty independent observations.** Every pair of training sets
> shares 19 of their 20 subjects, so fold-to-fold variance underestimates sampling variance and
> **every p-value and every confidence interval in this rung is anti-conservative.** That applies
> to roughly fifteen tests across §7's rung 11 and the tables in this section. This repo already
> flags exactly this hazard for the Wilson interval on n=45, *"it treats 45 cross-validated
> predictions as independent draws from one model when they come from five"* (§10.2, `README.md:47`)
>, and the same objection applies with more force to a 20-fold t-test on overlapping training
> sets. Read these numbers as **ordering evidence, not as calibrated tail probabilities.** No
> correction for this is applied anywhere in the repo, and applying one properly would need a
> resampling scheme this project has not built.

> **The rung's own stated prediction half-failed, and this document did not say so.**
> `regime_decomposition.py:133` keeps the prediction verbatim as it was written: *"EEGNet should
> score above chance here **and CSP should not**."* CSP scores **53.7%, p = 0.023 against 0.5**:
> above chance. The script now reports this itself (`:448-462`, *"AND HALF THE PREDICTION THIS
> CELL WAS BUILT ON FAILED"*); the prose here had not caught up, which is the wrong direction for
> that gap to run. The finding that survives is comparative, not categorical: **both** models read
> the cue onset, EEGNet far more of it. That is enough to relocate regime C's effect, and it is
> not the clean dissociation the prediction asked for.

EEGNet decodes the first second after cue onset at **61.1%**. And the decisive comparison: adding
the entire three-second imagery window on top of it buys **+1.9 points, 95% CI [−2.8, +6.6],
p = 0.41**: indistinguishable from nothing. CSP is not helped either (−2.2 points, p = 0.20).
That is the finding: **the imagery window adds nothing the cue-onset second did not already
carry.**

So EEGNet's 63.0% in regime C is, within measurement error, **entirely available from the cue
period alone**.

**And the control that check itself needed.** "EEGNet decodes 61.1% on the cue window" is only
about the cue if the same models score *nothing* on a window with no cue in it. A seventh cell
decodes the **second before the cue** (−1.0 to 0.0 s), band- and length-matched to the cue cell so
the two differ only in which side of the cue they sit on. That second is task-free: task onsets
are 8.3 s apart with a 4.2 s rest before each.

| cell | CSP + LDA | EEGNet |
|---|---|---|
| **pre-cue** (−1.0–0.0 s) | 47.7% (t = −1.69, p = 0.108, at chance) | 51.8% (t = +1.39, p = 0.181, at chance) |
| **cue** (0.0–1.0 s) | 53.7% (t = +2.48, p = 0.023) | 61.1% (t = +6.18, p < 0.0001) |
| paired difference, same 20 folds | **+6.0 pts** (t = +2.85, p = 0.0103) | **+9.3 pts** (t = +5.43, p < 0.0001) |

Neither model reaches significance before the cue, and both clear it after, so the effect is
located **post-cue** rather than in subject leakage, drift, or a harness bug. What licenses that
reading is the **paired** row, not the pre-cue row: +6.0 points for CSP and +9.3 for EEGNet across
the same 20 folds.

> **"The control passes" was too strong, and it is withdrawn.** This paragraph used to end *"Both
> models sit at chance before the cue and above it after ... **The control passes**."* A failure
> to reject is not a pass, and here the gap between the two is large enough to matter. The pre-cue
> EEGNet interval is **95% CI [49.1%, 54.5%]** (point estimate 51.8%, p = 0.181). **Provenance,
> stated because this document has been burned by not stating it:** that interval is computed in
> this prose, not by any committed script. It was derived from the 20 per-fold values stored in
> `regime_decomposition.json` under `pre-cue.eegnet`, as a t-interval on the fold mean, and the
> point estimate and p-value it sits with *are* printed by `regime_decomposition.py`. It belongs
> to the class §12.2 item 7 tracks, and closing that item means having the script print the
> interval alongside the p-value it already prints. **Its upper bound of 54.5% is above the
> cue-window CSP point
> estimate of 53.7%, which the table two rows up reports as significant at p = 0.023.** So the
> control cannot exclude a pre-cue effect the size of an effect the same rung reports as a
> finding. That interval appeared nowhere in this document until now, no power analysis was run,
> and no equivalence test was run, and an equivalence test is what "at chance" actually requires.
> §14 item 6 already warns against exactly this move, asserting a null the experiment has too
> little power to establish, and this control was doing it three sections earlier. The honest
> statement is the narrower one now in the paragraph above: **the paired increase is what carries
> the conclusion; the pre-cue cell is consistent with chance and underpowered to establish it.**

> Two limits on what it establishes, both of which the code now carries as comments. **(a)** A
> pre-cue null localises the effect to *after* the cue; it cannot split "cue flash" from "imagery
> onset," because in EEGBCI those begin at the same instant. Separating them needs a cue that
> looks identical across classes, and EEGBCI is position-confounded, target at screen top for
> fists, bottom for feet, so a class-discriminative visual evoked response necessarily exists
> post-cue whatever the subject imagines. This grid bounds **when** the effect starts, not **what**
> it is. **(b)** The filtering is applied to continuous data before cropping, and MNE's zero-phase
> 4–38 Hz firwin is a 265-tap symmetric FIR with a half-length of 0.825 s, so post-cue energy
> smears *backwards* into the pre-cue window. The direction is favorable, smear can only push a
> pre-cue score up, so a null is conservative, but an above-chance pre-cue result would have
> needed a truncated-filter re-run before it meant anything. And p = 0.108 is a failure to reject,
> not proof of a true null.

The interpretation is now a measurement: regime C's "the ranking flips" is the
CNN reading something **time-locked to cue onset**, not learning motor imagery better than CSP
does. What it is *not* is a story about CSP being blind.

> **The mechanism story is withdrawn, 2026-07-25.** This passage used to continue: *"The reason
> CSP cannot follow it there is structural. A phase-locked evoked deflection is a temporal
> pattern, and **CSP's log-variance features throw timing away by construction**, keeping only
> band power."* The rung's own data refutes it. CSP's paired post-cue-minus-pre-cue gain is
> **+6.0 points, p = 0.0103** across the same 20 folds, against EEGNet's +9.3. CSP follows the cue
> effect **less** than EEGNet does; it does not fail to follow it. "Cannot" was the wrong verb for
> a 6-point significant gain. The script says the same thing in its own output (`:457-462`): *"the
> MECHANISM story ... does not [stand], and should not be repeated anywhere downstream."*
>
> This is the project's signature error appearing inside the very rung built to catch it: the
> mechanism arrived in the same breath as the number, in a section whose opening paragraph names
> that exact failure mode and promises to test the interpretation rather than assert it. It tested
> the *effect* and asserted the *explanation*. The surviving claim is comparative and carries no
> mechanism: **both models read cue onset, EEGNet reads more of it, and this design cannot say
> what "it" is**: EEGBCI's cue position is confounded with the class, so a cue flash and imagery
> onset cannot be separated here at all.

This is the same shape of finding as rung 7's gaze confound, and it earns the same framing: build
a rung, believe it, then find the confound in your own result. **Two of the eleven rungs turned
out to be decoding the stimulus rather than the intention.** For a project about reading
*intent*, that is the most useful thing in the repository, and it is why regime C is kept and
re-analyzed rather than deleted.

> **The general lesson, worth carrying to any BCI result.** Both confounds were invisible to
> accuracy, to cross-validation, and to permutation testing, because in every case the model was
> finding real, reproducible structure. It was simply the wrong structure. The only checks that
> caught either one were **ablations that removed the thing the claim depends on**: frontopolar
> channels in rung 7, the imagery window here. If a result cannot survive deleting what it
> claims to be reading, the claim is about something else.

---

## 8. CSP: Common Spatial Patterns (the heart of the method)

This is the part I wanted to be able to derive on my whiteboard for genuine understanding,
so it goes intricately.

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

Mechanically it does this by simultaneously diagonalizing the two classes' covariance matrices,
it solves a generalized eigenvalue problem on `Cov(hands)` and `Cov(feet)`. The filters at the
extreme ends of the eigenvalue spectrum are the ones where the variance ratio between classes
is most lopsided, the most discriminative spatial patterns.

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

### 8.3 The `figures/csp_patterns.png` plot: *interesting, but NOT the honesty check*
`csp.plot_patterns(...)` draws each spatial filter as a **scalp map** (a top-down head with a
color heatmap).

> [!warning] **An earlier version of this section was wrong and it is worth understanding why.**
> It claimed the learned patterns were "focal over central / sensorimotor cortex" and offered that
> as proof the model found motor sources. Checking the actual channel weights: component 0 *is*
> sensorimotor (FC3/C3/FC1), component 1 mixes sensorimotor weights (FC4/FC2/C4) with occipital
> ones, and the component this document showcased peaks at **POz, PO4, Oz**: parieto-occipital
> *electrode positions*. In MNE topomaps posterior is at the **bottom**, and a lower-central blob
> was misread as the vertex. **Reading a topography by eye is not a control.**
>
> **A later layer on this same correction.** The sentence above used to end "…and correlates
> **r = 0.57** with the subject's own eyes-closed alpha map." That figure is withdrawn: no script
> in this repo computes any correlation, and it did not reproduce under the obvious definitions of
> one. It was load-bearing prose *inside a retraction*, which makes it the worst possible place for
> an unproduced number, the honest-self-correction section was itself resting on one. The
> **location** claim, that the component is posterior, survives on the channel weights alone; its
> **oscillatory** character was never measured, because no script in this repo computes a
> per-component spectrum, so "alpha-like" is withdrawn along with the figure.

**Why this matters enormously:** a model can hit 91% by cheating, locking onto an eye-blink
artifact, a neck-muscle tension that happens to correlate with the cue, or a per-run drift. The
control that gets closest to catching this is an **ablation**: restrict the montage to channels
that should *not* carry motor imagery and see whether the decoder survives. It does not.
Frontopolar-only lands at **51.1% (23/45)**, one trial *below* the 53.3% majority-class floor,
while keeping only the 17 sensorimotor channels holds accuracy at **95.6% (43/45)**.

> **Read that as a bound, not a proof, and here is exactly why.** ~~`ablate_channels.py:153-158`
> builds four conditions and only four: all-64, sensorimotor-only (17 ch), frontopolar-only
> (8 ch), and all-64 leave-one-run-out. **There is no arm that deletes sensorimotor cortex and
> keeps the rest of the montage.**~~ **The struck sentences were true until 2026-07-26 and are
> now false: the script builds six conditions, and two of them are the missing arm and a stricter
> version of it. They are kept visible because they are the record of the gap.** Frontopolar-only
> is still not that arm: it deletes 56 of 64
> channels, occipital, parietal, temporal *and* central, so its collapse to 51.1% is confounded
> with an eightfold reduction in channel count and feature dimension. A degenerate 8-channel
> decoder is what you would expect from that shrinkage alone. The script says the rest itself, in
> its own printed output: *"BOUND, NOT PROOF:
> the average reference is computed over all 64 channels before picking, so the subsets are not
> electrically independent, and EEGMMIDB ships no EOG channel to regress out. This ablation
> **bounds** the ocular contribution; **it cannot measure it**."* That sentence survives the new
> arms untouched, and the new arms add a second one to stand beside it: no channel-deletion
> experiment on a scalp montage can falsify a source hypothesis in *either* direction, because
> deleting the electrodes nearest a source does not delete the source from the ones that remain.
The intuition people reach for is: if CSP had learned an artifact, the scalp maps would light up at
the *edges* of the head (eyes, temples, neck) rather than the center, so centered patterns prove
motor sources. **That intuition is too weak to rely on, and this project is a worked example of why.**
Occipital alpha sits at the back of the head, not the edge; it is large, it is inside the 8-30 Hz
band, and at a glance in a topomap it is easy to mistake for something central.

**Accuracy alone never proves the signal is neural, and neither does a scalp map.** The ablation is
the strongest control this repo has, and it **bounds** rather than proves. The falsifiable prediction
it would take to prove the point is *"if the decoder is reading sensorimotor cortex, then deleting
sensorimotor cortex and keeping the rest of the montage must break it."*

**That experiment was run on 2026-07-26, and the prediction did not hold.** Until that date this
paragraph read *"**That experiment has not been run.** What has been run is the converse and a
shrinkage-confounded complement, which together bound the ocular contribution without measuring
it."* The concession is kept above as the record of what was owed, and here is what replaces it.
Delete the 17-channel FC/C/CP strip, keep the other 47 electrodes, refit the published pipeline
unchanged, and the decoder lands at **77.8% (35/45)**, ten-seed mean 79.3% over the range
[75.6%, 84.4%], against a majority floor of 53.3% and its own permutation null of
51.0% ± 8.8% at p <= 0.001. §10.5 carries the full arm, the pre-registered decision
rule, and the four separate things that keep it from being read as strongly in either direction.

Being able to say *"here is the control I ran, here is what would have falsified it, and here is
the arm I have not built yet"* is what separates a credible BCI result from a lucky number. Being
able to say *"here is the arm I finally built, and it went against me"* is the next thing after
that, and it is the state this section is now in.

> **Withdrawn, 2026-07-25.** This paragraph used to read *"What proves it is an ablation, because it
> makes a falsifiable prediction: if the decoder is reading sensorimotor cortex, then deleting
> sensorimotor cortex must break it. **Here that prediction holds, sharply.**"* It does not hold,
> because the experiment was never run: `ablate_channels.py:153-158` contains no
> sensorimotor-deleted arm, and the condition the sentence treated as one (frontopolar-only) drops
> 56 of 64 channels. The script's own stdout says *"BOUND, NOT PROOF ... it cannot measure it"*: so
> the document was asserting "proves" over an artifact that prints the opposite. The two nearby
> sentences that carried the same overclaim ("remove the cortex that should carry the signal and see
> whether the decoder dies. Here it does") are corrected above. Adding the missing arm was listed
> as §12.2 item 8.
>
> **A layer on that withdrawal, 2026-07-26.** The 2025 correction was right that the prediction had
> not been tested and right to downgrade *proves* to *bounds*. It was silently optimistic about
> what the missing arm would return: §12.2 item 8 called it "cheap" and predicted it would be
> "partially confounded", both of which are true, and neither of which prepared a reader for the
> prediction failing to hold at all. The register now records the outcome and not just the
> promise, because "the test I have not run yet" is a comfortable sentence and this project has
> written it more often than it has written "the test I ran, which cost me something."

---

## 9. LDA: Linear Discriminant Analysis (the classifier)

After CSP, each trial is just 4 numbers. LDA is a simple, fast linear classifier:
- It models each class as a Gaussian blob in the 4-D feature space and finds the straight
  boundary (a hyperplane) that best separates the two blobs, assuming they share a covariance
  shape.
- **Why LDA and not something fancier?** With only 45 trials, a complex model would overfit
  instantly. LDA has almost no free parameters, is the textbook partner to CSP log-variance
  features (which were designed to be Gaussian-ish for exactly this), and is the canonical
  baseline in the BCI literature. It's the *right* amount of model for the data size.

The `Pipeline([("CSP", csp), ("LDA"...)])` chains them so that, critically, **CSP is
re-fit on only the training data inside each cross-validation fold.** If you fit CSP on all data
once and then cross-validated only the LDA, CSP would have "seen" the test trials and your
accuracy would be inflated. Wrapping both in a Pipeline is what makes the evaluation honest.

---

## 10. Evaluation: cross-validation and the scoreboard

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
> actual interval, a Wilson bound on 45 trials gives roughly **[79%, 97%]**: and even that is
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

### 10.4 The permutation test: is 91% outside what noise produces?

Shuffle the labels, re-run the entire thing, see what randomness scores. Do it a thousand
times. If your real answer sits outside that whole cloud, you have something. It's a beautiful
way to show significance, which is the statistician's word for it.

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

### 10.4b Testing the tests: `permutation_design.py`

Two objections were lodged against the nulls in this repo and conceded here in prose, for weeks,
without either being measured.

**Objection (a), the partition.** `permutation_test_score` loops `for train, test in
cv.split(X, y...)` with the *permuted* labels as `y`, so `StratifiedKFold` re-derives the folds
from every shuffled vector. Each replicate is scored on a different partition, while the observed
91.1% is scored on the partition stratified on the **true** labels. The stated fix was to freeze
the partition at that true-label split, called P0.

**Objection (b), the reference set.** The labels are shuffled i.i.d. across all 45 trials, which
treats runs 6, 10 and 14 as one exchangeable pool. Run is a blocking factor with its own electrode
settling and drift, so the weaker and more defensible assumption is exchangeability *within* run
only. The within-run reference set is 14.2 times smaller than the i.i.d. one, and a smaller
reference set is a weaker assumption, not a stronger one.

`permutation_design.py` runs the full 2x2 of those two corrections at 10,000 draws per cell on
three subjects (1, plus 17 and 19 picked by a fixed median rule from `results/sweep_results.csv`). One list
of permuted label vectors feeds both partition rules, so the partition contrast is genuinely
paired and isolates the rule.

**Then it does the thing that made the run worth having: it tests the tests.** A permutation rule
is exact only if, when the null is true *by construction*, it rejects at most at its nominal rate.
That is checkable with no EEG at all. Run a majority-class classifier on an all-zero feature
matrix, where there is provably nothing to decode, draw 200 null label vectors, run 199 inner
permutations under each rule, and count how often each rule calls that classifier significant.
Every rejection it earns is a false one.

| rule | what it is | P(p ≤ 0.05), subject 1 |
|---|---|---|
| C1: i.i.d. shuffle, folds re-stratified | **the published null** | 0.0000 |
| C3: within-run shuffle, re-stratified | correction (b) alone | 0.0000 |
| C2: i.i.d. shuffle, folds frozen at P0 | correction (a) alone | **0.6550**, 13.1x nominal |
| C4: within-run shuffle, frozen at P0 | *"fully corrected"* | **0.6600**, 13.2x nominal |
| `KF_free`: i.i.d. shuffle, frozen at a **label-free** `KFold` partition | one valid version of (a); this is C5's rule | 0.0250 |
| `WITHIN`: labels permuted **inside each fold** of P0 | the other valid version of (a) | 0.0000 |

**The error runs the opposite way from the objection.** The cells this project built as "the
correction" are the ones that are not tests, and the published null is exact. The reason is one
line: P0 is `StratifiedKFold(...).split(X, y_true)`, so it is *a function of the labels being
permuted*. Freezing it makes the observed vector the unique point in the reference set whose fold
margins are balanced with respect to its own labels, while every replicate is scored on a partition
its labels did not produce. Computing the observed value and the replicates "the same way" is
necessary but **not sufficient**; the conditioning quantity must also be ancillary. Under
re-stratification the statistic is one function `S(y) = CV(X, y; SKF(y))` applied identically to
every vector including the observed one, so C1 and C3 are exact. sklearn was already right for its
own reference set.

Across the three subjects, 4 of 6 frozen-at-P0 cells are anti-conservative and **0 of 12**
re-stratified or label-free cells are. The defect does not bite at every class marginal, which is
worse rather than better: an analyst does not get to know which marginal they have before choosing
a rule. On the real pipeline the same defect appears as a size of 0.0752 against C1's 0.0444 at
nominal 0.05, and as rejecting at 29/45 where the exact partner needs 30/45.

**So everything derived from C2 and C4 is withdrawn**, including their p-values, their tail counts,
and the n_eff-corrected Wilson interval this project had recomputed from C4. The exact re-analysis
is smaller and it is the one that stands:

| subject | C1 (published, exact) | C3 (within-run, exact) | change at 0.05 |
|---|---|---|---|
| 1 (headline) | p ≤ 9.999e-05, C = 0 | p ≤ 9.999e-05, C = 0 | none |
| 17 (median) | p = 0.11529 | p = 0.052795 | none; factor 2.18, no crossing |
| 19 (median) | p = 0.055494 | **p = 0.044496** | crosses **downward** |

One objection of the two survives, then, and for one of the two median subjects: **run blocking is
a real correction and it moved one verdict.** Under C3 the two median subjects *disagree* at 0.05,
which is precisely the pre-registered neutral outcome, and nothing is concluded from a disagreement
between two subjects at 45 trials each. Subject 1's arm was uninformative **by construction** and
was registered as such in advance: at roughly 4.7 null standard deviations its p is pinned at the
resolution floor in every cell, so "unchanged" there is a resolution floor being read, not a
measurement, and it is **not** evidence the original design was right.

The one number the headline gains is a variance-corrected interval. The exact null's spread is
wider than binomial, VIF 1.323, so 45 cross-validated predictions are worth about 34 independent
ones and the Wilson interval widens from [79.3%, 96.5%] to **[77.0%, 96.9%]**. That is the
quantified version of the caveat §10.2 has always carried in words.

The valid form of objection (a) is worth seeing, because it shows why the shortcut was tempting.
Freeze the partition at a `KFold(shuffle=True)` split built **without** the labels and it *is*
ancillary, so the resulting cells (C5 with the i.i.d. shuffle, C6 with the within-run one) are
exact by construction, and the empirical check above confirms the i.i.d. version at 0.0250.
**But the observed value moves when the partition moves**, and by a lot: subject 1 goes from 41/45
to 42/45, subject 17 from 28/45 to 25/45, and subject 19 from 29/45 all the way down to 21/45. So
C5 and C6 are exact tests of a *different statistic*, accuracy on an unstratified partition, and
their p-values are not interchangeable with C1's and are not a correction to the published number.
That is the trap in miniature: the only way to freeze the partition validly is to freeze one the
published result was never computed on.

> **A registered falsification gate fired on this run and was overridden.** Assert 9 required every
> cell's null mean inside 0.45 to 0.55; subject 17's C4 centres at 0.4380, and the pre-registration
> then bars reporting anything in its Sections 6.1 to 6.4 from this run, which is both arms and not
> only the one that tripped it. It is reported anyway, on an argument added *after* a smoke run
> tripped the assert. The defence is a control rather than an argument: the same all-zero
> dummy classifier drops from exactly the majority rate with zero variance under re-stratification
> to 40.67% under a fixed partition, with no EEG involved, so sub-0.45 centring is a property of
> the partition rule. The exact label-free cells centre below the band too, which independently
> confirms the diagnosis. The pre-registration was internally inconsistent and the run found it:
> its own registered mechanism predicts a fixed-partition null centred below 0.50, and its assert 9
> declares exactly that fatal. Both cannot be right. **What the override does not buy** is a defence
> of C2 and C4, whose defect turned out to be exchangeability rather than centring. Right diagnosis,
> wrong problem, and fixing the first did not fix the second. A reader who rejects the override
> should treat this whole subsection as unreported.

**The cross-subject arm is where the objection holds cleanly**, because the defect there is the
reference set rather than the partition, and `LeaveOneGroupOut` is already label-invariant.
`cross_subject.py` draws one global permutation of all 900 pooled labels, which can deal a
45-trial subject a class split the protocol could not produce; one draw reached a 80.0% majority
rate. Under within-subject block permutation the observed 59.4% (535/900) survives at
p ≤ 0.00049975 with zero exceedances, and the global null is measurably inflated,
sd(global)/sd(block) = 1.5730, which is 16.3 Monte Carlo standard errors from 1. The deliverable is
a replacement for that script's hard-coded and underived `SHUFFLE_MAX = 0.60` leakage guard: the
block null's 99th percentile, **53.1% (478/900)**, or 53.4% (481/900) at the 99.5th. The existing
0.60 sits at the 100th percentile of the correct null, so it is *looser* than the null justifies
rather than tighter. Note the limit: no quantile here was ever validated against a **real** leak,
and `cross_subject.py` already discloses that its guard has never fired. Replacing an underived
threshold with a derived one does not measure its false-alarm rate against an actual defect.

That arm also corrected a stated fact about the corpus: the pooled class marginal is **not**
uniform across the 20 subjects. Six of them (1, 2, 3, 14, 16, 17) are 24/45 and fourteen are 23/45,
and that heterogeneity is exactly the structure a global shuffle destroys.

**This run was not blind**, and its own output says so first. A pilot had run both arms and its
numbers were read before the pre-registration was written. Agreement is confirmation at higher
resolution with a paired design, never independent discovery, and the pilot reproduced exactly, 4
of 4 cells.

### 10.5 The ablation: the control that actually rules out artifacts

A permutation test proves the model found *structure*. It cannot prove that structure is
**motor**. A decoder riding an eye-movement artifact that correlates with the cue would pass a
permutation test comfortably.

The control that discriminates is an **ablation**: refit the whole pipeline on the channel subset
where blinks and saccades are loudest, and see whether the result survives there. It now also runs
in the opposite direction, deleting the strip that should carry the signal. Both are **bounds
rather than proofs**, for reasons given below the table.

> **The framing sentence this section carried until 2026-07-26, kept as the record of what was
> owed.** It read: *"**No condition here deletes sensorimotor cortex while keeping the rest of the
> montage**, so the falsifiable form 'if it reads sensorimotor cortex, deleting sensorimotor cortex
> must break it' is **not tested and must not be attributed to this table**."* It is tested now.
> The result is rows 4 and 5, and it did not go the way the sentence implied it might.

Every row below is printed by **`ablate_channels.py`**, which reuses `decode_csp.py`'s
preprocessing verbatim, one seed (42), one pipeline, the same 45 trials, and **two splitters**:
stratified 5-fold for the first five rows, leave-one-run-out for the sixth, which is why that row
reports three per-fold values and the others report five. The all-64 row reproducing the published
91.1% exactly is the fidelity check that the pipeline was *reused* rather than reimplemented.

| channels used | ch | accuracy | trials | per-fold |
|---|---|---|---|---|
| sensorimotor only, FC3/1/z/2/4, C5…C6, CP3/1/z/2/4 | 17 | **95.6%** | 43/45 | 1.00 .89 1.00 .89 1.00 |
| all 64 | 64 | 91.1% | 41/45 | .89 .89 .89 .89 1.00 |
| frontopolar only, Fp1/z/2, AF7/3/z/4/8 | 8 | **51.1%** | 23/45 | .56 .78 .33 .56 .33 |
| **sensorimotor DELETED**, 47 kept | 47 | **77.8%** | 35/45 | .67 .89 .78 .78 .78 |
| wide FC/C/CP deleted, 43 kept | 43 | 71.1% | 32/45 | .67 .78 .67 .67 .78 |
| leave-one-run-out, all 64 | 64 | 93.3% | 42/45 | .87 .93 1.00 |

Restrict the montage to electrodes with no motor cortex under them and the decoder falls to the
**majority-class floor**: 51.1% is 23/45 against the 24/45 you get by ignoring the EEG entirely
and always answering "feet." The per-fold spread, 0.33 to 0.78, is the other tell; folds that
wide are a coin, not a decoder. Leave-one-run-out at 93.3% holds with no trial sharing a run with
its training set. That is a control. Reading a scalp map by eye, which is what this document
used to offer here, is not, and §8.3 is the full account of why.

> **Two words of scope on each of those, because both were stated more strongly than the design
> supports.** The first sentence used to open *"Delete the cortex that should carry the signal"*
>, the frontopolar arm does not do that; it keeps 8 of 64 channels and deletes everything else,
> sensorimotor cortex included. What it demonstrates is that an 8-electrode ring over the orbits
> carries no usable signal for this contrast, which is a real and useful negative control, and
> which is confounded with an eightfold cut in feature dimension. See §8.3.
>
> **A third sentence, missed by the edit above and corrected 2026-07-25.** The framing sentence
> introducing this table used to read *"because it makes a falsifiable prediction: if the decoder is
> reading sensorimotor cortex, then removing sensorimotor cortex must break it."* Same retracted
> inference as §1 and §12.1 round three, and it survived the first pass because that pass only fixed
> the sentence *below* the table. It now states the ocular hypothesis and the bound. The same edit
> withdrew *"one splitter"* from the provenance line: `ablate_channels.py:176-177` constructs a
> `StratifiedKFold` **and** a `LeaveOneGroupOut`, and the fourth row's three per-fold values against
> the others' five was the visible tell all along.
>
> The leave-one-run-out sentence used to end *"so the headline is not a **within-session** drift
> artifact."* That is withdrawn as stated. Runs 6, 10 and 14 are three recordings from **one
> session**, concatenated at `ablate_channels.py:132`. Holding one out removes the sharing of
> *within-run* drift between train and test; it cannot remove a session-level trend that runs
> monotonically across all three, because there is no second session to hold out. **EEGMMIDB is
> single-session, so nothing in this repo can test across-session stability at all.** The 93.3%
> (42/45) is sound and the control is worth having. The inference drawn from it was one level
> stronger than the design. The same overreaching sentence is printed by the script itself at
> `ablate_channels.py:213-214` and should be corrected there too.

#### Rows 4 and 5: the arm this document owed for three revisions

This is the arm §8.3 and §12.2 item 8 promised. Delete the 17 sensorimotor channels, keep the
other 47, and refit everything: average reference over all 64 before the pick, 8–30 Hz firwin,
epochs −1.0 to 4.0 cropped to 1.0–2.0 s, `CSP(n_components=4, log=True)` and LDA inside a fresh
`Pipeline` so CSP refits inside every training fold, `StratifiedKFold(5, shuffle=True)`. Seed 42
is primary, with a ten-seed sweep beside it.

**The complement scores well above the majority floor.** 77.8% (35/45) at seed 42, 79.3% over ten seeds across the range
[75.6%, 84.4%], permutation p <= 0.001 against a null of 51.0% ± 8.8%, Wilson 95% interval
[63.7%, 87.5%] whose lower bound clears the 53.3% majority floor by more than ten points. The
stricter deletion that also removes FC5/FC6/CP5/CP6 lands at 71.1% (32/45) at seed 42 and 76.7%
over ten seeds, so the four peri-Rolandic electrodes the 17-channel list happens to omit are not
what is carrying it.

**What the pre-registration allows to be said, which is less than the paragraph above suggests in
both directions.** The decision statistic was fixed before the run: G = (all-64 ten-seed mean)
minus (complement ten-seed mean), with a real loss requiring **both** G > 10.0 points **and** an
exact McNemar at p < 0.05 on the paired per-trial predictions. G is 94.0 − 79.3 = +14.7 points,
which clears its half. The McNemar is p = 0.0703 on a 2x2 of 34 both correct, 7 all-64 only, 1
complement only, 3 neither, which does not. The registered rule requires both halves and forbids
upgrading on whichever reads better, so:

> **REGISTERED VERDICT: a loss is suggested and not established at n = 45.** Undecided, leaned
> neither way. The sentence *"the sensorimotor strip is sufficient but not necessary"* is
> therefore **not written**, even though G points at it.

Read that p with its discordant count or not at all. At `n_disc = 8` the observed 7-vs-1 split
**is** the most lopsided split that still misses p < 0.05; only a clean 8-0 would have reached
it. The two arms agree on 34 of 45 trials and both miss 3 more, which is what keeps the count
small. *Not established* here means **underpowered**, not *evidence of no difference*, and the
pairing was verified rather than assumed: `StratifiedKFold(random_state=42)` returns identical
folds on repeated calls, and each arm's pooled `cross_val_predict` count equals its
`cross_val_score` fold mean.

**The rule could not have fired at its own threshold, and the script proves it from the design
alone.** In a paired 2x2 on the same 45 trials, `b − c` is identically the accuracy gap in trials,
so enumerating every attainable (b, c) shows the smallest gap that reaches p < 0.05 at *any*
discordant count is 6 trials, which is 13.3 points. The G half fires at 10.0 points, or 4.50
trials. Between 10.0 and 13.3 points the conjunction is unreachable no matter what the recording
does. That refutes the pre-registration's own stated justification for the rule, *"at a gap of 10
or more points the McNemar should fire comfortably"*, and it was checkable with two lines of
arithmetic before the run. The pre-registration was **not** edited to match; it is refuted, and
the refutation is recorded in its results section.

Three further facts, printed alongside, and two of them cut against the unflattering reading:

1. **The average-reference leak, now quantified.** The reference is computed over all 64
   electrodes *before* the 47 are picked, so every surviving channel carries −1/64 of every
   deleted one. Re-referencing inside the complement's own 47 costs it **6.0 points** over ten
   seeds (79.3% to 73.3%, and 35/45 to 31/45 at seed 42). Part of the complement's score is what
   volume conduction plus a 64-channel average reference *predicts*. This arm therefore **bounds**
   the strip's necessity and cannot establish it. The script also refuses to assign those 6.0
   points to the leak alone, because re-referencing removes one spatial dimension that mixes the
   strip's contribution with the complement's own global component, and that decomposition is not
   identified.
2. **Channel count is not the explanation, for this comparison only.** Deleting 17 channels *at
   random*, 50 times, moves the ten-seed mean to 93.5% with a G null centred at +0.5 points over
   the range [−1.8, +3.3]. Zero of 50 draws reached the observed +14.7, which sits 12.4 null
   standard deviations out. So the declared channel-count confound does not explain this arm's
   deficit. It is **not** retired for the 17-channel and 8-channel rows, which keep far fewer
   channels than this control ever does. That arm was added post-registration with the answer
   visible, and it is labeled as such.
3. **The verdict is not robust to which integer was typed as `random_state`.** Seed 42 gives
   p = 0.0703. The same exact test on the ten registered sweep seeds gives a median of 0.0391 with
   6 of 10 reaching p < 0.05, so seed 42 is worse for the loss than seven of those ten. And the two
   halves of the rule are evaluated on disjoint seed sets: G is a mean over seeds 0 through 9 and
   the McNemar sits at 42. None of that flips the verdict, because ten re-splits of the same 45
   trials are not ten samples. What it establishes is that the registered rule cannot certify band
   C at n = 45 and that the reported outcome is a property of the rule at least as much as a
   reading of the recording.

**What this arm is forbidden from saying.** The 47 retained electrodes include T7, T8, T9, T10,
TP7 and TP8, which is temporalis territory, so *"posterior cortex also decodes"* is blocked; the
permitted sentence is *"the 47 non-strip electrodes decode above the floor."* They also include
POz, PO4 and Oz, the peak of the strongest retained CSP component, so this arm sits on top of the
unresolved posterior question rather than resolving it. And the instrument limit is true in every
direction and outranks everything above: **no channel-deletion experiment on a 64-channel scalp
montage can falsify a source hypothesis**, because deleting the electrodes nearest a source does
not delete the source from the ones that remain. Forward-is-not-inverse refutes a negative source
claim exactly as hard as a positive one. Every row of that table is a sensor-space claim.

**One process note, because it changed whether the run was reportable.** The tenth registered
analysis-falsifier was first implemented too widely, firing whenever G >= 10.0 and the McNemar
missed, so the first run printed `1 FIRED`. The pre-registration scopes that falsifier to bands A
and B only, and band C registers the failed McNemar as its own downgrade path, which is the path
that was taken. The check was corrected to the registered scope **after** the first run, with the
answer visible. It changed no measured value, verified by diffing the two runs, but it changed
whether anything here is quotable, so it is recorded rather than quietly fixed.

**The accuracy half of this arm was not blind.** It had been run once, uncommitted, before the
pre-registration was written, and this run reproduced it to 0.0 points at both seed 42 and the
ten-seed mean. That is a replication of a known number and must not be reported as a confirmed
prediction. Only the permutation test, the Wilson interval and the McNemar were blind, and the
McNemar is the half that decided the verdict.

**The other direction is much weaker than this document claimed, and the distinction matters.**
Sensorimotor-only is 43/45 against all-64's 41/45. That is a **two-trial** difference on n=45,
well inside noise, and it is mostly a dimensionality effect anyway. A 64×64 covariance estimated
from 45 trials overfits, so almost any well-chosen small subset avoids that. The claim the
artifact control actually needs is *"removing 47 non-motor channels does not hurt,"* which the
data fully supports. *"The sensorimotor subset is better"* is not supported and is no longer
claimed. Note that rows 1 and 4 now sit awkwardly beside each other: keeping only the strip gives
95.6% and deleting only the strip gives 77.8%, and both are far above the floor. The strip is
richer than its complement and neither is empty.

> ### The correction that produced this script
>
> Until this revision the table above read **95.9%** for sensorimotor and **47.4%, i.e. chance**
> for frontopolar, and **no script in the repository produced any of the four rows**.
> `ablate_channels.py` did not exist. This was the repo's designated artifact control. The thing
> §8.3 says "replaced" the retracted scalp-map defence, so it was simultaneously the most
> load-bearing and the least sourced number here.
>
> Both bolded values were also *arithmetically unreachable*. With 45 trials and a partition that
> tests each exactly once, accuracy is k/45, steps of 2.222%. Attainable values near the
> headline: 39/45 = 86.7%, 40/45 = 88.9%, 41/45 = 91.1%, 42/45 = 93.3%, 43/45 = 95.6%,
> 44/45 = 97.8%. **No k gives 0.959 or 0.474.** The script now both prints this lattice and
> *asserts* it, and cross-checks `cross_val_score`'s fold mean against a pooled
> `cross_val_predict` count, so an unequal-fold scheme cannot silently emit an off-lattice number
> that still looks like one.
>
> | published | actual | reason |
> |---|---|---|
> | 95.9% | **95.6%** (43/45) | off-lattice, so it cannot have been measured on a 45-trial partition. Origin unknown; see the note below. |
> | 47.4%, "i.e. chance" | **51.1%** (23/45) | off-lattice, *and* the wrong reference, chance here is the 53.3% majority-class rate, not 50% |
> | 91.1% | 91.1% (41/45) | already correct |
> | 93.3% | 93.3% (42/45) | already correct |
>
> This is the same failure mode §12.1 names five times over, *the number and the story arrived
> together*: except here the number arrived without even a run behind it. It is why
> `check_provenance.py` now exists: it extracts the **percentages, p-values, correlations and a
> narrow class of counts** from this document and the README, and fails if no script's stdout
> contains one of them. That is narrower than "every figure", and the gap matters, see the note
> on what the guard cannot see, at the end of §12.1.
>
> **"A transcription slip of 95.6" is withdrawn, 2026-07-25.** That was an unbacked causal story
> about where the bad number came from, offered inside a retraction, which is the one place this
> project has learned to be most careful. It is also not the best-fitting reconstruction: a
> 20-seed mean over a 21-channel sensorimotor set lands on **0.9589 → 95.9%**, and an 8-channel
> frontopolar set on **0.4744 → 47.4%**, both on the 1/900 lattice a 20-seed mean lives on, which
> would make the pair a coherent readout of a *different, unrecorded experiment* rather than two
> independent typos. That reconstruction is itself unverified and is offered only to show the slip
> story is not the only account available. **What is established and what the retraction rests on
> is unchanged: on a 45-trial partition accuracy must be a multiple of 1/45, neither 95.9% nor
> 47.4% is, so neither can have been measured the way it was published.** How they got onto the
> page is not known, and the honest thing is to say so rather than to supply a tidy reason.
>
> **One limit on what any of this bounds.** The average reference is computed over all 64 channels
> *before* a subset is picked, which is what `decode_csp.py` does, so the subsets are not
> electrically independent, every channel carries −1/64 of every other. Re-referencing each
> subset separately would give a cleaner number but would no longer be the published pipeline. The
> ablation therefore **bounds** the ocular contribution rather than eliminating it. It is also
> **band-conditional**: at 8–30 Hz it says nothing about low-frequency ocular contamination, which
> is where rung 7's gaze confound lives.

### 10.5b The muscle probe: `emg_proxy.py`, and why an ablation could not do this

The ablation above cannot bound muscle, and it is important to see *why* rather than to take it
on trust. Surface EMG from temporalis or jaw is **broadband**, with most of its power well above
the mu/beta band. The published pipeline band-passes to 8–30 Hz on the continuous signal, before
any covariance is computed, so the frequencies where muscle is loudest are discarded before CSP
ever sees them. No arrangement of channel subsets inside that pipeline can probe them. A separate
instrument is required, and that is what this script is.

**Design, pre-registered before it was written.** Refit the *unmodified* CSP+LDA pipeline, same
splitter and same seed 42, on **40–75 Hz with a 60 Hz notch**, restricted to the eight-channel
temporal ring (T7 T8 T9 T10 TP7 TP8 FT7 FT8). If class information rides on muscle, a muscle-band
decoder at muscle-territory electrodes should find it. A positive control runs first and must
reproduce the published number: 8–30 Hz on all 64 channels returns 91.1% (41/45), so the harness
is the published pipeline and not a lookalike.

**Result: nothing there.**

| channel set | ch | accuracy | permutation p |
|---|---|---|---|
| temporal ring (**primary cell**) | 8 | **51.1%** (23/45) | 0.5175 |
| all 64 | 64 | 40.0% (18/45) | 0.9141 |
| sensorimotor | 17 | 37.8% (17/45) | 0.8941 |
| frontopolar | 8 | 33.3% (15/45) | 0.9810 |
| *floor* | | *53.3% (24/45)* | |

A univariate arm agrees: mean log high-band power does not differ by class on the ring
(Welch t p = 0.6922, Mann-Whitney U p = 0.7074, and 0 of 8 channels survive Holm-Bonferroni).
Robustness bands land within noise of the primary (40–55 Hz at 55.6%, 65–75 Hz at 57.8%, the
greedy 32–75 Hz at 51.1%), and their pre-registered role forbids them from promoting a null
primary in any case.

**Two readings of "below the floor" and only one of them is right.** The floor is what a *constant*
predictor scores, and this pipeline is not a constant predictor even on random labels: each
channel set's own permutation null has a median of 22 to 23 of 45. Against its own null the
temporal ring sits at the 48th to 57th percentile, which is *at chance*, not below it. The other
three sets sit in the lower tail of their own nulls, which is a different phenomenon that this run
explicitly declines to explain, because naming a mechanism in the same breath as a number is this
project's round-one failure mode and it gets its own registration or it gets nothing.

**The part that turns a null into a measurement.** A null is worth exactly what the instrument's
sensitivity is worth, so the script plants a synthetic class-correlated source in the ring at a
ladder of amplitudes and finds the smallest one it can detect. Detection is the smallest amplitude
whose median accuracy across ten injection seeds reaches 31/45, chosen over 30/45 because 30/45
sits at binomial p = 0.0490 and a criterion should not rest on a knife edge. Amplitude `a` is the
source's contribution to T8 as a fraction of T8's own measured high-band standard deviation.

Over four source shapes and both class directions the *worst* threshold is **a = 0.600**, against
0.150 in the easiest cell. That worst case is the bound, and it has three named holes:

- **Spectral.** The probe covers 40–75 Hz minus the notch, out of a recorded 0–80 Hz. 160 Hz
  sampling discards everything above the 80 Hz Nyquist, where temporalis EMG has real power, and
  the probe itself declines the 40 Hz below its own lower edge. **Nothing at all is bounded inside
  the decoder's own 8–30 Hz passband, which is the only band the headline can be contaminated
  in.** The measured PSD makes that worse, not better: the recorded high band is steeply
  attenuated, so any EMG present shows up preferentially where this probe is not looking.
- **Temporal.** Every rung injects a constant amplitude into *every* trial of a class, which is
  the shifted distribution the script's own text calls unrealistic. A bursty source in 25% of
  trials is **not bounded at any amplitude tested**, including ten times the ladder's top rung,
  because the registered criterion cannot adjudicate that duty cycle at all. That is an open
  exposure, not a larger bound.
- **Statistical.** 45 trials buys a large-effect bound and nothing finer: the univariate arm
  excludes Cohen's d at or above 0.837 aggregate and 1.069 per channel, both large.

**One pre-registered falsifier partially fired and it must not be buried.** Falsifier 7 pinned the
notch at 89 taps. MNE splits `trans_bandwidth=6.0` into 3.0 Hz per side, so the realised notch is
177 taps and the cascade half-length is 1.375 s rather than the pinned 1.100 s. The *frequency*
design is exactly as pinned; only the time-domain prediction was wrong. Two printed statements had
to be recomputed: the 1.0–2.0 s feature window can draw energy from as early as −0.375 s, and the
pre-cue diagnostic window is now **entirely** filter-contaminated, so its null (t p = 0.4140) is
uninformative exactly as pre-declared and says nothing about pre-cue high-band power. The
conclusion is unaffected because filter smear can only push a score *up*, so a null stays
conservative. The notch parameters were not re-tuned to make the pinned tap count come true; the
pre-registration pinned the *call*, and the tap count was a prediction about it that the script
was written to check rather than trust.

> **Withdrawn on 2026-07-26, one day after it was published, and kept visible.** This section
> first reported the bound as *"a = 0.300 of T8's own high-band SD"* and stated that *"the corpus
> line 'nothing in the repo bounds an EMG contribution' is now false."* Both overstated, in the
> same direction, three times over. The 0.300 was the worst of **two** registered shapes, both
> diffuse; adding the canonical focal one-electrode source doubles it, and a focal source at
> a = 0.500 sits *inside* this recording's tolerance while sitting *outside* the number that was
> published. The claim is also band-scoped and continuous-source-scoped as above. The corpus line
> is false **for 40–75 Hz only** and remains true inside 8–30 Hz. The failure mode is the
> familiar one wearing new clothes: a max over a convenience sample of two was written as a max.

**What is not run, stated because a conjunction with one arm run is not satisfied.** The closure
condition has two parts: (i) a temporal-channel-*deleted* ablation inside 8–30 Hz that should not
hurt appreciably, and (ii) this high-band probe landing at or below the floor. Only (ii) exists.
And EEGMMIDB ships no EMG reference channel, so there is no ground truth for "this is muscle" at
all: the probe measures high-band power at muscle-adjacent scalp sites. The word it is licensed to
use is **bounds**, never *eliminates*.

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
| `SUBJECT` | 1 | Which person | **Already swept**: rung 6 runs all 109. Median 60.0%, and subject 1 is the 91st percentile. |
| `RUNS` | 6,10,14 | Which task | **4,8,12** = *imagined* left vs. right fist, a much harder contrast on the same homunculus strip. **Not 3,7,11. Those are executed movement** (see the run table in §4). **Already built** as rung 7, where it found a gaze confound. |
| `L_FREQ,H_FREQ` | 8,30 | The band kept | Split into mu (8–12) and beta (13–30) and combine (filter-bank CSP), often a real gain, and still unbuilt here. Splitting them on the left/right contrast is what exposed it as an alpha-band decoder. |
| CSP `n_components` | 4 | # spatial filters | 6 or 8, more filters can help or overfit; cross-validate to decide. Untested here. |
| CSP `reg` | None | Covariance shrinkage | `'ledoit_wolf'`: stabilizes CSP when trials are few or channels many. Untested on this baseline. Worth noting that rung 9's Riemannian arm ran with OAS shrinkage while its CSP baseline ran `reg=None`: an asymmetry that favors the **Riemannian** arm, not the baseline. The baseline won anyway. |
| crop `1.0–2.0 s` | 1 s window | Which slice becomes features | Sliding it on the left/right contrast gave **66.7 / 55.6 / 73.3 / 64.4 / 46.7%** for starts at 0.0 / 0.5 / 1.0 / 1.5 / 2.0 s. A **26.7-point** range across *overlapping* windows of the same trials. Treat this knob as a **noise source**, not a tuning surface. |
| `cv` | `StratifiedKFold(5, shuffle=True)` | Evaluation rigor | **Leave-one-subject-out** once you go multi-subject (rungs 8–10 do). Leave-one-**run**-out is the cheaper *within*-session check: it holds at 93.3%. It is **not** a session-level check, which this document used to call it, runs 6/10/14 are three recordings from one session, and EEGMMIDB has no second session to hold out. |
| `random_state` | 42 | Reproducibility seed | Rung 5 already swept 100 seeds: 88.9–97.8%, mean **93.8%**, with 42 at the **3rd percentile**: so the published number understates its own estimator by ~2.7 points. That "3rd" is a *strictly-below* rank; the estimator is quantized to 1/45, many seeds tie exactly on 91.1%, and a tie-aware rank puts 42 higher. The 2.7-point gap does not depend on the convention; the word "3rd" does. Worth knowing that 42 was **inherited from MNE's CSP tutorial, not chosen**: which makes "the seed wasn't cherry-picked" true but vacuous. |

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
| 5 | Is that number an artifact of the estimator? | No. The estimator change *raises* the expectation by ~0.2 points (ShuffleSplit mean 93.6% vs. StratifiedKFold mean 93.8% over 100 seeds). The whole 3.3-point move is **seed placement**: seed 42 sits 2.7 points below the published estimator's own 93.8% mean |
| 6 | Does it hold across 109 people? | Median **60.0%**, IQR 52.8–75.6%. Subject 1 is the **91st percentile** |
| 7 | What does a harder contrast cost? | **Unanswerable as run**: the left/right rung is **gaze-confounded**, and overlapping crop windows of the same trials disagree by 26.7 points |
| 8 | Does it transfer to an unseen person? | Near-parity, 95% CI **[−1.9, +11.2]**, p = 0.181. Cannot distinguish a drop from no drop |
| 9 | Does a Riemannian method beat it? | No, but only **MDM-64 is significant** (p = 0.005, Holm 0.019); the rest is underpowered, and the comparison was tilted *toward* Riemannian by shrinkage. These p-values used to be computed in prose by no committed script; `inferential_stats.py` §2 now produces them |
| 10 | Does a CNN beat it? | Scores lower by **4 of 45 trials** net at n=45 (8.9 points; 10 discordant trials, exact McNemar **p = 0.344**), level at ~900 trials |
| 11 | What did rung 10's regime C measure? | **Cue onset, not the imagery window.** EEGNet scores 61.1% on the 0–1 s window alone against 51.8% on the second before it, and adding the whole imagery window buys +1.9 (p = 0.41). CSP is also above chance in that window (53.7%, p = 0.023), so this is a difference of degree, not a dissociation. Figures are 2026-07-23 checkpoint values (§7 rung 11) |
|, | Is the artifact ablation real? | **It is now.** `ablate_channels.py` produces it; two of its four published values were unreachable and are corrected below |
| guard | What does deleting the 17-channel sensorimotor strip cost? | **Undecided, leaned neither way.** 47 electrodes with the FC/C/CP strip removed reach **77.8% (35/45)**, ten-seed 79.3%, permutation p ≤ 0.001. G = +14.7 points clears its threshold; the paired McNemar (p = 0.0703, 8 discordant) does not. Registered verdict: *a loss is suggested and not established at n = 45* (§10.5) |
| guard | Is class information riding on muscle? | **Not in 40–75 Hz, and unbounded in 8–30 Hz.** A muscle-band decoder at the temporal ring lands at **51.1% (23/45)**, below the 53.3% floor, permutation p = 0.5175, univariate arm null. Sensitivity bound a = 0.600 of T8's high-band SD; a bursty source is not bounded at all (§10.5b) |
| guard | Is the permutation null itself correctly built? | **The within-subject one already was.** The published re-stratified null is exact; the "correction" this project favored is anti-conservative to 13x nominal on zero-information data and is withdrawn. Run blocking is real and moved one median subject across 0.05. Cross-subject, the block null replaces `SHUFFLE_MAX = 0.60` with **53.1% (478/900)** (§10.4b) |

### 12.1 What is retracted, and why that list matters

Claims this project published that did not survive adversarial review. They are listed here
rather than quietly deleted, because the list is more informative than the results. **This list
only grows**: when a correction is itself later found wanting, a layer is added rather than the
record rewritten.

**Round one, the six found by the first review:**

- **"27% BCI illiteracy."** Inverted inference. A pure-noise null predicts ~55% below chance;
  28% was observed. It is evidence *of* signal.
- **"EEGNet loses by 37.8 points."** A units bug. The network was never training. Real gap: 8.9.
- **"The ranking flips once both models get a wider band and a longer window."** Three factors
  moved at once, and the effect turns out to belong to none of the two that were credited. Rung
  11 traces it to an undocumented crop-start change, shows the stated band mechanism is
  backwards, and confirms by measurement that EEGNet scores **61.1% on the cue-onset window
  alone**. (This bullet used to say "on a window containing no imagery at all." That phrase is
  itself withdrawn in round three below: the 0–1 s window contains the cue *and* the first second
  of imagery, and this design cannot separate them.)
- **"73.3% is what a harder contrast costs."** The rung is **gaze-confounded**, and five
  overlapping 1-second windows of the *same* trials disagree by **26.7 points**
  (`harder_contrast.py`, cached stdout `:25`), so 73.3% is a window draw and not a cost of
  contrast at all.
  > **A second layer on this bullet, 2026-07-25.** It used to read *"**Group value is ~7 points**,
  > and the rung is gaze-confounded."* That "~7 points" is the 16-subject figure withdrawn at §7
  > rung 7 for lack of provenance. No script in this repo computes it and the unexplained *n* = 16
  > matches no other multi-subject rung. It was withdrawn there and survived here, in the very list
  > that is supposed to govern withdrawn claims. Withdrawn again, and named as the specific failure
  > it is: a retraction register that still asserts a retracted number is not doing its job. The
  > retraction stands on the window sweep, which is produced by code.
- **"No method dominates / the best method is subject-specific."** Per-subject optimality requires
  a **crossover** subject × method interaction, and none is demonstrated: the pairwise homogeneity
  test that could show one is null and underpowered (**χ² = 13.33 on 19 df, p = 0.821**, MDE 5.68
  points). (An earlier version of this line said no interaction *exists*, which asserts a null
  this experiment has too little power to establish. The same move §14 item 6 warns against.
  A second version said *"no subject × method interaction is detectable (p = 0.84)"*; that p and
  its χ² of 13.0 are withdrawn as unproduced, **and the sentence was wrong as well as unsourced**:
  the design-appropriate omnibus, Tukey's 1-df test on the 20 × 5 layout, **does** reject
  additivity at F = 13.4627, p = 0.0005. What it detects is a fan driven by MDM-64, not a
  crossover, so the retraction is unaffected. See §7 rung 9.)
- **"CSP patterns are focal over sensorimotor cortex, which is my evidence against artifacts."**
  False for the showcased component, which is parieto-occipital (§8.3).

The pattern is worth naming, because it is the same mistake five times: **the mechanism story was
invented in the same breath as the number**. Measuring and explaining are separate steps, and
doing them together is how a plausible narrative gets attached to noise.

**Round two, found by writing the scripts the round-one corrections had promised.** Every entry
below is a number that survived the first review because nobody checked whether any code produced
it. This is a *different* failure mode from round one, and arguably a worse one: round one
invented mechanisms for real measurements; round two published measurements that were never made.

- **"Sensorimotor only 95.9%, frontopolar only 47.4%, i.e. chance."** The repo's headline artifact
  control, quoted in the README, in §1, in §8.3, in §10.5 and in the one-minute talking points,
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
  73.3%** (§7 rung 7). The confound is real and is now better evidenced. A cue-window frontopolar
  asymmetry at **t = +7.71, p = 3.7e-09**: but by a different measurement than the one published.
- **"r = 0.57 between the showcased CSP component and the subject's eyes-closed alpha map."**
  Stated three times across the two flagship documents, computed nowhere, and not reproducible
  under the obvious definitions. Withdrawn. It sat *inside* the §8.3 retraction, so the
  self-correction was itself resting on an unproduced number.
- **"Ocular checks come back clean: HEOG p = 0.27, VEOG p = 0.44."** EEGMMIDB ships **zero EOG
  channels**, which the same page says two paragraphs earlier. These were undisclosed frontal-EEG
  surrogates, computed by no script, and an underpowered null presented as a clean bill of health.
  Withdrawn and replaced by the scripted frontopolar ablation (§14 item 4).
- **"Rung 9's shrinkage flattered the comparison in the baseline's favour."** Backwards. OAS sits
  on the **Riemannian** arm; the CSP baseline runs `reg=None`. The asymmetry favored Riemannian,
  and the baseline won anyway. §7 rung 9 had it right and §11 had it wrong. The document
  contradicted itself about its own most-cited negative result.
- **"An independent reimplementation landing on the same values."** Rung 11's control shares
  `load_subject`, `make_eegnet` and `seed_everything` with the rung it audits. It is a regression
  check on the plumbing, not independent validation; a shared bug would reproduce perfectly.
- **"EEGNet loses by 8.9 points at n=45, learned filters need volume."** The 8.9 points are
  **four trials out of 45**, no significance test was run on them, and the two regimes differ in
  optimisation budget (~100 gradient steps versus ~2600) as well as in sample size. The direction
  is fine; "learned filters need volume" is not what was measured.
- **"79 / 109 subjects above their own chance."** A bare inequality reported in a table that reads
  as a count of people with decodable signal. No per-subject test is run. Relabelled, not removed,
  rung 6's actual conclusion is a population-level comparison that does not use it.

**The round-two pattern has its own name, and it is the one worth carrying forward:** *a number
whose provenance you cannot state is not a result, however true it turns out to be.* Two of the
figures above (91.1%, 93.3%) were correct all along, which is precisely why the defect survived
review, a table that is half right reads as sourced.

**Round three, 2026-07-25, twelve entries, found by auditing the round-two corrections
themselves.** Three of these sat *inside* a passage whose job was to correct something else
("six orders of magnitude", "repeated runs have come back identical", and the "~7 points" that
outlived its own withdrawal two sections earlier), and a fourth is a correction block's
downstream conclusion. A retraction is prose like any other, and this project had been treating
its own retractions as trustworthy because of where they sat on the page rather than because
anyone had rechecked them.

- **"Roughly 2.7 points of seed luck and 0.6 points of real estimator change."** The 94.4 → 91.1
  decomposition, wrong in its *split* while right in its *total*, which is why it survived. The
  0.6 differences seed 42's ShuffleSplit **draw** against StratifiedKFold's **mean**, moving 0.8
  points of old-estimator seed luck onto the estimator's account and flipping that term's sign.
  The estimator is worth **−0.2 points**: it *raises* the expectation and cannot have contributed
  any share of a drop. Corrected to seed placement 3.5, estimator −0.2 (§7 rung 5).
- **"The end-to-end logits are only about 53× too small."** Unreachable from the document's own
  premises: under its stated ~31.6×-per-stage recovery the chain runs 4500 → 142 → 4.5 → 0.14, and
  `4500 / 31.6ˣ = 53` needs x = 1.29 stages. Withdrawn; the deficit at the first BN (~4500×) is
  what survives. The "cannot train out of it" conclusion was computed *from* the 53× and is
  downgraded from established to plausible (§7 rung 10).
  > **And the replacement was wrong too, measured 2026-07-25.** The correction above swapped the
  > unreachable 53× for the model's own **4.5×** and recomputed the training margin as a **2.3×
  > shortfall**. `inferential_stats.py` §§7–8 measures the chain with forward hooks instead of
  > modelling it: the per-stage deficits are **4744× / 459× / 59×** and the logits are **102×** too
  > small, so per-stage recovery is ~10× and ~8×, not 31.6×, and the margin is a **64×** shortfall
  > (mean-|dw| travel) or **139×** (sd travel), not 2.3×. Both the withdrawn 53× and its
  > replacement 4.5× are wrong; the measured value sits between them and closer to the former.
  > Note the shape of this one: it is the **fourth** consecutive layer on the same passage, and
  > every layer before this one was arithmetic on an unmeasured model.
- **"Both are wrong by about six orders of magnitude."** A correction block wrong twice about its
  own arithmetic: the two errors it describes are **2.85** and **7.47** orders, four and a half
  orders apart, and its explanation of the 1e-8 cannot coexist with its own stated 7e-6 activation
  scale (variance 4.9e-11). Withdrawn (§7 rung 10).
- **"What proves it is an ablation ... deleting sensorimotor cortex must break it. Here that
  prediction holds."** The experiment does not exist. `ablate_channels.py` builds four conditions
  and none of them deletes sensorimotor cortex while keeping the montage; the arm treated as that
  one drops 56 of 64 channels. The script's own stdout says **"BOUND, NOT PROOF ... it cannot
  measure it."** Downgraded from proves to bounds (§8.3).
- **"The fists-vs-feet headline passes both its ocular checks."** There are none. Every ocular
  analysis in `harder_contrast.py` runs on the **left-vs-right** runs 4/8/12; the fists-vs-feet
  reference gets only an accuracy and a permutation test. The headline has **one** artifact
  control, the channel ablation, and it bounds rather than checks. The claim also falsified §14's
  own universal six lines above it, which is demoted to a rule (§14 items 4 and 5).
- **"CSP's log-variance features throw timing away by construction."** The mechanism story
  attached to rung 11's finding, refuted by rung 11's own data: CSP's paired post-cue gain is
  **+6.0 points, p = 0.0103**. CSP follows the cue effect *less* than EEGNet, not not-at-all.
  Withdrawn, and it is the round-one failure mode reappearing inside the rung built to catch it
  (§7 rung 11).
- **"The cue window, which contains no imagery at all."** An assumption stated as a measurement,
  three times, while the script producing the cell said in its own docstring that it was an
  assumption. The subject begins imagining *at* the cue. Renamed the **cue-onset window** (§7
  rung 11).
- **"Both models sit at chance before the cue. The control passes."** A failure to reject sold as
  a pass. The pre-cue EEGNet interval is **[49.1%, 54.5%]**, whose upper bound exceeds the
  cue-window CSP estimate of 53.7% that the same rung calls significant. No power analysis, no
  equivalence test, and the interval was never shown. Rewritten onto the paired comparison, which
  is what actually carries the conclusion (§7 rung 11).
- **"Repeated runs on this machine have come back identical."** The round-one correction chose the
  wrong side: it retracted an accurate note about CNN non-determinism and replaced it with an
  unmeasured claim of determinism. Runs of a byte-identical regime-C configuration disagree by
  **0.3 points**: 63.0% in the committed `regime_decomposition.json` checkpoint against the 63.3%
  `eegnet_compare.py` prints for that cell, while CSP reproduces exactly. Restored, with the measurement
  attached (§13).
- **"One commit per rung."** 26 commits for 11 rungs, one commit introducing four rungs, and rung
  11 spanning three. Wrong in both directions, in a sentence inviting the reader to check it
  against the history (§6).
- **"`check_provenance.py` extracts every number in the docs."** Stated in four places; the script
  never claimed it. It cannot see multipliers, point-differences, µV, t-statistics or parameter
  counts, which is exactly where the two worst defects above live. Corrected to the docstring's
  own scope, with the blind spots tabulated below.
- **"Group value is ~7 points"** (round one, rung 7 bullet). A figure already withdrawn in §7 for
  having no script behind it and an unexplained *n* = 16, which then went on living in the
  retraction list itself. Withdrawn a second time. This is the entry that motivated auditing the
  corrections rather than the claims, and it is left visible in round one above with its own
  withdrawal note attached.

**What round three should change about how this repo is read.** Rounds one and two each had a
single-sentence lesson. This one's is narrower and less comfortable: **a retraction is not
evidence, and marking a passage as a correction does not make its arithmetic true.** Every defect
in this batch sat inside prose whose function was to be trustworthy. The mitigation is the same
one the project already applies to results, applied one level up: a correction has to state what
it recomputed and from which artifact, or it is just a confident paragraph in a different font.

**Round four, 2026-07-25 into 2026-07-26: six entries, found by running three tests this corpus
had only conceded.** These come from a different place than the first three rounds. A reviewer
read the finished corpus and returned one sentence: *this project is trained to concede exposures
with great precision and is not trained to measure them; nine of ten findings name a hazard, name
exactly what test would close it, and then do not run the test.* Three of those tests were then
built and run under pre-registration. Rounds one and three were about claims that were wrong.
**This round is mostly about claims that were merely unmade, plus two the measurements actively
refuted**, and the entries are ordered with the costly ones first.

- **"No condition here deletes sensorimotor cortex while keeping the rest of the montage."**
  Withdrawn as a *state of affairs*, not as an error: the condition now exists, and the finding is
  that the prediction it was shielding **did not hold**. 47 electrodes with the strip removed reach 77.8%
  (35/45), ten-seed 79.3%, permutation p ≤ 0.001. This entry is in the register because the
  concession was load-bearing everywhere it appeared, in §1, §8.3, §10.5, §14 and the README, and
  because the result it was concealing is the one that costs this project the most. The registered
  verdict is *suggested, not established*, and the sentence "the strip is sufficient but not
  necessary" is not written. See §10.5.
- **"The published permutation null marginalises over partitions the observed value conditions on,
  so it is the wrong null."** Conceded in this document and treated as a defect awaiting repair.
  **Measured, and the concession was wrong in its conclusion.** The published re-stratified null is
  an exact test. The repair this project favored, freezing the folds at the true-label partition,
  is **not** a test: on data with provably zero information it rejects at 0.6550 against a nominal
  0.05, 13.1x, because the frozen partition is itself a function of the labels being permuted.
  Every number computed from those cells is withdrawn, including a variance-inflation factor and an
  n_eff-corrected Wilson interval this document had already published from them. What survives of
  the objection is run blocking, which is real and moved one of two median subjects across 0.05.
  See §10.4b.
- **"Nothing in this repo bounds an EMG contribution."** True when written, false now, and the
  scope matters more than the reversal: it is false **for 40–75 Hz only** and remains true inside
  the decoder's own 8–30 Hz passband, which is the only band the headline can be contaminated in.
  §10.5b carries the probe and its three named holes.
- **"This recording contains no class-correlated broadband temporal source as large as a = 0.300
  of T8's high-band SD."** Published 2026-07-25, withdrawn 2026-07-26, one day of life. The 0.300
  was the worst of **two** shapes, both diffuse, and was written as though it were a maximum over
  topographies. Over four shapes including the canonical focal one-electrode source the worst
  threshold is **a = 0.600**, so a focal source at a = 0.500 sits inside this recording's tolerance
  and outside the published bound. The same sentence was also silently continuous-source-scoped: a
  bursty source at a 25% duty cycle is not bounded at any amplitude tested. Three scope failures,
  all in the direction that overstates coverage.
- **"At a gap of 10 or more points the McNemar should fire comfortably."** A pre-registration's own
  justification for its two-part decision rule, refuted by arithmetic available before the run. In
  a paired 2x2 on 45 trials the accuracy gap fixes `b − c`, so the smallest gap that can reach
  p < 0.05 at *any* discordant count is 6 trials, or 13.3 points. The rule's other half fires at
  10.0 points. Between those two the conjunction cannot fire whatever the data does. The
  pre-registration was not edited to match; it is refuted on the record.
- **"`ablate_channels.py` builds four conditions and only four."** A factual statement about the
  code, true when written and false now that the script builds six. Listed because this document
  has twice been caught describing its own scripts from memory, and a correction that leaves the
  old line unmarked is how the next reader gets misled.

**What round four should change about how this repo is read, and it is the least comfortable
lesson of the four.** The first three rounds were about numbers that were wrong. This one is about
a *habit* that reads as rigour and functions as avoidance: naming a hazard precisely, naming the
test that would settle it, and stopping there. A conceded exposure and a measured one are not the
same object, and this corpus had been treating them as though they were, because the concession is
written in the same confident voice as a result and sits in the same section. Two of the three
tests came back against the framing they were built to defend. **A disclosure is a promissory note,
and this round is what it costs to redeem three of them.** The remaining exposures listed in §12.2
should now be read as debts with a known price rather than as evidence of thoroughness.

`check_provenance.py` now extracts the
percentages, p-values, correlations and result-bearing counts from this document and the README
and fails if no script's stdout contains one of them.

**What the guard cannot see, stated so nobody trusts it further than it goes.** Four places in
this repo used to describe `check_provenance.py` as extracting *"every number"* or *"every
figure."* It does not, and the script's own docstring never claimed it did, `:13-14` says
"percentages, p-values, correlations, and a narrow class of counts," and `:25-27` states the
value-matching limit plainly. The docs overstated what the code says about itself. Concretely,
`PATTERNS` (`check_provenance.py:81-86`) matches only `N%`, `p`/`r` compared to a decimal, and
integers glued to `trials|subjects|shuffles|seeds|folds`. Everything else is invisible:

| class | example in these docs | seen by the guard? |
|---|---|---|
| multipliers | `4500×`, `31.6×` (§7, rung 10) | no |
| point-differences | `0.2 points`, `2.7 points` (§7, rung 5) | no |
| microvolts and t-statistics | `+11.89 µV`, `t = +7.71` (§8) | no |
| parameter counts | `2,290 parameters` (§7, rung 10) | no |
| scientific notation | `1.3e-5`, `1.6e-10` (§7, rung 10) | no |
| anything inside a code fence | the ladder block in §6 | no, by design |

**This is not a cosmetic gap.** The two worst defects this project has had to withdraw, the
"0.6 points of real estimator change" decomposition in rung 5, and the "53× too small" chain in
rung 10, are both in classes the extractor structurally cannot match. The guard could run every
day for a year and never see either. Two further limits were verified in the source, and **one of them has since been closed, so this
paragraph is half-withdrawn**. ~~The `INCOMPLETE` branch at `:407-412` returns exit 0 and blames
`--fast` **without checking whether `--fast` was passed**, so a full run in which one registered
script crashes reports pass.~~ **WITHDRAWN 2026-07-25 18:0x: that hole was closed at 17:40:43 and
the wording is kept visible only as the record of what it used to do.** On disk the branch is
gated: `:468 if missing:`, `:477 if args.fast:`, `:479` prints `INCOMPLETE` and returns 0 **only
under `--fast`**, and a full run falls past that gate to
`Result: FAIL (registered script produced no output on a full run)` and returns 1. `:407-412` is
now the unrelated `--list` branch. **The second limit stands, at a new address:** `:486` computes
`bad = bool(unbacked) or bool(stray)`, so the WEAK bucket is advisory and never fails the build.
Absence from the UNBACKED list means *not checked*, not *checked and passed*.

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
   where the crossover should be, so it becomes a **test of a prediction** rather than another
   isolated data point.
5. **Artifact rejection.** ICA-based ocular cleaning, and a paradigm with EOG channels. Rung 7
   demonstrated this project cannot currently monitor the confound it found, let alone remove it.
   The muscle half of this item moved from "unmonitored" to "bounded in one band" on 2026-07-25
   (§10.5b); the ocular half did not move at all, and both still lack a reference electrode of any
   kind, which is what would turn a bound into a measurement.
6. **Trial-count QC, and a live bug inside it.** 12 of 109 subjects reach the sweep with
   non-standard trial counts (36–57 instead of 45), and three record at 128 Hz rather than 160.
   The sweep reports the sampling-rate anomaly and not the timing one. More importantly, **most
   of those non-standard counts are this repo's own fault, not the dataset's**: `sweep_subjects.py`
   epochs at `tmin=-1.0, tmax=4.0`: inherited verbatim from MNE's CSP tutorial, but only ever
   uses the 1–2 s crop, so on subjects whose runs end early the 4-second tail runs off the end of
   the recording and MNE silently drops those trials. The script's own stated rule is "nothing is
   dropped silently," and at the trial level it does not hold. Right-sizing the window to
   `tmin=0.0, tmax=2.0` would recover them and would move the published per-subject numbers.
   Untouched here deliberately: the sweep's figures in this document are what the current script
   prints, and changing the script and the prose in the same pass is how a repo loses track of
   which is which.
7. **Statistics that live only in prose.** Several inferential figures in rungs 7–9, the
   cross-subject CI, rung 9's paired p-values and its subject × method interaction, are computed
   by no committed script, and rungs 8 and 9 persist no per-fold arrays, so a reader must edit the
   source to re-derive them. They were previously described here as "correct but uncomputed"; that
   overstates what is known, since being uncomputed is exactly the condition under which
   correctness has not been established. One of them now carries an open flag: rung 9's
   χ²₁₉ = 13.0 had degrees of freedom that do not match a five-way subject × method interaction
   (§7 rung 9). Rung 11's pre-cue interval **[49.1%, 54.5%]** joins this class as of 2026-07-25:
   it is derived in prose from `regime_decomposition.json`'s stored per-fold values, and the
   script prints the p-value but not the interval. Closing this item means printing the tests, and
   their intervals, from the scripts that own them.
   > **Mostly closed, 2026-07-25, by `inferential_stats.py`.** It computes and prints rung 8's gap
   > and interval, rung 9's four paired tests with Holm alongside them, both families of the
   > subject × method question, the Wilson and McNemar tests for rung 10, and every rung-11 test
   > including the pre-cue interval, from persisted arrays, with a ledger of what it still cannot
   > produce. The unproduced **χ²₁₉ = 13.0 / p = 0.84** is withdrawn and replaced (χ² = 13.33 on
   > 19 df, p = 0.82, plus the Tukey F = 13.4627, p = 0.0005 that the df objection called for).
   > What remains open is *ownership*: `riemannian.py` and `cross_subject.py` still persist no
   > per-fold arrays of their own, so `inferential_stats.py` reads a captured copy rather than a
   > fresh one, and `regime_decomposition.py` still prints no intervals. Closing the item fully
   > means each script printing its own tests.
8. ~~**The ablation arm that would turn a bound into a test.** `ablate_channels.py` has no condition
   that deletes sensorimotor cortex and keeps the rest of the montage, which is the arm that would
   make the artifact control a falsifiable prediction rather than a bound (§8.3). It is cheap: one
   more entry in the `conditions` list, `[c for c in ch_names if c not in SENSORIMOTOR]`, 47
   channels. It is listed here rather than done in this pass because changing the script and the
   prose describing it in the same commit is how a repo loses track of which is which. Note in
   advance that it will be *partially* confounded too, dropping 17 of 64 channels is a
   dimensionality change like any other, so the honest reading will be a comparison against the
   frontopolar arm's 8-channel collapse, not a clean proof.~~
   > **DONE, 2026-07-26, and the item is struck rather than deleted because its own prediction is
   > part of the record.** It was cheap and it was partially confounded, both as forecast. It did
   > not go as the item's tone implied. The 47-channel complement reaches 77.8% (35/45), ten-seed
   > 79.3%, and the prediction did not hold. The dimensionality worry the item raised was itself
   > measured and dismissed *for this comparison*, by deleting 17 channels at random 50 times and
   > watching the ten-seed mean barely move (93.5%). See §10.5. What remains open from this item is
   > only the thing no montage can close: deleting electrodes near a source does not delete the
   > source, so no channel ablation falsifies a source hypothesis in either direction.
9. **The second arm of the muscle check.** `emg_proxy.py` (§10.5b) is one half of a two-part
   condition. The other half is a temporal-channel-**deleted** ablation *inside* 8–30 Hz, which
   should not hurt appreciably if the decoder is not riding myogenic activity. It was declared out
   of scope on purpose so that the probe's pre-registration could not be widened after seeing its
   own result. A conjunction with one arm run is not satisfied, and no sentence in this document
   may be written as though it were.
10. **EMG inside the decoder's own passband.** The probe covers 40–75 Hz minus a notch and is
    blind at 8–30 Hz by construction, which is the only band that can actually contaminate the
    headline. The measured spectrum makes this more urgent rather than less: the recorded high
    band is steeply attenuated, so any myogenic activity present shows up preferentially at the
    low frequencies the probe excludes. Closing it needs either a recording with an EMG reference
    channel or a source-separation method this montage cannot support.
11. **A bursty EMG source is not bounded at all.** The injection ladder plants a constant amplitude
    into every trial of a class, which the probe's own text calls the unrealistic case. At a 25%
    duty cycle the registered detection criterion cannot adjudicate any amplitude tested, so this
    is an open exposure rather than a larger bound. Closing it means recalibrating the criterion
    for intermittent sources, which is a new pre-registration, not a re-run.
12. **Re-scoring `results/sweep_results.csv` on the exact block null.** Run blocking moved one of two
    median subjects across 0.05 (§10.4b), so per-subject significance across the 109 was computed
    with a null that is exact but not the best available. Three subjects is an existence proof
    that the correction *can* move a verdict, never a frequency, and the honest version of rung 6
    needs all 109 re-scored under C3 rather than an extrapolation from two.
13. **Validating the replacement leakage guard against a real leak.** The block null's 99th
    percentile, 53.1% (478/900), replaces an underived 0.60 in `cross_subject.py`. Neither number
    has ever been tested against an actual injected leak, and that script already discloses its
    guard has never fired. A derived threshold is better than a round one, and it is still not a
    measured false-alarm rate.

---

## 13. Reproduce it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline/decode_csp.py        # full pipeline + writes figures/csp_patterns.png
```
- The dataset downloads automatically on first run and is cached in `~/mne_data`, so later runs
  are fast and offline-capable. The first run of rung 6 pulls ~840 MB (all 109 subjects).
- `decode_csp.py` alone reproduces the headline number and the scalp-map figure. Rungs 1–3 are
  optional checkpoints you can run individually to inspect each stage; rungs 5–11 are the
  attacks on the result and each runs standalone.
- Because `random_state=42` is fixed, `decode_csp.py` prints **91.1% (+/- 4.4%)** and
  **p <= 0.001** exactly. The classical rungs are all deterministic in this way.
- **The CNN rungs report a single torch seed, and they are not bit-reproducible.** Seeds are fixed
  for torch, numpy and python. That fixes the *initialisation*; it does not fix the *arithmetic*,
  and on this machine it does not hold. Treat every EEGNet number as one draw, and treat the last
  decimal place of that draw as noise. Run-to-run stability would not be robustness to seed
  *choice* in any case, and no seed sweep is run.

  > **This bullet has been wrong twice, in opposite directions, and both layers are kept.**
  >
  > **Layer one (retracted).** It originally read *"MPS (Apple GPU) kernels are not guaranteed
  > bit-reproducible. Expect small run-to-run drift in the CNN numbers."* That was retracted for
  > reassuring the reader out of noticing the real issue, which is that a single initialisation was
  > being reported as an estimate.
  >
  > **Layer two (retracted, 2026-07-25).** The replacement said *"in practice repeated runs on this
  > machine have come back identical."* **That is false by measurement, and layer one was right on
  > the facts even though it was wrong on the emphasis.** Two runs of a byte-identical
  > configuration disagree: `regime_decomposition.json`'s `original-C` cell (900 trials, 641
  > samples, 4–38 Hz, 0.0–4.0 s) records EEGNet **0.630000**, while a later run of
  > `eegnet_compare.py` prints **63.3%** for the same regime on the same
  > data. That is **0.3 points, 3 trials of 900**. CSP over the same cell reproduces
  > *exactly* (51.4% both places), which is what isolates the disagreement to the CNN rather than
  > to the data or the folds.
  >
  > The lesson is the one this repo keeps relearning in a new costume: "I ran it twice and it
  > looked the same" is not a measurement, and it was asserted here without one. The current
  > bullet claims only what the two artifacts jointly support.
- Rungs 10 and 11 are the slow ones (LOSO with a CNN at every fold). `regime_decomposition.py`
  checkpoints to `regime_decomposition.json` after each cell, so it can be killed and resumed
  without losing completed work.
- **The three guard scripts split into two very different cost classes, and the registry records
  measured runtimes rather than guesses for all of them.** `ablate_channels.py` and
  `emg_proxy.py` are seconds-to-a-minute jobs despite doing four permutation tests and a
  400-run injection ladder between them, because every channel set except all-64 is 8 or 17
  channels wide. `permutation_design.py` is the opposite: a **cold run took about 4 hours
  50 minutes** on this machine, dominated by arm B, which is 4,000 replicates of a 20-fold LOSO
  over 900 pooled trials. It checkpoints each block of replicates to `.permutation_design_cache/`,
  stamped with a fingerprint of every input, so a re-run reuses completed blocks and finishes in
  under three minutes. `check_provenance.py`'s registry carries the **cold** number, because a
  fresh clone has no cache and a timeout there would be reported as a crashed script.
  > **A process note on that script, because it affects what is reproducible.** The first full run
  > was killed partway into arm B. Block checkpointing was added and the script was re-run from
  > scratch, detached, with arm A's output compared across both runs to confirm the refactor
  > changed nothing. The committed script therefore contains caching code the first, discarded run
  > did not have, and the authoritative stdout is the second run's.

---

## 14. The honest limitations (know these before a mentor asks)

**EMPHASIS HERE !** 45 trials total means every confidence interval here is about 17 points
wide. Probably the single biggest limitation on this page !!

The README states these plainly, which is itself a strength, over-claiming is the cardinal sin
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
   gave confidence the model was not riding artifacts. That claim was false**: the showcased
   component is parieto-occipital (§8.3). The real defence is the ablation in
   §10.5: on fists-vs-feet, a frontopolar-only decoder lands at **51.1%**, one trial below the
   majority-class floor. For the left/right rung the same kind of check emphatically fails.

   > **Two things have changed here since 2026-07-25, and one of them is bad news.** The
   > *muscle* half of this item is no longer unmonitored: `emg_proxy.py` refits the pipeline at
   > 40–75 Hz on the temporal ring and lands at 51.1% (23/45), below the floor, with a null
   > univariate arm and a measured sensitivity bound (§10.5b). That bound covers 40–75 Hz only and
   > covers **nothing** inside the decoder's own 8–30 Hz passband. The *ocular* half has not
   > moved. And the sentence this item leans on, that the ablation is "the real defence", is now
   > weaker than it reads: the ablation's own falsifiable form was finally run and did not
   > falsify. Deleting the sensorimotor strip leaves 77.8% (35/45). The frontopolar collapse is
   > still a real negative control; it is just no longer paired with a positive one.

   > **Correction, and it is about vocabulary as much as about numbers.** This item used to read
   > "for fists-vs-feet the ocular checks come back clean (HEOG p = 0.27, VEOG p = 0.44)." Both
   > figures are withdrawn. **EEGMMIDB ships no EOG channels**: which this same page says two
   > bullets down, so nothing here is an HEOG or a VEOG; they were undisclosed *frontal-EEG
   > surrogates* (a lateral AF7−AF8 difference and a mean over Fp1/Fp2), computed by no script,
   > and described with labels that imply electrodes the recording does not have. They were also
   > an underpowered null sold as a clean bill of health: at n=45 that test could not have detected
   > an ocular effect a third the size of the one rung 7 found. The scripted frontopolar ablation
   > above says the same thing honestly and with provenance. **The standing rule from here on:
   > every ocular statement in this repo is a frontal-EEG surrogate and must be named as one.**
   >
   > That used to be written as a statement of fact ("Any ocular statement in this repo *is* a
   > frontal-EEG surrogate and is named as one") and it was false **six lines below itself**:
   > item 5 said the fists-vs-feet headline "passes both its ocular checks," which is an ocular
   > statement, unnamed as a surrogate, inside the same bullet list. A universal that its own page
   > breaks is worse than no universal, so it is demoted to a rule and the counterexample is
   > corrected below.
5. **Two confirmed confounds, in rungs 7 and 11.** Rung 7's left/right decoding rides a
   lateralized gaze artifact, and EEGMMIDB ships no EOG channels, so it can be bounded but
   neither removed nor monitored. Rung 11's wide-window CNN comparison turned out to be reading
   something time-locked to **cue onset**: EEGNet scores **61.1% on the 0–1 s window alone**, and
   adding the whole imagery window on top of it buys nothing distinguishable from zero. That
   window holds the cue flash *and* the first second of imagery, and EEGBCI cannot separate them.
   Neither touches the fists-vs-feet headline directly, but the headline's own artifact defence is
   thinner than this item used to claim, and the claim is corrected here.

   > **Withdrawn, 2026-07-25.** This bullet used to end "…which passes **both its ocular checks**
   > and its channel ablation." There are not two ocular checks on fists-vs-feet, and what exists
   > is a bound rather than a check. Verified by reading `harder_contrast.py`: every
   > frontopolar/ocular analysis in that script (`:233-310`) runs on `epochs_low`, built from
   > `RUNS = [4, 8, 12]`: the **left-vs-right** contrast. The fists-vs-feet reference
   > (`REFERENCE_RUNS = [6, 10, 14]`, `:198-206`) receives only `csp_accuracy` and a permutation
   > test. **No ocular analysis is run on the headline contrast anywhere in that rung.** The only
   > ocular-adjacent evidence on fists-vs-feet in the whole repo is the frontopolar row of
   > `ablate_channels.py`, which is one control, not two, and which the script itself labels
   > *"BOUND, NOT PROOF ... it cannot measure it."* The honest version: the headline has **one**
   > artifact control, a channel ablation that bounds the ocular contribution without measuring
   > it. Together with the two confirmed confounds above, that is why this project treats "the
   > accuracy is real" and "the accuracy is measuring what I said it measures" as two separate
   > claims.
6. **Underpowered comparisons.** With n=20 subjects the minimum detectable difference is roughly
   6–10 points depending on the comparison (`inferential_stats.py` §3: noncentral-t MDEs of 5.68
   to 9.92 points across the five paired tests in rungs 8 and 9). Several "no difference" results
   in rungs 8 and 9 are really "this experiment cannot tell," which is a different statement.
   The subject × method interaction rung 9 leans on used to be listed here as one of them; it is
   not, and the correction runs the other way. **"No interaction exists"** overstated it and
   **"no interaction is detectable here"**: the wording this item used to endorse, is now
   withdrawn too: the design-appropriate omnibus test detects one (Tukey 1-df **F = 13.4627,
   p = 0.0005**). What it detects is a fan localised to the MDM-64 arm rather than the crossover
   per-subject optimality would need, and the *pairwise* test that could see a crossover is null
   and underpowered (χ² = 13.33 on 19 df, p = 0.821, MDE 5.68 points). The retraction it backs, that
   the best method is subject-specific, stands either way, because a positive claim is withdrawn
   for lack of support regardless of the test's power, and a fan is not the support it needs.
7. **Not a real-time system.** This decodes pre-recorded, pre-cut trials offline. A live BCI adds
   continuous decoding, latency, and no clean trial boundaries.
8. **The artifact control's own falsifiable prediction was run and did not hold.** This is the
   newest item and it belongs near the top rather than the bottom, so read it that way. The
   prediction was *if the decoder is reading sensorimotor cortex, deleting sensorimotor cortex
   must break it*. Deleting the strip and keeping the other 47 electrodes leaves 77.8% (35/45),
   ten-seed 79.3%, permutation p ≤ 0.001. The registered verdict is *a loss is suggested and not
   established at n = 45*, because the paired McNemar that had to confirm the loss came in at
   p = 0.0703 on eight discordant trials, which at that count is the most lopsided split that
   still misses. So this is an undecided result and not a refutation of the motor story: the
   average reference leaks the strip into the complement (removing that leak costs 6.0 points),
   the volume conductor means deleting electrodes near a source does not delete the source, and
   n = 45 gives the confirming test essentially no power. What is **not** available is the
   comfortable version, where the ablation is described as a falsifiable test that passed. It was
   run and it did not pass. §10.5.
9. **Two of this project's three newest measurements went against it.** The sensorimotor deletion
   above, and the permutation-null re-analysis, which found that the "correction" this repo
   favored is anti-conservative to 13x nominal and that the published null was already exact
   (§10.4b). Both were pre-registered, both were run anyway, and both are reported at the same
   volume as the results that flattered the project. That is the point of pre-registration and it
   is the only reason those two paragraphs are trustworthy.

None of these are bugs; they are the honest scope of a baseline. The project's credibility comes
precisely from *stating* them, and from the fact that item 4 is written against this document's
own earlier claim rather than quietly edited out. Items 8 and 9 are the harder version of the same
thing: stating a limitation costs nothing, and *running the experiment that could create one* is
what the first seven items were quietly substituting for.

---

## 15. Talking points: how to discuss this in one minute

If a mentor says *"walk me through it,"* this is the spine:

> "It decodes imagined movement, fists vs. feet, from scalp EEG. When you imagine moving,
> the motor rhythm (8–30 Hz mu/beta) drops in power over the specific patch of motor cortex for
> that body part, that's event-related desynchronization. Fists and feet map to different
> places on the motor homunculus, so the *spatial pattern* of that power drop differs. I
> band-pass to the motor band on the continuous signal, before epoching, to keep filter edge
> artifacts out of the trials, then use CSP to recombine all 64 electrodes into a few virtual
> channels whose variance best separates the classes, feed the log-variance into an LDA, and
> cross-validate. 91% versus a 53% chance baseline on one clean subject, with a thousand-shuffle
> permutation test at p under 0.001. My artifact control is an ablation, and it bounds rather than
> proves: on the 17 sensorimotor channels alone it holds at 95.6%, and on the 8 frontopolar channels
> alone, which is where blinks and saccades are loudest, it drops to 51.1%, which is *below* the
> 53.3% majority-class floor, so that decoder is one trial worse than always guessing feet. What it
> rules out is frontopolar variance in the 8 to 30 Hz band as the source. It does not rule out a
> low-frequency ocular contribution. And I should give you the arm that goes the other way before
> you ask for it: if I delete the sensorimotor strip and keep the other 47 electrodes, the decoder
> does *not* die. It sits at 77.8%, which is thirty-five of forty-five, well clear of the floor. So
> the falsifiable version of my own artifact defence was finally run and it did not falsify. My
> pre-registered rule says that is undecided rather than a loss, because the paired McNemar came in
> at p = 0.0703 on eight discordant trials, which at that count is the most lopsided result that
> still misses significance. The honest caveats are that it's one subject and an easy
> contrast, and that subject sits in the top decile of the 109 I swept, where the median is 60%."

And when they ask the follow-up that actually matters, *"what did you get wrong?"*: this is the
stronger half of the answer:

> "Four things, and two of them are the same mistake. I built a left-versus-right rung and
> reported what a harder contrast costs. It was decoding gaze: the cue sits on one side of the
> screen for the whole trial, and in the cue window the frontopolar left-minus-right asymmetry is
> plus-twelve microvolts on left cues and minus-thirteen on right, t of seven-point-seven. A
> decoder on frontopolar mean amplitude alone gets 87% there. Then I compared against a CNN and
> reported that it wins once
> you give it a wider band and a longer window. It wasn't the band or the window. I had also
> moved the crop start without noting it, which let the cue-evoked response into the window. The
> CNN scores 61% on the first second after the cue, and adding the entire imagery window on top
> buys it two points I can't distinguish from zero. I'd be careful about how I say that second
> part: that window holds the cue flash and the first second of imagery both, because the subject
> starts imagining at the cue, so I can't separate them in this dataset. What I can say is that
> the rest of the imagery window adds nothing. Both times the model
> was finding real, reproducible structure that survived permutation testing. It was just the
> wrong structure. The third was a units bug: MNE returns volts, and the network's batch-norm
> epsilon is **seven orders of magnitude** larger than the signal variance, so normalization never
> engaged, it never trained, and it scored exactly the majority-class rate, which reads as 'CNN
> performs at chance on small data,' a completely plausible finding I wrote up as a headline.
>
> "The fourth is the one I'd lead with now, because it isn't about a mechanism. My headline
> artifact control, the channel ablation, was a table in the README that **no script in the repo
> produced**. Two of its four numbers weren't even reachable: with 45 trials tested once each,
> accuracy has to be a multiple of one-forty-fifth, and 95.9% and 47.4% aren't on that grid. The
> real numbers are 95.6% and 51.1%, and 51.1% is *below* the majority-class floor, so the control
> is actually stronger than the version I'd published. I wrote the script, corrected the table, and
> then wrote a second script that pulls the percentages, p-values and trial counts out of my docs
> and fails if no script prints them, because the same defect turned out to be sitting in three
> other places, including inside one of my own retractions. I'd want to say what that guard does
> not do, before you asked. It matches on value, so it cannot tell a right number from a
> coincidence, and it cannot see a multiplier or a difference in points at all. Both of the
> defects I found after building it were in exactly those classes, so it never had a chance at
> either. Absence from its failure list means not checked, not checked and passed."

And there is a third answer now, to the question a good reviewer asks after those two: *"fine, but
what did you find when you tested your own defence?"*

> "The most useful criticism I've had was that I'd got very good at *conceding* problems and had
> never *measured* one. I'd name a hazard, name the exact experiment that would settle it, and
> stop. So I pre-registered three of those experiments and ran them, writing down what each
> possible outcome would mean before the script existed. Two came back against me.
>
> "First: my artifact control was an ablation, and its falsifiable form is that deleting
> sensorimotor cortex should break the decoder. I finally deleted the strip and kept the other
> forty-seven electrodes. The prediction didn't hold. Seventy-eight percent, thirty-five of forty-five, well
> clear of the floor. My pre-registered rule needed two things to call that a real loss and only
> one of them fired, so the verdict is *suggested, not established* and I don't get to write the
> tidy sentence in either direction. I'd also tell you the rule was badly built: the arithmetic
> says its two halves are calibrated to effect sizes that can't both be met, which I could have
> checked before the run and didn't.
>
> "Second: I'd conceded for weeks that my permutation null was built wrong, because scikit-learn
> re-derives the folds from every shuffled label vector. I measured it by running each candidate
> null on data with provably zero information, where every rejection is a false one. My published
> null is exact. The *fix* I'd been planning to apply rejects sixty-five percent of the time at a
> nominal five, because freezing the folds at the true-label split freezes something that depends
> on the labels you're permuting. So I withdrew every number I'd derived from my own correction.
> One half of the objection survived, blocking the shuffle within run, and it moved one subject
> across the line.
>
> "Third, the one that went my way: I refit the pipeline on forty to seventy-five hertz at the
> temporal electrodes to look for muscle. It lands below the majority floor and the univariate
> test is null. But a null is only worth its sensitivity, so I planted synthetic sources until the
> probe could see them, and the honest bound is the worst shape and worst direction, not the best.
> And I'd flag the three holes in it myself: it's blind inside my own eight-to-thirty band, which
> is the only band that can contaminate the headline; it can't adjudicate a bursty source at all;
> and there's no EMG reference channel in this corpus, so it measures high-band power at
> muscle-adjacent sites rather than muscle."

That answer is worth more than the 91%, and the third part is worth more than the first two.
Anyone can report an accuracy. Most careful people can report a limitation. The thing a lab is
actually screening for is whether you run the experiment that could take your result away from
you, and then say what it returned.

If you can say all three, you own this repo.

---

## 16. Mini-glossary

- **BCI**: Brain–Computer Interface. Turning brain signals into commands.
- **EEG**: Electroencephalography. Recording brain electrical activity from scalp electrodes.
- **Motor imagery**: Imagining a movement without performing it.
- **Homunculus**: The body-map layout of motor/sensory cortex; hands lateral, feet central.
- **Mu / beta rhythms**: ~8–12 Hz / ~13–30 Hz oscillations over idling motor cortex.
- **ERD (Event-Related Desynchronization)**: The drop in mu/beta power when a region engages.
- **Raw / Epochs**: MNE objects for a continuous recording vs. a stack of cut trials.
- **Annotation / event**: A timestamped cue marker; the ground-truth label source.
- **Montage**: The 3-D scalp positions of the electrodes.
- **Average reference**: Re-referencing each channel to the mean of all channels.
- **Band-pass filter**: Keep only frequencies in a chosen band (here 8–30 Hz).
- **Edge artifact**: Filter distortion at the start/end of the filtered segment.
- **CSP (Common Spatial Patterns)**: Learns channel-mixings that maximize the between-class
  variance ratio; the core spatial-filtering step.
- **Log-variance feature**: Log of a virtual channel's variance = its band power; the feature CSP feeds LDA.
- **LDA (Linear Discriminant Analysis)**: Simple linear classifier separating two Gaussian feature clouds.
- **Cross-validation**: Repeatedly train and test on different splits to get an honest accuracy.
- **StratifiedKFold**: A cross-validation *partition* that tests every trial exactly once and
  preserves class balance in each fold. What this project uses.
- **ShuffleSplit**: Independently resampled train/test splits. Neither a partition nor
  stratified; the estimator this project moved away from.
- **Chance level**: Accuracy of always guessing the majority class (here 53.3%), *not* 50%.
- **Permutation test**: Shuffle the labels many times, re-run the whole pipeline, and see where
  the real result falls in the resulting null distribution.
- **Exact test**: a test whose false-positive rate is at most its nominal level when the null is
  true. Checkable empirically by running it on data with provably no signal, which is what
  `permutation_design.py` does.
- **Anti-conservative**: a test that rejects more often than its nominal rate. It looks like extra
  sensitivity and is actually a broken instrument, because the extra rejections are false ones.
- **Ancillary**: a quantity you condition on that does not depend on what you are permuting.
  Freezing a fold partition is only valid if the partition is ancillary; a partition stratified on
  the true labels is not.
- **Block permutation**: shuffling labels *within* blocks (here, within run, or within subject)
  rather than across the whole set. A smaller reference set and therefore a weaker assumption.
- **McNemar test**: the paired test for two classifiers on the same trials. It looks only at the
  trials where they disagree, so its power is set by the *discordant* count, not by n.
- **Ablation**: Removing a part of the input to test whether the model depends on it. The
  artifact control this project relies on.
- **LOSO (leave-one-subject-out)**: Train on N−1 people, test on the held-out one, rotate.
- **Degenerate classifier**: A model predicting one class for every trial. It scores the
  majority-class rate, which is easily mistaken for chance performance.
- **EEGNet**: A compact CNN for EEG; the deep-learning comparison in rungs 10 and 11.
- **MDM / Tangent Space**: Riemannian classifiers operating on covariance matrices: nearest
  class mean along the manifold, and a projection to a flat space where ordinary classifiers work.

---

## 17. The result section README.md carried until 2026-08-26, in full

Moved here verbatim from the README's `## Result` on 2026-08-26, when the README was cut
down to a summary table. Nothing below was reworded in the move, so "this page" and
"this README" in the text mean the README this was written on.

| Metric | Value |
|---|---|
| CSP+LDA accuracy (stratified 5-fold CV) | **91.1%** |
| Chance (majority class) | 53.3% |
| Permutation test (1000 shuffles) | **p ≤ 0.001** (null 50.7% ± 8.5%) |
| Wilson 95% CI on n=45 | [79.3%, 96.5%] |
| Per-fold scores | 8/9, 8/9, 8/9, 8/9, 9/9 |
| Trials | 45 (21 hands, 24 feet), one subject |

Trying to emphasize how every number on this page carries its scope. A percentage without a
denominator and a subject count isn't easy to understand or calibrate from.

The per-fold row appears instead of a ± because a 9-trial test set can only score multiples of
1/9. A standard deviation over those five values is a step on that ladder, not a spread, which is
the same objection that retired the earlier "± 5.6%" below. Take the Wilson interval as the
honest uncertainty, and as mildly optimistic: it treats 45 cross-validated predictions as
independent draws from one model when they come from five. **That optimism is named here and not
currently quantified.** An earlier revision of this table quantified it, from a variance-inflation
factor and an `n_eff`-corrected interval computed on the fixed-partition cell C4, and that cell
was withdrawn on exchangeability grounds, so every figure derived from it is withdrawn with it.
The same applies to the 10,000-draw row this table used to carry. See
`docs/process/OVERRIDE-RULING-2026-07-30.md` §1.5, which enumerates them. Requantifying the optimism from a
cell that survives is open work, not a finished result. The permutation p is reported as
**p ≤ 0.001** rather than `0.0010` because 1/1001 is the resolution floor of a 1000-shuffle test,
not a measurement; the scripts print the bound directly.

Across 1000 label shuffles, not one matched or exceeded the real result, so the
decoding is finding real structure rather than fitting noise. ("Matched or
exceeded" rather than "beat" because that is the comparison scikit-learn
actually counts.)

![Permutation null distribution](figures/permutation_null.png)

### The null's design was challenged, `permutation_design.py` measured it, and the challenge was mostly wrong

The objection: `permutation_test_score` re-derives the folds from *each shuffled label
vector*, so every replicate is scored on a different partition while the observed 91.1%
is scored on the partition stratified on the **true** labels. A second objection sits one
level up: the labels are shuffled i.i.d. across all 45 trials, which treats runs 6, 10
and 14 as one exchangeable pool when run is a blocking factor. Both were conceded in
prose here for weeks and neither was measured.

`permutation_design.py` runs the full 2x2 at 10,000 draws per cell on three subjects,
with one list of permuted label vectors feeding both partition rules so the contrast is
paired. **Then it tests the tests.** A permutation rule is exact only if, when the null
is true by construction, it rejects at most at its nominal rate. That is checkable with
no EEG at all: run a majority-class classifier on an all-zero feature matrix, where there
is provably nothing to decode, and count how often each rule calls it significant.

| rule | what it is | false-positive rate at α = 0.05 (subject 1) |
|---|---|---|
| re-stratified, i.i.d. (**the published null**) | C1 | 0.0000 |
| re-stratified, within-run | C3 | 0.0000 |
| **fixed at the true-label partition, i.i.d.** | C2 | **0.6550**, 13.1x nominal |
| **fixed at the true-label partition, within-run** | C4 | **0.6600**, 13.2x nominal |
| fixed at a label-*free* `KFold` partition | `KF_free`, which is C5's rule | 0.0250 |
| labels permuted *inside* each fold of the true-label partition | `WITHIN` | 0.0000 |

**The direction of the error is the opposite of the objection.** The cells this project
built as "the correction" are the ones that are not tests, and the published null is
exact. Freezing the folds at `StratifiedKFold(...).split(X, y_true)` freezes a quantity
that is *a function of the labels being permuted*, so the observed vector becomes the
unique point in the reference set whose fold margins are balanced. Across the three
subjects, 4 of 6 fixed-at-true-label cells are anti-conservative and 0 of 12 re-stratified
or label-free cells are. On the real pipeline the same defect shows up as a size of
0.0752 against C1's 0.0444, and as rejecting at 29/45 where the exact partner needs
30/45. That is how a verdict gets flipped by the rule rather than by the data.

So every number this project computed from C2 and C4 is withdrawn, including the
n_eff-corrected Wilson interval. The exact re-analysis is narrower:

| subject | C1 (published, exact) | C3 (within-run, exact) | verdict change |
|---|---|---|---|
| 1 (headline) | p ≤ 9.999e-05, C = 0 | p ≤ 9.999e-05, C = 0 | none, and uninformative by construction |
| 17 (median) | p = 0.11529 | p = 0.052795 | none; factor 2.18, no crossing |
| 19 (median) | p = 0.055494 | **p = 0.044496** | crosses 0.05 downward |

One of the two objections survives, then, and only for one of the two median subjects:
**run blocking is a real correction and it moved one verdict.** The headline is untouched
either way. Subject 1's effect is about 4.7 null standard deviations out, so its p is
pinned at the resolution floor in every cell no matter which null is used; that is a fact
about the effect size, **not** evidence the design was right, and it was registered as
near-certain before the run. Under C3 the two median subjects **disagree** at 0.05, which
is exactly the pre-registered neutral outcome, and no conclusion is drawn from a
disagreement between two subjects at n = 45 each.

Subject 1's Wilson interval, corrected for the exact null's variance inflation, is
[77.0%, 96.9%] at n_eff 34.0 against the face-value [79.3%, 96.5%].

The cross-subject arm is the one where the objection holds cleanly, because the defect
there is in the *reference set* rather than the partition: `cross_subject.py` shuffles all
900 pooled labels globally, which can deal a 45-trial subject a class split the protocol
could not produce. Under the correct within-subject block permutation the 59.4%
(535/900) cross-subject result survives at p ≤ 0.00049975, and the global null is
measurably inflated, sd(global)/sd(block) = 1.5730. The deliverable is a replacement for
that script's underived `SHUFFLE_MAX = 0.60` leakage guard: the block null's 99th
percentile, **53.1% (478/900)**, or 53.4% (481/900) at the 99.5th.

> **A registered falsification gate fired on this run and was overridden, and that has
> not changed.** Subject 17's C4 null centres at 43.80%, outside the pre-registered 45%
> to 55% band, and the pre-registration's consequence clause then bars reporting anything
> in its Sections 6.1 to 6.4 from this run, which is both arms, not only the one that
> tripped it. It is reported anyway, on an argument added *after* a smoke run tripped the
> assert. The justification is a control rather than an argument: the same all-zero dummy
> classifier shows the identical downward shift under a fixed partition (40.67% for
> subject 1 against its 53.33% majority rate), so sub-45% centring is a property of the
> partition rule and not of a mis-specified null. The exact label-free cells C5 and C6
> centre below the band too, which is independent confirmation that assert 9 was the wrong
> assert. Note what that does and does not buy: the band argument was the right diagnosis
> of the wrong problem, and it was still not a defence of C2 and C4, whose defect turned
> out to be exchangeability rather than centring. Two separate defects, and fixing the
> first did not fix the second. **A reader who rejects the override should treat both arms
> as unreported from this run**: Section 6.4 is arm B, so the cross-subject p, the sd
> ratio and both replacement thresholds in the paragraph above fall with arm A. That is a
> wider blast radius than the exchangeability defect has, `LeaveOneGroupOut` reads only
> `groups`, so arm B never had that defect, and wider than arm B's own centring earns,
> since neither of its nulls leaves the band. The registered remedy is a run-level halt
> and it does not ask why a cell is fine. The 59.4% (535/900) itself stands either way:
> `cross_subject.py` produced it, and this run only re-tested it.

The evidence that this is motor activity and not eye or muscle artifact is an
**ablation**, not a picture. Every row is printed by `ablate_channels.py`, one
seed (42), one pipeline, on the same 45 trials:

| Channels used | ch | Accuracy | Trials |
|---|---|---|---|
| sensorimotor only (FC/C/CP strip) | 17 | **95.6%** | 43/45 |
| all 64 | 64 | 91.1% | 41/45 |
| frontopolar only (Fp/AF ring) | 8 | **51.1%**: *below* the 53.3% majority-class floor | 23/45 |
| **sensorimotor strip DELETED**, 47 kept | 47 | **77.8%** | 35/45 |
| wide FC/C/CP strip deleted, 43 kept | 43 | 71.1% | 32/45 |
| leave-one-run-out (all 64) | 64 | 93.3% | 42/45 |

Refit on the eight electrodes where blinks and saccades are loudest and the
decoder falls to the majority-class floor: 51.1% is one trial *worse* than
ignoring the EEG and always answering "feet", with folds scattered from 0.33 to
0.78. One limit on what that bounds: frontopolar-only keeps 8 of 64 channels and
deletes the other 56, so the collapse is confounded with an eightfold cut in
feature dimension. It **bounds** the ocular contribution rather
than proving the signal is motor. That is still a control; a scalp map is not.

### Deleting the 17-channel sensorimotor strip: the falsifiable prediction did not hold

Rows four and five are new, and they are the ones that cost something. Until
2026-07-26 this README said, in bold, that **"no condition here deletes sensorimotor
cortex while keeping the rest of the montage,"** and EXPLAINER.md §8.3 named the exact
experiment that would close it. That concession is kept above as the record of what was
owed. Here is the measurement instead.

Delete the 17-channel FC/C/CP strip, keep the other 47 electrodes, refit the entire
published pipeline, and the decoder does **not** collapse. It lands at 77.8% (35/45) at
seed 42 and 79.3% over ten seeds, range [75.6%, 84.4%], with a 1000-shuffle permutation
test at p ≤ 0.001 against a null of 51.0% ± 8.8%, and a Wilson 95% interval of
[63.7%, 87.5%] whose lower bound clears the 53.3% majority floor by more than ten
points. The stricter deletion, which also removes FC5/FC6/CP5/CP6, still reaches
71.1% (32/45) at seed 42 and 76.7% over ten seeds.

So the falsifiable form of the artifact defence, *"if it reads sensorimotor cortex,
deleting sensorimotor cortex must break it,"* was finally run and **the prediction did not
hold**. That is the unflattering result and it is the headline of this section.

What the pre-registration allows to be concluded is narrower than that sentence, in both
directions. The decision statistic was fixed in advance as G = (all-64 ten-seed mean)
minus (complement ten-seed mean), with a real loss requiring **both** G > 10.0 points
**and** an exact McNemar at p < 0.05. G came out at 94.0 − 79.3 = +14.7 points, which
clears its threshold. The McNemar did not: on the paired seed-42 predictions the 2x2 is
34 both correct, 7 all-64 only, 1 complement only, 3 neither, giving p = 0.0703 on eight
discordant trials. The registered rule requires both halves, so the verdict is
**a loss is suggested and not established at n = 45**, and the sentence *"the strip is
sufficient but not necessary"* is therefore **not written**, even though G points that
way.

Read that p with its discordant count attached or not at all. At eight discordant trials
the observed 7-vs-1 split *is* the most lopsided split that still misses p < 0.05; only
a clean 8-0 would have reached it. The two arms agree on 34 of 45 trials and both miss 3
more, which is what leaves the count small. "Not established" here means **underpowered**,
not "evidence of no difference". The script prints the arithmetic that makes this worse:
the McNemar half of the rule cannot reach p < 0.05 below a six-trial gap, which is 13.3
points, while the G half fires at 10.0 points, so between 10.0 and 13.3 points the
conjunctive rule **cannot fire no matter what the data does**. That is a defect in the
rule, found by the run that the rule was written for.

Three further things bound this arm, and two of them cut against the unflattering
reading:

- **The average reference leaks.** It is computed over all 64 electrodes *before* the 47
  are picked, so every surviving channel carries −1/64 of every deleted one. Re-referencing
  inside the complement's own 47 costs it 6.0 points over ten seeds (79.3% to 73.3%, and
  35/45 to 31/45 at seed 42). Part of the complement's score is what volume conduction
  plus a 64-channel average reference *predicts*.
- **It is not "sensorimotor cortex deleted."** The 17-channel list has no FC5, FC6, CP5 or
  CP6, so the complement keeps four peri-Rolandic electrodes. The wide-21 deletion above
  removes them and still lands far above the floor, so that leak does not explain the
  result.
- **Channel count is not the explanation, for this comparison.** Deleting 17 channels *at
  random*, 50 times, barely moves the ten-seed mean (93.5%, G null centred at +0.5 points,
  range [−1.8, +3.3]). Not one of the 50 draws reached the observed +14.7. Deleting the
  strip costs far more than deleting the same number of arbitrary electrodes. That control
  does not transfer to the 17-channel and 8-channel rows, which keep far fewer channels
  than it ever does.

**The instrument limit is true in every direction and should be stated before any of the
numbers.** No channel-deletion experiment on a 64-channel scalp montage can falsify a
source hypothesis, because deleting the electrodes nearest a source does not delete the
source from the remaining electrodes. Forward-is-not-inverse refutes a negative source
claim exactly as hard as a positive one. Everything here is a sensor-space claim.

One thing that arm is explicitly **not** allowed to say: the 47 surviving electrodes
include T7, T8, T9, T10, TP7 and TP8, which is temporalis muscle territory, and they
also include POz, PO4 and Oz, the peak of the strongest CSP component. "Posterior cortex
also decodes" is a sentence this table cannot support. The permitted sentence is *"the
47 non-strip electrodes decode above the floor."* Which is why the next section exists.

### Muscle: a probe instead of a concession

Until 2026-07-25 the muscle hypothesis was handled by a sentence saying nobody had
measured it. `emg_proxy.py` measures it. Surface EMG is broadband and sits mostly above
the mu/beta band, so the probe refits the **unmodified** CSP+LDA pipeline on 40 to 75 Hz
with a 60 Hz notch, restricted to the eight-channel temporal ring
(T7 T8 T9 T10 TP7 TP8 FT7 FT8). If class information is riding on jaw or temporalis
activity, a muscle-band decoder at muscle-territory electrodes should find it.

It does not. The primary cell lands at **51.1% (23/45)**, *below* the 53.3%
majority-class floor, permutation p = 0.5175. All four channel sets are at or below the
floor in that band (temporal 23/45, all-64 18/45, sensorimotor 17/45, frontopolar 15/45),
and a univariate arm agrees: log band power does not differ by class
(Welch t p = 0.6922, Mann-Whitney U p = 0.7074, 0 of 8 channels surviving Holm). The
positive control passes first, in the same script: 8 to 30 Hz on all 64 channels
reproduces 91.1% (41/45) exactly.

Do not read "below the floor" as "worse than chance". The floor is what a *constant*
predictor scores, and this pipeline is not a constant predictor even on shuffled labels:
the temporal ring's own permutation null has a median of 23/45, and the observed 23/45
sits at its 48th to 57th percentile. Against the correct reference the probe is **at
chance**, which is the claim. The other three channel sets sit in the lower tail of their
own nulls, which is a different phenomenon that the script names and deliberately does
not explain, because attaching a mechanism to a number in the same breath is the failure
mode this whole pre-registration exists to prevent.

**A null is only worth what its sensitivity is worth, so the script measures that too.**
It plants a synthetic class-correlated source in the temporal ring at a ladder of
amplitudes and finds the smallest one this probe can see. Report the *worst* cell, never
the most flattering: over four source shapes and both class directions the detection
threshold is **a = 0.600** of T8's own high-band standard deviation, where a = 0.150 is
the easiest cell. So the honest statement is a bound with three explicit holes in it:

- **Spectrally**, it covers 40 to 75 Hz minus the notch, out of a recorded 0 to 80 Hz.
  The headline decoder lives at 8 to 30 Hz, where this probe is blind by construction.
  **EMG inside the decoder's own passband remains entirely unbounded**, and the measured
  spectrum makes that worse rather than better, because the high band is steeply
  attenuated so any EMG present shows up preferentially where this probe is not looking.
- **Temporally**, the ladder injects a constant amplitude into every trial of a class,
  which is the *shifted distribution* the script itself calls unrealistic. A bursty
  source in 25% of trials is **not bounded at any amplitude tested**, up to ten times the
  ladder's top rung, because the registered detection criterion cannot adjudicate that
  duty cycle at all. That exposure is open, not closed.
- **Statistically**, 45 trials buys a large-effect bound and nothing finer: the univariate
  arm can only exclude Cohen's d at or above 0.837 aggregate, 1.069 per channel.

The word this run is allowed to use is **bounds**, not eliminates. What is bounded is
worth having anyway: inside 40 to 75 Hz at the temporal ring, this recording contains no
class-correlated broadband source large enough for the probe to see, and no aggregate
log-power difference as large as a very large effect. EEGMMIDB also ships no EMG
reference channel, so this measures high-band power at muscle-adjacent scalp sites; it
does not measure muscle.

> **Two sentences this section printed on 2026-07-25 and withdrew on 2026-07-26.** The
> first was *"this recording contains no class-correlated broadband temporal source as
> large as a = 0.300 times T8's own high-band SD"*, and the second was *"the line
> 'nothing bounds an EMG contribution' is now false."* Both overstated, in the same
> direction, three times over. The 0.300 was the worst of **two** registered shapes, both
> diffuse; adding the canonical focal one-electrode source doubles it to 0.600, and a
> focal source at 0.500 sits *inside* this recording's tolerance while sitting *outside*
> the number that was published. The bound is also band-scoped and continuous-source-scoped,
> as above. The corpus line is false **for 40 to 75 Hz only** and remains true inside
> 8 to 30 Hz.

The other direction is weaker than this README used to claim it. Sensorimotor-only
is 43/45 against all-64's 41/45, a **two-trial** difference, which on n=45 is
inside noise. The defensible statement is *"dropping 47 non-motor channels does
not hurt"*, not *"the sensorimotor subset is better."* The load-bearing half of
the ablation is the collapse, not the gain.

> **Correction, and it is the reason `ablate_channels.py` now exists.** Until this
> commit the two bolded rows read **95.9%** and **47.4%, i.e. chance**, and *no
> script in the repo produced them*. Both are also arithmetically unreachable:
> with 45 trials tested exactly once each, accuracy can only be k/45, steps of
> 2.222%, and neither 0.959 nor 0.474 is on that lattice. The real values are
> 43/45 = 95.6% and 23/45 = 51.1%. The framing was wrong twice over too:
> frontopolar-only is not "chance," because chance here is the 53.3%
> majority-class rate, not 50%. The table is kept in corrected form rather than
> deleted, because a headline control that no code produced is the most useful
> thing this repo found about itself.
>
> One honest limit on what the ablation bounds: the average reference is computed
> over all 64 channels *before* any subset is picked, so the subsets are not
> electrically independent, every channel carries −1/64 of every other. This
> **bounds** the ocular contribution; it does not eliminate it.

The learned CSP patterns are plotted below because they are interesting, **not as
proof**. An earlier version of this README claimed they were "focal over central
sensorimotor cortex" and offered that as the artifact defence. That was wrong:
one of the four plotted components peaks at **POz, PO4 and Oz**, which are
parieto-occipital electrode positions. A second component mixes
sensorimotor weights (FC3/C3/FC1, FC4/FC2/C4) with occipital ones. Reading
topographies by eye is not a control, which is why the ablation above replaced it.

> **A second correction, inside the first one.** This paragraph used to add that
> the showcased component "correlates r = 0.57 with this subject's own eyes-closed
> alpha map." No script in this repo computes any correlation, and the figure did
> not reproduce under the obvious definitions. It has been withdrawn rather than
> re-derived: the **location** claim, that the component is posterior, stands on
> the channel weights; its **oscillatory** character was never measured, because
> no script in this repo computes a per-component spectrum, so "alpha-like" is
> withdrawn too. A retraction passage resting on an unproduced number would be
> the exact defect it was written to correct.

![CSP spatial patterns](figures/csp_patterns.png)

### A note on the earlier number

An earlier version of this README reported **94.4% ± 5.6%** under "10-fold CV."
Two things were wrong with that. The evaluation was `ShuffleSplit`, which is not
k-fold: it never tested 5 of the 45 trials while testing others up to 6 times, and
it left class balance swinging from 2:7 to 7:2 across folds. And the ± 5.6% was a
quantization artifact rather than a spread, because a 9-trial test set can only
score multiples of 1/9, so the ten folds landed on just two distinct values.

Stratified 5-fold puts the figure at 91.1%, and the lower number is the better
claim because it now carries a significance test.

**But most of that 3.3-point drop is seed placement, not a correction.** Sweeping 100
cross-validation seeds gives a mean of 93.6% for ShuffleSplit and 93.8% for
stratified 5-fold, so the two estimators agree in expectation to about 0.2 points,
and stratified k-fold is the *higher* of the two. The switch of estimator therefore
did not lower the headline; in expectation it raises it slightly. What moved the
number is where seed 42 happens to fall: the 49th percentile of the retracted
estimator (94.4%) and the **3rd percentile** of the published one (91.1%), 2.7
points below that estimator's own 93.8% mean. Seed placement accounts for
0.8 + 2.7 = 3.5 points and the estimator offsets 0.2 points in the opposite
direction, which is why the visible drop is 3.3. The switch is still the right
call, for coverage and stratification reasons, but presenting it as an integrity
correction would be its own small dishonesty.

> **Correction, and the arithmetic is what gives it away.** Until this revision this
> paragraph closed with *"the estimator change is still right, for coverage and
> stratification reasons, but it is worth roughly 0.6 points, not 3.3."* That 0.6 is
> `94.4 − 93.8`, which differences seed 42's ShuffleSplit **draw** against
> StratifiedKFold's **mean**. It silently moves 0.8 points of old-estimator seed luck
> onto the estimator's account and flips the sign of the estimator term, which over
> 100 seeds is `93.6 − 93.8 = −0.2`: the estimator *raises* the expectation rather
> than lowering it, so it is arithmetically incapable of having lowered anything. The
> split survived review because the total came out right, `0.6 + 2.7 = 3.3`. The
> total was right; the split was wrong. Worse, the paragraph contradicted itself,
> stating the correct 0.2 three lines above the wrong 0.6.

**So the headline understates itself, deliberately and on the record.** 91.1% is a
low draw from its own estimator: the same pipeline averages **93.8%** across 100
cross-validation seeds, over a range of 88.9–97.8%. The number published here is
the conservative one, and it is the one carrying the permutation test.

Read those percentiles with one caveat the script makes visible: the estimator is
quantized to 1/45, so the 100 seeds land on only a handful of distinct values and
many of them tie exactly on 91.1%. `evaluate_honestly.py` ranks seeds *strictly
below*, which is the most flattering of the available tie conventions, counting
ties as at-or-below would place seed 42 materially higher. The 2.7-point gap
between 91.1% and the 93.8% mean does not depend on the convention; the word
"3rd" does.

`evaluate_honestly.py` reproduces the whole comparison, sweeping both estimators
so that neither one's diagnostics get attached to the other's number.

## 18. The script guide and the guard notes the README carried, in full

Moved here verbatim from the README's `## Reproduce` section on 2026-08-26, same move as
§17. The README keeps the three repo-map tables; the rung-by-rung guide and the notes on
`check_provenance.py`'s buckets live here now.

### Scripts (built rung by rung)

Rungs 1–4 build the result. Rungs 5–11 attack it, and five of them found something wrong:
rung 5 (the reported precision was a quantization artifact, and 5 of 45 trials were never
tested), rung 6 (the BCI-illiteracy inference was backwards, what was observed is evidence
*of* signal), rung 7 (a gaze confound in this project's own data), rung 10 (a units bug meant
the CNN was never training), and rung 11 (the "EEGNet wins" regime was the model reading the
cue, not the imagery).

> **Correction.** This line used to read *"Rungs 5–11 attack it, and three of them found
> something wrong."* Three is what §12 of EXPLAINER.md names in prose, and it undercounts.
> The retraction register in §12.1 traces round-one retractions to attack rungs 6, 7, 9, 10 and
> 11, and rung 5's correction of the headline's precision is documented in "A note on the
> earlier number" above. The five named in the sentence are 5, 6, 7, 10 and 11. Rung 9 is
> arguable and is deliberately left out: its retraction ("no method dominates") replaced an
> asserted null with an undetectable one, which is a correction to how the rung was *described*
> rather than a defect the rung caught. Counting it would make six.

| # | Script | Does |
|---|---|---|
| 1 | `load_and_plot.py` | Load one run, plot raw EEG → `figures/raw_eeg.png` |
| 2 | `epoch_trials.py` | Cut runs 6/10/14 into labeled hands/feet trials |
| 3 | `filter_and_epoch.py` | Add 8–30 Hz band-pass + average reference |
| 4 | `decode_csp.py` | CSP + LDA, cross-validated, permutation test, spatial patterns |
| 5 | `evaluate_honestly.py` | Stress-test the number: stratification, coverage, permutation test, 100-seed sweep of both estimators |
| 6 | `sweep_subjects.py` | All 109 subjects, per-subject chance, against the pure-noise expectation |
| 7 | `harder_contrast.py` | Left vs. right fist (runs 4/8/12). Found a lateralized gaze confound |
| 8 | `cross_subject.py` | Leave-one-subject-out across 20 subjects |
| 9 | `riemannian.py` | MDM and Tangent Space vs. the CSP baseline on identical folds |
| 10 | `eegnet_compare.py` | EEGNet vs. CSP+LDA at two sample sizes. Where the units bug was found |
| 11 | `regime_decomposition.py` | Decomposes rung 10's confounded third regime. The "EEGNet wins" result is the CNN reading the **cue**, not the imagery. Includes the pre-cue control |

Four more scripts exist that are controls on the repo rather than rungs of the ladder:

| Script | Does |
|---|---|
| `ablate_channels.py` | Produces the artifact-ablation table above including the sensorimotor-deleted arm, and asserts every reported accuracy lands on the k/45 lattice |
| `emg_proxy.py` | Refits the pipeline on 40–75 Hz at the temporal ring, and converts the null into a numeric sensitivity bound with an injection ladder |
| `permutation_design.py` | Tests the permutation tests. Measures each null's false-positive rate on data with provably zero information, then re-scores every result on the cells that survive |
| `check_provenance.py` | Extracts the figure classes its own docstring names, percentages, p-values, correlations, and integers bound to trials/subjects/shuffles/seeds/folds, from README.md and EXPLAINER.md, and fails if one of *those* is not printed by some script's stdout |

The first three are the three exposures this repo used to concede in prose and now
measures. Two of the three came back **against** the framing they were built to defend:
the sensorimotor-strip deletion left the decoder well above the majority floor, and the permutation objection turned
out to be right about the cross-subject null and wrong about the within-subject one, in
the direction that says the project's own "correction" was the invalid test.

`check_provenance.py` exists because of the defect the ablation table turned out to
be: a specific figure, published as a headline control, produced by no code. It is
the guard that makes that class of error loud instead of silent. It is not a guard
against every kind of wrong number: the patterns it matches do not see point
differences, multipliers, µV values, t-statistics or parameter counts, so a claim
written in any of those forms is invisible to it however often it runs. Matching is
also by value rather than by meaning, which the docstring states plainly.

Three things about it are worth knowing before you run it. It has a **WEAK** bucket for
a number whose only backing line reads as a retraction, without that, a script
printing "95.9% and 47.4% are off this lattice" *in order to withdraw them* would have
marked the fabricated originals as sourced, defeating the whole point on this repo.

WEAK is a *softer* verdict than UNBACKED, not a harder one. A claim can only reach it
if some script really did print that number, on a line whose wording reads as a
withdrawal. A number no script prints at all is UNBACKED, and UNBACKED is what fails
the run. WEAK does not: `bad` is computed from UNBACKED and unregistered scripts only,
so the bucket is advisory and has to be read by hand.

Reading it by hand matters, because it is not made up only of withdrawn figures.
`RETRACTION_HINT` matches on words, and one of the words it matches is *retracted*
inside the estimator label `ShuffleSplit (retracted)`. So the live, correctly measured
93.6% quoted above lands in WEAK next to the genuinely withdrawn 95.9% and 47.4%.
Six of the twenty-four WEAK rows in the 2026-07-26 run are current measurements of
that kind. The withdrawn figures do stay flagged, which is the intended state: the
figure is gone from every live claim and kept only in the record of its own
withdrawal. But the bucket does not mean "withdrawn," and it never meant "unproduced."

> **A count in this paragraph was stale and is corrected 2026-07-26.** It read *"Three of the
> fifteen WEAK rows in the 2026-07-25 cold run."* The bucket has grown to twenty-four rows as the
> retraction register has grown, which is the expected direction: every withdrawn figure that
> some script still prints in the act of withdrawing it lands here forever. A hand-counted total
> written into prose goes stale the moment the next retraction lands, and this one did. It is
> corrected rather than removed because the *ratio* is the point: most of the bucket is genuinely
> withdrawn figures, and a minority are live measurements caught by the word "retracted" sitting
> inside an estimator label.

One claim in these two documents is currently **UNBACKED**, so `check_provenance.py` exits FAIL as
of 2026-07-26. It is a withdrawn ocular-decoder figure quoted inside its own withdrawal at
EXPLAINER.md §7 rung 7, and it is deliberately not repeated here. No script computes it, which is
correct, because the claim is retracted. That is a gap in the guard's design rather than a defect
in the prose: the WEAK bucket exists precisely for withdrawn numbers, but it can only catch them
when some script happens to print the number in the act of retracting it. A figure that was never
produced by any code, and now survives only in the record of its own withdrawal, is
indistinguishable to this tool from a fabrication. It is left failing rather than papered over,
because deleting a retraction to turn a guard green is the exact trade this repo exists to refuse.

> **Correction, on this section's account of its own guard.** Two claims here are
> withdrawn. The script table used to say `check_provenance.py` *"extracts every number
> from README.md and EXPLAINER.md"*; it extracts the classes its docstring names, and
> the figure classes it cannot see are exactly the ones this repo's confirmed errors
> have lived in. And this passage used to explain the WEAK bucket as flagging withdrawn
> figures *"because by construction no script produces them any more."* That is
> backwards. No production means UNBACKED. WEAK means the number **was** produced, on a
> line that reads like a withdrawal, which is why three live measurements sit in the
> bucket today. Both sentences overstated the guard in the direction that makes it look
> stronger, which is the direction this repo has to be most suspicious of.

If you got this far: genuinely thank you ! if you see something I got wrong or am inaccurate
about PLEASE tell me !! :3
