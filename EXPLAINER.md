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
feet*. On one clean subject it reaches **94% accuracy** (chance is 53%), and it visualizes
the brain map the model learned to prove it locked onto real motor cortex, not noise. This is
the canonical "hello world" of brain–computer interfaces (BCIs), done honestly and reproducibly.

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
- **Structure:** each subject did 14 runs. Different runs are different *tasks*:
  - Some runs = real movement; some = **imagined** movement.
  - **Runs 6, 10, 14 = Task 4 = imagine both fists vs. both feet.** These are the three runs
    this project uses. Using all three gives us more trials than one run alone.

### 4.1 Annotations: how the data knows what the subject was doing
Inside each EDF file are **annotations** — timestamped markers the experimenters recorded when
they showed the subject a cue. Three labels appear:
- **T0** = rest (do nothing). *We drop these.*
- **T1** = the cue to imagine **fists** (in these runs). We call this class **"hands"**.
- **T2** = the cue to imagine **feet**. Class **"feet"**.

These annotations are the *ground-truth labels*. Without them we'd have EEG but no idea what
the person was told to imagine, and supervised learning would be impossible.

> ⚠️ Gotcha worth knowing for a mentor conversation: T1/T2 mean *different things in different
> runs*. In the hand runs (3/7/11) T1/T2 mean left/right fist. In *these* runs (6/10/14) T1 =
> both fists, T2 = both feet. The code hard-codes the runs so this mapping is correct — but if
> you ever swap runs, the labels silently change meaning. This is a classic EEGBCI footgun.

---

## 5. The software stack (`requirements.txt`)

The pinned versions matter for reproducibility, but conceptually there are only four libraries
doing real work:

| Library | Role in this project |
|---|---|
| **MNE** (`mne==1.12.1`) | The EEG/MEG workhorse. Downloads the data, reads EDF, holds the signal in `Raw`/`Epochs` objects, does filtering, referencing, epoching, and even ships the CSP implementation and the scalp-map plotting. **~90% of the domain logic is MNE.** |
| **scikit-learn** (`scikit-learn==1.9.0`) | The generic ML layer: `LinearDiscriminantAnalysis` (the classifier), `Pipeline` (chain CSP→LDA), and `cross_val_score`/`ShuffleSplit` (honest evaluation). |
| **NumPy** (`numpy==2.5.1`) | Array math under everything; used directly only for the chance-level calculation and rounding. |
| **matplotlib** (`matplotlib==3.11.0`) | Renders the two PNGs (raw signal, CSP scalp maps). |

Everything else in `requirements.txt` (certifi, scipy, joblib, pooch, tqdm, pillow, …) is a
transitive dependency — pulled in *by* the four above, not used directly by your code. `pooch`
is worth a mention: it's the downloader MNE uses to fetch and cache the dataset.

`.gitignore` keeps the virtual environment (`.venv/`) and Python bytecode caches out of git —
standard hygiene so the repo stays just source + results.

---

## 6. The architecture: a "ladder" of four scripts

The defining design choice of this repo is that it's built **rung by rung**. Each script is a
complete, runnable checkpoint that adds exactly one new idea on top of the previous one. The
git history mirrors this — one commit per rung. This is deliberate and worth articulating to a
mentor: *it makes the pipeline debuggable and teachable, because every stage can be run and
inspected in isolation before the next stage is added.*

```
load_and_plot.py      Rung 1:  Can I load the data and see a signal?
        │
        ▼
epoch_trials.py       Rung 2:  Can I cut it into labeled trials?
        │
        ▼
filter_and_epoch.py   Rung 3:  Can I isolate the motor rhythms first?
        │
        ▼
decode_csp.py         Rung 4:  Can I actually classify it — and prove it's real?
```

`decode_csp.py` is the *only* script you need to run to reproduce the headline result; it
re-does the work of rungs 1–3 internally. The earlier scripts are kept as **inspectable
checkpoints**, not as an import chain (nothing imports anything else — each file is standalone).

The next four sections walk each rung in depth.

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
cv  = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
scores = cross_val_score(clf, train_data, labels, cv=cv)
```
CSP and LDA are explained in full in Sections 8 and 9; cross-validation in Section 10.

Then it computes and prints the honest scoreboard:
```python
chance = max(np.mean(labels == 2), np.mean(labels == 3))   # majority-class baseline
```
and finally **visualizes what CSP learned** by fitting it on all trials and drawing the top 4
spatial patterns as scalp maps → **`csp_patterns.png`**. This plot is the credibility check
(Section 8.3).

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

### 8.3 The `csp_patterns.png` plot — *why it's the honesty check*
`csp.plot_patterns(...)` draws each spatial filter as a **scalp map** (a top-down head with a
color heatmap). The README's screenshot shows the learned patterns are **focal over central /
sensorimotor cortex** — bright spots right where hands and feet motor areas live (§3.1).

**Why this matters enormously:** a model can hit 94% by cheating — locking onto an eye-blink
artifact, a neck-muscle tension that happens to correlate with the cue, or a per-run drift.
If CSP had learned those, the scalp maps would light up at the *edges* of the head (eyes,
temples, neck), not the center. Because the patterns are centered on motor cortex, you have
**physiological evidence the model found real motor sources.** Accuracy alone never proves this;
the scalp map does. Being able to say *"and here's how I know it's not an artifact"* is what
separates a credible BCI result from a lucky number.

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

### 10.2 ShuffleSplit ×10
`ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)`:
- Randomly hold out **20%** of trials (≈9) as a test set, train on the other 80%, score.
- Repeat **10 times** with different random splits.
- Report the **mean ± standard deviation** across the 10 runs.
- `random_state=42` fixes the randomness so the result is **reproducible** — you get the same
  94.4% every time you run it.

The reported **±5.6%** spread is not a footnote — it's honesty. It tells you the estimate is
*noisy* (because n is small), so you shouldn't over-trust the exact 94.4%.

### 10.3 The chance baseline
```python
chance = max(mean(labels==2), mean(labels==3))   # = 24/45 = 53.3%
```
A model that always guesses the majority class ("feet") would be right 53.3% of the time. So the
number that matters is not "94%" in a vacuum but **"94% vs. a 53% floor"** — the model is doing
far better than guessing. *Always report accuracy against chance;* a raw accuracy with no
baseline is a red flag a mentor will catch immediately.

### 10.4 How to read the final printout
```
CSP+LDA accuracy: 94.4%  (+/- 5.6%)
Chance (majority class): 53.3%
Per-fold: [1.   0.89 1.   0.89 ...]
```
The per-fold list shows individual splits ranged from ~89% to 100% — consistent with a real,
strong effect plus small-sample noise.

---

## 11. The tunable knobs (your iteration surface)

Every constant near the top of the scripts is a lever. Here's what each does and what happens
if you turn it:

| Knob | Current | What it controls | Try changing it to… |
|---|---|---|---|
| `SUBJECT` | 1 | Which person | Loop over all 109; report mean accuracy — the real test of the method (see §12). |
| `RUNS` | 6,10,14 | Which task | **3,7,11** = imagine *left vs. right fist* — a much harder contrast (same homunculus strip). Accuracy will drop; that's the point. |
| `L_FREQ,H_FREQ` | 8,30 | The band kept | Split into mu (8–12) and beta (13–30) separately (filter-bank CSP) and combine — often a real accuracy gain. |
| CSP `n_components` | 4 | # spatial filters | 6 or 8 — more filters can help or overfit; cross-validate to decide. |
| CSP `reg` | None | Covariance shrinkage | `'ledoit_wolf'` — stabilizes CSP when trials are few or channels many; important for cross-subject. |
| crop `1.0–2.0 s` | 1 s window | Which slice = features | Widen to 0.5–2.5 s, or slide the window; the imagery signal isn't perfectly time-locked. |
| `cv` splits/test_size | 10 / 0.2 | Evaluation rigor | Stratified K-fold; or **leave-one-subject-out** once you go multi-subject. |
| `random_state` | 42 | Reproducibility seed | Change it and re-run to feel how much the 94% wobbles with n=45. |

---

## 12. How to extend it (concrete next projects, roughly in order of value)

1. **Multi-subject sweep (highest value, lowest effort).** Wrap the whole pipeline in a loop
   over subjects 1–109, collect per-subject accuracy, and plot the distribution. This converts
   "94% on one lucky subject" into a defensible claim about the *method*. Expect a wide spread —
   some subjects are "BCI illiterate" and barely beat chance. This single change is what turns a
   demo into a result.
2. **Cross-subject generalization.** Train on subjects 1–108, test on 109 (leave-one-subject-out).
   This is much harder — every brain/skull is different — and is where naive CSP struggles.
   Motivates transfer-learning and covariance-alignment methods.
3. **Harder contrasts.** Switch to runs 3/7/11 (left vs. right fist). Lower accuracy but far
   more useful for a real BCI (a left/right decision drives a cursor).
4. **Filter-bank CSP (FBCSP).** Run CSP separately in several sub-bands and let the classifier
   combine them. A well-known, reliable accuracy bump — and still classical/interpretable.
5. **Swap in EEGNet (the README's stated "Next").** A compact convolutional neural net that
   learns spatial + temporal filters end-to-end, no hand-designed CSP. The right way to compare
   is *against this CSP+LDA baseline on the exact same splits* — if the CNN can't beat a simple
   baseline on 45 trials, that itself is the finding (it usually needs more data to shine).
6. **Riemannian geometry methods.** Classify the covariance matrices directly on their curved
   (Riemannian) manifold (e.g. `pyriemann`). Current state-of-the-art for classical BCI and a
   strong, still-interpretable alternative to deep learning.

---

## 13. Reproduce it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python decode_csp.py        # full pipeline + writes csp_patterns.png
```
- The dataset downloads automatically on first run and is cached in `~/mne_data` (so later runs
  are fast and offline-capable).
- `decode_csp.py` alone reproduces the headline number and the scalp-map figure. The other three
  scripts are optional checkpoints you can run individually to inspect each stage.
- Because `random_state=42` is fixed, you should get **94.4% ± 5.6%** exactly.

---

## 14. The honest limitations (know these before a mentor asks)

The README states these plainly, which is itself a strength — over-claiming is the cardinal sin
in BCI. Be ready to volunteer them:

1. **Within-subject, small-n.** One subject, 45 trials. The result says nothing about whether
   this works on a *new* person. The ±5.6% spread openly signals the estimate is noisy.
2. **Easy contrast.** Fists-vs-feet are far apart on the homunculus, so their scalp patterns
   differ a lot. This is close to the easiest possible motor-imagery discrimination.
3. **Clean subject.** Per-subject quality varies enormously across the 109 subjects; subject 1
   is a good recording. Picking it is fair for a baseline demo but not representative.
4. **No artifact rejection.** There's no explicit removal of eye-blinks or muscle artifacts. The
   band-pass and the *centered* CSP patterns give confidence the model isn't riding artifacts,
   but a production pipeline would add ICA-based artifact cleaning.
5. **Not a real-time system.** This decodes pre-recorded, pre-cut trials offline. A live BCI adds
   the hard problems of continuous decoding, latency, and no clean trial boundaries.

None of these are bugs — they're the honest scope of a baseline. The project's credibility comes
precisely from *stating* them instead of hiding behind the 94%.

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
> cross-validate. 94% versus a 53% chance baseline on one clean subject. And I plotted the CSP
> patterns — they're focal over central sensorimotor cortex, which is my evidence the model
> locked onto real motor sources, not eye or muscle artifacts. The honest caveats are that it's
> one subject, small-n, and an easy contrast; the natural next step is a 109-subject sweep and
> an EEGNet comparison against this baseline."

If you can say that, you own this repo.

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
- **Cross-validation / ShuffleSplit** — Repeatedly train/test on different random splits for an honest accuracy.
- **Chance level** — Accuracy of always guessing the majority class (here 53.3%).
- **EEGNet** — A compact CNN for EEG; the intended deep-learning comparison.
```
