# EEG Motor-Imagery Decoding

I wanted to find out whether I could read *imagined* movement off the scalp, and then
whether I could prove the number I got was real. This repository is both halves of that.

When you imagine moving, your sensorimotor cortex changes its mu (8–12 Hz) and beta
(13–30 Hz) rhythm power in a spatially specific way. This project reads that pattern with
a classic **CSP + LDA** baseline and guesses which movement was imagined.

The decoder itself is a tutorial baseline. What I think
is worth looking at is mainly the second half: the verification suite I built to attack my own result, and the
corrections I had to publish. The headline used to read 94.4%. It reads 91.1% now,
and the reason is written up next to it :)

**Dataset:** [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/)
motor-imagery set (109 subjects, 64-channel EEG @ 160 Hz), loaded via
`mne.datasets.eegbci`. This baseline uses **subject 1**, runs 6/10/14
(Task 4 = imagine *both fists* vs. *both feet*).

## Repo map: what to read, in what order

There are a lot of scripts here, and they aren't all equally worth your time. They fall into
three groups, one folder each: `src/pipeline/`, `src/attacks/`, `src/checks/`. **If you read three files, read `src/pipeline/decode_csp.py`,
`src/attacks/evaluate_honestly.py` and `src/attacks/ablate_channels.py`, in that order.**

**1. Building the result** (rungs 1–4). Start here.

| Script | What it does |
|---|---|
| `src/pipeline/load_and_plot.py` | Load one run, look at the raw signal |
| `src/pipeline/filter_and_epoch.py` | Band-pass to 8–30 Hz, average reference |
| `src/pipeline/epoch_trials.py` | Cut the continuous signal into labeled trials |
| **`src/pipeline/decode_csp.py`** | **The pipeline and the headline number. The entry point.** |

**2. Attacking the result** (rungs 5–11). This is the half that matters.

| Script | What it found |
|---|---|
| **`src/attacks/evaluate_honestly.py`** | **Moved the headline 94.4% → 91.1%** |
| `src/attacks/sweep_subjects.py` | Median across all 109 subjects is 60.0%, not 91.1% |
| `src/attacks/cross_subject.py` | 59.4% on an unseen person, with no calibration |
| `src/attacks/harder_contrast.py` | Left vs right fist is much harder, and gaze is a confound |
| **`src/attacks/ablate_channels.py`** | **The artifact control, and a prediction I lost** |
| `src/attacks/emg_proxy.py` | Whether jaw muscle could be driving it (bounded, not closed) |
| `src/attacks/riemannian.py` | Covariance geometry as an alternative classifier |
| `src/attacks/eegnet_compare.py` | A CNN comparison, and the units bug that faked a finding |
| `src/attacks/regime_decomposition.py` | What the CNN was actually decoding (the cue, not the imagery) |

**3. Checking the checks.** Infrastructure. Read only if you want to audit.

| Script | Job |
|---|---|
| `src/checks/permutation_design.py` | Tests whether the permutation null is itself valid |
| `src/checks/validity_gate.py` | The registered independence gate |
| `src/checks/inferential_stats.py` | Every CI and p-value in the docs, computed in one place |
| `src/checks/check_provenance.py` | Refuses numbers in the docs that no script produces |
| `src/checks/check_wording.py` | Catches banned phrasings and retracted claims |
| `src/checks/test_pipeline.py` | 21 regression tests, one per mistake actually made |
| `src/common.py` | The shared pipeline definition, imported by everything |

**Directories:** `src/` all the code, sorted into the three groups above, with
`src/common.py` the shared definition · `prereg/` the registered designs, each dated in `prereg/README.md` ·
`figures/` generated plots · `results/` generated data · `.provenance_cache/` captured stdout
that `check_provenance.py` checks the docs against.

## Pipeline

```
raw EEG  →  average reference  →  band-pass 8–30 Hz  →  epoch around cues
         →  crop to 1–2 s imagery window  →  CSP spatial filters
         →  log-variance features  →  LDA  →  cross-validated accuracy
```

- **Filtering** keeps only the mu/beta motor rhythms; applied to the
  continuous signal (not epochs) to avoid edge artifacts.
- **CSP (Common Spatial Patterns)** learns channel-weightings that maximize
  the variance ratio between classes, a handful of spatial filters that
  separate hands from feet. The log-variance along the top axes is the feature.
- **LDA** separates the two feature clouds with a linear boundary.
- **Stratified 5-fold cross-validation** tests every trial exactly once and
  holds class balance steady across folds.
- **A 1000-shuffle permutation test** asks whether the result could have come
  from chance at all.

## Result

| Metric | Value |
|---|---|
| CSP+LDA accuracy, subject 1 (stratified 5-fold CV) | **91.1%** (41/45) |
| Chance (majority class) | 53.3% |
| Permutation test (1000 shuffles) | **p ≤ 0.001** (null 50.7% ± 8.5%) |
| Wilson 95% CI on n=45 | [79.3%, 96.5%] |
| Per-fold scores | 8/9, 8/9, 8/9, 8/9, 9/9 |
| Trials | 45 (21 hands, 24 feet), one subject |
| Median across all 109 subjects, same pipeline | 60.0% |
| Cross-subject (leave-one-subject-out, 20 subjects) | 59.4% (535/900) |

The full treatment, including the ablation and muscle controls, the permutation-design
audit, and every correction along the way, is in [EXPLAINER.md](EXPLAINER.md) §17.

## Honest limitations

- **Within-subject, small-n.** One subject, 45 trials. The number doesn't claim cross-subject
  generalization, and the honest interval is roughly [79%, 97%].
- **Easy contrast.** Fists vs. feet are far apart on the motor homunculus (lateral vs.
  top-central), so their scalp patterns differ a lot. Left-hand-vs-right-hand is harder, and
  `harder_contrast.py` found it's also **gaze-confounded** in this dataset. The cue sits on one
  side of the screen for the whole trial, and on subject 1 the frontopolar asymmetry
  (Fp1+AF7+AF3 minus Fp2+AF8+AF4) is **+11.89 µV on left cues and −12.99 µV on right cues** in
  the cue window (Welch t = +7.71, p = 3.7e-09). A decoder using **frontopolar mean amplitude
  alone reaches 86.7%** on that window (p ≤ 0.001).

  > **Correction.** This bullet used to say "a decoder using only frontopolar channels at
  > 0.5–5 Hz matches the 64-channel result on this subject." At matched settings it doesn't.
  > Same trials, same folds, same 1.0–2.0 s window: all-64 at 8–30 Hz is 73.3%, frontopolar at
  > 0.5–5 Hz is **53.3%**: 20 points below, not level. The old "match" compared a frontopolar
  > decoder on the whole 0–4 s epoch against a 64-channel decoder on a 1-second crop, which is
  > two different experiments. The confound is real; the evidence quoted for it was
  > window-shopped, and CSP's log-variance features are close to blind to it anyway, because a
  > sustained gaze deviation is a DC shift. Swapping the *feature* to mean amplitude is what
  > actually finds it.

- **Clean subject.** Subject 1 is the **91st percentile** of the 109; the median subject scores
  60.0%. Picking it's fair for a baseline, and quoting it without the distribution wouldn't be.
- **No artifact rejection.** No ICA, and EEGMMIDB ships **no EOG channels**, so ocular
  contamination can be bounded by ablation but never removed or directly measured. Any "ocular
  check" in this repo is a *frontal-EEG surrogate*, not an eye electrode, and it's named as one
  wherever it appears. The same holds for muscle: there's no EMG reference channel either, so
  `emg_proxy.py` is a *high-band-power-at-muscle-adjacent-sites* probe and is named as one. It
  bounds a myogenic contribution inside 40 to 75 Hz and bounds **nothing** inside the decoder's
  own 8 to 30 Hz passband.
- **The falsifiable artifact test was run and didn't falsify.** Deleting the sensorimotor strip
  leaves 77.8% (35/45). That's the single most important caveat on this page, it's written up
  in full in [EXPLAINER.md](EXPLAINER.md) §17 rather than buried here, and its registered
  verdict is *suggested, not established*.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline/decode_csp.py        # full pipeline + CSP patterns → figures/csp_patterns.png
```

Data downloads automatically on first run (cached in `~/mne_data`).

### Tests

```bash
python src/checks/test_pipeline.py       # or, with pytest installed: python -m pytest src/checks/test_pipeline.py -q
```

21 regression tests, under a second, no data download. They aren't "does it run" tests.
Each one guards a mistake this project actually made and had to retract, so a future edit
that reintroduces it fails here instead of in this README. The three that matter most:

- **95.9% and 47.4% aren't attainable accuracies.** With 45 trials tested exactly once
  each, accuracy is a count over 45, so it can only land on multiples of 2.222%. Both
  numbers were published here as measurements. Neither is on the lattice, so neither ever
  was one.
- **CSP sits inside the Pipeline.** If it's fitted once outside, the spatial filters see
  the test trials and every accuracy in this repo is invalid.
- **`for_torch` rejects volts-scale data.** Feeding EEGNet volts leaves BatchNorm unable to
  normalize; the network scores exactly the majority-class rate while being dead, which
  reads as a plausible finding and isn't one.

Shared definitions live in `common.py`: the classifier, the Wilson interval, Holm
correction, the channel sets and the loader. They used to be copied into each script,
because importing a script that defined them also ran its multi-minute analysis. Every
script has a `__main__` guard now, so importing costs nothing.

The statistics helpers are genuinely shared. The classifier is not, and the README used to say
otherwise. Seven scripts still build their own CSP+LDA, and `decode_csp.py`, the one that
produces the headline number, imports nothing from `common.py` at all. All of those
constructions put CSP inside the Pipeline, so nothing leaks and no published number is affected,
but `common.py` isn't yet one definition of what "the published pipeline" means, and claiming it
was is an overclaim I'd rather correct than keep.

### Why `.provenance_cache/` is committed

The cache is the evidence pool: `src/checks/check_provenance.py` resolves every number in
README.md and EXPLAINER.md into some script's captured stdout, and those captures live in
`.provenance_cache/`, keyed by script name with a source hash in `meta.json`. If the cache
were untracked, a fresh clone couldn't check a single claim without first re-running every
registered script, which is hours of compute, and the slowest script alone is most of a
working day cold. A claim that can only be checked by people who already have the cache is
not checkable, so the cache ships with the repo. Entries are verbatim stdout, never edited,
and the recorded source hash stops matching the
moment the registered script itself changes. That hash covers the entry file alone, not the
modules it imports, so since the 2026-08-26 split into `src/pipeline`, `src/attacks` and
`src/checks`, an edit inside an imported module leaves the key unchanged and the entry gets
reported as cached. I tested it: a within-subject count changed from 41 to 39 inside a split
module and the check still passed the script. That's a real hole and it's mine to close. The longer form of this argument sits where the rule is enforced,
in `.gitignore`.

## Next

Cross-subject, harder contrasts and the EEGNet comparison are all built now (rungs 7–11). What
is actually left:

- **More trials per subject.** Almost every limitation above traces back to 45 trials: the
  quantized folds, the underpowered comparisons, the learning curve that can't be run. Public
  corpora exist with 2,000–5,000 trials per subject. It's also what would give the
  sensorimotor-deletion McNemar enough discordant trials to decide anything.
- **Filter-bank CSP (FBCSP).** CSP per sub-band, combined by the classifier. A reliable gain, and
  still classical and interpretable.
- **The learning curve**, holding subject fixed and sweeping training-set size. That would settle
  a claim this project made and then retracted, that the barrier is sample size rather than
  anatomy.
- **ICA-based artifact rejection**, and a paradigm that records EOG.
- **The second arm of the muscle check.** `emg_proxy.py` is one half of a two-part condition:
  the high-band probe at muscle territory, which came back null. The other half is a
  temporal-channel-**deleted** ablation *inside* 8–30 Hz, which shouldn't hurt appreciably if
  the decoder isn't riding muscle. It hasn't been run, and a conjunction with one arm run is
  not satisfied.
- **EMG inside the decoder's own passband.** The probe covers 40–75 Hz and is blind at 8–30 Hz
  by construction, which is the only band the headline can actually be contaminated in. Closing
  that needs an instrument this montage and this sampling rate can't provide.
- **Re-scoring `results/sweep_results.csv` on an exact null.** Run blocking moved one of the two median
  subjects across 0.05, so per-subject significance across the 109 was computed with a null
  that's exact but not the best available. Three subjects is an existence proof, not a survey.

See [EXPLAINER.md](EXPLAINER.md) §12 for the full scoreboard, including the complete list of
claims this project published and later retracted.

The list only grows :sob: good to keep track & iterate, iterate, iterate tho.
