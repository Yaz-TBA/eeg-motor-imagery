# EEG Motor-Imagery Decoding

Decoding *imagined* movement from scalp EEG with a classic **CSP + LDA**
baseline. When you imagine moving, your sensorimotor cortex changes its
mu (8–12 Hz) and beta (13–30 Hz) rhythm power in a spatially specific way;
this project reads that pattern to guess which movement was imagined.

**Dataset:** [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/)
motor-imagery set (109 subjects, 64-channel EEG @ 160 Hz), loaded via
`mne.datasets.eegbci`. This baseline uses **subject 1**, runs 6/10/14
(Task 4 = imagine *both fists* vs. *both feet*).

## Pipeline

```
raw EEG  →  average reference  →  band-pass 8–30 Hz  →  epoch around cues
         →  crop to 1–2 s imagery window  →  CSP spatial filters
         →  log-variance features  →  LDA  →  cross-validated accuracy
```

- **Filtering** keeps only the mu/beta motor rhythms; applied to the
  continuous signal (not epochs) to avoid edge artifacts.
- **CSP (Common Spatial Patterns)** learns channel-weightings that maximize
  the variance ratio between classes — a handful of spatial filters that
  separate hands from feet. The log-variance along the top axes is the feature.
- **LDA** separates the two feature clouds with a linear boundary.
- **Stratified 5-fold cross-validation** tests every trial exactly once and
  holds class balance steady across folds.
- **A 1000-shuffle permutation test** asks whether the result could have come
  from chance at all.

## Result

| Metric | Value |
|---|---|
| CSP+LDA accuracy (stratified 5-fold CV) | **91.1%** |
| Chance (majority class) | 53.3% |
| Permutation test (1000 shuffles) | **p ≤ 0.001** (null 50.7% ± 8.5%) |
| Wilson 95% CI on n=45 | [79.3%, 96.5%] |
| Per-fold scores | 8/9, 8/9, 8/9, 8/9, 9/9 |
| Trials | 45 (21 hands, 24 feet), one subject |

The per-fold row appears instead of a ± because a 9-trial test set can only score multiples of
1/9. A standard deviation over those five values is a step on that ladder, not a spread, which is
the same objection that retired the earlier "± 5.6%" below. Take the Wilson interval as the
honest uncertainty, and as mildly optimistic: it treats 45 cross-validated predictions as
independent draws from one model when they come from five. The permutation p is reported as
**p ≤ 0.001** rather than `0.0010` because 1/1001 is the resolution floor of a 1000-shuffle test,
not a measurement; the scripts print the bound directly.

Across 1000 label shuffles, not one matched or exceeded the real result, so the
decoding is finding real structure rather than fitting noise. ("Matched or
exceeded" rather than "beat" because that is the comparison scikit-learn
actually counts.)

![Permutation null distribution](permutation_null.png)

The evidence that this is motor activity and not eye or muscle artifact is an
**ablation**, not a picture. Every row is printed by `ablate_channels.py`, one
seed (42), one pipeline, on the same 45 trials:

| Channels used | ch | Accuracy | Trials |
|---|---|---|---|
| sensorimotor only (FC/C/CP strip) | 17 | **95.6%** | 43/45 |
| all 64 | 64 | 91.1% | 41/45 |
| frontopolar only (Fp/AF ring) | 8 | **51.1%** — *below* the 53.3% majority-class floor | 23/45 |
| leave-one-run-out (all 64) | 64 | 93.3% | 42/45 |

Remove the cortex that should carry the signal and the decoder falls to the
majority-class floor — 51.1% is one trial *worse* than ignoring the EEG and
always answering "feet", with folds scattered from 0.33 to 0.78. That is a
control; a scalp map is not.

The other direction is weaker than this README used to claim it. Sensorimotor-only
is 43/45 against all-64's 41/45 — a **two-trial** difference, which on n=45 is
inside noise. The defensible statement is *"dropping 47 non-motor channels does
not hurt"*, not *"the sensorimotor subset is better."* The load-bearing half of
the ablation is the collapse, not the gain.

> **Correction, and it is the reason `ablate_channels.py` now exists.** Until this
> commit the two bolded rows read **95.9%** and **47.4%, i.e. chance**, and *no
> script in the repo produced them*. Both are also arithmetically unreachable:
> with 45 trials tested exactly once each, accuracy can only be k/45 — steps of
> 2.222% — and neither 0.959 nor 0.474 is on that lattice. The real values are
> 43/45 = 95.6% and 23/45 = 51.1%. The framing was wrong twice over too:
> frontopolar-only is not "chance," because chance here is the 53.3%
> majority-class rate, not 50%. The table is kept in corrected form rather than
> deleted, because a headline control that no code produced is the most useful
> thing this repo found about itself.
>
> One honest limit on what the ablation bounds: the average reference is computed
> over all 64 channels *before* any subset is picked, so the subsets are not
> electrically independent — every channel carries −1/64 of every other. This
> **bounds** the ocular contribution; it does not eliminate it.

The learned CSP patterns are plotted below because they are interesting, **not as
proof**. An earlier version of this README claimed they were "focal over central
sensorimotor cortex" and offered that as the artifact defence. That was wrong:
the strongest component here peaks at **POz, PO4 and Oz**, which is
parieto-occipital and looks like occipital alpha. A second component mixes
sensorimotor weights (FC3/C3/FC1, FC4/FC2/C4) with occipital ones. Reading
topographies by eye is not a control, which is why the ablation above replaced it.

> **A second correction, inside the first one.** This paragraph used to add that
> the showcased component "correlates r = 0.57 with this subject's own eyes-closed
> alpha map." No script in this repo computes any correlation, and the figure did
> not reproduce under the obvious definitions. It has been withdrawn rather than
> re-derived: the qualitative claim (that component is posterior and alpha-like)
> stands on the channel weights, and a retraction passage resting on an
> unproduced number would be the exact defect it was written to correct.

![CSP spatial patterns](csp_patterns.png)

### A note on the earlier number

An earlier version of this README reported **94.4% ± 5.6%** under "10-fold CV."
Two things were wrong with that. The evaluation was `ShuffleSplit`, which is not
k-fold: it never tested 5 of the 45 trials while testing others up to 6 times, and
it left class balance swinging from 2:7 to 7:2 across folds. And the ± 5.6% was a
quantization artifact rather than a spread, because a 9-trial test set can only
score multiples of 1/9, so the ten folds landed on just two distinct values.

Stratified 5-fold puts the figure at 91.1%, and the lower number is the better
claim because it now carries a significance test.

**But most of that 3.3-point drop is seed luck, not a correction.** Sweeping 100
cross-validation seeds gives a mean of 93.6% for ShuffleSplit and 93.8% for
stratified 5-fold, so the two estimators agree in expectation to about 0.2 points.
Seed 42 happens to sit at the 49th percentile for the old estimator and the **3rd
percentile** for the new one. The estimator change is still right, for coverage
and stratification reasons, but it is worth roughly 0.6 points, not 3.3.

**So the headline understates itself, deliberately and on the record.** 91.1% is a
low draw from its own estimator: the same pipeline averages **93.8%** across 100
cross-validation seeds, over a range of 88.9–97.8%. The number published here is
the conservative one, and it is the one carrying the permutation test.

Read those percentiles with one caveat the script makes visible: the estimator is
quantized to 1/45, so the 100 seeds land on only a handful of distinct values and
many of them tie exactly on 91.1%. `evaluate_honestly.py` ranks seeds *strictly
below*, which is the most flattering of the available tie conventions — counting
ties as at-or-below would place seed 42 materially higher. The 2.7-point gap
between 91.1% and the 93.8% mean does not depend on the convention; the word
"3rd" does.

`evaluate_honestly.py` reproduces the whole comparison, sweeping both estimators
so that neither one's diagnostics get attached to the other's number.

## Honest limitations

- **Within-subject, small-n.** One subject, 45 trials. The number does not claim cross-subject
  generalization, and the honest interval is roughly [79%, 97%].
- **Easy contrast.** Fists vs. feet are far apart on the motor homunculus (lateral vs.
  top-central), so their scalp patterns differ a lot. Left-hand-vs-right-hand is harder, and
  `harder_contrast.py` found it is also **gaze-confounded** in this dataset. The cue sits on one
  side of the screen for the whole trial, and on subject 1 the frontopolar asymmetry
  (Fp1+AF7+AF3 minus Fp2+AF8+AF4) is **+11.89 µV on left cues and −12.99 µV on right cues** in
  the cue window (Welch t = +7.71, p = 3.7e-09). A decoder using **frontopolar mean amplitude
  alone reaches 86.7%** on that window (p ≤ 0.001).

  > **Correction.** This bullet used to say "a decoder using only frontopolar channels at
  > 0.5–5 Hz matches the 64-channel result on this subject." At matched settings it does not.
  > Same trials, same folds, same 1.0–2.0 s window: all-64 at 8–30 Hz is 73.3%, frontopolar at
  > 0.5–5 Hz is **53.3%** — 20 points below, not level. The old "match" compared a frontopolar
  > decoder on the whole 0–4 s epoch against a 64-channel decoder on a 1-second crop, which is
  > two different experiments. The confound is real; the evidence quoted for it was
  > window-shopped, and CSP's log-variance features are close to blind to it anyway, because a
  > sustained gaze deviation is a DC shift. Swapping the *feature* to mean amplitude is what
  > actually finds it.

- **Clean subject.** Subject 1 is the **91st percentile** of the 109; the median subject scores
  60.0%. Picking it is fair for a baseline, and quoting it without the distribution would not be.
- **No artifact rejection.** No ICA, and EEGMMIDB ships **no EOG channels**, so ocular
  contamination can be bounded by ablation but never removed or directly measured. Any "ocular
  check" in this repo is a *frontal-EEG surrogate*, not an eye electrode, and it is named as one
  wherever it appears.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python decode_csp.py        # full pipeline + CSP patterns → csp_patterns.png
```

Data downloads automatically on first run (cached in `~/mne_data`).

### Scripts (built rung by rung)

Rungs 1–4 build the result. Rungs 5–11 attack it, and three of them found something wrong.

| # | Script | Does |
|---|---|---|
| 1 | `load_and_plot.py` | Load one run, plot raw EEG → `raw_eeg.png` |
| 2 | `epoch_trials.py` | Cut runs 6/10/14 into labeled hands/feet trials |
| 3 | `filter_and_epoch.py` | Add 8–30 Hz band-pass + average reference |
| 4 | `decode_csp.py` | CSP + LDA, cross-validated, permutation test, spatial patterns |
| 5 | `evaluate_honestly.py` | Stress-test the number: stratification, coverage, permutation test, 100-seed sweep of both estimators |
| 6 | `sweep_subjects.py` | All 109 subjects, per-subject chance, against the pure-noise expectation |
| 7 | `harder_contrast.py` | Left vs. right fist (runs 4/8/12). Found a lateralised gaze confound |
| 8 | `cross_subject.py` | Leave-one-subject-out across 20 subjects |
| 9 | `riemannian.py` | MDM and Tangent Space vs. the CSP baseline on identical folds |
| 10 | `eegnet_compare.py` | EEGNet vs. CSP+LDA at two sample sizes. Where the units bug was found |
| 11 | `regime_decomposition.py` | Decomposes rung 10's confounded third regime. The "EEGNet wins" result is the CNN reading the **cue**, not the imagery. Includes the pre-cue control |

Two more scripts exist that are controls on the repo rather than rungs of the ladder:

| Script | Does |
|---|---|
| `ablate_channels.py` | Produces the artifact-ablation table above, and asserts every reported accuracy lands on the k/45 lattice |
| `check_provenance.py` | Extracts every number from README.md and EXPLAINER.md and fails if one is not printed by some script's stdout |

`check_provenance.py` exists because of the defect the ablation table turned out to
be: a specific figure, published as a headline control, produced by no code. It is
the guard that makes that class of error loud instead of silent.

Two things about it are worth knowing before you run it. It has a **WEAK** bucket for
a number whose only backing line reads as a retraction — without that, a script
printing "95.9% and 47.4% are off this lattice" *in order to withdraw them* would have
marked the fabricated originals as sourced, defeating the whole point on this repo. And
it will always flag the handful of **withdrawn** figures that this project quotes inside
its retraction passages, because by construction no script produces them any more. That
is the intended state: the figure is gone from every live claim and kept only in the
record of its own withdrawal.

## Next

Cross-subject, harder contrasts and the EEGNet comparison are all built now (rungs 7–11). What
is actually left:

- **More trials per subject.** Almost every limitation above traces back to 45 trials: the
  quantized folds, the underpowered comparisons, the learning curve that cannot be run. Public
  corpora exist with 2,000–5,000 trials per subject.
- **Filter-bank CSP (FBCSP).** CSP per sub-band, combined by the classifier. A reliable gain, and
  still classical and interpretable.
- **The learning curve**, holding subject fixed and sweeping training-set size. That would settle
  a claim this project made and then retracted, that the barrier is sample size rather than
  anatomy.
- **ICA-based artifact rejection**, and a paradigm that records EOG.

See [EXPLAINER.md](EXPLAINER.md) §12 for the full scoreboard, including the complete list of
claims this project published and later retracted. That list only grows — corrections are added
to it, never swapped in over the record of the claim they correct.
