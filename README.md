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
| CSP+LDA accuracy (stratified 5-fold CV) | **91.1% ± 4.4%** |
| Chance (majority class) | 53.3% |
| Permutation test (1000 shuffles) | **p = 0.0010** (null 50.7% ± 8.5%) |
| Wilson 95% CI on n=45 | [79.3%, 96.5%] |
| Trials | 45 (21 hands, 24 feet), one subject |

Shuffled labels never once beat the real result across 1000 permutations, so the
decoding is finding real structure rather than fitting noise:

![Permutation null distribution](permutation_null.png)

The learned CSP patterns are focal over central/sensorimotor cortex, which is the
evidence the model found real motor sources and not eye or muscle artifact:

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
and stratification reasons, but it is worth roughly 0.6 points, not 3.3. The
published 91.1% is a conservative draw from an 88.9-97.8% seed distribution.

`evaluate_honestly.py` reproduces the whole comparison, sweeping both estimators
so that neither one's diagnostics get attached to the other's number.

## Honest limitations

- **Within-subject, small-n.** One subject, 45 trials. The number does not
  claim cross-subject generalization; the ±5.6% spread shows the estimate is
  noisy.
- **Easy contrast.** Fists vs. feet are far apart on the motor homunculus
  (lateral vs. top-central), so their scalp patterns differ a lot.
  Left-hand-vs-right-hand would be harder.
- **Clean subject.** Per-subject accuracy varies widely across the 109
  subjects; subject 1 is a good recording.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python decode_csp.py        # full pipeline + CSP patterns → csp_patterns.png
```

Data downloads automatically on first run (cached in `~/mne_data`).

### Scripts (built rung by rung)

| Script | Does |
|---|---|
| `load_and_plot.py` | Load one run, plot raw EEG → `raw_eeg.png` |
| `epoch_trials.py` | Cut runs 6/10/14 into labeled hands/feet trials |
| `filter_and_epoch.py` | Add 8–30 Hz band-pass + average reference |
| `decode_csp.py` | CSP + LDA, cross-validated, plot spatial patterns |
| `evaluate_honestly.py` | Stress-test the number: stratification, coverage, permutation test, seed sweep |

## Next

- Cross-subject / harder contrasts (left vs. right hand) to test robustness.
- Swap CSP+LDA for **EEGNet** (compact CNN) and compare against this baseline.
