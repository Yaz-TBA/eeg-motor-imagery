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
- **10× ShuffleSplit cross-validation** gives an honest accuracy, not a
  single lucky split.

## Result

| Metric | Value |
|---|---|
| CSP+LDA accuracy (10-fold CV) | **94.4% ± 5.6%** |
| Chance (majority class) | 53.3% |
| Trials | 45 (21 hands, 24 feet) |

The learned CSP patterns are focal over central/sensorimotor cortex — the
model found real motor sources, not noise:

![CSP spatial patterns](csp_patterns.png)

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

## Next

- Cross-subject / harder contrasts (left vs. right hand) to test robustness.
- Swap CSP+LDA for **EEGNet** (compact CNN) and compare against this baseline.
