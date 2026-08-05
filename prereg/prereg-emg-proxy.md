# Pre-registration: the high-band EMG proxy

**Written 2026-07-25, before any probe code exists.**
**Target script:** `/Users/yaz/Documents/Projects/eeg-motor-imagery/emg_proxy.py` (not yet written).
**Status at time of writing:** nothing executable has been produced for this measurement. The design
below is fixed. If it changes, the change gets appended to section 13 with a date and a reason, and
the original text stays.

## 0. Reading this document: no number below is a result

Every number in this file is one of three things, and none of them is a measurement of this project's
data:

- a **property of the design** (160 Hz sampling, 45 trials, 21/24 class split, the k/45 lattice, filter
  transition bandwidths, filter half-lengths),
- a **pre-computed threshold** (the binomial critical count, the minimum detectable effect size),
- a **prediction or a decision boundary** that exists so a future result cannot be renarrated.

Nothing here is an outcome. A claim-checker that scrapes percentages out of this file will find
thresholds, not findings. This warning exists because this project has published numbers that were
never computed, and a pre-registration full of numeric thresholds is exactly the document where that
mistake would recur.

## 1. Why this measurement exists

The corpus names this exposure at canon level and then does not close it.

- `CANON.md:355` (A93): "The fourth retained component peaks at T8, with T10 and TP8 in its top five,
  which is an open EMG exposure."
- `00-GATE-PACK.md:249-250`: "The fourth retained CSP component peaks at T8/T10/TP8, which is
  temporalis territory. The ablation tests frontopolar channels only. Nothing in the repo bounds an
  EMG contribution."
- `gate/01-csp-derivation.md:429`: "Nothing in this repo bounds an EMG contribution ... So the honest
  label is **open exposure**, not controlled."
- `DRILLS.md:1995` (S2-A6): "Status: **open exposure, not controlled.** Neither probe exists."
- `strands/1-signal.md:979-983` (A5) specifies the probe that would close it and predicts its result.

**Pointer appended 2026-07-26, after the run. It changes no threshold, no hypothesis and no text
above it, and it is the only post-run addition to this file.** The five quotations above are the
**pre-run** state of the corpus and are kept verbatim on purpose. Four of the five have since been
corrected in place: `CANON.md` A93, `00-GATE-PACK.md`, `gate/01-csp-derivation.md` section 5.4 and
section 8.5, and `DRILLS.md` `S2-A6` now carry `UPDATED 2026-07-26` notes recording that probe (ii)
exists and probe (i) does not. **Do not read the quotations above as the current state of those
files**, and do not read the corrections as closing the exposure: the closure condition is a
conjunction of two probes and only one ran.

The repo's only artifact control is `ablate_channels.py`, whose frontopolar-only row (23/45, 51.1%,
against a 24/45 majority floor) addresses **ocular** contamination. It says nothing about **muscle**,
for a reason the corpus already states: the pipeline band-passes to 8-30 Hz, so everything above 30 Hz
is discarded before any covariance is computed, and the EMG signature lives mostly above the passband.
The filter, not the feature, decides what is findable. An EMG probe inside 8-30 Hz cannot see the thing
it is probing for.

This document converts that disclosure into a measurement.

**One correction to the corpus's own proposal, made here and on purpose.** `strands/1-signal.md:975`
and `DRILLS.md:1897` both propose the band **50 to 78 Hz**. That band straddles 60 Hz, which is the
line frequency of the recording site (BCI2000 / Wadsworth Center, United States) and is below the
80 Hz Nyquist, so it is present in the data. A 50-78 Hz probe would have been substantially a
line-noise detector. The band used here is stated in section 4 and avoids 60 Hz explicitly.

## 2. The one question

**Does a decoder restricted to muscle-band frequencies at muscle-territory electrodes decode hands
versus feet in subject 1?**

Stated so it has a wrong answer: the project's implicit position is that the 91.1% headline reflects
sensorimotor rhythm modulation. If that is right, a CSP+LDA decoder given only 40-75 Hz power at the
temporal ring should land at or below the 24/45 majority floor. If it lands well above the floor, the
project's position is wrong or incomplete, and the headline carries an unbounded muscle or gaze
contribution that the frontopolar ablation was structurally incapable of seeing.

Two arms, and the second is the sharp one:

- **(a)** Does high-band power at temporal channels differ by class? Univariate, per channel plus an
  aggregate.
- **(b)** Does high-band temporal power carry class information on its own? The same CSP+LDA pipeline,
  same seed, same folds, refit on the high-band temporal data.

(b) governs. (a) is reported because a difference in power with no decodability, and decodability with
no univariate marginal, are different situations and both are informative. A null on (a) does not
rescue a positive on (b): CSP is multivariate and can find a spatial combination with no per-channel
marginal.

## 3. Data, held fixed

Identical to `decode_csp.py` and `ablate_channels.py`, so the numbers are comparable to the existing
table and to nothing else.

| item | value |
|---|---|
| dataset | PhysioNet EEGMMIDB via `mne.datasets.eegbci`, cached locally |
| subject | 1 |
| runs | 6, 10, 14 |
| contrast | imagined both fists (T1, label 2) vs imagined both feet (T2, label 3) |
| trials | 45 total, 21 hands, 24 feet |
| majority-class floor | 24/45 = 53.3%. Chance is this, not 50% |
| accuracy lattice | every CV here is a partition testing each trial once, so accuracy is k/45, steps of 2.222%. Off-lattice values are impossible |
| channels | 64, standardised via `eegbci.standardize` + `standard_1005` |
| sampling | 160 Hz, Nyquist 80 Hz |
| reference | average, over all 64 channels, computed BEFORE any channel subset is picked |
| epochs | tmin -1.0, tmax 4.0, `baseline=None` |
| feature window | crop 1.0 to 2.0 s, same as the headline |
| seed | 42 |
| CV | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

**Verified before writing this document, not assumed:** all eight temporal-ring channels
(T7, T8, T9, T10, TP7, TP8, FT7, FT8) are present in subject 1's standardised montage, and `sfreq` is
160.0 with 64 channels. A pre-registration that names channels the montage does not carry would be a
design defect, and `ablate_channels.py` already asserts on exactly this.

## 4. Bands

The decoder's own filter is 8-30 Hz. MNE's `firwin` default upper transition bandwidth is
`min(max(0.25 * h_freq, 2), sfreq/2 - h_freq) = 7.5 Hz`, so the **decoder's stopband edge sits at
37.5 Hz**. Any probe band whose passband starts below 37.5 Hz overlaps something the headline pipeline
partially passes, which would blur the claim that the probe looks where the decoder cannot.

**PRIMARY band: 40-75 Hz, then a 60 Hz notch.**

| parameter | value | why |
|---|---|---|
| `l_freq`, `h_freq` | 40.0, 75.0 | lower edge clears the decoder's 37.5 Hz stopband edge; upper edge leaves room below the 80 Hz Nyquist for a real transition band |
| `l_trans_bandwidth` | 2.0 | lower stopband edge lands at 38 Hz, above the decoder's 37.5 Hz. Pinned explicitly: MNE's default here would be 10 Hz, putting the probe's stopband edge at 30 Hz and overlapping the decoder's passband |
| `h_trans_bandwidth` | 2.0 | upper stopband edge at 77 Hz, strictly inside Nyquist |
| line handling | `notch_filter(freqs=60.0, notch_widths=2.0, trans_bandwidth=6.0)` | empties roughly 56-64 Hz. This is the "exclude 55-65 Hz" option, implemented as a notch so the surrounding band is retained |

Handling 60 Hz by notch rather than by band choice is deliberate: the probe is a **bounding**
instrument, so it should have the maximum EMG bandwidth the sampling rate permits. A narrow band would
buy cleanliness at the cost of sensitivity, and an underpowered probe returning a null is worth
nothing. The injection ladder in section 7 is what makes the sensitivity claim quantitative rather
than rhetorical.

**Filter length, which is the one number that silently rescales with `sfreq`, so it is pinned and
printed.** At 160 Hz with a 2.0 Hz minimum transition, `firwin` gives a 265-tap symmetric FIR:
half-length 132 samples = **0.825 s**. That is the same half-length as the decoder's own 8-30 Hz
filter (whose minimum transition is also 2.0 Hz, from the lower edge), and the same 0.825 s already
computed in `regime_decomposition.py:170-171`. Matching the transition bandwidths matches the temporal
smear budget, so probe and decoder contaminate their analysis windows by the same amount. The notch at
`trans_bandwidth=6.0` adds an 89-tap filter, half-length 0.275 s. **Cascade half-length is therefore
about 1.100 s**, so the primary band's 1.0-2.0 s window can draw energy from as early as -0.100 s,
i.e. just before the cue. The script prints the realised lengths rather than trusting these figures.

**Pre-registered decision rule on the smear.** Smear can only push a score UP, never down, which is
the direction argument `regime_decomposition.py:178` already establishes. So a null in the primary
band is conservative and needs no repair. A **positive** in the primary band must be arbitrated by R1
and R2 below, which use a single filter with a 0.825 s half-length and therefore draw only from
+0.175 s onward, strictly post-cue.

**Pre-registered robustness bands. Their role is fixed now. They cannot promote a null primary to a
positive result. They can only qualify a positive primary or expose line contamination.**

| id | band | notch | role |
|---|---|---|---|
| R1 | 40-55 Hz | none needed | entirely below line. Single filter, 0.825 s half-length, window strictly post-cue. Arbiter for a positive primary |
| R2 | 65-75 Hz | none needed | entirely above line. Same smear properties as R1. Second arbiter |
| R3 | 32-75 Hz | 60 Hz | the greedy band, deliberately dipping into the decoder's 30-37.5 Hz transition region. Reported so a reader can see whether the 40 Hz lower edge is doing any work |

Pre-declared reading: if the primary is positive but **both** R1 and R2 are at floor, the primary
result is attributed to line residual or to the notch cascade's extra smear, not to muscle, and it is
reported that way. If the primary is positive and R1 and R2 agree with it, the smear and line
explanations are excluded and the result stands as a high-band finding.

**Pre-declared contingency on R2 being empty.** If the amplifier's anti-alias filter rolls off before
75 Hz, R2's band may sit at the noise floor and its null would be uninformative. The script prints the
median PSD in each of 40-55, 56-64 and 65-75 Hz at the temporal ring. **If the 65-75 Hz median is
below 0.10 times the 40-55 Hz median, R2 is declared uninformative and is not used as an arbiter.**
The 0.10 figure is a judgement call fixed in advance rather than a principled constant, and it is
stated as such.

## 5. Channel sets

All four run in the primary band, so the result has a spatial profile rather than a single number. A
class difference that is present everywhere means something quite different from one confined to the
temporal ring, and without the comparison sets the primary number cannot distinguish them.

| set | channels | n | role |
|---|---|---|---|
| TEMPORAL | T7, T8, T9, T10, TP7, TP8, FT7, FT8 | 8 | the probe. Temporalis territory, and where component 3's pattern peaks |
| FRONTOPOLAR | Fp1, Fpz, Fp2, AF7, AF3, AFz, AF4, AF8 | 8 | matched size, existing control set, and the site where saccadic spike potentials and frontalis EMG are largest |
| SENSORIMOTOR | FC3 FC1 FCz FC2 FC4, C5 C3 C1 Cz C2 C4 C6, CP3 CP1 CPz CP2 CP4 | 17 | where genuine high-gamma would appear if it appears at all |
| ALL64 | all | 64 | is there any high-band class information anywhere |

FRONTOPOLAR is deliberately the same size as TEMPORAL (8 channels), because `ablate_channels.py`'s own
second caveat is that its frontopolar row confounds region with an 8x cut in channel count. Matching
the counts removes that confound from the TEMPORAL versus FRONTOPOLAR comparison specifically.

## 6. The two tests

### (a) Univariate: does high-band power differ by class?

Per trial, per channel: mean band power over the 1.0-2.0 s crop, then log. Log because raw band power
is right-skewed and heteroscedastic in a way that scales with the class mean, which is the same
argument the corpus already makes for `log=True` in CSP.

- **Aggregate (headline for arm a):** mean log power across the 8 TEMPORAL channels, one value per
  trial, one test. Welch's t-test, two-sided, plus Mann-Whitney U on the same values.
- **Per channel:** the same two tests on each of the 8 channels, with **Holm-Bonferroni** across the 8
  at family alpha 0.05. Raw and corrected p both printed.
- **Effect size:** Cohen's d on log power, plus the ratio of median raw power between classes, so
  direction and magnitude are both legible.
- **Two-sided.** There is no directional prior. The cue is a bar at the top of the screen for fists and
  the bottom for feet (`regime_decomposition.py:186-188`), which gives a class-dependent gaze
  direction but no defensible prediction about which direction produces more high-band temporal power.
- **Repeated on FRONTOPOLAR, SENSORIMOTOR and ALL64 aggregates**, descriptive only.
- **Pre-cue diagnostic:** the same aggregate test on the -1.0 to 0.0 s window. Reported with its
  caveat: at a 1.100 s cascade half-length only the first 0.175 s of that window is strictly
  filter-clean, which at B ~ 27 Hz is about 9 effective degrees of freedom and far too noisy for a
  stable band-power estimate. So a null here is uninformative, and only a **strong positive** is
  meaningful. **Pre-registered contingency:** if the pre-cue aggregate is significant, re-run it on
  epochs built from raw segments that physically END at t = 0.0 s before filtering, which is the
  remedy `regime_decomposition.py:174-177` already established for this exact problem.

**Pre-computed power, so a null cannot be oversold.** With 21 versus 24 trials at 80% power:
minimum detectable Cohen's d is **0.837** for the single aggregate test at alpha 0.05, and **1.069**
for a single channel at the Bonferroni floor alpha 0.00625. Both are large effects. A null on arm (a)
bounds large effects only, and that sentence goes in the output verbatim.

### (b) Decoding: the sharp test

The pipeline is not modified in any respect except the filter band and the channel pick.

```
Pipeline([("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
          ("LDA", LinearDiscriminantAnalysis())])
```

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, the same splitter and seed as the
  headline and the ablation.
- **CSP is fit inside every training fold**, via the `Pipeline`, never on the full dataset. The
  band-pass and notch are applied to the continuous raw before epoching, outside the fold, which is
  label-blind and fixed a priori and therefore not leakage. This asymmetry is stated in the output.
- `error_score="raise"`, so a rank failure on an 8-channel high-band covariance surfaces as a
  traceback rather than a silent NaN. This is a live risk here, not a hypothetical: high-band power at
  8 channels after an average reference computed over 64 could be near-singular.
- `cross_val_predict` alongside `cross_val_score`, so accuracy is reported as an integer count k out
  of 45 and not only as a fold mean.
- **Assert** that the fold mean equals the pooled k/45 to 1e-9, the same assert `ablate_channels.py`
  carries, because unequal folds would produce a fold-mean off the lattice that still looks like an
  accuracy.
- **Permutation test**, 1000 permutations, `random_state=42`, on the four primary-band channel sets
  only (not on every band, for runtime). p reported as `<= 0.001` when it hits the floor, because
  sklearn computes `(C+1)/(n+1)` and 1/1001 is the test's resolution limit, not a measurement. Same
  convention as `decode_csp.py:135-136`.
- **Analytic cross-check printed alongside:** one-sided exact binomial against p0 = 24/45. The
  pre-computed lattice is fixed here so it cannot be chosen after the fact:

| k/45 | accuracy | one-sided binomial p vs 24/45 |
|---|---|---|
| 29/45 | 64.4% | 0.0886 |
| **30/45** | **66.7%** | **0.0490, the first k inside alpha 0.05** |
| 31/45 | 68.9% | 0.0249 |
| 32/45 | 71.1% | 0.0115 |
| 33/45 | 73.3% | 0.0048 |
| 35/45 | 77.8% | 0.0006 |

30/45 clears alpha by 0.001, which is a knife edge. **Significance for arm (b) is called from the
permutation test, not the binomial**, and a result of exactly 30/45 is reported as marginal rather
than as detection regardless of which instrument it clears.

- **Positive control on the harness, run first.** The 8-30 Hz, all-64 condition must reproduce
  41/45 = 91.1%. If it does not, the harness is not the published pipeline and nothing else in the run
  is comparable to the existing table. This is checked and printed before any probe result.

## 7. The sensitivity calibration, which is what turns a null into a bound

This is the part that distinguishes measuring from conceding. Without it, a probe at floor supports
only "we looked and found nothing", which is another disclosure. With it, the probe supports "we can
detect a class-correlated broadband temporal source of size X, and the data contains less than X".

**Injection design.** A realistic muscle artifact is a *source*: one generator, projecting to several
electrodes with a fixed topography. Injecting independent noise per channel would be both unrealistic
and unfair to the probe, since CSP's whole business is finding coherent spatial directions.

- Per trial, draw one latent Gaussian time series, band-limit it with the **same filter cascade** as
  the probe band, and project it onto the 8 TEMPORAL channels with a fixed unit-norm topography.
- **Stipulated topography, right-lateralised to mimic the observed component 3:** T8 1.00, T10 0.80,
  TP8 0.80, FT8 0.50, T7 0.20, T9 0.15, TP7 0.15, FT7 0.10, then normalised. This is **stipulated, not
  measured**, and the script says so. A second, spatially flat topography (all 8 equal) is run as a
  robustness check on whether the sensitivity figure depends on the assumed shape.
- **Amplitude** `a` is defined as the injected source's contribution to T8 as a fraction of T8's own
  measured high-band SD, so the ladder is in interpretable units.
- **Ladder:** a in {0.000, 0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.400, 0.600, 0.800}.
  a = 0.000 is the real data and is the reference row.
- **10 injection seeds per rung**, distinct from the CV seed, which stays at 42 throughout.
- **Both directions:** injected into hands (21 trials) and, separately, into feet (24 trials). The
  **worse** of the two, meaning the higher detection threshold, is the one reported as the bound. A
  bound must not be computed from the easier direction.
- **Detection criterion, fixed now:** the smallest `a` whose **median accuracy across the 10 seeds is
  at least 31/45 (68.9%)**. 31/45 rather than 30/45 because 30/45 sits at p = 0.0490 and a detection
  criterion should not rest on a knife edge.
- **The script prints the realised aggregate Cohen's d at each rung**, so the bound can be stated in
  measured effect-size units and compared directly with arm (a)'s observed d and its 0.837 detection
  floor. This is measured per rung, not taken from the analytic approximation.

**The sentence this produces**, with X and Y filled from the run and not from this document:

> The probe detects a class-correlated broadband temporal source injected at a = X times the T8
> high-band SD, corresponding to an aggregate Cohen's d of Y in log high-band power. The real data
> yields [observed]. Any real class-correlated high-band temporal source in this recording is
> therefore smaller than that. This BOUNDS an EMG contribution. It does not eliminate one.

## 8. Pre-registered outcomes: every plausible result, with its meaning fixed now

### 8.1 Primary cell: TEMPORAL, primary band, arm (b), imagery window

**One cell is declared primary. Everything else in this document is descriptive.** This is stated so
the four channel sets by four bands by two arms cannot be mined for whichever cell reads best.

| result | meaning, fixed in advance | verdict |
|---|---|---|
| **<= 24/45 (53.3% or below)** | At or below the majority floor. No decodable class information in muscle-band power at muscle-territory electrodes. Combined with section 7, this yields a numeric bound on any EMG contribution. The corpus line "nothing bounds an EMG contribution" becomes false and gets replaced by the bound. Say **bounds**, never **eliminates** | good |
| **25/45 to 29/45 (55.6% to 64.4%), permutation p > 0.05** | Above floor, not distinguishable from noise at n = 45. **AMBIGUOUS, and reported as ambiguous.** Not a null. The exposure is narrowed, not closed, and the ladder bound still applies | depends |
| **30/45 to 32/45 (66.7% to 71.1%), permutation p <= 0.05** | A real but modest class-correlated high-band temporal signal exists. This is an EMG exposure that is now measured rather than open. It does not invalidate 91.1%, but the honest statement changes from "nothing bounds an EMG contribution" to "a muscle-band, muscle-location decoder reaches this accuracy, so class-correlated non-cortical activity is present" | bad |
| **33/45 to 39/45 (73.3% to 86.7%), permutation p <= 0.05** | A serious confound. A decoder with no access to mu or beta, using only muscle-band frequencies at temporal electrodes, carries a large share of the information the project attributes to motor imagery. The framing that the frontopolar ablation addresses artifact contamination must be withdrawn as insufficient, and the headline must be restated with an explicit unbounded-muscle caveat | bad |
| **>= 40/45 (88.9% or above)**, at or above the headline's own 41/45 | The worst case, and it is reported **first** in the output, not last. The probe matches or beats the headline while being blind to the entire band the headline claims to use. The parsimonious reading is that 91.1% rides a muscle or gaze artifact. Correct action: treat the headline as unsupported pending a dataset with EOG and EMG reference channels. This is the outcome the project would least like to see and the one this pre-registration exists to make unspinnable | bad |
| **<= 20/45 (44.4% or below)** | Clearly below the majority floor. No usable signal, the same reading `ablate_channels.py` already gives its 23/45 frontopolar row. **Not** evidence of anti-information: at n = 45 a value this low is a coin, and the correct description is a degenerate classifier, not an inverted one | good |

### 8.2 Spatial profile, primary band, arm (b)

| pattern | meaning, fixed in advance | verdict |
|---|---|---|
| all four sets at or below floor | No high-band class information anywhere. Cleanest outcome. The bound is then set entirely by the ladder | good |
| TEMPORAL above floor, others at floor | Localised to temporal territory. Consistent with temporalis EMG, and also with a right-lateralised source this design cannot distinguish from it | bad |
| TEMPORAL and FRONTOPOLAR above floor, SENSORIMOTOR at floor | Anterior and lateral, not central. The leading candidate is the **saccadic spike potential**, a high-frequency frontotemporal artifact, which is highly plausible here because the cue is position-confounded with the label (top bar for fists, bottom for feet). A different confound from temporalis, equally serious, and it must not be reported as EMG | bad |
| all four above floor, including ALL64 | A **global** broadband class difference. Candidates: global muscle tone, arousal, or amplifier gain drift correlated with block structure. The least specific and the most alarming, because something broadband differs by class everywhere on the scalp | bad |
| SENSORIMOTOR above floor, TEMPORAL at floor | Possibly genuine sensorimotor high-gamma, which is a **finding, not a confound**, and must not be reported as EMG. It would still need its own artifact control before being claimed | depends |
| FRONTOPOLAR above floor, TEMPORAL at floor | Ocular, not muscular, and it would mean the existing frontopolar ablation's null at 8-30 Hz was band-limited rather than conclusive | bad |

### 8.3 Arm (a) against arm (b)

| combination | meaning, fixed in advance | verdict |
|---|---|---|
| (a) null, (b) at floor | Consistent. Bound reported at d = 0.837 for the aggregate, d = 1.069 per channel, plus the ladder's bound from (b) | good |
| (a) positive, (b) at floor | High-band temporal power differs by class but is not linearly separable at this n. A partial exposure, reported as partial. The presence of a power difference is the more conservative reading and it governs the write-up | bad |
| (a) null, (b) above floor | A multivariate effect with no univariate marginal, which CSP is built to find. **(a)'s null does not rescue (b).** (b) is the sharper test and it governs | bad |
| (a) positive and (b) above floor | Confound confirmed on both instruments. No ambiguity left to report | bad |
| pre-cue diagnostic significant | Class-correlated high-band temporal power exists **before the cue**, which cannot be imagery. Triggers the section 6 contingency (rebuild epochs from segments ending at t = 0). If it survives that, the finding is block structure, drift or a labelling artifact, and it damages more than the EMG claim | bad |

### 8.4 The ladder itself

| ladder result | meaning, fixed in advance | verdict |
|---|---|---|
| clean detection threshold somewhere in 0.05 to 0.40, near-ceiling by 0.80 | The probe works and its sensitivity is quantified. Any null from section 8.1 now has teeth | good |
| never detects, even at a = 0.800 | **The probe cannot see a planted signal.** Any null in section 8.1 is then uninterpretable and must be reported as "we built an instrument that cannot detect what it was built to detect", which falsifies the measurement rather than the hypothesis. Nothing about EMG may be concluded | neutral |
| detects at a = 0.025, the smallest rung | Suspiciously sensitive. Before reporting, audit the injection code for a leak, most plausibly an injection correlated with the label beyond the intended amplitude, or an injection applied after rather than before the fold split | neutral |
| flat and stipulated topographies give thresholds differing by more than about 2x | The sensitivity figure depends materially on an assumed shape that was never measured. Report the **worse** threshold as the bound and state the dependence | neutral |

## 9. What would falsify the analysis itself, as opposed to the hypothesis

These are checks on the instrument. Each is asserted or printed, and each one failing means the run
produces no usable claim about EMG at all.

1. **The ladder never detects at a = 0.800.** The primary falsifier. A probe that cannot recover a
   planted source at 80% of a channel's own high-band SD is not an instrument, and its null bounds
   nothing.
2. **The positive control fails:** 8-30 Hz on all 64 channels does not return 41/45 = 91.1%. The
   harness is then not the published pipeline and no number in the run is comparable to the existing
   table.
3. **Trial counts are not 45 total, 21 hands, 24 feet.** Asserted.
4. **Any accuracy lands off the k/45 lattice**, or the fold mean disagrees with the pooled count by
   more than 1e-9. The CV is then not an equal-fold partition and the reported number is not what it
   claims to be. Asserted, following `ablate_channels.py:210-214`.
5. **The PSD diagnostic shows the probe band dominated by a narrow spike** rather than broadband
   content. The feature is then line residual or an alias, not band power. Note specifically that a
   120 Hz line harmonic, if the amplifier's anti-alias filter did not remove it before the 160 Hz
   sampling, aliases to **40 Hz**, which is the probe's lower passband edge. Aliased content is
   indistinguishable from genuine 40 Hz content after the fact and cannot be notched out, so this is
   checked by inspection of the printed PSD, not by filtering.
6. **CSP raises on a near-singular 8-channel high-band covariance.** `error_score="raise"` makes this
   a traceback rather than a NaN quietly dragging a mean down.
7. **The realised filter lengths do not match the pinned values** (265 taps for the band-pass,
   half-length 0.825 s; 89 taps for the notch, half-length 0.275 s). The smear budget in section 4 is
   then wrong and every window-based statement has to be recomputed. Printed and checked, because this
   is the one quantity in the pipeline that silently rescales with `sfreq`.

## 10. Limitations that survive every possible outcome

These are stated now so they cannot be quietly dropped from whichever result arrives.

1. **160 Hz sampling truncates the EMG spectrum.** Surface temporalis EMG has substantial power well
   above 80 Hz, and none of it was recorded. Even a perfect null bounds only the recorded part of the
   spectrum. This is a hard limit of the dataset, not of the analysis.
2. **The average reference is computed over all 64 channels before any subset is picked**, exactly as
   in `decode_csp.py` and `ablate_channels.py`. Every channel therefore carries -1/64 of every other,
   so the temporal ring is not electrically sealed off. This bounds a contribution, it does not isolate
   one. Re-referencing the subset separately would no longer be the published pipeline.
3. **EEGMMIDB ships no EOG and no EMG channel.** There is no ground truth for "this is muscle". The
   probe measures high-band power at muscle-adjacent scalp sites. It does not measure muscle.
4. **A positive result cannot distinguish temporalis EMG from a saccadic spike potential**, and the
   position-confounded cue makes the latter genuinely plausible. Both are confounds; they are different
   confounds with different remedies.
5. **n = 45.** Every accuracy here has a Wilson interval well over 20 points wide, and a four-trial
   difference between conditions is noise. Arm (a) can only detect large effects, d >= 0.837.
6. **Single subject, single session.** Runs 6, 10 and 14 are three recordings from one session.
   Nothing here generalises past subject 1, and a session-level trend across all three runs survives
   every control in this design.
7. **A null does not license "EMG is not a risk for this project."** Imagined fists versus feet is a
   low-EMG task. The corpus's own worked example, imagined speech (`DRILLS.md:1858-1860`), is a task
   where sub-vocalisation makes EMG class-correlated by construction. A null here licenses exactly one
   sentence: for this subject, this task, this recording, within the recorded band, a class-correlated
   temporal high-band source is smaller than the ladder's threshold.
8. **The injection topography is stipulated, not measured.** It is a plausible right-temporalis shape,
   not this subject's.
9. **The probe shares the headline's filter-before-fold structure.** The band-pass is fitted on all
   data. It is label-blind and fixed a priori, so it is not leakage, but it is not inside the fold
   either and the output says so rather than letting a reader assume otherwise.

## 11. Design choices considered and rejected

Recorded so the primary design cannot later be presented as the only option that existed.

- **50-78 Hz, the corpus's own proposal.** Rejected: it straddles 60 Hz line noise, which is present
  below the 80 Hz Nyquist. It would have been substantially a line-noise detector. See section 1.
- **Two separate band-passes (40-55 and 65-75) with the filtered channels stacked**, avoiding the
  notch entirely. Rejected: it doubles CSP's input dimension from 8 to 16 virtual channels, which
  changes the covariance estimation problem and breaks comparability with the 8-channel frontopolar
  row. The two bands are run separately as R1 and R2 instead.
- **A narrow, maximally clean band as primary.** Rejected: for a bounding probe, sensitivity dominates
  cleanliness, because an underpowered null bounds nothing. Cleanliness is recovered through R1 and R2.
- **Independent per-channel noise injection.** Rejected as both unrealistic and unfair to CSP, which
  finds coherent spatial directions. A muscle artifact is a source with a topography.
- **Injecting into the majority class only.** Rejected: both directions are run and the worse
  detection threshold is reported, because a bound computed from the easier direction overstates the
  instrument.
- **A temporal-channel-DELETED ablation row (all 64 minus the temporal ring, in the 8-30 Hz band).**
  This is the other arm the corpus asks for (`DRILLS.md:1987-1989`, `strands/1-signal.md:977`), and it
  is a legitimate and cheap measurement. It is **explicitly out of scope here** and is not run, because
  it lives in the decoder's own band and answers a different question: whether the headline needs the
  temporal channels, rather than whether muscle-band activity is class-correlated. Pre-registering it
  loosely alongside this probe would let a null on one be read as covering the other. It should get its
  own pre-registration.

## 12. Output contract

House style, and load-bearing because `check_claims.py` and `check_provenance.py` match against
stdout. Everything below is printed, not returned.

- Module docstring saying why the script exists, what it guards against, and what it does not show.
- Printed section headers.
- **Every accuracy printed with its k/45 count and against the 24/45 = 53.3% majority floor**, in the
  form `95.6% (43/45)`. Chance is the majority rate, never 50%.
- The attainable-accuracy lattice printed once, as `ablate_channels.py` does.
- Asserts on: trial counts, channel presence in the montage, the k/45 lattice, fold-mean versus pooled
  count, and the positive control reproducing 41/45.
- The realised filter lengths, half-lengths, and the cascade smear figure, printed.
- The PSD table for 40-55, 56-64 and 65-75 Hz at the temporal ring, printed.
- A closing **"What this does and does not show"** section that states, in whichever direction the
  result lands: that the average reference makes the subsets electrically non-independent; that 160 Hz
  sampling truncates the EMG spectrum; that there is no EOG or EMG reference channel; and, if the
  result is a null, the word **bounds** and not the word **eliminates**.
- Permutation p at the floor printed as `<= 0.001`, never as `0.0010`.
- No em dashes anywhere in the script or its output.
- Registry entry in `check_provenance.py`. The runtime must be **measured on this machine and the
  measured figure written into the registry with the date**, following the convention at
  `check_provenance.py:91-97` that runtimes are measured, not guessed. A placeholder guess would
  violate house style in the specific way this project is trying to stop doing.

## 13. Amendments

None. Any change to sections 2 through 9 after the first run of `emg_proxy.py` gets appended here with
a date and a reason, and the superseded text stays visible in place, following the repo's practice of
keeping withdrawn statements rather than deleting them.

---

### Amendment 1, 2026-07-26: five corrections after an adversarial pass

Appended here rather than edited into sections 2 through 9, per the rule above. **The
superseded text stays visible in place.** Re-run of `emg_proxy.py` after the repairs: exit 0,
positive control still 41/45, primary cell still 23/45 with permutation p = 0.5175, arm (a)
still null, and the registered ladder's two-topography thresholds unchanged. **No registered
number moved.** All five corrections are about what the run SAYS and about arms it did not run.

#### 1. The permutation null was computed and discarded, so the run never established where chance is

`emg_proxy.py` wrote `_, _, p_perm = permutation_test_score(...)`, throwing away the 1000-shuffle
null distribution and leaving the majority floor as the only reference in the write-up. The null
is kept now. Measured, in trials out of 45:

| channel set | observed | null median | null mean | null sd | observed percentile in own null |
|---|---|---|---|---|---|
| TEMPORAL | 23/45 | 23.0 | 22.79 | 3.92 | 48.3 to 57.2 |
| FRONTOPOLAR | 15/45 | 23.0 | 22.99 | 3.81 | 1.9 to 2.9 |
| SENSORIMOTOR | 17/45 | 22.0 | 22.55 | 3.96 | 6.7 to 11.0 |
| ALL64 | 18/45 | 23.0 | 23.06 | 3.89 | 8.6 to 12.1 |

The null median sits **below** the 24/45 majority floor for all four sets. Two consequences.

First, the sentence "lands at 51.1% (23/45), BELOW the 53.3% majority floor" invites the reading
that the probe underperformed chance. It did not. It landed at the **48th percentile of its own
null**, i.e. exactly at chance for this pipeline. The "BELOW the floor" framing carried
rhetorical weight the data does not supply, and the floor is what a CONSTANT predictor scores,
not what CSP+LDA scores under H0.

Second, and not previously diagnosed anywhere: the other three sets are in the **LOWER tail of
their own nulls**. No-information predicts performance AT the null. What was measured is
performance systematically below it at three of four sets. The one-sided permutation test is
structurally incapable of seeing this, because sklearn scores P(null >= observed) only.

**NO MECHANISM IS OFFERED.** Naming a cause in the same breath as the number is this project's
round-one failure mode. It is recorded as an unexplained systematic property of the instrument,
printed as limitation 12, and it needs its own pre-registration.

#### 2. The primary cell was one partition with no spread reported, and the pinned seed is in the minority branch

`random_state=42` was pinned in section 3, correctly. It was never varied. Over 100 CV seeds the
primary cell gives median **19/45 = 42.2%**, IQR [17, 21], range [13/45, 25/45], mean 18.86/45,
sd 2.51 trials. The reported 23/45 sits at the **93rd percentile** of that distribution.

Against the section 8.1 branch table, over the 100 seeds (permutation significance assumed false
throughout, which is the conservative assignment for a null):

- **73/100** fire "<= 20/45, degenerate classifier, NOT evidence of anti-information"
- **26/100** fire the branch that was actually printed, "21 to 24/45, at or below the floor"
- **1/100** fire AMBIGUOUS
- **0/100** reach the "bad" band at 30/45 or above

This is not p-hacking: the seed was pinned in advance. It is a bounding instrument that reported
one draw and no spread, in a repo whose `evaluate_honestly.py` already sweeps 100 seeds.

**This STRENGTHENS the conclusion and is reported for that reason.** 0 of 100 seeds reach the
pre-registered bad band, so that band is unreachable under every partition tried, which is a
stronger claim than any single seed can make. The section 8.1 anti-information caveat, which is
the correct caveat for what this probe typically does, never printed at the pinned seed. It
prints now.

The ladder threshold, by contrast, IS seed-stable: re-derived at CV seeds 42, 0, 1, 2 and 3 the
stipulated/into-hands threshold is 0.300 at all five. Those are different quantities and the
seed sensitivity of one must not be read onto the other.

#### 3. The bound is not a bound over topographies, and section 8.4's "no consequence" reading is WITHDRAWN

`a` is defined **only** by the source's contribution to T8, while detectability depends on the
TOTAL power injected across the ring. Those are related by the shape: injected amplitude is
`a * SD_T8 / w[T8]`, so total injected power scales as `1 / w[T8]^2`. At fixed `a`, flat injects
**8.00x** the total power of a T8-only source and stipulated injects **2.62x**. The registered
"max over topographies" is a max over TWO shapes, both diffuse.

Two focal shapes added post-registration, same seeds, same rungs, same CV:

| topography | w[T8] | worse-direction threshold `a` | same in ring-RMS units |
|---|---|---|---|
| flat | 0.3536 | 0.150 | 0.150 |
| stipulated | 0.6172 | 0.300 | 0.172 |
| focal T8+T10 | 0.8575 | 0.600 | 0.247 |
| focal T8-only | 1.0000 | 0.600 | 0.212 |

**Section 8.4's alarm did fire.** The true spread over four shapes is 0.150 to 0.600, a **4x**
range, not a 2.00x knife edge, and the run's reading that the knife edge "has no consequence" is
refuted by data: the consequence is a factor of two on the only number the measurement produces.
A focal generator directly under one electrode is the canonical superficial-artifact geometry,
not an exotic case, and a source at `a = 0.5` sitting focally under T8 is INSIDE this recording's
tolerance while sitting OUTSIDE the registered bound.

**WITHDRAWN**: the unqualified sentence *"this recording contains no class-correlated broadband
temporal source as large as a = 0.300 times T8's own high-band SD"*, and the "knife edge, no
consequence" reading. The honest figure over the shapes actually measured is **a = 0.600**.

Re-expressing `a` as the source's RMS across the eight ring channels (equivalently
`a / (sqrt(8) * w[T8])`) collapses the shape dependence from 4x to **1.65x**, which is the
evidence that most of the spread was a units artifact of pinning `a` to one electrode. That
re-expression is printed beside the registered units; it does not replace them.

#### 4. The ladder calibrates against a source model this script itself calls unrealistic, and the bursty case is NOT bounded

`emg_proxy.py` prints that the realistic EMG failure mode is "a few trials with a clench, not a
shifted distribution", and then injects a **constant amplitude into every trial of the target
class**, which is precisely a shifted distribution. Nothing in the registered ladder is
intermittent. Section 10 limitation 8 disclosed the stipulated TOPOGRAPHY; there was no matching
disclosure for the stipulated TEMPORAL PROFILE, and section 7 considered and rejected only one
alternative (independent per-channel noise), never intermittency.

A bursty arm was added: the same total injected variance concentrated in a random 25% of the
target class's trials, per-trial amplitude scaled by `1/sqrt(f)`, same rungs, same seeds, same CV.

**Result, and it is worse than "a higher threshold".** Seven of the eight shape-by-direction
cells never reach the 31/45 detection criterion at any registered rung. Pushed far past the
ladder's top rung, to `a = 2.0`, `4.0` and `8.0`, the bursty arm **saturates at 26/45 at best**
and **0 of 8 cells** ever reach 31/45.

The competing explanation was checked rather than assumed: at a 25% duty cycle the source is
present in only 5 of 21 hands trials or 6 of 24 feet trials, so it can carry information about at
most that many trials, while 31/45 is 7 trials above the majority floor. **So the criterion
itself, which was calibrated against continuous injection, is not transportable to a 25% duty
cycle.** This run therefore does NOT claim that bursty sources are undetectable. It records that
the registered detection criterion **cannot adjudicate them**, which is a smaller and more exact
statement, and that the bursty exposure is consequently **OPEN, not bounded at a larger number**.

#### 5. Scope substitution in the headline sentence

`emg_proxy.py` printed that the corpus line "nothing in the repo bounds an EMG contribution" is
now FALSE, unqualified. The measurement covers 40 to 75 Hz minus a 60 Hz notch, out of a recorded
0 to 80 Hz. **The headline decoder lives at 8 to 30 Hz, which is exactly where this probe is blind
BY CONSTRUCTION**, and the corpus's own position elsewhere is that temporalis EMG is broadband and
not excluded by any plausible band-pass. The docstring's argument that "the filter, not the
feature, decides what is findable" cuts symmetrically. The PSD table makes this worse rather than
better: 65 to 75 Hz is 0.329x of 40 to 55 Hz and 40 to 55 Hz is already about half of 25 to 30 Hz,
so the recorded high band is steeply attenuated and any EMG present shows up preferentially at the
low frequencies this probe excludes.

Limitation 1 named the wrong boundary in the same direction: it said "even a perfect null bounds
only the RECORDED part of the spectrum". The recorded spectrum is 0 to 80 Hz. This bounds 40 to
75 Hz minus 56 to 64 Hz. Both are rewritten.

The scoped claim, which is what the script prints now: **the corpus line is false FOR THE 40 TO
75 Hz BAND ONLY. EMG inside the decoder's own 8 to 30 Hz passband remains entirely unbounded, and
the temporal-channel-deleted arm the corpus asks for alongside this one has not been run.**

New limitation 10 records that the corpus's stated closure condition
(`strands/1-signal.md:977-983`) is a **conjunction** of two arms, (i) a temporal-channel-deleted
ablation row inside 8 to 30 Hz and (ii) this high-band probe, and that only arm (ii) exists.
A conjunction with one arm run is not satisfied.
