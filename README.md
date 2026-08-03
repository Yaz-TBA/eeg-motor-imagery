# EEG Motor-Imagery Decoding

I wanted to find out whether I could read *imagined* movement off the scalp, and then
whether I could prove the number I got was real. This repository is both halves of that.

When you imagine moving, your sensorimotor cortex changes its mu (8–12 Hz) and beta
(13–30 Hz) rhythm power in a spatially specific way. This project reads that pattern with
a classic **CSP + LDA** baseline and guesses which movement was imagined.

The decoder itself is a tutorial baseline and I want to be upfront about that. What I think
is worth looking at is the second half: the suite I built to attack my own result, and the
corrections it forced me to publish. The headline used to read 94.4%. It reads 91.1% now,
and the reason is written up next to it rather than quietly edited out :)

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
independent draws from one model when they come from five. **That optimism is named here and not
currently quantified.** An earlier revision of this table quantified it, from a variance-inflation
factor and an `n_eff`-corrected interval computed on the fixed-partition cell C4, and that cell
was withdrawn on exchangeability grounds, so every figure derived from it is withdrawn with it.
The same applies to the 10,000-draw row this table used to carry. See
`OVERRIDE-RULING-2026-07-30.md` §1.5, which enumerates them. Requantifying the optimism from a
cell that survives is open work, not a finished result. The permutation p is reported as
**p ≤ 0.001** rather than `0.0010` because 1/1001 is the resolution floor of a 1000-shuffle test,
not a measurement; the scripts print the bound directly.

Across 1000 label shuffles, not one matched or exceeded the real result, so the
decoding is finding real structure rather than fitting noise. ("Matched or
exceeded" rather than "beat" because that is the comparison scikit-learn
actually counts.)

![Permutation null distribution](permutation_null.png)

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
  > 0.5–5 Hz is **53.3%**: 20 points below, not level. The old "match" compared a frontopolar
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
  wherever it appears. The same holds for muscle: there is no EMG reference channel either, so
  `emg_proxy.py` is a *high-band-power-at-muscle-adjacent-sites* probe and is named as one. It
  bounds a myogenic contribution inside 40 to 75 Hz and bounds **nothing** inside the decoder's
  own 8 to 30 Hz passband.
- **The falsifiable artifact test was run and did not falsify.** Deleting the sensorimotor strip
  leaves 77.8% (35/45). That is the single most important caveat on this page, it is written up
  in full above rather than buried here, and its registered verdict is *suggested, not
  established*.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python decode_csp.py        # full pipeline + CSP patterns → csp_patterns.png
```

Data downloads automatically on first run (cached in `~/mne_data`).

### Tests

```bash
python test_pipeline.py       # or: python -m pytest test_pipeline.py -q
```

19 regression tests, under a second, no data download. They are not "does it run" tests.
Each one guards a mistake this project actually made and had to retract, so a future edit
that reintroduces it fails here instead of in this README. The three that matter most:

- **95.9% and 47.4% are not attainable accuracies.** With 45 trials tested exactly once
  each, accuracy is a count over 45, so it can only land on multiples of 2.222%. Both
  numbers were published here as measurements. Neither is on the lattice, so neither ever
  was one.
- **CSP sits inside the Pipeline.** If it is fitted once outside, the spatial filters see
  the test trials and every accuracy in this repo is invalid.
- **`for_torch` rejects volts-scale data.** Feeding EEGNet volts leaves BatchNorm unable to
  normalise; the network scores exactly the majority-class rate while being dead, which
  reads as a plausible finding and is not one.

Shared definitions live in `common.py`: the classifier, the Wilson interval, Holm
correction, the channel sets and the loader. They used to be copied into each script,
because importing a script that defined them also ran its multi-minute analysis. Every
script has a `__main__` guard now, so importing costs nothing and there is one definition
of what "the published pipeline" means rather than five.

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

## Next

Cross-subject, harder contrasts and the EEGNet comparison are all built now (rungs 7–11). What
is actually left:

- **More trials per subject.** Almost every limitation above traces back to 45 trials: the
  quantized folds, the underpowered comparisons, the learning curve that cannot be run. Public
  corpora exist with 2,000–5,000 trials per subject. It is also what would give the
  sensorimotor-deletion McNemar enough discordant trials to decide anything.
- **Filter-bank CSP (FBCSP).** CSP per sub-band, combined by the classifier. A reliable gain, and
  still classical and interpretable.
- **The learning curve**, holding subject fixed and sweeping training-set size. That would settle
  a claim this project made and then retracted, that the barrier is sample size rather than
  anatomy.
- **ICA-based artifact rejection**, and a paradigm that records EOG.
- **The second arm of the muscle check.** `emg_proxy.py` is one half of a two-part condition:
  the high-band probe at muscle territory, which came back null. The other half is a
  temporal-channel-**deleted** ablation *inside* 8–30 Hz, which should not hurt appreciably if
  the decoder is not riding muscle. It has not been run, and a conjunction with one arm run is
  not satisfied.
- **EMG inside the decoder's own passband.** The probe covers 40–75 Hz and is blind at 8–30 Hz
  by construction, which is the only band the headline can actually be contaminated in. Closing
  that needs an instrument this montage and this sampling rate cannot provide.
- **Re-scoring `sweep_results.csv` on an exact null.** Run blocking moved one of the two median
  subjects across 0.05, so per-subject significance across the 109 was computed with a null
  that is exact but not the best available. Three subjects is an existence proof, not a survey.

See [EXPLAINER.md](EXPLAINER.md) §12 for the full scoreboard, including the complete list of
claims this project published and later retracted. That list only grows, corrections are added
to it, never swapped in over the record of the claim they correct.
