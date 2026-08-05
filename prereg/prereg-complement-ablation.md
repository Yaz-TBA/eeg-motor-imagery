# PRE-REGISTRATION: THE COMPLEMENT ABLATION

**Measurement 1 of 3 (W6).** Written 2026-07-25, before the committed script exists.
**Everything above the RESULTS heading at the bottom is frozen once this file is written.**
The results section is appended. Nothing above it is edited, including anything this run
turns out to contradict.

Target script: `/Users/yaz/Documents/Projects/eeg-motor-imagery/ablate_channels.py`
Comparison arms already in that script: `all 64`, `sensorimotor only (17)`, `frontopolar only (8)`,
`all 64, leave-one-run-out`.

---

## 0. THE DISCLOSURE THAT HAS TO COME FIRST: THIS IS NOT A BLIND PRE-REGISTRATION

**The complement arm has already been run and I have read the number before writing this file.**
Not disclosing that would make this document a lie of exactly the kind the workflow exists to stop.

On 2026-07-25 the hostile pass ran the arm in
`/Users/yaz/Documents/Projects/neuro-canon/runs/hostile-pass-2026-07-25/hostile_verify_A.py`
and its stdout at line 17 of `hostile_verify_A.stdout` reads:

```
sensorimotor DELETED (47 kept)         seed42  77.8% (35/45)   10-seed mean  79.3%  [75.6%, 84.4%]
```

That number is recorded in `CANON.md` at **A255**, marked `hostile-pass verification 2026-07-25
(uncommitted)`. It is not in `.provenance_cache`, no committed script produces it, and
`check_provenance.py` cannot back it. `CANON.md` A51 has already been marked SPENT on the strength
of it. `CURRICULUM.md` (the seven-row ladder) and `DRILLS.md` both quote it.

**So the outcome bands in section 6 were written with the answer visible.** A reader is entitled to
assume the bands were drawn around 79.3%. They were not drawn to flatter it, and section 6 puts the
unflattering band before the expected one, but the reader should not have to take my word for that,
which is why the prior number is printed here rather than buried.

**What this pre-registration still buys, honestly stated:**

1. **A committed, provenance-backed number replaces an uncommitted one.** A255's own remedy says
   "Land the occipito-parietal and complement conditions in `ablate_channels.py` ... then re-run
   `check_provenance.py` cold and record the date. Until then, section 8 is a section of things you
   have measured and cannot yet cite." This measurement is that remedy.
2. **Three registered quantities are genuinely blind.** They do not exist anywhere in the corpus:
   the **permutation test on the complement**, the **Wilson interval on the complement count**, and
   the **exact McNemar between all-64 and the complement on paired per-trial predictions**. A255
   carries none of them; A256's permutation test is on the occipito-parietal set and A257's McNemar
   is sensorimotor-only against occipito-parietal-only. Section 6's verdict rules turn on the McNemar,
   which is blind, not on the accuracy, which is not.
3. **A disagreement between the two implementations is itself a registered outcome** with a fixed
   meaning (section 7), rather than something to be explained after the fact.

Every registered quantity below is tagged **[KNOWN]** or **[BLIND]**. Do not let a KNOWN quantity be
reported as a confirmation of a prediction.

---

## 1. THE QUESTION

**Are the sensorimotor channels NECESSARY for the 91.1% headline, or only SUFFICIENT?**

Stated so it has a wrong answer: *delete the 17-channel FC/C/CP strip, keep the other 47 electrodes,
refit the whole pipeline. If the decoder depends on the sensorimotor strip, this must break it.*

The existing ablation establishes only the sufficiency half. `sensorimotor only (17 ch)` scores 95.6%
(43/45), so those channels **suffice**. No condition in the committed script deletes them and keeps
the rest, so **necessity is untested**, and `ablate_channels.py` says so in its own docstring
(lines 40-48): "the falsifiable form 'if the decoder reads sensorimotor cortex then deleting
sensorimotor cortex must break it' is NOT tested here and must not be attributed to this file."

This measurement builds that arm.

**SAY THE INSTRUMENT LIMIT BEFORE THE NUMBER, NOT AFTER IT.** On a 64-channel scalp montage with a
centimetre-scale point-spread function, **no channel-deletion experiment can falsify a source
hypothesis**, because deleting the electrodes nearest a source does not delete the source from the
remaining electrodes. That is a property of the instrument, not a hedge on the result, and it is true
whichever way the number lands. This measurement is a **sensor-space** measurement and licenses only
sensor-space claims.

---

## 2. CHANNEL DEFINITIONS, AS AN EXACT SET DIFFERENCE

### 2.1 The 64, in montage order

Read from the cached EDF after `eegbci.standardize(raw)`, subject 1 run 6, on 2026-07-25:

```
FC5 FC3 FC1 FCz FC2 FC4 FC6  C5 C3 C1 Cz C2 C4 C6  CP5 CP3 CP1 CPz CP2 CP4 CP6
Fp1 Fpz Fp2  AF7 AF3 AFz AF4 AF8  F7 F5 F3 F1 Fz F2 F4 F6 F8  FT7 FT8
T7 T8 T9 T10  TP7 TP8  P7 P5 P3 P1 Pz P2 P4 P6 P8  PO7 PO3 POz PO4 PO8  O1 Oz O2 Iz
```

### 2.2 SENSORIMOTOR, verbatim from `ablate_channels.py` lines 102-106 (17 channels)

```
FC3 FC1 FCz FC2 FC4
C5 C3 C1 Cz C2 C4 C6
CP3 CP1 CPz CP2 CP4
```

Reused unmodified. Not re-derived, not widened, not corrected. If this list is wrong, this
measurement is wrong in exactly the same way the existing sufficiency arm is wrong, which is the
point of reusing it: the two arms must be the same cut of the montage seen from both sides.

### 2.3 COMPLEMENT, the exact set difference, montage order preserved (47 channels)

Computed as `[c for c in ch_names if c not in SENSORIMOTOR]`, which is the same expression the
hostile-pass script used at its line 84, so the two implementations pick identically:

```
FC5 FC6  CP5 CP6
Fp1 Fpz Fp2  AF7 AF3 AFz AF4 AF8
F7 F5 F3 F1 Fz F2 F4 F6 F8  FT7 FT8
T7 T8 T9 T10  TP7 TP8
P7 P5 P3 P1 Pz P2 P4 P6 P8
PO7 PO3 POz PO4 PO8
O1 Oz O2 Iz
```

17 + 47 = 64. Asserted in code, not trusted to arithmetic done here (section 5.4).

### 2.4 THREE THINGS THE COMPLEMENT RETAINS, REGISTERED NOW SO THEY CANNOT BE DISCOVERED LATER

**(a) Four peri-Rolandic electrodes: FC5, FC6, CP5, CP6.** The 17-channel SENSORIMOTOR set does not
contain them, so the complement keeps them. **This arm is therefore not "sensorimotor cortex deleted."
It is "the 17-channel strip deleted."** Lateral sensorimotor coverage survives. Registered mitigation:
the script also runs the **wide FC/C/CP 21 complement (43 kept)**, which deletes FC5/FC6/CP5/CP6 as
well, so the peri-Rolandic leak is bounded rather than assumed away. Prior value for that stricter
arm, also **[KNOWN]**: 71.1% (32/45) at seed 42, 76.7% over ten seeds.

**(b) The peak of retained CSP component 2: POz, PO4, Oz.** From `gate/csp_pattern_channels.stdout`,
the strongest pattern component by extremity peaks parieto-occipital. The complement keeps that
entire territory, so a complement that decodes well is decoding partly from the electrodes the
posterior/visual-attentional hypothesis (A122, A256) already flagged.

**(c) The peak of retained CSP component 3: T8, T10, TP8.** That is temporalis muscle territory and
**nothing in the repo bounds an EMG contribution.** The complement keeps the whole temporal ring.
**Registered in advance: a high complement score is CONFOUNDED with the EMG hypothesis and must not
be reported as "posterior cortex also decodes" until measurement 2 (the high-band EMG proxy) exists.**
The permitted sentence is "the 47 non-strip electrodes decode above the floor, and I have not yet
bounded how much of that is myogenic."

**(d) No condition here controls for channel count.** 47 channels is a different CSP estimation
problem from 17 or 64 at a fixed `n_components=4`. This is the same confound `ablate_channels.py`
already declares for frontopolar-only (its docstring lines 40-48, canon A50 (iv)). It is not fixed
here. The script prints the channel count beside every accuracy so no reader can miss it.

---

## 3. THE PIPELINE, IDENTICAL TO THE COMMITTED ONE

Copied from `ablate_channels.py`, which is itself copied from `decode_csp.py` so the numbers are
comparable. No parameter is changed for this arm.

| stage | setting |
|---|---|
| data | PhysioNet EEGBCI, subject 1, runs 6/10/14, imagined both fists vs both feet |
| montage | `eegbci.standardize`, `standard_1005` |
| reference | average over **all 64 channels, BEFORE any subset is picked** |
| band-pass | 8.0 to 30.0 Hz, `fir_design="firwin"`, zero-phase, on the CONTINUOUS recording |
| epochs | tmin -1.0, tmax 4.0, `baseline=None`, `picks="eeg"` |
| crop | 1.0 to 2.0 s |
| features | `CSP(n_components=4, reg=None, log=True, norm_trace=False)` |
| classifier | `LinearDiscriminantAnalysis()` |
| pipeline | sklearn `Pipeline([("CSP", ...), ("LDA", ...)])`, **fresh instance per condition and per seed** |
| splitter | `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)` |
| scoring | `cross_val_score(..., error_score="raise")` and `cross_val_predict` on the same folds |
| n | 45 trials, 21 hands, 24 feet |
| chance | **majority class = 53.3% (24/45)**, never 50% |

**CSP IS FIT INSIDE EACH FOLD.** The pipeline object is constructed by a `make_clf()` factory and
handed to `cross_val_score`, which clones it per fold, so CSP never sees a test trial. Any change
that fits CSP outside the CV loop invalidates every number in this document.

**The average reference is the leak, and it is registered here, not discovered later.** It is
computed over all 64 electrodes before the 47 are picked, so **every complement channel carries
-1/64 of every sensorimotor channel.** The complement is not electrically independent of the strip.
This measurement therefore **BOUNDS** the strip's necessity; it cannot establish it. Re-referencing
the complement within its own 47 would make the subsets independent but would no longer be the
published pipeline, so it is registered as a **secondary** arm (section 4, arm 6), reported beside
the primary and never in place of it.

---

## 4. THE ARMS

Seeds: **seed 42 primary** (the published seed, the one every existing number uses), plus a
**ten-seed sweep over `range(10)`**, which is the protocol the hostile pass used and therefore the
only sweep that makes the replication a replication. Seed 42 is reported as a k/45 count; the sweep
is reported as mean and [min, max].

| # | arm | ch | status | why it is here |
|---|---|---|---|---|
| 1 | all 64 | 64 | [KNOWN] 91.1% (41/45), 10-seed 94.0% | the baseline every gap is measured against; also the pipeline liveness check |
| 2 | sensorimotor only | 17 | [KNOWN] 95.6% (43/45), 10-seed 94.4% | the sufficiency arm already in the script |
| 3 | frontopolar only | 8 | [KNOWN] 51.1% (23/45), 10-seed 47.1% | the ocular negative control already in the script; also the floor sanity check |
| 4 | **sensorimotor 17 DELETED, 47 kept** | **47** | **[KNOWN] 77.8% (35/45), 10-seed 79.3%** | **THE MEASUREMENT** |
| 5 | wide FC/C/CP 21 DELETED, 43 kept | 43 | [KNOWN] 71.1% (32/45), 10-seed 76.7% | bounds the FC5/FC6/CP5/CP6 leak in arm 4 |
| 6 | arm 4, re-referenced within its own 47 | 47 | **[BLIND]** | removes the -1/64 average-reference leak; secondary, reported beside arm 4 |
| 7 | permutation test on arm 4, 1000 shuffles, seed 42 | 47 | **[BLIND]** | is the complement above its own null at all |
| 8 | Wilson 95% CI on arm 4's seed-42 count | 47 | **[BLIND]** | the interval on 35/45 or whatever it lands on |
| 9 | exact McNemar, arm 1 vs arm 4, paired per-trial predictions, seed 42 folds | - | **[BLIND]** | **the decision statistic in section 6** |

The `wide FC/C/CP 21` set is derived the same way the hostile-pass script derived it
(`c[:2] in ("FC","CP") or (c[0]=="C" and c[1] != "P")`), and the script asserts it has 21 members.

Arms 2 and 3 are already in the committed script and are re-run unchanged so that a shift in the
environment shows up as a shift in a number that is already published, rather than silently inside
the new arm.

---

## 5. STATISTICS AND ASSERTS

### 5.1 Wilson 95% CI
Same function as `evaluate_honestly.py` lines 94-100, z = 1.96, on the seed-42 count out of 45.

### 5.2 Permutation test
`permutation_test_score`, 1000 shuffles, `random_state=42`, accuracy, same `StratifiedKFold(5,
shuffle=True, random_state=42)`. p reported as **`<= 0.001` when it bottoms out**, because
sklearn computes `p = (C+1)/(n+1)` and 1/1001 is the **resolution floor of the test, not a
measurement**. Assert the null is centred within 5 points of 50%, as `evaluate_honestly.py` does.

### 5.3 Exact McNemar, all-64 against complement
`cross_val_predict` on the same seed-42 folds for both arms. Print the full 2x2: both correct,
all-64 only (b), complement only (c), neither. p from `scipy.stats.binomtest(b, b+c, 0.5)`,
two-sided exact, **not** the chi-square approximation, because the discordant count will be small.

**The p is quoted with its discordant count or not at all.** Precedent A56: at 10 discordant pairs a
one-trial shift moves the exact p from 0.109 to 0.754, a factor of about seven. The script must also
print, for the observed `n_disc`, **the most lopsided split that would still have failed to reach
p < 0.05**, which is an honest statement of what this test could and could not have detected at that
n_disc, computed rather than asserted.

### 5.4 Asserts that must be in the code
1. `len(SENSORIMOTOR) == 17`, `len(COMPLEMENT) == 47`, `len(WIDE) == 21`, `len(NOT_WIDE) == 43`.
2. `set(SENSORIMOTOR) | set(COMPLEMENT) == set(ch_names)` and
   `set(SENSORIMOTOR) & set(COMPLEMENT) == set()`. The complement is a partition of the montage or
   the arm is not a complement.
3. Every channel named is present in the montage (`ablate_channels.py` line 189 pattern).
4. **The k/45 lattice.** Every accuracy is `k/45` for integer k, steps of 2.2222%. Assert
   `abs(scores.mean() - n_correct/45) < 1e-9` for every condition, as `ablate_channels.py` lines
   210-214 do. A value off the lattice is a tell that it was never computed.
5. **Positive control:** arm 1 must beat the majority rate by more than `TOL = 1e-9`. A majority-class
   dummy scores exactly 24/45 and floating point puts it 1.1e-16 above a bare `> chance`, so the
   tolerance is load-bearing, not decoration (`decode_csp.py` lines 111-118).
6. **Negative control:** arm 3 (frontopolar) must stay within 3 trials of the majority rate. If the
   negative control has come alive, the run is not interpretable and nothing else in it should be read.

---

## 6. PRE-REGISTERED OUTCOMES

### 6.1 The noise band, fixed before the run

On n = 45 tested once each, **one trial is 2.222 points and two trials are 4.444 points.**
The corpus has already committed to this at **A48**: "43/45 against 41/45 is two trials. Say
'dropping 47 non-motor channels does not hurt'. Never say the sensorimotor subset is better, even
though the script prints `+4.4 points`." A rule the project applied when the difference flattered it
applies here unchanged.

**Pre-committed thresholds.** Let **G = (all-64 ten-seed mean) minus (complement ten-seed mean)**,
in points.

- **G of 4.4 points or less, either sign, is NOT a difference.** Two trials.
- The seed-42 point difference **is not the decision statistic.** `evaluate_honestly.py` section 6
  sweeps 100 seeds and finds the all-64 headline moves several points on seed placement alone, and
  `ablate_channels.py` already annotates its own `+4.4 points` row as "one draw, not an effect size"
  (lines 248-254). A single quantized draw cannot carry this.
- **The verdict requires two independent things to agree:** a ten-seed mean gap **G > 10.0 points**
  (more than twice the noise band, more than 4.5 trials) **AND** exact McNemar **p < 0.05** on the
  seed-42 paired predictions. The McNemar is the blind half.
- **If only one fires, the registered verdict is "suggestive, not established at n = 45."** It is not
  upgraded by choosing whichever of the two reads better.
- **"No difference detected" is never written as "no difference."** At n = 45 with a small discordant
  count this design has poor power, and section 5.3's printed line says how poor.

### 6.2 The bands

`is_this_good_news` is scored against **the project's leaned-on framing**, that the decoder reads
sensorimotor cortex. Good = supports it. Bad = undercuts it. Scoring it this way, in advance, is what
stops the write-up from redefining "good" after the fact.

The bands are defined on **G first**, because G is the decision quantity and it stays meaningful even
if arm 1 replicates to something other than 94.0%. The absolute column is the arithmetic equivalent
**assuming arm 1 lands at its prior ten-seed mean of 94.0%**; if arm 1 moves, G governs and the
absolute column moves with it.

| band | gap G (points) | complement ten-seed mean, if all-64 = 94.0% | verdict, fixed in advance | for the framing |
|---|---|---|---|---|
| **A. COLLAPSE** | G >= 38.4 | **at or below 55.6%** (25/45, within one trial of the 53.3% floor) | The 47 non-strip electrodes carry nothing. The strip is **necessary as well as sufficient**. The falsifiable form survives its test. | **GOOD**, and the strongest result available here |
| **B. SEVERE LOSS** | 22.9 to 38.4 | 55.6% to 71.1% | Complement decodes above the floor but is crippled. Signal strongly concentrated over the strip; a real residual elsewhere that must be disclosed, not rounded to zero. | **GOOD**, with a disclosed residual |
| **C. SUBSTANTIAL LOSS, NOT NECESSARY** | 10.0 to 22.9 | 71.1% to 84.0% | Complement decodes **well above the floor and well below all-64**, and the gap clears the 10.0-point threshold. If McNemar also gives p < 0.05: the strip is **sufficient but NOT necessary**, and "deleting sensorimotor cortex breaks it" is **FALSE at this instrument**. The permitted sentence is the density one: 17 channels reach 94.4% and the 47 that exclude them reach the complement value, so per-channel discriminative density is several times higher over the strip, **and that is a sensor-space claim, not a source claim**. If McNemar gives p >= 0.05, the verdict downgrades to band C2's wording even though G cleared. | **BAD.** This is where the prior run sits (79.3%, G = 14.7). Registered as the expected outcome. |
| **C2. SUGGESTIVE, NOT ESTABLISHED** | 4.4 to 10.0 | 84.0% to 89.6% | Complement is below all-64 by more than two trials but less than the pre-committed 10.0-point threshold. **Registered verdict: a loss is suggested and not established at n = 45.** Do not upgrade it with the McNemar; the rule in 6.1 requires both, and this band fails the G half by construction. | **NEUTRAL**, and it must be reported as undecided rather than leaned either way |
| **D. MATCH** | \|G\| <= 4.4 (two trials or less, either sign) | 89.6% to 98.4% | The complement is **not distinguishable from the full montage**. The strip contributes nothing the rest of the montage does not already carry. Every sensorimotor framing in the corpus weakens, and the sufficiency arm is reduced to "these 17 channels are one of several sets that work." | **BAD, and the worst realistic case.** Write it plainly if it lands. |
| **E. EXCEEDS** | G < -4.4 | above 98.4% | The complement **beats** the full montage by more than two trials. Either the strip carries noise the classifier is better off without at `n_components=4`, or something is wrong. **Treat as suspect first** (section 7), and only report as a result once the section 7 checks pass. | **BAD**, and requires an integrity check before it is reported at all |

The bands are exhaustive on G and they do not overlap: A and B are absolute-floor bands that also
satisfy G > 22.9, C covers 10.0 to 22.9, C2 covers 4.4 to 10.0, D covers -4.4 to 4.4, E covers
below -4.4. If a run lands on a boundary, the **more conservative** band applies, meaning the one
that claims less about the strip being necessary.

**The two-part rule of 6.1 applies to A, B and C alike, not only to C.** A, B and C all assert a real
loss, so all three require the McNemar to fire as well as G. At a gap of 10 or more points the McNemar
should fire comfortably, so **if G lands in A or B and the McNemar does NOT reach p < 0.05, that is a
red flag about the pairing, not a licence to report the gap anyway**: check that both arms were
predicted on the same folds with the same seed, and that `cross_val_predict` and `cross_val_score`
agree on each arm's count before reading anything into the discordant split.

Two things that are true in **every** band and must be printed in every band:

- **The average-reference leak (section 3).** The complement is not electrically free of the strip.
  A high complement score is partly what volume conduction plus a 64-channel average reference
  **predicts**, so band C or D is weaker evidence against the sensorimotor framing than it looks,
  and band A would be **stronger** evidence for it than a clean-subset design could give.
- **The instrument limit (section 1).** No band licenses a source claim in either direction.
  A91 pointed both ways: forward-is-not-inverse refutes a negative source claim exactly as hard as a
  positive one.

### 6.3 What gets said out loud, per band

- Bands A and B: "I ran the arm the script declines to build. Deleting the strip costs N points."
- Band C: **"The committed script did not build that arm. I built it. The strip is sufficient, not
  necessary: 47 electrodes that exclude it still reach X%. Per-channel density is several times
  higher over the strip, and that is a sensor-space claim."** Instrument limit first, number second.
- Band C2: **"The complement is lower, by less than this design can establish at n = 45. Undecided."**
- Band D: **"The sensorimotor framing does not survive this. 47 non-strip electrodes match the full
  montage within two trials. The strip is sufficient and interchangeable."**
- Band E: run section 7 before saying anything.

---

## 7. WHAT WOULD FALSIFY THE ANALYSIS RATHER THAN THE HYPOTHESIS

Each of these means **the measurement is broken**, not that the hypothesis moved. If any fires, no
number from the run is quotable until it is resolved.

1. **Arm 1 does not reproduce 91.1% (41/45) at seed 42.** The all-64 arm is the pipeline's fingerprint.
   If it moves, the environment or the pipeline moved, and every other arm is measuring something
   other than the published pipeline. Stop and diff.
2. **Arms 2 or 3 do not reproduce 95.6% (43/45) and 51.1% (23/45) at seed 42.** Same reason, one level
   down. These are already in the committed script's stdout.
3. **`len(COMPLEMENT) != 47`**, or SENSORIMOTOR and COMPLEMENT are not a disjoint partition of the 64.
   The arm is then not a complement and its name is false.
4. **Any accuracy off the k/45 lattice**, or `fold-mean != pooled count`. That is the exact defect the
   old README table carried (95.9%, 47.4%) and it is a tell that a number was not computed.
5. **The complement ten-seed mean differs from the prior 79.3% by more than 2.2 points (one trial).**
   Two implementations of the same set difference on the same data disagree. **Registered response:
   report both, do not average them, do not pick the better one, and resolve the discrepancy before
   either enters CANON.** The most likely cause is a channel-picking or ordering difference, which is
   diagnosable by printing both channel lists and comparing them as sets and as ordered sequences.
6. **The complement falls at or below the frontopolar-8 value (47.1% ten-seed, 51.1% seed 42).**
   47 channels including the entire posterior montage scoring at or below 8 frontopolar channels is
   not a plausible neurophysiological result; it is a picking, labelling or crop bug.
7. **The permutation null for the complement is not centred within 5 points of 50%.** The null is
   mis-specified and the p does not mean what it appears to.
8. **The negative control (arm 3) comes alive**, i.e. moves more than 3 trials off the majority rate.
9. **Band E lands.** A complement that beats the full montage is registered as suspect-first: check
   channel picking, check that CSP is inside the fold, check the crop, and re-run before reporting.

---

## 8. DELIVERABLE AND PROVENANCE

- The arm **lands in `ablate_channels.py`** as new conditions, not in a side script. A255's remedy
  requires exactly this, and a number that only exists in `runs/` is the problem being fixed.
- The docstring lines 40-48, which currently declare the arm absent, **must be amended in the same
  edit, with the old text kept visible as a withdrawal** in the house style already used at lines
  50-61. The script must not simultaneously build the arm and claim it does not.
- Every accuracy prints **with its baseline and its k/n count**, e.g. `77.8% (35/45)`, plus channel
  count, ten-seed mean and range, and the gap in points against all-64.
- A **"what this does NOT show"** block prints: the average-reference leak, the instrument limit, the
  retained FC5/FC6/CP5/CP6, the retained T8/T10/TP8 and the unbounded EMG confound, the channel-count
  confound, n = 45 one subject one session.
- **`REGISTRY` in `check_provenance.py` must be updated.** `ablate_channels.py` is currently budgeted
  at 180 s for four single-seed conditions. Adding ten-seed sweeps and a permutation test will exceed
  that. Per the file's own rule the runtime is **measured, not guessed**: time it twice, take the
  higher, round up, and let the 4x headroom rule set the timeout.
- Then **`check_provenance.py` cold**, and record the date. Until that passes, the complement number
  stays in the "measured and cannot yet cite" column no matter how good it looks.
- No number from this run enters `CANON.md`, `CURRICULUM.md` or `DRILLS.md` before it appears in a
  committed script's stdout.

---

## 9. REGISTERED RISKS

1. **Non-blind on the headline accuracy.** Disclosed in section 0. The decision statistic (McNemar)
   and three of the nine arms are blind; the accuracy is not.
2. **The average reference leaks the strip into the complement at -1/64 per channel.** Bounds, does
   not eliminate. Cuts against the unflattering reading, which is why it is stated in every band and
   not only in the band where it helps.
3. **The complement is not "sensorimotor deleted."** It keeps FC5, FC6, CP5, CP6. Arm 5 bounds this.
4. **The complement keeps T8/T10/TP8, and no EMG bound exists yet.** A complement that decodes well
   is confounded with a myogenic explanation until measurement 2 runs. Registered as a hard block on
   the "posterior cortex also decodes" sentence.
5. **The complement keeps POz/PO4/Oz**, the peak of the strongest retained CSP pattern component, so
   this arm overlaps the unresolved posterior/visual-attentional question at A257. That question is
   **unresolved**, and this measurement does not resolve it.
6. **No channel-count control.** 47 vs 17 vs 64 at fixed `n_components=4` are different estimation
   problems. Same declared confound as frontopolar-only (A50 (iv)).
7. **Ten seeds is a small sample of splits.** Report the range, never the mean alone.
8. **McNemar on a small discordant count is unstable** (A56: one trial, factor of seven). The count
   travels with the p, always.
9. **n = 45, one subject, one session, three runs.** Everything here is subject-1 specific and the
   leave-one-run-out control does not remove a session-level trend, because EEGMMIDB has no second
   session.
10. **Confirmation pressure toward band A.** Band A is the outcome the project would prefer and it is
    also the outcome that would most likely indicate a bug (a 47-channel montage collapsing to the
    floor). Section 7 item 6 exists so band A gets the same suspicion band E gets.
11. **`ablate_channels.py` is a document as much as a script.** Editing it to add the arm risks
    breaking the numbers other documents quote from its stdout. Arms 1 to 3 are re-run unchanged as
    the guard.

---

## RESULTS

Run 2026-07-25. Script: `/Users/yaz/Documents/Projects/eeg-motor-imagery/ablate_channels.py`
(the arm landed in the target file, as section 8 requires, not in a side script).
Command: `.venv/bin/python ablate_channels.py`. Wall time 15.48 s and 15.65 s on two runs;
every measured value byte-identical across both. mne 1.12.1, scikit-learn 1.9.0, scipy 1.18.0.
Nothing above this line was edited.

### The nine arms

| # | arm | ch | seed 42 | ten-seed mean | ten-seed range | tag |
|---|---|---|---|---|---|---|
| 1 | all 64 | 64 | 91.1% (41/45) | 94.0% | [91.1%, 95.6%] | [KNOWN] reproduced exactly |
| 2 | sensorimotor only | 17 | 95.6% (43/45) | 94.4% | [88.9%, 97.8%] | [KNOWN] reproduced exactly |
| 3 | frontopolar only | 8 | 51.1% (23/45) | 47.1% | [42.2%, 53.3%] | [KNOWN] reproduced exactly |
| 4 | **sensorimotor DELETED** | **47** | **77.8% (35/45)** | **79.3%** | **[75.6%, 84.4%]** | **[KNOWN] reproduced exactly** |
| 5 | wide FC/C/CP DELETED | 43 | 71.1% (32/45) | 76.7% | [66.7%, 91.1%] | [KNOWN] reproduced exactly |
| 6 | arm 4, re-referenced within its own 47 | 47 | 68.9% (31/45) | 73.3% | [64.4%, 82.2%] | **[BLIND]** |
| 7 | permutation on arm 4, 1000 shuffles | 47 | p <= 0.001 | null 51.0% +/- 8.8%, max 73.3% | | **[BLIND]** |
| 8 | Wilson 95% CI on 35/45 | 47 | [63.7%, 87.5%], width 23.7 points | | | **[BLIND]** |
| 9 | exact McNemar, arm 1 vs arm 4 | | b = 7, c = 1, n_disc = 8, p = 0.0703 | | | **[BLIND]** |

Leave-one-run-out, re-run unchanged: 93.3% (42/45).
2x2 for arm 9: both correct 34, all-64 only 7, complement only 1, neither 3, sums to 45.

Arm 4 reproduced the prior uncommitted hostile-pass value to 0.0 points at both seed 42 and
the ten-seed mean. Two independent implementations of the same set difference agree exactly.

### Falsifiers: 10 of 10 passed

No analysis-falsifier fired. Positive control passed (all-64 beats the majority rate by more
than TOL = 1e-9), negative control passed (frontopolar 1 trial off the majority rate), the
permutation null sits 1.0 points off 50%, every accuracy is on the k/45 lattice and every
fold-mean equals its pooled `cross_val_predict` count, and the complement is a disjoint
47-channel partition of the montage with the 17.

### Verdict

**G = 94.0 - 79.3 = +14.7 points**, which lands in **band C** on G. The McNemar gives
**p = 0.0703 on n_disc = 8**, which does **not** reach p < 0.05.

Band C's own registered rule applies: *"If McNemar gives p >= 0.05, the verdict downgrades to
band C2's wording even though G cleared."*

**REGISTERED VERDICT: a loss is SUGGESTED and NOT ESTABLISHED at n = 45. Undecided, leaned
neither way.** The band C conclusion, that the strip is sufficient but not necessary, is **not
established by this run** and is not written. Both halves of the two-part rule get said or
neither does.

Why the McNemar failed, in the terms fixed in advance: at n_disc = 8 the observed 7-vs-1 split
**is** the most lopsided split that still misses p < 0.05. Only an 8-0 sweep would have reached
it. The test had essentially no power at the discordant count it got, and that count is small
because the two arms agree on 34 of 45 trials and both miss 3 more. The pairing worked; the
sample is small.

### The two statements that print in every band

1. **The average-reference leak.** The reference is computed over all 64 electrodes before the
   47 are picked, so every complement channel carries -1/64 of every sensorimotor channel. This
   measurement bounds the strip's necessity; it cannot establish it. The blind secondary arm
   quantifies the leak for the first time: removing it costs the complement 6.0 points over ten
   seeds (79.3% to 73.3%), so the leak is real and it inflates the complement.
2. **The instrument limit.** No band licenses a source claim in either direction.

### One process deviation, disclosed

Falsifier (10) was first implemented too widely, firing whenever G >= 10.0 and the McNemar
missed, and the first run therefore printed "1 FIRED". Section 6.2 of this document scopes that
falsifier to bands A and B only, and band C registers the failed-McNemar case as its own
downgrade path. The check was corrected to the registered scope **after** the first run, with
the answer visible. It changes no measured value: the 2x2, the p, G and every accuracy are
identical either way. What it changes is whether the run is reportable at all, which is why it
is recorded here and in a comment in the script rather than quietly fixed.

### Provenance

`REGISTRY` in `check_provenance.py` updated from a guessed 180 s to a measured 20 s (higher of
two timings, rounded up, 80 s timeout at the 4x rule). `check_provenance.py` run with the
`ablate_channels.py` cache deleted: the script ran cold inside its budget and its stdout was
recaptured. Every number in the table above now appears in a script's stdout in the repo, which
is what A255's remedy asked for. The repo-level provenance exit is still FAIL, for seven
pre-existing UNBACKED claims in `EXPLAINER.md` (57.5%, 53.9%, HEOG p = 0.27, VEOG p = 0.44) and
one unregistered sibling script, none of which this measurement touches.

---

## RESULTS ADDENDUM, 2026-07-26: FOUR CORRECTIONS AFTER AN ADVERSARIAL PASS

Appended, not edited in. Everything above this heading is left exactly as it was written,
including the sentences this addendum withdraws, because a withdrawal that deletes the
withdrawn text hides what was claimed. Re-run of `ablate_channels.py` after the repairs:
exit 0, all ten registered falsifiers still pass, and **every registered number above is
unchanged**. Arms 1 to 9 reproduce to the digit. What changed is what the run SAYS.

### Correction 1: the McNemar half of the registered rule cannot fire at the registered G threshold

In a paired 2x2 on the same 45 trials, `b - c` is identically (all-64 correct) minus
(complement correct), so the accuracy gap in trials fixes `b - c` and only `c` is free.
Enumerating every `(b, c)` with `b + c <= 45`, the smallest trial gap that reaches
p < 0.05 at ANY discordant count is **6 trials = 13.3 points** (first at n_disc = 6, 6 vs 0,
p = 0.0312). The registered G threshold is 10.0 points = 4.5 trials, which is **below that
floor**. Single-seed gaps between 10.0 and 13.3 points cannot reach p < 0.05 at any n_disc.

Section 6.2's sentence **"At a gap of 10 or more points the McNemar should fire comfortably"
is therefore REFUTED**, and it was checkable as false when it was written, from n = 45 and
alpha = 0.05 alone. That sentence is load-bearing twice: it is the stated reason falsifier
(10) exists, and it is the reason a failed McNemar inside band C was booked as a registered
outcome rather than as a defect in the rule.

**The pre-registration is not edited to fix this.** The registered thresholds stand and the
registered verdict stands. What is recorded is that **the two-part rule as written cannot
certify band C at n = 45**, so "a loss is SUGGESTED and NOT ESTABLISHED" is a property of the
rule at least as much as a reading of the data. The script now computes the detection floor
from the design and prints it BEFORE the arms run.

Scope, stated so this is not read as more than it is: G is a ten-seed MEAN gap and the McNemar
is computed on ONE partition, so the 13.3-point floor binds the seed-42 gap directly and G
only through their correlation.

### Correction 2: the verdict is not robust to the seed, and the two halves of the rule were evaluated on disjoint seed sets

G is a mean over `range(10)`. The McNemar was computed at seed 42, which is **not a member of
that sweep**. The exact McNemar on all ten registered sweep seeds, now printed by the script:

| seed | all-64 | complement | gap | b | c | n_disc | p |
|---|---|---|---|---|---|---|---|
| 0 | 42/45 | 36/45 | 6 | 7 | 1 | 8 | 0.0703 |
| 1 | 43/45 | 36/45 | 7 | 8 | 1 | 9 | **0.0391** |
| 2 | 42/45 | 35/45 | 7 | 8 | 1 | 9 | **0.0391** |
| 3 | 41/45 | 34/45 | 7 | 9 | 2 | 11 | 0.0654 |
| 4 | 43/45 | 34/45 | 9 | 10 | 1 | 11 | **0.0117** |
| 5 | 43/45 | 36/45 | 7 | 8 | 1 | 9 | **0.0391** |
| 6 | 42/45 | 38/45 | 4 | 5 | 1 | 6 | 0.2188 |
| 7 | 43/45 | 35/45 | 8 | 10 | 2 | 12 | **0.0386** |
| 8 | 42/45 | 35/45 | 7 | 7 | 0 | 7 | **0.0156** |
| 9 | 42/45 | 38/45 | 4 | 6 | 2 | 8 | 0.2891 |

Median 0.0391. **Six of ten reach p < 0.05.** Seed 42 gives 0.0703 and has a smaller p than
only three of the ten. Additionally, seed 42 was near-predetermined to miss: with marginals
41/45 and 35/45, both KNOWN before this pre-registration was written and both printed in its
section 0 and arm table, `b - c = 6` is forced, and of the 20 attainable configurations
**exactly one** (c = 0, b = 6, n_disc = 6, p = 0.0312) reaches p < 0.05. The observed c = 1
gives 0.0703; c = 2 gives 0.1094; larger c is worse.

**The verdict is NOT flipped.** Ten non-independent re-splits of the same 45 trials are not
ten samples, so "established" is not licensed either. What is recorded is that the registered
outcome is substantially a fact about which integer was typed as `random_state`.

### Correction 3: arm 6 is a rank-1 common-mode projection, not "the leak removed"

The secondary arm's manipulation is now measured rather than described. The re-referenced data
equals the primary complement data minus its own across-channel mean:
max |difference| = 9.15e-20 against a data scale of 1.58e-04, i.e. 5.8e-16 relative, and the
difference is uniform across channels. Rank drops **47 to 46**. Both facts are now asserted in
the script.

So arm 6 **deletes one spatial dimension, the within-47 common mode**. The direction it deletes
has time course m64(x) - m47(x) = -(17/64)(m47(x) - m17(x)), which mixes the average-referenced
strip contribution with (17/64) of the complement's OWN global component. **The 6.0 points
cannot be assigned to the leak alone.**

WITHDRAWN from the RESULTS section above: *"The blind secondary arm quantifies the leak for the
first time: removing it costs the complement 6.0 points over ten seeds (79.3% to 73.3%), so the
leak is real and it inflates the complement."* The causal clause is unidentified. A clean arm
would project out only an estimate of the strip's common mode and leave the complement's
intact; that arm is not built and is not claimed.

Applying the project's own standard, which arm 6 had been exempt from: 6.0 points is 2.7 trials,
above the 4.444-point two-trial band and below the 10.0-point G threshold, with no confidence
interval, no significance test and no registered threshold of its own. Section 5 registered
arm 6 as a measurement and registered **no interpretation rule for it**, so the entire reading
was post hoc. Under the rule this project already hard-codes, arm 6 is **SUGGESTED, NOT
ESTABLISHED**, on exactly the standard applied to G.

### Correction 4: the declared channel-count confound, measured (POST-REGISTRATION arm 10)

Section 2.4(d) and registered risk 6 declare "no condition here controls for channel count" and
then do not run the control. It is now run, as arm 10 in `ablate_channels.py`.

**NOT BLIND, and disclosed as such**: this arm was added on 2026-07-26 with the answer already
visible, on an adversarial pass's instruction. It is not a prediction that was tested. It is a
declared confound that was finally measured.

Delete 17 channels **at random**, keep 47, identical pipeline, identical ten seeds, 50 seeded
draws:

- random-47 ten-seed mean **93.5%**, range [90.7%, 95.8%]
- random-47 G null: mean **+0.5** points, range [-1.8, +3.3], sd 1.1
- observed G (strip deleted) = **+14.7** points; draws at or beyond it: **0/50**
- draws at or below the complement's 79.3%: **0/50**
- empirical p = (C+1)/(N+1) = **0.0196**, whose resolution floor is 1/51 = 0.0196
- observed G sits **12.4 null SDs** above the null mean

Deleting 17 channels costs essentially nothing. Deleting THE STRIP costs far more than any
random deletion reached.

**What this retires**: caveat (v) in `ablate_channels.py` and the 2.4(d) / risk 6 confound, **for
the 47-vs-64 comparison only**. It is NOT retired for the 17-channel or 8-channel arms: this
control deletes 17 and keeps 47, so it says nothing about how a 17-channel or an 8-channel CSP
estimation problem differs from a 64-channel one.

**What this does NOT do**: it does not substitute for the registered decision rule, which stays
exactly as registered; it is not blind; and a permutation over channel SETS is not a permutation
over LABELS, so it cannot speak to whether the complement decodes at all. It tests "is the strip
special among 17-channel deletions", which is a different and more relevant null for a NECESSITY
question than the paired per-trial McNemar, but it is not the registered one.

### What the registered verdict still is

**BAND C on G. McNemar p = 0.0703 on n_disc = 8, which does not reach p < 0.05. Band C's own
registered clause applies and the verdict downgrades to band C2's wording: a loss is SUGGESTED
and NOT ESTABLISHED at n = 45. Undecided, leaned neither way.** That is unchanged. Corrections
1 and 2 say the rule could not have said otherwise across the lower third of band C; correction
4 says a differently-calibrated and confound-controlled test comes out decisively. Those are
reported side by side and neither is allowed to overwrite the other.

### A wording prohibition, recorded because it was crossed in a report of this run

A report of the first run opened with "Deleting the sensorimotor strip does NOT break the
decoder" and listed, under "Two things are established regardless", that "the falsifiable form
... did not produce a break". Neither phrasing appears anywhere in this repository:
`grep` over the script's stdout, this pre-registration, `README.md`, `EXPLAINER.md` and every
`.py` and `.md` in the project tree returns no match for "does not break", "did not produce a
break", or "established regardless". They were a reporter's additions on top of a run whose
printed verdict is undecided, and the reworded version was **stronger** than the version the
rule suppressed.

Both are prohibited. "No break" is band C's conclusion in paraphrase (band B is registered as
"crippled", which is also a break), so it is a band-C-and-below claim, not a floor claim.
"Deleting sensorimotor cortex" is source-level wording, which section 1, section 6.2 statement
2 and the script's own block (ii) all forbid in either direction; and the arm keeps FC5, FC6,
CP5 and CP6, so it is "the 17-channel strip deleted" and not "sensorimotor cortex deleted".

The permitted floor statement, scoped and in sensor space: **the 47 electrodes remaining after
the 17-channel strip is deleted score 35/45 = 77.8% at seed 42, above the 53.3% majority floor,
with a 1000-shuffle permutation p <= 0.001.** Do not write "break", "does not break",
"sensorimotor cortex" or "necessary" in any sentence describing this run's outcome, and do not
place any statement of the falsifiable form's fate under a heading containing the word
"established".
