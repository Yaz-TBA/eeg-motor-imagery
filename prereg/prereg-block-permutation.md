# PRE-REGISTRATION: THE BLOCK-PERMUTATION NULL

**Written 2026-07-25. Nothing executable exists yet. No number in Section 6 may be edited after the
script runs.**

This converts a disclosure into a measurement. The disclosure, from the hostile statistician:

> Yaz can read a permutation test but cannot design one. He cannot state the exchangeable unit;
> sklearn re-stratifies on the permuted labels so the null marginalises over a partition the observed
> value conditions on; and the global 900-trial LOSO shuffle destroys the subject blocking that makes
> the null meaningful. The words "exchangeability", "block permutation" and "fixed partition" appear
> nowhere in the curriculum or the repo.

The last sentence was true of `eeg-motor-imagery/` at the time it was written and is still true of the
code. Verified 2026-07-25 by grep over `*.py` and `*.md` in the repo: `exchangeab` 0 files,
`block permut` 0 files, `fixed partition` 0 files, `re-stratif` 0 files. Every occurrence of those
terms in this project lives in `neuro-canon/`, which is commentary, not the artifact a PI would run.

---

## 0. THE DISCLOSURE THAT HAS TO COME FIRST: A PILOT ALREADY EXISTS AND I HAVE SEEN ITS NUMBERS

This is **not** a blind pre-registration and saying otherwise would be the exact failure mode this
workflow exists to correct. Before writing a line of this document I read
`neuro-canon/runs/hostile-pass-2026-07-25/hostile_verify_A.py` and its stdout, and canon entries
A268, A269, A270. A pilot of both arms already ran on 2026-07-25. It is marked `uncommitted` and
`DERIVED`, it has no committed script in `eeg-motor-imagery/`, and its numbers are:

**A269, within-subject, subject 1, observed 91.1% (41/45), 300 draws per cell:**

| pilot cell | null mean | null sd | null max | p |
|---|---|---|---|---|
| (i) i.i.d. shuffle, folds RE-STRATIFIED each draw (what sklearn does) | 50.7% | 8.9% | 86.7% | 0.0033 |
| (ii) i.i.d. shuffle, partition HELD FIXED at the observed one | 47.7% | 8.4% | 71.1% | 0.0033 |
| (iii) within-run cyclic shift, folds re-stratified | 49.0% | 8.6% | 73.3% | 0.0033 |
| (iv) within-run permutation, folds re-stratified | 50.2% | 8.2% | 68.9% | 0.0033 |

**A270, cross-subject LOSO, observed 59.4%, 200 draws per cell:** global null 49.6% +/- 2.0, range
[42.4%, 54.8%]; within-subject block null 50.0% +/- 1.2, range [46.2%, 52.8%]. Both p = 0.0050.

**So what is this document for, if the pilot exists?** Five things the pilot cannot deliver, each of
which is a defect in the pilot and not a presentation problem:

1. **Every p in the pilot is at its floor.** 0.0033 is (0+1)/(300+1) and 0.0050 is (0+1)/(200+1).
   All six cells report the resolution limit of their own draw count. The pilot therefore **cannot
   answer the question the objection poses**, which is whether the p-value moves. Six numbers that
   are all the same number by construction are not a comparison.
2. **The pilot's cells are unpaired.** `hostile_verify_A.py:240` creates one `default_rng(42)` and the
   four cells consume it sequentially, so cell (i) and cell (ii) see **different label vectors**. The
   3.0-point gap between 50.7% and 47.7% is therefore confounded with Monte Carlo noise, and the
   max-statistic comparison (86.7% against 71.1%) is the one the canon entry itself withdraws as
   "too noisy a statistic to reason from". The correction under test is a change of **partition rule
   only**, so the two cells must be fed identical labels or the contrast is not the contrast.
3. **The pilot never ran the fully corrected cell.** Cells (iii) and (iv) change the shuffling scheme
   but keep the re-stratified partition, so they change two things at once relative to (i). The cell
   that is (within-run shuffle) AND (fixed partition) does not exist in the pilot. That cell is the
   corrected within-subject null.
4. **The pilot tests only subject 1, where the p cannot move.** See Section 5.1: at a 4.5 sd effect no
   feasible draw count produces a non-floor p. An arm where the answer is fixed in advance by the
   effect size cannot detect a design error.
5. **The pilot produces no artifact.** No committed script, no stdout in `eeg-motor-imagery/`, nothing
   `check_claims.py` or `check_provenance.py` can match against. The PI's criticism is that this
   project concedes rather than measures; a measurement that lives only in a canon footnote marked
   `uncommitted` is closer to a concession than to a result.

**Consequence for how this must be read and reported.** A269 and A270 are the **prior**, stated in
advance and in full above. Anything in Section 6 that agrees with them is a **confirmation at higher
resolution with a paired design**, and must be described that way. It is not an independent
discovery, and this document may not be cited as one. Where Section 6 disagrees with them, the
disagreement is the finding, and the pilot numbers above are what it is measured against.

**One prediction has already failed and I am not allowed to make it again.** A269 records that a
heavier right tail was predicted for the restricted (fixed-partition) null and the direction
reversed. Section 7 registers a mechanism that predicts the direction the pilot actually observed,
and registers it as a **separate, secondary probe** whose failure does not touch the primary result.
Measuring and explaining are separate steps.

---

## 1. THE QUESTION

**Does correcting the null change the verdict, and if not, what does it change?**

Stated so it has a wrong answer: for each of two designs there is a published null and a corrected
null. The corrected null either moves the p-value across a decision threshold, or moves the null's
shape without moving the p-value, or moves neither. Section 6 fixes what each of those means before
the script runs.

Two sub-questions, one per design:

**(a) Within-subject.** `decode_csp.py:124` and `evaluate_honestly.py:185` call
`permutation_test_score` with `cv=StratifiedKFold(5, shuffle=True, random_state=42)`. Does holding
the partition fixed at the observed one change the null, the p-value, or neither?

**(b) Cross-subject.** `cross_subject.py:135` draws one global permutation of all 900 pooled labels
and checks it against `SHUFFLE_MAX = 0.60`. Does permuting within subject instead change the null,
the p-value, or the threshold that guard should be using?

---

## 2. THE MECHANISM, VERIFIED IN THE INSTALLED SOURCE RATHER THAN ASSERTED

Environment: scikit-learn 1.9.0, numpy 2.5.1, mne 1.12.1, `eeg-motor-imagery/.venv`. Every claim
below was read out of the installed source on 2026-07-25, not recalled.

### 2.1 sklearn re-derives the partition from the permuted labels

`sklearn/model_selection/_validation.py`, `_permutation_test_score`, first line of its loop:

```python
for train, test in cv.split(X, y, **split_params):
```

and the caller passes `_shuffle(y, groups, random_state)` as `y`. `StratifiedKFold.split` derives
fold membership from `y`. Therefore **every permutation replicate is scored on a different
partition**, each one stratified on its own permuted labels, while the observed value is scored on
the partition stratified on the true labels.

Measured by index arithmetic alone, no EEG and no classifier, on subject 1's 21/24 label vector with
`StratifiedKFold(5, shuffle=True, random_state=42)` over 500 permutations:

- the re-stratified partition differs from the observed partition on **80.0% of trials on average**
  (range 57.8% to 95.6%). This is not a marginal perturbation of the partition, it is a different
  partition nearly every time.
- under re-stratification every test fold is forced to (4 hands, 5 feet) or (5 hands, 4 feet). Only
  those two balances ever occur.
- under a **fixed** partition the same permuted labels produce test folds spanning **(0,9) through
  (8,1)**, the full range.

That last pair of lines is the entire mechanism of arm (a) and it is a fact about the splitter, not a
result: the two nulls are not two estimates of one quantity, they are two different reference
distributions, because they evaluate on structurally different folds.

### 2.2 An iterable of splits freezes the partition, using public API only

`check_cv` wraps an iterable of `(train, test)` index arrays in `_CVIterableWrapper`, whose `split`
ignores `X`, `y` and `groups` and replays the stored list. Verified: passing a materialised
`list(skf.split(X, y_true))` as `cv` returns **identical test sets under a permuted y**, and is
reusable across calls because `__init__` does `self.cv = list(cv)`. The correction in arm (a) is
therefore a one-line change to a published script, not a reimplementation.

### 2.3 sklearn already implements block permutation, and the repo does not use it

`_shuffle(y, groups, random_state)` permutes **within each group** when `groups` is not None. So
`permutation_test_score(..., groups=subject_ids)` is a block permutation for free.
`cross_subject.py` does not call `permutation_test_score` at all. It calls
`rng.permutation(y_all)` once, globally, at line 135. The correction was available in the library the
script already imports.

### 2.4 The LOSO partition is ALREADY invariant to the labels, and this asymmetry matters

`LeaveOneGroupOut._iter_test_masks` reads only `groups`. Verified: LOSO test sets are identical under
a permuted `y`. **Arm (a)'s defect does not exist in arm (b).** The two objections are not the same
objection applied twice:

- **arm (a)** is a defect of the **partition**: the null marginalises over partitions the observed
  value conditions on.
- **arm (b)** is a defect of the **reference set**: the null draws label vectors the experiment could
  not have produced.

Both are the same underlying error, "the null must condition on whatever the observed statistic
conditions on", at two different levels. Saying they are one defect would be wrong, and saying they
are unrelated would miss the point.

---

## 3. THE EXCHANGEABLE UNIT, STATED EXPLICITLY

This is the sentence the corpus could not produce. Both arms get one.

### 3.1 Arm (a), within-subject, subject 1

> **The exchangeable unit is the trial, conditional on run, and the statistic is evaluated on a fixed
> partition.**

Under the null, within each of runs 6, 10 and 14 the 15 trial labels are exchangeable. Nothing is
exchangeable across runs, because run is a blocking factor: the runs are separate recordings with
their own electrode settling and drift. `ablate_channels.py:319` already computes the run index for
every epoch via `np.searchsorted`, so the blocking variable exists in the repo and has never been
used in a null.

Subject 1's labels by run, from A272: `FHHFHFFHHFFHHFF / HFFHFHHFFHHFFHF / FHFHHFHFHFFHFHF`, which
is **8 feet and 7 hands in each of the three runs**, summing to the published 24 feet / 21 hands.
Reference set sizes: within-run permutation gives C(15,7)^3 = 2.66e11 label vectors; i.i.d.
permutation gives C(45,21) = 3.77e12, a factor of 14.2 larger. The block null is the **smaller**
reference set and therefore the **weaker assumption**. It does not assume trials are exchangeable
across runs. Both are far larger than 10,000 draws, so sampling with replacement from the reference
set is not a concern.

The published null assumes i.i.d. exchangeability across all 45 trials AND re-derives the partition.
It is the strongest assumption of the four cells in Section 4.

### 3.2 Arm (b), cross-subject LOSO

> **The exchangeable unit is the trial, conditional on subject. The partition needs no correction
> because LOSO already conditions on subject.**

Under the null, within each subject's 45 trials the labels are exchangeable; nothing is exchangeable
across subjects. Reference set: the product over 20 subjects of C(45, n_hands_i), of order 1e252.
The global shuffle's reference set is C(900, n_hands_total), of order 1e269.

**What the global shuffle's null actually tests.** It tests the compound hypothesis:

> the labels are unrelated to the EEG **AND** the labels are exchangeable across subjects.

The second conjunct is **false by protocol**, independent of any decoding. Each subject's class
marginal is fixed by the experiment. From `sweep_results.csv`, subjects 1 to 20 do not share one
marginal: six of them (1, 2, 3, 14, 16, 17) have a majority-class rate of 0.5333 = 24/45 and the
other fourteen have 0.5111 = 23/45. Canon A268 says the marginal is "fixed by the protocol at a
near-balanced 21/24"; that is right for subject 1 and **wrong as a statement about the pooled set**,
and this pre-registration flags it as an entry needing correction. The heterogeneity is small, and it
is exactly the structure a global shuffle destroys.

**Why it is the wrong null for a LOSO design specifically.** In LOSO the estimand is per-subject
generalisation and every fold's test set is exactly one subject. The accuracy on that fold is read
against that subject's own majority-class rate, which in the observed data is a **fixed constant**.
Under a global shuffle it becomes a **random variable**: a subject with 45 trials can be dealt 30
feet and 15 hands, a draw the experiment could not have produced. The null therefore carries a
variance component that has nothing to do with decoding and everything to do with re-dealing the
class marginals. Rejecting it does not isolate "the decoder reads the imagery", because the rejection
can be driven by between-subject marginal structure alone.

Under block permutation each subject's marginal is preserved exactly, the subject-to-fold map is
untouched, and the only thing that varies is the within-subject label assignment, which is the thing
the null is actually about.

---

## 4. THE ARMS

Pipeline is the committed one and does not change: average reference over all 64 channels, 8 to 30 Hz
FIR, epochs -1.0 to 4.0 s, cropped to 1.0 to 2.0 s, `CSP(n_components=4, reg=None, log=True,
norm_trace=False)` then `LinearDiscriminantAnalysis()` inside one `Pipeline`, so **CSP refits inside
every training fold in every replicate**. Any leakage invalidates the result.
`error_score="raise"` on every CV call.

### 4.1 Arm A, within-subject: a 2x2 factorial, paired on the labels

Two factors, fully crossed. The pilot ran three of these four cells and never crossed them.

|  | partition RE-STRATIFIED each draw | partition FIXED at P0 |
|---|---|---|
| **i.i.d. shuffle** (exchangeable across all 45) | **C1** = the published null | **C2** = correction (a) alone |
| **within-run shuffle** (exchangeable within run) | **C3** = pilot cell (iv) | **C4** = FULLY CORRECTED |

P0 is `list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y_true))`, which is
the partition the published 91.1% was computed on. The observed statistic is scored on P0 in every
cell, so the observed value is the published 41/45 and does not vary by cell.

**Pairing, and this is the design's main contribution over the pilot.** One list of 10,000 i.i.d.
permuted label vectors is generated once from `default_rng(42)` and fed to **both** C1 and C2. A
second list of 10,000 within-run permuted vectors is generated once and fed to **both** C3 and C4.
Therefore:

- C1 against C2 is **paired**: same labels, only the partition rule differs. This isolates the
  correction.
- C3 against C4 is **paired** in the same way.
- C1 against C3, and C2 against C4, are **unpaired by construction**, because they draw from
  different reference sets. They are compared on distributions only, never replicate by replicate.

The script asserts that the label lists handed to C1 and C2 are element-wise identical, and likewise
for C3 and C4. A paired design that is not actually paired is worse than an unpaired one.

**Subjects.** Three, chosen by a rule fixed before any null runs:

- **Subject 1**, the headline. Non-negotiable, it is the number under dispute.
- **The median subject of the 20 LOSO subjects by published within-subject accuracy in
  `sweep_results.csv`.** Sorting subjects 1 to 20 gives a median of 0.6333 between the 10th and 11th
  values, and **subjects 17 (0.6222 = 28/45) and 19 (0.6444 = 29/45) are exactly equidistant from
  it**. Rather than invent a tie-break, both are included. This removes the last discretionary choice
  from subject selection.

**Why the median subjects are here, stated in advance so it cannot be read as cherry-picking later.**
Subject 1's effect is roughly 4.5 null sd out. Under a normal approximation to the pilot null, the
expected number of exceedances in 10,000 draws is **0.028**, so the p is at its floor with
probability about 97%, in every cell, no matter which null is used. **An arm whose answer is fixed by
the effect size cannot detect a design error.** The same approximation puts subjects 17 and 19 at
about **p = 0.080 and p = 0.048**, interior to the range and straddling 0.05. That is the only place
in this design where a corrected null can flip a verdict, which is precisely why it must be
pre-registered rather than added after subject 1 comes back uninformative. These are planning
estimates used to choose N and nothing else; A271 records that the null is visibly not normal, so
they are not predictions and Section 6 does not score them.

**N = 10,000 draws per cell**, floor p = 1/10001 = 9.999e-05. Measured cost: one 5-fold CV is 128 ms
serial, so 4 cells x 3 subjects x 10,000 is about 4.3 h serial and well under an hour on 16 cores.

### 4.2 Arm B, cross-subject: global against block

Subjects 1 to 20, runs 6/10/14, 45 trials each, 900 pooled trials, `LeaveOneGroupOut` over subject,
20 folds of 45. Identical to `cross_subject.py`.

- **G** = global permutation of all 900 labels. What `cross_subject.py:135` does, but 2,000 times
  instead of once.
- **B** = within-subject block permutation. Correction (b).

Unpaired by construction and reported as such. **N = 2,000 draws per cell**, floor
p = 1/2001 = 4.998e-04. Measured cost: one 855-trial fit plus predict is 0.33 s at the true array
shape, so one 20-fold LOSO pass is about 6.6 s and 2 cells x 2,000 draws is about 7.3 h serial, under
an hour on 16 cores.

### 4.3 Arm A-repro: reproduce the pilot exactly, at the pilot's own settings

Before anything else runs, the harness reproduces A269: 300 draws, `default_rng(42)` consumed
sequentially in the pilot's cell order, subject 1, cells (i) through (iv). See Section 7.

---

## 5. STATISTICS AND ASSERTS

### 5.1 The p-value, and the honest statement of what it can resolve

Reported as `p = (C + 1) / (N + 1)` where `C = #{null >= observed}`, matching sklearn exactly, with
`>=` and not `>`. When `C = 0` it is printed as a **bound**, `p <= 9.999e-05`, never as a measured
value, for the same reason `p <= 0.001` is a bound in the published work.

**Registered in advance: for subject 1 and for the LOSO arm, the p-value is expected to be at its
floor in every cell, and this is a property of the effect size, not evidence about the design.** The
exceedance count `C` is reported alongside every p precisely so that "both at the floor" is
distinguishable from "both actually equal".

### 5.2 Primary statistics for arm A

Per cell: mean, sd, min, max, and the 50th, 90th, 95th, 99th and 99.9th percentiles of the null;
exceedance count `C`; p.

**Paired difference, the primary statistic for the correction:** for the 10,000 shared i.i.d. label
vectors, `d_j = acc_C2(perm_j) - acc_C1(perm_j)`. Report mean(d), sd(d), the Monte Carlo standard
error of mean(d), the fraction of replicates with `d != 0`, and the fraction with `d > 0`. Same for
`acc_C4 - acc_C3`. This is the statistic that can move even when both p-values are pinned at the
floor, and it is the statistic the pilot could not compute.

**Tail mass at fixed lattice points**, so tails are compared at a common location rather than by the
maximum, which A269 already withdrew as too noisy: count of replicates at or above 32/45 (71.1%),
34/45 (75.6%) and 36/45 (80.0%).

**Standardised distance** (observed minus null mean) / null sd, per cell, reported as descriptive
only. The null is not normal and no p is derived from it.

### 5.3 Primary statistics for arm B

Per cell: mean, sd, min, max, 50th, 90th, 95th, 99th, 99.5th percentiles; exceedance count; p.

**The sd ratio** sd(G) / sd(B), with a Monte Carlo standard error. At the pilot's values, 2.0 / 1.2 =
1.7, and the MC standard error of each sd at N=2,000 is about 0.032 points.

**The guard replacement, which is the arm's deliverable.** `SHUFFLE_MAX = 0.60` is a round number
with no derivation. Report the **99th and 99.5th percentiles of the block null on the k/900 lattice**
as its replacement, and report where 0.60 sits in each null in both sd units and percentile. This
turns a threshold nobody derived into a measured quantile of the correct reference distribution.

**The mechanism exhibit, which is descriptive and not a test:** the standard deviation across
replicates of each held-out subject's own majority-class rate, under G and under B. Under B it must
be exactly 0.0, because block permutation preserves marginals; under G it will not be. This is the
clearest possible demonstration of what the global shuffle destroys, and it costs nothing.

### 5.4 Asserts that must be in the code

1. **Lattice, arm A.** Every null replicate score is an integer multiple of 1/45 to within 1e-9. Five
   equal folds of 9 make the fold-mean equal the pooled count over 45. This holds in both partition
   rules, verified: fold sizes are {9} under re-stratification as well.
2. **Lattice, arm B.** Every LOSO null replicate score is an integer multiple of 1/900 to within
   1e-9. Twenty equal folds of 45.
3. **Observed value.** Subject 1's observed accuracy on P0 is exactly 41/45 = 91.1%. If it is not,
   P0 is not the published partition and the whole comparison is against the wrong baseline.
4. **Pairing.** The label array handed to C1 at replicate j is element-wise identical to the one
   handed to C2 at replicate j. Same for C3 and C4.
5. **Partition really is fixed.** In C2 and C4 the realised test-index sets are identical to P0's on
   every replicate. Asserted, not assumed, because the entire correction is this one property.
6. **Partition really does move.** In C1 and C3 the realised partition differs from P0 on at least
   one replicate. A "correction" that changed nothing because both cells were secretly fixed would
   otherwise look like a clean null result.
7. **Block shuffle really blocks.** In C3 and C4, each run's class counts are preserved exactly on
   every replicate (8 feet / 7 hands per run for subject 1). In arm B, each subject's class counts
   are preserved exactly on every replicate.
8. **Global shuffle really does not block.** In arm B cell G, at least one replicate changes at least
   one subject's class marginal. Same reasoning as assert 6.
9. **Null centring.** Each cell's null mean is within 0.05 of 0.50. Wider than that means the null is
   mis-specified and no p from it means what it appears to.
10. **Marginal preservation, global.** Even the global shuffle preserves the **pooled** class counts,
    since it is a permutation. Assert it, so that a bug that resamples rather than permutes is caught.

---

## 6. PRE-REGISTERED OUTCOMES

**Definitions of "material", fixed now.**

- A **p-value** change is material if it crosses 0.05, or if both values are off the floor and they
  differ by a factor of 2 or more.
- A **paired null difference** mean(d) is material if |mean(d)| exceeds 3 Monte Carlo standard errors
  AND exceeds 1/45 = 2.22 points, that is, one whole trial. Statistical resolution alone is not
  enough; at N=10,000 the MC standard error of mean(d) will be small enough to certify differences
  far below one trial, and a sub-trial difference in a 45-trial null is not a difference anyone
  should act on.
- An **sd ratio** is material if it differs from 1 by more than 3 of its Monte Carlo standard errors.

### 6.1 Arm A, the paired partition correction, C2 against C1

| result | meaning, fixed in advance | good news? |
|---|---|---|
| mean(d) materially **negative**: fixed-partition null sits BELOW the re-stratified null | The published null is **too high**, so the published p is **conservative**. sklearn's re-stratification made the headline's significance harder to attain, not easier. The design error runs against the project's own interest. Report as "the error was in the safe direction, which we did not know until we measured it", never as vindication of the design. Confirms A269's direction at higher resolution with a paired design. | good for the headline, bad for the design |
| mean(d) materially **positive**: fixed-partition null sits ABOVE the re-stratified null | The published null is **too low** and the published p is **anti-conservative**. The headline's p must be re-quoted from the corrected null in every document that carries it. This contradicts the pilot and the pilot's direction is then the thing that needs explaining. | **bad, and it gets published** |
| mean(d) not material, or material by MC error but under one trial | The partition rule does not change the null at a magnitude anyone should act on for this dataset. The objection is correct in principle and empirically null here. Must NOT be written as "the objection was wrong": it was a real defect that turned out not to bite at n=45. | neutral |
| both p-values at the floor, C = 0 in both cells | **The near-certain outcome for subject 1, registered as such.** The correction does not change the verdict on the headline. This is NOT evidence the original null was correctly designed; it is evidence the effect is large enough that the design error cannot change the answer. That distinction is the whole finding for this cell. | neutral |
| p-values differ off the floor | Only possible if the effect is far weaker than every prior. Would mean the published headline is far less secure than 41/45 suggests, and would take priority over everything else in this document. | bad |
| null sd differs materially between C1 and C2 while mean(d) does not | The correction changes the null's spread without changing its centre. Consequences flow to A271's variance-inflation factor and the n_eff-corrected Wilson interval, which are computed from the null sd. Those numbers must be recomputed from the corrected null. | depends |

### 6.2 Arm A, the block correction and the fully corrected cell

| result | meaning, fixed in advance | good news? |
|---|---|---|
| C3 and C4 (within-run) statistically indistinguishable from C1 and C2 (i.i.d.) | Run blocking does not matter for subject 1: the trials behave as exchangeable across runs. This **supports** the published i.i.d. null as adequate here, and it is the first evidence in this project that it is adequate. Pair it with A272, which bounds monotone drift at one trial by an independent route. | good |
| within-run null materially **higher** than i.i.d. | Within-run label structure was carrying part of the apparent effect. The honest null is the block one, the headline's p must be quoted from C4, and the leave-one-run-out result (93.3%) needs re-reading in that light. | **bad, and it gets published** |
| within-run null materially **lower** than i.i.d. | The i.i.d. null was conservative for a second, independent reason. Same reporting discipline as the first row of 6.1: safe direction, not vindication. | good for the headline |
| C4 (fully corrected) differs materially from C1 (published) | **This is the headline result of arm A**, whichever direction it goes, because C4 is the null with the weakest assumptions and the correct conditioning. Its p is the one that should be published going forward. | depends |
| C4 indistinguishable from C1 | The published null, despite being wrong on both counts, delivers the same answer as the null with the correct exchangeable unit and the correct conditioning. Report as "the two errors do not bite at n=45 on this subject". Explicitly do not generalise it to other subjects, other n, or other designs. | neutral |

### 6.3 Arm A, subjects 17 and 19, where the p can actually move

| result | meaning, fixed in advance | good news? |
|---|---|---|
| corrected p crosses 0.05 **upward** for either subject | A design error of exactly this kind can flip a per-subject verdict in this dataset. That is a **demonstrated** consequence, not a hypothetical one, and it is the strongest possible argument that the correction is required rather than pedantic. It also means `sweep_results.csv` and anything built on per-subject significance is affected. | bad for the corpus, **strong finding, publish** |
| corrected p crosses 0.05 **downward** | Same demonstration with the opposite sign: the published null was costing real detections. Equally publishable, and it must not be sold as a bonus. | depends |
| corrected and published p both interior and not materially different | The correction does not move a verdict even where the p is free to move. This is the **strongest available evidence that the objection is principled but empirically inert on this data**, and it is a stronger form of that claim than subject 1 can supply. | good, strengthens the original |
| the two median subjects disagree with each other | n=2 subjects, and per-subject nulls at 45 trials are noisy. Report both, draw no conclusion from the disagreement, and do not pick the agreeable one. | neutral |
| recomputed observed accuracy for 17 or 19 does not match `sweep_results.csv` | Not an outcome, a defect. Halt the arm and reconcile the pipeline before reporting anything from it. See 7.6. | broken |

### 6.4 Arm B, cross-subject

| result | meaning, fixed in advance | good news? |
|---|---|---|
| sd(B) materially **smaller** than sd(G) | Confirms A270 at 10x the draws. The global shuffle inflates the null by re-dealing each held-out subject's class marginal, which is a variance source unrelated to decoding. The published guard is loose **for a nameable reason**, and the replacement threshold in 5.3 is the deliverable. | good for rigour, bad for the existing guard |
| sd(B) materially **larger** than sd(G) | The stated mechanism is wrong. The global shuffle would then have been **anti**-conservative and the 59.4% cross-subject result is weaker than published. Section 3.2's account gets withdrawn and rewritten from the data. | **bad, and it gets published** |
| sd(B) and sd(G) not materially different | Subject blocking does not change the null's spread on this data. The objection stands as a design principle, and the practical consequence is nil. The guard threshold still gets replaced, because 0.60 was never derived from anything either way. | neutral |
| observed 59.4% still far outside the block null, p at floor | The cross-subject finding survives the correct null. Expected, and it must be reported with the floor convention and not as p = 0.0005. | good |
| observed 59.4% falls **inside** the block null, p > 0.05 | The cross-subject result does not survive a correctly blocked null. This is the most damaging outcome available in this document and it gets published in full, at the top, ahead of everything else. `cross_subject.py`'s own docstring already licenses it: a cross-subject score at chance is a legitimate finding about transfer. | **very bad, publish first** |
| per-subject majority-rate sd under B is not exactly 0.0 | Not an outcome, a defect. The block shuffle is not blocking. Halt. | broken |
| the replacement threshold lands above 0.60 | The existing guard would be **tighter** than the correct null justifies, meaning it could fire on a clean run. Contradicts A41's reading and A41 gets corrected. | depends |

### 6.5 The outcome that applies to every cell

If any cell's null mean falls outside 0.45 to 0.55, or any replicate score lands off the k/45 or
k/900 lattice, **nothing in Sections 6.1 to 6.4 may be reported from that run**. See Section 7.

---

## 7. WHAT WOULD FALSIFY THE ANALYSIS RATHER THAN THE HYPOTHESIS

Each of these says the harness is broken, not that the corrected null is interesting.

1. **The pilot does not reproduce.** Arm A-repro at 300 draws with `default_rng(42)` consumed in the
   pilot's order must return A269's numbers: (i) 50.7 / 8.9 / 86.7, (ii) 47.7 / 8.4 / 71.1, (iii)
   49.0 / 8.6 / 73.3, (iv) 50.2 / 8.2 / 68.9, all with p = 0.0033. A mismatch is diagnosed before
   anything else runs, and the two causes are distinguishable: **all four cells shifting together**
   points at version drift in mne or sklearn between the pilot run and this one, which is
   reportable and not fatal; **one cell shifting** points at a bug in that cell, which is fatal.
   Nothing from Section 6 may be reported until this is resolved either way.
2. **My re-stratified cell is not what sklearn does.** C1 at N=1,000 must agree with a direct call to
   `permutation_test_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
   n_permutations=1000, random_state=42)` on null mean and null sd to within 3 Monte Carlo standard
   errors. If it does not, then C1 is not the published null, and the entire framing "this is what
   sklearn does" is unfounded. This is the single most important check in the document, because every
   claim of a **difference** is measured against C1.
3. **Any replicate score off the lattice.** k/45 for arm A, k/900 for arm B. Off-lattice means unequal
   folds or a scorer that is not accuracy, and the fold-mean is then not the accuracy at all. This is
   the check that caught two fabricated numbers in an earlier README.
4. **The observed value is not 41/45 for subject 1 on P0.** Then P0 is not the published partition,
   and every "corrected against published" comparison in the document is against the wrong baseline.
5. **A fixed cell whose partition is not fixed, or a re-stratified cell whose partition never moves.**
   Asserts 5 and 6. Either one makes the paired difference meaningless while leaving it looking clean.
6. **Subjects 17 or 19 do not reproduce `sweep_results.csv`** at 28/45 and 29/45 on the published
   pipeline. Then the median-selection rule selected on numbers this pipeline does not produce, and
   the subject choice has to be redone from recomputed values before any null is run on them.
7. **A block shuffle that changes a marginal, or a global shuffle that changes the pooled marginal.**
   Asserts 7, 8 and 10. The first means the blocking is broken; the second means the code is
   resampling rather than permuting.
8. **Null mean outside 0.45 to 0.55.** Mis-specified null, as `evaluate_honestly.py:220` already
   asserts for the published one.

**Registered separately, as a secondary mechanism probe whose failure does not touch any primary
result.** A269 records a failed prediction of a heavier right tail for the fixed-partition null. Here
is a mechanism that predicts the direction actually observed, registered before the run so it can
fail on the record: under a fixed partition, a permuted-label test fold can be strongly unbalanced,
and the complementary training folds are then unbalanced the **other** way, so the classifier's
induced prior is anti-correlated with the test fold's majority class and the fold scores **worse**
than a balanced fold would. Re-stratification forbids those folds entirely, which is why its null
sits higher. **The probe:** in C2, correlate per-fold test-set imbalance against per-fold accuracy.
The mechanism predicts a **negative** correlation. If the correlation is zero or positive, the
mechanism is wrong, it gets withdrawn in the write-up, and **mean(d) is unaffected**, because the
measurement does not depend on the explanation. This is the separation the round-one failure mode
demands.

---

## 8. DELIVERABLE AND PROVENANCE

- Script: `eeg-motor-imagery/permutation_design.py`, house style. Module docstring saying why the
  script exists and what it guards against; printed section headers; every accuracy printed with its
  baseline and its k/n count; explicit statements of what each number does not show; asserts on the
  lattice.
- Stdout captured to `neuro-canon/runs/`, so `check_claims.py` and `check_provenance.py` can match
  against it. Every number in the write-up must be **printed by the script**, never interpolated from
  a literal. `evaluate_honestly.py` lines 236 to 248 record what happens when a hardcoded number ends
  up in stdout and then backs itself in a provenance check.
- Chance is the **majority-class rate** everywhere: 53.3% (24/45) for subject 1, 24/45 or 23/45 per
  subject for 17 and 19, and the pooled rate printed by the script for arm B. Never 50%.
- Canon entries to update on completion: **A41** (the guard threshold and its replacement), **A268**
  (the "21/24 fixed by protocol" claim, which is wrong for the pooled set), **A269** and **A270**
  (promoted from `uncommitted` to a committed script and stdout, or corrected).
- No em dashes anywhere in the script, its stdout, or the write-up.

---

## 9. REGISTERED RISKS

1. **The headline arm is uninformative by construction and I knew it before running.** Subject 1's
   p cannot move. Presenting "p unchanged" as a clean confirmation would be reading a resolution
   floor as a measurement, the exact error `decode_csp.py:132-136` exists to prevent. The finding for
   subject 1 is the paired null difference and the null's shape, and the write-up must lead with that.
2. **Confirming a pilot I have already read is weak evidence.** I knew A269 and A270's numbers before
   designing this. The pairing, the fourth cell, the resolution and the interior-p subjects are real
   additions, but no part of this run is blind, and Section 0 has to travel with the result.
3. **n=45 per subject, 20 subjects.** Every null here is small-sample. A271 already shows the
   within-subject null is not binomial, with a variance-inflation factor near 1.4. Nothing in this
   document repairs that; it measures the null's shape more carefully and inherits the same n.
4. **Three subjects is not a survey.** Subjects 17 and 19 were selected by a fixed rule from an
   existing CSV, but they are still 2 of 109, chosen for statistical position rather than at random.
   No claim about "how often the correction matters" can be made from them. The claim available is
   existence: whether it **can** move a verdict.
5. **Runs 6, 10 and 14 are one session.** Within-run blocking addresses drift inside a run. A
   session-level trend across all three runs survives every null in this document, exactly as it
   survives the leave-one-run-out ablation. `ablate_channels.py:255-259` says so and this does not
   improve on it.
6. **The LOSO arm reuses `cross_subject.py`'s 20-subject budget**, which is a documented budget and
   not the full 109. Nothing here changes that, and the block-permutation conclusion is scoped to
   those 20.
7. **Two of the six comparisons are unpaired by construction** (C1 against C3, C2 against C4, and
   both arm B cells). Their differences carry Monte Carlo noise that the paired comparisons do not.
   Reporting them at the same confidence as the paired ones would repeat the pilot's error at a
   larger N.
8. **A corrected p that is unchanged strengthens the original result, and that is a pleasant
   conclusion reached by a route I designed.** The registered guard against motivated reading is that
   the unflattering cells in 6.1, 6.2, 6.3 and 6.4 are written down here, before the run, in the same
   detail as the flattering ones.

---

## RESULTS

Run 2026-07-25, **re-run with corrections 2026-07-26**. Script:
`/Users/yaz/Documents/Projects/eeg-motor-imagery/permutation_design.py`, at the registered N
(arm A 10,000 per cell, arm B 2,000 per cell, repro 300, gate 2 1,000). Nothing above this
line was edited.

**READ THIS FIRST.** The run's original reading has been substantially withdrawn. An adversarial
pass showed that **two of the four arm A cells are not exact permutation tests**, and they are
the two the run called "corrected". The measured values did not change. The conclusions did.

### The gates

- **Gate 1 (pilot reproduction)** passes: arm A-repro at 300 draws with `default_rng(42)`
  consumed in A269's order returns the pilot's four cells.
- **Gate 2 (is C1 what sklearn does)** passes: C1 at N = 1,000 agrees with a direct
  `permutation_test_score` call on null mean and null sd within 3 MC standard errors, and the
  observed value matches at 41/45.
- **Falsification 4 and 6** pass: subject 1 is exactly 41/45 on P0; subjects 17 and 19
  reproduce `sweep_results.csv` at 28/45 and 29/45.
- **Asserts 5 and 6** pass: the fixed cells replayed P0 on 10,000 of 10,000 replicates, and
  the re-stratified cells moved off it.
- **Assert 9 (null centring, 0.45 to 0.55)** fires on the fixed cells. See below.

### The cells, as measured

| subject | observed on P0 | C1 (published) | C2 | C3 | C4 |
|---|---|---|---|---|---|
| 1 | 41/45 | p <= 9.999e-05 | p <= 9.999e-05 | p <= 9.999e-05 | p <= 9.999e-05 |
| 17 | 28/45 | p = 0.11529 | p = 0.079592 | p = 0.052795 | p = 0.025197 |
| 19 | 29/45 | p = 0.055494 | p = 0.037396 | p = 0.044496 | p = 0.030197 |

Null means: subject 17 C1 51.11%, C2 48.26%, C3 47.64%, **C4 43.80%**.

### The correction: C2 and C4 are not exact tests, and C1 and C3 are

`P0 = StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_TRUE)` is **a function of the
labels being permuted**. Freezing it makes the statistic a function of (X, y', y_true), and the
observed vector is then the unique point in the reference set whose fold margins are balanced
against its own labels. Every replicate is scored on those same folds under labels that make the
margins arbitrary, and LDA takes its priors from the training fold. Exchangeability is broken.

Under re-stratification the statistic is ONE function, S(y) = CV(X, y; SKF(y)), applied
identically to the observed vector and to every replicate. **C1 and C3 are exact.** sklearn's
published null was already correct for its own reference set.

Section 2B's licensing sentence, *"A permutation p is exact when the observed statistic and the
null replicates are computed the SAME way. The observed value is scored on P0; C2 and C4 score
every replicate on P0"*, is **withdrawn**. Computing them the same way is necessary, not
sufficient; the conditioning quantity must not depend on the labels.

**Section 2B's OTHER argument is NOT withdrawn.** The 0.45 to 0.55 band really does not transfer
to a fixed partition, and the new C5/C6 cells confirm it independently: those cells are exact
tests and they also centre below the band (subject 17: C5 48.18%, C6 43.65%). Assert 9 was the
wrong assert. That was a correct diagnosis of a different defect, and fixing it did not fix this
one.

#### Measured with zero information (no EEG at all)

A majority-class predictor on an all-zero feature matrix, so H0 is exactly true by construction
and every rejection is false. 200 H0 label vectors x 199 inner permutations per rule per subject:

| rule | S1 P(p<=0.05) | S17 | S19 |
|---|---|---|---|
| RE (= C1) | 0.0000 | 0.0000 | 0.0000 |
| **FX (= C2)** | **0.6550** | 0.0000 | 0.0500 (0.8350 at alpha 0.10) |
| RE_blk (= C3) | 0.0000 | 0.0000 | 0.0000 |
| **FX_blk (= C4)** | **0.6600** | 0.0000 | 0.0550 (0.8450 at alpha 0.10) |
| KF_free (label-free fixed partition) | 0.0250 | 0.0000 | 0.0150 |
| WITHIN (within-fold restricted) | 0.0000 | 0.0000 | 0.0000 |

Counted from the table: **4 of 6 fixed-at-P0 cells are anti-conservative** (by more than 2 MC
standard errors at alpha 0.05 or 0.10); **0 of 12** re-stratified or label-independent cells are.
The defect does not bite equally at every class marginal, and the analyst does not get to know
which marginal they have before choosing a rule.

#### Measured on the real pipeline, from this run's own replicates

Under H0 the observed value is distributed like a self-stratified replicate, so each rule is
judged against the H0 it assumes, on disjoint halves of the 10,000:

| subject | C1 (exact) | C2 (withdrawn) | C3 (exact) | C4 (withdrawn) |
|---|---|---|---|---|
| 1 | .0444 (>= 30/45) | .0752 (>= 29/45) | .0332 (>= 30/45) | .0594 (>= 29/45) |
| 17 | .0386 (>= 30/45) | .0386 (>= 30/45) | .0556 (>= 28/45) | **.0876 (>= 27/45)** |
| 19 | .0298 (>= 30/45) | .0528 (>= 29/45) | .0438 (>= 29/45) | .0438 (>= 29/45) |

Subject 17's C4 runs at 1.8x nominal, and it is the cell that produced the reported verdict flip.
On every subject each withdrawn cell has size at least as large as its exact partner and a
rejection threshold no higher, never the reverse.

### What is withdrawn

Every number derived from C2 and C4: their p-values, percentiles, tail counts, the C4-based
`n_eff` and the `n_eff`-corrected Wilson interval, the C2-C1 and C4-C3 paired `mean(d)` rows read
as corrections, the C4-minus-C1 "headline result of arm A", and the recommendation that C4's p is
the one to publish. They are still printed, so what was claimed stays visible, but nothing
concludes from them.

Also withdrawn: **"the published null sits TOO HIGH, so the published p-values were
CONSERVATIVE"**. That presupposes the fixed cell is a reference distribution for the same
statistic. The displacement measures the cost of breaking exchangeability, not a mis-centred
published null. C1 needs no re-centring.

Also withdrawn: **"THE CORRECTION CHANGES VERDICTS. Both median subjects go from non-significant
to significant under the fully corrected null."**

### The exact version of the fixed-partition idea (C5, C6, added post-registration)

The partition is built by `KFold(5, shuffle=True, random_state=42).split(X)`, **without the
labels**, so it is ancillary and freezing it is exact. N = 2,000.

| subject | observed on PF | C5 (i.i.d.) | C6 (within-run) |
|---|---|---|---|
| 1 | 42/45 | p <= 0.0005 | p <= 0.0005 |
| 17 | 25/45 | p = 0.25487 | p = 0.11444 |
| 19 | 21/45 | p = 0.56522 | p = 0.48576 |

**The observed value moves because the partition moved** (subject 19 falls from 29/45 to 21/45).
C5 and C6 are exact tests of a **different statistic**, accuracy on an unstratified partition, so
their p is not interchangeable with C1's and is not a correction to the published number.

### Re-scored against section 6, using only the exact cells

- **6.1 (C2 against C1): the row does not apply.** C2 is not a valid reference distribution, so
  none of the three directional readings in that table can be assigned. The registered table has
  no cell for "the comparison arm is invalid", and that is now a known gap in it.
- **6.2 (block correction):** run blocking on its own **is** a valid correction, because C3
  re-derives the folds from each permuted vector. Subject 17 moves 0.11529 to 0.052795, a factor
  of 2.18, which is material by the registered factor-of-2 rule, and **does not cross 0.05**.
  Subject 19 moves 0.055494 to 0.044496 and **does cross**, by 0.011 of p. Subject 1 stays at the
  floor throughout. The "C4 is the headline result" row is withdrawn with C4.
- **6.3 (subjects 17 and 19):** the outcome that fires is **"the two median subjects disagree
  with each other"**, registered as **[neutral]**. It is NOT the "[depends] the corrected p
  crosses 0.05 downward" row, which was reported before and required both subjects. Under C3,
  subject 19 crosses and subject 17 does not. Under C5 and C6 neither crosses.
- **6.4 (arm B): unaffected and unchanged.** `LeaveOneGroupOut` reads only `groups`, so arm A's
  defect does not exist there. sd(B) is materially smaller than sd(G), ratio 1.573, confirming
  A270 at 10x the draws; the observed 59.4% (535/900) stays far outside the block null with
  p <= 0.0005 under both rules; and the replacement for the underived `SHUFFLE_MAX = 0.60` is
  the block null's 99th percentile at 53.1% (478/900), which lands **below** 0.60.
- **6.5 (the halt):** assert 9 fired on the fixed cells and section 2B overrode it. **That
  override is upheld, for the reason 2B gave and now with independent support**: C5 and C6 are
  exact and also centre below 0.45, so the band genuinely does not transfer to a fixed partition.
  Reinstating the halt on centring grounds would be the wrong remedy for the right problem. The
  correct remedy is the one applied here: C2 and C4 are withdrawn on **exchangeability** grounds,
  which is a separate defect that the override never addressed and that assert 9 was never
  testing.

### The corrected reading of arm A

**Run blocking alone (C3), which is exact, moves subject 17's p by a factor of 2.18 without
crossing 0.05, and moves subject 19's across the line by 0.011 of p. The two median subjects
therefore DISAGREE, which is the pre-registered [neutral] outcome. Subject 1 sits at the
resolution floor in every cell, which the pre-registration predicted and which cannot detect a
design error either way. The demonstrated existence claim is ONE subject, not two, and it is
delivered by the block correction alone, not by the fixed-partition step.**

Stated with the caveats the pre-registration requires: subjects 17 and 19 were selected precisely
because their p was expected to straddle 0.05 (section 4.1); no multiplicity adjustment is applied
across 3 subjects by 4 cells; and subject 19's crossing is 0.0445 against a 0.05 line, which is
1.1 percentage points of p and about 2.7 exceedance-count Monte Carlo standard errors inside.

### What this run refutes about its own pre-registration

Section 6's outcome tables assume each cell is a valid test of its stated null and provide no
outcome for "this cell is not a test". Section 5.4's ten asserts check the lattice, the blocking,
the partition replay and the null centring, and **none of them checks exchangeability**, which is
the property that decides whether a permutation p means anything. Assert 9 was reaching for that
property through a proxy (null centring) that is neither necessary nor sufficient for it: C5 and
C6 are exact and fail the band, C2 and C4 are not exact and the band was the only thing that ever
flagged them. A future pre-registration in this family needs a zero-information type-I control as
a registered gate, not as a post-hoc repair.

