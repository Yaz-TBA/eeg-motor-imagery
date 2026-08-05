# PRE-REGISTRATION: A STRUCTURAL AND EMPIRICAL VALIDITY GATE FOR THE WITHIN-SUBJECT PERMUTATION NULL

**Written 2026-07-30. Committed BEFORE the run it governs. Nothing above the RESULTS line may be
edited after the script executes. Amendments append below that line under Section 11 and are never
made in place.**

This replaces the null-centring gate of `neuro-canon/measurements/prereg-block-permutation.md`
(assert 9 at :377-378, falsifier 8 at :477-478, consequence clause at :442-443, *"nothing in Sections
6.1 to 6.4 may be reported from that run"*). That gate fired, `permutation_design.py:705` ("Section
2B") overrode it inside the analysis script, and a twelve-agent adjudication
(`eeg-motor-imagery/OVERRIDE-RULING-2026-07-30.md`) has since established that the gate was
**unsatisfiable by any valid design on this data**, that half of the override's reasoning was
**false**, and that the other half was **true**.

The old document named its own replacement at :716-720:

> none of them checks exchangeability, which is the property that decides whether a permutation p
> means anything ... A future pre-registration in this family needs a zero-information type-I control
> as a registered gate, not as a post-hoc repair.

This is that document, with one correction to that instruction. **A zero-information type-I control
is not sufficient either, and Section 0.4 shows why with numbers measured today.** The primary gate
here is a *structural* test of exchangeability that runs in microseconds and cannot be argued with.
The type-I control is the backstop, and this document is largely about the honest limits of that
backstop.

---

## 0. THE DISCLOSURE THAT HAS TO COME FIRST

### 0.1 This is not blind

Before writing a line of this document I read `prereg-block-permutation.md` in full including its
RESULTS section, `permutation_design.py` Sections 2B and 2C, the cached stdout at
`.provenance_cache/permutation_design.py.txt`, and the 2026-07-30 adjudication. I know which rules
the completed run found anti-conservative, in which direction, and roughly by how much. I then ran
the planning computations in 0.3 and 0.4 myself. **A gate designed after seeing the defect it must
catch will catch it. That is a construction, not a prediction**, and Sections 8 and 16 are written so
the difference stays visible.

I am the analyst, the author of the code under test, and the owner of the result. There is no
blinding and no second party.

### 0.2 What is already known, so nothing below can be presented as a discovery

**(a) The repo's zero-information type-I control** (`permutation_design.py` Section 2C, added
2026-07-26 post-registration; 200 H0 label vectors by 199 inner permutations; majority-class dummy on
`np.zeros((45,1))`, so H0 is true by construction and every rejection is false). `P(p <= 0.05)`:

| rule | S1 (21/24) | S17 (24/21) | S19 (22/23) |
|---|---|---|---|
| RE, re-stratified i.i.d. (= C1, the published rule) | 0.0000 | 0.0000 | 0.0000 |
| **FX, fixed at `P0 = SKF(y_obs)` (= C2)** | **0.6550** | 0.0000 | 0.0500 (0.8350 at alpha 0.10) |
| RE_blk, re-stratified within-run (= C3) | 0.0000 | 0.0000 | 0.0000 |
| **FX_blk (= C4)** | **0.6600** | 0.0000 | 0.0550 (0.8450 at alpha 0.10) |
| KF_free, label-free `KFold` partition (= C5) | 0.0250 | 0.0000 | 0.0150 |
| WITHIN, permute within folds of `P0` | 0.0000 | 0.0000 | 0.0000 |

The adjudication independently reproduced S1's FX cell at **0.675** against the repo's 0.6550, about
0.6 Monte Carlo standard errors apart at 200 draws.

**(b) The zero-information null centres.** A label-free `KFold(5, shuffle=True, random_state=42)`
partition reads only the shape of `X`, is therefore ancillary, and freezing it yields a provably
exact permutation test. Its zero-information null centres at **0.4054** (S1), **0.4728** (S17),
**0.3761** (S19). Two of three are outside the old 0.45 to 0.55 band. The **re-stratified** dummy
sits at **0.4444** at S19's marginal, also outside.

**(c) The tie-break.** A large share of that displacement is an artifact of sklearn's argmax rule,
which sends ties to the lower class label (`permutation_design.py:891`,
`pred = 2 if n2 >= n3 else 3`). At the same 24/45 majority marginal, opposite class codings give
**0.4056** and **0.4725**; under a **random** tie-break both collapse to **0.4385**.

**(d) The withdrawal.** `permutation_design.py:705` gave two arguments for overriding the centring
gate. Its **exactness** argument was false and was withdrawn 2026-07-26. Its **centring** argument,
that the 0.45 to 0.55 band does not transfer to a fixed partition, is **true** and is not withdrawn.

**(e) Observed values, carried forward unchanged and not results of this run.** On `P0`: S1 41/45,
S17 28/45, S19 29/45. On the label-free partition `PF`: S1 42/45, S17 25/45, S19 21/45. Cross-subject
LOSO 535/900.

### 0.3 What I measured today, before writing the gate, at the same three marginals

House rule inherited from the previous document: **disclose first, argue second.** Everything in 0.3
and 0.4 is a **planning input**. It is not a result, Section 9 does not score it, and Section 13
(F14) registers what happens if the run fails to reproduce it. All of it is pure label arithmetic and
synthetic features: **no EEG, no MNE, no repo data**, `eeg-motor-imagery/.venv`, scikit-learn 1.9.0,
numpy 2.5.1.

**(i) Reproduction of the known centring numbers.** 20,000 draws, `default_rng(0)`, sklearn
tie-break, majority-class dummy on `np.zeros((45,1))`:

| subject | marginal | label-free `PF` mean | `PF` sd | re-stratified mean | re-stratified sd | distinct values |
|---|---|---|---|---|---|---|
| 1 | 21/24 | 0.4055 | 0.0583 | 0.5333 | **0.0000** | **1** |
| 17 | 24/21 | 0.4735 | 0.0693 | 0.5333 | **0.0000** | **1** |
| 19 | 22/23 | 0.3757 | 0.0427 | 0.4444 | **0.0000** | **1** |

The `PF` column reproduces 0.4054 / 0.4728 / 0.3761 to within Monte Carlo error. The last three
columns are 0.4.

**(ii) Tie-break sensitivity of the centring**, label-free `PF`, 20,000 draws, with the fraction of
training folds carrying an exact 18/18 tie:

| subject | marginal | tie to lower | tie to higher | random tie | training-fold tie rate |
|---|---|---|---|---|---|
| 1 | 21/24 | 0.4057 | 0.4730 | 0.4388 | 0.2032 |
| 17 | 24/21 | 0.4729 | 0.4052 | 0.4388 | 0.2026 |
| 19 | 22/23 | 0.3754 | 0.4071 | 0.3912 | 0.2767 |

This reproduces the adjudication's 0.4056 / 0.4725 / 0.4385.

**(iii) Support size of the W0 reference sets**, B = 199, 300 outer draws per cell,
`default_rng(77)`, median number of distinct values in one reference set:

| rule | S1 low | S1 rand | S17 low | S17 rand | S19 low | S19 rand |
|---|---|---|---|---|---|---|
| KF_free (exact) | 11 | 10 | **6** | 10 | 10 | 13 |
| RE (published) | **1** | **1** | **1** | **1** | **1** | **4** |

These two rows set the informativeness threshold in 6.1 and their boundary values are named there.

**(iv) Measured per-unit costs, this machine, n = 45, 5 folds.** `StratifiedKFold(...).split`, all
folds materialised: **186.3 us**. Majority-class accuracy, direct path: **11.3 us**. `cross_val_score`
with `LinearDiscriminantAnalysis`: **2918 us**. Hand-written two-class LDA, 5 folds: **226.0 us**,
which agrees with sklearn to a maximum absolute difference of **1.11e-16** over 40 random label
vectors. The committed CSP-plus-LDA pipeline at **128 ms** per 5-fold pass is quoted from the previous
pre-registration Section 4.1 and is **not** re-measured here; Section 5.5 registers a timing probe
because this document's most expensive arm depends on it.

### 0.4 The three findings that made me redesign the gate, and that cut against the project

These were not in the adjudication's established-facts list. Two of the three make this project's
existing prose weaker. All three are measured, replicated, and registered before any gate is written.

**FINDING 1. The zero-information dummy null of the published rule is a point mass, so the repo's
own type-I control has exactly zero power against it.**

Under re-stratification the dummy's accuracy takes **one value** at each of the three marginals
(0.5333, 0.5333, 0.4444; sd exactly 0.0000; one distinct value in 20,000 draws, table 0.3i). The
reference set is therefore constant, `p = 1.0` on every draw, and `P(p <= alpha) = 0.0000` **by
construction**. The same holds for the WITHIN rule, and for the same reason with a shorter
derivation: permuting labels inside each fold of a frozen partition preserves every fold's class
counts, hence every training fold's counts, hence the majority-class prediction, hence the number
correct in each test fold. Measured: 0.5333, 0.5333, 0.4444, sd 0.0000, one distinct value.

**Consequence.** The three `0.0000` rows for RE, RE_blk and WITHIN in the Section 2C table quoted in
0.2a are **not evidence of validity**. They are arithmetic. A pre-registration that makes the dummy
world its primary arm registers a gate that the published rule **cannot fail**, which is decoration.

Under a **random** tie-break at S19's marginal the re-stratified null becomes barely non-degenerate
(mean 0.4777, sd 0.0191, four distinct values, training-fold tie rate 0.6000). That is the only cell
in which the dummy world has any grip on the published rule at all, and four atoms is not a
distribution.

**FINDING 2. The famous 0.6550 figure is a tie-break artifact at that marginal. The defect is real,
but it does not live where the repo says it lives.**

`P(p <= 0.05)` for FX, dummy world, M = 1000 outer draws by B = 199 inner permutations,
`default_rng(3)`:

| marginal | tie to lower | tie to higher | random tie |
|---|---|---|---|
| 21/24 (S1) | **0.6740** | 0.0000 | **0.0000** |
| 24/21 (S17) | 0.0000 | **0.6770** | **0.0000** |
| 22/23 (S19) | 0.0570 | 0.0010 | **0.3340** |

Replicated at M = 1500 with two further independent seeds: 21/24 lower gives 0.6487 and 0.6713 and
21/24 random gives 0.0000 and 0.0000; 22/23 random gives 0.3487 and 0.3400.

So the headline demonstration that C2 is broken (0.6550 at subject 1) **disappears entirely under a
fair coin**, and the defect reappears at subject 19's marginal, where sklearn's own tie-break shows
almost nothing. Any document that registers "the positive control fails at 13x nominal at subject 1"
as its proof that the gate bites is registering a fact about `permutation_design.py:891`.

**FINDING 3. The measured size of a structurally invalid rule depends almost entirely on which
zero-information estimator it is pointed at, so a PASS transfers to nothing.**

The same rule, FX, whose invalidity is a structural fact independent of any estimator:

| zero-information world | estimator | measured size at alpha 0.05 | ratio to nominal |
|---|---|---|---|
| W0 dummy, 21/24, tie to lower | majority-class on `zeros((45,1))` | 0.6740 | 13.5x |
| W0 dummy, 21/24, random tie | majority-class | 0.0000 | 0.0x |
| W1 noise, 21/24, d = 1 | LDA on 1 N(0,1) feature | 0.0717 | 1.43x |
| W1 noise, 21/24, d = 4 | LDA on 4 N(0,1) features | 0.0520 | 1.04x |
| real pipeline, S1 (repo's own replicate-based estimate) | CSP(4) + LDA on real EEG | 0.0752 | 1.50x |

(W1 rows at M = 3000, B = 99, `default_rng(7)`; the real-pipeline row is the repo's own
`.provenance_cache/permutation_design.py.txt` "REAL-PIPELINE TYPE-I" line for C2 against C1's 0.0444.)
At d = 4 the mean displacement `obs - mean(reference)` is a real **+0.02685**, and it still produces
no measurable size inflation.

**Consequence.** The empirical gate's power is a property of the (world, marginal, tie-break) cell
and not of the rule. Section 6 is built entirely around that fact.

**FINDING 3b, the calibration number that must travel with every pass.** Over a 10-marginal by
3-tie-break sweep in W0 at M = 600 (`default_rng(55)`), the **known-invalid** rule FX exceeds the gate
line in **4 of 30 cells** (21/24 lower at 0.6933, 24/21 higher at 0.6667, 22/23 random at 0.3600,
23/22 random at 0.3533) and sits at 0.0000 in 18 of 30. The exact rule KF_free never exceeds it
anywhere, with a maximum of 0.0383. **The gate's demonstrated detection rate against the one defect
we are certain exists is 13% of cells.** A pass is weak evidence and Section 9 forbids reporting it
otherwise.

### 0.5 The direction that flatters this project, named per outcome, before the run

| if the run finds ... | who does that flatter? |
|---|---|
| RE and RE_blk clear both gates | **flatters the project.** The published `p <= 0.001` on subject 1's 41/45 keeps its reference distribution. Expected, and the outcome I have the strongest incentive to reach. |
| FX and FX_blk fail | **flatters my own self-correction.** I authored the fixed-partition idea and I wrote its retraction. A gate I designed that kills it is not an independent test of it. Registered as a **positive control**, never as a finding. |
| KF_free / KF_free_blk clear both gates | **the most self-serving available outcome**, because it partly restores an idea I authored and retracted. Fenced in 9.3: it licenses nothing about any published p, since on `PF` subject 19's observed value falls from 29/45 to 21/45. |
| RE or RE_blk fails | **maximally damaging.** Every within-subject permutation p in the repo is void, including the headline's. Leads the write-up ahead of everything (9.1, 10.4). |
| the empirical gate returns NO VERDICT for RE | **awkward for this document, not for the project**, and that asymmetry is itself a risk (Section 16.9). It is the outcome Findings 1 and 3 predict. |
| Finding 2 reproduces | **costs me directly.** I have already written prose calling the displacement "a property of the arithmetic of a fixed partition, not of EEG". Section 9.5 registers the correction. |

### 0.6 What this document may not be cited as

Not an independent replication of Section 2C. Not a blind test. Not evidence that the analyst's
judgement is reliable. It is a pre-committed, higher-resolution, structurally-gated re-run of a check
the analyst already ran and already believes the answer to. Its contribution is that the thresholds,
the consequences, the positive controls, the power calibration and the failure procedure are fixed
**before** the run rather than argued **during** it, which is the exact thing that went wrong last
time.

---

## 1. THE QUESTION

**For each candidate permutation rule on this design: is it a test at all?**

Stated so it has a wrong answer: a permutation rule is a test if and only if, when the null it claims
to test is true, `P(p <= alpha) <= alpha` for every alpha. A rule that rejects a provably true null
more often than its nominal rate is not a test, and no property of where its null centres, how
principled its motivation is, or who proposed it can repair that.

Two sub-questions, answered by two different instruments:

**(a) Does the rule satisfy the hypotheses of the exchangeability theorem?** Answered
**deterministically**, by a mechanical test on the code, in Section 4. This is the primary gate.

**(b) Does the rule show measurable type-I inflation under a null that is true by construction?**
Answered by measurement in Section 5. This is the backstop, and Section 6 registers exactly how weak
it is.

**What this question is NOT.** It is not "is the headline significant". Subject 1's p sits at the
Monte Carlo resolution floor under every rule tried to date and this run cannot move it (9.6). It is
not "which rule gives the smallest p". It is not a re-run of the block-permutation comparison, whose
exact cells C1 and C3 and whose arm B results stand.

---

## 2. WHY THE OLD GATE GOES, AND WHY THAT IS NOT A LICENCE

The single most dangerous sentence available to me is "the registered gate fired, but the gate was
wrong". It is dangerous because it is **true here**, and because it is exactly what a motivated
analyst says when a gate is merely inconvenient. So the argument is made in a form that does not
depend on my judgement.

### 2.1 The old gate is UNSATISFIABLE, and this is a proof rather than a preference

Take the most defensible fixed-partition design available: build the partition with
`KFold(5, shuffle=True, random_state=42).split(X)`, which reads only the shape of `X` and never
touches `y`. It is ancillary, so freezing it yields an exact permutation test by Theorem 3.1. Its
zero-information null centres at 0.4054, 0.4728, 0.3761 (0.2b; reproduced at 0.4055, 0.4735, 0.3757
in 0.3i). **Two of three are outside 0.45 to 0.55.** Even the **re-stratified** dummy, which is what
the band was originally written for, centres at 0.4444 at subject 19's marginal, also outside.

There is therefore no re-run, no reparameterisation and no choice of subject that makes a valid
design pass assert 9 on this data. "Re-run at a design that does not trip the gate" is not a remedy
that exists. **The gate is broken, not the design.**

### 2.2 The old gate is also INSUFFICIENT, counted from the repo's own stdout rather than asserted

This is the sharpest available statement and it is fully checkable against
`.provenance_cache/permutation_design.py.txt`. On the **real pipeline**, assert 9's band flagged:

| family | cells (3 subjects) | null means printed by the run | outside 0.45 to 0.55 |
|---|---|---|---|
| invalid, fixed at `P0` | C2, C4 | 48.26, 47.26, 48.26, **43.80**, 47.04, 45.72 | **1 of 6** |
| exact, fixed at label-free `PF` | C5, C6 | 48.14, 47.28, 48.18, **43.65**, 47.18, 45.43 | **1 of 6** |
| exact, re-stratified | C1, C3 | 50.53, 50.09, 51.11, 47.64, 48.83, 48.07 | 0 of 6 |

**The band flagged invalid fixed-partition cells and exact fixed-partition cells at exactly the same
rate, 1 in 6.** Within the family where the defect actually lived, its verdict carried no information
at all. The single cell it did flag, subject 17's C4, is the one the run recorded as
`8. null means in [0.45,0.55] : FIRES on [(17, 'C4')]`, and the exact rule C6 at the same subject
(43.65) sits outside the band too and was flagged by nothing, because C5 and C6 were added after
registration and were never in the assert's scope.

This is a stronger claim than "the band is neither necessary nor sufficient". It is: on this data,
the band's output and the validity property are **statistically indistinguishable within the family
under test**, and a gate like that manufactures the appearance of scrutiny.

### 2.3 The bundling, which is the mechanism of the actual damage

`permutation_design.py:705+` overrode the gate with two arguments: a **false** exactness criterion
and a **true** centring observation. **The damage was done by bundling.** One true claim and one
false claim were used jointly to license one departure, and the true one carried the false one
through. Section 11.6 registers the rule that forbids the bundle.

Both arguments' fates are treated as **settled inputs**. This document does not re-derive either, and
no clause below depends on re-deriving either. **The 0.45 to 0.55 band appears nowhere below as a
gate, an assert, or a halt condition.** Where centring is printed at all it carries the literal
annotation `descriptive only, no reportability depends on this`, so that a future reader cannot find
a centring clause here and mistake it for a binding one.

### 2.4 What removing a gate is allowed to mean, registered now, before it can be convenient

A gate may be removed only when it is shown that **no valid design can pass it**, demonstrated at the
actual parameters of the data, by a route that uses no EEG, and exhibited as a concrete design rather
than argued. That standard is met in 2.1. A gate may **never** be removed because the design in hand
fails it. The replacement must be strictly harder to satisfy than nothing, and Section 8 discharges
that obligation with measurements rather than assurances.

---

## 3. THE PROPERTY BEING GATED

### 3.1 The theorem, and the two hypotheses it needs

Let `G` be a finite group acting on label vectors: the full permutation group on the orbit of `y` for
the i.i.d. rules, the within-run subgroup for the block rules. Let `P` be the cross-validation
partition and `S(y) = CVacc(X, y; P)`. Suppose:

1. under H0 the conditional law of `y` given `X` is invariant under `G`; and
2. `S` is **one fixed function of its argument**: neither `P` nor `G` may depend on the observed
   vector `y_obs` except through the argument being scored.

Then `(S(y), S(g_1 y), ..., S(g_B y))` is exchangeable for `g_j` i.i.d. uniform on `G`, and

> `p = (1 + #{j : S(g_j y) >= S(y)}) / (1 + B)` satisfies **`P(p <= alpha) <= alpha` for every
> alpha**, with equality where `alpha(B+1)` is an integer and there are no ties. Ties push the rate
> strictly below alpha.

Hypothesis 2 is what `P0 = StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_obs)`
violates: `P0` is a function of the very labels being permuted, so replicate `j` computes
`CVacc(X, g_j y; P0(y_obs))`, which is not `S(g_j y)` for any fixed `S`. Re-stratification restores
hypothesis 2 in a different way, by making `P` a deterministic function of whichever vector is being
scored.

**Hypothesis 1 is an assumption about the data and this document does not test it.** Hypothesis 2 is
a property of the code and Section 4 tests it exactly.

### 3.2 What the gates do not check, stated so a pass is not over-read

1. **Power.** A rule that always returns `p = 1.0` passes both gates trivially and is useless.
   Section 5.6 reports informativeness as an explicitly non-gating companion.
2. **Hypothesis 1.** If the trials are not exchangeable under `G` in the real data, both gates can
   pass and the p can still be meaningless. That is the residual and it is not closed here.
3. **The rest of the pipeline.** Leakage, lattice violations and the defects the old document's
   asserts 1 to 8 and 10 covered are carried forward in Section 13, unchanged in substance.
4. **Alphas below 0.05.** The headline quotes `p <= 0.001`. The empirical gate certifies size at
   alpha 0.05 and 0.10 only. See 9.6 item 2.
5. **Marginals and designs not run.** n = 45, 5 folds, these marginals, these estimators.

---

## 4. THE PRIMARY GATE: A STRUCTURAL TEST OF HYPOTHESIS 2

This is the gate the old document needed and did not have. It is deterministic, costs microseconds,
has no Monte Carlo error, no estimator dependence, no tie-break dependence, and no power problem.

### 4.1 The test, stated as an executable procedure

> **G-S, the `y_obs`-independence test.** For a rule `R`, fix a label vector `y_s` to be scored. Draw
> two different observed vectors `y_obs^(1)` and `y_obs^(2)` from the same class marginal, with
> `y_obs^(1) != y_obs^(2)`. Instantiate `R` under each and record (i) the realised partition, as an
> ordered list of test-index sets, and (ii) the realised permutation group, as the ordered list of
> index blocks within which labels may move.
>
> **`R` PASSES G-S** if and only if both are element-wise identical across the two instantiations,
> for **every** one of 200 registered `(y_s, y_obs^(1), y_obs^(2))` triples per rule per subject.
>
> **`R` FAILS G-S** on a single mismatch.

A rule that passes G-S satisfies hypothesis 2 by direct verification, and is therefore exact by
Theorem 3.1 given hypothesis 1. A rule that fails G-S has a statistic that depends on `y_obs`, the
exchangeability argument does not apply to it, and no measurement is required to disqualify it.

### 4.2 Satisfiable and failable, both by construction, before the run

| rule | partition | group | G-S verdict, predicted with its reason |
|---|---|---|---|
| RE | `SKF(X, y_s)`, a function of the scored vector | full permutation | **PASS.** Neither object mentions `y_obs`. |
| RE_blk | `SKF(X, y_s)` | within-run, from run indices only | **PASS.** Run indices are protocol, not labels. |
| KF_free | `KFold(X)`, reads shape only | full permutation | **PASS.** |
| KF_free_blk | `KFold(X)` | within-run | **PASS.** |
| **WITHIN** | frozen at `P0(y_obs)` | blocks are `P0(y_obs)`'s folds | **FAIL.** Both objects move with `y_obs`. |
| **FX** | frozen at `P0(y_obs)` | full permutation | **FAIL.** Partition moves with `y_obs`. |
| **FX_blk** | frozen at `P0(y_obs)` | within-run | **FAIL.** Same. |

**Four rules pass and three fail, and both directions are exhibited before any number exists.** This
discharges, for the primary gate, the obligation the old document never discharged for assert 9.

### 4.3 The consequence, and the one place it costs the project a rule

A rule failing G-S is **disqualified**. It produces no p, no percentile, no null mean, no tail count
and no figure, and there is no measurement that can rehabilitate it, because G-S is a statement about
the code and not about the data.

**WITHIN fails G-S and is therefore disqualified.** A conditional argument might rescue it: one could
try to condition on the observed fold margins and argue that the within-fold group acts validly on
that conditional law. **I have not constructed that argument and this document does not adjudicate
it.** Under the conflict rule of 11.5 the more restrictive reading wins, so WITHIN is out. If someone
constructs the proof, that is an amendment binding a future run (11.8), never this one. This costs the
project a candidate rule and flatters nothing, and it is registered in that direction deliberately.

### 4.4 What G-S would have done on 2026-07-25

Run against C2 and C4 as they were written, G-S returns FAIL on the first triple, in microseconds,
before any EEG is loaded and before any p exists. There is no null mean to interpret, no band to
argue about, no smoke run to be surprised by, and nothing for a Section 2B to be written about. The
entire episode is foreclosed at the cost of one deterministic assert.

---

## 5. THE BACKSTOP: THE ZERO-INFORMATION TYPE-I MEASUREMENT

G-S verifies hypothesis 2 as written. It cannot catch a harness in which the code that G-S inspects
differs from the code that runs, a seed silently derived from the labels, or a rule that satisfies
hypothesis 2 and is still broken for a reason nobody has thought of. The empirical gate is the
backstop for those, and it is also the instrument that produces the transferable finding.

### 5.1 The worlds

**W0, the dummy.** `X = np.zeros((45, 1))`, `DummyClassifier(strategy="most_frequent")`. H0 is true
because the mutual information between features and labels is exactly zero by construction. Pure
integer arithmetic over fold index sets, so it is cheap enough for a marginal sweep at three
tie-breaks. **Degenerate for RE, RE_blk and WITHIN (Finding 1).**

**W1, LDA on noise, indexed by feature count `d` in {1, 2, 4}.** `X ~ N(0, I)` of shape `(45, d)`,
drawn from an RNG stream that never touches the label stream, so `X` and `y` are independent by
construction. Estimator `LinearDiscriminantAnalysis()`, the class the committed pipeline ends in;
`d = 4` matches `CSP(n_components=4)`. Non-degenerate for every rule. `d` is registered as a family
because Finding 3 shows the gate's power varies by an order of magnitude across it, and **the worst
case over `d` governs**.

**W2, the committed pipeline on real EEG.** Subject 1's real `X` through the committed
`CSP(4) + LDA` pipeline, with `y0` drawn uniformly from the permutations of subject 1's real 21/24
label vector. H0 is exactly true because `y0` is independent of `X` by construction, while `X` keeps
its real covariance structure, its fold-to-fold heterogeneity, and the CSP refit inside every
training fold. This is the only arm containing the estimator whose p-values this project reports, and
it is the only arm that can produce a live cell for the published rule with realistic features.

### 5.2 The rules

All seven, because all seven have numbers in the repo's stdout that a reader can find. C5 and C6
carry printed p-values (S17 0.25487 and 0.11444; S19 0.56522 and 0.48576) and must be gated rather
than quietly dropped.

| id | rule | repo cell | role |
|---|---|---|---|
| RE | i.i.d. shuffle, re-stratified per scored vector | C1 | **the published rule.** Candidate. |
| RE_blk | within-run shuffle, re-stratified | C3 | the one correction the 2026-07-26 re-reading still stands behind. Candidate. |
| KF_free | i.i.d. shuffle, frozen at label-free `PF` | C5 | candidate, **different statistic** (9.3). |
| KF_free_blk | within-run shuffle, frozen at `PF` | C6 | candidate, different statistic. |
| WITHIN | permute within folds of `P0` | new | **disqualified by G-S (4.3).** Measured for the record only. |
| **FX** | i.i.d. shuffle, frozen at `P0(y_obs)` | C2 | **POSITIVE CONTROL for i.i.d. rules.** Withdrawn 2026-07-26. Never a candidate. |
| **FX_blk** | within-run shuffle, frozen at `P0(y_obs)` | C4 | **POSITIVE CONTROL for within-run rules.** Never a candidate. |

FX and FX_blk are **not eligible to be a reported rule under any outcome**, whatever either gate says.
Registering that now removes the temptation to rehabilitate them if they happen to pass somewhere,
which Finding 3b says they will in 87% of W0 cells.

### 5.3 The procedure, written as an analyst would actually run it

```
for each cell (world, marginal, tie-break):
  for i in 1..M:                                    # outer draws
      y0  <- uniform permutation of the cell's marginal vector      # H0 is now true
      X   <- the world's feature matrix (fresh per draw in W1)
      P0  <- StratifiedKFold(5, shuffle=True, random_state=42).split(X, y0)   # analyst stratifies
      PF  <- KFold(5, shuffle=True, random_state=42).split(X)                 # label-free
      obs_RE   <- score(y0, restratified on y0)     # handed to RE, RE_blk
      obs_FX   <- score(y0, P0)                     # handed to WITHIN, FX, FX_blk
      obs_free <- score(y0, PF)                     # handed to KF_free, KF_free_blk
      for j in 1..B:
          y' <- draw under the rule's group; score under the rule's partition
      p_i <- (1 + #{ref >= obs - 1e-9}) / (1 + B)                 # sklearn's convention exactly
  r_hat(alpha) <- (1/M) * #{i : p_i <= alpha}
```

Each rule receives **its own** observed value, the one that rule's analyst would compute. Handing
every rule the same observed value would test a different and easier question (assert A4).

**Pairing.** At outer draw `i` and inner replicate `j` the **same** i.i.d. label vector feeds RE, FX
and KF_free; the **same** within-run vector feeds RE_blk, FX_blk and KF_free_blk. Between-rule
differences therefore carry no between-rule Monte Carlo noise. The script asserts element-wise
identity of the shared vectors (F7); a paired design that is not actually paired is worse than an
unpaired one, and the previous run's unpaired cells are in the record.

### 5.4 M, B, the thresholds, and the family-wise budget

`B = 199` in W0 and `B = 99` in W1 and W2. Both make `alpha(B + 1)` an integer at both binding alphas
(10 and 20; 5 and 10), so `p <= alpha` is exactly representable and no part of the measured rate is a
rounding artifact. Increasing `B` buys little here because the statistic is discrete on the k/45
lattice and its atoms, not `B`, limit p-resolution.

The gate line is an **exact one-sided binomial** tail, not a normal approximation. Registered
per-cell false-alarm budget `q = 1e-4`: the line is the smallest integer `k` with
`P(Binom(M, alpha) >= k) <= q`.

| arm | M | line at alpha 0.05 | line at alpha 0.10 |
|---|---|---|---|
| W0, 3 real marginals x 3 tie-breaks | 2,000 | `k >= 139`, rate **0.0695** | `k >= 253`, rate **0.1265** |
| W0 sweep, 10 marginals x 3 tie-breaks | 1,000 | `k >= 78`, rate **0.0780** | `k >= 138`, rate **0.1380** |
| W1, d in {1,2,4}, 3 marginals | 2,000 | rate **0.0695** | rate **0.1265** |
| W2, subject 1, 4 rules | 3,000 | `k >= 197`, rate **0.0657** | `k >= 364`, rate **0.1213** |

**Family-wise budget, and the unit distinction that must not be blurred.** A **cell** is a (world,
marginal, tie-break) triple and is the unit of Section 6.1's liveness. A **comparison** is a
(cell, alpha) pair and is the unit of the false-alarm budget, because each alpha is a separate
one-sided test. The census: W0 at the three real marginals, 3 x 3 = 9 cells, 18 comparisons; the W0
sweep, 10 x 3 = 30 cells, 60 comparisons; W1 at d in {1,2,4} x 3 marginals = 9 cells, 18 comparisons;
W2, 1 cell, 2 comparisons. **49 cells and 98 comparisons per rule.** Five candidate rules (RE,
RE_blk, KF_free, KF_free_blk; WITHIN is counted although 4.3 already disqualifies it, which makes the
budget conservative) gives **490 comparisons**. At `q = 1e-4` the expected number of false
disqualifications under exactness is **0.049** and `P(at least one) <= 0.0478`. There is **no re-run
and no appeal**: a rule
that fails is disqualified on its first failure. Registered direction of this policy's error: it can
only **disqualify a valid rule**, which costs the project and flatters nothing, and it can never admit
an invalid one.

**Power, printed so no pass is over-read:**

| arm | vs 1.3x nominal | vs 1.5x | vs 2x |
|---|---|---|---|
| M = 1,000, alpha 0.05 | 0.057 | 0.376 | 0.993 |
| M = 2,000, alpha 0.05 | 0.219 | 0.835 | 1.000 |
| M = 2,000, alpha 0.10 | 0.689 | 0.999 | 1.000 |
| M = 3,000, alpha 0.05 | 0.452 | 0.978 | 1.000 |

W2 at M = 3,000 has **power 0.980** against the repo's own real-pipeline estimate of the C2 defect
(0.0752). That number is why M is 3,000 there and not less.

### 5.5 Cost, the timing probe, and the reduction ladder

Costs from 0.3iv. W0 and W1 are minutes: the sweep is 10 marginals x 3 tie-breaks x 7 rules x 1,000 x
200 dummy evaluations at 11.3 us plus re-stratification at 186.3 us, about 1.9 core-hours; the three
real marginals at M = 2,000 add about 1.3; W1 at three `d` values, three marginals, seven rules,
M = 2,000, B = 99 at 226 us adds about 2.6. **W2 dominates**: 4 rules x 3,000 x 100 x 128 ms is
**42.7 core-hours**, about 2.7 hours on 16 cores. Total under **49 core-hours**, comparable to the
previous registered run's 4.3 hours plus 7.3 hours serial.

**The timing probe and the ladder, applied without discretion and BEFORE any gate cell is
evaluated.** A probe at M = 20 extrapolates each arm. If the total exceeds a registered ceiling of
**6 hours on 16 cores**, apply in this order until it does not: (L1) W2 rules from {RE, RE_blk, FX,
FX_blk} to {RE, FX}; (L2) W0 sweep M 1,000 to 600; (L3) W1 M 2,000 to 1,200; (L4) W2 M 3,000 to
2,000. Every line is recomputed from the exact binomial formula at the reduced M; **no threshold is
ever hand-edited.**

Three clauses that close the obvious abuses:

- **The ladder is frozen before the first gate cell is computed** and may not be re-entered
  afterwards. Reducing M after seeing a failure would raise the line and rescue the rule.
- **Every reduction RAISES the line and therefore makes passing EASIER, which flatters the project.**
  Any applied reduction is the first line of stdout and must be repeated in the first paragraph of
  any write-up.
- **If L1 to L4 are all applied and the ceiling is still exceeded, W2 is not silently dropped.** It
  is replaced by the registered fallback **W2-lite**: LDA on subject 1's real `X` projected onto a
  **label-free** 4-component PCA basis, which keeps the real feature covariance and drops the CSP
  refit. If W2-lite also cannot run, W2 returns **NO VERDICT** and Section 9.1's no-live-cell row
  fires. Registering the fallback now means the substitution is not chosen at the keyboard.

### 5.6 Informativeness, reported and explicitly NON-GATING

For each rule that survives, at each real marginal, the script reports the rejection rate at alpha
0.05 under a **planted** effect: in W1, a between-class mean shift of `delta` in {0.5, 1.0} sd on one
feature, 200 outer draws. **This number cannot change any gate verdict in either direction.** It
exists so that "RE_blk survived" is never reported without "and here is what it can detect", and so
that a degenerate rule which never rejects anything cannot be sold as the safest choice.

### 5.7 RNG discipline

One `numpy.random.SeedSequence(20260730)` at the top; every cell takes its own stream via `.spawn()`
on a key printed to stdout, so cells are independently reproducible and can run in any order on any
number of cores without stream collisions. `SEED_PARTITION = 42` for every splitter, matching the
committed pipeline. In W1 the feature stream and the label stream are spawned from different keys and
the independence is measured (F13).

---

## 6. THE DECISION PROCEDURE, WITH NO DISCRETION ANYWHERE

Findings 1 and 3 mean a naive "passes everywhere therefore valid" rule would certify rules on the
strength of cells that had no power to fail them. The following three definitions close that, and
every one of them is a computed boolean.

### 6.1 Cell status

A **cell** is a (world, marginal, tie-break) triple. Within a cell the gate binds at both alphas.

- **INFORMATIVE for rule R** iff **both**: (i) the fraction of outer draws in which R's reference set
  has `sd > 0` is at least **0.95**; and (ii) the **median number of distinct values in the reference
  set is at least 6**.

  **Where the 6 comes from, disclosed because it is a tuned number.** From table 0.3iii, measured
  2026-07-30 at B = 199, 300 outer draws per cell, `default_rng(77)`: the exact rule KF_free has
  median distinct counts of 11, 6, 10 under sklearn's tie-break and 10, 10, 13 under a random
  tie-break at subjects 1, 17, 19; the re-stratified rule RE has **1** everywhere except
  (19, random tie), where it has **4**. The
  threshold is set **below KF_free's minimum (6, at subject 17) and above RE's maximum (4)**, and
  both boundary numbers are named here so the tuning is visible rather than discoverable later.
  Direction of the choice: setting it at 6 rather than 10 **admits KF_free**, which 0.5 names as the
  most self-serving rule available, and **excludes RE from every W0 cell**, which costs the published
  rule its cheapest route to certification. The two effects run opposite ways, and neither is a
  reason to move the number after the run.

  Criterion (ii) exists because criterion (i) alone is defeated by a four-atom reference set. Without
  it, the (19, random tie) cell would be LIVE for RE, and the published rule could be certified on a
  reference distribution with four support points. That is exactly the degeneracy trap of Finding 1
  wearing a thinner disguise. Finding 1's derivation already predicts which cells fail; criterion (ii)
  is the backstop against that derivation being wrong.
- **CALIBRATED** iff the matched positive control is disqualified **in that same cell**: FX for the
  i.i.d. rules, FX_blk for the within-run rules.
- **LIVE for R** iff INFORMATIVE for R **and** CALIBRATED.

### 6.2 The verdict, per rule

> **DISQUALIFIED** if R fails G-S, **or** if R exceeds the gate line in **any** cell at **any** alpha,
> **live or not**. Liveness governs certification only. It never excuses a failure. There is no cell
> whose failure can be discounted.
>
> **CERTIFIED** if R passes G-S, is disqualified nowhere, and passes in **at least three LIVE cells
> spanning at least two distinct worlds.**
>
> **NO VERDICT (empirical)** otherwise: R passes G-S, is disqualified nowhere, and the empirical gate
> had insufficient demonstrated power to say anything. R's standing then rests on Theorem 3.1 alone,
> and the sentence *"exact by Theorem 3.1; the empirical gate returned no verdict because no live
> cell was available"* must accompany every mention of its p.

The two-world, three-cell requirement exists because of Finding 3. A single live cell in a single
world is exactly the estimator-dependence trap, and certifying on one would be reading a property of
the dummy as a property of the pipeline.

### 6.3 Ordering, enforced by the script and auditable from stdout

1. G-S runs for all seven rules. Verdicts printed.
2. Harness asserts A1 to A10 and F1 to F14 run and pass.
3. W0, W1, W2 run. Every rate is printed with its M, its exact-binomial line, and its cell status.
4. The **cell status table** is printed: INFORMATIVE, CALIBRATED, LIVE, per rule per cell.
5. The **verdict list** is printed.
6. **Only then** is any EEG p-value computed or printed.

Any stdout in which step 6 precedes steps 4 and 5 invalidates the run (F12). The point is that the
rule is chosen at a moment when its p-value is not knowable.

### 6.4 The priority list, fixed now, on grounds independent of any p-value

Among rules that are CERTIFIED, then among rules at NO VERDICT, the reported rule is the **first** of:

1. **RE_blk.** Exact by 3.1; makes the **weakest** exchangeability assumption (within run only);
   computes the **same statistic** as the published result, so its p attaches to 41/45.
2. **RE.** Exact by 3.1; same statistic; assumes exchangeability across runs, a stronger assumption.
   `= C1`, the published rule.
3. **KF_free_blk**, then 4. **KF_free.** Exact by 3.1, but a **different statistic** (9.3).

CERTIFIED always outranks NO VERDICT. Within a status class the order above is fixed. The criteria
are assumption strength and statistic identity, neither of which is a function of any p-value. All
other surviving rules are reported as sensitivity analyses in full, including any whose p is less
flattering than the reported rule's.

---

## 7. WHEN A GATE FIRES: NO DISCRETION, NO OVERRIDE

### 7.1 The consequence, stated without conditions

A disqualified rule **produces no number that is reported anywhere**: not a p, not a percentile, not a
null mean, not a tail count, not a figure. It may be **printed** so that what was computed stays
visible, and it may be **discussed as a failure**, and that is the whole of its permitted use. There
is no conditional reportability, no reportability with a caveat, no reportability because the
mechanism is understood, and no reportability because the failure was small. Section 5.4's thresholds
are the definition of small.

### 7.2 The mechanism, because a rule enforced by prose is defeatable by prose

This is the direct lesson of `permutation_design.py:705+`, where a registered halt was overridden by
a well-written section inside the analysis script. The remedy is to take the decision out of prose.

```python
STRUCTURAL = run_gs(RULES)                       # {rule: bool}, deterministic
GATE_PASS  = evaluate_gate(STRUCTURAL, TYPE1, CELL_STATUS)   # {rule: 'CERT'|'NOVERDICT'|'DISQ'}
REPORTABLE = {r: v != 'DISQ' for r, v in GATE_PASS.items()}  # constructed once, never widened
def emit(rule, label, value):
    if not REPORTABLE[rule]:
        print(f"  WITHDRAWN (gate failed: {GATE_REASON[rule]}): {label} = {value}")
    else:
        print(f"  {label} = {value}")
```

Registered requirements on the script:

1. `evaluate_gate` reads only the measured rates, the G-S booleans, and the constants in Section 5.4.
   It takes no argument a human sets after seeing results.
2. **Every** p-value, percentile and derived statistic printed anywhere goes through `emit`. Enforced
   by a source check in the style of the repo's existing `check_wording.py`: no bare p-value format
   string may appear outside `emit`.
3. Assert A6 runs at the end: `REPORTABLE` is element-wise no wider than `GATE_PASS`.
4. **The script contains no prose section arguing against a gate outcome.** Adding one is a violation
   of this pre-registration **regardless of whether the argument in it is correct**. The argument at
   `:705` was half correct and still produced a run whose conclusions had to be withdrawn, because it
   was made after the gate fired, by the person who wanted the result.
5. On any `[HARNESS]` condition the process exits non-zero and **nothing from the run is reported**,
   including numbers already printed and numbers unrelated to the failure.

### 7.3 What is explicitly not available

- Adding a rule after seeing results and reporting it as though registered. Any post-run rule is
  labelled `POST-HOC` in every table and may never be a headline number.
- **Adding a WORLD after seeing results.** The worlds are W0, W1 at d in {1, 2, 4}, and W2 (or its
  registered fallback W2-lite). A world invented after the run cannot supply a LIVE cell, cannot
  contribute to 6.2's two-world count, and may not change any verdict. This is registered because
  8.4 makes the exact incentive visible in advance: the cheapest way to certify RE without W2 would
  be to bolt on a fourth cheap world, and it is closed here rather than after the temptation exists.
- Re-running at a different M until a rule passes. A5 asserts `M` equals the registered value; any
  exploratory run at reduced M prints `NOT A REGISTERED RUN` on every line and may not be quoted.
- Re-entering the reduction ladder after gate evaluation begins (5.5).
- Choosing an alpha, a p convention, a tie-break arm, a feature count `d`, or a world after seeing
  which is kinder. All fixed in Sections 5.1 to 5.4; worst case governs across tie-breaks, `d`, and
  marginals.
- Discounting a failure because its cell was not LIVE. Forbidden by 6.2 in that direction explicitly.
- Treating a Section 11 amendment as an override. Closed at 11.8.

### 7.4 The discreteness direction, disclosed because it runs my way

The W0 statistic lives on the k/45 lattice, the reference distribution is heavily tied, and the `>=`
convention with the `+1/+1` correction makes p **conservative**. That biases the measured rate
**downward**, which biases the gate **toward passing**, which flatters the project. Registered
consequence, fixed now: **the conventional p binds**, because it is the p this project reports. The
mid-p variant `(0.5 * #{ref == obs} + #{ref > obs} + 1) / (1 + B)` is computed and reported as
**descriptive only**. If a rule passes on the conventional p and fails on the mid-p, the write-up
must state that in the same sentence as the pass, using the phrase **"the pass depends on the
discreteness convention"**.

### 7.5 If everything fails

If no rule is CERTIFIED or at NO VERDICT, no permutation p may be reported for the within-subject
design at all. The headline is re-quoted using only evidence that does not depend on a permutation
null: **41/45 = 91.1% on one subject at n = 45, against that subject's majority-class floor of
24/45 = 53.3%**, with an exact binomial or Wilson interval and the explicit statement that no valid
permutation reference distribution was found. That is a publishable negative result about the design
and 9.7 registers it as leading the write-up.

---

## 8. SATISFIABILITY AND BITE, DEMONSTRATED IN BOTH DIRECTIONS BEFORE THE RUN

The old document's fatal flaw was a gate no valid design could pass. This section discharges the
obligation for every gate registered here, with measurements rather than assurances.

### 8.1 G-S: satisfiable and failable by construction

Section 4.2. Four rules pass, three fail, and each verdict follows from reading the rule's definition.
No measurement is involved, so there is no resolution at which the demonstration could be too weak.

### 8.2 The empirical gate is SATISFIABLE, exhibited with numbers

**KF_free, provably exact by 3.1, passes the line everywhere I have measured it.** Over the 10 x 3 W0
sweep at M = 600 its maximum is **0.0383** (22/23, random tie) against an exact-binomial line of
**0.0883** at that M; at the three real marginals at M = 1,000 its maximum is **0.0400** against a
line of 0.0780; across the four W1 cells of 8.4 at M = 1,200 its maximum is **0.0467 at alpha 0.05
and 0.0858 at alpha 0.10**, against lines of 0.0758 and 0.1342; the repo's own Section 2C reads
0.0250 / 0.0000 / 0.0150. **A rule the OLD gate killed at 0.4054 passes the NEW gate with room.** The
replacement is not the same taste relabelled: it accepts a design the old one rejected, and it does so
for a reason that is a theorem.

Under exactness the probability of a rule exceeding a `q = 1e-4` line in any of its 98 cells is at
most 0.0098, so a correct design passes the whole family with probability at least 0.990.

### 8.3 The empirical gate BITES, exhibited with numbers, and its power is honestly bounded

FX exceeds the line at M = 1,000 or better in: W0 21/24 lower (0.6740, replicated 0.6487 and 0.6713);
W0 24/21 higher (0.6770); W0 22/23 random (0.3340, replicated 0.3487 and 0.3400); W0 23/22 random
(0.3533); W1 d = 1 at 21/24 (0.0717 at alpha 0.05, 0.1477 at alpha 0.10) and at 22/23 (0.0813,
0.1660). Power against the 22/23 random cell at M = 2,000 is 1.000; against the W1 d = 1 cell it is
0.66 at alpha 0.05 and 0.997 at alpha 0.10.

**And the bound, which is the number that must travel with every pass.** Over the 10 x 3 W0 sweep, FX
exceeds the line in **4 of 30 cells** and reads exactly 0.0000 in 18 of 30. In W1 at d = 4 it reads
0.0520 and 0.0560, indistinguishable from nominal, despite a real mean displacement of +0.027. **A
structurally invalid rule shows no measurable inflation in the large majority of cells.** The
empirical gate is a backstop with a demonstrated detection rate near 13% of cells, and Section 6's
LIVE-cell machinery exists precisely so that the 87% cannot be read as evidence of anything.

### 8.4 LIVE cells exist for the published rule, exhibited

W1 at M = 1,200, B = 99, `default_rng(9)`, exact-binomial lines 0.0758 at alpha 0.05 and 0.1342 at
alpha 0.10:

| cell | RE, alpha 0.05 / 0.10 | RE non-degenerate | FX (the matched control) | cell status for RE |
|---|---|---|---|---|
| d = 1, 21/24 | 0.0283 / 0.0775 | 1.000 | 0.0675 / **0.1517** | **LIVE, RE passes** |
| d = 1, 22/23 | 0.0367 / 0.0775 | 1.000 | 0.0750 / **0.1558** | **LIVE, RE passes** |
| d = 2, 21/24 | 0.0458 / 0.0942 | 1.000 | **0.0850** / **0.1483** | **LIVE, RE passes** |
| d = 2, 22/23 | 0.0450 / 0.0825 | 1.000 | 0.0758 / **0.1550** | **LIVE, RE passes** |

So the empirical gate is not decoration for the published rule: there are four cells in which RE's
reference distribution is live, the known-broken rule demonstrably fails, and RE passes. KF_free and
WITHIN pass in all four as well.

**And the precise, falsifiable consequence of 6.2's two-world requirement.** All four of those cells
are in **one** world, W1. With the measurements in hand, **RE therefore cannot be CERTIFIED without a
calibrated cell in W0 or W2**, and W0 is degenerate for it (Finding 1). **RE's certification hinges
entirely on W2**, which requires FX to fail the line in the real-pipeline arm. The repo's own
replicate-based real-pipeline estimate for C2 (0.0752 against C1's 0.0444) suggests it will, and
W2's M = 3,000 gives 0.980 power against exactly that value. If it does not, RE lands at NO VERDICT
and 10's verbatim sentence fires. That prediction is registered here so it can fail on the record,
and the two-world requirement is deliberately **not** relaxed to let d = 1 and d = 2 count as
separate worlds, because they are the same estimator on the same feature distribution and counting
them separately would be weakening a rule to reach a nicer verdict.

### 8.5 The old gate side by side, so the improvement is measured and not asserted

| property | old gate, centring in 0.45 to 0.55 | G-S | empirical gate |
|---|---|---|---|
| a valid design that passes exists | **no** (2.1) | yes, four exhibited (4.2) | yes, exhibited with numbers (8.2) |
| an invalid design fails | 1 of 6 invalid cells (2.2) | yes, three exhibited, deterministically (4.2) | yes, 4 of 30 W0 cells (8.3) |
| flags exact designs falsely | 1 of 6 (2.2), same rate as invalid | never, by construction | 0.0098 per rule family (8.2) |
| verdict depends on the estimator | not applicable | **no** | **yes, by up to 13x** (Finding 3) |
| defeasible by a paragraph | **yes, and it was** | no | no |

The honest reading of the last two rows: **G-S is the gate that decides, and the empirical arm is
calibration and insurance.** Any document that made the empirical arm primary would be building on
the row that says "depends on the estimator by up to 13x".

### 8.6 What would show this section wrong

F9 (no positive control fails anywhere, so the empirical gate has no exhibited failure mode) and F15
(no candidate is CERTIFIED or at NO VERDICT, so Section 8.2's satisfiability claim is refuted). Both
have rows in Section 9 and both trigger Section 11 against **this** document.

---

## 9. PRE-REGISTERED OUTCOMES

Every table carries the flattering direction in the same detail as the unflattering one.

### 9.1 The published rule, RE, and the block rule, RE_blk

| result | meaning, fixed in advance | who does it flatter? |
|---|---|---|
| RE and RE_blk pass G-S and are CERTIFIED | The published within-subject null is a valid test and the empirical arm corroborated it in live cells across two worlds. Every published within-subject p, including subject 1's `p <= 0.001` on 41/45, stands as published. Must be written as **"no inflation was detectable at the resolutions in 5.4"**, never as "proven exact", and Section 0.6 travels with it. | **the project.** Expected, and the outcome I am most motivated to reach. |
| RE and RE_blk pass G-S but reach **NO VERDICT** | Exact by Theorem 3.1; the empirical arm had no live cell. The registered sentence of 6.2 attaches to every mention of their p. **This is the outcome Findings 1 and 3 predict** if W2 does not deliver a calibrated cell. | awkward for this document; neutral for the project. See 16.9. |
| **RE or RE_blk DISQUALIFIED** | Either the code violates hypothesis 2 in a way its definition does not show, or the rule over-rejects a provably true null. Every within-subject permutation p in this repository is void, including the headline's. The 41/45 survives as an accuracy against a 24/45 floor; the p does not. | **maximally damaging. Publishes first, at the top, ahead of everything, with the verbatim sentence of 10.4.** |
| RE and RE_blk differ in verdict | The i.i.d. and within-run groups are not interchangeable. The surviving one governs by 6.4 and the difference is itself reportable. | neutral, informative |

### 9.2 The degeneracy finding

| result | meaning, fixed in advance | who does it flatter? |
|---|---|---|
| RE, RE_blk and WITHIN confirmed non-informative in W0 at all three marginals under sklearn's tie-break | Confirms Finding 1 at higher resolution. **The repo's Section 2C `0.0000` rows for those rules are arithmetic, not evidence**, and any document citing them as evidence of validity is corrected in place. Reported as confirmation of a fact measured 2026-07-30, never as a discovery. | **bad for the repo's existing prose**, which I wrote, and it gets corrected |
| the (19, random tie) cell reproduces at four atoms and is therefore NOT informative | Confirms 6.1's criterion (ii) is doing the work it was registered for. RE gets **no** live cell anywhere in W0, and its certification depends entirely on W2 (8.4). | costs the project its cheapest certification route. Registered before the run for that reason. |
| that cell reproduces at **6 or more** atoms, making it INFORMATIVE | Then RE has a W0 live cell and 6.2's two-world requirement can be met without W2. Reported with the atom count in the same sentence as the certification, every time, and the write-up must state that the second world was supplied by a reference distribution with fewer than ten support points. | flattering, and fenced |
| any of them is found INFORMATIVE in a cell where 0.3i predicts a point mass | Not an outcome. A defect. Halt (F14). | broken |

### 9.3 The label-free rules, KF_free and KF_free_blk. This is where my conflict lives.

| result | meaning, fixed in advance | who does it flatter? |
|---|---|---|
| KF_free CERTIFIED | A fixed-partition permutation test **is** available on this data, and an idea I authored and retracted is partly restored. Registered now, before the number exists: **this licenses nothing about any published p.** It tests a **different statistic on a different partition**; on `PF` subject 19's observed value falls from 29/45 to 21/45 and subject 17's from 28/45 to 25/45. A pass may not be reported as "the fixed-partition correction was right after all", may not be used to re-quote any headline, and if it ever becomes the reported rule under 6.4 the different-statistic caveat appears on **every** mention. | **the most self-serving outcome available. Held to the highest bar in this document.** |
| KF_free CERTIFIED and its null again centres outside 0.45 to 0.55 | The clearest demonstration that the replacement was necessary: the rule the old gate killed passes the new one. **Known in advance** (0.2b, 0.3i); a re-measurement, not a discovery. | the argument I am making. Disclosed. |
| KF_free or KF_free_blk DISQUALIFIED | A rule that is provably exact by 3.1 over-rejects, which is a contradiction. It points at the harness or at an error in 3.1, not at the rule. Halt under F16 and diagnose. | **broken** |

### 9.4 The positive controls

| result | meaning, fixed in advance | who does it flatter? |
|---|---|---|
| FX and FX_blk fail G-S, and fail the empirical line in at least one cell | Expected. The 2026-07-26 withdrawal of C2 and C4 is confirmed by an instrument registered in advance. Report the excess **and the cells where it does not bite**, because "0.6740 at 21/24 under one tie-break and 0.0000 under a fair coin" is the finding, not "anti-conservative everywhere". | good, expected, confirms a correction I wrote. Not independent. |
| FX and FX_blk fail G-S but the empirical line **nowhere** | F9. The empirical arm has no exhibited failure mode in this run, **no cell is CALIBRATED**, no rule can be CERTIFIED, and every candidate falls to NO VERDICT on the strength of G-S alone. | **bad for this document.** Publishes at the top. |
| FX or FX_blk **passes G-S** | The structural test does not detect a dependence on `y_obs` that is visible in the rule's definition. The harness is not implementing G-S. Halt (F17). | **broken** |

### 9.5 The tie-break check

| result | meaning, fixed in advance | who does it flatter? |
|---|---|---|
| Finding 2 reproduces: FX's inflation at 21/24 vanishes under a fair tie-break and appears at 22/23 instead | The repo's flagship 0.6550 figure is a property of the marginal, the class coding and `permutation_design.py:891` jointly. Every sentence describing the displacement as *"a property of the arithmetic of a fixed partition, not of EEG"*, in `permutation_design.py` Section 2B and in the completed run's write-up, is **over-claimed and is corrected in place** to name the interaction with the tie-break. **Section 2B's centring argument is NOT withdrawn by this**: the band still does not transfer, as 2.1 proves independently. Only the stated cause is corrected. | **bad for prose I wrote.** Publishes. |
| a candidate's verdict differs across tie-break arms | Worst case governs (6.2), so the rule is disqualified whichever arm favoured it. The write-up may not report the favourable arm as the verdict and may not call the disqualification conservative. `permutation_design.py:891` would then be a load-bearing line in a validity argument, which it should never have been. | bad for that rule, and it is the check working |
| null displacement and validity verdicts move independently across arms | The registered reading: **displacement and validity are different quantities**, the same lesson as 2.2 arriving by a second route. It does not license reopening the centring band. | neutral, informative |

**Two sentences forbidden in advance**, because both are available readings of 0.3ii and both are
wrong. *"The fixed-partition null is displaced downward because a fixed partition forces
complementary train/test imbalance"* attributes the whole displacement to the partition rule; 6.7
points of it at the 24/45 marginal are carried by a class-coding convention with no scientific
content. *"The displacement is just a tie-break artifact"* is also false: under a random tie-break the
label-free null still sits at 0.4388, 0.4388 and 0.3912, all below 0.45. **The registered honest
statement: the displacement is jointly produced by the partition rule and by an argmax tie-break
convention, and neither alone accounts for it.**

### 9.6 The outcome that applies to every cell

1. **Nothing here is evidence that the decoder decodes.** Every world contains zero information by
   construction. A rule that passes has not found anything.
2. **Size is gated at alpha 0.05 and 0.10 only.** The headline quotes `p <= 0.001`. Extrapolating a
   size guarantee from 0.05 down to 0.001 is an assumption this run does not test, and **no write-up
   may say the headline's alpha was validated.** The Theorem 3.1 guarantee does hold at every alpha,
   for rules that pass G-S, conditional on hypothesis 1; that is a proof and must be labelled as one
   rather than as a measurement.
3. **This run cannot move the headline.** Subject 1's p sits at the Monte Carlo resolution floor under
   every rule tried to date, at roughly 4.5 null sd. Reported as `p <= 0.001`, as a bound, never as a
   measured value. Presenting "p unchanged" as the gate vindicating the headline would be reading a
   resolution floor as a measurement.
4. **Scoped to n = 45, 5 folds, three real marginals plus ten swept ones, and these estimators.**
5. **The LOSO arm is out of scope.** `LeaveOneGroupOut._iter_test_masks` reads only `groups`, so its
   partition passes G-S by construction and arm A's defect does not exist there. Its results (sd ratio
   1.5730, observed 535/900 outside the block null at `p <= 0.0005`, replacement guard at the block
   null's 99th percentile of 478/900) are unaffected and unchanged.
6. **The block-permutation comparison itself is out of scope.** C1 and C3 are exact and their
   comparison stands. This document decides validity, not the block question.

### 9.7 Reporting order, fixed now so it cannot be arranged later

The write-up leads with, in this order, whichever applies first: no rule survived; RE or RE_blk
disqualified; a positive control passed G-S (harness broken); no cell was CALIBRATED; the Finding 2
correction; a tie-break disqualification; the degeneracy correction to the repo's Section 2C reading;
then any pass. **The flattering outcome is last in the queue by construction.**

---

## 10. VERBATIM SENTENCES, PRE-COMMITTED SO THE WRITE-UP CANNOT BE SOFTENED

If the corresponding outcome occurs, the following sentence appears in the write-up, in these words,
before any other conclusion.

- **RE or RE_blk disqualified:** *"The published within-subject permutation p-values in this
  repository, including the p <= 0.001 attached to subject 1's 41/45, are produced by a rule that
  fails a validity gate registered before this run. They are withdrawn. This project currently has no
  valid permutation p for its headline number, which survives as an accuracy of 91.1% on one subject
  at n = 45 against that subject's 53.3% majority-class floor."*
- **No cell CALIBRATED:** *"The empirical arm of this run failed its own positive control everywhere.
  No rule is certified by it. The surviving rules stand on Theorem 3.1 alone, and this run provides no
  independent corroboration of them."*
- **RE at NO VERDICT:** *"The published rule is exact by Theorem 3.1 and passed the structural gate.
  The zero-information type-I arm returned no verdict on it, because its null under a zero-information
  estimator is a point mass and therefore cannot fail. The repository's existing 0.0000 type-I figures
  for this rule are arithmetic and are not evidence of validity."*
- **Everything passes as predicted:** *"This run confirms a conclusion the analyst reached before
  writing this document, using gates the analyst designed after seeing the data that motivated them.
  It is a confirmation at higher resolution against pre-committed thresholds, and it is not an
  independent discovery."*

---

## 11. IF THIS PRE-REGISTRATION IS ITSELF DEFECTIVE

The failure of 2026-07-25 was not that a gate fired. It was that the document **contained two clauses
that could not both hold**, the run discovered it, and the analyst resolved the conflict mid-run,
inside the analysis script, in the direction that lowered a p-value. A pre-registration with no
procedure for its own defects delegates that resolution to the person with the strongest interest in
the answer. This section removes the delegation.

### 11.1 Definition

An **internal inconsistency** is any of: **(i)** two registered clauses that no possible outcome can
satisfy simultaneously; **(ii)** a registered gate that no valid design can pass, or that no invalid
design can fail; **(iii)** a registered outcome table with no row for an outcome the run produces;
**(iv)** a registered threshold that cannot be evaluated as written.

The 2026-07-26 event was type (i): assert 9 required every cell's null mean inside 0.45 to 0.55 while
the same document's Section 7 mechanism predicted a fixed-partition null below 0.50. It was **also**
type (ii), which nobody noticed until the adjudication, and **also** type (iii), which the old
document's own line 713 records: *"The registered table has no cell for 'the comparison arm is
invalid'."*

### 11.2 The distinction that must be made first

- **(A) A gate fires.** Section 7. Mechanical, no discretion, and overwhelmingly the likely case.
- **(B) A gate is unsatisfiable.** A defect in **this document**, not in the design under test.

The 2026-07-26 override conflated these: its centring argument was a correct type-(B) observation and
it was used to license type-(A) behaviour.

### 11.3 The admissibility bar for a type-(B) claim

A claim that a gate here is unsatisfiable is **inadmissible without an exhibit**: a concrete design,
plus an argument that it is valid, plus a measurement showing it fails the gate. Argument alone is not
enough, and the person making it does not get to be the only reader. For the old gate that exhibit
exists and is decisive (2.1). On 2026-07-26 the argument had been produced and the exhibit had not,
and the run had already reported from the overridden cells. **The order is the entire content of the
word "pre-registration".**

**The asymmetry that distinguishes these gates from the one they replace, stated so it can be held
against this document.** To exhibit G-S as unsatisfiable one would have to produce a rule whose
partition and group provably do not depend on `y_obs` and which nonetheless fails an element-wise
identity test; that is a harness bug (F17), not a document defect. To exhibit the empirical gate as
unsatisfiable one would have to produce a rule satisfying Theorem 3.1's hypotheses that over-rejects
under a constructed true null; that contradicts the theorem, so it is either an error in 3.1 or a
harness bug (F16), both of which have their own halts. **If that reasoning is wrong, this paragraph is
where to attack the document.**

### 11.4 The consistency pass, executed before commit

- **(a) Satisfiability.** For every registered gate, exhibit at least one design that provably passes
  and at least one that provably fails. **Executed: Section 8**, for both gates, with measurements.
  Had this check existed in July, assert 9 could not have been written, because no passer exists.
- **(b) Cross-clause compatibility.** **Executed: Section 11.9.**
- **(c) Outcome coverage.** **Executed.** Per rule: {CERTIFIED, NO VERDICT, DISQUALIFIED by G-S,
  DISQUALIFIED by the empirical line, non-informative everywhere}, all covered by 9.1 to 9.4. Across
  rules: {all survive, some survive, none survive}, covered by 9.1 and 7.5. Controls: {both fail,
  neither fails empirically, one passes G-S}, covered by 9.4. Harness failures: Section 13. **The row
  the old document was missing, "this cell is not a valid test", is the row this entire document
  produces.**

### 11.5 The conflict tie-break rule, fixed now and deliberately biased against this project

> **When two registered clauses conflict, the clause that is MORE RESTRICTIVE of what may be reported
> wins.** If restrictiveness is ambiguous, the clause that would license the **smaller p-value** or the
> **more favourable conclusion** for this project **loses**. If still ambiguous, nothing from the
> affected arm is reported and the arm is re-registered and re-run.

The asymmetry is intentional and is not a claim to neutrality. The analyst's bias runs one way, so a
symmetric rule returns the decision to the analyst, which is the failure being fixed. The cost is that
it will sometimes discard a correct analysis; that cost is accepted here, in advance, in exchange for
removing discretion at the moment discretion is least trustworthy. Section 4.3 already applies it, at
this document's own expense, to WITHIN.

**Tested retrospectively on the case in the record.** Conflict: assert 9 against Section 7's
mechanism. Applying 11.5: assert 9 halts Sections 6.1 to 6.4 outright while the mechanism licenses
continuing, so assert 9 is more restrictive and **wins. C2 and C4 halt immediately.** That is the same
practical outcome the adjudication reached three days later on independent exchangeability grounds,
and it is reached **without the analyst having to be right about the statistics**. It would also have
blocked the verdict-flip claim that had to be retracted, which is the single most consequential error
in the record. Registered as a check on this rule: if any future conflict resolves under 11.5 to a
conclusion the project later has to retract, this clause is itself defective and must be re-registered.

### 11.6 One claim per amendment. Bundled amendments are void.

An amendment that advances two independent arguments for one departure is **rejected in full**, and
both arguments must be re-submitted separately. This is the specific rule that would have stopped the
2026-07-26 damage: the true centring argument carried the false exactness argument through because
they travelled together (2.3).

### 11.7 The asymmetric evidentiary standard

If an amendment runs in the direction that **flatters** the project, it additionally requires a
control using **no EEG**, exhibited at the actual parameters of the data, **and** a demonstration that
the amended gate still disqualifies FX and FX_blk. If it runs against the project's interest,
disclosure alone suffices. Every amendment carries a mandatory field stating **which direction it
flatters the project**; the absence of that field is itself a violation and must be reported as one.

### 11.8 The loophole closed explicitly

Section 11 is an amendment procedure and Section 7 forbids overrides. Without a clause distinguishing
them, an analyst facing a failed gate could declare an inconsistency, amend, and continue. That is the
override wearing a different hat, and it is the most likely way this document gets defeated.

> **Section 11 may amend this document only for a FUTURE run. It may never restore reportability
> within the run whose gate fired.** A gate failure in run k is permanent for run k. If the amendment
> is correct, the correct analysis is available at run k+1 at the cost of one re-run. If the amendment
> is motivated, the cost of the re-run is the price of finding that out.

Corollary: **declaring an inconsistency is never cheaper than accepting a gate failure.** Both halt
the affected arm; only the timing of the fix differs. This removes the incentive to manufacture an
inconsistency.

Further hard limits. An amendment may **remove a gate proven unsatisfiable by every valid design**, by
the standard of 2.4 and the bar of 11.3. An amendment may **never** remove, weaken, raise the
threshold of, or add an exception to a gate that the design in hand merely **fails**; nor make FX or
FX_blk eligible (5.2); nor introduce an override for Section 7; nor reorder 6.4 after any p-value is
known; nor re-enter the reduction ladder (5.5). **If Section 11 is invoked against Section 11, the
only permitted action is to report nothing from this run and write a fresh pre-registration.**
Recursion terminates in silence, not in discretion.

### 11.9 The compatibility table, check 11.4(b) executed

| clause pair constraining the same quantity | compatible? | why |
|---|---|---|
| G-S (Section 4) against the empirical gate (Section 5) | yes | Both can only disqualify. G-S is decisive for disqualification; the empirical arm can add a disqualification but never remove one. No outcome exists in which one demands reporting and another forbids it. |
| 6.2 "a failure disqualifies even in a non-live cell" against 6.1 "a non-live cell cannot certify" | yes | Liveness is a one-way condition on certification only. Stated in both directions in 6.2 so the asymmetry is explicit. |
| Section 8 (a passer exists) against 9.1's DISQUALIFIED rows and F15 | yes | The prediction and its contradiction both have rows. This is exactly the coverage the old Section 6 lacked. |
| 7.4 (conventional p binds) against 5.4 (thresholds) | yes | Which p binds is fixed before any threshold is evaluated. |
| 3.2 (this gate does not check power) against 5.6 (informativeness) | yes | 5.6 is registered with no reportability attached and prints that fact. |
| 5.5 (reduction ladder) against 5.4 (registered M) and A5 | yes | The ladder runs before any gate cell is computed and is frozen thereafter; A5 asserts the post-ladder M, which is printed first. |
| 4.3 (WITHIN disqualified) against 9.x | yes | WITHIN has no outcome row that reports a p, only a row that records the measurement. |
| **Section 7 (no override) against Section 11 (amendment)** | yes, **and this is the loophole that must be named** | 11.8. |
| any centring clause against anything | **not applicable** | No centring clause is registered anywhere. The strings "0.45", "0.55" and "centring" appear only in Sections 0, 2, 9 and 15, always attached to an explicit statement that no reportability depends on them. |

---

## 12. THE BINDING DEVICES, LISTED SO AN AUDITOR CAN CHECK THEM AGAINST THE ARTIFACT

- **B1. Monotonicity.** Every check can only **remove** a rule from eligibility. Nothing, in any world,
  at any marginal, under any tie-break, can restore one.
- **B2. Reportability is a boolean in code** (7.2), not a paragraph and not a judgement.
- **B3. A deterministic primary gate.** G-S has no Monte Carlo error and no estimator dependence, so
  its verdict cannot be moved by choosing a world.
- **B4. Per-cell positive-control calibration.** A cell may certify only if the matched known-broken
  rule demonstrably fails in it (6.1).
- **B5. Two-world, three-cell certification.** Closes the estimator-dependence trap (6.2).
- **B6. Registered negative controls with expected failure.** FX and FX_blk, never eligible whatever
  they do.
- **B7. Worst case governs** across tie-breaks, feature counts and marginals.
- **B8. An asymmetric conflict tie-break**, biased against the project by design and said to be (11.5).
- **B9. Direction-of-flattery named per outcome before the run** (0.5, 9), and a mandatory
  direction-of-flattery field in every amendment (11.7).
- **B10. Amendment is never cheaper than failure** (11.8), and bundled amendments are void (11.6).
- **B11. Verbatim pre-committed sentences** for the four decisive outcomes (Section 10).
- **B12. Registered-run assertion** (A5): a reduced-M run cannot produce a quotable number.
- **B13. The document can refute itself:** F9, F15, 9.2's last row, and 11.5's self-check.
- **B14. The ladder is frozen before gate evaluation** and every reduction is printed first (5.5).

---

## 13. ASSERTS AND FALSIFIERS

`[GATE]` disqualifies a rule. `[HARNESS]` means the code is broken: halt, exit non-zero, report
nothing from the run including numbers already printed and numbers unrelated to the failure.

**F1 `[GATE]` Structural failure.** A rule fails G-S on any of its 200 registered triples (4.1).

**F2 `[GATE]` Empirical over-rejection.** A rule exceeds the exact-binomial line of 5.4 in any cell at
alpha 0.05 or 0.10, live or not.

**F3 `[HARNESS]` The labels are not the recorded labels.** Class counts exactly 21/24, 24/21, 22/23
for subjects 1, 17, 19; run counts exactly 15/15/15 over runs 6, 10, 14. Exact arithmetic.

**F4 `[HARNESS]` The fast paths are not the estimators.** The direct majority-class path must equal
`cross_val_score(DummyClassifier(strategy="most_frequent"), ...)` and the hand-written LDA must equal
`cross_val_score(LinearDiscriminantAnalysis(), ...)`, each on **200** random label vectors **per
tie-break arm**, to `|difference| < 1e-9`. The repo checks 25 at one tie-break; the tie-break variants
are new code and need their own gate. Measured 2026-07-30: the LDA path agrees to 1.11e-16 on 40
vectors.

**F5 `[HARNESS]` H0 is not actually true.** Every drawn `y0` and every replicate must carry exactly the
cell's class counts. A resample rather than a permutation would silently change the estimand.

**F6 `[HARNESS]` The permutation machinery does not do what its name says.** (a) FX, FX_blk, WITHIN
replay `P0` on every replicate; (b) KF_free, KF_free_blk replay `PF` on every replicate; (c) RE and
RE_blk move off the observed partition on at least one replicate; (d) block shuffles preserve each
run's class counts; (e) within-fold shuffles preserve each fold's class counts.

**F7 `[HARNESS]` The pairing is not paired.** The shared label vector handed to RE, FX and KF_free at
outer draw `i`, replicate `j` must be element-wise identical; likewise the block vector across RE_blk,
FX_blk and KF_free_blk.

**F8 `[HARNESS]` Lattice.** Every W0, W1 and W2 score is an integer multiple of 1/45 to within 1e-9.
Off-lattice means unequal folds or a scorer that is not accuracy. This is the check that caught two
fabricated numbers in an earlier README.

**F9 `[HARNESS-CLASS, but reported not halted]` No cell is CALIBRATED.** Neither FX nor FX_blk exceeds
the line anywhere. The empirical arm has no exhibited failure mode, no rule may be CERTIFIED, and
9.4's row and 10's verbatim sentence apply. The run is **not** halted, because G-S stands on its own
and its verdicts remain reportable.

**F10 `[HARNESS]` Vacuity is not measured.** The non-degenerate fraction must be computed and printed
for **every** rule in **every** cell. A cell whose vacuity fraction is not printed may not contribute
to any verdict. A W0 cell reporting `0.0000` rejection without its vacuity fraction alongside is
exactly the error this document exists to correct.

**F11 `[HARNESS]` Check T is vacuous where it should not be.** The training-fold tie count is zero at a
marginal where 0.3ii predicts a nonzero rate (0.2032, 0.2026, 0.2767 at the three real marginals;
0.6000 for the re-stratified rule at subject 19). Means the tie-break arms are not reaching the code
path they claim to test.

**F12 `[HARNESS]` Ordering violation.** Any EEG p-value printed before the cell-status table and the
verdict list (6.3). The run is invalid regardless of its contents.

**F13 `[HARNESS]` W1's H0 is not H0.** Over the outer draws the maximum absolute point-biserial
correlation between any noise feature and the label must not exceed the
`1 - 0.01 / (d * M)` quantile of its null. Catches a seeding bug entangling the feature stream with
the label stream. A failure voids W1; W0 and W2 are unaffected because the three worlds share no
stream, and this is the one place where a partial report is permitted, permitted **here in advance**
rather than negotiated later.

**F14 `[HARNESS]` A Section 0.3 or 0.4 planning number fails to reproduce** within 3 Monte Carlo
standard errors at the registered M, or a predicted point mass is found non-degenerate. The planning
numbers are load-bearing for Sections 6 and 8, so a discrepancy means the gate was designed against a
measurement that does not exist.

**F15 `[DOCUMENT]` No candidate survives.** Section 8.2 claims a valid design exists and exhibits it.
If no candidate reaches CERTIFIED or NO VERDICT, Section 8 is refuted, that is a type-(ii) internal
inconsistency by 11.1, Section 11 governs, and nothing in Section 9 is reported.

**F16 `[HARNESS]` A rule that passes G-S is empirically disqualified.** Theorem 3.1 says this cannot
happen under hypothesis 1. A violation is a harness bug, an error in 3.1, or a failure of hypothesis 1
in a constructed world where it holds by construction. All three must be resolved before anything is
reported. They are partly distinguishable: a harness bug usually moves several rules together; a proof
error moves exactly the rule whose ancillarity claim is wrong.

**F17 `[HARNESS]` A rule that visibly depends on `y_obs` passes G-S.** FX, FX_blk or WITHIN returns
PASS. G-S is not implemented.

**Asserts A1 to A10**, in the code: A1 W0's `X` is exactly `np.zeros((45,1))`; A2 W1's feature stream
never touches the label stream, by seed provenance; A3 the fast paths are the estimators (F4); A4 each
rule receives its own observed value and the three differ on at least one outer draw; A5 `M` and `B`
equal their post-ladder registered values, printed first; A6 `REPORTABLE` is element-wise no wider
than `GATE_PASS`; A7 lattice (F8); A8 marginal and block preservation (F6); A9 `PF` is index-identical
across all draws and replicates; A10 the re-stratified partition really moves (F6c).

**Not registered as an assert, deliberately: null centring.** No band, no halt, no consequence. Every
null mean is printed with the literal annotation `descriptive only, no reportability depends on this`.

---

## 14. SCOPE: THE ONLY NUMBERS THIS PROJECT MAY QUOTE

Registered here because a validity document is exactly where an inflated headline would be laundered.

- The project's **only defensible headline** is **91.1% (41/45) on ONE subject (subject 1), n = 45
  trials**, stratified 5-fold CV, against **that subject's majority-class floor of 53.3% (24/45)**, at
  **p <= 0.001 reported as a bound and never as a measured value**.
- Across all 109 subjects the **median is 60.0%**.
- Cross-subject **LOSO is 59.4% (535/900)**, over the documented 20-subject budget, not the full 109.
- The complement ablation is **77.8% (35/45), one subject, n = 45**, and ten-seed **79.3%**.
- Observed values on the label-free partition, if a `PF` rule ever becomes the reported one:
  **42/45 (subject 1), 25/45 (subject 17), 21/45 (subject 19)**, each n = 45, each against that
  subject's own majority-class floor.
- **94% and 94.4% are retracted** and may not appear in any document, figure, caption, abstract, slide
  or commit message except with the word "retracted" in the same sentence.
- **Chance is the majority-class rate everywhere, never 50%.**
- A single-subject result at n = 45 is an **existence claim**, not a population estimate. This
  document adds no subjects, no trials and no generalisation. **A validity gate can only remove a
  p-value's standing; it can never raise an accuracy.** If RE is disqualified, the 91.1% survives as
  an accuracy and the `p <= 0.001` does not; the two are separable and must then be reported
  separately.
- Two rows in `README.md` are override-dependent and are **not** covered by the defensible-figures
  list: `README.md:39` ("p <= 9.999e-05") and `README.md:41` (Wilson at n_eff 34.0, [77.0%, 96.9%]).
  They are withdrawn independently of this run's outcome.

---

## 15. DELIVERABLE AND PROVENANCE

- Script: `eeg-motor-imagery/permutation_validity_gate.py`, house style. Module docstring saying why it
  exists and what it guards against; printed section headers; every rate printed with its M, its
  Monte Carlo standard error, its exact-binomial line and its cell status on the same line; every
  accuracy printed with its baseline and its k/n count; explicit statements of what each number does
  not show; asserts A1 to A10 and F1 to F17 in the code.
- W0 and W1 import no MNE and read no EEG. W2 reads subject 1's epoched artifact only.
- Stdout captured to `neuro-canon/runs/`, so `check_claims.py` and `check_provenance.py` can match
  against it. Every number in any write-up must be **printed by the script**, never interpolated from
  a literal.
- Any applied reduction from the 5.5 ladder is the **first line** of stdout.
- The cell-status table and the verdict list appear in stdout **before** any EEG p-value (F12), so the
  ordering is auditable from the artifact and not only from my account of it.
- The three checks of 11.4 are executed and committed **with** this document, before the run.
- Canon entries to update on completion: the assert-9 entry (withdrawn, with 2.1's unsatisfiability
  proof and 2.2's 1-in-6 count); the Section 2B entries (the exactness half already withdrawn
  2026-07-26; the centring half stands, but its "property of the arithmetic of a fixed partition"
  sentence is **narrowed** by Finding 2); the Section 2C entry (its `0.0000` rows for RE, RE_blk and
  WITHIN are **arithmetic, not evidence**, per Finding 1); and any entry citing the 0.45 to 0.55 band
  as a validity criterion.
- `OVERRIDE-RULING-2026-07-30.md` §1.6 refers to `neuro-canon/measurements/prereg-permutation-validity.md`
  as a document that exists and has been run. **It does not exist.** That sentence is forward-looking
  and must be corrected to a future tense, or to this file's name once committed. Recorded here
  because an unnoticed reference to a non-existent pre-registration is exactly the provenance defect
  the ruling itself was written to catch.
- No em dashes anywhere in the script, its stdout, or the write-up.

---

## 16. REGISTERED RISKS AND THE OBJECTIONS THAT SURVIVE

1. **The gates were designed after seeing the defect they catch.** They will catch it. That is not
   evidence the analyst would catch the next one.
2. **A pass is much weaker evidence than a fail.** Finding 3b bounds it: the empirical gate detects a
   known-invalid rule in about 13% of cells. Discreteness (7.4) biases it further toward passing.
   Every mention of a pass carries the resolution and the detection rate.
3. **G-S tests hypothesis 2 and not hypothesis 1.** If the trials are not exchangeable within run in
   the real data, every rule here can pass and every p can still be wrong. **This is the residual and
   it is not closed by this document.** It is the same class of defect as the one that was found, one
   level up, and I have no instrument for it.
4. **The positive control is one defect, tested twice.** FX and FX_blk share a single mechanism. A cell
   calibrated against them may have no power against a different defect, so "CALIBRATED" means
   "calibrated against the C2-type defect" and nothing wider. This is the sharpest surviving objection
   and I cannot close it: a gate demonstrated against one defect class is not a general validity
   certificate.
5. **W2's cost rests on a 128 ms figure I did not re-measure.** The probe and the ladder (5.5) handle
   it, but if the figure is badly wrong the arm that decides RE's certification is the arm that gets
   cut, and the ladder is deliberately ordered to cut W2 last and to substitute W2-lite rather than
   drop it.
6. **No world contains CSP except W2, and W2 is one subject.** A defect living specifically in CSP's
   refit inside training folds is invisible to W0 and W1, and is tested at n = 45 on one subject in
   W2. No claim about the committed pipeline may be stronger than that.
7. **W1 uses Gaussian noise; CSP's log-variance features are not Gaussian.** W1 tests "continuous
   statistic" rather than "the real feature distribution". W2 has the real distribution and poor
   resolution. Neither arm has both.
8. **Ten swept marginals plus three real ones is a grid, not a proof.** Finding 3b shows the defect is
   sharply marginal-dependent and coding-dependent. A rule that passes at all thirteen could fail
   between them or outside them.
9. **NO VERDICT is a comfortable place for this document to land.** If RE reaches NO VERDICT, the
   published p survives on a theorem and the empirical gate cost the project nothing, which is
   precisely the configuration in which a gate is least likely to be scrutinised. The mitigation is
   10's verbatim sentence, which states the emptiness in plain words, and 9.7, which puts it ahead of
   any pass in the reporting queue. It is a mitigation and not a fix.
10. **WITHIN is disqualified by a rule I chose, over an argument I did not construct** (4.3). If the
    conditional argument exists, this document destroys a valid rule. That is the registered cost of
    11.5's asymmetry and I accept it rather than adjudicate it here.
11. **This document is longer and more careful than the one it replaces, and that is not evidence it
    is right.** The previous document was also careful, was internally inconsistent, and its
    inconsistency was found by a run rather than by a reader. Section 11 assumes the same will happen
    here. The mitigation is that the load-bearing parts are in code (7.2, A6) rather than in prose, and
    B1 to B14 are listed in one place so an auditor can check them in one pass.
12. **Confirming my own retraction is weak evidence.** The C2/C4 defect is the prior. What is new here
    is G-S, the degeneracy finding, the tie-break finding, the estimator-dependence finding, the
    per-cell calibration machinery, and the fact that the check now runs **before** the analysis
    instead of after it. None of that makes the result independent of what I already knew.

---

## AMENDMENTS

*(None. Any amendment appends here with its date, the clauses involved, the reason, what each candidate
resolution would license, and which direction each candidate resolution flatters the project. The last
field is mandatory per 11.7. One claim per amendment; bundled amendments are void per 11.6. An
amendment binds a future run only, per 11.8.)*

---

## RESULTS

Run 2026-07-30. Script: `eeg-motor-imagery/validity_gate.py`, committed after this document.
Nothing above this line was edited after the run.

**Partial run, and the partial-ness is the first thing to say.** The primary gate of Section 4 ran in
full. The empirical backstop of Section 5 has **not been run**. No number below is a type-I
measurement, and this document may not be cited as having measured one.

### The primary gate, G-S (Section 4.1), 200 triples per rule per subject, three subjects, seed 42

| rule | predicted §4.2 | observed | mismatching triples |
|---|---|---|---|
| RE | PASS | PASS | 0 of 600 |
| RE_blk | PASS | PASS | 0 of 600 |
| KF_free | PASS | PASS | 0 of 600 |
| KF_free_blk | PASS | PASS | 0 of 600 |
| WITHIN | FAIL | FAIL | 600 of 600 |
| FX (= C2) | FAIL | FAIL | 600 of 600 |
| FX_blk (= C4) | FAIL | FAIL | 600 of 600 |

**Seven of seven verdicts match the prediction fixed in §4.2 before the script existed.** The three
failing rules fail on the *first* triple at every subject, which is what a structural defect looks
like: 600 of 600, not a rate. `F-GS` did not fire.

**What this establishes.** C2 and C4 are disqualified by a deterministic check costing two seconds,
with no EEG loaded, no null mean computed, no estimator chosen and no tie-break involved. The
withdrawal reached on 2026-07-26 by an adversarial pass and a Monte Carlo control is reached here by
a route that needs none of them. Run on 2026-07-25, this gate would have foreclosed the entire
Section 2B episode before the first p-value existed.

**What this does not establish, stated because a pass is the easiest thing to over-read.** G-S tests
hypothesis 2 of Theorem 3.1 and not hypothesis 1. If subject 1's trials are not exchangeable within
run, all four surviving rules pass this gate and every p from them is still wrong. That is risk 16.3
and it remains open. G-S also says nothing about the *magnitude* of any defect, and a rule passing it
is not thereby certified — the empirical arm that would do the certifying has not run.

### Outcomes fired

- **§9, the type-(B) outcome ("the old gate is unsatisfiable"):** confirmed by the §2.1 proof and not
  contradicted by anything here. `F14` did not fire; no exact rule centred inside 0.45-0.55.
- **No verdict is claimed for any rule.** Under §6.2 a rule is certified only by the empirical arm,
  which did not run. `RE_blk` remains the priority-list head by §6.4, on assumption strength, which
  is not a function of any p-value and was fixed before the run.

### What remains, and the honest reason it remains

Section 5's three worlds, the marginal sweep and the two-stage family-wise budget are a multi-hour
job whose cost is dominated by W2, the only arm that can certify the published rule and the one
Section 15 flags as costed on an un-re-measured 128 ms figure. It was not started rather than started
and truncated, because a truncated type-I sweep reported as a pass is precisely the failure this
document exists to prevent. The primary gate was chosen to run first for exactly this reason: it is
the arm that decides the question the override turned on, and it is the arm that costs nothing.

Nothing in `README.md` or `EXPLAINER.md` changes as a result of this run. The 91.1% headline never
depended on the disqualified cells, and the cells this gate disqualifies were already withdrawn.
