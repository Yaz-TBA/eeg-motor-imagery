"""The artifact control: refit on the channels where the EYES are loudest and see
whether the result survives there.

A scalp topography is not evidence. It is a picture of the model's weights, and
a model riding an eye-movement artifact will happily draw a picture too. The
only cheap control that actually bites is an ABLATION: refit the entire pipeline
on a channel subset that CANNOT see sensorimotor cortex, and check that the
accuracy collapses to the majority-class rate -- NOT to 50%. With 21 hands and
24 feet the do-nothing baseline is 53.3%, and a control that lands there has
failed correctly.

Six conditions, TWO splitters -- stratified 5-fold for (a)-(c), (e) and (f),
leave-one-run-out for (d), which is why (d) reports three per-fold values and
the others report five:

  (a) all 64 channels          -- reproduces the published headline
  (b) sensorimotor only        -- FC/C/CP strip. Should hold or improve.
  (c) frontopolar only         -- Fp/AF. This is where blinks and saccades are
                                  LOUDEST. If the decoder were reading the eyes,
                                  this is the subset that would keep working.
  (d) leave-one-run-out        -- all 64 channels, but folds are whole recording
                                  runs, so no fold can share a session-drift or
                                  electrode-settling trend with its training set.
  (e) sensorimotor DELETED     -- the 17-channel strip removed, the OTHER 47
                                  electrodes kept. THE NECESSITY ARM, added
                                  2026-07-25. See the second caveat below.
  (f) wide FC/C/CP DELETED     -- the 21-channel FC/C/CP block removed, 43 kept.
                                  Bounds the FC5/FC6/CP5/CP6 that (e) retains.

Seed 42 is primary, and (a)-(c), (e) and (f) also carry a ten-seed sweep over
range(10), because a seed-42 point difference is ONE quantized draw -- this file
already annotates its own +4.4 point row as "one draw, not an effect size" and
the same rule now binds the arm that does not flatter. The decision statistic
for (e) is a ten-seed mean gap AND an exact McNemar, both fixed in advance in
neuro-canon/measurements/prereg-complement-ablation.md, which also records that
the accuracy half of (e) had already been run once and was NOT blind.

WHY THIS README TABLE EXISTS AS A SCRIPT NOW. An earlier README published these
four numbers as 91.1 / 95.9 / 47.4 / 93.3 with no script behind them. Two of
them were arithmetically impossible: with 45 trials in five equal folds of 9,
overall accuracy is a count of correct trials over 45, so it can only land on
multiples of 1/45 = 2.222%. 95.9% and 47.4% are not on that lattice -- there is
no k with k/45 = 0.959. They were not measurements. This file replaces them.

A CAVEAT THAT SURVIVES THE ABLATION. The average reference is computed across
all 64 electrodes BEFORE any subset is picked, exactly as in decode_csp.py. So
the frontopolar channels are not hermetically sealed off from occipital or
central activity -- every channel carries -1/64 of every other. The ablation
therefore bounds the artifact contribution rather than eliminating it. Making
the subsets independent would mean re-referencing each subset separately, which
would no longer be the published pipeline. Bounding is the honest claim.

A SECOND CAVEAT, ON WHAT (c) IS AND IS NOT. Frontopolar-only is not "sensorimotor
cortex deleted." It KEEPS 8 of 64 electrodes and deletes the other 56 -- occipital,
parietal and temporal along with the central strip -- so its collapse is confounded
with an 8x cut in channel count and feature dimension. What (c) does test is the
OCULAR hypothesis: the subset where blinks and saccades are loudest does not carry
the result.

THE ARM THIS FILE USED TO DECLARE ABSENT, NOW BUILT (2026-07-25). Conditions (e)
and (f) delete the strip and keep the rest, which is the falsifiable form: "if the
decoder reads sensorimotor cortex then deleting sensorimotor cortex must break it."
Read what (e) IS before reading its number. It is "the 17-channel strip deleted,"
not "sensorimotor cortex deleted": SENSORIMOTOR does not contain FC5, FC6, CP5 or
CP6, so (e) keeps four peri-Rolandic electrodes, and (f) exists to bound exactly
that leak. (e) also keeps T8/T10/TP8, which is temporalis muscle territory that
NOTHING in this repo yet bounds, and POz/PO4/Oz, the peak of the strongest
retained CSP pattern. And the instrument limit comes BEFORE the number, not after
it: no channel-deletion experiment on a 64-channel scalp montage can falsify a
SOURCE hypothesis in either direction, because deleting the electrodes nearest a
source does not delete the source from the remaining ones. These are sensor-space
measurements and they license only sensor-space claims.

WHAT THIS SCRIPT WITHDREW, KEPT VISIBLE (2026-07-25):
  - "Four conditions, one seed, one splitter." False: the conditions block below
    constructs a StratifiedKFold AND a LeaveOneGroupOut, and hands the second to
    condition (d). "One seed" is true, and (d)'s three per-fold values against
    the others' five were the visible tell all along.
  - The opening line used to read "take the motor cortex away and see if the
    decoder dies," which describes an arm this script does not build. See the
    second caveat above.
  - The printed line "so the result is not a within-session drift artifact"
    claimed more than the design supports. Runs 6, 10 and 14 are three
    recordings from ONE session, so holding out a run removes drift shared
    inside a run but not a session-level trend running across all three.
  - "No condition in this script deletes the sensorimotor strip and retains the
    rest of the montage, so the falsifiable form 'if the decoder reads
    sensorimotor cortex then deleting sensorimotor cortex must break it' is NOT
    tested here and must not be attributed to this file." True from the day it
    was written until 2026-07-25, false from the edit that added conditions (e)
    and (f). Kept visible rather than deleted because that sentence was quoted
    elsewhere in the corpus as the evidence that the arm was missing, and a
    reader who meets it there has to be able to see that it was retired by
    BUILDING the arm rather than by rewording the disclosure.

WHAT THIS SCRIPT WITHDREW, KEPT VISIBLE (2026-07-26). All four came from an
adversarial pass, all four are defects in what the run SAID rather than in what
it computed, and no measured value below changes because of them.
  - "The complement re-referenced within its own 47 (LEAK REMOVED)." False. The
    secondary arm is provably the primary minus its own across-channel mean, a
    RANK-1 COMMON-MODE PROJECTION that drops the rank 47 to 46. The direction it
    deletes carries the average-referenced strip contribution AND the
    complement's own global component, so its cost is not assignable to the leak.
    Both the identity and the rank drop are now measured and asserted in the run.
  - The McNemar power line conditioned only on the observed n_disc, which is a
    random draw. The binding constraint is the DESIGN: in a paired 2x2 on the
    same 45 trials, b - c is identically the trial gap, so the smallest gap that
    can ever reach p < 0.05 is 6 trials = 13.3 points, while the registered G
    threshold is 10.0 points = 4.5 trials. The conjunctive rule cannot fire in
    between, at any discordant count. The pre-registration's stated reason for
    the rule ("at a gap of 10 or more points the McNemar should fire
    comfortably") is refuted by that arithmetic and was checkable when it was
    written. The pre-registration is NOT edited; the refutation is reported.
  - The McNemar was computed at seed 42 only, while G is a mean over range(10),
    so the two halves of one conjunctive rule were evaluated on DISJOINT seed
    sets. The test now runs on every sweep seed. The registered verdict stays at
    seed 42, where it was registered, and the spread is printed beside it.
  - "NO CHANNEL-COUNT CONTROL", declared as a live confound in this file and in
    the pre-registration and never run. It is now run, as arm 10, and it is
    retired for the 47-vs-64 comparison only.
"""

import matplotlib

matplotlib.use("Agg")

import os

# joblib workers are fresh processes that re-import mne at its default log level,
# so mne.set_log_level() below never reaches them. The environment does.
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")

import warnings

import numpy as np
import mne
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from scipy import stats

from common import FRONTOPOLAR, SENSORIMOTOR, make_clf, wilson_interval


mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --- the registered constants. Module level so load_data() and derive_channel_sets()
# --- can be imported and called without running the analysis. Changing any of these
# --- makes the run something other than the registered one.
SUBJECT = 1
RUNS = [6, 10, 14]
TMIN, TMAX = -1.0, 4.0
L_FREQ, H_FREQ = 8.0, 30.0
SEED = 42
SEEDS = range(10)          # the sweep the hostile pass used, so this replicates it
N_PERMUTATIONS = 1000
TOL = 1e-9


def load_data():
    """Load, average-reference, filter and epoch subject 1. Identical to decode_csp.py,
    so the numbers here are comparable with the headline.

    Returns (cropped_epochs, labels, groups, ch_names, n, n_hands, n_feet, majority).
    `groups` is the run index per epoch, which condition (d) uses as its fold variable.
    """
    edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
    raws = [mne.io.read_raw_edf(p, preload=True) for p in edf_paths]

    # Record each run's length BEFORE concatenating, so every epoch can be traced
    # back to the run it came from. concatenate_raws consumes the list in place and
    # the run boundary is not recoverable from the result.
    run_edges = np.cumsum([r.n_times for r in raws])

    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")

    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    epochs = mne.Epochs(
        raw, events, dict(hands=2, feet=3),
        tmin=TMIN, tmax=TMAX, picks="eeg", baseline=None, preload=True,
    )
    labels = epochs.events[:, -1]
    cropped = epochs.copy().crop(tmin=1.0, tmax=2.0)

    # np.searchsorted maps each epoch's onset sample to the run whose span contains
    # it: 0 for samples before the first edge, 1 before the second, 2 after.
    onsets = cropped.events[:, 0] - raw.first_samp
    groups = np.searchsorted(run_edges, onsets, side="right")

    n = len(labels)
    n_hands, n_feet = int((labels == 2).sum()), int((labels == 3).sum())
    majority = max(n_hands, n_feet) / n
    return cropped, labels, groups, cropped.ch_names, n, n_hands, n_feet, majority


def derive_channel_sets(ch_names):
    """COMPLEMENT, WIDE and NOT_WIDE, derived from the montage rather than typed.

    A hand-typed 47-channel list can silently stop being the complement of SENSORIMOTOR
    the moment either one is edited, and then the arm keeps its name while measuring
    something else. Returns (COMPLEMENT, WIDE, NOT_WIDE).
    """
    complement = [c for c in ch_names if c not in SENSORIMOTOR]
    # The WIDE set adds FC5, FC6, CP5 and CP6 to the strip: the full FC/C/CP block.
    # Derived the same way, from the name, for the same reason.
    wide = [c for c in ch_names
            if c[:2] in ("FC", "CP") or (c[0] == "C" and c[1] != "P")]
    not_wide = [c for c in ch_names if c not in wide]
    return complement, wide, not_wide


def main():
    """The analysis. Lives in a function so that importing this module for its
    helpers does not run a multi-minute experiment as a side effect."""

    # Pre-registered decision thresholds. Fixed in
    # neuro-canon/measurements/prereg-complement-ablation.md BEFORE this script ran.
    # One trial on n = 45 is 2.222 points, two trials are 4.444. A48 already refused
    # to call two trials a difference in the direction that flattered the project;
    # the same refusal is hard-coded here in the direction that does not.
    NOISE_BAND = 100 * 2 / 45   # 4.444 points = two trials
    G_THRESHOLD = 10.0          # more than twice the noise band, more than 4.5 trials
    ALPHA = 0.05

    # POST-REGISTRATION arm 10 (added 2026-07-26): the channel-count control that the
    # pre-registration declared at 2.4(d) and risk 6 and then did not run. 50 draws
    # puts the empirical-p resolution floor at 1/51 = 0.0196, which is below alpha;
    # 30 draws would put it at 1/31 = 0.0323, also below alpha but with less headroom
    # and a coarser null. The RNG seed is fixed here so the draws are reproducible.
    N_RANDOM_DRAWS = 50
    RANDOM_DELETION_SEED = 20260726

    # SENSORIMOTOR (the FC/C/CP strip straddling the central sulcus) and FRONTOPOLAR
    # (the Fp/AF ring over the orbits, the negative control) are defined in common.py
    # and imported at the top of this file, so the ablation conditions mean the same
    # thing here, in emg_proxy.py and in test_pipeline.py.
    #
    # make_clf and wilson_interval come from there too. wilson_interval used to be
    # copied in with a note that it was copied "byte-for-byte ... because an interval
    # computed two different ways is two intervals". That reasoning was right, and one
    # definition enforces it rather than asking each copy to stay in step.


    def most_lopsided_failing_split(n_disc, alpha=ALPHA):
        """At this discordant count, the most extreme split that STILL misses alpha.

    This is the honest statement of what the McNemar could and could not have
    detected at the n_disc it actually got, computed rather than asserted. The
    exact two-sided p falls monotonically as the split gets more lopsided, so
    the last split with p >= alpha is the most extreme failure available.
    Returns None when every split at this n_disc reaches alpha, and n_disc = 0
    is returned as None too, because there is no split to speak of.
    """
        if n_disc == 0:
            return None
        worst = None
        for b in range(n_disc // 2, n_disc + 1):
            p = float(stats.binomtest(b, n_disc, 0.5).pvalue)
            if p >= alpha:
                worst = (b, n_disc - b, p)
        return worst


    def min_detectable_trial_gap(n_max, alpha=ALPHA):
        """The smallest |b - c| that reaches p < alpha at ANY discordant count.

    THE DESIGN QUESTION, ASKED BEFORE THE RUN INSTEAD OF AFTER IT. The function
    above answers "what could this test have seen at the n_disc it got", which
    conditions on a random draw. This one conditions on nothing: it enumerates
    every (b, c) with b + c <= n_max and returns the smallest trial gap that ever
    clears alpha, together with the discordant count where it first does.

    Why it matters here. In a paired 2x2 on the same n trials, b - c is not free:
    it is identically (arm-1 correct) minus (arm-2 correct). So the accuracy gap
    in TRIALS fixes b - c exactly, and the only freedom left is c. The number
    this function returns is therefore a floor on the accuracy gap the McNemar
    half of the registered rule can detect, in trials, at any n_disc, ever.

    Returns (min_gap, n_disc_where_first_reached, b, c, p).
    """
        best = None
        for n_disc in range(1, n_max + 1):
            for b in range(0, n_disc + 1):
                c = n_disc - b
                p = float(stats.binomtest(b, n_disc, 0.5).pvalue)
                if p < alpha:
                    gap = abs(b - c)
                    if best is None or gap < best[0]:
                        best = (gap, n_disc, b, c, p)
        return best


    def forced_mcnemar_grid(k_a, k_b, n_total, alpha=ALPHA):
        """Every 2x2 compatible with two KNOWN marginals, and which of them fire.

    b - c = k_a - k_b is algebraic, not empirical: both arms score the same
    n_total trials, so the difference in correct counts IS the difference b - c.
    Enumerating c from 0 upward therefore enumerates every 2x2 the pair could
    possibly have produced. Returns [(b, c, n_disc, p, fires), ...].
    """
        d = k_a - k_b
        rows = []
        c = 0
        while c + max(d, 0) <= n_total and c + abs(d) + c <= n_total:
            b = c + d
            if b < 0:
                break
            n_disc = b + c
            if n_disc == 0:
                rows.append((b, c, 0, 1.0, False))
            else:
                p = float(stats.binomtest(b, n_disc, 0.5).pvalue)
                rows.append((b, c, n_disc, p, p < alpha))
            c += 1
        return rows


    # --- load + preprocess (identical to decode_csp.py, so the numbers are comparable) ---
    cropped, labels, groups, ch_names, n, n_hands, n_feet, majority = load_data()
    # The SECONDARY arm below re-references from the raw EDFs, so it needs the paths.
    # This is a cached path lookup, not a second read.
    edf_paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)

    print(f"\nSubject {SUBJECT}, runs {RUNS} (imagined both fists vs. both feet)")
    print(f"{n} trials ({n_hands} hands, {n_feet} feet) | "
          f"majority class = {majority:.1%}")
    print(f"Trials per run: " + ", ".join(
        f"run {r}={int((groups == i).sum())}" for i, r in enumerate(RUNS)))

    # --- the complement, as an exact set difference -------------------------------
    # All three are derived from the montage, never typed. The expression is the same
    # one the hostile pass used at its line 84, so the two implementations pick
    # identically or the disagreement is real rather than a transcription slip.
    COMPLEMENT, WIDE, NOT_WIDE = derive_channel_sets(ch_names)

    print("\n--- The deleted sets, as exact set differences ---")
    assert len(SENSORIMOTOR) == 17, f"SENSORIMOTOR is {len(SENSORIMOTOR)} channels, not 17"
    assert len(COMPLEMENT) == 47, f"COMPLEMENT is {len(COMPLEMENT)} channels, not 47"
    assert len(WIDE) == 21, f"WIDE is {len(WIDE)} channels, not 21"
    assert len(NOT_WIDE) == 43, f"NOT_WIDE is {len(NOT_WIDE)} channels, not 43"
    # A complement is a PARTITION or it is not a complement, and the name would then
    # be false. Both halves are checked: the union covers the montage and the
    # intersection is empty. Either one alone can pass on a set that overlaps.
    assert set(SENSORIMOTOR) | set(COMPLEMENT) == set(ch_names), (
        "SENSORIMOTOR + COMPLEMENT do not cover the montage")
    assert set(SENSORIMOTOR) & set(COMPLEMENT) == set(), (
        "SENSORIMOTOR and COMPLEMENT overlap, so COMPLEMENT is not a complement")
    assert set(WIDE) | set(NOT_WIDE) == set(ch_names) and not (set(WIDE) & set(NOT_WIDE))
    print(f"montage {len(ch_names)} ch | SENSORIMOTOR {len(SENSORIMOTOR)} + "
          f"COMPLEMENT {len(COMPLEMENT)} = {len(SENSORIMOTOR) + len(COMPLEMENT)}, "
          f"disjoint partition: asserted, not counted by hand")
    print(f"WIDE FC/C/CP {len(WIDE)} + NOT_WIDE {len(NOT_WIDE)} = "
          f"{len(WIDE) + len(NOT_WIDE)}, disjoint partition: asserted")
    print(f"COMPLEMENT ({len(COMPLEMENT)} ch, montage order): {' '.join(COMPLEMENT)}")
    # Printed so a disagreement with any other implementation of the same set
    # difference is diagnosable as a set difference AND as an ordering difference,
    # rather than argued about.
    print(f"COMPLEMENT keeps these four peri-Rolandic electrodes: "
          f"{[c for c in COMPLEMENT if c in ('FC5', 'FC6', 'CP5', 'CP6')]}")
    print(f"COMPLEMENT keeps this temporal ring (temporalis EMG territory, UNBOUNDED "
          f"in this repo): {[c for c in COMPLEMENT if c in ('T7', 'T8', 'T9', 'T10', 'TP7', 'TP8')]}")

    # --- the attainable-accuracy lattice -----------------------------------------
    # Every condition below tests each of the 45 trials exactly once (5 stratified
    # folds of 9, or 3 run-folds of 15), so the reported accuracy is a count of
    # correct trials divided by 45. It cannot take any other value. Quoting a number
    # off this lattice is a tell that it was never computed.
    print(f"\n--- Attainable-accuracy lattice (n = {n}) ---")
    print(f"Every fold scheme here is a PARTITION: each trial is tested once, so the")
    print(f"overall accuracy is k/{n} for integer k, i.e. steps of {1/n:.3%}.")
    print("Values near the headline: " + ", ".join(
        f"{k}/{n}={k/n:.1%}" for k in range(39, 45)))
    print("So 95.9% and 47.4% -- the two numbers the old README table carried --")
    print(f"are OFF this lattice ({round(0.959*n)}/{n} = {round(0.959*n)/n:.1%}, "
          f"{round(0.474*n)}/{n} = {round(0.474*n)/n:.1%}) and cannot have been measured.")

    # --- WHAT THE REGISTERED DECISION RULE CAN DETECT, COMPUTED BEFORE THE RUN ----
    # ADDED 2026-07-26, after an adversarial pass showed the registered rule cannot
    # fire across the lower third of its own band C. This block conditions on NOTHING
    # that the run produces. It is arithmetic on the design, and every input to it
    # (n = 45, alpha = 0.05, the paired 2x2 structure, G_THRESHOLD = 10.0) was fixed
    # before the first run. It could have been printed then. It was not, and that is
    # the defect being repaired.
    print(f"\n--- The McNemar half's detection floor, from the design alone ---")
    _min_gap, _mg_ndisc, _mg_b, _mg_c, _mg_p = min_detectable_trial_gap(n)
    print(f"In a paired 2x2 on the same {n} trials, b - c is IDENTICALLY (arm-1 "
          f"correct) - (arm-2 correct),")
    print(f"so the accuracy gap in trials fixes b - c and only c is free. Enumerating "
          f"every (b, c) with")
    print(f"b + c <= {n}: the smallest trial gap that reaches p < {ALPHA} at ANY "
          f"discordant count is")
    print(f"|b - c| = {_min_gap} trials (first at n_disc = {_mg_ndisc}, {_mg_b} vs "
          f"{_mg_c}, p = {_mg_p:.4f}) = {100*_min_gap/n:.1f} points.")
    print(f"THE REGISTERED G THRESHOLD IS {G_THRESHOLD:.1f} POINTS = "
          f"{G_THRESHOLD*n/100:.2f} TRIALS, WHICH IS BELOW THAT FLOOR.")
    print(f"The two halves of the registered two-part rule are therefore calibrated to")
    print(f"incommensurable effect sizes: single-seed gaps from {G_THRESHOLD:.1f} to "
          f"{100*_min_gap/n:.1f} points cannot")
    print(f"reach p < {ALPHA} at any n_disc, so in that range the conjunctive rule "
          f"cannot fire no")
    print("matter what the data does. This is a fact about the rule, not about the "
          "recording.")
    print(f"It also REFUTES the pre-registration's own stated justification for the "
          f"rule")
    print(f"(prereg section 6.2: 'At a gap of {G_THRESHOLD:.0f} or more points the "
          f"McNemar should fire")
    print(f"comfortably'). At exactly {G_THRESHOLD:.0f} points it cannot fire at all. "
          f"The pre-registration is")
    print("NOT edited to match this; it is refuted, and the refutation is recorded in "
          "its")
    print("RESULTS section as an outcome of the run.")
    print(f"CAVEAT ON SCOPE, stated so this is not read as more than it is: G is a "
          f"TEN-SEED MEAN")
    print(f"gap and the McNemar is computed on ONE partition, so the {100*_min_gap/n:.1f}-"
          f"point floor binds the")
    print("SEED-42 gap directly and G only through their correlation. The two halves "
          "of the")
    print("rule are not even evaluated on the same seed set. See the per-seed McNemar "
          "table below.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    logo = LeaveOneGroupOut()

    conditions = [
        ("all 64 channels", ch_names, skf, None),
        (f"sensorimotor only ({len(SENSORIMOTOR)} ch)", SENSORIMOTOR, skf, None),
        (f"frontopolar only ({len(FRONTOPOLAR)} ch)", FRONTOPOLAR, skf, None),
        (f"sensorimotor DELETED ({len(COMPLEMENT)} kept)", COMPLEMENT, skf, None),
        (f"wide FC/C/CP DELETED ({len(NOT_WIDE)} kept)", NOT_WIDE, skf, None),
        ("all 64, leave-one-run-out", ch_names, logo, groups),
    ]

    results = []
    per_trial = {}   # condition -> boolean "was this trial predicted correctly"
    for name, picks, cv, grp in conditions:
        missing = [c for c in picks if c not in ch_names]
        assert not missing, f"{name}: channels not in this montage: {missing}"
        data = cropped.copy().pick(picks).get_data(copy=False)

        # error_score="raise" so a CSP rank failure or a degenerate fold surfaces as
        # a traceback instead of a silent np.nan that quietly drags the mean down.
        scores = cross_val_score(make_clf(), data, labels, cv=cv, groups=grp,
                                 error_score="raise")
        pred = cross_val_predict(make_clf(), data, labels, cv=cv, groups=grp)
        n_correct = int((pred == labels).sum())
        results.append((name, data.shape[1], scores, n_correct))
        per_trial[name] = (pred == labels)

    # Look results up BY NAME, not by index. The old code read results[0] through
    # results[3] and would have kept running, silently reading the wrong condition,
    # if anyone ever inserted a row above the one they wanted.
    by_name = {r[0]: r for r in results}
    ALL64 = "all 64 channels"
    SMC = f"sensorimotor only ({len(SENSORIMOTOR)} ch)"
    FP = f"frontopolar only ({len(FRONTOPOLAR)} ch)"
    COMP = f"sensorimotor DELETED ({len(COMPLEMENT)} kept)"
    NWIDE = f"wide FC/C/CP DELETED ({len(NOT_WIDE)} kept)"
    LORO = "all 64, leave-one-run-out"

    print(f"\n--- Ablation (CSP+LDA, seed {SEED}, same pipeline throughout) ---")
    print(f"{'condition':<32} {'ch':>3} {'acc':>7} {'correct':>9}  per-fold")
    for name, n_ch, scores, n_correct in results:
        per_fold = " ".join(f"{s:.2f}" for s in scores)
        print(f"{name:<32} {n_ch:>3} {n_correct/n:>6.1%} {f'{n_correct}/{n}':>9}  {per_fold}")

    # The mean of the per-fold scores equals the pooled count only when the folds are
    # equal-sized. They are here (9, 9, 9, 9, 9 and 15, 15, 15), so the two agree --
    # but assert it rather than assume it, because an unequal split would make the
    # fold-mean a number that is NOT on the k/45 lattice while still looking like one.
    for name, _, scores, n_correct in results:
        assert abs(scores.mean() - n_correct / n) < 1e-9, (
            f"{name}: fold-mean {scores.mean():.4f} != pooled {n_correct}/{n}. "
            "Folds are unequal, so the fold-mean is not the accuracy."
        )
        assert abs(n_correct - round(n_correct)) < 1e-9
    print(f"\nAll {len(results)} accuracies land on the k/{n} lattice, as they must.")

    # --- controls, before anything is read off the table --------------------------
    # Both of these run BEFORE the ten-seed sweep and the permutation test, because a
    # dead pipeline or a live negative control makes every number below unreadable
    # and there is no reason to spend three minutes discovering that.
    maj_correct = max(n_hands, n_feet)
    _all64_scores = by_name[ALL64][2]
    # TOL is NOT decoration. A majority-class DummyClassifier scores exactly 24/45,
    # and floating point puts it 1.1e-16 ABOVE a bare `> majority`, so a bare
    # comparison passes a model that has learned nothing. Same trap, same 1e-9
    # tolerance, as decode_csp.py lines 111-118 and sweep_subjects.py.
    assert _all64_scores.mean() > majority + TOL, (
        f"POSITIVE CONTROL FAILED: all-64 scored {_all64_scores.mean():.1%}, which "
        f"does not beat the majority-class rate {majority:.1%}. The pipeline is not "
        "decoding anything and no other row in this table means what it says.")
    _fp_off = abs(by_name[FP][3] - maj_correct)
    assert _fp_off <= 3, (
        f"NEGATIVE CONTROL FAILED: frontopolar-only is {_fp_off} trials off the "
        f"majority rate ({by_name[FP][3]}/{n} vs {maj_correct}/{n}). The control has "
        "come alive; this run is not interpretable.")
    print(f"Positive control: all 64 at {by_name[ALL64][3]}/{n} beats majority "
          f"{maj_correct}/{n} by more than TOL={TOL:g}. PASS.")
    print(f"Negative control: frontopolar-only is {_fp_off} trial"
          f"{'' if _fp_off == 1 else 's'} off the majority rate, within the "
          f"registered 3. PASS.")

    # --- ten seeds, because one seed is one draw ----------------------------------
    # THE SEED-42 POINT DIFFERENCE IS NOT THE DECISION STATISTIC, and this block is
    # why. evaluate_honestly.py section 6 sweeps 100 seeds and finds the all-64
    # headline moving several points on split placement alone; a single quantized
    # draw from that distribution cannot carry a necessity claim in either direction.
    # range(10) is not chosen here -- it is the sweep the hostile pass ran, which is
    # what makes this a replication of that number rather than a fresh one.
    print(f"\n--- Ten-seed sweep, seeds {list(SEEDS)} (leave-one-run-out has no seed "
          f"to sweep, so it is absent) ---")
    print(f"{'condition':<32} {'ch':>3} {'seed42':>8} {'10-seed':>8}  {'range':>16}")
    sweeps = {}
    for name, picks, cv, grp in conditions:
        if grp is not None:                       # leave-one-run-out: no shuffle seed
            continue
        data = cropped.copy().pick(picks).get_data(copy=False)
        vals = np.array([
            cross_val_score(make_clf(), data, labels,
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=s),
                            error_score="raise").mean()
            for s in SEEDS
        ])
        # Every INDIVIDUAL seed is a k/45 count and must land on the lattice. Their
        # MEAN is an average of ten such counts and is NOT on the lattice, which is
        # correct and is stated here so nobody reads an off-lattice sweep mean as the
        # defect the old README table had.
        for s, v in zip(SEEDS, vals):
            assert abs(v * n - round(v * n)) < 1e-9, (
                f"{name}: seed {s} scored {v:.6f}, off the k/{n} lattice")
        sweeps[name] = vals
        n_ch = data.shape[1]
        print(f"{name:<32} {n_ch:>3} {by_name[name][3]/n:>7.1%} {vals.mean():>7.1%}  "
              f"[{vals.min():>5.1%}, {vals.max():>5.1%}]")
    print(f"Each of the {len(SEEDS)} per-seed values above is on the k/{n} lattice "
          f"(asserted). Their MEAN is not, and should not be.")

    # --- the McNemar on EVERY sweep seed, not only on seed 42 ---------------------
    # ADDED 2026-07-26. The registered rule is conjunctive: G AND the McNemar. G is a
    # mean over range(10). The McNemar was computed at seed 42, which is NOT a member
    # of range(10). The two halves of one rule were being evaluated on DISJOINT seed
    # sets, and the half that decided the verdict rested on a single partition. The
    # per-trial predictions needed to fix that were already being computed by
    # cross_val_predict at seed 42 and cost one extra call per arm per seed here.
    # This does NOT change the registered verdict, which is defined at seed 42 and
    # stays there. It measures how much of that verdict is the seed.
    print(f"\n--- Exact McNemar on EVERY sweep seed (all 64 vs. the complement) ---")
    _X_all_for_mcn = cropped.copy().pick(ch_names).get_data(copy=False)
    _X_comp_for_mcn = cropped.copy().pick(COMPLEMENT).get_data(copy=False)


    def mcnemar_at(seed):
        """(k_all, k_comp, b, c, n_disc, p) for one StratifiedKFold seed."""
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        _pa = cross_val_predict(make_clf(), _X_all_for_mcn, labels, cv=_cv) == labels
        _pc = cross_val_predict(make_clf(), _X_comp_for_mcn, labels, cv=_cv) == labels
        _b = int((_pa & ~_pc).sum())
        _c = int((~_pa & _pc).sum())
        _nd = _b + _c
        _p = 1.0 if _nd == 0 else float(stats.binomtest(_b, _nd, 0.5).pvalue)
        return int(_pa.sum()), int(_pc.sum()), _b, _c, _nd, _p


    print(f"{'seed':>5} {'all64':>8} {'comp':>8} {'gap(tr)':>8} {'b':>3} {'c':>3} "
          f"{'n_disc':>7} {'p':>8}")
    sweep_mcn = {}
    for s in SEEDS:
        ka, kc, b_, c_, nd_, p_ = mcnemar_at(s)
        sweep_mcn[s] = (ka, kc, b_, c_, nd_, p_)
        print(f"{s:>5} {f'{ka}/{n}':>8} {f'{kc}/{n}':>8} {ka - kc:>8} {b_:>3} {c_:>3} "
              f"{nd_:>7} {p_:>8.4f}{'  <alpha' if p_ < ALPHA else ''}")
    _sweep_ps = np.array([sweep_mcn[s][5] for s in SEEDS])
    _n_fire = int((_sweep_ps < ALPHA).sum())
    print(f"median p over the {len(SEEDS)} registered sweep seeds = "
          f"{np.median(_sweep_ps):.4f}; {_n_fire} of {len(SEEDS)} reach p < {ALPHA}.")
    print(f"Every row obeys the algebra above: b - c equals the trial gap exactly, in "
          f"all {len(SEEDS)} rows.")
    for s in SEEDS:
        ka, kc, b_, c_, _nd, _p = sweep_mcn[s]
        assert b_ - c_ == ka - kc, f"seed {s}: b - c = {b_-c_} but gap = {ka-kc}"

    # --- SECONDARY: the complement re-referenced inside its own 47 ----------------
    # WHAT THIS ARM PROVABLY IS, CORRECTED 2026-07-26. It used to be described here
    # and in the pre-registration's RESULTS as "the leak removed". It is not. Write
    # out the algebra: the primary is x_i - m64(x), the secondary is x_i - m47(x), so
    # secondary = primary - m47(primary), which is the primary MINUS ITS OWN
    # ACROSS-CHANNEL MEAN. That is a RANK-1 COMMON-MODE PROJECTION: it deletes one
    # spatial dimension, the uniform direction over the 47, and the rank of the data
    # drops from 47 to 46. Both facts are measured and asserted below rather than
    # argued.
    #
    # The time course it deletes is m64(x) - m47(x) = -(17/64) * (m47(x) - m17(x)),
    # which MIXES the average-referenced strip contribution -(17/64) m17 with
    # (17/64) of the complement's OWN global component m47. So the points this arm
    # costs cannot be assigned to the strip leak alone. A clean arm would project out
    # only an estimate of the strip's common mode and leave the complement's intact.
    # That arm is not built here and is not claimed.
    def build_reref_within(picks):
        """Rebuild from the EDFs with the average reference taken over PICKS alone."""
        _raw = mne.concatenate_raws(
            [mne.io.read_raw_edf(p, preload=True) for p in edf_paths])
        eegbci.standardize(_raw)
        _raw.set_montage("standard_1005")
        _raw.pick(picks)                    # pick BEFORE referencing: that is the point
        _raw.set_eeg_reference("average", projection=False)
        _raw.filter(L_FREQ, H_FREQ, fir_design="firwin", skip_by_annotation="edge")
        _ev, _ = mne.events_from_annotations(_raw, event_id=dict(T1=2, T2=3))
        _ep = mne.Epochs(_raw, _ev, dict(hands=2, feet=3), tmin=TMIN, tmax=TMAX,
                         picks="eeg", baseline=None, preload=True)
        return (_ep.copy().crop(tmin=1.0, tmax=2.0).get_data(copy=False),
                _ep.events[:, -1])


    print(f"\n--- SECONDARY: complement re-referenced within its own "
          f"{len(COMPLEMENT)} (ONE SPATIAL DIMENSION DELETED, not the leak removed) ---")
    X_ref, y_ref = build_reref_within(COMPLEMENT)
    assert np.array_equal(y_ref, labels), "re-referenced arm did not reproduce the labels"
    assert X_ref.shape[1] == len(COMPLEMENT)

    # WHAT THE MANIPULATION IS, MEASURED. Two facts, both checked here so neither has
    # to be taken on the author's word, and both printed so no reader meets the 6.0
    # points without meeting them.
    _X_comp_primary = cropped.copy().pick(COMPLEMENT).get_data(copy=False)
    _demeaned = _X_comp_primary - _X_comp_primary.mean(axis=1, keepdims=True)
    _reref_resid = float(np.abs(X_ref - _demeaned).max())
    _scale = float(np.abs(_X_comp_primary).max())
    _rank_primary = int(np.linalg.matrix_rank(
        _X_comp_primary.transpose(1, 0, 2).reshape(len(COMPLEMENT), -1)))
    _rank_ref = int(np.linalg.matrix_rank(
        X_ref.transpose(1, 0, 2).reshape(len(COMPLEMENT), -1)))
    print(f"IDENTITY CHECK: max |secondary - (primary - its own across-channel mean)| "
          f"= {_reref_resid:.2e}")
    print(f"      against a data scale of {_scale:.2e}, i.e. "
          f"{_reref_resid/_scale:.1e} relative. The secondary IS the primary")
    print(f"      with the uniform spatial direction removed. Nothing else changes.")
    print(f"RANK CHECK: rank drops {_rank_primary} -> {_rank_ref}. ONE spatial "
          f"dimension deleted, out of {len(COMPLEMENT)}.")
    assert _reref_resid / _scale < 1e-12, (
        "the secondary arm is NOT the primary minus its own across-channel mean, so "
        "the rank-1 description printed above is wrong and must be rewritten")
    assert _rank_ref == _rank_primary - 1, (
        f"expected a rank-1 projection ({_rank_primary} -> {_rank_primary - 1}), "
        f"got {_rank_primary} -> {_rank_ref}")
    print(f"WHAT THAT DIRECTION CARRIES: m64(x) - m47(x) = -(17/64)(m47(x) - m17(x)).")
    print(f"      It mixes the average-referenced strip contribution -(17/64)*m17 "
          f"WITH (17/64) of the")
    print(f"      complement's OWN global component m47. The cost below is therefore "
          f"NOT assignable")
    print(f"      to the strip leak alone, and this arm does not identify a causal "
          f"contribution.")
    ref_scores = cross_val_score(make_clf(), X_ref, labels,
                                 cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                    random_state=SEED),
                                 error_score="raise")
    ref_correct = int(round(ref_scores.mean() * n))
    assert abs(ref_scores.mean() - ref_correct / n) < 1e-9
    ref_vals = np.array([
        cross_val_score(make_clf(), X_ref, labels,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=s),
                        error_score="raise").mean()
        for s in SEEDS
    ])
    comp_seed42 = by_name[COMP][3]
    comp_sweep = sweeps[COMP]
    print(f"average ref over all 64, then pick 47 (PRIMARY): seed42 "
          f"{comp_seed42/n:.1%} ({comp_seed42}/{n})  10-seed {comp_sweep.mean():.1%} "
          f"[{comp_sweep.min():.1%}, {comp_sweep.max():.1%}]")
    print(f"average ref over the 47 alone   (SECONDARY): seed42 "
          f"{ref_correct/n:.1%} ({ref_correct}/{n})  10-seed {ref_vals.mean():.1%} "
          f"[{ref_vals.min():.1%}, {ref_vals.max():.1%}]")
    _ref_delta = 100 * (ref_vals.mean() - comp_sweep.mean())
    print(f"difference, secondary minus primary: "
          f"{_ref_delta:+.1f} points over ten seeds "
          f"(two trials = {NOISE_BAND:.3f} points, so anything inside that is nothing).")
    # THE PROJECT'S OWN EVIDENTIARY STANDARD, APPLIED TO THE ARM THAT WAS EXEMPT FROM
    # IT. Until 2026-07-26 this arm's 6.0 points was written up as established new
    # information ("removing the leak costs the complement 6.0 points, so the primary
    # complement score is inflated by the leak") while 14.7 points WITH a permutation
    # test behind it was written up as not established. That is two standards. The
    # rule this script already hard-codes for G is applied here instead.
    _ref_trials = abs(_ref_delta) * n / 100
    print(f"UNDER THE RULE THIS SCRIPT ALREADY HARD-CODES: {abs(_ref_delta):.1f} "
          f"points is {_ref_trials:.1f} trials,")
    print(f"      above the two-trial band ({NOISE_BAND:.3f}) and below the "
          f"{G_THRESHOLD:.1f}-point threshold, with NO")
    print(f"      confidence interval, NO significance test and NO registered "
          f"threshold of its own")
    print(f"      (the pre-registration registered this arm as a measurement and "
          f"registered no")
    print(f"      interpretation rule for it). So it is SUGGESTED, NOT ESTABLISHED, "
          f"on exactly the")
    print(f"      standard the {G_THRESHOLD:.1f}-point rule imposes on G. Any reading "
          f"stronger than that is")
    print(f"      post hoc. Any reading that assigns these points to the strip leak "
          f"specifically is")
    print(f"      also unidentified, per the decomposition printed above.")

    # --- permutation test on the complement ---------------------------------------
    print(f"\n--- Permutation test on the complement ({len(COMPLEMENT)} ch), "
          f"{N_PERMUTATIONS} shuffles, seed {SEED} ---")
    X_comp = cropped.copy().pick(COMPLEMENT).get_data(copy=False)
    perm_obs, perm_null, perm_p = permutation_test_score(
        make_clf(), X_comp, labels, scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_permutations=N_PERMUTATIONS, random_state=SEED, n_jobs=-1,
    )
    assert abs(perm_obs * n - round(perm_obs * n)) < 1e-9
    assert abs(round(perm_obs * n) - comp_seed42) < 1e-9, (
        f"permutation_test_score's observed value ({round(perm_obs*n)}/{n}) disagrees "
        f"with cross_val_score's ({comp_seed42}/{n}) on identical folds")
    # sklearn computes p = (C + 1)/(n + 1), so 1/1001 is the RESOLUTION FLOOR of a
    # 1000-shuffle test and not a measurement. Printing "0.0010" invites reading the
    # limit of the instrument as a result. Same rule as decode_csp.py.
    p_floor = 1.0 / (N_PERMUTATIONS + 1)
    perm_p_str = (f"<= {p_floor:.3f}" if perm_p <= p_floor + 1e-12
                  else f"=  {perm_p:.4f}")
    print(f"observed {perm_obs:.1%} ({round(perm_obs*n)}/{n})   p {perm_p_str}")
    print(f"null mean {perm_null.mean():.1%} +/- {perm_null.std():.1%}, "
          f"max {perm_null.max():.1%}")
    null_off_50 = abs(100 * perm_null.mean() - 50.0)
    print(f"null centred {null_off_50:.1f} points off 50%, inside the registered 5. "
          f"{'PASS' if null_off_50 <= 5.0 else 'FIRED'}")
    assert null_off_50 <= 5.0, (
        f"permutation null centred at {perm_null.mean():.1%}, more than 5 points off "
        "50%. The null is mis-specified and the p does not mean what it looks like.")

    # --- Wilson 95% CI on the complement's seed-42 count --------------------------
    w_lo, w_hi = wilson_interval(comp_seed42, n)
    print(f"\n--- Wilson 95% CI on the complement, z = 1.96 ---")
    print(f"{comp_seed42}/{n} = {comp_seed42/n:.1%}, Wilson [{w_lo:.1%}, {w_hi:.1%}], "
          f"width {100*(w_hi-w_lo):.1f} points, against the majority floor "
          f"{majority:.1%} ({maj_correct}/{n}).")
    print(f"An interval {100*(w_hi-w_lo):.1f} points wide is what n = {n} buys. It is "
          f"printed so the point estimate is never read alone.")

    # --- exact McNemar: all 64 against the complement, paired on the same folds ----
    # THE DECISION STATISTIC, and the blind half of it: nothing in the corpus carries
    # this comparison. Paired per-trial, because the two arms score the SAME 45
    # trials on the SAME seed-42 partition, and an unpaired comparison throws that
    # pairing away.
    print(f"\n--- Exact McNemar: all 64 vs. the complement, seed {SEED} folds, "
          f"paired per trial ---")
    c_all, c_comp = per_trial[ALL64], per_trial[COMP]
    both = int((c_all & c_comp).sum())
    only_all = int((c_all & ~c_comp).sum())          # b
    only_comp = int((~c_all & c_comp).sum())         # c
    neither = int((~c_all & ~c_comp).sum())
    n_disc = only_all + only_comp
    assert both + only_all + only_comp + neither == n
    assert both + only_all == by_name[ALL64][3] and both + only_comp == comp_seed42
    print(f"2x2: both correct {both}, all-64 only (b) {only_all}, "
          f"complement only (c) {only_comp}, neither {neither}. Sums to {n}.")
    if n_disc == 0:
        mcn_p = 1.0
        print("discordant pairs: 0. There is no test to run; p is 1.0 by definition.")
    else:
        mcn_p = float(stats.binomtest(only_all, n_disc, 0.5).pvalue)
        print(f"discordant pairs n_disc = {n_disc} ({only_all} vs {only_comp}), "
              f"exact two-sided p = {mcn_p:.4f}")
    # THE P IS QUOTED WITH ITS DISCORDANT COUNT OR NOT AT ALL. A56: at 10 discordant
    # pairs a one-trial shift moves the exact p from 0.109 to 0.754, a factor of about
    # seven. The line below is what this test could and could not have detected at the
    # n_disc it actually got, computed rather than asserted.
    worst = most_lopsided_failing_split(n_disc)
    if worst is None and n_disc > 0:
        print(f"At n_disc = {n_disc}, EVERY split reaches p < {ALPHA}. The test could "
              "not have returned a null result at this discordant count.")
    elif worst is not None:
        wb, wc, wp = worst
        print(f"POWER, at n_disc = {n_disc}: the most lopsided split that would STILL "
              f"have missed p < {ALPHA} is {wb} vs {wc} (p = {wp:.4f}). Anything less "
              f"lopsided than that is undetectable by this test at this n_disc.")
    else:
        print(f"At n_disc = 0 there is no split and no power to report.")

    # THE POWER LINE ABOVE CONDITIONS ON A RANDOM DRAW, AND THIS ONE DOES NOT.
    # n_disc = 8 is itself an outcome. The two MARGINALS, however, were known before
    # the pre-registration was written (they are printed in its section 0 and in its
    # arm table), and they fix b - c algebraically. So the set of 2x2 tables this
    # comparison could possibly have produced at seed 42 was enumerable in advance.
    print(f"MARGINAL-FORCED POWER, which conditions on nothing the run produced: at "
          f"seed {SEED} the two")
    print(f"      marginals are {by_name[ALL64][3]}/{n} and {comp_seed42}/{n}, so "
          f"b - c = {by_name[ALL64][3]} - {comp_seed42} = {by_name[ALL64][3]-comp_seed42} "
          f"is forced. Enumerating c:")
    _grid = forced_mcnemar_grid(by_name[ALL64][3], comp_seed42, n)
    for _b, _c, _nd, _p, _fires in _grid[:5]:
        print(f"        c = {_c}, b = {_b}, n_disc = {_nd}, p = {_p:.4f}"
              f"{'   FIRES' if _fires else ''}")
    _n_possible = sum(1 for r in _grid if r[4])
    print(f"      {_n_possible} of the {len(_grid)} attainable configurations reaches "
          f"p < {ALPHA}, and it is c = 0.")
    print(f"      The observed c = {only_comp}. So at these marginals the test was "
          f"near-predetermined to miss,")
    print(f"      and that was checkable with two lines of arithmetic at "
          f"prereg-writing time.")

    # --- POST-REGISTRATION ARM 10: the channel-count control the prereg declared ---
    # ADDED 2026-07-26, AFTER the answer was visible. Disclosed in full rather than
    # presented as blind. An adversarial pass ran this control, reported that it comes
    # out decisively, and told this script to add it. So it is NOT a prediction that
    # was tested; it is a declared confound that was finally measured, by a script
    # that already knew roughly what it would say. Read it as that.
    #
    # WHAT IT ANSWERS. Pre-registration 2.4(d) and registered risk 6, and this
    # script's own caveat (v), all declared: "47 vs 17 vs 64 channels at a fixed
    # n_components=4 are different CSP estimation problems", and then did not run the
    # control. That declared confound is exactly the alternative explanation for the
    # complement arm's deficit: maybe deleting ANY 17 channels costs 14.7 points.
    # This measures it. Delete 17 channels AT RANDOM, keep 47, run the identical
    # pipeline over the identical ten seeds.
    #
    # WHAT NULL IT TESTS, stated so it is not confused with the registered one. This
    # asks "is the strip special among 17-channel deletions". The registered McNemar
    # asks "does all-64 beat the complement on paired per-trial predictions at seed
    # 42". They are different questions and this one does NOT substitute for the
    # registered decision rule, which stays exactly as registered.
    print(f"\n--- POST-REGISTRATION arm 10: random-17-channel-deletion null "
          f"({N_RANDOM_DRAWS} draws) ---")
    print(f"Registered confound 2.4(d) / risk 6 / caveat (v), measured instead of "
          f"declared. NOT BLIND:")
    print(f"added 2026-07-26 with the answer already visible, on an adversarial "
          f"pass's instruction.")
    _rng = np.random.default_rng(RANDOM_DELETION_SEED)
    _null_means, _null_G = [], []
    _G_obs = 100 * (sweeps[ALL64].mean() - comp_sweep.mean())
    for _d in range(N_RANDOM_DRAWS):
        _drop = set(_rng.choice(len(ch_names), size=len(SENSORIMOTOR),
                                replace=False).tolist())
        _keep = [c for i, c in enumerate(ch_names) if i not in _drop]
        assert len(_keep) == len(COMPLEMENT), (
            f"random deletion kept {len(_keep)}, not {len(COMPLEMENT)}")
        _Xd = cropped.copy().pick(_keep).get_data(copy=False)
        _v = np.array([
            cross_val_score(make_clf(), _Xd, labels,
                            cv=StratifiedKFold(n_splits=5, shuffle=True,
                                               random_state=s),
                            error_score="raise").mean()
            for s in SEEDS
        ])
        for _s, _val in zip(SEEDS, _v):
            assert abs(_val * n - round(_val * n)) < 1e-9, (
                f"random draw {_d}, seed {_s} scored {_val:.6f}, off the k/{n} lattice")
        _null_means.append(_v.mean())
        _null_G.append(100 * (sweeps[ALL64].mean() - _v.mean()))
    _null_means = np.array(_null_means)
    _null_G = np.array(_null_G)
    _at_or_beyond = int((_null_G >= _G_obs - 1e-9).sum())
    _below_comp = int((_null_means <= comp_sweep.mean() + 1e-9).sum())
    _emp_p = (_at_or_beyond + 1) / (N_RANDOM_DRAWS + 1)
    print(f"random-47 ten-seed mean: {_null_means.mean():.1%}, range "
          f"[{_null_means.min():.1%}, {_null_means.max():.1%}]")
    print(f"random-47 G null: mean {_null_G.mean():+.1f} points, range "
          f"[{_null_G.min():+.1f}, {_null_G.max():+.1f}], sd {_null_G.std(ddof=1):.1f}")
    print(f"observed G (strip deleted) = {_G_obs:+.1f} points. Draws at or beyond it: "
          f"{_at_or_beyond}/{N_RANDOM_DRAWS}.")
    print(f"Draws landing at or below the complement's {comp_sweep.mean():.1%}: "
          f"{_below_comp}/{N_RANDOM_DRAWS}.")
    print(f"Empirical p = (C+1)/(N+1) = {_emp_p:.4f}, whose RESOLUTION FLOOR is "
          f"1/{N_RANDOM_DRAWS + 1} = {1/(N_RANDOM_DRAWS+1):.4f}.")
    print(f"observed G sits {(_G_obs - _null_G.mean())/_null_G.std(ddof=1):.1f} null "
          f"SDs above the null mean.")
    print("READING, and its limits. Deleting 17 channels costs essentially nothing on")
    print("average; deleting THE STRIP costs far more than any random deletion "
          "reached. So the")
    print("declared channel-count confound does NOT explain the complement arm's "
          "deficit, and")
    print("caveat (v) is retired FOR THE 47-vs-64 COMPARISON ONLY. It is NOT retired "
          "for the")
    print(f"{len(SENSORIMOTOR)}-channel or {len(FRONTOPOLAR)}-channel arms: this "
          f"control deletes 17 and keeps 47, so it says")
    print("nothing about how a 17-channel or 8-channel CSP estimation problem "
          "differs from a")
    print(f"64-channel one. What it does NOT do: it does not substitute for the "
          f"registered")
    print("decision rule, it is not blind, and a permutation over channel sets is not "
          "a")
    print("permutation over labels, so it cannot speak to whether the complement "
          "decodes at all.")

    # --- the registered analysis-falsifiers, all ten, printed --------------------
    # These fire when the MEASUREMENT is broken, not when the hypothesis moved. If any
    # fires, no number above is quotable until it is resolved. Printed rather than
    # asserted where the registered response is "report both and resolve", because an
    # assert would prevent the reporting the pre-registration requires.
    print("\n--- Registered analysis-falsifiers (prereg section 7) ---")
    PRIOR_COMPLEMENT_SWEEP = 0.793     # hostile_verify_A.stdout line 17, uncommitted
    ONE_TRIAL = 100 / n                # 2.222 points
    checks = []


    def check(idx, label, ok, detail):
        checks.append((idx, label, ok, detail))
        print(f"({idx}) {'PASS ' if ok else 'FIRED'} {label}: {detail}")


    check(1, "all-64 reproduces 41/45 at seed 42",
          by_name[ALL64][3] == 41,
          f"got {by_name[ALL64][3]}/{n} = {by_name[ALL64][3]/n:.1%}, "
          f"expected 41/{n} = {41/n:.1%}")
    check(2, "sensorimotor 43/45 and frontopolar 23/45 at seed 42",
          by_name[SMC][3] == 43 and by_name[FP][3] == 23,
          f"got {by_name[SMC][3]}/{n} and {by_name[FP][3]}/{n}, expected 43/45 and 23/45")
    check(3, "COMPLEMENT is 47 and partitions the montage with SENSORIMOTOR",
          len(COMPLEMENT) == 47
          and set(SENSORIMOTOR) | set(COMPLEMENT) == set(ch_names)
          and not (set(SENSORIMOTOR) & set(COMPLEMENT)),
          f"{len(SENSORIMOTOR)} + {len(COMPLEMENT)} = {len(ch_names)}, disjoint")
    _lattice_ok = all(abs(r[2].mean() - r[3] / n) < 1e-9 for r in results)
    check(4, "every accuracy on the k/45 lattice, fold-mean == pooled count",
          _lattice_ok,
          f"all {len(results)} conditions checked, plus every one of the "
          f"{len(SEEDS)} per-seed values in the sweep")
    _delta_prior = 100 * abs(comp_sweep.mean() - PRIOR_COMPLEMENT_SWEEP)
    check(5, "complement ten-seed mean within one trial of the prior uncommitted run",
          _delta_prior <= ONE_TRIAL + 1e-9,
          f"this run {comp_sweep.mean():.1%}, prior (uncommitted) "
          f"{PRIOR_COMPLEMENT_SWEEP:.1%}, difference {_delta_prior:.1f} points, "
          f"one trial = {ONE_TRIAL:.3f}")
    check(6, "complement is above the frontopolar-8 floor",
          comp_sweep.mean() > sweeps[FP].mean() and comp_seed42 > by_name[FP][3],
          f"complement {comp_sweep.mean():.1%} ten-seed / {comp_seed42}/{n} seed42 vs "
          f"frontopolar {sweeps[FP].mean():.1%} / {by_name[FP][3]}/{n}")
    check(7, "permutation null centred within 5 points of 50%",
          null_off_50 <= 5.0, f"null at {perm_null.mean():.1%}, {null_off_50:.1f} points off")
    check(8, "negative control stays dead (within 3 trials of majority)",
          _fp_off <= 3, f"frontopolar is {_fp_off} trial(s) off {maj_correct}/{n}")
    G = 100 * (sweeps[ALL64].mean() - comp_sweep.mean())
    check(9, "band E did not land (complement does not beat all 64 by > two trials)",
          G >= -NOISE_BAND, f"G = {G:+.1f} points, band-E edge is {-NOISE_BAND:.3f}")
    # SCOPE, and it is the registered scope, not a convenient one. Falsifier (10)
    # fires only in bands A and B: "if G lands in A or B and the McNemar does NOT
    # reach p < 0.05, that is a red flag about the pairing." Band C is NOT in that
    # scope, because band C already registers its own handling of exactly this
    # configuration: "if McNemar gives p >= 0.05, the verdict downgrades to band C2's
    # wording even though G cleared." A failed McNemar in band C is a REGISTERED
    # OUTCOME, not a broken measurement.
    #
    # WRITTEN DOWN BECAUSE IT WAS GOT WRONG ONCE. The first version of this check
    # fired whenever G >= 10.0 and p >= 0.05, which is wider than the pre-registration
    # and would have declared a registered band-C downgrade to be an analysis failure.
    # It was corrected against the prereg text AFTER the first run, which is a change
    # made with the answer visible, so it is recorded here rather than quietly fixed.
    # It changes no measured value: the 2x2, the p, G and every accuracy are identical
    # either way. What it changes is whether this run is reportable at all.
    _band_AB = G > 22.9
    check(10, "in band A or B, the McNemar also reaches p < 0.05 (scoped to A/B)",
          not (_band_AB and mcn_p >= ALPHA),
          f"G = {G:+.1f} is {'in' if _band_AB else 'NOT in'} band A/B; McNemar "
          f"p = {mcn_p:.4f} on n_disc = {n_disc}"
          + ("" if _band_AB else ", so this check does not apply"))
    # The pairing diagnostic falsifier (10) tells you to run. Cheap, so it runs
    # unconditionally rather than only when the falsifier fires.
    _folds_all = [te.tolist() for _, te in skf.split(np.zeros((n, 1)), labels)]
    _folds_again = [te.tolist() for _, te in skf.split(np.zeros((n, 1)), labels)]
    print(f"      pairing diagnostic: StratifiedKFold(random_state={SEED}) returns "
          f"identical test folds on repeated calls: {_folds_all == _folds_again}. "
          f"Both arms were scored on it, and for each arm cross_val_predict's pooled "
          f"count equals cross_val_score's fold-mean (asserted above), so the 2x2 is "
          f"a genuine per-trial pairing.")
    n_fired = sum(1 for c in checks if not c[2])
    print(f"{len(checks) - n_fired} of {len(checks)} passed. "
          + ("No analysis-falsifier fired; the numbers above are quotable."
             if n_fired == 0 else
             f"{n_fired} FIRED. Nothing above is quotable until they are resolved."))

    # --- THE PRE-REGISTERED VERDICT ------------------------------------------------
    # G and the bands were fixed in writing before this script existed. The band is
    # selected by arithmetic here, not by reading, so the edge cannot be resolved in
    # the project's favour after the fact.
    print("\n--- THE PRE-REGISTERED VERDICT ---")
    print(f"G = (all-64 ten-seed mean) - (complement ten-seed mean) = "
          f"{100*sweeps[ALL64].mean():.1f} - {100*comp_sweep.mean():.1f} = "
          f"{G:+.1f} points.")
    print(f"Two trials on n = {n} is {NOISE_BAND:.3f} points (the prereg rounds this "
          f"to 4.4). The pre-committed threshold for a real loss is G > "
          f"{G_THRESHOLD:.1f} points AND McNemar p < {ALPHA}.")

    # Edges resolve DOWNWARD in G, to the band that claims LESS about the strip being
    # necessary. Fixed in advance so a boundary cannot be argued afterwards.
    if G > 38.4:
        band, band_name = "A", "COLLAPSE"
    elif G > 22.9:
        band, band_name = "B", "SEVERE LOSS"
    elif G > G_THRESHOLD:
        band, band_name = "C", "SUBSTANTIAL LOSS, NOT NECESSARY"
    elif G > NOISE_BAND:
        band, band_name = "C2", "SUGGESTIVE, NOT ESTABLISHED"
    elif G >= -NOISE_BAND:
        band, band_name = "D", "MATCH"
    else:
        band, band_name = "E", "EXCEEDS"

    mcnemar_fired = mcn_p < ALPHA
    print(f"BAND {band}: {band_name}. McNemar p = {mcn_p:.4f} on n_disc = {n_disc}, "
          f"which {'DOES' if mcnemar_fired else 'does NOT'} reach p < {ALPHA}.")

    if band in ("A", "B", "C") and mcnemar_fired:
        if band == "A":
            print("VERDICT: the 47 non-strip electrodes carry nothing. The sensorimotor")
            print("strip is NECESSARY as well as sufficient, and the falsifiable form")
            print("survives its test. Registered as SUSPECT-FIRST all the same: a")
            print("47-channel montage including the whole posterior and temporal rings")
            print("falling to the floor is not a plausible neurophysiological result.")
        elif band == "B":
            print("VERDICT: the complement decodes above the majority floor but is")
            print("crippled. The signal is strongly concentrated over the strip, and")
            print("there is a REAL RESIDUAL elsewhere that is disclosed, not rounded to")
            print("zero. The residual cannot be attributed to posterior cortex until an")
            print("EMG bound exists, because the complement retains T8, T10 and TP8.")
        else:
            print("VERDICT: the sensorimotor strip is SUFFICIENT BUT NOT NECESSARY.")
            print("The falsifiable form 'if it reads sensorimotor cortex, deleting")
            print("sensorimotor cortex must break it' is FALSE AT THIS INSTRUMENT.")
            print(f"{len(COMPLEMENT)} electrodes that exclude the strip still reach "
                  f"{comp_sweep.mean():.1%} over ten seeds, {comp_seed42}/{n} at seed "
                  f"{SEED}.")
            print("The framing this project leans on gets WEAKER, and that is what is")
            print("written. The one permitted positive sentence is the DENSITY one, and")
            print("it is a sensor-space claim, not a source claim:")
            print(f"  {len(SENSORIMOTOR)} channels reach {sweeps[SMC].mean():.1%} and "
                  f"the {len(COMPLEMENT)} that exclude them reach "
                  f"{comp_sweep.mean():.1%}, so per-channel")
            print(f"  discriminative density is roughly "
                  f"{(sweeps[SMC].mean()/len(SENSORIMOTOR))/(comp_sweep.mean()/len(COMPLEMENT)):.1f}x "
                  f"higher over the strip.")
    elif band in ("A", "B", "C") and not mcnemar_fired:
        print(f"VERDICT DOWNGRADED, by the rule and not by preference. G = {G:+.1f} "
              f"cleared the")
        print(f"{G_THRESHOLD:.1f}-point threshold; the McNemar did not reach "
              f"p < {ALPHA}. The registered rule")
        print("requires BOTH halves and forbids upgrading on whichever reads better.")
        print(f"REGISTERED VERDICT: a loss is SUGGESTED and NOT ESTABLISHED at n = {n}.")
        print("Reported as undecided, leaned neither way.")
        print(f"NOT ESTABLISHED BY THIS RUN, therefore not written: that the strip is")
        print("'sufficient but not necessary'. G points that way and the test that was")
        print("registered to confirm it did not. Both halves get said, or neither does.")
        if band == "C":
            # Band C registers this path explicitly, so it is a registered outcome and
            # not an analysis failure. Say why the McNemar failed, in the terms the
            # prereg fixed in advance: the count, then the power at that count.
            print(f"WHY IT FAILED, at the discordant count it actually got: n_disc = "
                  f"{n_disc} ({only_all} vs {only_comp}).")
            if worst is not None and worst[0] == only_all:
                print(f"      The observed split IS the most lopsided split that misses "
                      f"p < {ALPHA} at")
                print(f"      n_disc = {n_disc}. Only a {n_disc}-0 sweep would have "
                      f"reached it. This design had")
                print("      essentially no power here, and that is a property of n = 45")
                print("      with a strongly agreeing pair, not evidence for either side.")
            print(f"      {both} of {n} trials were called correctly by BOTH arms and "
                  f"{neither} by neither,")
            print("      which is what leaves the discordant count small. The pairing is")
            print("      working as designed; it is the sample that is small.")
            # ADDED 2026-07-26. The four sentences above are true and they are not
            # the whole truth. They describe the discordant count as if it were the
            # binding constraint. It is not: the rule itself cannot fire here.
            print(f"WHAT THE FOUR LINES ABOVE LEAVE OUT, and it is the larger part:")
            print(f"      (1) THE RULE CANNOT FIRE AT ITS OWN THRESHOLD. The McNemar "
                  f"half needs a")
            print(f"          {_min_gap}-trial = {100*_min_gap/n:.1f}-point gap at "
                  f"minimum, at any n_disc. The registered G half fires at")
            print(f"          {G_THRESHOLD:.1f} points = {G_THRESHOLD*n/100:.2f} "
                  f"trials. Between them the conjunction is unreachable by")
            print(f"          construction. This is a defect in the rule, and the "
                  f"pre-registration's stated")
            print(f"          justification for the rule ('at a gap of "
                  f"{G_THRESHOLD:.0f} or more points the McNemar should")
            print(f"          fire comfortably', section 6.2) is REFUTED by this "
                  f"arithmetic.")
            print(f"      (2) THE VERDICT IS NOT ROBUST TO SEED CHOICE. Seed {SEED} "
                  f"gives p = {mcn_p:.4f}. The same")
            print(f"          exact test on the {len(SEEDS)} seeds of this script's "
                  f"OWN registered sweep gives a")
            print(f"          median of {np.median(_sweep_ps):.4f}, with "
                  f"{_n_fire} of {len(SEEDS)} reaching p < {ALPHA}. Seed {SEED} is worse")
            print(f"          for the loss than {int((_sweep_ps < mcn_p).sum())} of "
                  f"those {len(SEEDS)} seeds and worse than their median.")
            print(f"          So 'not established' is substantially a fact about "
                  f"which integer was")
            print(f"          typed as random_state, not about the recording.")
            print(f"      (3) THE CONJUNCTION IS EVALUATED ON DISJOINT SEED SETS. G "
                  f"is a mean over")
            print(f"          {list(SEEDS)}; the McNemar is at seed {SEED}, which is "
                  f"not in that list.")
            print(f"      NONE OF THIS FLIPS THE VERDICT. Ten non-independent "
                  f"re-splits of the same {n}")
            print(f"      trials are not ten samples, so 'ESTABLISHED' is not "
                  f"licensed either. What is")
            print(f"      established is that the registered rule, as written, "
                  f"CANNOT certify band C at")
            print(f"      n = {n}, and that the reported outcome is a property of "
                  f"the rule at least as much")
            print(f"      as a reading of the data.")
        if band in ("A", "B"):
            print("At a gap this size the McNemar should have fired comfortably, so this")
            print("is a RED FLAG about the pairing rather than a licence to report the")
            print("gap anyway. See falsifier (10).")
    elif band == "C2":
        print(f"VERDICT: the complement is below all-64 by more than two trials but by")
        print(f"less than the pre-committed {G_THRESHOLD:.1f}-point threshold. A loss is")
        print(f"SUGGESTED and NOT ESTABLISHED at n = {n}. It is NOT upgraded with the")
        print("McNemar, because the rule requires both halves and this band fails the G")
        print("half by construction. Reported as undecided, leaned neither way.")
    elif band == "D":
        print("VERDICT: the complement is NOT DISTINGUISHABLE from the full montage.")
        print("The sensorimotor strip contributes nothing the rest of the montage does")
        print("not already carry. Every sensorimotor framing in this corpus weakens, and")
        print("the sufficiency arm is reduced to 'these 17 channels are one of several")
        print("sets that work'. A48 binds here: this corpus already refuses to call two")
        print("trials a difference in the direction that flattered it, so it refuses in")
        print("this direction too. 'No difference DETECTED' is not 'no difference'.")
    else:
        print("VERDICT WITHHELD. Band E: the complement BEATS the full montage by more")
        print("than two trials. Registered as SUSPECT FIRST, not as a result. Check the")
        print("channel picking, check that CSP is inside the fold, check the crop, and")
        print("re-run before reporting anything. Only reportable once those pass, and")
        print("then it is bad news for the framing.")

    # --- printed in EVERY band, without exception ---------------------------------
    print("\nTRUE IN EVERY BAND (1): THE AVERAGE-REFERENCE LEAK. The reference is")
    print(f"computed over all {len(ch_names)} electrodes BEFORE the "
          f"{len(COMPLEMENT)} are picked, so every")
    print(f"complement channel carries -1/{len(ch_names)} of every sensorimotor "
          f"channel. The complement is")
    print("NOT electrically independent of the strip. This measurement BOUNDS the")
    print("strip's necessity; it cannot establish it. A high complement score is")
    print("partly what volume conduction plus a 64-channel average reference PREDICTS.")
    print("TRUE IN EVERY BAND (2): THE INSTRUMENT LIMIT. No band licenses a SOURCE")
    print("claim in either direction. Forward-is-not-inverse refutes a negative source")
    print("claim exactly as hard as a positive one, so 'the strip is not necessary'")
    print("is a statement about 64 electrodes, not about cortex.")

    print("\n--- What this does and does not show ---")
    all64 = by_name[ALL64][3] / n
    smc = by_name[SMC][3] / n
    fp = by_name[FP][3] / n
    loro = by_name[LORO][3] / n
    fp_correct = by_name[FP][3]
    print(f"Frontopolar-only lands at {fp:.1%} ({fp_correct}/{n}) against a "
          f"majority-class rate of {majority:.1%} ({maj_correct}/{n}).")
    gap = abs(fp_correct - maj_correct)
    # EVERY NUMBER IN THIS BLOCK IS INTERPOLATED, and that is the point of the
    # rewrite. Until 2026-07-25 the lines below were bare print() calls with no
    # f-prefix, carrying the literals "21/24", "51.1%", "one trial" and "0.33 to
    # 0.78". They agreed with the table above by construction rather than by
    # measurement, and would have kept printing the old values if the ablation ever
    # moved -- which also means a provenance check that looks for a number in some
    # script's stdout would have found these no matter what the pipeline did.
    fp_folds = by_name[FP][2]
    print(f"That is {gap} trial{'' if gap == 1 else 's'} off the rate you get by ignoring")
    print("the EEG entirely and always answering 'feet'. The frontopolar")
    print("decoder has no usable signal. Note the framing: the honest reference here")
    print(f"is the MAJORITY rate, not 50% -- with {n_hands}/{n_feet} classes, a "
          f"{fp:.1%} result is not")
    print(f"'above chance', it is a degenerate classifier {gap} "
          f"trial{'' if gap == 1 else 's'} short of guessing.")
    print(f"The per-fold spread ({fp_folds.min():.2f} to {fp_folds.max():.2f}) is "
          f"the other tell: folds that wide are")
    print("a coin, not a decoder.")
    # SINGLE-SEED DRAW, not an effect size. The gain printed on the next line is one
    # seed-42 partition. A 20-seed reference on the same two conditions puts the
    # expected difference near +1 point, so this row is roughly 4x the expectation
    # and its DIRECTION is what the write-up treats as inside noise. Read it as one
    # draw, not as "dropping non-motor channels buys you 4 points."
    print(f"Sensorimotor-only ({smc:.1%}) vs. all 64 ({all64:.1%}): "
          f"{100*(smc-all64):+.1f} points from dropping "
          f"{64-len(SENSORIMOTOR)} non-motor channels (one seed, one partition).")
    print(f"Leave-one-run-out ({loro:.1%}) holds up with no trial sharing a run with")
    print("its training set, so the result is not an artifact of drift shared INSIDE")
    print(f"a run. It does not rule out a session-level trend: runs {RUNS} are three")
    print("recordings from one session, and a drift monotonic across all three")
    print("survives this control. EEGMMIDB has no second session to test against.")
    print("\nBOUND, NOT PROOF: the average reference is computed over all 64 channels")
    print("before picking, so the subsets are not electrically independent, and")
    print("EEGMMIDB ships no EOG channel to regress out. This ablation bounds the")
    print("ocular contribution; it cannot measure it.")

    # --- WHAT THE COMPLEMENT ARM DOES NOT SHOW ------------------------------------
    # Registered in advance, printed unconditionally, so none of it can be discovered
    # later and presented as a caveat the author thought of afterwards.
    print("\n--- WHAT THE COMPLEMENT ARM DOES NOT SHOW ---")
    print(f"(i)   THE AVERAGE-REFERENCE LEAK. Reference over all {len(ch_names)}, "
          f"then pick {len(COMPLEMENT)}. Every")
    print(f"      complement channel carries -1/{len(ch_names)} of every deleted "
          f"channel. BOUNDS, not proves.")
    print("(ii)  THE INSTRUMENT LIMIT. Deleting the electrodes nearest a source does")
    print("      not delete the source from the remaining electrodes. No")
    print("      channel-deletion result on a 64-channel scalp montage falsifies a")
    print("      source hypothesis, in EITHER direction. Sensor-space only.")
    print(f"(iii) THIS IS NOT 'SENSORIMOTOR CORTEX DELETED'. The complement keeps "
          f"{[c for c in COMPLEMENT if c in ('FC5', 'FC6', 'CP5', 'CP6')]},")
    print(f"      four peri-Rolandic electrodes. The wide-{len(WIDE)} arm above deletes "
          f"them too and lands")
    print(f"      at {by_name[NWIDE][3]}/{n} = {by_name[NWIDE][3]/n:.1%} at seed "
          f"{SEED}, {sweeps[NWIDE].mean():.1%} over ten seeds. That is the bound.")
    print(f"(iv)  UNBOUNDED EMG. The complement keeps "
          f"{[c for c in COMPLEMENT if c in ('T8', 'T10', 'TP8')]}, temporalis muscle")
    print("      territory, and NOTHING in this repo bounds a myogenic contribution.")
    print("      The permitted sentence is 'the non-strip electrodes decode above the")
    print("      floor, and how much of that is muscle is NOT yet bounded'. The")
    print("      sentence 'posterior cortex also decodes' is BLOCKED until it is.")
    print(f"(v)   CHANNEL-COUNT CONTROL: MEASURED 2026-07-26, PARTIALLY RETIRED. This "
          f"line used to")
    print(f"      read 'NO CHANNEL-COUNT CONTROL' and declare {len(COMPLEMENT)} vs "
          f"{len(SENSORIMOTOR)} vs {len(ch_names)} channels an")
    print("      unmeasured confound. It is now measured for the "
          f"{len(COMPLEMENT)}-vs-{len(ch_names)} comparison: arm 10")
    print(f"      deletes {len(SENSORIMOTOR)} channels at random {N_RANDOM_DRAWS} "
          f"times and the ten-seed mean barely moves")
    print(f"      ({_null_means.mean():.1%}, G null mean {_null_G.mean():+.1f} "
          f"points), while deleting the strip costs {_G_obs:+.1f}.")
    print("      So channel count does not explain the complement arm's deficit. The")
    print(f"      confound STANDS UNMEASURED for the {len(SENSORIMOTOR)}-channel and "
          f"{len(FRONTOPOLAR)}-channel arms, which keep")
    print("      far fewer channels than arm 10 ever does. Channel counts still print")
    print("      beside every row.")
    print(f"(vi)  n = {n}, ONE SUBJECT, ONE SESSION, three runs. Ten seeds is a small")
    print("      sample of splits, which is why the range prints beside every mean,")
    print("      and the McNemar sits on a discordant count of "
          f"{n_disc}, which is why that count")
    print("      travels with its p everywhere it is quoted.")
    print("(vii) THE ACCURACY HALF OF THIS ARM WAS NOT BLIND. It had been run once,")
    print("      uncommitted, before the pre-registration was written, and the prior")
    print(f"      value ({PRIOR_COMPLEMENT_SWEEP:.1%} over ten seeds) is checked "
          f"against this run at falsifier (5).")
    print("      The permutation test, the Wilson interval and the McNemar are blind.")



if __name__ == "__main__":
    main()
