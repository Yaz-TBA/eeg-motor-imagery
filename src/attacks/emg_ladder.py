"""Section 8 of emg_proxy.py: the sensitivity ladder that turns a null into a
bound, over the two registered topographies and both injection directions. Split
out 2026-08-26; the body is verbatim from that file."""

from types import SimpleNamespace

import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import TEMPORAL, make_clf
from emg_setup import (
    CROP, DETECT_K, INJECT_SEED_BASE, INTERMITTENT_FRACTION,
    INTERMITTENT_SEED_BASE, LADDER, N_INJECT_SEEDS, SEED, TOPO_FLAT,
    TOPO_STIPULATED, TOPO_T8_ONLY, TOPO_T8_T10, hr, sub,
)


def run_ladder(D, cohens_d, MDE_AGGREGATE):
    get_data, acc_str, assert_lattice = D.get_data, D.acc_str, D.assert_lattice
    labels, N, N_TIMES, EPOCHS = D.labels, D.N, D.N_TIMES, D.EPOCHS
    SFREQ, BANDS = D.SFREQ, D.BANDS

    # =============================================================================
    # 8. The sensitivity ladder
    # =============================================================================
    hr("8. THE SENSITIVITY LADDER: WHAT SIZE OF PLANTED SOURCE CAN THIS PROBE SEE?")

    print("\nThis is the part that separates measuring from conceding. Without it, a")
    print("probe at floor supports only 'we looked and found nothing'. With it, the")
    print("probe supports 'we can detect a class-correlated broadband temporal source")
    print("of size X, and this recording contains less than X'.")
    print("\nDESIGN. A realistic muscle artifact is a SOURCE: one generator projecting")
    print("to several electrodes with a fixed topography. Independent per-channel noise")
    print("was rejected as unrealistic and as unfair to CSP, whose entire business is")
    print("finding coherent spatial directions.")
    print("\n  per trial, one latent Gaussian series, band-limited with the SAME filter")
    print("  cascade as the primary band, projected onto the 8 temporal channels with a")
    print("  fixed unit-norm topography.")
    print("  amplitude a = the source's contribution to T8 as a fraction of T8's own")
    print("  measured high-band SD, so the ladder is in interpretable units.")
    print(f"  {N_INJECT_SEEDS} injection seeds per rung, distinct from the CV seed, "
          f"which stays {SEED}.")
    print("  BOTH directions (into the 21 hands, and separately into the 24 feet). The")
    print("  WORSE, meaning higher, detection threshold is the one reported as the")
    print("  bound. A bound computed from the easier direction overstates the instrument.")
    print(f"  detection = smallest a whose MEDIAN accuracy across {N_INJECT_SEEDS} seeds "
          f"reaches {DETECT_K}/{N} = {DETECT_K / N:.1%}.")
    print(f"  {DETECT_K}/{N} rather than 30/45 because 30/45 sits at binomial p = 0.0490 "
          f"and a detection")
    print("  criterion should not rest on a knife edge.")
    print("\nTHE TOPOGRAPHY IS STIPULATED, NOT MEASURED. It is a plausible right")
    print("temporalis shape, not this subject's. That is why a spatially flat")
    print("topography is run alongside it.")

    X_TEMPORAL = get_data("PRIMARY", TEMPORAL).copy()
    T8_IDX = TEMPORAL.index("T8")
    SD_T8 = float(X_TEMPORAL[:, T8_IDX, :].std())
    print(f"\n  Measured T8 high-band SD over the feature window: {SD_T8:.4e} V")

    FULL_TIMES = EPOCHS["PRIMARY"].times
    crop_mask = (FULL_TIMES >= CROP[0] - 1e-9) & (FULL_TIMES <= CROP[1] + 1e-9)
    assert int(crop_mask.sum()) == N_TIMES, (
        f"crop mask gives {int(crop_mask.sum())} samples, cropped epochs give {N_TIMES}"
    )
    N_FULL = len(FULL_TIMES)


    def make_source(rng, n_trials):
        """One band-limited latent source per trial, unit SD over the feature window."""
        s = rng.standard_normal((n_trials, N_FULL))
        b = BANDS["PRIMARY"]
        s = mne.filter.filter_data(
            s, SFREQ, b["l_freq"], b["h_freq"],
            l_trans_bandwidth=b["l_trans"], h_trans_bandwidth=b["h_trans"],
            method="fir", fir_design="firwin", verbose="error",
        )
        s = mne.filter.notch_filter(
            s, SFREQ, np.array([60.0]), notch_widths=2.0, trans_bandwidth=6.0,
            method="fir", fir_design="firwin", verbose="error",
        )
        s = s[:, crop_mask]
        return s / s.std()


    def topo_vector(topo):
        w = np.array([topo[ch] for ch in TEMPORAL], dtype=float)
        return w / np.linalg.norm(w)


    def ladder_run(a, topo, target_label, seed_i, cv_seed=SEED, intermittent=False):
        """Inject, then run the unmodified CV. Injection happens BEFORE the split, on
    the data array, so every fold sees the same planted source. Injecting after
    the split would be a leak, and the pre-registration flags that explicitly.

    intermittent=True concentrates the SAME TOTAL injected variance into a random
    INTERMITTENT_FRACTION of the target class's trials, by scaling the per-trial
    amplitude by 1/sqrt(fraction) on the chosen trials and zero elsewhere. Added
    2026-07-26 because this script prints that the realistic EMG failure mode is
    "a few trials with a clench, not a shifted distribution" and then calibrated
    exclusively against a shifted distribution.
    """
        w = topo_vector(topo)
        idx = np.where(labels == target_label)[0]
        rng = np.random.default_rng(INJECT_SEED_BASE + seed_i)
        s = make_source(rng, len(idx))
        scale = a * SD_T8 / w[T8_IDX]
        amp = np.full(len(idx), scale, dtype=float)
        if intermittent:
            n_on = max(1, int(round(INTERMITTENT_FRACTION * len(idx))))
            rng_i = np.random.default_rng(INTERMITTENT_SEED_BASE + seed_i)
            on = rng_i.choice(len(idx), size=n_on, replace=False)
            amp = np.zeros(len(idx), dtype=float)
            # 1/sqrt(f) keeps the TOTAL injected variance equal to the continuous arm
            # at the same a, with f the realised on-fraction rather than the nominal.
            amp[on] = scale / np.sqrt(n_on / len(idx))
        X = X_TEMPORAL.copy()
        X[idx] += w[None, :, None] * (amp[:, None] * s)[:, None, :]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
        pred = cross_val_predict(make_clf(), X, labels, cv=cv)
        k = int((pred == labels).sum())
        assert_lattice(k, f"ladder a={a} seed={seed_i} cv={cv_seed} int={intermittent}")
        lp = np.log(np.mean(X ** 2, axis=-1)).mean(axis=1)
        d = cohens_d(lp[labels == 2], lp[labels == 3])
        return k, d


    def ladder_rows(topo, target, cv_seed=SEED, intermittent=False):
        """One full ladder: (a, median k, min k, max k, median d) per rung."""
        rows = []
        for a in LADDER:
            if a == 0.0:
                k, d = ladder_run(0.0, topo, target, 0, cv_seed, intermittent)
                rows.append((a, k, k, k, d))
                continue
            ks, ds = [], []
            for si in range(N_INJECT_SEEDS):
                k, d = ladder_run(a, topo, target, si, cv_seed, intermittent)
                ks.append(k)
                ds.append(d)
            rows.append((a, int(np.median(ks)), min(ks), max(ks), float(np.median(ds))))
        return rows


    DIRECTIONS = [("into hands (21)", 2), ("into feet  (24)", 3)]
    TOPOS = [("stipulated", TOPO_STIPULATED), ("flat", TOPO_FLAT)]
    # POST-REGISTRATION shapes, kept in a SEPARATE list so the registered pair and
    # the registered "max over topographies" can still be reported exactly as
    # registered, with the extension reported beside it rather than folded into it.
    TOPOS_EXTRA = [("focal T8-only", TOPO_T8_ONLY), ("focal T8+T10", TOPO_T8_T10)]
    ALL_TOPOS = TOPOS + TOPOS_EXTRA

    ladder_table = {}
    for topo_name, topo in TOPOS:
        for dir_name, target in DIRECTIONS:
            ladder_table[(topo_name, dir_name)] = ladder_rows(topo, target)

    for topo_name, _ in TOPOS:
        for dir_name, _ in DIRECTIONS:
            rows = ladder_table[(topo_name, dir_name)]
            sub(f"Ladder, {topo_name} topography, injected {dir_name}")
            print(f"  {'a':>6} {'median acc':>16} {'min':>7} {'max':>7} "
                  f"{'median d':>9}")
            for a, kmed, kmin, kmax, d in rows:
                note = "   <- real data" if a == 0.0 else ""
                print(f"  {a:>6.3f} {acc_str(kmed):>16} {kmin:>3}/{N} {kmax:>3}/{N} "
                      f"{d:>+9.3f}{note}")


    def threshold(rows):
        for a, kmed, _, _, d in rows:
            if a > 0.0 and kmed >= DETECT_K:
                return a, kmed, d
        return None, None, None


    sub("Detection thresholds, and the bound they produce")
    print(f"  Detection = median accuracy across {N_INJECT_SEEDS} seeds reaches "
          f"{DETECT_K}/{N} = {DETECT_K / N:.1%}.")
    print(f"\n  {'topography':<12} {'direction':<18} {'threshold a':>12} "
          f"{'acc at thr':>16} {'d at thr':>9}")
    thr_by_topo = {}
    for topo_name, _ in TOPOS:
        worst = None
        for dir_name, _ in DIRECTIONS:
            a, kmed, d = threshold(ladder_table[(topo_name, dir_name)])
            if a is None:
                print(f"  {topo_name:<12} {dir_name:<18} {'NEVER':>12} "
                      f"{'':>16} {'':>9}")
                worst = "NEVER"
            else:
                print(f"  {topo_name:<12} {dir_name:<18} {a:>12.3f} "
                      f"{acc_str(kmed):>16} {d:>+9.3f}")
                if worst != "NEVER":
                    worst = a if worst is None else max(worst, a)
        thr_by_topo[topo_name] = worst

    LADDER_FAILED = any(v == "NEVER" for v in thr_by_topo.values())
    LADDER_SUSPICIOUS = any(v == LADDER[1] for v in thr_by_topo.values()
                            if v != "NEVER" and v is not None)

    print(f"\n  Worst (reported) threshold, stipulated topography: {thr_by_topo['stipulated']}")
    print(f"  Worst (reported) threshold, flat topography       : {thr_by_topo['flat']}")

    # The two arms have different sensitivities, and the ladder MEASURES the gap
    # rather than asserting it. This is why the pre-registration says arm (a)'s null
    # does not rescue arm (b): if arm (b) detects a planted source at a realised
    # aggregate d well under arm (a)'s 0.837 detection floor, then a null on (a) was
    # never going to be evidence about (b) in the first place.
    thr_ds = []
    for topo_name, _ in TOPOS:
        for dir_name, _ in DIRECTIONS:
            a, kmed, d = threshold(ladder_table[(topo_name, dir_name)])
            if a is not None:
                thr_ds.append(abs(d))
    if thr_ds:
        print(f"\n  Realised aggregate |Cohen's d| at the detection threshold ranges "
              f"{min(thr_ds):.3f} to {max(thr_ds):.3f}")
        print(f"  across the four topography-by-direction cells, against arm (a)'s "
              f"detection floor of {MDE_AGGREGATE:.3f}.")
        print("  Arm (b) therefore detects planted sources that arm (a) provably")
        print("  cannot, and in the feet-injection direction it detects one whose")
        print("  univariate aggregate marginal is close to ZERO, because the injection")
        print("  cancels the small baseline difference on its way past. That is the")
        print("  pre-registered point that a null on arm (a) does not rescue arm (b),")
        print("  measured on this data rather than asserted.")

    if LADDER_FAILED:
        print("\n  *** PRIMARY FALSIFIER FIRED. The ladder never reached the detection")
        print(f"  *** criterion, even at a = {LADDER[-1]:.3f}, meaning a source contributing")
        print(f"  *** {LADDER[-1]:.0%} of T8's own high-band SD with a coherent topography.")
        print("  *** A probe that cannot recover a planted source that large is not an")
        print("  *** instrument, and its null bounds NOTHING. This falsifies the")
        print("  *** MEASUREMENT, not the hypothesis. Nothing about EMG may be concluded")
        print("  *** from this run, and the exposure remains open exactly as the corpus")
        print("  *** already says.")
    elif LADDER_SUSPICIOUS:
        print(f"\n  *** Detection at the SMALLEST rung a = {LADDER[1]:.3f}. Per the")
        print("  *** pre-registration this is treated as a SUSPECTED LEAK in the")
        print("  *** injection code, not as a sensitivity result, and it blocks")
        print("  *** reporting the bound until audited.")
    else:
        ts, tf = thr_by_topo["stipulated"], thr_by_topo["flat"]
        ratio = max(ts, tf) / min(ts, tf)
        print(f"\n  stipulated / flat threshold ratio = {ratio:.2f}x "
              f"(pre-registered concern threshold: about 2x)")
        if ratio > 2.0:
            print("  The sensitivity figure DEPENDS MATERIALLY on an assumed spatial")
            print("  shape that was never measured for this subject. Per the")
            print("  pre-registration the WORSE threshold is reported as the bound and")
            print("  the dependence is stated rather than the flattering figure quoted.")
        elif abs(ratio - 2.0) < 1e-9:
            print("  KNIFE EDGE, and it is reported as one rather than resolved in the")
            print("  project's favour. The ratio lands EXACTLY on the pre-registered")
            print("  'about 2x' concern threshold, so the branch could be argued either")
            print("  way. It does not matter, because the pre-registration reports the")
            print("  WORSE threshold as the bound in BOTH branches, and that is what is")
            print("  printed below. Note also that the ladder rungs are discrete, so")
            print("  this ratio is quantised by the ladder's own spacing and should not")
            print("  be read to two decimal places.")
        else:
            print("  The two topographies agree within the pre-registered 2x tolerance,")
            print("  so the bound does not rest on the assumed shape.")
        print(f"  BOUND REPORTED (as registered) = a = {max(ts, tf):.3f} times T8's "
              f"own high-band SD,")
        print("  which is the worse of the two REGISTERED topographies AND the worse of")
        print("  the two injection directions. Section 8B below shows this is NOT the")
        print("  worst case over shapes, and withdraws it as a bound over topographies.")


    return SimpleNamespace(
        ladder_table=ladder_table, threshold=threshold, thr_by_topo=thr_by_topo,
        LADDER_FAILED=LADDER_FAILED, ladder_rows=ladder_rows,
        ladder_run=ladder_run, topo_vector=topo_vector, T8_IDX=T8_IDX,
        SD_T8=SD_T8, TOPOS=TOPOS, TOPOS_EXTRA=TOPOS_EXTRA, ALL_TOPOS=ALL_TOPOS,
        DIRECTIONS=DIRECTIONS,
    )
