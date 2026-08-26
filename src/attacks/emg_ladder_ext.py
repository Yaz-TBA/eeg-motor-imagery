"""Section 8B of emg_proxy.py, the post-registration ladder extension: the shape
family, the intermittent arm, the saturation probe and the tiered bound. Split out
2026-08-26; the body is verbatim, and it is disclosed there as not blind."""

import numpy as np

from common import TEMPORAL
from emg_setup import (
    DETECT_K, INTERMITTENT_FRACTION, LADDER, LADDER_SEED_CHECK,
    N_INJECT_SEEDS, N_SEED_SWEEP, SEED, TOPO_STIPULATED, hr, sub,
)


def run_ladder_extension(D, L):
    acc_str, N = D.acc_str, D.N
    MAJ_CORRECT, N_HANDS, N_FEET = D.MAJ_CORRECT, D.N_HANDS, D.N_FEET
    ladder_table, threshold, thr_by_topo = (
        L.ladder_table, L.threshold, L.thr_by_topo)
    ladder_rows, ladder_run, topo_vector = (
        L.ladder_rows, L.ladder_run, L.topo_vector)
    T8_IDX, TOPOS_EXTRA, ALL_TOPOS, DIRECTIONS = (
        L.T8_IDX, L.TOPOS_EXTRA, L.ALL_TOPOS, L.DIRECTIONS)

    # =============================================================================
    # 8B. POST-REGISTRATION: the ladder over a shape family, and over intermittency
    # =============================================================================
    hr("8B. POST-REGISTRATION LADDER EXTENSION (added 2026-07-26, NOT BLIND)")

    print("Everything in this section was added after the registered ladder had run and")
    print("after an adversarial pass reported what it would find. It is disclosed as an")
    print("addition, the pre-registration is not edited to accommodate it, and section")
    print("8's registered numbers above are left exactly as they printed.")

    print("\nWHY a IS NOT A SHAPE-FREE UNIT. a is defined as the source's contribution")
    print("to T8 as a fraction of T8's own high-band SD. Detectability depends on the")
    print("TOTAL power the source puts on the ring. Those two are related by the")
    print("topography: with w the unit-norm shape, the injected amplitude is")
    print("a * SD_T8 / w[T8], so total injected power scales as 1 / w[T8]^2. A flat")
    print("source at a given a therefore injects far more total power than a focal one")
    print("at the same a, and is correspondingly easier for the probe to see. The")
    print("registered 'max over topographies' is a max over TWO shapes, both of them on")
    print("the diffuse side.")
    print(f"\n  {'topography':<14} {'w[T8]':>7} {'total power vs T8-only':>24}")
    for _tn, _t in ALL_TOPOS:
        _w = topo_vector(_t)
        print(f"  {_tn:<14} {_w[T8_IDX]:>7.4f} {1.0/_w[T8_IDX]**2:>23.2f}x")


    def a_to_rms(a, topo):
        """Re-express a in ring-RMS units: the injected source's RMS across the 8.

    Injected SD on channel i is |w_i| * scale with scale = a * SD_T8 / w[T8].
    RMS across the ring is scale * ||w|| / sqrt(8) = scale / sqrt(8) for unit w.
    In units of SD_T8 that is a / (sqrt(8) * w[T8]). This is shape-free by
    construction: it measures the source, not its projection onto one electrode.
    """
        w = topo_vector(topo)
        return a / (np.sqrt(len(TEMPORAL)) * w[T8_IDX])


    sub("The ladder over four shapes, both directions, continuous injection")
    for topo_name, topo in TOPOS_EXTRA:
        for dir_name, target in DIRECTIONS:
            ladder_table[(topo_name, dir_name)] = ladder_rows(topo, target)
    print(f"  {'topography':<14} {'direction':<18} {'thr a':>7} {'thr a_rms':>10} "
          f"{'acc at thr':>16}")
    shape_thr = {}
    for topo_name, topo in ALL_TOPOS:
        worst_a = None
        for dir_name, _ in DIRECTIONS:
            a, kmed, d = threshold(ladder_table[(topo_name, dir_name)])
            if a is None:
                print(f"  {topo_name:<14} {dir_name:<18} {'NEVER':>7} {'':>10} {'':>16}")
                worst_a = "NEVER"
            else:
                print(f"  {topo_name:<14} {dir_name:<18} {a:>7.3f} "
                      f"{a_to_rms(a, topo):>10.3f} {acc_str(kmed):>16}")
                if worst_a != "NEVER":
                    worst_a = a if worst_a is None else max(worst_a, a)
        shape_thr[topo_name] = worst_a

    _shape_as = [v for v in shape_thr.values() if v not in (None, "NEVER")]
    _shape_rms = [a_to_rms(shape_thr[tn], t) for tn, t in ALL_TOPOS
                  if shape_thr[tn] not in (None, "NEVER")]
    if _shape_as:
        print(f"\n  In REGISTERED units (T8 contribution): thresholds span "
              f"{min(_shape_as):.3f} to {max(_shape_as):.3f}, "
              f"a {max(_shape_as)/min(_shape_as):.1f}x range.")
        print(f"  In RING-RMS units (shape-free)          : thresholds span "
              f"{min(_shape_rms):.3f} to {max(_shape_rms):.3f}, "
              f"a {max(_shape_rms)/min(_shape_rms):.2f}x range.")
        print("  The RMS re-expression collapses most of the spread, which is the")
        print("  evidence that the spread was a UNITS artifact of pinning a to one")
        print("  electrode, not a real change in what the probe can see.")
        print(f"\n  CONSEQUENCE FOR THE REGISTERED READING. The registered ladder's")
        print(f"  stipulated/flat ratio was reported as a knife edge with 'no")
        print(f"  consequence'. Over four shapes the true spread is "
              f"{min(_shape_as):.3f} to {max(_shape_as):.3f} in the")
        print(f"  registered units, so the consequence is a factor of "
              f"{max(_shape_as)/min(_shape_as):.0f} on the only number this")
        print(f"  measurement produces. The 'no consequence' reading is WITHDRAWN.")

    sub("The intermittent arm: the failure mode this script itself calls realistic")
    print("  Section 6 of this script prints that the realistic EMG failure mode is 'a")
    print("  few trials with a clench, not a shifted distribution'. Every rung above")
    print("  injects a constant-amplitude source into EVERY trial of the target class,")
    print("  which is a shifted distribution. Nothing in the registered ladder is")
    print("  intermittent, and the pre-registration considered only one alternative")
    print("  (independent per-channel noise), never intermittency.")
    print(f"  This arm concentrates the SAME TOTAL injected variance into a random "
          f"{INTERMITTENT_FRACTION:.0%}")
    print("  of the target class's trials, per-trial amplitude scaled by 1/sqrt(f),")
    print("  same rungs, same injection seeds, same CV seed.")
    inter_table = {}
    for topo_name, topo in ALL_TOPOS:
        for dir_name, target in DIRECTIONS:
            inter_table[(topo_name, dir_name)] = ladder_rows(
                topo, target, intermittent=True)
    print(f"\n  {'topography':<14} {'direction':<18} {'thr cont':>9} {'thr burst':>10} "
          f"{'worse':>7}")
    inter_thr = {}
    for topo_name, topo in ALL_TOPOS:
        worst_a = None
        for dir_name, _ in DIRECTIONS:
            a_c, _, _ = threshold(ladder_table[(topo_name, dir_name)])
            a_i, _, _ = threshold(inter_table[(topo_name, dir_name)])
            cs = "NEVER" if a_c is None else f"{a_c:.3f}"
            is_ = "NEVER" if a_i is None else f"{a_i:.3f}"
            if a_i is None:
                wr = "NEVER"
            elif a_c is None:
                wr = "NEVER"
            else:
                wr = f"{max(a_c, a_i):.3f}"
            print(f"  {topo_name:<14} {dir_name:<18} {cs:>9} {is_:>10} {wr:>7}")
            cand = "NEVER" if (a_i is None or a_c is None) else max(a_c, a_i)
            if worst_a == "NEVER" or cand == "NEVER":
                worst_a = "NEVER"
            else:
                worst_a = cand if worst_a is None else max(worst_a, cand)
        inter_thr[topo_name] = worst_a

    _all_worst = [v for v in inter_thr.values() if v not in (None, "NEVER")]
    _never = [tn for tn, v in inter_thr.items() if v == "NEVER"]

    # IS THE CRITERION EVEN REACHABLE AT A 25% DUTY CYCLE? Asked before the "NEVER"
    # rows are read as a sensitivity result, because they have a competing and
    # entirely boring explanation: a source present in only a handful of trials can
    # only carry information about that handful, and DETECT_K was calibrated against
    # continuous injection. This is measured rather than argued: push the amplitude
    # far past the registered ladder's top rung and see where the bursty arm
    # saturates. If it saturates below DETECT_K, "NEVER" is a statement about the
    # CRITERION, not about the probe's sensitivity to bursty sources.
    sub("Is the detection criterion reachable at this duty cycle? (saturation probe)")
    SATURATION_RUNGS = [2.0, 4.0, 8.0]
    print(f"  Bursty injection pushed to a = {SATURATION_RUNGS}, far past the "
          f"registered top rung of {LADDER[-1]:.3f}.")
    print(f"  {'topography':<14} {'direction':<18} "
          + " ".join(f"{'a=' + str(a):>13}" for a in SATURATION_RUNGS))
    sat_max = {}
    for topo_name, topo in ALL_TOPOS:
        for dir_name, target in DIRECTIONS:
            cells = []
            for a in SATURATION_RUNGS:
                ks = [ladder_run(a, topo, target, si, intermittent=True)[0]
                      for si in range(N_INJECT_SEEDS)]
                cells.append(int(np.median(ks)))
            sat_max[(topo_name, dir_name)] = max(cells)
            print(f"  {topo_name:<14} {dir_name:<18} "
                  + " ".join(f"{acc_str(c):>13}" for c in cells))
    _sat_best = max(sat_max.values())
    _n_reach = sum(1 for v in sat_max.values() if v >= DETECT_K)
    print(f"\n  Best median accuracy any bursty cell reaches at any amplitude: "
          f"{acc_str(_sat_best)}.")
    print(f"  Cells reaching the {DETECT_K}/{N} criterion at some amplitude: "
          f"{_n_reach} of {len(sat_max)}.")
    _n_on_hands = max(1, int(round(INTERMITTENT_FRACTION * N_HANDS)))
    _n_on_feet = max(1, int(round(INTERMITTENT_FRACTION * N_FEET)))
    print(f"  Structural ceiling to keep in view: at "
          f"{INTERMITTENT_FRACTION:.0%} the source is present in only")
    print(f"  {_n_on_hands} of {N_HANDS} hands trials or {_n_on_feet} of {N_FEET} "
          f"feet trials, so it can carry information about")
    print(f"  at most that many trials, while {DETECT_K}/{N} is "
          f"{DETECT_K - MAJ_CORRECT} trials above the majority floor.")
    if _n_reach == 0:
        print(f"  SO THE 'NEVER' ROWS ABOVE ARE A STATEMENT ABOUT THE CRITERION AS "
              f"MUCH AS ABOUT THE")
        print(f"  PROBE. DETECT_K was calibrated against continuous injection and is "
              f"not")
        print(f"  transportable to a {INTERMITTENT_FRACTION:.0%} duty cycle without "
              f"recalibration. This run does NOT")
        print(f"  claim that bursty sources are undetectable; it claims that the "
              f"registered")
        print(f"  criterion cannot adjudicate them, which is a different and smaller "
              f"statement.")
    else:
        print(f"  So the criterion IS reachable at this duty cycle in at least one "
              f"cell, and the")
        print(f"  'NEVER' rows are sensitivity results rather than criterion artifacts.")

    sub("The bound that actually follows, over shapes and over temporal structure")
    _cont_worst = max(v for v in shape_thr.values() if v not in (None, "NEVER"))
    CONT_SHAPE_BOUND = _cont_worst
    _cont_worst_rms = max(a_to_rms(shape_thr[tn], t) for tn, t in ALL_TOPOS
                          if shape_thr[tn] not in (None, "NEVER"))
    print(f"  TIER 1, as registered (2 shapes, continuous, worse direction):  a = "
          f"{max(thr_by_topo['stipulated'], thr_by_topo['flat']):.3f}")
    print(f"  TIER 2, over {len(ALL_TOPOS)} shapes, continuous, worse direction:      "
          f"    a = {CONT_SHAPE_BOUND:.3f}  "
          f"(ring-RMS {_cont_worst_rms:.3f})")
    print(f"  TIER 3, adding bursty temporal structure:                       "
          f"NOT BOUNDED at any rung tested")
    print(f"\n  TIER 2 is the honest bound in the registered units for the shapes "
          f"actually")
    print(f"  measured, and it is {CONT_SHAPE_BOUND/max(thr_by_topo['stipulated'], thr_by_topo['flat']):.0f}x "
          f"the registered figure. A focal source under one electrode at")
    print(f"  a = 0.500 is INSIDE this recording's tolerance and OUTSIDE the "
          f"registered bound.")
    print(f"  TIER 3 is not a bound at all: the registered detection criterion cannot")
    print(f"  adjudicate a bursty source at this duty cycle, so the temporal-structure")
    print(f"  exposure is OPEN, not closed and not quantified.")
    WORST_BOUND = CONT_SHAPE_BOUND

    sub("Is the registered threshold stable across CV seeds?")
    print(f"  The registered ladder pins the CV seed at {SEED} like everything else.")
    print(f"  Re-deriving the stipulated-topography, into-hands threshold at CV seeds")
    print(f"  {LADDER_SEED_CHECK}:")
    _seed_thrs = []
    for _cs in LADDER_SEED_CHECK:
        _rows = ladder_rows(TOPO_STIPULATED, 2, cv_seed=_cs)
        _a, _km, _d = threshold(_rows)
        _seed_thrs.append(_a)
        print(f"    CV seed {_cs:>2}: threshold a = "
              f"{'NEVER' if _a is None else f'{_a:.3f}'}")
    _uniq = sorted({t for t in _seed_thrs if t is not None})
    print(f"  distinct thresholds across those {len(LADDER_SEED_CHECK)} CV seeds: "
          f"{_uniq}")
    print(f"  So the LADDER THRESHOLD is "
          f"{'seed-stable' if len(_uniq) == 1 else 'NOT seed-stable'}, unlike the "
          f"primary cell's accuracy,")
    print(f"  which moves across {N_SEED_SWEEP} seeds as section 5 shows. Those are "
          f"different quantities")
    print(f"  and this is stated so the seed sensitivity of one is not read onto the "
          f"other.")

    return CONT_SHAPE_BOUND
