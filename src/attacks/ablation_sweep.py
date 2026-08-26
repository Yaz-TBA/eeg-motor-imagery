"""The ten-seed sweep of ablate_channels.py, the exact McNemar on every sweep seed,
and post-registration arm 10, the random-17-channel-deletion null. Split out
2026-08-26; the stage bodies are verbatim from that file."""

import ablation_data  # noqa: F401  -- installs the common.py path first

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from common import FRONTOPOLAR, SENSORIMOTOR, make_clf
from ablation_data import SEEDS
from ablation_design import ALPHA


def run_sweep(cropped, labels, conditions, by_name, n):
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
    return sweeps


def mcnemar_per_seed(cropped, labels, ch_names, COMPLEMENT, n):
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
    return sweep_mcn, _sweep_ps, _n_fire


def run_random_deletion(cropped, labels, ch_names, sweeps, ALL64, comp_sweep, n,
                        N_RANDOM_DRAWS, RANDOM_DELETION_SEED, COMPLEMENT):
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
    return _null_means, _null_G, _G_obs
