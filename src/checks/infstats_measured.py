"""Sections 6-8 of inferential_stats.py: the three figures no persisted array can
supply, measured directly on subject 1. Split out 2026-08-26. The seeding order and the
constants are identical to eegnet_compare.py's experiment A, which is the whole point."""

import numpy as np
from scipy import stats

from infstats_lib import (
    BATCH_SIZE, BN_EPS, LR, NARROW, N_EPOCHS, N_SCALE_SEEDS, RUNS, SEED, SUBJECT,
    cannot, head, sub,
)

# ---------------------------------------------------------------------------
# 6-8. The three figures that no persisted array can supply
# ---------------------------------------------------------------------------

def load_subject_one():
    """Load subject 1 exactly as eegnet_compare.py's experiment A does."""
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    paths = eegbci.load_data(subjects=SUBJECT, runs=RUNS, update_path=True)
    raw = mne.concatenate_raws(
        [mne.io.read_raw_edf(p, preload=True) for p in paths])
    eegbci.standardize(raw)
    raw.set_montage("standard_1005")
    raw.set_eeg_reference("average", projection=False)
    raw.filter(NARROW["l_freq"], NARROW["h_freq"], fir_design="firwin",
               skip_by_annotation="edge")
    events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    epochs = mne.Epochs(raw, events, dict(hands=2, feet=3), tmin=-1.0, tmax=4.0,
                        picks="eeg", baseline=None, preload=True)
    X = epochs.copy().crop(*NARROW["crop"]).get_data(copy=False)
    y = (epochs.events[:, -1] == 3).astype(np.int64)
    return X, y

def section_mcnemar(X, y):
    head("6. RUNG 10 -- McNEMAR ON THE WITHIN-SUBJECT COMPARISON")
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import EEGNet
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    print("NO ARRAY ON DISK HOLDS THIS. McNemar needs the discordant pair counts,")
    print("which need the two models' PER-TRIAL predictions on the same folds.")
    print("eegnet_compare.py scores its folds and discards the predictions, so this")
    print("section re-runs experiment A to recover them: same 45 trials, same")
    print("StratifiedKFold(5, shuffle=True, random_state=42), same seed, same model.")
    print("THE SEEDING ORDER IS COPIED, NOT APPROXIMATED. eegnet_compare.py seeds")
    print("once and lets cross_val_score clone across folds, so the five folds do")
    print("not restart from the same RNG state. Reseeding per fold changes four of")
    print("the five folds and moves the EEGNet count by several trials, which is")
    print("enough on its own to make a McNemar table irreproducible.")
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    csp = Pipeline([
        ("CSP", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ("LDA", LinearDiscriminantAnalysis()),
    ])
    pred_csp = cross_val_predict(csp, X, y, cv=cv)

    seed_torch()
    net = EEGClassifier(
        module=EEGNet, module__n_chans=X.shape[1], module__n_outputs=2,
        module__n_times=X.shape[2], optimizer=torch.optim.AdamW,
        optimizer__lr=LR, optimizer__weight_decay=1e-4,
        batch_size=BATCH_SIZE, max_epochs=N_EPOCHS, train_split=None,
        device=device, verbose=0, callbacks=[])
    # microvolts: the working configuration every reported accuracy comes from
    pred_net = cross_val_predict(net, (X * 1e6).astype(np.float32), y, cv=cv)

    k_csp = int((pred_csp == y).sum())
    k_net = int((pred_net == y).sum())
    print(f"  CSP + LDA correct     {k_csp}/{len(y)} = {100 * k_csp / len(y):.1f}%")
    print(f"  EEGNet correct        {k_net}/{len(y)} = {100 * k_net / len(y):.1f}%")
    print(f"  EEGNet - CSP          {100 * (k_net - k_csp) / len(y):+.1f} points")
    counts = np.bincount(pred_net, minlength=2)
    print(f"  EEGNet predicted class counts {counts.tolist()} "
          f"(true {np.bincount(y, minlength=2).tolist()})")

    b = int(((pred_csp == y) & (pred_net != y)).sum())
    c = int(((pred_csp != y) & (pred_net == y)).sum())
    both = int(((pred_csp == y) & (pred_net == y)).sum())
    neither = int(((pred_csp != y) & (pred_net != y)).sum())
    sub("the 2x2 agreement table")
    print(f"  both correct            {both}")
    print(f"  CSP only  (b)           {b}")
    print(f"  EEGNet only (c)         {c}")
    print(f"  neither correct         {neither}")
    if b + c == 0:
        print("  No discordant pairs; McNemar is undefined here.")
        return
    p_exact = float(stats.binomtest(b, b + c, 0.5).pvalue)
    print(f"  McNemar exact, two-sided, binomial on {b + c} discordant pairs")
    print(f"  McNemar p = {p_exact:.3f}")
    print("  ASSUMPTION: exact binomial rather than the chi-square approximation,")
    print(f"  because {b + c} discordant pairs is far too few for the asymptotic form.")
    print()
    print("  THE MARGINAL COUNTS DO NOT DETERMINE THE SPLIT. Any (b, c) with")
    print(f"  b - c = {k_csp - k_net} is consistent with {k_csp}/{len(y)} against "
          f"{k_net}/{len(y)}, and the p-value")
    print("  depends on which one it is, not on the difference. The maximally nested")
    print(f"  split, b = {k_csp - k_net} and c = 0, would give p = "
          f"{float(stats.binomtest(k_csp - k_net, k_csp - k_net, 0.5).pvalue):.3f}; the")
    print(f"  measured split of b = {b}, c = {c} gives p = {p_exact:.3f}. Deriving a")
    print("  McNemar p from two accuracies alone is arithmetic on an assumption about")
    print("  agreement that the predictions themselves settle.")
    print("  What this does NOT show: a non-significant McNemar on 45 trials leaves")
    print("  the direction of the difference undecided. Only the sign of b - c is")
    print("  consistent with the point estimate; its magnitude is not established.")
    print("  DETERMINISM CAVEAT: MPS kernels are not bit-reproducible, so the EEGNet")
    print("  half of this table can shift by a trial or two between runs.")

def seed_torch(seed=SEED):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def section_bn_scale(X):
    head("7. RUNG 10 -- THE BATCHNORM ACTIVATION-SCALE DEFICIT, MEASURED")
    import torch
    import torch.nn as nn
    from braindecode.models import EEGNet

    print("NO ARRAY ON DISK HOLDS THIS EITHER. No persisted artefact in this repo")
    print("records a per-layer activation standard deviation, so the deficit figure")
    print("has never been produced by a script. This section measures it directly:")
    print("forward hooks on every BatchNorm2d, one batch in a no-grad forward pass,")
    print("dropout disabled so the only thing varying is scale.")
    print()
    print("HOW THE MEASUREMENT IS DEFINED. BatchNorm divides by sqrt(var + eps). If")
    print("the input variance is far below eps the divisor is essentially sqrt(eps),")
    print("so the stage's output lands sqrt(eps)/sigma too small instead of at unit")
    print(f"scale. eps = {BN_EPS:g}, so sqrt(eps) = {np.sqrt(BN_EPS):.4f}.")
    print()
    print(f"Signal scale, subject 1, {NARROW['l_freq']:.0f}-{NARROW['h_freq']:.0f} Hz, "
          f"{NARROW['crop'][0]:.1f}-{NARROW['crop'][1]:.1f} s:")
    print(f"  standard deviation in volts   {X.std():.3e}")
    print(f"  variance in volts             {X.var():.3e}")
    print(f"  ratio eps / variance          {BN_EPS / X.var():.2e}")

    bn_names = ["bnorm_temporal", "bnorm_1", "bnorm_2"]
    per_seed = {name: [] for name in bn_names}
    clf_in = {"volts": [], "microvolts": []}
    logits = {"volts": [], "microvolts": []}

    for s in range(N_SCALE_SEEDS):
        seed_torch(SEED + s)
        model = EEGNet(n_chans=X.shape[1], n_outputs=2, n_times=X.shape[2])
        caps = {}

        def make_hook(tag):
            def hook(_mod, inp, out):
                caps[tag] = (inp[0].detach(), out.detach())
            return hook

        for name, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) or name.endswith("conv_classifier"):
                mod.register_forward_hook(make_hook(name))
        model.train()
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.eval()

        for scale, tag in ((1.0, "volts"), (1e6, "microvolts")):
            batch = torch.tensor((X[:BATCH_SIZE] * scale).astype(np.float32))
            with torch.no_grad():
                model(batch)
            if tag == "volts":
                for name in bn_names:
                    act = caps[name][0]
                    var = act.transpose(0, 1).reshape(act.shape[1], -1).var(
                        dim=1, unbiased=False).mean().item()
                    per_seed[name].append((np.sqrt(var), np.sqrt(var + BN_EPS)))
            key = [k for k in caps if k.endswith("conv_classifier")][0]
            clf_in[tag].append(caps[key][0].std().item())
            logits[tag].append(caps[key][1].std().item())

    sub(f"per-stage deficit at volts scale (mean over {N_SCALE_SEEDS} seeds)")
    print(f"{'stage':<18}{'sigma in':>13}{'divisor':>13}{'deficit':>12}{'sd':>10}")
    for name in bn_names:
        sig = np.array([a for a, _ in per_seed[name]])
        div = np.array([b for _, b in per_seed[name]])
        ratio = div / sig
        print(f"{name:<18}{sig.mean():>13.4e}{div.mean():>13.4e}"
              f"{ratio.mean():>11.0f}x{ratio.std(ddof=1):>10.0f}")
    first = np.array([b / a for a, b in per_seed["bnorm_temporal"]])
    print()
    print(f"THE DEFICIT AT THE FIRST BATCHNORM IS {first.mean():.0f}x "
          f"(sd {first.std(ddof=1):.0f} over {N_SCALE_SEEDS} seeds).")
    print("  Its input sigma is the signal after one temporal convolution; the")
    print("  divisor is sqrt(eps) because the variance is seven orders below eps.")

    sub("end-to-end, measured rather than modeled")
    v_in, u_in = np.array(clf_in["volts"]), np.array(clf_in["microvolts"])
    v_lg, u_lg = np.array(logits["volts"]), np.array(logits["microvolts"])
    print(f"  classifier-input sd, volts       {v_in.mean():.4e}")
    print(f"  classifier-input sd, microvolts  {u_in.mean():.4e}")
    print(f"  END-TO-END DEFICIT AT THE CLASSIFIER INPUT  "
          f"{(u_in / v_in).mean():.0f}x  (sd {(u_in / v_in).std(ddof=1):.0f})")
    print(f"  logit sd, volts                  {v_lg.mean():.4e}")
    print(f"  logit sd, microvolts             {u_lg.mean():.4e}")
    print(f"  DEFICIT AT THE LOGITS            {(u_lg / v_lg).mean():.0f}x")
    print()
    print("  A RECOVERY MODEL OF 31.6x PER STAGE IS AN ASSUMPTION, NOT A MEASUREMENT.")
    print("  That model implies the deficit runs 4500x, then 142x, then 4.5x, then")
    print("  0.14x across three stages. The measured per-stage deficits above do not")
    print("  decay at that rate, and the measured end-to-end deficit printed here is")
    print("  the figure any downstream scale argument should use.")
    print("  What this does NOT show: this is the BROKEN volts configuration, which")
    print("  eegnet_compare.py's assertion now refuses to run. It is a diagnosis of a")
    print("  configuration kept for the record, not a current result. The microvolt")
    print("  column is the configuration every reported accuracy comes from.")
    return float((u_lg / v_lg).mean())

def section_weight_travel(X, y, end_to_end):
    head("8. RUNG 10 -- FINAL-LAYER WEIGHT TRAVEL AGAINST THE SCALE GAP")
    import torch
    from braindecode import EEGClassifier
    from braindecode.models import EEGNet
    from sklearn.model_selection import StratifiedKFold

    print("NO ARRAY ON DISK HOLDS THIS. Nothing records final-layer weight statistics")
    print("at init or after training, so the 'the network cannot train out of it'")
    print("argument has never had a measured margin. This section measures both ends.")
    print()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, _ = next(iter(cv.split(X, y)))
    print(f"One fold of experiment A: {len(train)} training trials, "
          f"{N_EPOCHS} epochs of AdamW at lr = {LR:g}, batch {BATCH_SIZE}.")
    print()

    results = {}
    for scale, tag in ((1.0, "volts"), (1e6, "microvolts")):
        seed_torch()
        clf = EEGClassifier(
            module=EEGNet, module__n_chans=X.shape[1], module__n_outputs=2,
            module__n_times=X.shape[2], optimizer=torch.optim.AdamW,
            optimizer__lr=LR, optimizer__weight_decay=1e-4,
            batch_size=BATCH_SIZE, max_epochs=N_EPOCHS, train_split=None,
            device=device, verbose=0, callbacks=[])
        clf.initialize()

        def final_weight():
            layer = dict(clf.module_.named_modules())["final_layer.conv_classifier"]
            return layer.weight.detach().cpu().numpy().copy()

        # Re-fetched rather than held: skorch may rebind module_ during fit, and
        # a stale reference silently reports zero travel, which reads exactly
        # like the finding this section exists to test.
        w0 = final_weight()
        clf.partial_fit((X[train] * scale).astype(np.float32), y[train])
        w1 = final_weight()
        results[tag] = (w0, w1)
        print(f"  {tag:<11} weight sd {w0.std():.4f} -> {w1.std():.4f}   "
              f"travel in sd {abs(w1.std() - w0.std()):.4f}   "
              f"mean |dw| {np.abs(w1 - w0).mean():.4f}")

    w0, w1 = results["volts"]
    init_sd = float(w0.std())
    achieved_sd = float(abs(w1.std() - w0.std()))
    achieved_abs = float(np.abs(w1 - w0).mean())
    if achieved_sd == 0.0 or achieved_abs == 0.0:
        cannot("the training margin",
               "the final layer did not move at all, which means the weights were "
               "read from a stale module reference rather than the trained one",
               "check that final_weight() is re-fetched after partial_fit")
        return
    print()
    print(f"  init sd is {init_sd:.4f}, and the mean per-weight move is "
          f"{achieved_abs:.4f}.")
    print("  Two different quantities have both been called 'the travel': the change")
    print("  in the weight standard deviation, and the mean absolute per-weight")
    print(f"  change. Here they differ by {achieved_abs / achieved_sd:.1f}x, so a "
          f"margin quoted without")
    print("  saying which one it is cannot be checked.")

    sub("the margin, computed from the MEASURED end-to-end deficit")
    required_sd = init_sd * end_to_end
    print(f"  end-to-end deficit at the logits          {end_to_end:.0f}x")
    print(f"  final-layer sd needed to close it         {required_sd:.3f}")
    print(f"  required travel                           {required_sd - init_sd:.3f}")
    print(f"  travel achieved in {N_EPOCHS} epochs (sd)         {achieved_sd:.4f}")
    print(f"  shortfall, sd definition                  "
          f"{(required_sd - init_sd) / achieved_sd:.0f}x")
    print(f"  travel achieved in {N_EPOCHS} epochs (mean |dw|)  {achieved_abs:.4f}")
    print(f"  shortfall, mean-|dw| definition           "
          f"{(required_sd - init_sd) / achieved_abs:.0f}x")
    print()
    print("  A MARGIN COMPUTED FROM AN ASSUMED 4.5x RESIDUAL GAP would give a")
    print(f"  required travel of {init_sd * 4.5 - init_sd:.3f} and a shortfall near "
          f"{(init_sd * 4.5 - init_sd) / achieved_abs:.1f}x.")
    print("  That gap comes from the 31.6x-per-stage recovery model, which section 7")
    print("  measures and does not confirm.")
    print("  What this does NOT show: a scale argument is not a training experiment.")
    print("  The direct evidence that the volts configuration does not train is the")
    print("  degenerate-prediction behavior, which the guard in eegnet_compare.py now")
    print("  refuses to reproduce. This section bounds the optimizer's reach; it does")
    print("  not prove the optimizer could never find another route.")
