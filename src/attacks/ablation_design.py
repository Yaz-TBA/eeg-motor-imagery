"""The decision-rule arithmetic of ablate_channels.py: what the registered McNemar
half can and cannot detect, computed from the design alone. Split out 2026-08-26.
Everything here conditions on nothing the run produces."""

from dataclasses import dataclass

from scipy import stats

ALPHA = 0.05   # the registered two-sided alpha, shared by every stage

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


# --- WHAT THE REGISTERED DECISION RULE CAN DETECT, COMPUTED BEFORE THE RUN ----
# ADDED 2026-07-26, after an adversarial pass showed the registered rule cannot
# fire across the lower third of its own band C. This block conditions on NOTHING
# that the run produces. It is arithmetic on the design, and every input to it
# (n = 45, alpha = 0.05, the paired 2x2 structure, G_THRESHOLD = 10.0) was fixed
# before the first run. It could have been printed then. It was not, and that is
# the defect being repaired.
def print_detection_floor(n, G_THRESHOLD):
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
    return _min_gap, _mg_ndisc, _mg_b, _mg_c, _mg_p


@dataclass(frozen=True)
class Ablation:
    """Everything the three verdict stages read but none of them change.

    These twenty values were threaded through run_falsifiers, run_verdict and
    run_closing as positional arguments, which put three functions at 20+ parameters.
    Nothing about the analysis changed when they were bundled: each stage still
    unpacks them into the same local names on its first line, so every body below
    that line is the code that produced the committed stdout.
    """

    by_name: object
    results: object
    ALL64: object
    SMC: object
    FP: object
    LORO: object
    NWIDE: object
    WIDE: object
    n: int
    majority: object
    maj_correct: object
    n_hands: object
    n_feet: object
    sweeps: object
    comp_sweep: object
    comp_seed42: object
    COMPLEMENT: object
    ch_names: object
    NOISE_BAND: object
    G_THRESHOLD: object
    N_RANDOM_DRAWS: object
    skf: object
    labels: object
