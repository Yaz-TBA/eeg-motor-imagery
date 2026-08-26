#!/usr/bin/env python3
# SIZE-WAIVER: one guard, one contract. The registry with its measured-runtime
# comments, the extractor, the matcher and the exit contract are a single promise to
# the reader, and splitting the matcher from the registry it guards would hide that
# promise. Most of the overhang is the docstring and the registry comments.
"""Every number in the docs must come out of a script. This checks that.

Refuses to let a number appear in the docs unless a script in this repo produces
it, and exits FAIL on figures quoted inside retraction prose too, which is
expected; those get adjudicated by hand, not fixed. The failure mode it guards
against actually happened here: ablation accuracies (95.9% / 47.4% / 93.3%) sat
in README.md for weeks while no script produced them, and two were
arithmetically impossible, since on 45 trials accuracy is a multiple of 1/45.
Prose drifts off code silently. This makes it loud.

How it works:

  1. Extract numeric claims from README.md and EXPLAINER.md: percentages,
     p-values, correlations, a narrow class of counts.
  2. Run each analysis script, capture stdout, cache it in .provenance_cache/
     (tracked in git, deliberately; .gitignore says why), keyed by a hash of
     the script source so reruns are free until the script changes.
  3. Assert every claim appears in some captured stdout, rounding-tolerant
     (91.1 in the docs matches 91.11111 in the output).
  4. Print BACKED / ALLOWLISTED / WEAK / UNBACKED, where WEAK means the only
     backing line reads like a retraction; see RETRACTION_HINT.

The exit contract, stated exactly, because an overstated one is worse than none:

  exit 1  any UNBACKED claim, or any analysis script on disk the registry does
          not list, or, on a full run only, any registered script that produced
          no output (a crashed script must not turn a FAIL into a pass)
  exit 0  everything else, including INCOMPLETE, which is reachable only under
          --fast, where skipped slow scripts are expected to contribute nothing

WEAK is advisory and never affects the exit code. It cannot: this repo keeps
corrected claims visible inside retraction passages, and RETRACTION_HINT also
fires on innocent lines (the word "retracted" inside an estimator label puts
live figures in WEAK). Every WEAK entry needs a human to decide quoting from
republishing; a green exit does not mean the list was reviewed.

Known limits: matching is by value, not meaning, so a small integer can be
backed by coincidence and any printed number backs a claim. This catches
fabrication and drift, not misinterpretation.

Usage:
    python src/checks/check_provenance.py            # everything (~3 h, sum of REGISTRY)
    python src/checks/check_provenance.py --fast     # skip slow scripts, use their cache
    python src/checks/check_provenance.py --list     # show scripts + claims, run nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The repo root: this file lives in src/checks/, two levels down.
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / ".provenance_cache"
PYTHON = str(ROOT / ".venv" / "bin" / "python")
DOCS = ["README.md", "EXPLAINER.md"]

# script -> (expected runtime in seconds, slow?). "slow" means --fast skips it
# and falls back to whatever stdout is already cached. Runtimes are measured,
# not guessed; they set the subprocess timeout (4x headroom).
REGISTRY = {
    "load_and_plot.py":         (30,    False),
    "epoch_trials.py":          (10,    False),
    "filter_and_epoch.py":      (10,    False),
    "decode_csp.py":            (90,    False),
    # Six channel conditions at seed 42, a ten-seed sweep over five of them, a
    # secondary arm that rebuilds from the EDFs to re-reference inside 47
    # channels, and a 1000-shuffle permutation test on the complement. The old
    # 180 was a guess carried from when this script ran four single-seed
    # conditions. Runtime MEASURED 2026-07-25 on this machine, EEGBCI files
    # already fetched, permutation test at n_jobs=-1: 15.48 s and 15.65 s wall,
    # identical stdout on both except for an edit made between them. 20 is the
    # higher measurement rounded up, giving an 80 s timeout at the 4x rule above.
    # It is LOWER than the number it replaces because the old one was never
    # measured, not because the script got faster.
    #
    # RAISED 20 -> 60 on 2026-07-26. Three arms were added after an adversarial
    # pass: the exact McNemar re-run on all ten sweep seeds (two extra
    # cross_val_predict calls per seed), a rank and identity check on the
    # re-referenced secondary arm, and POST-REGISTRATION arm 10, a 50-draw
    # random-17-channel-deletion null swept over the same ten seeds, which is 500
    # extra 47-channel CV fits on its own. Runtime MEASURED on this machine after
    # those additions: 52.19 s and 53.36 s wall, stdout byte-identical across
    # both. 60 is the higher measurement rounded up, giving a 240 s timeout at
    # the 4x rule above.
    "ablate_channels.py":       (60,    False),
    # The EMG probe: the pipeline refit on 40-75 Hz at the temporal ring, which
    # is the arm ablate_channels.py structurally CANNOT run, because the
    # published pipeline band-passes to 8-30 Hz and discards the band an EMG
    # probe needs before any covariance is computed. Pre-registered in
    # prereg/prereg-emg-proxy.md. Cheap despite four 1000-shuffle
    # permutation tests and a 400-run injection ladder, because every channel set
    # except ALL64 is 8 or 17 channels wide. Runtime MEASURED 2026-07-25 on this
    # machine, EEGBCI files already fetched, permutation tests at n_jobs=-1:
    # 32.69 s and 35.21 s wall, stdout byte-identical across both runs apart from
    # the line that prints the runtime itself. 40 is the higher measurement
    # rounded up, giving a 160 s timeout at the 4x rule above.
    #
    # RAISED 40 -> 90 on 2026-07-26. The script grew four post-registration arms
    # after an adversarial pass: a 100-CV-seed sweep of the primary cell, the
    # ladder over two extra focal topographies, an intermittent-injection ladder
    # over all four shapes and both directions, and a bursty saturation probe at
    # three amplitudes past the top rung. Runtime MEASURED on this machine after
    # those additions: 69.2 s and 75.8 s wall. 80 is the higher measurement
    # rounded up; 90 leaves headroom for the machine being busy, giving a 360 s
    # timeout at the 4x rule. It is HIGHER than the number it replaces because
    # the script does more, not because anything got slower.
    "emg_proxy.py":             (90,    False),
    "harder_contrast.py":       (120,   False),
    "evaluate_honestly.py":     (300,   True),   # 100-seed x 2-estimator sweep
    "sweep_subjects.py":        (1800,  True),   # all 109 subjects
    "cross_subject.py":         (900,   True),   # leave-one-subject-out, n=20
    # 5 pipeline arms: 2 methods (MDM, TangentSpace+LR) x 2 montages (64 ch,
    # sensorimotor subset) + the CSP+LDA baseline they are paired against.
    "riemannian.py":            (900,   True),   # 20 subjects, 5 pipeline arms
    "eegnet_compare.py":        (3600,  True),   # CNN training, 3 regimes
    # 7 cells, not 4: the 2x2 band x crop factorial plus original-C, cue-only and
    # pre-cue. The pre-cue control is the one rung 11's conclusion rests on, so a
    # comment that says "2x2" understates what this script has to run.
    "regime_decomposition.py":  (3600,  True),   # CNN training, 7 cells
    # Re-analysis of arrays already on disk, plus three small measurements on
    # subject 1. Cheap because it trains nothing it does not have to, so it is
    # NOT slow: --fast runs it rather than falling back to a cache. Runtime
    # MEASURED 2026-07-25 on this machine, EEGBCI files already fetched:
    # 11.90 s and 12.18 s wall, byte-identical stdout both times. 15 is that
    # measurement rounded up, giving a 60 s timeout at the 4x rule above.
    "inferential_stats.py":     (15,    False),
    # Arm A: 3 subjects x 4 cells x 10,000 permutation replicates, each one a
    # full 5-fold CSP+LDA refit, plus 2 exact label-free cells at N=2,000 and a
    # 6-rule x 3-subject exactness study (200 H0 vectors x 199 inner
    # permutations). Arm B: 2 cells x 2,000 replicates of a 20-fold LOSO over
    # 900 pooled trials, which is the expensive half by an order of magnitude.
    #
    # TWO RUNTIMES, MEASURED, and the registry has to carry the LARGER one
    # because the smaller one is only available to a machine that has already
    # paid for it. The script checkpoints each block of replicates to
    # .permutation_design_cache/, stamped with a fingerprint of every input, so
    # a re-run reuses completed blocks:
    #   COLD  (no cache): 20:44:01 -> 01:33:11 on 2026-07-25/26, i.e. 17,350 s
    #                     wall, read off the script's own timestamped stderr.
    #   WARM  (all blocks cached): 159.99 s wall, MEASURED 2026-07-26.
    # 17400 is the cold measurement rounded up. That gives a 69,600 s timeout at
    # the 4x rule, which is absurdly generous for a warm run and is exactly
    # right for a cold one -- and a cold run is what a fresh clone gets. Marked
    # SLOW so --fast skips it and uses the cache, like the other multi-hour
    # scripts. Do NOT lower this to the warm number: deleting the cache would
    # then turn a working script into a timeout, which check_provenance.py
    # reports as FAIL on a full run.
    "permutation_design.py":    (17400, True),

    # The G-S gate registered in prereg-permutation-validity.md §4. Deterministic
    # and data-free: it instantiates each rule under two observed vectors and
    # compares the realised partition and permutation group, so there is no
    # estimator, no Monte Carlo and nothing to cache. Runtime MEASURED 2026-07-30
    # on this machine: 2.09 s and 2.08 s wall, identical stdout. 10 gives a 40 s
    # timeout at the 4x rule. NOT slow: --fast runs it rather than reading a cache,
    # which is the point of a gate that costs two seconds.
    "validity_gate.py":         (10,    False),
}

# ---------------------------------------------------------------------------
# Script locations
# ---------------------------------------------------------------------------

# The scripts live under src/, in the three groups the README names. REGISTRY,
# NON_ANALYSIS, the .provenance_cache/ filenames and meta.json all stay keyed by bare
# script name, so the tracked cache survives moves byte-for-byte; this is the one
# place a name resolves to a path. "src" and "." keep common.py and any root-level
# script visible to unregistered_scripts().
SCRIPT_DIRS = ("src/pipeline", "src/attacks", "src/checks", "src", ".")


def script_path(script: str) -> Path:
    for d in SCRIPT_DIRS:
        p = ROOT / d / script
        if p.exists():
            return p
    return ROOT / script


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# A "claim" is a number the docs assert about THIS project's results. Three
# kinds are extracted, in priority order (first match on a span wins):
#   pct    91.1%          any percentage
#   stat   p <= 0.001     p-values and correlations, r = 0.57
#   count  45 trials      integers bound to a result-bearing noun
# Everything else in the prose is INVISIBLE to this tool, and not all of it is
# harmless. Two different things get conflated here, so keep them apart:
#
#   Ignored on purpose, and safe: Hz, seconds, run numbers, section numbers,
#   electrode names. These are settings and structure, not results.
#
#   ALSO INVISIBLE, AND THESE ARE RESULTS: point-differences ("0.2 points",
#   "3.5 points", "17.8 points"), multipliers ("53x", "4500x"), microvolts
#   ("+11.89 uV"), t-statistics ("t = +7.71"), k/45 fractions, parameter counts
#   ("2,290 parameters"), and scientific notation ("1.3e-5", "1.6e-10").
#
# None of that second group is excluded by UNIT_SUFFIX or ALLOWLIST. It is
# excluded because PATTERNS below simply never matches it. A fabricated
# point-difference or multiplier passes this tool in silence -- which is how a
# retracted causal claim, expressed entirely as a point-difference, survived
# several review rounds without this checker ever seeing it. Extending PATTERNS to
# point-differences and multipliers would close the largest part of the gap.
PATTERNS = [
    ("pct",   re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*%")),
    ("stat",  re.compile(r"\b(?:p|r)\s*(?:=|<=|<|≤)\s*(\d*\.\d+)")),
    ("count", re.compile(r"(?<![\w.])(\d[\d,]{0,6})[\s-]+"
                         r"(?:trials?|subjects?|shuffles?|seeds?|folds?)\b")),
]

# Numbers carrying these units are never result claims -- they are experimental
# settings (band edges, crop windows, sampling rate) that live in the code as
# constants, not in its stdout.
UNIT_SUFFIX = re.compile(r"^\s*(?:hz|khz|s\b|sec|ms|µv|uv|v\b|db|gb|mb)", re.I)

# "95% CI" / "95% confidence" is a confidence LEVEL, not a measured quantity.
# Excluded by context rather than by value so that a genuine 95% accuracy
# elsewhere in the docs still gets checked.
CONTEXT_EXEMPT = re.compile(r"^\s*(?:CI\b|confidence)", re.I)

# Values excused from needing a script behind them. Each one needs a reason.
ALLOWLIST = {
    # Dataset facts, fixed by PhysioNet, not measured by us.
    "count:109": "EEGBCI subject count -- a property of the dataset",
    "count:64":  "channel count -- a property of the dataset",
    "count:14":  "runs per subject -- a property of the dataset",
    # Design constants chosen in code, printed only sometimes.
    "count:1000": "permutation shuffle count -- a knob, not a result",
    "count:100": "seed-sweep size -- a knob, not a result",
    "count:5":   "k in stratified 5-fold -- a knob, not a result",
    "count:10":  "n_splits in the retired ShuffleSplit(10, 0.2) estimator -- a "
                 "knob, not a result. NOT a 10-fold: that framing is retracted, "
                 "ShuffleSplit is not a partition (see evaluate_honestly.py 2)",
    "count:20":  "subject subset size for LOSO -- a knob, not a result",
    # Forward-looking prose in README "Next" / EXPLAINER roadmap. These describe
    # corpora this project has NOT run, so no script can back them.
    "count:2000": "aspirational trial count for other public corpora",
    "count:5000": "aspirational trial count for other public corpora",
    # NO percentage is allowlisted, deliberately. This block used to carry
    # "pct:50.0", "pct:100.0" and "pct:0.0", and both halves of that were wrong:
    #
    #   1. They never fired. Claim.key renders a whole-number value as "pct:50",
    #      not "pct:50.0" (see the int() in Claim.key), so those three strings
    #      could not match an extracted claim. They read as active policy while
    #      exempting nothing.
    #   2. The policy itself was unsafe, because these keys exempt by VALUE, with
    #      no regard for context:
    #        - 50.0 is not only "theoretical chance". regime_decomposition.py
    #          tests against 0.5 while EXPLAINER.md describes a pooled
    #          majority-class floor of 50.1%. Exempting every 50.0% would hide
    #          exactly that mismatch.
    #        - 100.0 is not only a rhetorical ceiling. It is a MEASURED maximum
    #          here: sweep_subjects.py prints "max 100.0% (S070)", a clean 45/45.
    #
    # If a percentage ever genuinely needs exemption, exempt it by CONTEXT the way
    # CONTEXT_EXEMPT handles "95% CI", not by value.
}


@dataclass
class Claim:
    kind: str
    value: float
    raw: str
    doc: str
    line: int
    context: str

    @property
    def key(self) -> str:
        # 45.0 -> "45" so the allowlist reads naturally for counts.
        v = int(self.value) if self.value == int(self.value) else self.value
        return f"{self.kind}:{v}"

    @property
    def decimals(self) -> int:
        return len(self.raw.split(".")[1]) if "." in self.raw else 0


def strip_code_fences(text: str) -> tuple[list[str], int]:
    """Blank out fenced code blocks; return (lines, numbers_dropped).

    Fenced blocks are pasted terminal output and shell commands -- they are
    reproductions, not prose claims, and they are checked by rerunning the
    script rather than by string-matching. Their numbers are counted and
    reported so the exemption stays visible.
    """
    lines, out, dropped, in_fence = text.split("\n"), [], 0, False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
        elif in_fence:
            dropped += len(re.findall(r"\d", ln))
            out.append("")
        else:
            out.append(ln)
    return out, dropped


def extract_claims(doc: Path) -> tuple[list[Claim], int]:
    lines, dropped = strip_code_fences(doc.read_text())
    claims, seen = [], set()
    for i, line in enumerate(lines, 1):
        # Markdown headings ("## 12. Scoreboard") and link targets are structure.
        if line.lstrip().startswith("#") or line.lstrip().startswith("["):
            continue
        for kind, pat in PATTERNS:
            for m in pat.finditer(line):
                tail = line[m.end():]
                if UNIT_SUFFIX.match(tail) or CONTEXT_EXEMPT.match(tail):
                    continue
                raw = m.group(1).replace(",", "")
                c = Claim(kind, float(raw), raw, doc.name, i,
                          line.strip()[:90])
                if (c.key, c.context) in seen:
                    continue
                seen.add((c.key, c.context))
                claims.append(c)
    return claims, dropped


# ---------------------------------------------------------------------------
# Running scripts / caching stdout
# ---------------------------------------------------------------------------

# Scripts on disk that are not analysis: they produce no number this project
# claims, so the registry must not run them and their stdout must never enter
# the evidence pool. Exempting by name with a written reason mirrors ALLOWLIST;
# a pattern-based rule would make "deliberately exempt" indistinguishable from
# "forgotten". Adding a script here asserts that its stdout backs no doc claim.
# If that stops being true, register it instead.
NON_ANALYSIS = {
    "status.py": "writes STATUS.json, a machine-written account of git state "
                 "(branch, dirty counts, default-branch SHA, gh visibility). "
                 "It measures the repo, not the EEG data, and no claim in "
                 "README.md or EXPLAINER.md comes from it. Registering it "
                 "would be harmful: its stdout carries incidental integers "
                 "that could back a doc claim by coincidence, it shells out "
                 "to `gh` over the network, and it writes a file.",
    "check_wording.py": "a guard, not analysis. It greps README.md and "
                        "EXPLAINER.md for outcome wording a pre-registration "
                        "banned and prints only pass/fail plus offending "
                        "excerpts. It measures the prose, not the EEG data.",
    "common.py": "a library. It defines the published pipeline, the Wilson "
                 "and Holm helpers, the lattice check and the units guard, "
                 "and running it produces no stdout at all, so there is "
                 "nothing for a claim to match against. The scripts that "
                 "import it are registered; that is where its numbers "
                 "surface. Added 2026-08-04, after it was extracted in "
                 "8419ddd and left in neither list -- which made the guard "
                 "exit FAIL for a reason README.md never disclosed.",
    "test_pipeline.py": "a test suite. Its stdout is nineteen PASS lines and "
                        "a count, and every figure inside it is an assertion "
                        "about a number that is already registered somewhere "
                        "else. Registering it would let a doc claim be backed "
                        "by its own test rather than by the run that produced "
                        "it, which inverts the direction the guard is for. "
                        "Added 2026-08-04, same commit gap as common.py.",
    # The five infstats_* modules are inferential_stats.py, split into readable
    # pieces on 2026-08-26. Each is import-only: its printing happens inside that
    # script's registered run, so registering any of them would double-count the
    # same stdout, the same argument as common.py above.
    "infstats_lib.py": "estimators, loaders and constants of inferential_stats.py; "
                       "prints nothing on import.",
    "infstats_rungs_8_9.py": "sections 0-4 of inferential_stats.py.",
    "infstats_rungs_10_11.py": "sections 5, 9 and 10 of inferential_stats.py.",
    "infstats_measured.py": "sections 6-8 of inferential_stats.py, the three "
                            "figures measured rather than re-derived.",
    "infstats_ledger.py": "sections 11 and 12 of inferential_stats.py.",
    # The six ablation_* modules are ablate_channels.py, split the same way on the
    # same day. Import-only for the same reason.
    "ablation_data.py": "constants, loader and channel sets of ablate_channels.py.",
    "ablation_design.py": "the decision-rule arithmetic of ablate_channels.py.",
    "ablation_conditions.py": "the six registered conditions and controls.",
    "ablation_sweep.py": "the ten-seed sweep, per-seed McNemar and arm 10.",
    "ablation_secondary.py": "the secondary arms: re-reference, permutation, "
                            "Wilson, paired McNemar.",
    "ablation_verdict.py": "the falsifiers, the registered verdict and the "
                           "every-band caveats.",
    # The seven emg_* modules are emg_proxy.py, split the same way on the same
    # day. Import-only for the same reason.
    "emg_setup.py": "sections 0-2 of emg_proxy.py: constants, data, bands, epochs.",
    "emg_psd.py": "section 3 of emg_proxy.py, the PSD diagnostic.",
    "emg_sharp.py": "sections 4-5 of emg_proxy.py: positive control and arm (b).",
    "emg_univariate.py": "sections 6-7 of emg_proxy.py: arm (a) and the pre-cue "
                         "diagnostic.",
    "emg_ladder.py": "section 8 of emg_proxy.py, the sensitivity ladder.",
    "emg_ladder_ext.py": "section 8B of emg_proxy.py, the post-registration "
                         "ladder extension.",
    "emg_verdict.py": "section 9 of emg_proxy.py, the pre-registered reading.",
    # The nine permdesign_* modules are permutation_design.py, split the same way
    # on the same day. Import-only for the same reason.
    "permdesign_lib.py": "constants and shared helpers of permutation_design.py.",
    "permdesign_workers.py": "schemes, scoring workers and block checkpointing.",
    "permdesign_setup.py": "section 0 of permutation_design.py.",
    "permdesign_gates.py": "sections 1-2, the two falsification gates.",
    "permdesign_dummy.py": "sections 2B-2C, the centring departure and the "
                          "zero-information exactness study.",
    "permdesign_arm_a.py": "section 3, arm A and its per-subject report.",
    "permdesign_mechanism.py": "section 4, the secondary mechanism probe.",
    "permdesign_arm_b.py": "section 5, arm B and the SHUFFLE_MAX replacement.",
    "permdesign_verdict.py": "section 6, the pre-registered outcomes.",
}


def unregistered_scripts() -> list[str]:
    """Analysis scripts on disk that the registry does not know about.

    Half of this tool's value is noticing when the repo grows a script whose
    output nobody is checking. Two exemptions, both explicit: this file, and
    anything named in NON_ANALYSIS. There is no implicit exemption -- a helper
    nobody has justified in writing shows up here and fails the run.
    """
    return sorted(p.name for d in SCRIPT_DIRS for p in (ROOT / d).glob("*.py")
                  if p.name not in REGISTRY
                  and p.name not in NON_ANALYSIS
                  and p.name != Path(__file__).name)


def cache_path(script: str) -> Path:
    return CACHE / f"{script}.txt"


def source_hash(script: str) -> str:
    return hashlib.sha256(script_path(script).read_bytes()).hexdigest()[:12]


def run_script(script: str, budget: int) -> str | None:
    """Run a script and return its STDOUT, or None if it failed to produce any.

    Only stdout counts. MNE's own logging goes to stderr and is full of
    incidental numbers (filter lengths, sample counts, file sizes) that would
    accidentally "back" a doc claim it has nothing to do with.
    """
    print(f"  running {script} (budget {budget}s) ...", flush=True)
    try:
        proc = subprocess.run([PYTHON, str(script_path(script))], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=budget * 4)
    except subprocess.TimeoutExpired:
        print(f"  ! {script} timed out after {budget * 4}s", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"  ! {script} exited {proc.returncode}:", file=sys.stderr)
        print("    " + "\n    ".join(proc.stderr.strip().split("\n")[-5:]),
              file=sys.stderr)
        return None
    return proc.stdout


def gather_outputs(fast: bool) -> tuple[str, list[str]]:
    """Return (concatenated stdout, list of scripts with no usable output)."""
    CACHE.mkdir(exist_ok=True)
    meta_file = CACHE / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    blobs, missing = [], []

    for script, (budget, slow) in REGISTRY.items():
        if not script_path(script).exists():
            missing.append(f"{script} (registered but not on disk)")
            continue
        path, cached = cache_path(script), None
        if path.exists():
            cached = path.read_text()
        stale = meta.get(script) != source_hash(script)

        if fast and slow:
            if cached is None:
                missing.append(f"{script} (slow, skipped, no cache)")
                continue
            if stale:
                print(f"  ~ {script}: cache is stale (source changed) -- "
                      f"using it anyway under --fast")
        elif cached is None or stale:
            # Hash BEFORE running: a concurrent edit mid-run must not be
            # recorded as "this cache matches that source".
            h = source_hash(script)
            out = run_script(script, budget)
            if out is None:
                missing.append(f"{script} (failed or timed out)")
                continue
            path.write_text(out)
            meta[script] = h
            meta_file.write_text(json.dumps(meta, indent=2))
            cached = out
        else:
            print(f"  cached {script}")
        blobs.append(cached)

    return "\n".join(blobs), missing


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

NUM_IN_OUTPUT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")

# Several scripts here print a number in order to RETRACT it -- ablate_channels.py
# prints "So 95.9% and 47.4% -- the two numbers the old README table carried"
# precisely because those were fabricated. String-matching stdout would happily
# call those numbers "backed". A claim whose only backing line reads like
# retraction prose is reported separately (WEAK) instead of silently passing.
RETRACTION_HINT = re.compile(
    r"\b(?:old|older|earlier|previous|retract\w*|wrong|impossible|"
    r"fabricat\w*|README|not a measurement|used to)\b", re.I)


def half_up(value: float, digits: int) -> float:
    """Round half away from zero, the way prose does it.

    Python's round() is banker's rounding: round(96.5) == 96, which would
    wrongly flag a doc that writes a 96.5% Wilson bound as "roughly 97%".
    """
    scale = 10 ** digits
    return math.floor(abs(value) * scale + 0.5) / scale * (1 if value >= 0 else -1)


def output_lines(blob: str) -> list[tuple[float, str, str]]:
    """Every number in captured stdout, with its line and what follows it.

    The trailing text is kept because a bare float carries no meaning: "57.5"
    is an accuracy, a frequency or a millisecond count depending entirely on
    the characters after the digits, and dropping them makes those the same
    number.
    """
    nums = []
    for line in blob.split("\n"):
        for m in NUM_IN_OUTPUT.finditer(line):
            nums.append((float(m.group(1)), line.strip(), line[m.end():]))
    return nums


def find_backing(claim: Claim, nums: list[tuple[float, str, str]]) -> str | None:
    """Return the best stdout line backing this claim, or None.

    Rounding-tolerant: 91.1 in docs matches 91.11111 in stdout. Percentages
    additionally match their fractional form (91.1% <- 0.911), because some
    scripts print fractions and some print percents. A non-retraction line
    always wins over a retraction line.
    """
    d, weak = claim.decimals, None
    for value, line, tail in nums:
        hit = (half_up(value, d) == claim.value or
               (claim.kind == "pct" and half_up(value * 100, d) == claim.value))
        if not hit:
            continue
        # A percentage is never backed by a number wearing a physical unit:
        # "57.5 Hz" is a power-spectrum bin, not an accuracy. UNIT_SUFFIX is the
        # same rule the doc side already applies at extraction, read here on the
        # output side.
        #
        # The rule is ONE-DIRECTIONAL and its mirror is NOT available: an
        # explicit disqualifying unit rejects, a missing "%" never does. Scripts
        # print a bare "91.1" meaning percent, and every fraction-form match two
        # lines up is a number no "%" can follow. MEASURED 2026-07-30 on this
        # repo: demanding the sign instead takes BACKED 498 -> 492, unbacking a
        # CI bound that stdout prints as "[ 50.6,  56.8]" and five more like it.
        if claim.kind == "pct" and UNIT_SUFFIX.match(tail):
            continue
        if RETRACTION_HINT.search(line):
            weak = weak or line
        else:
            return line
    return weak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fast", action="store_true",
                    help="skip slow scripts; use their cached stdout")
    ap.add_argument("--list", action="store_true",
                    help="show scripts and extracted claims, run nothing")
    args = ap.parse_args()

    claims, dropped = [], 0
    for name in DOCS:
        c, d = extract_claims(ROOT / name)
        claims += c
        dropped += d

    if args.list:
        print("SCRIPTS")
        for s, (b, slow) in REGISTRY.items():
            mark = "SLOW" if slow else "fast"
            hit = "cached" if cache_path(s).exists() else "no cache"
            gone = "" if script_path(s).exists() else "  MISSING FROM DISK"
            print(f"  {s:<28} {mark:<5} ~{b:>5}s  [{hit}]{gone}")
        for s in unregistered_scripts():
            print(f"  {s:<28} NOT IN REGISTRY -- its output is never checked")
        # Print the exemptions too. An exemption that prints nothing is
        # indistinguishable from a check that never ran, which is the confusion
        # this tool exists to remove.
        for s in sorted(NON_ANALYSIS):
            gone = "" if script_path(s).exists() else "  NOT ON DISK -- stale exemption"
            print(f"  {s:<28} NOT ANALYSIS -- exempt by name{gone}")
        print(f"\nCLAIMS ({len(claims)} extracted, "
              f"{dropped} digits inside code fences exempted)")
        for c in claims:
            tag = "allowlist" if c.key in ALLOWLIST else "check"
            print(f"  {c.doc}:{c.line:<5} {c.kind:<6} {c.raw:<8} {tag:<10} "
                  f"{c.context}")
        return 0

    print(f"Gathering script output{' (--fast)' if args.fast else ''} ...")
    blob, missing = gather_outputs(args.fast)
    nums = output_lines(blob)
    print(f"  {len(nums)} numbers captured from "
          f"{len(REGISTRY) - len(missing)} scripts")
    for m in missing:
        print(f"  ! no output for {m}")

    backed, weak, unbacked, allowed = [], [], [], []
    for c in claims:
        if c.key in ALLOWLIST:
            allowed.append(c)
            continue
        line = find_backing(c, nums)
        if line is None:
            unbacked.append(c)
        elif RETRACTION_HINT.search(line):
            weak.append((c, line))
        else:
            backed.append(c)

    stray = unregistered_scripts()
    for s in stray:
        print(f"  ! {s} is not in REGISTRY -- its output is never checked")

    print(f"\n{'=' * 72}\nPROVENANCE REPORT\n{'=' * 72}")
    print(f"BACKED      {len(backed):>4}   number appears in some script's stdout")
    print(f"ALLOWLISTED {len(allowed):>4}   exempt by documented rule")
    print(f"WEAK        {len(weak):>4}   only backed by a line that reads as retraction")
    print(f"UNBACKED    {len(unbacked):>4}   no script produces this number")

    if weak:
        print(f"\n{'-' * 72}\nWEAKLY BACKED (check by hand: is the doc quoting a "
              f"retracted number,\nor republishing it?)\n{'-' * 72}")
        for c, line in sorted(weak, key=lambda x: (x[0].doc, x[0].line)):
            print(f"{c.doc}:{c.line} [{c.kind}] {c.raw}")
            print(f"    doc:  {c.context}")
            print(f"    only: {line[:88]}")

    if unbacked:
        print(f"\n{'-' * 72}\nUNBACKED CLAIMS\n{'-' * 72}")
        for c in sorted(unbacked, key=lambda x: (x.doc, x.line)):
            print(f"{c.doc}:{c.line} [{c.kind}] {c.raw}")
            print(f"    {c.context}")

    if missing:
        print(f"\nNOTE: {len(missing)} script(s) contributed no output, so some "
              f"UNBACKED entries\n      above may simply be unverified.")
        # Gated on --fast. Under --fast, missing output is EXPECTED (slow scripts
        # are skipped by design) and an incomplete run must not fail the build.
        # On a FULL run it means a registered script crashed, timed out, or is not
        # on disk -- and returning 0 there would let a broken script convert a
        # genuine FAIL into a pass, which is the exact failure this tool exists to
        # catch.
        if args.fast:
            print("      Run without --fast to settle them.")
            print("Result: INCOMPLETE (exit 0 -- --fast cannot fail the build)")
            return 0
        print("      This is a FULL run, so that is a broken script, not a "
              "skipped one.")
        print("Result: FAIL (registered script produced no output on a full run)")
        return 1

    bad = bool(unbacked) or bool(stray)
    print(f"\nResult: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
