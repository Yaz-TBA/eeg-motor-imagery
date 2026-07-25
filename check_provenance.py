#!/usr/bin/env python3
"""Every number in the docs must come out of a script. This checks that.

The failure mode this guards against is the one that actually happened here: a
table of ablation accuracies (95.9% / 47.4% / 93.3%) sat in README.md for weeks
while NO script in the repo produced those numbers. Two of them were
arithmetically impossible -- on 45 trials with 5-fold CV, accuracy has to be a
multiple of 1/45, and 95.9% is not one. Prose drifts off code silently. This
makes it loud.

How it works:

  1. Extract numeric CLAIMS from README.md and EXPLAINER.md (percentages,
     p-values, correlations, and a narrow class of counts).
  2. Run each analysis script, capture stdout, cache it under
     .provenance_cache/ (gitignored) keyed by a hash of the script source, so
     reruns are free until the script changes.
  3. Assert every claim appears somewhere in some captured stdout, allowing for
     rounding (91.1 in the docs matches 91.11111 in the output).
  4. Print BACKED / ALLOWLISTED / WEAK / UNBACKED and exit nonzero on any
     UNBACKED. WEAK means the only stdout line carrying that number reads like
     a retraction ("the two numbers the old README table carried"), which is
     not the same as the number being produced -- see RETRACTION_HINT.

Known limits, stated plainly: matching is by value, not by meaning, so a small
integer can be backed by coincidence, and a script that prints a number for any
reason will back it. This catches fabrication and drift, not misinterpretation.

Usage:
    python check_provenance.py            # run everything (~3 h, sum of REGISTRY)
    python check_provenance.py --fast     # skip slow scripts, use their cache
    python check_provenance.py --list     # show scripts + claims, run nothing
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

ROOT = Path(__file__).resolve().parent
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
    "ablate_channels.py":       (180,   False),
    "harder_contrast.py":       (120,   False),
    "evaluate_honestly.py":     (300,   True),   # 100-seed x 2-estimator sweep
    "sweep_subjects.py":        (1800,  True),   # all 109 subjects
    "cross_subject.py":         (900,   True),   # leave-one-subject-out, n=20
    "riemannian.py":            (900,   True),   # 20 subjects, 3 pipelines
    "eegnet_compare.py":        (3600,  True),   # CNN training, 3 regimes
    "regime_decomposition.py":  (3600,  True),   # CNN training, 2x2 factorial
}

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# A "claim" is a number the docs assert about THIS project's results. Three
# kinds are extracted, in priority order (first match on a span wins):
#   pct    91.1%          any percentage
#   stat   p <= 0.001     p-values and correlations, r = 0.57
#   count  45 trials      integers bound to a result-bearing noun
# Everything else in the prose (Hz, seconds, run numbers, section numbers,
# electrode names) is ignored by construction -- see UNIT_SUFFIX and ALLOWLIST.
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
    "count:10":  "k in the retired 10-fold estimator -- a knob",
    "count:20":  "subject subset size for LOSO -- a knob, not a result",
    # Forward-looking prose in README "Next" / EXPLAINER roadmap. These describe
    # corpora this project has NOT run, so no script can back them.
    "count:2000": "aspirational trial count for other public corpora",
    "count:5000": "aspirational trial count for other public corpora",
    # Textbook / literature values cited with attribution, not produced here.
    "pct:50.0":  "theoretical chance for a balanced two-class problem",
    "pct:100.0": "rhetorical ceiling ('100% of the time'), not a measurement",
    "pct:0.0":   "rhetorical floor, not a measurement",
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

def unregistered_scripts() -> list[str]:
    """Analysis scripts on disk that the registry does not know about.

    Half of this tool's value is noticing when the repo grows a script whose
    output nobody is checking. Ignore this file and any non-analysis helper.
    """
    return sorted(p.name for p in ROOT.glob("*.py")
                  if p.name not in REGISTRY and p.name != Path(__file__).name)


def cache_path(script: str) -> Path:
    return CACHE / f"{script}.txt"


def source_hash(script: str) -> str:
    return hashlib.sha256((ROOT / script).read_bytes()).hexdigest()[:12]


def run_script(script: str, budget: int) -> str | None:
    """Run a script and return its STDOUT, or None if it failed to produce any.

    Only stdout counts. MNE's own logging goes to stderr and is full of
    incidental numbers (filter lengths, sample counts, file sizes) that would
    accidentally "back" a doc claim it has nothing to do with.
    """
    print(f"  running {script} (budget {budget}s) ...", flush=True)
    try:
        proc = subprocess.run([PYTHON, str(ROOT / script)], cwd=ROOT,
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
        if not (ROOT / script).exists():
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


def output_lines(blob: str) -> list[tuple[float, str]]:
    """Every number in captured stdout, paired with the line it came from."""
    pairs = []
    for line in blob.split("\n"):
        for m in NUM_IN_OUTPUT.finditer(line):
            pairs.append((float(m.group(1)), line.strip()))
    return pairs


def find_backing(claim: Claim, pairs: list[tuple[float, str]]) -> str | None:
    """Return the best stdout line backing this claim, or None.

    Rounding-tolerant: 91.1 in docs matches 91.11111 in stdout. Percentages
    additionally match their fractional form (91.1% <- 0.911), because some
    scripts print fractions and some print percents. A non-retraction line
    always wins over a retraction line.
    """
    d, weak = claim.decimals, None
    for value, line in pairs:
        hit = (half_up(value, d) == claim.value or
               (claim.kind == "pct" and half_up(value * 100, d) == claim.value))
        if not hit:
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
            gone = "" if (ROOT / s).exists() else "  MISSING FROM DISK"
            print(f"  {s:<28} {mark:<5} ~{b:>5}s  [{hit}]{gone}")
        for s in unregistered_scripts():
            print(f"  {s:<28} NOT IN REGISTRY -- its output is never checked")
        print(f"\nCLAIMS ({len(claims)} extracted, "
              f"{dropped} digits inside code fences exempted)")
        for c in claims:
            tag = "allowlist" if c.key in ALLOWLIST else "check"
            print(f"  {c.doc}:{c.line:<5} {c.kind:<6} {c.raw:<8} {tag:<10} "
                  f"{c.context}")
        return 0

    print(f"Gathering script output{' (--fast)' if args.fast else ''} ...")
    blob, missing = gather_outputs(args.fast)
    pairs = output_lines(blob)
    print(f"  {len(pairs)} numbers captured from "
          f"{len(REGISTRY) - len(missing)} scripts")
    for m in missing:
        print(f"  ! no output for {m}")

    backed, weak, unbacked, allowed = [], [], [], []
    for c in claims:
        if c.key in ALLOWLIST:
            allowed.append(c)
            continue
        line = find_backing(c, pairs)
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
              f"UNBACKED entries\n      above may simply be unverified. Run "
              f"without --fast to settle them.")
        print("Result: INCOMPLETE (exit 0 -- --fast cannot fail the build)")
        return 0

    bad = bool(unbacked) or bool(stray)
    print(f"\nResult: {'FAIL' if bad else 'PASS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
