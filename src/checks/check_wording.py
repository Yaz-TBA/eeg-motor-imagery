#!/usr/bin/env python3
"""Fail when outcome-describing prose reuses wording a pre-registration banned.

Checks wording, not truth: it catches the banned phrasings and cannot tell
whether the surrounding sentence is correct.

Why check_provenance.py cannot do this:

That guard matches by value, so it is blind to a sentence with no number in it,
which is where this project's worst defects have lived: the estimator-vs-seed
misattribution, "each compared on identical folds", and the one guarded here,
describing the complement-ablation outcome as the decoder "not breaking".

The ban, verbatim from prereg/prereg-complement-ablation.md:650-655:

    Do not write "break", "does not break", "sensorimotor cortex" or "necessary"
    in any sentence describing this run's outcome, and do not place any statement
    of the falsifiable form's fate under a heading containing the word
    "established".

The permitted floor statement, scoped and in sensor space: the 47 electrodes
remaining after the 17-channel strip is deleted score 35/45 = 77.8% at seed 42,
above the 53.3% majority floor, with a 1000-shuffle permutation p <= 0.001.

What it checks, and deliberately does not:

The bare word "break" is not bannable, since the documents must be free to quote
the falsifiable prediction in order to report it did not hold. So the guard
matches the specific outcome phrasings the prereg's own supporting text at
:633-648 names. Whitespace is normalized first: three of the eleven violations
found on 2026-07-29 wrapped across a newline, invisible to line-oriented grep,
which is very likely why the prereg's own grep claim was written as passing.

Absence from this file's failure list means not checked, never checked-and-passed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # the repo root: this file lives in src/checks/
DOCS = ("README.md", "EXPLAINER.md")

# Each entry: (compiled pattern, what to write instead). Patterns run against
# whitespace-normalized text, so `\s+` is unnecessary -- single spaces suffice.
BANNED = [
    (r"does not break the decoder",
     'the prediction did not hold / the complement scores above the majority floor'),
    (r"did not produce a break",
     '"the prediction did not hold"'),
    (r"failing to break the decoder",
     '"the prediction failing to hold"'),
    (r"deletion did not break",
     '"the deletion left the decoder above the majority floor"'),
    (r"[Nn]othing broke",
     'state the measured floor instead, or say nothing'),
    (r"[Ii]t did ?n[o']t break",
     '"the prediction did not hold"'),
    (r"the decoder does not break",
     '"the complement scores well above the majority floor"'),
]

HEADING_ESTABLISHED = re.compile(r"^#+ .*established", re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def main() -> int:
    failures: list[str] = []

    for name in DOCS:
        path = ROOT / name
        if not path.exists():
            failures.append(f"{name}: MISSING -- cannot check")
            continue
        raw = path.read_text(encoding="utf-8")
        flat = normalize(raw)

        for pattern, remedy in BANNED:
            for m in re.finditer(pattern, flat):
                start = max(0, m.start() - 70)
                failures.append(
                    f"{name}: banned outcome wording {m.group(0)!r}\n"
                    f"    ...{flat[start:m.end() + 70]}...\n"
                    f"    write instead: {remedy}"
                )

        # The prereg also forbids putting the falsifiable form's fate under a
        # heading containing "established".
        for i, line in enumerate(raw.splitlines(), 1):
            if HEADING_ESTABLISHED.match(line):
                failures.append(
                    f"{name}:{i}: heading contains 'established' -- the prereg "
                    f"forbids filing the falsifiable form's fate under one\n"
                    f"    {line.strip()}"
                )

    print("=" * 74)
    print("WORDING GUARD -- prereg-complement-ablation.md:650-655")
    print("=" * 74)
    print(f"Checked {len(DOCS)} documents against {len(BANNED)} banned outcome "
          f"phrasings, whitespace-normalized.")
    print()

    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print()
        print(f"Result: FAIL ({len(failures)} violation(s))")
        return 1

    print("No banned outcome wording found.")
    print()
    print("This checks WORDING only. It cannot tell whether the surrounding")
    print("claim is true, and it does not look outside README.md/EXPLAINER.md.")
    print("Result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
