#!/usr/bin/env python3
"""Fail when outcome-describing prose reuses wording a pre-registration banned.

NAVIGATION. Checks WORDING, not truth. It catches banned phrasings (unscoped accuracy
claims, 'chance is 50%', the withdrawn scalp-map defence) and can't tell you whether the
surrounding sentence is correct.

WHY THIS EXISTS, AND WHY check_provenance.py CANNOT DO IT

check_provenance.py matches by VALUE. It asks whether some committed script
prints the number a document claims. It is blind to a sentence that contains no
number, which is exactly where this project's worst defects have lived: the
estimator-versus-seed misattribution, "each compared on identical folds", and
the one this file guards -- describing the complement-ablation outcome as the
decoder "not breaking".

THE BAN, verbatim from prereg/prereg-complement-ablation.md:650-655 in the
study corpus:

    Do not write "break", "does not break", "sensorimotor cortex" or "necessary"
    in any sentence describing this run's outcome, and do not place any statement
    of the falsifiable form's fate under a heading containing the word
    "established".

The permitted floor statement, scoped and in sensor space, is: the 47 electrodes
remaining after the 17-channel strip is deleted score 35/45 = 77.8% at seed 42,
above the 53.3% majority floor, with a 1000-shuffle permutation p <= 0.001.

WHAT THIS CHECKS, AND WHAT IT DELIBERATELY DOES NOT

The bare word "break" is not bannable: these documents must be free to QUOTE the
falsifiable prediction ("if it reads sensorimotor cortex, deleting sensorimotor
cortex must break it") in order to report that it did not hold. Banning the word
outright would forbid stating the hypothesis at all. So this guard matches the
specific OUTCOME phrasings, which is what the prereg's own supporting text at
:633-648 names.

Whitespace is normalized before matching. Three of the eleven violations found
on 2026-07-29 wrapped across a newline and were invisible to line-oriented grep,
which is very likely why the prereg's own grep claim was written as passing.

Absence from this file's failure list means NOT CHECKED, not CHECKED AND PASSED.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the repo root: this file lives in checks/
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
