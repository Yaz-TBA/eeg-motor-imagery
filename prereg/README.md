# Pre-registrations

The four documents here are the registered designs behind the measurements this repo reports.
They don't all claim the same thing about their own priority, so here it is per document rather
than as the one blanket sentence this paragraph used to carry:

- `prereg-emg-proxy.md` and `prereg-block-permutation.md` each state in their opening lines that
  nothing executable existed when they were written.
- `prereg-permutation-validity.md` states it was committed **before the run it governs**. That is
  priority over the run, not over the script.
- `prereg-complement-ablation.md` states it was written before the committed script existed, and
  git disagrees: `ablate_channels.py` was committed 2026-07-24 21:22, the day before that
  document's own stated write date of 2026-07-25. Read its design as registered and its priority
  claim as wrong.

They are here because `README.md` and `EXPLAINER.md` cite "the pre-registration" as the authority for
what may and may not be concluded, and until 2026-08-04 a reader had no way to read it. An authority
nobody can check is not an authority.

| File | Governs | Written |
|---|---|---|
| `prereg-complement-ablation.md` | `../ablate_channels.py` | 2026-07-25 |
| `prereg-emg-proxy.md` | `../emg_proxy.py` | 2026-07-25 |
| `prereg-block-permutation.md` | `../permutation_design.py` | 2026-07-25 |
| `prereg-permutation-validity.md` | `../validity_gate.py` | 2026-07-30 |

`runs/permutation-design-2026-07-25.stdout` is the captured run output that `../validity_gate.py:29`
cites for its subject marginals.

## These files are byte-identical to the originals. Nothing was edited.

Every file in this directory is an **exact copy** of its source in the private study corpus where it
was authored. No path was rewritten, no reference retargeted, no word changed.

That is deliberate, and it is a correction. **An earlier version of this vendoring, on 2026-08-04,
rewrote paths inside these documents and inserted two short clauses of explanatory prose.** Every one
of these files carries a freeze clause — `prereg-complement-ablation.md:5`, `prereg-block-permutation.md:552`,
`prereg-emg-proxy.md:6-7`, `prereg-permutation-validity.md:3-5` — and one of them says amendments are
*"never made in place."* Editing them, however cosmetically, and then asserting in this file that only
paths had changed, was the exact failure mode the documents exist to prevent. The edits are reverted
and this paragraph is the record of them.

## What that costs, stated plainly

**Some paths inside these documents do not resolve from this repo.** They were written inside another
directory tree and they point at it:

- References to `neuro-canon/measurements/...` mean the sibling documents now beside this file.
- References to `neuro-canon/runs/...` mean `runs/` here, for the one run log that was vendored.
- `prereg-block-permutation.md` and `prereg-complement-ablation.md` cite
  `neuro-canon/runs/hostile-pass-2026-07-25/hostile_verify_A.py` and its stdout. **That harness is
  not vendored.** It is pilot material the registered designs cite as the *source of the disclosure
  they respond to*, never as evidence for a reported number. Every figure this repo publishes traces
  to a script in the parent directory and to `.provenance_cache/`.
- Three absolute paths of the form `/Users/<user>/...` survive, in
  `prereg-emg-proxy.md`, `prereg-block-permutation.md` and `prereg-complement-ablation.md`.
- `prereg-permutation-validity.md:11` sends a reader to `OVERRIDE-RULING-2026-07-30.md` for the
  twelve-agent adjudication of the fired gate. **That document is not in this repo and it is not
  on my disk either.** It is the one citation in this directory with nothing behind it, and it
  was missing from this list until 2026-08-29. The two claims it is cited for, that the gate was
  unsatisfiable by any valid design and that half the override's reasoning was false, should be
  read as assertions rather than as findings anyone can check.

A broken link in a frozen document is a smaller problem than an edited frozen document. That is the
trade, and it is why it went this way.

**One sentence reads strangely and is correct.** `prereg-permutation-validity.md:1243` says of another
document: *"It does not exist."* That was true on 2026-07-30 when it was written, about a
forward-looking reference. It is a dated statement inside a frozen file, not a live claim.

## How to read one

Read **Section 0 and the registered-outcomes section first, before the appended RESULTS.** That
ordering is the entire point. A rule read after you already know the answer is not a rule.
