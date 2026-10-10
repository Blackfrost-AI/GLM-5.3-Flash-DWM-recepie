# Agent contract: GLM-5.3-Flash DWM alpha 2.8

Read these files completely and in this order before proposing a command:

1. `recipe.lock.json`
2. `RUNBOOK.md`
3. `assets/directions.asset.json`
4. `evidence/capture-receipt.json`
5. `evidence/historical-alpha3-lineage.json`

The method parameters are immutable. Alpha is `2.8`, rank is one, the
operation is one pass, main layers 2 through 44 are edited, and MTP layer 45 is
excluded and restored exactly from the clean BF16 source. Never substitute a
source revision, direction tensor, prompt bank, capture site, target surface,
layer range, arithmetic rule, or pass count and call the result this method.

Run `python3 scripts/verify_package.py` before execution. The original BF16
direction tensor is identified by exact size and SHA-256 but is absent from
this archive. Recover it or reproduce it with the pinned capture path, then
run the verifier with `--directions`. Stop on any mismatch.

The bundled Ablagent plan is a read-only audit. Its tools inspect this package;
they do not run Docker, capture activations, edit weights, serve a model, or
approve a release. Those actions require the shell commands and gates in the
runbook.

Alpha 2.8 is an operator-selected variant. Historical alpha-3.0 measurements
must be labeled as lineage and must never be reported as alpha-2.8 evidence.
A new output is unqualified until the current evaluation and red-line gates
pass.
