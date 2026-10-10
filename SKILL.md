---
name: glm53-flash-dwm-a2p8
description: Audit or execute Blackfrost AI's GLM-5.3-Flash BF16 DWM alpha-2.8 method with exact direction, source, geometry, and MTP verification.
---

# GLM-5.3-Flash DWM alpha 2.8

1. Read `AGENTS.md`, `recipe.lock.json`, and `RUNBOOK.md` completely.
2. Treat the method fields in `recipe.lock.json` as immutable.
3. Run `python3 scripts/verify_package.py` before acquiring large inputs.
4. Recover or recapture the exact direction artifact described in
   `assets/directions.asset.json` and verify it with `--directions`.
5. Verify the pinned clean source with `--source`.
6. Run inspect, build, and output verification in the runbook order.
7. Record alpha-2.8 results separately from the historical alpha-3.0 lineage.
8. Require fresh serving, capability, refusal, and red-line evidence before
   promoting an output.

Use `plans/ablagent-read-only-audit.json` with Ablagent Standalone Harness
1.0.0. The harness audits the method and does not execute weight edits.
