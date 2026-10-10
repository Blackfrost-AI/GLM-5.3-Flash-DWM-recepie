# GLM-5.3-Flash DWM alpha 2.8

This is a public, Apache-2.0 agent package from **Blackfrost AI** for applying
the GLM-5.3-Flash BF16 Domain Weight Modification at **alpha 2.8**. It preserves
the exact prompt pair, direction definition, source revision, layer range,
target surface, one-pass operator, arithmetic, and MTP handling recorded by the
original Flash method.

The repository contains the method and its evidence. It does not redistribute
GLM weights or claim that an alpha-2.8 checkpoint has already passed release
qualification. See [`MODEL-NOTICE.md`](MODEL-NOTICE.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Current state

| Item | State |
|---|---|
| Text package | Complete and independently verifiable |
| Alpha | Locked to `2.8`; no command line override |
| Direction provenance | Historical BF16 capture is fully recorded |
| Exact direction tensor | Identified by size and SHA-256; missing from the retained archive |
| Alpha-2.8 checkpoint | Not built by this package release |
| Alpha-2.8 behavior | Not yet evaluated |
| Historical evidence | Alpha `3.0` lineage only |

The required direction file is exactly `151,148,124` bytes with SHA-256
`9de7f65c147bb25f15ea4dea411d67048f6eb13c79a2513153fbde96dfcf343e`.
Recover it or reproduce it with the pinned capture environment. The executor
rejects every different digest. This prevents an agent from silently replacing
the BF16 direction with the earlier FP8 direction.

The original metadata is preserved as a path-sanitized semantic reference at
[`evidence/directions.metadata.reference.json`](evidence/directions.metadata.reference.json),
and its original byte hash remains recorded in the receipts.
A fresh capture naturally changes volatile fields such as elapsed time, so its
sidecar is checked against the locked model, geometry, prompts, capture method,
and artifact shape instead of being required to have the historical byte hash.

## Method lock

| Item | Locked value |
|---|---|
| Clean source | `zai-org/GLM-5.3-Flash-BF16@b1967181a3917ae70a437f4884748f6b8e3a1f4d` |
| Model geometry | 45 main layers, hidden width 4096, four mHC streams |
| Prompt material | 32 harmful and 32 harmless prompts paired by array index |
| Prompt formatting | Chat template, generation prompt, left padding, `reasoning_effort=max` |
| Capture position | Last token |
| Primary direction | Unweighted mean of four mHC layer-end streams |
| Alpha | `2.8` |
| Rank and passes | Rank one, one pass |
| Main layers edited | `2` through `44` inclusive |
| Exclusions | Main layers `0–1`; MTP layer `45` |
| Edited tensors | 43 attention output, 1 dense down, 12,096 routed-expert down, 42 shared-expert down; 12,182 total |
| Arithmetic | FP32 projection and subtraction, then BF16 rounding |
| Norm restoration | None |
| MTP handling | 889 tensors transplanted exactly from clean BF16 |

[`recipe.lock.json`](recipe.lock.json) is the machine-readable authority.

## The exact 32 pairs

[`assets/pairs.json`](assets/pairs.json) contains two ordered arrays. Harmful
row `i` is paired with harmless row `i`, for 32 pairs. Capture applies the
model's chat template separately to every user prompt, adds the generation
prompt, left pads the batch, and records the last-token activations.

At each main layer, the capture keeps three model-specific views:

- `layer_end_mean`: the unweighted mean across the four mHC streams;
- `attn_input`: the mHC-collapsed input to attention; and
- `ffn_input`: the mHC-collapsed input to the FFN/MoE.

The edit uses `layer_end_mean`. For layer `l`:

```text
harmful_mean[l] = mean of the 32 harmful last-token activations
harmless_mean[l] = mean of the 32 harmless last-token activations
direction[l] = unit(harmful_mean[l] - harmless_mean[l])
```

The artifact also retains the alternate sites, raw activations, differences,
effect sizes, and split-half statistics. It has 32 tensors in total. Capture
does not modify model weights.

## The alpha-2.8 edit

For unit direction `r` and selected matrix `W`, the executor performs:

```text
W' = BF16(W.float() - 2.8 * r[:, None] * (r @ W.float())[None, :])
```

The selected output-space writes are:

- `self_attn.o_proj.weight` in layers 2–44;
- dense `mlp.down_proj.weight` in layer 2;
- all 288 routed expert `down_proj.weight` tensors in layers 3–44; and
- shared expert `down_proj.weight` in layers 3–44.

Layer 45 belongs to the separate NextN/MTP draft block. It receives no
direction and no DWM edit. After the main shards are written, all 889 layer-45
tensors are copied from the clean BF16 source and compared tensor by tensor.

The projection remaining after the edit is `(1 - alpha)` times the original.
At alpha 2.8, its norm ratio should be about `1.8` and its direction should be
reversed. The executor checks representative tensors for both properties after
BF16 rounding.

## Why alpha 2.8 is separate from the historical result

The retained historical Flash run used alpha 3.0. It established the capture,
geometry, target counts, MTP procedure, serialization path, and historical
behavior of that alpha. The operator selected alpha 2.8 for this package.

Changing alpha changes every edited tensor. Historical alpha-3.0 refusal
counts therefore remain useful lineage, but they cannot validate alpha 2.8.
[`evidence/historical-alpha3-lineage.json`](evidence/historical-alpha3-lineage.json)
states this boundary in machine-readable form.

## Repository map

| Path | Purpose |
|---|---|
| `RUNBOOK.md` | Ordered download, recovery/capture, inspect, build, and verify procedure |
| `recipe.lock.json` | Immutable source, direction, operation, geometry, and claim contract |
| `environment.lock.json` | Recorded image and Python package versions |
| `assets/pairs.json` | Exact 32 harmful and 32 harmless prompts |
| `assets/directions.asset.json` | Required direction size, hashes, status, and recovery rules |
| `evidence/` | Canonical direction metadata, capture receipt, and separately labeled alpha-3.0 lineage |
| `scripts/capture_directions.py` | Locked BF16 direction capture implementation |
| `scripts/run_locked.py` | Locked alpha-2.8 BF16 editor with no alpha override |
| `scripts/verify_output.py` | Independent source/output/MTP and projection verifier |
| `scripts/verify_package.py` | Package, lock, plan, prompt, and optional large-input verifier |
| `plans/ablagent-read-only-audit.json` | Ablagent Standalone Harness 1.0.0 audit plan |

## Quick integrity audit

```bash
python3 scripts/verify_package.py
```

After the direction and clean source are available:

```bash
python3 scripts/verify_package.py \
  --directions artifacts/directions.safetensors \
  --source /data/GLM-5.3-Flash-BF16-b1967181
```

The complete commands and expected results are in [`RUNBOOK.md`](RUNBOOK.md).

## Ablagent Standalone Harness 1.0.0

The included plan lets an Ablagent-compatible model audit the package as a
bounded, read-only workspace:

```bash
cp config.ablagent.example.json config.local.json
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" --probe
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" \
  --workspace "$PWD" \
  --plan plans/ablagent-read-only-audit.json
```

The optional paired model is `Alblagent-9B`, configured through the ordinary
OpenAI-compatible endpoint fields. The plan uses only `read_file`, `grep`, and
`glob`. A shell-capable operator performs the model download, capture, and
weight edit described in the runbook.

## Qualification gates

An alpha-2.8 output becomes a candidate only after the offline checks pass. A
release decision additionally requires:

1. successful model load and deterministic smoke generation;
2. capability comparison against the pinned clean source;
3. the current 443-row refusal measurement corpus;
4. the separate 9-row release-only red-line holdout; and
5. retained run receipts and an output hash manifest.

Do not copy historical alpha-3.0 scores into an alpha-2.8 result report.

## License

Copyright 2026 Blackfrost AI. Blackfrost-authored code, documentation, prompt
material, manifests, and plans in this repository are licensed under
Apache-2.0. Upstream model weights and third party software retain their own
licenses.
