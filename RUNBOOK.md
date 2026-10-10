# Agent runbook: GLM-5.3-Flash DWM alpha 2.8

This is the ordered procedure for the alpha-2.8 method. Read `AGENTS.md`,
`recipe.lock.json`, `environment.lock.json`, the direction descriptor, and both
evidence receipts completely before running commands. Stop on a mismatch.

## 1. Execution and claims boundary

This runbook captures or verifies a direction, performs an offline BF16 weight
transformation, and verifies the result. It does not by itself qualify or
publish a model. Alpha 2.8 has no inherited behavior score from alpha 3.0.

The edit is method-exact only when the pinned source and exact recorded
direction tensor pass verification. Byte-identical output is not promised
because output serialization can vary with runtime and hardware.

## 2. Required resources

- Linux with Docker and the NVIDIA Container Toolkit.
- Eight visible CUDA GPUs. The recorded capture allowed 160 GiB per GPU.
- A large CPU-memory allowance; the recorded capture allowed 3,000 GiB.
- The ability to build the pinned Docker image.
- Roughly 650 GB for the clean source and 650 GB for the BF16 output, plus
  temporary and receipt space. Use separate volumes or at least 1.4 TB free.
- The exact direction artifact and its matching metadata, or resources to
  recapture them.

The recorded image is `local/glm53-direction-capture:20260826` with image ID
`sha256:4078efee2f9bad820f1a5bf4822e1dc3cc2ab923dde5a8319b0d0f313f104f6d`.
If that local image is unavailable, rebuild from `runtime/Dockerfile` and
record the new image ID. A rebuild is expected to be method-equivalent and is
not automatically byte-identical.

## 3. Set isolated paths

Run from the repository root. Choose new output and run directories.

```bash
export PACKAGE_ROOT="$PWD"
export SOURCE_ROOT="/data/GLM-5.3-Flash-BF16-b1967181"
export ARTIFACT_ROOT="$PACKAGE_ROOT/artifacts"
export DIRECTION_FILE="$ARTIFACT_ROOT/directions.safetensors"
export OUTPUT_ROOT="/data/GLM-5.3-Flash-DWM-a2p8-BF16"
export RUN_ROOT="/data/GLM-5.3-Flash-DWM-a2p8-run"
mkdir -p "$ARTIFACT_ROOT" "$RUN_ROOT"
```

Do not reuse an alpha-3.0 output directory or a partial directory from another
method.

## 4. Verify the repository

```bash
python3 scripts/verify_package.py
```

Expected: `passed: true`, lock ID `glm-5.3-flash-bf16-a2p8`, alpha `2.8`, 32
pairs, eight read-only Ablagent stages, and complete `SHA256SUMS` coverage.

## 5. Obtain the pinned clean source

Fetch the exact commit instead of a moving branch:

```bash
hf download zai-org/GLM-5.3-Flash-BF16 \
  --revision b1967181a3917ae70a437f4884748f6b8e3a1f4d \
  --local-dir "$SOURCE_ROOT"
```

Verify the source metadata and structure:

```bash
python3 scripts/verify_package.py --source "$SOURCE_ROOT"
```

Expected source facts:

| Check | Required value |
|---|---:|
| `config.json` SHA-256 | `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943` |
| index SHA-256 | `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f` |
| Index keys | 38,770 |
| Shards | 120 |
| Indexed tensor bytes | 642,646,653,816 |
| Main layers | 45 |
| Hidden size | 4,096 |
| Routed experts | 288 |
| First dense layers | 3 |
| NextN/MTP layers | 1 |

## 6. Obtain the exact direction

The original artifact is not in this Git repository or the retained archive.
Preferred action: recover these two files from the recorded run directory:

```text
20260826T1742Z-bf16-direction-capture/directions.safetensors
20260826T1742Z-bf16-direction-capture/directions.metadata.json
```

Place both in `ARTIFACT_ROOT`, then verify:

```bash
python3 scripts/verify_package.py \
  --directions "$DIRECTION_FILE" \
  --source "$SOURCE_ROOT"
```

Required direction facts:

- file size `151,148,124` bytes;
- SHA-256 `9de7f65c147bb25f15ea4dea411d67048f6eb13c79a2513153fbde96dfcf343e`;
- metadata SHA-256 `84217794ae4b0c55161f3ac087cb5fdd53f89b311b780d8666d1c11ddf02b4b5`;
- 32 tensors; and
- `dirs` shape `[45, 4096]` and exact alias content with
  `directions.layer_end_mean`.

### Recapture path

Use this only when recovery fails. Build the environment while network is
available:

```bash
docker build \
  --tag local/glm53-direction-capture:20260826 \
  runtime
docker image inspect local/glm53-direction-capture:20260826 \
  --format '{{.Id}}' > "$RUN_ROOT/capture-image-id.txt"
```

Then capture offline from the read-only clean source:

```bash
docker run --rm \
  --name glm53-flash-bf16-direction-capture \
  --gpus all \
  --network none \
  --ipc host \
  --shm-size 128g \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e USE_HUB_KERNELS=NO \
  -e OUTPUT_UID="$(id -u)" \
  -e OUTPUT_GID="$(id -g)" \
  -v "$SOURCE_ROOT:/model:ro" \
  -v "$PACKAGE_ROOT/scripts:/work:ro" \
  -v "$ARTIFACT_ROOT:/run" \
  local/glm53-direction-capture:20260826 \
  python3 /work/capture_directions.py \
    --model /model \
    --out /run/directions.safetensors \
    --log /run/capture.log
```

The capture script has no revision, repository, batch-size, padding, reasoning,
or capture-site override. Run the package verifier immediately afterward. If
either recorded digest differs, stop and preserve the recapture as diagnostic
evidence. Do not edit the lock or continue the alpha-2.8 run under this method
name.

## 7. Inspect without writing weights

This dry run verifies source hashes and geometry, the exact direction and
metadata, unit norms, target counts, shard presence, and MTP inventory:

```bash
docker run --rm \
  --name glm53-flash-a2p8-inspect \
  --gpus all \
  --network none \
  --ipc host \
  --shm-size 128g \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$ARTIFACT_ROOT:/directions:ro" \
  -v "$PACKAGE_ROOT/scripts:/work:ro" \
  local/glm53-direction-capture:20260826 \
  python3 /work/run_locked.py \
    --source /source \
    --output /output-not-created \
    --directions /directions/directions.safetensors \
    --source-display-path "$SOURCE_ROOT" \
    --output-display-path "$OUTPUT_ROOT" \
    --directions-display-path "$DIRECTION_FILE" \
    --dry-run | tee "$RUN_ROOT/inspect.log"
```

Expected target counts are 43 attention output, 1 dense down, 12,096 routed
expert down, and 42 shared expert down tensors: 12,182 total.

## 8. Build alpha 2.8 once

Confirm that `OUTPUT_ROOT` is absent or empty. The output volume needs space
for 120 BF16 shards. Then run:

```bash
mkdir -p "$OUTPUT_ROOT"
docker run --rm \
  --name glm53-flash-bf16-dwm-a2p8 \
  --gpus all \
  --network none \
  --ipc host \
  --shm-size 128g \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$OUTPUT_ROOT:/output" \
  -v "$ARTIFACT_ROOT:/directions:ro" \
  -v "$PACKAGE_ROOT/scripts:/work:ro" \
  local/glm53-direction-capture:20260826 \
  python3 /work/run_locked.py \
    --source /source \
    --output /output \
    --directions /directions/directions.safetensors \
    --output-uid "$(id -u)" \
    --output-gid "$(id -g)" \
    --source-display-path "$SOURCE_ROOT" \
    --output-display-path "$OUTPUT_ROOT" \
    --directions-display-path "$DIRECTION_FILE" \
  2>&1 | tee "$RUN_ROOT/apply.log"
```

There is deliberately no `--alpha`, `--revision`, `--skip-early`, `--passes`,
or `--workers` option. The executable constants are alpha 2.8, one pass,
layers 2–44, MTP layer 45 excluded, and eight workers.

## 9. Verify the output independently

```bash
docker run --rm \
  --name glm53-flash-a2p8-verify \
  --gpus all \
  --network none \
  -v "$SOURCE_ROOT:/source:ro" \
  -v "$OUTPUT_ROOT:/output:ro" \
  -v "$ARTIFACT_ROOT:/directions:ro" \
  -v "$PACKAGE_ROOT/scripts:/work:ro" \
  local/glm53-direction-capture:20260826 \
  python3 /work/verify_output.py \
    --source /source \
    --output /output \
    --directions /directions/directions.safetensors \
  | tee "$RUN_ROOT/output-verification.json"
```

Required results:

| Check | Required value |
|---|---:|
| Output shards | 120 |
| Index keys | 38,770 |
| Edited tensors recorded | 12,182 |
| Alpha | 2.8 |
| Passes | 1 |
| Projection norm ratio | approximately 1.8 |
| Projection cosine | less than -0.999 |
| MTP tensors byte-equal | 889 |
| MTP bytes recorded | 14,865,185,408 |

## 10. Hash and retain the candidate

```bash
(cd "$OUTPUT_ROOT" && \
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256) \
  > "$RUN_ROOT/output.SHA256SUMS"
shasum -a 256 "$RUN_ROOT/output.SHA256SUMS" \
  > "$RUN_ROOT/output.SHA256SUMS.sha256"
```

Retain the image ID, package commit, source revision, direction hashes, inspect
log, apply log, output verification, output manifests, and generated
`dwm_provenance.json` together.

## 11. Later qualification

Serve and test the candidate under a new authorized run. Record alpha 2.8 in
every result path. Run model load, smoke generation, capability comparison,
the versioned 443-row measurement corpus, and the separate 9-row release-only
red-line holdout. A failed red-line gate is a hard stop.

Historical alpha-3.0 evaluation is context only. Never merge its scores into
the alpha-2.8 report.

## 12. Ablagent audit

The read-only plan can be run before any large input is acquired:

```bash
cp config.ablagent.example.json config.local.json
/Users/craigmac/Desktop/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" --probe
/Users/craigmac/Desktop/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" \
  --workspace "$PWD" \
  --plan plans/ablagent-read-only-audit.json
```

The plan verifies what an agent is expected to know and report. It cannot run
the shell commands above because Harness 1.0.0 exposes read-only tools.
