#!/usr/bin/env python3
"""Fail-closed verification for the GLM-5.3-Flash alpha-2.8 package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_REPO = "zai-org/GLM-5.3-Flash-BF16"
SOURCE_REVISION = "b1967181a3917ae70a437f4884748f6b8e3a1f4d"
SOURCE_CONFIG_SHA256 = "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943"
SOURCE_INDEX_SHA256 = "e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f"
PROMPTS_SHA256 = "08f90187547463dbf0fbe76c4003e9ab40add53fa111e4a7553f7da749bfdc10"
DIRECTION_BYTES = 151_148_124
DIRECTION_SHA256 = "9de7f65c147bb25f15ea4dea411d67048f6eb13c79a2513153fbde96dfcf343e"
DIRECTION_METADATA_REFERENCE_SHA256 = "d77801ddc2eebad5b698439f68c4d7b95f539f09755b11edc25b93d1bb5d5cf4"
DIRECTION_METADATA_RECORDED_SHA256 = "84217794ae4b0c55161f3ac087cb5fdd53f89b311b780d8666d1c11ddf02b4b5"
RUNTIME_EXCLUSIONS = {"SHA256SUMS", "config.local.json", ".DS_Store"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value_at(document, dotted: str):
    value = document
    for part in dotted.split("."):
        value = value[part]
    return value


def verify_lock() -> dict:
    lock = read_json(ROOT / "recipe.lock.json")
    expected = {
        "schema": "blackfrost.dwm.locked-method.v1",
        "recipe_id": "glm-5.3-flash-bf16-a2p8",
        "version": "1.0.0",
        "source.repo": SOURCE_REPO,
        "source.revision": SOURCE_REVISION,
        "source.config_sha256": SOURCE_CONFIG_SHA256,
        "source.index_sha256": SOURCE_INDEX_SHA256,
        "source.indexed_tensors": 38770,
        "source.weight_shards": 120,
        "source.indexed_tensor_bytes": 642646653816,
        "directions.bytes": DIRECTION_BYTES,
        "directions.sha256": DIRECTION_SHA256,
        "directions.metadata_reference": "evidence/directions.metadata.reference.json",
        "directions.metadata_reference_sha256": DIRECTION_METADATA_REFERENCE_SHA256,
        "directions.recorded_original_metadata_sha256": DIRECTION_METADATA_RECORDED_SHA256,
        "directions.prompt_manifest_sha256": PROMPTS_SHA256,
        "directions.harmful_count": 32,
        "directions.harmless_count": 32,
        "directions.tensor_count": 32,
        "directions.primary_shape": [45, 4096],
        "directions.primary_site": "layer_end_mean",
        "operation.alpha": 2.8,
        "operation.passes": 1,
        "operation.rank": 1,
        "operation.skip_early": 2,
        "operation.first_layer": 2,
        "operation.last_layer": 44,
        "operation.excluded_mtp_layer": 45,
        "operation.targets.o_proj": 43,
        "operation.targets.dense_down": 1,
        "operation.targets.expert_down": 12096,
        "operation.targets.shared_down": 42,
        "operation.targets.total": 12182,
        "operation.mtp.expected_tensors": 889,
        "operation.mtp.expected_bytes": 14865185408,
        "execution.gpus": 8,
        "execution.workers": 8,
        "qualification.alpha_2_8_checkpoint_built": False,
        "qualification.alpha_2_8_served": False,
        "qualification.alpha_2_8_capability_tested": False,
        "qualification.alpha_2_8_refusal_tested": False,
        "qualification.alpha_2_8_redline_tested": False,
        "qualification.historical_alpha_3_results_are_alpha_2_8_evidence": False,
        "reproducibility.byte_identical_output_promised": False,
    }
    for key, wanted in expected.items():
        actual = value_at(lock, key)
        require(actual == wanted, f"recipe lock mismatch: {key}: {actual!r} != {wanted!r}")
    require(lock["operation"]["norm_restoration"] is False, "norm restoration must be false")
    require(lock["operation"]["mtp"]["require_all_exact"] is True, "MTP exactness must be required")
    return lock


def verify_prompts() -> dict:
    path = ROOT / "assets/pairs.json"
    require(sha256(path) == PROMPTS_SHA256, "prompt manifest hash mismatch")
    prompts = read_json(path)
    harmful = prompts["harmful"]
    harmless = prompts["harmless"]
    require(len(harmful) == len(harmless) == 32, "expected 32 prompt pairs")
    require(all(isinstance(item, str) and item for item in harmful + harmless), "empty prompt")
    compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    require(
        hashlib.sha256(compact(harmful)).hexdigest()
        == "178acbc0c022698ba37477a66c3ed75b9379aecbcacd71baffb4f0ee6475c90f",
        "harmful prompt order/content mismatch",
    )
    require(
        hashlib.sha256(compact(harmless)).hexdigest()
        == "8caf820292fd7c30b5a95add58c2bfc11a3504be21d594c8d39cdb57d328ec6a",
        "harmless prompt order/content mismatch",
    )
    return {"pairs": 32, "manifest_sha256": PROMPTS_SHA256}


def verify_descriptor() -> dict:
    descriptor = read_json(ROOT / "assets/directions.asset.json")
    require(descriptor["asset_name"] == "directions.safetensors", "direction filename mismatch")
    require(descriptor["bytes"] == DIRECTION_BYTES, "direction descriptor size mismatch")
    require(descriptor["sha256"] == DIRECTION_SHA256, "direction descriptor hash mismatch")
    require(descriptor["metadata_reference_sha256"] == DIRECTION_METADATA_REFERENCE_SHA256, "metadata reference hash mismatch")
    require(descriptor["recorded_original_metadata_sha256"] == DIRECTION_METADATA_RECORDED_SHA256, "recorded metadata hash mismatch")
    reference = ROOT / descriptor["metadata_reference"]
    require(reference.is_file(), "metadata reference is missing")
    require(sha256(reference) == DIRECTION_METADATA_REFERENCE_SHA256, "metadata reference file hash mismatch")
    require(descriptor["archive_status"] == "missing", "archive status must remain explicit")
    require(descriptor["public_release_asset_published"] is False, "unpublished asset marked published")
    require(descriptor["required_for_execution"] is True, "direction must be required")
    return {"status": descriptor["archive_status"], "required": True}


def assignments(path: pathlib.Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return result


def verify_scripts() -> dict:
    run_path = ROOT / "scripts/run_locked.py"
    capture_path = ROOT / "scripts/capture_directions.py"
    run_text = run_path.read_text(encoding="utf-8")
    capture_text = capture_path.read_text(encoding="utf-8")
    ast.parse(run_text)
    ast.parse(capture_text)
    run = assignments(run_path)
    capture = assignments(capture_path)
    required = {
        "REPO_ID": SOURCE_REPO,
        "REVISION": SOURCE_REVISION,
        "ALPHA": 2.8,
        "PASSES": 1,
        "SKIP_EARLY": 2,
        "WORKERS": 8,
        "DIRECTION_BYTES": DIRECTION_BYTES,
        "DIRECTION_SHA256": DIRECTION_SHA256,
    }
    for key, wanted in required.items():
        require(run.get(key) == wanted, f"executor constant mismatch: {key}")
    for key, wanted in {
        "REPO_ID": SOURCE_REPO,
        "REVISION": SOURCE_REVISION,
        "BATCH_SIZE": 32,
        "SKIP_EARLY": 2,
    }.items():
        require(capture.get(key) == wanted, f"capture constant mismatch: {key}")
    for forbidden in ("--alpha", "--revision", "--repo-id", "--skip-early", "--workers"):
        require(f'add_argument("{forbidden}"' not in run_text, f"executor exposes {forbidden}")
    for forbidden in ("--revision", "--repo-id", "--batch-size", "--skip-early"):
        require(f'add_argument("{forbidden}"' not in capture_text, f"capture exposes {forbidden}")
    require("work.add_(r[:, None] * projection[None, :], alpha=-alpha)" in run_text, "operator changed")
    return {"alpha": run["ALPHA"], "passes": run["PASSES"], "runtime_overrides": False}


def verify_direction(path: pathlib.Path) -> dict:
    require(path.is_file(), f"direction artifact not found: {path}")
    require(path.name == "directions.safetensors", "unexpected direction filename")
    require(path.stat().st_size == DIRECTION_BYTES, "direction byte count mismatch")
    require(sha256(path) == DIRECTION_SHA256, "direction SHA-256 mismatch")
    metadata_path = path.with_suffix(".metadata.json")
    require(metadata_path.is_file(), "direction metadata is missing")
    metadata_sha256 = sha256(metadata_path)
    metadata = read_json(metadata_path)
    require(metadata["model"]["repo"] == SOURCE_REPO, "direction source repository mismatch")
    require(metadata["model"]["revision"] == SOURCE_REVISION, "direction source revision mismatch")
    require(metadata["model"]["checkpoint_precision"] == "bf16", "direction is not BF16-captured")
    require(metadata["model"]["hidden_size"] == 4096, "direction hidden size mismatch")
    require(metadata["model"]["num_hidden_layers"] == 45, "direction layer count mismatch")
    require(metadata["model"]["hc_mult"] == 4, "direction mHC width mismatch")
    require(metadata["model"]["num_nextn_predict_layers"] == 1, "direction MTP geometry mismatch")
    method = metadata["method"]
    require(method["primary_site"] == "layer_end_mean", "direction primary site mismatch")
    require(method["sites"] == ["layer_end_mean", "attn_input", "ffn_input"], "direction sites mismatch")
    require(method["last_token"] is True and method["left_padding"] is True, "direction token-position policy mismatch")
    require(method["reasoning_effort"] == "max" and method["batch_size"] == 32, "direction capture settings mismatch")
    require(method["weights_modified"] is False and method["mtp_layer_captured"] is False, "direction capture boundary mismatch")
    require(metadata["prompts"] == {
        "harmful_count": 32,
        "harmless_count": 32,
        "harmful_sha256": "178acbc0c022698ba37477a66c3ed75b9379aecbcacd71baffb4f0ee6475c90f",
        "harmless_sha256": "8caf820292fd7c30b5a95add58c2bfc11a3504be21d594c8d39cdb57d328ec6a",
    }, "direction prompt metadata mismatch")
    require(metadata["summary"]["artifact_shape"] == [45, 4096], "direction artifact shape metadata mismatch")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        tensor_names = [name for name in header if name != "__metadata__"]
        require(len(tensor_names) == 32, "direction tensor count mismatch")
        for key in ("dirs", "directions.layer_end_mean"):
            require(header[key]["dtype"] == "F32", f"{key} dtype mismatch")
            require(header[key]["shape"] == [45, 4096], f"{key} shape mismatch")
        data_start = 8 + header_size
        offsets = [header[key]["data_offsets"] for key in ("dirs", "directions.layer_end_mean")]
        chunks = []
        for start, end in offsets:
            handle.seek(data_start + start)
            chunks.append(handle.read(end - start))
        require(chunks[0] == chunks[1], "dirs alias differs from primary direction")
    return {
        "bytes": DIRECTION_BYTES,
        "sha256": DIRECTION_SHA256,
        "tensors": 32,
        "shape": [45, 4096],
        "metadata_sha256": metadata_sha256,
        "bundled_metadata_reference": metadata_sha256 == DIRECTION_METADATA_REFERENCE_SHA256,
        "metadata_semantics_locked": True,
    }


def verify_source(path: pathlib.Path) -> dict:
    require(path.is_dir(), f"source directory not found: {path}")
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    require(sha256(config_path) == SOURCE_CONFIG_SHA256, "source config hash mismatch")
    require(sha256(index_path) == SOURCE_INDEX_SHA256, "source index hash mismatch")
    config = read_json(config_path)
    require(config.get("quantization_config") is None, "source is quantized")
    text = config["text_config"]
    for key, wanted in {
        "num_hidden_layers": 45,
        "hidden_size": 4096,
        "hc_mult": 4,
        "first_k_dense_replace": 3,
        "n_routed_experts": 288,
        "num_nextn_predict_layers": 1,
    }.items():
        require(text[key] == wanted, f"source geometry mismatch: {key}")
    index = read_json(index_path)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    require(len(weight_map) == 38_770, "source tensor count mismatch")
    require(len(shards) == 120, "source shard count mismatch")
    require(index["metadata"]["total_size"] == 642_646_653_816, "source indexed byte count mismatch")
    missing = [name for name in shards if not (path / name).is_file()]
    require(not missing, f"source shards missing: {missing[:8]}")
    patterns = {
        "o_proj": re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
        "dense_down": re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),
        "expert_down": re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.down_proj\.weight$"),
        "shared_down": re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"),
    }
    counts = {}
    for label, pattern in patterns.items():
        counts[label] = sum(
            1 for name in weight_map
            if (match := pattern.fullmatch(name)) and 2 <= int(match.group(1)) < 45
        )
    require(counts == {"o_proj": 43, "dense_down": 1, "expert_down": 12096, "shared_down": 42}, "source target surface mismatch")
    mtp = sum(name.startswith("model.language_model.layers.45.") for name in weight_map)
    require(mtp == 889, "source MTP tensor count mismatch")
    return {"tensors": len(weight_map), "shards": len(shards), "targets": counts, "mtp_tensors": mtp}


def verify_harness_plan() -> dict:
    plan = read_json(ROOT / "plans/ablagent-read-only-audit.json")
    require(plan["name"] == "glm53-flash-dwm-a2p8-audit", "unexpected plan name")
    require(len(plan["stages"]) == 8, "unexpected plan stage count")
    allowed = {"read_file", "grep", "glob"}
    for stage in plan["stages"]:
        require(set(stage["allowedTools"]).issubset(allowed), f"write-capable tool in {stage['id']}")
        for relative in stage.get("exactPaths", []):
            pure = pathlib.PurePosixPath(relative)
            require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe exact path: {relative}")
            require((ROOT / relative).is_file(), f"missing exact path: {relative}")
    require(plan["stages"][-1]["allowedTools"] == [], "synthesis stage must use no tools")
    return {"name": plan["name"], "stages": 8, "tools": sorted(allowed)}


def verify_manifest() -> dict:
    path = ROOT / "SHA256SUMS"
    require(path.is_file() and not path.is_symlink(), "SHA256SUMS is missing")
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        wanted, relative = line.split("  ", 1)
        pure = pathlib.PurePosixPath(relative)
        require(re.fullmatch(r"[0-9a-f]{64}", wanted) is not None, f"invalid hash: {relative}")
        require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe manifest path: {relative}")
        require(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        target = ROOT / relative
        require(target.is_file() and not target.is_symlink(), f"invalid manifest target: {relative}")
        require(sha256(target) == wanted, f"manifest mismatch: {relative}")
    expected = set()
    for target in ROOT.rglob("*"):
        if not target.is_file() or target.is_symlink():
            continue
        relative = target.relative_to(ROOT).as_posix()
        parts = pathlib.PurePosixPath(relative).parts
        if relative in RUNTIME_EXCLUSIONS or ".git" in parts or "__pycache__" in parts:
            continue
        if relative.endswith(".pyc") or relative.startswith(("artifacts/", "output/", "run/")):
            continue
        expected.add(relative)
    require(seen == expected, f"manifest coverage mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {"files": len(seen), "complete": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", type=pathlib.Path)
    parser.add_argument("--source", type=pathlib.Path)
    args = parser.parse_args()
    for path in ROOT.rglob("*"):
        require(not path.is_symlink(), f"symlink is not allowed: {path.relative_to(ROOT)}")
    lock = verify_lock()
    result = {
        "passed": True,
        "lock": lock["recipe_id"],
        "alpha": lock["operation"]["alpha"],
        "prompts": verify_prompts(),
        "asset": verify_descriptor(),
        "scripts": verify_scripts(),
        "harness": verify_harness_plan(),
        "manifest": verify_manifest(),
        "directions": verify_direction(args.directions.resolve()) if args.directions else "not supplied",
        "source": verify_source(args.source.resolve()) if args.source else "not supplied",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
