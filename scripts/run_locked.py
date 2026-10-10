#!/usr/bin/env python3
"""Stream the locked GLM-5.3-Flash BF16 alpha-2.8 DWM edit.

The source checkpoint is never modified. Main layers 2..44 receive the
single-shot GLM-5.2 operator W' = W - alpha * r(r^T W). Layer 45 is the
separate MTP draft layer and is explicitly transplanted from the clean source
after all main-layer shards have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import time
from pathlib import Path
from queue import Empty

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


TARGET_PATTERNS = {
    "o_proj": re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$"),
    "dense_down": re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$"),
    "expert_down": re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.\d+\.down_proj\.weight$"
    ),
    "shared_down": re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.weight$"
    ),
}
REPO_ID = "zai-org/GLM-5.3-Flash-BF16"
REVISION = "b1967181a3917ae70a437f4884748f6b8e3a1f4d"
ALPHA = 2.8
PASSES = 1
SKIP_EARLY = 2
WORKERS = 8
DIRECTION_BYTES = 151_148_124
DIRECTION_SHA256 = "9de7f65c147bb25f15ea4dea411d67048f6eb13c79a2513153fbde96dfcf343e"
SOURCE_CONFIG_SHA256 = "33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943"
SOURCE_INDEX_SHA256 = "e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f"
EXPECTED_COUNTS = {
    "o_proj": 43,
    "dense_down": 1,
    "expert_down": 12096,
    "shared_down": 42,
}
MTP_PREFIX = "model.language_model.layers.45."


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path: Path, uid: int, gid: int) -> None:
    path.chmod(0o644)
    os.chown(path, uid, gid)


def classify_target(key: str, skip_early: int, num_layers: int):
    for label, pattern in TARGET_PATTERNS.items():
        match = pattern.match(key)
        if match:
            layer = int(match.group(1))
            if skip_early <= layer < num_layers:
                return label, layer
            return None
    return None


def inspect_inputs(args):
    source = Path(args.source)
    directions_path = Path(args.directions)
    if source.resolve() == Path(args.output).resolve():
        raise RuntimeError("source and output must be different directories")
    config = json.loads((source / "config.json").read_text())
    if sha256_file(source / "config.json") != SOURCE_CONFIG_SHA256:
        raise RuntimeError("source config checksum mismatch")
    if sha256_file(source / "model.safetensors.index.json") != SOURCE_INDEX_SHA256:
        raise RuntimeError("source index checksum mismatch")
    text_config = config["text_config"]
    num_layers = int(text_config["num_hidden_layers"])
    hidden_size = int(text_config["hidden_size"])
    if num_layers != 45 or hidden_size != 4096 or int(text_config["num_nextn_predict_layers"]) != 1:
        raise RuntimeError(
            f"unexpected GLM-5.3 structure layers={num_layers} hidden={hidden_size} "
            f"nextn={text_config.get('num_nextn_predict_layers')}"
        )
    if config.get("quantization_config") is not None:
        raise RuntimeError("source checkpoint is quantized; BF16 source required")

    direction_tensors = load_file(directions_path, device="cpu")
    if directions_path.stat().st_size != DIRECTION_BYTES:
        raise RuntimeError("direction byte count mismatch")
    if sha256_file(directions_path) != DIRECTION_SHA256:
        raise RuntimeError("direction checksum mismatch")
    if "dirs" not in direction_tensors:
        raise RuntimeError("direction artifact lacks the GLM-compatible dirs key")
    directions = direction_tensors["dirs"].float().contiguous()
    if tuple(directions.shape) != (num_layers, hidden_size):
        raise RuntimeError(f"bad direction shape {tuple(directions.shape)}")
    if not torch.isfinite(directions).all():
        raise RuntimeError("directions contain non-finite values")
    norms = torch.linalg.vector_norm(directions, dim=-1)
    if float(norms.min()) < 0.999 or float(norms.max()) > 1.001:
        raise RuntimeError(f"directions are not unit normalized: {float(norms.min())}..{float(norms.max())}")

    metadata_path = directions_path.with_suffix(".metadata.json")
    direction_metadata = json.loads(metadata_path.read_text())
    model_record = direction_metadata["model"]
    if model_record["repo"] != args.repo_id or model_record["revision"] != args.revision:
        raise RuntimeError(f"direction provenance mismatch: {model_record}")
    if model_record.get("checkpoint_precision") != "bf16":
        raise RuntimeError("directions were not captured from the BF16 checkpoint")
    if (
        model_record.get("hidden_size") != 4096
        or model_record.get("num_hidden_layers") != 45
        or model_record.get("hc_mult") != 4
        or model_record.get("num_nextn_predict_layers") != 1
    ):
        raise RuntimeError("direction metadata geometry mismatch")
    method_record = direction_metadata["method"]
    if (
        method_record.get("primary_site") != "layer_end_mean"
        or method_record.get("sites") != ["layer_end_mean", "attn_input", "ffn_input"]
        or method_record.get("last_token") is not True
        or method_record.get("left_padding") is not True
        or method_record.get("reasoning_effort") != "max"
        or method_record.get("batch_size") != 32
        or method_record.get("weights_modified") is not False
        or method_record.get("mtp_layer_captured") is not False
    ):
        raise RuntimeError("direction capture-method metadata mismatch")
    prompt_record = direction_metadata["prompts"]
    if prompt_record != {
        "harmful_count": 32,
        "harmless_count": 32,
        "harmful_sha256": "178acbc0c022698ba37477a66c3ed75b9379aecbcacd71baffb4f0ee6475c90f",
        "harmless_sha256": "8caf820292fd7c30b5a95add58c2bfc11a3504be21d594c8d39cdb57d328ec6a",
    }:
        raise RuntimeError("direction prompt metadata mismatch")
    if direction_metadata.get("summary", {}).get("artifact_shape") != [45, 4096]:
        raise RuntimeError("direction metadata artifact-shape mismatch")

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    targets_by_shard: dict[str, list[tuple[str, str, int]]] = {}
    counts = {label: 0 for label in TARGET_PATTERNS}
    for key, shard in weight_map.items():
        target = classify_target(key, args.skip_early, num_layers)
        if target is None:
            continue
        label, layer = target
        counts[label] += 1
        targets_by_shard.setdefault(shard, []).append((key, label, layer))
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"DWM surface mismatch observed={counts} expected={EXPECTED_COUNTS}")

    shards = sorted(set(weight_map.values()))
    if len(shards) != 120:
        raise RuntimeError(f"expected 120 BF16 shards, found {len(shards)}")
    missing = [name for name in shards if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"missing source shards: {missing[:8]}")
    mtp_keys = sorted(key for key in weight_map if key.startswith(MTP_PREFIX))
    if not mtp_keys:
        raise RuntimeError("no layer-45 MTP tensors found")
    mtp_by_shard: dict[str, list[str]] = {}
    for key in mtp_keys:
        mtp_by_shard.setdefault(weight_map[key], []).append(key)

    return {
        "config": config,
        "text_config": text_config,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "directions": directions,
        "direction_metadata": direction_metadata,
        "direction_metadata_path": metadata_path,
        "index": index,
        "weight_map": weight_map,
        "targets_by_shard": targets_by_shard,
        "counts": counts,
        "shards": shards,
        "mtp_keys": mtp_keys,
        "mtp_by_shard": mtp_by_shard,
    }


@torch.inference_mode()
def edit_weight(weight: torch.Tensor, direction: torch.Tensor, alpha: float, device: torch.device):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise RuntimeError(f"expected BF16 matrix, got dtype={weight.dtype} shape={tuple(weight.shape)}")
    if weight.shape[0] != direction.numel():
        raise RuntimeError(f"output dimension {weight.shape[0]} != direction {direction.numel()}")
    work = weight.to(device=device, dtype=torch.float32)
    r = direction.to(device=device, dtype=torch.float32)
    projection = torch.matmul(r, work)
    work.add_(r[:, None] * projection[None, :], alpha=-alpha)
    if not torch.isfinite(work).all():
        raise RuntimeError("edit produced non-finite values")
    result = work.to(dtype=torch.bfloat16).cpu().contiguous()
    del work, r, projection
    return result


def worker_main(
    worker_id: int,
    gpu_id: int,
    shard_names: list[str],
    source_string: str,
    output_string: str,
    targets_by_shard,
    directions,
    alpha: float,
    uid: int,
    gid: int,
    queue,
):
    source = Path(source_string)
    output = Path(output_string)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    edited_counts = {label: 0 for label in TARGET_PATTERNS}
    processed = 0
    started = time.monotonic()
    try:
        for position, shard_name in enumerate(shard_names, 1):
            source_path = source / shard_name
            output_path = output / shard_name
            manifest_path = output / f".{shard_name}.dwm.json"
            if output_path.exists() and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "complete" or manifest.get("alpha") != alpha:
                    raise RuntimeError(f"invalid resume manifest for {shard_name}")
                for label, value in manifest["edited_counts"].items():
                    edited_counts[label] += int(value)
                processed += 1
                log(f"worker={worker_id} resume {position}/{len(shard_names)} {shard_name}")
                continue
            if output_path.exists() or manifest_path.exists():
                raise RuntimeError(f"partial output without matching manifest: {shard_name}")

            shard_targets = targets_by_shard.get(shard_name, [])
            shard_counts = {label: 0 for label in TARGET_PATTERNS}
            if not shard_targets:
                shutil.copy2(source_path, output_path)
            else:
                tensors = load_file(source_path, device="cpu")
                for key, label, layer in shard_targets:
                    tensors[key] = edit_weight(tensors[key], directions[layer], alpha, device)
                    shard_counts[label] += 1
                    edited_counts[label] += 1
                temporary = output / f".{shard_name}.tmp"
                temporary.unlink(missing_ok=True)
                save_file(tensors, temporary, metadata={"format": "pt"})
                os.replace(temporary, output_path)
                del tensors
            publish(output_path, uid, gid)
            manifest = {
                "status": "complete",
                "worker": worker_id,
                "source_shard": shard_name,
                "alpha": alpha,
                "edited_counts": shard_counts,
                "completed_at_utc": utc_now(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            publish(manifest_path, uid, gid)
            processed += 1
            torch.cuda.empty_cache()
            log(
                f"worker={worker_id} shard={position}/{len(shard_names)} {shard_name} "
                f"targets={sum(shard_counts.values())} elapsed={time.monotonic()-started:.0f}s"
            )
        queue.put({"status": "ok", "worker": worker_id, "processed": processed, "counts": edited_counts})
    except Exception as error:
        queue.put({"status": "error", "worker": worker_id, "error": f"{type(error).__name__}: {error}"})
        raise


def transplant_mtp(source: Path, output: Path, mtp_by_shard, uid: int, gid: int):
    transplanted_keys = 0
    transplanted_bytes = 0
    for shard_name, keys in sorted(mtp_by_shard.items()):
        source_tensors = load_file(source / shard_name, device="cpu")
        output_tensors = load_file(output / shard_name, device="cpu")
        for key in keys:
            clean = source_tensors[key].clone().contiguous()
            output_tensors[key] = clean
            transplanted_keys += 1
            transplanted_bytes += clean.numel() * clean.element_size()
        temporary = output / f".{shard_name}.mtp.tmp"
        temporary.unlink(missing_ok=True)
        save_file(output_tensors, temporary, metadata={"format": "pt"})
        os.replace(temporary, output / shard_name)
        publish(output / shard_name, uid, gid)
        del source_tensors, output_tensors
        log(f"MTP_TRANSPLANT shard={shard_name} tensors={len(keys)}")
    return transplanted_keys, transplanted_bytes


def verify_mtp(source: Path, output: Path, mtp_by_shard):
    verified = 0
    for shard_name, keys in sorted(mtp_by_shard.items()):
        with safe_open(source / shard_name, framework="pt", device="cpu") as clean_file:
            with safe_open(output / shard_name, framework="pt", device="cpu") as edited_file:
                for key in keys:
                    if not torch.equal(clean_file.get_tensor(key), edited_file.get_tensor(key)):
                        raise RuntimeError(f"MTP verification failed for {key}")
                    verified += 1
    return verified


def get_tensor(root: Path, weight_map, key: str):
    with safe_open(root / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def projection_metric(source: Path, output: Path, weight_map, directions, key: str, alpha: float):
    layer_match = re.search(r"\.layers\.(\d+)\.", key)
    layer = int(layer_match.group(1))
    before = get_tensor(source, weight_map, key).float()
    after = get_tensor(output, weight_map, key).float()
    r = directions[layer].float()
    before_projection = torch.matmul(r, before)
    after_projection = torch.matmul(r, after)
    denominator = before_projection.norm().clamp_min(1e-12)
    ratio = float(after_projection.norm() / denominator)
    cosine = float(torch.nn.functional.cosine_similarity(before_projection, after_projection, dim=0))
    residual = float((after_projection - (1.0 - alpha) * before_projection).norm() / denominator)
    return {
        "key": key,
        "layer": layer,
        "changed": not torch.equal(before.to(torch.bfloat16), after.to(torch.bfloat16)),
        "projection_norm_ratio": ratio,
        "projection_cosine": cosine,
        "operator_relative_error_after_bf16_rounding": residual,
    }


def finalize(args, state, output: Path):
    source = Path(args.source)
    uid, gid = args.output_uid, args.output_gid
    shutil.copy2(source / "model.safetensors.index.json", output / "model.safetensors.index.json")
    publish(output / "model.safetensors.index.json", uid, gid)
    for source_file in source.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == "config.json" or source_file.name == "model.safetensors.index.json":
            continue
        if source_file.name.endswith(".safetensors"):
            continue
        shutil.copy2(source_file, output / source_file.name)
        publish(output / source_file.name, uid, gid)

    mtp_count, mtp_bytes = transplant_mtp(source, output, state["mtp_by_shard"], uid, gid)
    verified_mtp = verify_mtp(source, output, state["mtp_by_shard"])
    if mtp_count != len(state["mtp_keys"]) or verified_mtp != mtp_count:
        raise RuntimeError(f"MTP count mismatch transplanted={mtp_count} verified={verified_mtp}")

    sentinels = [
        "model.language_model.layers.2.self_attn.o_proj.weight",
        "model.language_model.layers.2.mlp.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.44.mlp.experts.287.down_proj.weight",
        "model.language_model.layers.44.self_attn.o_proj.weight",
    ]
    metrics = [
        projection_metric(source, output, state["weight_map"], state["directions"], key, args.alpha)
        for key in sentinels
    ]
    if not all(metric["changed"] for metric in metrics):
        raise RuntimeError("one or more DWM sentinel tensors were unchanged")
    if not all(abs(metric["projection_norm_ratio"] - abs(1.0 - args.alpha)) < 0.02 for metric in metrics):
        raise RuntimeError(f"projection ratio validation failed: {metrics}")
    if not all(metric["projection_cosine"] < -0.999 for metric in metrics):
        raise RuntimeError(f"projection sign validation failed: {metrics}")

    unchanged_sentinels = [
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.1.self_attn.o_proj.weight",
        "model.language_model.layers.45.mlp.experts.0.down_proj.weight",
    ]
    for key in unchanged_sentinels:
        if not torch.equal(
            get_tensor(source, state["weight_map"], key),
            get_tensor(output, state["weight_map"], key),
        ):
            raise RuntimeError(f"untouched sentinel changed: {key}")

    directions_path = Path(args.directions)
    provenance = {
        "status": "complete_verified",
        "model": args.repo_id,
        "source_revision": args.revision,
        "source_path": args.source_display_path,
        "output_path": args.output_display_path,
        "operator": "W' = W - alpha * r(r^T W)",
        "alpha": args.alpha,
        "passes": 1,
        "skip_early": args.skip_early,
        "main_layers_edited": [args.skip_early, state["num_layers"] - 1],
        "target_counts": state["counts"],
        "target_total": sum(state["counts"].values()),
        "directions": {
            "path": args.directions_display_path,
            "sha256": sha256_file(directions_path),
            "metadata_sha256": sha256_file(state["direction_metadata_path"]),
            "shape": list(state["directions"].shape),
            "site": "layer_end_mean",
        },
        "mtp": {
            "layer": 45,
            "source": "clean BF16 checkpoint",
            "transplanted_tensors": mtp_count,
            "transplanted_bytes": mtp_bytes,
            "all_tensors_exact": True,
        },
        "verification": {
            "edited_sentinels": metrics,
            "unchanged_sentinels_exact": unchanged_sentinels,
            "output_shards": len(state["shards"]),
            "output_index_keys": len(state["weight_map"]),
        },
        "completed_at_utc": utc_now(),
    }

    config = state["config"]
    config["blackfrost_dwm"] = {
        "method": "single-shot residual-write output-space direction subtraction",
        "alpha": args.alpha,
        "skip_early": args.skip_early,
        "direction_site": "layer_end_mean",
        "direction_sha256": provenance["directions"]["sha256"],
        "source_repo": args.repo_id,
        "source_revision": args.revision,
        "mtp_layer_45_transplanted_from_clean_bf16": True,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    publish(output / "config.json", uid, gid)
    (output / "dwm_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    publish(output / "dwm_provenance.json", uid, gid)
    readme = (
        "# GLM-5.3-Flash DWM BF16\n\n"
        "Blackfrost-AI single-shot DWM build from the pinned official BF16 checkpoint.\n\n"
        f"- Alpha: `{args.alpha}`\n"
        f"- Skip early: `{args.skip_early}`\n"
        f"- Edited residual-write tensors: `{sum(state['counts'].values())}`\n"
        f"- MTP layer 45 tensors transplanted from clean BF16: `{mtp_count}`\n"
        "- NVFP4 conversion: not performed in this pass\n"
    )
    (output / "README_BLACKFROST_DWM.md").write_text(readme)
    publish(output / "README_BLACKFROST_DWM.md", uid, gid)
    return provenance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply the immutable GLM-5.3-Flash BF16 DWM alpha-2.8 method"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--directions", required=True)
    parser.set_defaults(
        repo_id=REPO_ID,
        revision=REVISION,
        alpha=ALPHA,
        skip_early=SKIP_EARLY,
        workers=WORKERS,
    )
    parser.add_argument("--output-uid", type=int, default=os.getuid())
    parser.add_argument("--output-gid", type=int, default=os.getgid())
    parser.add_argument("--source-display-path", required=True)
    parser.add_argument("--output-display-path", required=True)
    parser.add_argument("--directions-display-path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.repo_id != REPO_ID
        or args.revision != REVISION
        or args.alpha != ALPHA
        or args.skip_early != SKIP_EARLY
        or args.workers != WORKERS
        or PASSES != 1
    ):
        raise RuntimeError("locked GLM-5.3-Flash alpha-2.8 invariants changed")
    state = inspect_inputs(args)
    log(
        f"INSPECT source=BF16 layers={state['num_layers']} hidden={state['hidden_size']} "
        f"shards={len(state['shards'])} targets={json.dumps(state['counts'], sort_keys=True)} "
        f"mtp_keys={len(state['mtp_keys'])}"
    )
    if args.dry_run:
        log("DRY_RUN_DONE no output created and no weights modified")
        return
    if torch.cuda.device_count() != args.workers:
        raise RuntimeError(f"expected {args.workers} visible GPUs, found {torch.cuda.device_count()}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.chown(output, args.output_uid, args.output_gid)
    shard_partitions = [state["shards"][worker::args.workers] for worker in range(args.workers)]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    started = time.monotonic()
    for worker_id, shard_names in enumerate(shard_partitions):
        process = context.Process(
            target=worker_main,
            args=(
                worker_id,
                worker_id,
                shard_names,
                args.source,
                args.output,
                state["targets_by_shard"],
                state["directions"],
                args.alpha,
                args.output_uid,
                args.output_gid,
                queue,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    results = []
    while True:
        try:
            results.append(queue.get_nowait())
        except Empty:
            break
    errors = [result for result in results if result["status"] != "ok"]
    bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
    if errors or bad_exits or len(results) != args.workers:
        raise RuntimeError(f"worker failure results={results} exitcodes={bad_exits}")
    aggregate = {label: 0 for label in TARGET_PATTERNS}
    for result in results:
        for label, value in result["counts"].items():
            aggregate[label] += value
    if aggregate != EXPECTED_COUNTS:
        raise RuntimeError(f"aggregate edit mismatch observed={aggregate} expected={EXPECTED_COUNTS}")
    log(f"DWM_WRITE_DONE counts={json.dumps(aggregate, sort_keys=True)} elapsed={time.monotonic()-started:.0f}s")

    provenance = finalize(args, state, output)
    log(
        f"DWM_BF16_DONE output={args.output_display_path} targets={provenance['target_total']} "
        f"mtp_exact={provenance['mtp']['all_tensors_exact']} elapsed={time.monotonic()-started:.0f}s"
    )


if __name__ == "__main__":
    main()
