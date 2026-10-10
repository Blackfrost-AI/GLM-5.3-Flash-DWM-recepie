#!/usr/bin/env python3
"""Independently verify a GLM-5.3-Flash DWM alpha-2.8 BF16 output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_locked as locked


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--directions", required=True)
    args = parser.parse_args()

    args.output_uid = 0
    args.output_gid = 0
    args.repo_id = locked.REPO_ID
    args.revision = locked.REVISION
    args.alpha = locked.ALPHA
    args.skip_early = locked.SKIP_EARLY
    state = locked.inspect_inputs(args)
    source = Path(args.source)
    output = Path(args.output)

    require(output.is_dir(), "output directory is missing")
    provenance = json.loads((output / "dwm_provenance.json").read_text())
    require(provenance["status"] == "complete_verified", "output provenance is incomplete")
    require(provenance["model"] == locked.REPO_ID, "output repository binding mismatch")
    require(provenance["source_revision"] == locked.REVISION, "output revision binding mismatch")
    require(provenance["alpha"] == locked.ALPHA, "output alpha is not 2.8")
    require(provenance["passes"] == locked.PASSES, "output pass count is not one")
    require(provenance["skip_early"] == locked.SKIP_EARLY, "output skip-early mismatch")
    require(provenance["target_counts"] == locked.EXPECTED_COUNTS, "target counts mismatch")
    require(provenance["target_total"] == sum(locked.EXPECTED_COUNTS.values()), "target total mismatch")
    require(provenance["directions"]["sha256"] == locked.DIRECTION_SHA256, "direction binding mismatch")
    require(provenance["mtp"]["transplanted_tensors"] == 889, "MTP tensor count mismatch")
    require(provenance["mtp"]["transplanted_bytes"] == 14_865_185_408, "MTP byte count mismatch")
    require(provenance["mtp"]["all_tensors_exact"] is True, "MTP provenance is not exact")

    source_index = (source / "model.safetensors.index.json").read_bytes()
    output_index = (output / "model.safetensors.index.json").read_bytes()
    require(source_index == output_index, "output index differs from source index")
    require(len(state["weight_map"]) == 38_770, "unexpected index key count")
    require(len(state["shards"]) == 120, "unexpected shard count")
    require(all((output / name).is_file() for name in state["shards"]), "output shard missing")

    verified_mtp = locked.verify_mtp(source, output, state["mtp_by_shard"])
    require(verified_mtp == 889, "full MTP comparison failed")

    sentinels = [
        "model.language_model.layers.2.self_attn.o_proj.weight",
        "model.language_model.layers.2.mlp.down_proj.weight",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
        "model.language_model.layers.44.mlp.experts.287.down_proj.weight",
        "model.language_model.layers.44.self_attn.o_proj.weight",
    ]
    metrics = [
        locked.projection_metric(
            source,
            output,
            state["weight_map"],
            state["directions"],
            key,
            locked.ALPHA,
        )
        for key in sentinels
    ]
    require(all(item["changed"] for item in metrics), "an edited sentinel is unchanged")
    require(
        all(abs(item["projection_norm_ratio"] - 1.8) < 0.02 for item in metrics),
        "projection norm ratio is outside the BF16 tolerance",
    )
    require(
        all(item["projection_cosine"] < -0.999 for item in metrics),
        "projection direction did not reverse",
    )

    untouched = [
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.1.self_attn.o_proj.weight",
        "model.language_model.layers.45.mlp.experts.0.down_proj.weight",
    ]
    for key in untouched:
        require(
            torch.equal(
                locked.get_tensor(source, state["weight_map"], key),
                locked.get_tensor(output, state["weight_map"], key),
            ),
            f"untouched tensor changed: {key}",
        )

    print(
        json.dumps(
            {
                "passed": True,
                "alpha": locked.ALPHA,
                "passes": locked.PASSES,
                "output_shards": 120,
                "index_keys": 38_770,
                "edited_tensors": sum(locked.EXPECTED_COUNTS.values()),
                "mtp_tensors_exact": verified_mtp,
                "projection_metrics": metrics,
                "untouched_sentinels_exact": untouched,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
