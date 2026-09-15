#!/usr/bin/env python3
"""Describe the supported Q4_K TP layout without reading weight payloads.

The model descriptors prove topology; successful runtime allocation messages
prove that both ranks installed that layout. This is not an expert-ID split.
"""
import argparse
import importlib.util
from pathlib import Path
import re


def model_fields(path):
    spec = importlib.util.spec_from_file_location(
        "glm5_gguf", Path(__file__).with_name("check-glm5-next-gguf.py"))
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    meta, tensors = reader.read(Path(path))
    reader.require(meta, "general.architecture", "glm5-next")
    for name, value in {
        "block_count": 46, "trunk_block_count": 45,
        "embedding_length": 4096, "expert_count": 288,
        "expert_used_count": 8, "expert_feed_forward_length": 2048,
    }.items():
        reader.require(meta, "glm5-next." + name, value)
    # Same routed trunk range and packing accepted by the runtime installer.
    for layer in range(3, meta["glm5-next.trunk_block_count"]):
        for projection in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            dims = (2048, 4096, 288) if projection == "down" else (4096, 2048, 288)
            tensor = tensors.get(name)
            if tensor is None or tensor[:2] != (dims, 12):
                raise ValueError(f"unsupported Q4_K TP tensor: {name}")
    return {
        "tp_weight_layout": "q4k-ffn-intermediate",
        "tp_intermediate_size": "2048",
        "tp_intermediate_shards": "1024/1024",
        "tp_expert_count": "288", "tp_experts_used": "8",
        "tp_reduce_op": "sum", "tp_reduce_scope": "all-ranks",
        "tp_reduce_count": "42", "tp_reduce_width": "4096",
        "tp_reduce_dtype": "f32",
    }


def verify_logs(coordinator, worker):
    pattern = re.compile(
        r"^ds4: GLM5 compact Q4_K K-shard active: rank=(\d+) "
        r"layers=(\d+) rows=(\d+):(\d+) down-bytes=(\d+):(\d+)$",
        re.MULTILINE)
    for rank, path in enumerate((coordinator, worker)):
        matches = pattern.findall(Path(path).read_text())
        expected = tuple(map(str, (
            rank, 42, rank * 1024, (rank + 1) * 1024,
            rank * 576, (rank + 1) * 576)))
        if matches != [expected]:
            raise ValueError(f"rank {rank} lacks a unique matching TP allocation: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--logs", nargs=2, metavar=("COORDINATOR", "WORKER"))
    args = parser.parse_args()
    try:
        if not args.model and not args.logs:
            parser.error("provide a model or --logs")
        fields = model_fields(args.model) if args.model else {}
        if args.logs:
            verify_logs(*args.logs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print("".join(f"{key}={value}\n" for key, value in fields.items()), end="")


if __name__ == "__main__":
    main()
