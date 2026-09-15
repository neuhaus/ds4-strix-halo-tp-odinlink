#!/usr/bin/env python3
"""Exercise GGUF metadata + runtime layout validation with no GPU/payload."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from glm5_tp_layout import model_fields, verify_logs


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def gguf(path, *, used=8, quant=12, width=2048):
    meta = {"block_count": 46, "trunk_block_count": 45,
            "embedding_length": 4096, "expert_count": 288,
            "expert_used_count": used, "expert_feed_forward_length": 2048}
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 42 * 3, len(meta) + 1))
    data += string("general.architecture") + struct.pack("<I", 8) + string("glm5-next")
    for key, value in meta.items():
        data += string("glm5-next." + key) + struct.pack("<II", 4, value)
    for layer in range(3, 45):
        for projection in ("gate", "up", "down"):
            dims = (width, 4096, 288) if projection == "down" else (4096, width, 288)
            data += string(f"blk.{layer}.ffn_{projection}_exps.weight")
            data += struct.pack("<I3QIQ", 3, *dims, quant, 0)
    path.write_bytes(data)


class LayoutTest(unittest.TestCase):
    def test_model_descriptors_and_fail_closed_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "header.gguf"
            gguf(model)
            fields = model_fields(model)
            self.assertEqual(fields["tp_intermediate_shards"], "1024/1024")
            self.assertEqual(fields["tp_reduce_count"], "42")
            for mismatch in ({"used": 7}, {"quant": 10}, {"width": 1024}):
                gguf(model, **mismatch)
                with self.assertRaises(ValueError):
                    model_fields(model)

    def test_both_allocations_required(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Path(directory) / "coordinator.log"
            worker = Path(directory) / "worker.log"
            left = "ds4: GLM5 compact Q4_K K-shard active: rank=0 layers=42 rows=0:1024 down-bytes=0:576\n"
            right = "ds4: GLM5 compact Q4_K K-shard active: rank=1 layers=42 rows=1024:2048 down-bytes=576:1152\n"
            coordinator.write_text(left)
            worker.write_text(right)
            verify_logs(coordinator, worker)
            for bad in ("", left, right * 2, right.replace("576:1152", "0:576")):
                worker.write_text(bad)
                with self.assertRaises(ValueError):
                    verify_logs(coordinator, worker)


if __name__ == "__main__":
    unittest.main()
