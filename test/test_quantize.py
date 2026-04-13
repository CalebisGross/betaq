#!/usr/bin/env python3
"""Unit tests for quantize_rq4.py — validates Q8_0 and RQ4 block quantization."""

import struct
import sys
from pathlib import Path

import numpy as np

# Add parent dir so we can import the quantizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from quantize_rq4 import (
    QK_Q8_0, QK_RQ4, RQ4_CODEBOOK_FLOAT,
    quantize_block_q8_0, quantize_block_rq4,
)


def test_q8_0_roundtrip():
    """Q8_0: quantize -> dequantize should be close to original."""
    rng = np.random.default_rng(42)
    block = rng.standard_normal(QK_Q8_0).astype(np.float32)

    raw = quantize_block_q8_0(block)
    assert len(raw) == 34, f"expected 34 bytes, got {len(raw)}"

    # Dequantize
    scale = struct.unpack('<e', raw[:2])[0]
    quants = np.frombuffer(raw[2:], dtype=np.int8)
    reconstructed = quants.astype(np.float32) * scale

    # Q8_0 should be within ~1% of original for typical values
    max_err = np.abs(block - reconstructed).max()
    assert max_err < 0.05, f"Q8_0 max error {max_err:.6f} exceeds threshold"
    print(f"  Q8_0 roundtrip: max_err={max_err:.6f}")


def test_q8_0_zero_block():
    """Q8_0: zero block should produce zero scale and zero quants."""
    block = np.zeros(QK_Q8_0, dtype=np.float32)
    raw = quantize_block_q8_0(block)
    scale = struct.unpack('<e', raw[:2])[0]
    assert scale == 0.0
    assert raw[2:] == bytes(QK_Q8_0)
    print(f"  Q8_0 zero block: OK")


def test_rq4_roundtrip():
    """RQ4: quantize -> dequantize should reconstruct within codebook resolution."""
    rng = np.random.default_rng(42)
    block = rng.standard_normal(QK_RQ4).astype(np.float32) * 0.1

    scale, packed = quantize_block_rq4(block)
    assert len(packed) == QK_RQ4 // 2, f"expected {QK_RQ4//2} bytes, got {len(packed)}"

    # Dequantize
    indices = np.zeros(QK_RQ4, dtype=np.uint8)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    for j in range(QK_RQ4 // 2):
        indices[j * 2] = packed_arr[j] & 0x0F
        indices[j * 2 + 1] = (packed_arr[j] >> 4) & 0x0F

    reconstructed = RQ4_CODEBOOK_FLOAT[indices] * scale

    # 4-bit codebook has ~16 levels so error can be larger
    max_err = np.abs(block - reconstructed).max()
    assert max_err < 0.1, f"RQ4 max error {max_err:.6f} exceeds threshold"
    print(f"  RQ4 roundtrip: max_err={max_err:.6f}")


def test_rq4_zero_block():
    """RQ4: zero block should produce zero scale."""
    block = np.zeros(QK_RQ4, dtype=np.float32)
    scale, packed = quantize_block_rq4(block)
    assert scale == 0.0
    assert packed == bytes(QK_RQ4 // 2)
    print(f"  RQ4 zero block: OK")


def test_rq4_indices_in_range():
    """RQ4: all indices should be 0-15 (4-bit)."""
    rng = np.random.default_rng(123)
    block = rng.standard_normal(QK_RQ4).astype(np.float32)
    scale, packed = quantize_block_rq4(block)
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    for j in range(QK_RQ4 // 2):
        lo = packed_arr[j] & 0x0F
        hi = (packed_arr[j] >> 4) & 0x0F
        assert lo < 16, f"lo nibble {lo} out of range at byte {j}"
        assert hi < 16, f"hi nibble {hi} out of range at byte {j}"
    print(f"  RQ4 indices in range: OK")


def test_embed_pattern_detection():
    """Verify embed_patterns match the tensor names we expect."""
    embed_patterns = ("embd", "embed")
    should_match = [
        "token_embd.weight",
        "per_layer_token_embd.weight",
        "model.embed_tokens.weight",
        "embedding.weight",
    ]
    should_not_match = [
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
    ]
    for name in should_match:
        assert any(p in name for p in embed_patterns), f"{name} should match embed_patterns"
    for name in should_not_match:
        assert not any(p in name for p in embed_patterns), f"{name} should NOT match embed_patterns"
    print(f"  Embed pattern detection: OK")


def test_bool_metadata_type():
    """Verify bool() preserves Python bool type (not int)."""
    # Simulating what the metadata loop does
    raw_val = np.array([1], dtype=np.uint8)  # how gguf stores bools
    via_int = int(raw_val[0])
    via_bool = bool(raw_val[0])
    assert type(via_bool) is bool, f"bool() should return bool, got {type(via_bool)}"
    assert type(via_int) is int, f"int() should return int, got {type(via_int)}"
    assert via_bool is True
    # The bug: int(1) and bool(True) have different GGUF type dispatch
    print(f"  Bool metadata type: OK")


if __name__ == "__main__":
    print("Running quantizer tests...\n")
    test_q8_0_roundtrip()
    test_q8_0_zero_block()
    test_rq4_roundtrip()
    test_rq4_zero_block()
    test_rq4_indices_in_range()
    test_embed_pattern_detection()
    test_bool_metadata_type()
    print("\nAll tests passed.")
