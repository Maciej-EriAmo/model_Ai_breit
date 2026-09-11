"""Uszczelnienie Most: archive/TOMB (T niskie) nie dostaje wagi jako key."""

from __future__ import annotations

import torch

from config import CANON
from most_seal import apply_t_key_seal


def test_hard_mask_zeros_tomb_key() -> None:
    B, h, N, nd = 1, 2, 4, 6
    A = torch.ones(B, h, N, N)
    tracer = torch.zeros(B, N, nd)
    tracer[0, 0, 3] = 0.0  # CLS
    tracer[0, 1, 3] = 0.95  # HOT
    tracer[0, 2, 3] = 0.08  # archive TOMB
    tracer[0, 3, 3] = 0.70  # HOT-ish
    out = apply_t_key_seal(A, tracer)
    # key index 2 (archive) → 0
    assert torch.all(out[:, :, :, 2] == 0), out[0, 0, :, 2]
    # HOT keys żyją
    assert torch.all(out[:, :, :, 1] > 0)
    assert torch.all(out[:, :, :, 3] > 0)
    # CLS key otwarty
    assert torch.all(out[:, :, :, 0] > 0)


def test_cold_penalty_reduces_warmish() -> None:
    B, h, N, nd = 1, 1, 3, 6
    A = torch.ones(B, h, N, N)
    tracer = torch.zeros(B, N, nd)
    tracer[0, 0, 3] = 1.0
    tracer[0, 1, 3] = 0.95
    tracer[0, 2, 3] = 0.30  # COLD band (>= floor, < cold)
    out = apply_t_key_seal(A, tracer)
    assert float(out[0, 0, 0, 2]) < float(out[0, 0, 0, 1])
    assert float(out[0, 0, 0, 2]) > 0  # nie hard-zero


def test_seal_can_disable() -> None:
    A = torch.ones(1, 1, 2, 2)
    tracer = torch.zeros(1, 2, 6)
    tracer[0, 1, 3] = 0.05
    out = apply_t_key_seal(A, tracer, enabled=False)
    assert torch.allclose(out, A)


def test_canon_defaults() -> None:
    assert CANON.t_attn_seal is True
    assert CANON.t_attn_floor == 0.20
    assert CANON.t_attn_cold == 0.45


if __name__ == "__main__":
    test_hard_mask_zeros_tomb_key()
    test_cold_penalty_reduces_warmish()
    test_seal_can_disable()
    test_canon_defaults()
    print("OK test_most_seal")
