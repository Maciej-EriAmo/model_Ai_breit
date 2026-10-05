"""Uszczelnienie Most: archive/TOMB (T niskie) nie dostaje wagi jako key."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

import torch

from config import CANON
from most_seal import apply_t_key_seal
from phimost_learn import seal_for_source
from recall_model import PhiBridgeEncoder


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


def test_energy_proxy_is_not_sealed() -> None:
    assert seal_for_source("struktura") is False
    assert seal_for_source("qm") is False
    assert seal_for_source("glyphs_photo") is False
    assert seal_for_source("karmazyn") is None

    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    hot = torch.zeros(1, 4, 6)
    hot[0, :, 0] = 1.0
    hot[0, :, 3] = 0.95
    cold = hot.clone()
    cold[0, 2, 3] = 0.32  # Fe 3d, m=-2
    sealed = PhiBridgeEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32, h=2, t_seal=True)
    opened = PhiBridgeEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32, h=2, t_seal=False)
    opened.load_state_dict(sealed.state_dict())
    with torch.no_grad():
        hot_s, _ = sealed(x, hot)
        hot_o, _ = opened(x, hot)
        cold_s, _ = sealed(x, cold)
        cold_o, _ = opened(x, cold)
    assert torch.allclose(hot_s, hot_o, atol=1e-5)
    assert not torch.allclose(cold_s, cold_o, atol=1e-4)


def test_canon_defaults() -> None:
    assert CANON.t_attn_seal is True
    assert CANON.t_attn_floor == 0.20
    assert CANON.t_attn_cold == 0.45


if __name__ == "__main__":
    test_hard_mask_zeros_tomb_key()
    test_cold_penalty_reduces_warmish()
    test_seal_can_disable()
    test_energy_proxy_is_not_sealed()
    test_canon_defaults()
    print("OK test_most_seal")
