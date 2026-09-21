"""Kryterium #2 — BF16: zamek + krótki trening bez NaN/Inf."""

from __future__ import annotations

import _labroot  # noqa: F401

import torch

from bridge import BridgeTransformer
from config import CANON
from precision import (
    amp_context,
    assert_finite,
    bf16_supported,
    set_finite_check,
)
from recall_data import generate_recall_batch
from recall_model import PhiBridgeEncoder, train_encoder
from synthetic_phi import generate_phi_tree


def test_bf16_supported_on_this_host() -> None:
    assert bf16_supported(), "CPU/GPU musi liczyć bf16 (gate #2)"


def test_bridge_lock_finite_under_bf16_autocast() -> None:
    set_finite_check(True)
    try:
        torch.manual_seed(0)
        batch = generate_phi_tree(4, seq_len=16, d_x=CANON.d_x, n_dim=CANON.n_dim)
        model = BridgeTransformer(
            input_dim=CANON.d_x,
            d=CANON.d_model,
            h=CANON.n_heads,
            n_classes=CANON.n_classes,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            coupling=CANON.coupling,
        )
        with amp_context(torch.bfloat16):
            logits, w, dE = model(batch.content, batch.tracer, batch.mask)
        assert_finite(logits, "logits")
        assert_finite(w, "w")
        assert_finite(dE, "dE")
        assert torch.isfinite(logits.float()).all()
    finally:
        set_finite_check(False)


def test_encoder_train_bf16_no_nan() -> None:
    set_finite_check(True)
    try:
        torch.manual_seed(1)
        train = [generate_recall_batch(4) for _ in range(6)]
        val = [generate_recall_batch(4) for _ in range(2)]
        model = PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
        )
        m = train_encoder(
            model,
            train,
            val,
            epochs=4,
            lr=CANON.lr,
            return_curve=True,
            amp_dtype=torch.bfloat16,
            finite_check=True,
        )
        assert m["amp_dtype"] in ("bfloat16", "torch.bfloat16")
        assert torch.isfinite(torch.tensor(m["r@1"]))
        for row in m["curve"]:
            assert torch.isfinite(torch.tensor(row["loss"]))
            assert torch.isfinite(torch.tensor(row["r@1"]))
    finally:
        set_finite_check(False)


def test_stress_softplus_square_bf16_path() -> None:
    """Duże S w bf16 — softplus² nie może dać NaN w zamku (g²+dE²+eps)."""
    set_finite_check(True)
    try:
        torch.manual_seed(2)
        model = BridgeTransformer(
            input_dim=8, d=32, h=2, n_classes=4, dim_tracer=6, num_layers=2
        )
        x = torch.randn(2, 24, 8) * 3.0
        tr = torch.randn(2, 24, 6)
        with amp_context(torch.bfloat16):
            logits, w, dE = model(x, tr, None)
        assert torch.isfinite(logits.float()).all()
        assert w is None or torch.isfinite(w.float()).all()
        assert dE is None or torch.isfinite(dE.float()).all()
    finally:
        set_finite_check(False)


if __name__ == "__main__":
    test_bf16_supported_on_this_host()
    test_bridge_lock_finite_under_bf16_autocast()
    test_encoder_train_bf16_no_nan()
    test_stress_softplus_square_bf16_path()
    print("OK test_bf16")
