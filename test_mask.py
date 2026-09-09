"""P1.2 — mask w InfoNCE / metrics / uwadze (padding nie rankuje)."""

from __future__ import annotations

import torch

from recall_model import (
    PhiBridgeEncoder,
    info_nce_loss,
    recall_metrics,
)


def test_pad_never_top1():
    torch.manual_seed(0)
    B, N, d = 2, 6, 8
    # embeddingi: slot 0=query, 1=pozytyw, 5=pad (ale sztucznie największy cosinus)
    tok = torch.randn(B, N, d)
    tok[:, 0] = torch.tensor([1.0] + [0.0] * (d - 1))
    tok[:, 1] = torch.tensor([0.9] + [0.1] * (d - 1))
    tok[:, 5] = torch.tensor([1.0] + [0.0] * (d - 1))  # identyczny z query → wygrałby bez mask
    query = torch.zeros(B, dtype=torch.long)
    pos = torch.ones(B, dtype=torch.long)
    mask = torch.ones(B, N)
    mask[:, 4:] = 0.0  # pad 4,5

    m = recall_metrics(tok, query, pos, mask=mask)
    assert m["r@1"] == 1.0, m

    # bez maski pad/klon query wygrywa → r@1 spada
    m_bad = recall_metrics(tok, query, pos, mask=None)
    assert m_bad["r@1"] < 1.0, m_bad


def test_info_nce_ignores_pad():
    torch.manual_seed(1)
    B, N, d = 4, 5, 8
    tok = torch.randn(B, N, d)
    query = torch.zeros(B, dtype=torch.long)
    pos = torch.ones(B, dtype=torch.long)
    mask = torch.ones(B, N)
    mask[:, -1] = 0.0
    loss = info_nce_loss(tok, query, pos, mask=mask)
    assert torch.isfinite(loss)


def test_encoder_forward_mask():
    torch.manual_seed(2)
    B, N, d_x, n = 2, 7, 8, 6
    x = torch.randn(B, N, d_x)
    t = torch.randn(B, N, n)
    mask = torch.ones(B, N)
    mask[:, 5:] = 0.0
    enc = PhiBridgeEncoder(input_dim=d_x, dim_tracer=n, num_layers=1, d=32)
    tok, _ = enc(x, t, mask)
    assert tok.shape == (B, N, 32)
    # paddowane sloty mogą mieć wektor, ale nie powinny psuć kształtu
    loss = info_nce_loss(
        tok,
        torch.zeros(B, dtype=torch.long),
        torch.ones(B, dtype=torch.long),
        mask=mask,
    )
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_pad_never_top1()
    test_info_nce_ignores_pad()
    test_encoder_forward_mask()
    print("OK — mask honored")
