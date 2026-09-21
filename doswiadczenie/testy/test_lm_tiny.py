"""#1b — tiny LM forward/backward Most vs Softmax."""

from __future__ import annotations

import _labroot  # noqa: F401

import torch

from lm_tiny import TinyLM, TinyLMConfig, count_params, lm_loss


def test_params_well_under_1m() -> None:
    cfg = TinyLMConfig()
    m = TinyLM(cfg, attn="most")
    s = TinyLM(cfg, attn="softmax")
    assert count_params(m) < 1_000_000
    assert count_params(s) < 1_000_000


def test_most_and_softmax_train_one_step() -> None:
    cfg = TinyLMConfig(d=32, num_layers=1, max_seq=32)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    for attn in ("most", "softmax"):
        model = TinyLM(cfg, attn=attn)
        logits = model(idx)
        assert logits.shape == (4, 16, cfg.vocab_size)
        loss = lm_loss(logits, idx)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, attn
        assert all(torch.isfinite(g).all() for g in grads), attn


if __name__ == "__main__":
    test_params_well_under_1m()
    test_most_and_softmax_train_one_step()
    print("OK test_lm_tiny")
