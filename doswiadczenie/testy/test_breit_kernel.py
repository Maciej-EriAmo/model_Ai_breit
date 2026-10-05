"""Most-1: szerokość rośnie ze szczytem C, znak delta_E nic nie zmienia."""

import _labroot  # noqa: F401

import torch
import torch.nn.functional as F

from breit_kernel import breit_wigner_rows
from bridge import SoftmaxTransformer, masked_softmax
from lm_tiny import MostKnobs, TinyLM, TinyLMConfig
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder


def test_row_matches_the_closed_form_and_ignores_the_sign():
    coupling = torch.tensor([[[[1.0, 0.2, 0.05]]]], dtype=torch.float64)
    delta = torch.tensor([[[[0.0, 1.5, -1.5]]]], dtype=torch.float64)
    log_gamma0 = torch.zeros(1, 1, 1, dtype=torch.float64)
    beta_raw = torch.zeros(1, 1, 1, dtype=torch.float64)
    got = breit_wigner_rows(coupling, delta, log_gamma0, beta_raw)

    mass = coupling.sum(dim=-1, keepdim=True)
    kappa = (coupling / mass).amax(dim=-1, keepdim=True)
    gamma0 = F.softplus(log_gamma0) + 1e-3
    beta = F.softplus(beta_raw)
    gamma = gamma0 * (1.0 + beta * kappa)
    raw = gamma * coupling / (gamma.square() + delta.square() + 1e-12)
    assert torch.allclose(got, raw)
    flipped = breit_wigner_rows(coupling, -delta, log_gamma0, beta_raw)
    assert torch.allclose(got, flipped)


def test_a_larger_beta_lowers_the_peak_share():
    coupling = torch.tensor([[[[5.0, 0.1, 0.1]]]], dtype=torch.float64)
    delta = torch.tensor([[[[0.0, 2.0, 2.0]]]], dtype=torch.float64)
    log_gamma0 = torch.zeros(1, 1, 1, dtype=torch.float64)
    narrow = breit_wigner_rows(coupling, delta, log_gamma0, torch.full((1, 1, 1), -4.0))
    wide = breit_wigner_rows(coupling, delta, log_gamma0, torch.full((1, 1, 1), 2.0))

    def peak_share(weights: torch.Tensor) -> float:
        return float(weights[..., 0] / weights.sum(dim=-1))

    assert peak_share(wide) < peak_share(narrow)


def test_blocked_key_stays_zero_and_does_not_set_kappa():
    coupling = torch.tensor([[[[0.0, 4.0, 1.0]]]], dtype=torch.float64)
    delta = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    log_gamma0 = torch.zeros(1, 1, 1, dtype=torch.float64)
    beta_raw = torch.zeros(1, 1, 1, dtype=torch.float64)
    got = breit_wigner_rows(coupling, delta, log_gamma0, beta_raw)
    assert float(got[0, 0, 0, 0]) == 0.0
    visible = coupling[..., 1:]
    kappa = (visible / visible.sum()).amax()
    gamma = (F.softplus(log_gamma0) + 1e-3) * (1.0 + F.softplus(beta_raw) * kappa)
    assert torch.allclose(got[..., 1:], gamma * visible / (gamma.square() + 1e-12))


def test_sign_modes_are_rejected_and_a_forward_stays_finite():
    try:
        PhiBridgeEncoder(delta_mode="onesided")
    except ValueError:
        pass
    else:
        raise AssertionError("onesided powinien paść")
    enc = PhiBridgeEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32, h=2)
    x = torch.randn(2, 5, 8)
    tracer = torch.randn(2, 5, 6)
    tok, _dE = enc(x, tracer)
    assert tok.shape == (2, 5, 32)
    assert torch.isfinite(tok).all()

    lm = TinyLM(TinyLMConfig(d=32, h=2, num_layers=1, max_seq=16), attn="most")
    logits = lm(torch.randint(0, 128, (2, 8)))
    assert logits.shape == (2, 8, 128)
    assert torch.isfinite(logits).all()
    assert MostKnobs().delta_mode == "square"


def test_softmax_twin_survives_a_huge_self_score():
    n = 6
    eye = torch.eye(n, dtype=torch.bool)
    scores = torch.zeros(1, 1, n, n)
    scores[..., range(n), range(n)] = 80.0
    got = masked_softmax(scores, eye)
    assert torch.isfinite(got).all()
    assert float(got[0, 0, 0, 0]) == 0.0
    assert torch.allclose(got[0, 0, 0, 1:], torch.full((n - 1,), 1.0 / (n - 1)))
    assert torch.allclose(got.sum(-1), torch.ones(1, 1, n))

    plain = torch.randn(2, 2, n, n)
    pre = torch.softmax(plain.masked_fill(eye, -1e9), dim=-1)
    pre = pre.masked_fill(eye, 0.0)
    pre = pre / pre.sum(-1, keepdim=True)
    assert torch.allclose(masked_softmax(plain, eye), pre, atol=1e-5)

    sm = SoftmaxTransformer(input_dim=8, d=32, h=2, n_classes=2, dim_tracer=6, num_layers=1)
    enc = PhiSoftmaxEncoder(input_dim=8, dim_tracer=6, num_layers=1, d=32, h=2)
    x = torch.randn(2, 5, 8)
    tr = torch.randn(2, 5, 6)
    logits, w, _ = sm(x, tr)
    tok, _ = enc(x, tr)
    assert torch.isfinite(logits).all() and torch.isfinite(w).all()
    assert torch.isfinite(tok).all()
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


if __name__ == "__main__":
    test_row_matches_the_closed_form_and_ignores_the_sign()
    test_a_larger_beta_lowers_the_peak_share()
    test_blocked_key_stays_zero_and_does_not_set_kappa()
    test_sign_modes_are_rejected_and_a_forward_stays_finite()
    test_softmax_twin_survives_a_huge_self_score()
    print("ok")
