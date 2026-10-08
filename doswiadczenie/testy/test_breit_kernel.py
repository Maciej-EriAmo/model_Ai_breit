"""Most-1: szerokość rośnie ze szczytem C, znak delta_E nic nie zmienia."""

import _labroot  # noqa: F401

import torch
import torch.nn.functional as F

from breit_kernel import breit_wigner_rows, breit_wigner_weights
from bridge import BridgeTransformer, SoftmaxTransformer, masked_softmax
from most_seal import apply_t_key_seal
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


def test_beta_zero_ignores_the_row_and_the_raw_parameter():
    coupling = torch.tensor([[[[5.0, 0.2, 0.2]]]], dtype=torch.float64)
    delta = torch.tensor([[[[0.0, 1.0, -2.0]]]], dtype=torch.float64)
    log_gamma0 = torch.zeros(1, 1, 1, dtype=torch.float64)
    beta_raw = torch.full((1, 1, 1), 4.0, dtype=torch.float64)
    got = breit_wigner_rows(
        coupling, delta, log_gamma0, beta_raw, beta_exact=0.0
    )
    gamma0 = F.softplus(log_gamma0) + 1e-3
    raw = gamma0 * coupling / (gamma0.square() + delta.square() + 1e-12)
    assert torch.allclose(got, raw)


def test_excess_scale_starts_flat_rows_at_gamma0():
    log_gamma0 = torch.zeros(1, 1, 1, dtype=torch.float64)
    beta_raw = torch.zeros(1, 1, 1, dtype=torch.float64)
    gamma0 = F.softplus(log_gamma0) + 1e-3

    def amplitude_over_c(n: int, scale: str) -> torch.Tensor:
        coupling = torch.ones(1, 1, 1, n, dtype=torch.float64)
        delta = torch.zeros_like(coupling)
        got = breit_wigner_rows(
            coupling, delta, log_gamma0, beta_raw, scale=scale
        )
        return got / coupling

    short = amplitude_over_c(2, "excess")
    long = amplitude_over_c(8, "excess")
    assert torch.allclose(short, 1.0 / gamma0)
    assert torch.allclose(long, 1.0 / gamma0)
    historical_short = float(amplitude_over_c(2, "kappa").reshape(-1)[0])
    historical_long = float(amplitude_over_c(8, "kappa").reshape(-1)[0])
    assert abs(historical_short - historical_long) > 1e-4


def test_width_arms_build_and_stay_finite():
    cfg = TinyLMConfig(d=32, h=2, num_layers=1, max_seq=16)
    beta0 = TinyLM(cfg, attn="beta0")
    excess = TinyLM(cfg, attn="excess")
    historical = TinyLM(cfg, attn="most")
    assert beta0.arm == "beta0" and beta0.attn == "most"
    assert beta0.blocks[0].beta_exact == 0.0
    assert beta0.blocks[0].beta_raw.requires_grad is False
    assert excess.blocks[0].width_scale == "excess"
    assert excess.blocks[0].beta_raw.requires_grad is True
    assert historical.blocks[0].width_scale == "kappa"
    assert historical.blocks[0].beta_exact is None
    idx = torch.randint(0, 128, (2, 8))
    assert torch.isfinite(beta0(idx)).all()
    assert torch.isfinite(excess(idx)).all()


def test_weights_equal_normalised_rows():
    torch.manual_seed(0)
    coupling = F.softplus(torch.randn(3, 2, 5, 7, dtype=torch.float64)).square()
    coupling[0, 0, 1] = 0.0  # wiersz bez kluczy zostaje zerem
    delta = torch.randn(3, 2, 5, 7, dtype=torch.float64)
    log_gamma0 = torch.randn(2, 1, 1, dtype=torch.float64)
    beta_raw = torch.randn(2, 1, 1, dtype=torch.float64)
    A = breit_wigner_rows(coupling, delta, log_gamma0, beta_raw)
    ref = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
    got = breit_wigner_weights(coupling, delta, log_gamma0, beta_raw)
    assert torch.allclose(got, ref, atol=1e-12)
    assert torch.all(got[0, 0, 1] == 0)


def _bridge_reference(model, x_raw, tracer, mask):
    """Dawna ścieżka forward: maski i seal w każdej warstwie, potem A / sum A."""
    B, N0, _ = x_raw.shape
    x = torch.cat([model.cls.expand(B, 1, -1), model.embed(x_raw)], dim=1)
    tracer = torch.cat([torch.zeros(B, 1, tracer.shape[-1]), tracer], dim=1)
    N = x.shape[1]
    full = torch.cat([torch.ones(B, 1), mask], dim=1)
    key_pad = (full < 0.5).view(B, 1, 1, N)
    for li, layer in enumerate(model.layers):
        q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, model.h, model.dh).transpose(1, 2) for t in (q, k, v)]
        S = q @ k.transpose(-2, -1) / model.dh**0.5
        E = layer["e"](tracer).transpose(1, 2).unsqueeze(-1)
        dE = E - E.transpose(-2, -1)
        C = model._couple(S).masked_fill(torch.eye(N, dtype=torch.bool), 0.0)
        C = apply_t_key_seal(C, tracer, enabled=model.t_seal)
        C = C.masked_fill(key_pad, 0.0)
        A = breit_wigner_rows(C, dE, model.log_gammas[li], model.beta_raw[li])
        w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
        y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
        x = x + layer["o"](y)
        x = x + layer["ff"](x)
    return model.head(x[:, 0]), w


def test_bridge_matches_per_layer_masking():
    torch.manual_seed(1)
    B, N, dx, nd = 3, 9, 8, 6
    x = torch.randn(B, N, dx)
    tracer = torch.rand(B, N, nd)
    tracer[:, 2, 3] = 0.05  # TOMB — twarde odcięcie
    tracer[:, 4, 3] = 0.30  # COLD — kara
    mask = torch.ones(B, N)
    mask[1, 6:] = 0.0
    model = BridgeTransformer(input_dim=dx, dim_tracer=nd, n_classes=3, t_seal=True)
    for p in model.parameters():
        p.data.add_(0.1 * torch.randn_like(p))
    logits, w, _ = model(x, tracer, mask)
    ref_logits, ref_w = _bridge_reference(model, x, tracer, mask)
    assert torch.allclose(logits, ref_logits, atol=1e-5)
    assert torch.allclose(w, ref_w, atol=1e-6)
    logits.sum().backward()
    g = model.log_gammas[0].grad.clone()
    model.zero_grad()
    ref_logits.sum().backward()
    assert torch.allclose(g, model.log_gammas[0].grad, atol=1e-5)
