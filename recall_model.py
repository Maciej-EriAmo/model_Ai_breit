"""
Encoder Most/Softmax + InfoNCE do recall nad Φ.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from breit_kernel import breit_wigner_rows
from most_seal import apply_t_key_seal
from recall_data import RecallBatch


def _key_padding(
    mask: torch.Tensor | None,
    B: int,
    N0: int,
    device: torch.device,
) -> torch.Tensor | None:
    """mask [B,N0] 1=valid → bool [B,1,1,1+N0] True=pad (CLS always valid)."""
    if mask is None:
        return None
    m = mask.to(device=device)
    if m.dim() != 2 or m.shape[0] != B or m.shape[1] != N0:
        raise ValueError(f"mask shape {tuple(m.shape)} ≠ [{B},{N0}]")
    cls = torch.ones(B, 1, device=device, dtype=m.dtype)
    full = torch.cat([cls, m], dim=1)  # [B, 1+N0]
    return (full < 0.5).view(B, 1, 1, 1 + N0)


class PhiBridgeEncoder(nn.Module):
    """Most-1 Breita–Wignera. Zwraca embeddingi tokenów [B,N,d] (bez CLS)."""

    def __init__(
        self,
        input_dim: int = 8,
        d: int = 32,
        h: int = 2,
        dim_tracer: int = 6,
        num_layers: int = 2,
        coupling: str = "softplus",
        t_seal: bool | None = None,
        *,
        gamma_init: float = 0.0,
        e_scale: float = 1.0,
        couple_temp: float = 1.0,
        delta_mode: str = "square",
        delta_alpha: float = 1.0,
        directed_bias: float = 0.25,
    ):
        super().__init__()
        if delta_mode != "square":
            raise ValueError(
                "ten fork ma jądro Breita–Wignera: delta_mode musi być 'square'"
            )
        self.h, self.dh = h, d // h
        self.d = d
        self.coupling = coupling
        self.t_seal = t_seal
        self.e_scale = float(e_scale)
        self.couple_temp = max(float(couple_temp), 1e-3)
        self.delta_mode = delta_mode
        self.delta_alpha = float(delta_alpha)
        self.directed_bias = float(min(max(directed_bias, 0.0), 1.0))
        self.embed = nn.Linear(input_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv": nn.Linear(d, 3 * d, bias=False),
                        "o": nn.Linear(d, d, bias=False),
                        "e": nn.Linear(dim_tracer, h),
                        "n1": nn.LayerNorm(d),
                        "ff": nn.Sequential(
                            nn.LayerNorm(d),
                            nn.Linear(d, 4 * d),
                            nn.GELU(),
                            nn.Linear(4 * d, d),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        for layer in self.layers:
            nn.init.ones_(layer["e"].weight)
            nn.init.zeros_(layer["e"].bias)
        self.log_gammas = nn.ParameterList(
            [
                nn.Parameter(torch.full((h, 1, 1), float(gamma_init)))
                for _ in range(num_layers)
            ]
        )
        self.beta_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(h, 1, 1)) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d)

    def _couple(self, S: torch.Tensor) -> torch.Tensor:
        S = S / self.couple_temp
        if self.coupling == "softplus":
            return F.softplus(S).square()
        if self.coupling == "relu":
            return F.relu(S).square()
        raise ValueError(self.coupling)

    def forward(
        self,
        x_raw: torch.Tensor,
        tracer: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        B, N0, _ = x_raw.shape
        x = self.embed(x_raw)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        tracer = torch.cat(
            [torch.zeros(B, 1, tracer.shape[-1], device=x.device), tracer], dim=1
        )
        N = x.shape[1]
        dE_last = None
        key_pad = _key_padding(mask, B, N0, x.device)  # [B,1,1,N] True=pad

        for li, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
            q, k, v = [
                t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
            ]
            S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
            E = layer["e"](tracer).transpose(1, 2).unsqueeze(-1)
            dE = self.e_scale * (E - E.transpose(-2, -1))
            C = self._couple(S)
            C = C.masked_fill(torch.eye(N, device=x.device, dtype=torch.bool), 0.0)
            C = apply_t_key_seal(C, tracer, enabled=self.t_seal)
            if key_pad is not None:
                C = C.masked_fill(key_pad, 0.0)
            A = breit_wigner_rows(C, dE, self.log_gammas[li], self.beta_raw[li])
            w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
            from precision import assert_finite

            assert_finite(S, f"enc.S[L{li}]")
            assert_finite(dE, f"enc.dE[L{li}]")
            assert_finite(A, f"enc.A[L{li}]")
            assert_finite(w, f"enc.w[L{li}]")
            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
            dE_last = dE.detach()

        tok = self.out_norm(x[:, 1:, :])  # drop CLS → [B,N0,d]
        from precision import assert_finite

        assert_finite(tok, "enc.tok")
        return tok, dE_last


class PhiSoftmaxEncoder(nn.Module):
    """Twin Softmax — tracer ignorowany."""

    def __init__(
        self,
        input_dim: int = 8,
        d: int = 32,
        h: int = 2,
        dim_tracer: int = 6,
        num_layers: int = 2,
    ):
        super().__init__()
        self.h, self.dh = h, d // h
        self.d = d
        self.embed = nn.Linear(input_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "qkv": nn.Linear(d, 3 * d, bias=False),
                        "o": nn.Linear(d, d, bias=False),
                        "n1": nn.LayerNorm(d),
                        "ff": nn.Sequential(
                            nn.LayerNorm(d),
                            nn.Linear(d, 4 * d),
                            nn.GELU(),
                            nn.Linear(4 * d, d),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d)

    def forward(
        self,
        x_raw: torch.Tensor,
        tracer: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        B, N0, _ = x_raw.shape
        x = self.embed(x_raw)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        N = x.shape[1]
        key_pad = _key_padding(mask, B, N0, x.device)
        for layer in self.layers:
            q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
            q, k, v = [
                t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
            ]
            S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
            eye = torch.eye(N, device=x.device, dtype=torch.bool)
            if key_pad is not None:
                S = S.masked_fill(key_pad, -1e9)
            w = F.softmax(S, dim=-1)
            w = w.masked_fill(eye, 0.0)
            if key_pad is not None:
                w = w.masked_fill(key_pad, 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
        tok = self.out_norm(x[:, 1:, :])
        return tok, None


def gather_idx(tok: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """tok [B,N,d], idx [B] → [B,d]"""
    B, N, d = tok.shape
    return tok[torch.arange(B, device=tok.device), idx]


def _apply_cand_mask(
    scores: torch.Tensor,
    query_idx: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Wyzeruj self-query i paddowane sloty (mask<0.5)."""
    B, N = scores.shape
    ar = torch.arange(N, device=scores.device).view(1, N).expand(B, N)
    scores = scores.masked_fill(ar == query_idx.view(B, 1), -1e9)
    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError(f"mask {tuple(mask.shape)} ≠ scores {tuple(scores.shape)}")
        scores = scores.masked_fill(mask.to(scores.device) < 0.5, -1e9)
    return scores


def info_nce_loss(
    tok: torch.Tensor,
    query_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    temperature: float = 0.1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Score = cos(q, cand) / temp; pozytyw = pos_idx; kandydaci = wszyscy ≠ query.
    `mask` [B,N]: 1=atom obecny — padding nie wchodzi do Softmax CE.
    """
    B, N, d = tok.shape
    q = F.normalize(gather_idx(tok, query_idx), dim=-1)  # [B,d]
    all_h = F.normalize(tok, dim=-1)  # [B,N,d]
    scores = torch.einsum("bd,bnd->bn", q, all_h) / temperature  # [B,N]
    scores = _apply_cand_mask(scores, query_idx, mask)
    return F.cross_entropy(scores, pos_idx)


@torch.no_grad()
def recall_metrics(
    tok: torch.Tensor,
    query_idx: torch.Tensor,
    pos_idx: torch.Tensor,
    ks: tuple[int, ...] = (1, 3, 5),
    mask: torch.Tensor | None = None,
) -> dict:
    B, N, d = tok.shape
    q = F.normalize(gather_idx(tok, query_idx), dim=-1)
    all_h = F.normalize(tok, dim=-1)
    scores = torch.einsum("bd,bnd->bn", q, all_h)
    scores = _apply_cand_mask(scores, query_idx, mask)

    # rank: higher score better
    order = scores.argsort(dim=-1, descending=True)  # [B,N]
    # position of positive (0-based)
    pos_exp = pos_idx.view(B, 1).expand_as(order)
    hits = (order == pos_exp).float()
    ranks = hits.argmax(dim=-1)  # first True index
    # if somehow missing, rank = N
    has = hits.max(dim=-1).values
    ranks = torch.where(has > 0, ranks, torch.full_like(ranks, N))

    out = {"mrr": (1.0 / (ranks.float() + 1.0)).mean().item()}
    for k in ks:
        out[f"r@{k}"] = (ranks < k).float().mean().item()
    return out


def _avg_val_metrics(
    model: nn.Module,
    val_batches: list[RecallBatch],
    ablate_kw: dict,
) -> dict:
    metrics = []
    dE_vals = []
    model.eval()
    with torch.no_grad():
        for batch in val_batches:
            b = batch.ablate(**ablate_kw)
            tok, dE = model(b.content, b.tracer, b.mask)
            metrics.append(
                recall_metrics(tok, b.query_idx, b.pos_idx, mask=b.mask)
            )
            if dE is not None:
                dE_vals.append(dE.abs().mean().item())
    keys = metrics[0].keys()
    avg = {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}
    avg["mean_dE"] = float(sum(dE_vals) / len(dE_vals)) if dE_vals else 0.0
    return avg


def train_encoder(
    model: nn.Module,
    train_batches: list[RecallBatch],
    val_batches: list[RecallBatch],
    epochs: int = 8,
    lr: float = 1e-3,
    ablate_kw: dict | None = None,
    *,
    return_curve: bool = False,
    amp_dtype: torch.dtype | None = None,
    finite_check: bool | None = None,
) -> dict:
    """
    return_curve=True → dodaje klucz ``curve``: lista
    {epoch, loss, r@1, r@3, mrr} po każdej epoce
    (loss = średnia strata treningowa danej epoki; r@1/r@3/mrr = walidacyjne).

    amp_dtype=torch.bfloat16 → kryterium #2 (BF16) przez autocast.
    finite_check=True (domyślnie gdy amp_dtype ustawione) → assert na zamku/loss.
    """
    from precision import (
        amp_context,
        assert_finite_loss,
        set_finite_check,
    )

    ablate_kw = ablate_kw or {}
    if finite_check is None:
        finite_check = amp_dtype is not None
    prev = None
    try:
        from precision import finite_check_enabled

        prev = finite_check_enabled()
        set_finite_check(bool(finite_check))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        curve: list[dict] = []
        last_avg: dict | None = None
        for ep in range(epochs):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for batch in train_batches:
                b = batch.ablate(**ablate_kw)
                opt.zero_grad()
                with amp_context(amp_dtype):
                    tok, _ = model(b.content, b.tracer, b.mask)
                    loss = info_nce_loss(tok, b.query_idx, b.pos_idx, mask=b.mask)
                    assert_finite_loss(loss, step=f"train_ep{ep+1}")
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach().float().item())
                n_batches += 1
            if return_curve:
                with amp_context(amp_dtype):
                    last_avg = _avg_val_metrics(model, val_batches, ablate_kw)
                curve.append(
                    {
                        "epoch": ep + 1,
                        "loss": loss_sum / max(n_batches, 1),
                        "r@1": last_avg["r@1"],
                        "r@3": last_avg["r@3"],
                        "mrr": last_avg["mrr"],
                    }
                )

        if last_avg is not None:
            avg = last_avg
        else:
            with amp_context(amp_dtype):
                avg = _avg_val_metrics(model, val_batches, ablate_kw)
        if return_curve:
            avg["curve"] = curve
        if amp_dtype is not None:
            avg["amp_dtype"] = str(amp_dtype).replace("torch.", "")
        return avg
    finally:
        if prev is not None:
            set_finite_check(prev)
        else:
            set_finite_check(False)
