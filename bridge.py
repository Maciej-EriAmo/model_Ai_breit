"""
Fork jądra Breita–Wignera (Most-1) pod tym samym szkieletem PhiMost.

QK daje C = softplus². Tracer daje delta_E.
gamma = gamma0 * (1 + beta * max(C/sum C)).
A = gamma * C / (gamma² + delta_E²).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from breit_kernel import breit_wigner_rows
from most_seal import apply_t_key_seal
from phi_schema import PhiGraphBatch


class BridgeTransformer(nn.Module):
    """QK z x, jądro Breita–Wignera na tracerze n-D, values z treści."""

    def __init__(
        self,
        input_dim: int = 8,
        d: int = 32,
        h: int = 2,
        n_classes: int = 4,
        dim_tracer: int = 6,
        num_layers: int = 2,
        coupling: str = "softplus",
        t_seal: bool | None = None,
    ):
        super().__init__()
        self.h, self.dh = h, d // h
        self.dim_tracer = dim_tracer
        self.num_layers = num_layers
        self.coupling = coupling
        self.t_seal = t_seal
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
            [nn.Parameter(torch.zeros(h, 1, 1)) for _ in range(num_layers)]
        )
        self.beta_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(h, 1, 1)) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d, n_classes)

    def _couple(self, S: torch.Tensor) -> torch.Tensor:
        if self.coupling == "relu":
            return F.relu(S).square()
        if self.coupling == "softplus":
            return F.softplus(S).square()
        raise ValueError(self.coupling)

    def forward(
        self,
        x_raw: torch.Tensor,
        tracer: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """mask [B,N] 1=valid (przed CLS); pad nie bierze udziału w uwadze."""
        B, N0, _ = x_raw.shape
        x = self.embed(x_raw)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        tracer = torch.cat(
            [torch.zeros(B, 1, tracer.shape[-1], device=x.device), tracer], dim=1
        )
        N = x.shape[1]
        key_pad = None
        if mask is not None:
            m = mask.to(device=x.device)
            cls = torch.ones(B, 1, device=x.device, dtype=m.dtype)
            full = torch.cat([cls, m], dim=1)
            key_pad = (full < 0.5).view(B, 1, 1, N)
        dE_last = None
        w_last = None

        for li, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
            q, k, v = [
                t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
            ]
            S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)

            E = layer["e"](tracer).transpose(1, 2).unsqueeze(-1)
            dE = E - E.transpose(-2, -1)

            C = self._couple(S)
            C = C.masked_fill(torch.eye(N, device=x.device, dtype=torch.bool), 0.0)
            C = apply_t_key_seal(C, tracer, enabled=self.t_seal)
            if key_pad is not None:
                C = C.masked_fill(key_pad, 0.0)
            A = breit_wigner_rows(C, dE, self.log_gammas[li], self.beta_raw[li])
            w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)

            from precision import assert_finite

            assert_finite(S, f"bridge.S[L{li}]")
            assert_finite(C, f"bridge.C[L{li}]")
            assert_finite(dE, f"bridge.dE[L{li}]")
            assert_finite(A, f"bridge.A[L{li}]")
            assert_finite(w, f"bridge.w[L{li}]")

            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
            dE_last, w_last = dE.detach(), w

        logits = self.head(x[:, 0])
        from precision import assert_finite

        assert_finite(logits, "bridge.logits")
        return logits, w_last, dE_last


class SoftmaxTransformer(nn.Module):
    """Ten sam szkielet co Bridge, ale wagi = softmax(QK). Tracer ignorowany."""

    def __init__(
        self,
        input_dim: int = 8,
        d: int = 32,
        h: int = 2,
        n_classes: int = 4,
        dim_tracer: int = 6,
        num_layers: int = 2,
    ):
        super().__init__()
        self.h, self.dh = h, d // h
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
        self.head = nn.Linear(d, n_classes)

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
        key_pad = None
        if mask is not None:
            m = mask.to(device=x.device)
            cls = torch.ones(B, 1, device=x.device, dtype=m.dtype)
            full = torch.cat([cls, m], dim=1)
            key_pad = (full < 0.5).view(B, 1, 1, N)
        w_last = None

        for layer in self.layers:
            q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
            q, k, v = [
                t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)
            ]
            S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
            if key_pad is not None:
                S = S.masked_fill(key_pad, -1e9)
            eye = torch.eye(N, device=x.device, dtype=torch.bool)
            w = F.softmax(S, dim=-1)
            w = w.masked_fill(eye, 0.0)
            if key_pad is not None:
                w = w.masked_fill(key_pad, 0.0)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)

            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
            w_last = w

        return self.head(x[:, 0]), w_last, None


def batch_to_xy(batch: PhiGraphBatch, zero_tracer: bool = False):
    x = batch.content
    y = batch.labels
    if zero_tracer:
        tracer = torch.zeros_like(batch.tracer)
    else:
        tracer = batch.tracer
    return x, y, tracer


def train_eval(
    model: nn.Module,
    train_batches: list[PhiGraphBatch],
    val_batches: list[PhiGraphBatch],
    zero_tracer: bool = False,
    epochs: int = 6,
    lr: float = 1e-3,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in train_batches:
            x, y, tracer = batch_to_xy(batch, zero_tracer=zero_tracer)
            opt.zero_grad()
            logits, _, _ = model(x, tracer)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    dE_vals = []
    with torch.no_grad():
        for batch in val_batches:
            x, y, tracer = batch_to_xy(batch, zero_tracer=zero_tracer)
            logits, _, dE = model(x, tracer)
            loss = crit(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * x.shape[0]
            total_acc += (pred == y).float().sum().item()
            count += x.shape[0]
            if dE is not None:
                dE_vals.append(dE.abs().mean().item())
    return {
        "loss": total_loss / count,
        "acc": total_acc / count,
        "mean_dE": float(np.mean(dE_vals)) if dE_vals else 0.0,
    }
