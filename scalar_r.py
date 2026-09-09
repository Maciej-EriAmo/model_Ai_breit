"""
Skalarne R Mazura (baseline produktowy) — bez zależności od KarmazynOs.

R = g · V² / (g² + dE²)
V = cosine(content_q, content_c)
dE = |depth_q - depth_c|  (geometria Φ; opcjonalnie L2 tracera)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recall_data import RecallBatch
from recall_model import recall_metrics


def lorentz_R(V: torch.Tensor, dE: torch.Tensor, g: float = 0.3) -> torch.Tensor:
    g = max(float(g), 1e-9)
    return g * (V ** 2) / (g * g + dE ** 2 + 1e-12)


@torch.no_grad()
def score_scalar_R(
    batch: RecallBatch,
    g: float = 0.3,
    *,
    use_tracer_l2: bool = False,
    ignore_dE: bool = False,
) -> torch.Tensor:
    """
    Zwraca scores [B,N] dla każdego kandydata względem query.
    Query self → -inf.
    """
    B, N, d = batch.content.shape
    q_idx = batch.query_idx
    ar = torch.arange(B)

    q_x = batch.content[ar, q_idx]  # [B,d]
    q_x = F.normalize(q_x, dim=-1)
    c_x = F.normalize(batch.content, dim=-1)  # [B,N,d]
    V = torch.einsum("bd,bnd->bn", q_x, c_x).clamp(0.0, 1.0)

    if ignore_dE:
        dE = torch.zeros(B, N)
    elif use_tracer_l2:
        q_t = batch.tracer[ar, q_idx]  # [B,n]
        dE = (batch.tracer - q_t.unsqueeze(1)).pow(2).sum(-1).sqrt()
    else:
        q_d = batch.depths[ar, q_idx]  # [B]
        dE = (batch.depths - q_d.unsqueeze(1)).abs()

    scores = lorentz_R(V, dE, g=g)
    eye = torch.arange(N).view(1, N).expand(B, N) == q_idx.view(B, 1)
    scores = scores.masked_fill(eye, -1e9)
    if batch.mask is not None:
        scores = scores.masked_fill(batch.mask < 0.5, -1e9)
    return scores


@torch.no_grad()
def eval_scalar_R(
    batches: list[RecallBatch],
    g: float = 0.3,
    **score_kw,
) -> dict:
    metrics = []
    for batch in batches:
        scores = score_scalar_R(batch, g=g, **score_kw)
        # Direct rank from scores (padding już -inf w score_scalar_R):
        B, N = scores.shape
        order = scores.argsort(dim=-1, descending=True)
        pos_exp = batch.pos_idx.view(B, 1).expand_as(order)
        hits = (order == pos_exp).float()
        ranks = hits.argmax(dim=-1)
        has = hits.max(dim=-1).values
        ranks = torch.where(has > 0, ranks, torch.full_like(ranks, N))
        m = {"mrr": (1.0 / (ranks.float() + 1.0)).mean().item()}
        for k in (1, 3, 5):
            m[f"r@{k}"] = (ranks < k).float().mean().item()
        metrics.append(m)
    keys = metrics[0].keys()
    return {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}


@torch.no_grad()
def eval_cosine_only(batches: list[RecallBatch]) -> dict:
    """Ablacja: samo V (cosine content), bez dE — jak Softmax-ish ranking."""
    return eval_scalar_R(batches, ignore_dE=True)
