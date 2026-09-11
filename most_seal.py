"""
Uszczelnienie Mostu: bramka T na kluczach uwagi.

Hard mask: T < t_attn_floor → waga 0 (TOMB / archive).
Kara COLD: T < t_attn_cold → A *= exp(-k * (cold - T)).
CLS (pozycja 0) zawsze otwarty — ma τ=0 po padzie.
Softmax twin NIE używa tej bramki.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import CANON


def apply_t_key_seal(
    A: torch.Tensor,
    tracer: torch.Tensor,
    *,
    t_dim: int = 3,
    t_floor: float | None = None,
    t_cold: float | None = None,
    exp_k: float | None = None,
    enabled: bool | None = None,
) -> torch.Tensor:
    """
    A: [B, h, N, N] wagi przed normalizacją.
    tracer: [B, N, n_dim] (z CLS na indeksie 0).
    """
    if enabled is None:
        enabled = bool(CANON.t_attn_seal)
    if not enabled:
        return A
    if t_floor is None:
        t_floor = float(CANON.t_attn_floor)
    if t_cold is None:
        t_cold = float(CANON.t_attn_cold)
    if exp_k is None:
        exp_k = float(CANON.t_attn_exp_k)

    T = tracer[..., t_dim]  # [B, N]
    T_key = T.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]

    # kara COLD (ciągła)
    penalty = torch.exp(-exp_k * F.relu(t_cold - T_key))
    penalty = penalty.clone()
    penalty[..., 0] = 1.0  # CLS
    A = A * penalty

    # hard mask TOMB / bardzo zimne
    cold_hard = T_key < t_floor
    cold_hard = cold_hard.clone()
    cold_hard[..., 0] = False
    A = A.masked_fill(cold_hard, 0.0)
    return A
