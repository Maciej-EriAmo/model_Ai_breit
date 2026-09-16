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

    if tracer.shape[-1] <= t_dim:
        # brak kanału T w tym tracerze (np. n_dim<=3, tylko depth) — nic do uszczelnienia
        return A

    T = tracer[..., t_dim]  # [B, N]
    T_key = T.unsqueeze(1).unsqueeze(2)  # [B,1,1,N]

    # Ablacja zero_tracer (RecallBatch.ablate / bridge.batch_to_xy) zeruje CAŁY
    # tracer, więc T=0 wszędzie — bez tego guarda hard floor (T<t_floor)
    # maskowałby TO SAMO co "archive/TOMB" na 100% kluczy, zamieniając test
    # degeneracji Mostu (ΔE=0 → A=C/γ, wciąż pełna macierz) w martwą uwagę.
    # Prawdziwy atom (nawet COLD) zawsze ma niezerowy depth one-hot; ablowany
    # ma cały wektor = 0 — to odróżnia "brak τ do testu" od "realnie zimny".
    real_atom = tracer.abs().sum(dim=-1) > 0  # [B, N]; CLS też 0, ale i tak bypass niżej
    row_is_ablated = (~real_atom[:, 1:]).all(dim=-1) if tracer.shape[1] > 1 else torch.zeros(
        tracer.shape[0], dtype=torch.bool, device=tracer.device
    )
    if bool(row_is_ablated.all()):
        return A
    ablated = row_is_ablated.view(-1, 1, 1, 1)

    # kara COLD (ciągła)
    penalty = torch.exp(-exp_k * F.relu(t_cold - T_key))
    penalty = penalty.clone()
    penalty[..., 0] = 1.0  # CLS
    penalty = torch.where(ablated, torch.ones_like(penalty), penalty)
    A = A * penalty

    # hard mask TOMB / bardzo zimne
    cold_hard = T_key < t_floor
    cold_hard = cold_hard.clone()
    cold_hard[..., 0] = False
    cold_hard = cold_hard & ~ablated
    A = A.masked_fill(cold_hard, 0.0)
    return A
