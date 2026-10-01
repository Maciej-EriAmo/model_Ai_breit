"""Most-1 Breita–Wignera.

Szerokość jednego wiersza:

    gamma = gamma0 * (1 + beta * kappa),  kappa = max(C / sum(C))

    A = gamma * C / (gamma^2 + delta_E^2)

Znak delta_E nie wchodzi do wagi. C ma być już wyzerowane na zabronionych kluczach,
żeby padding i maska nie ustawiały kappa.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def breit_wigner_rows(
    coupling: torch.Tensor,
    delta_e: torch.Tensor,
    log_gamma0: torch.Tensor,
    beta_raw: torch.Tensor,
) -> torch.Tensor:
    """Surowe A, ten sam kształt co coupling. Ostatnia oś to klucze."""
    if coupling.shape != delta_e.shape:
        raise ValueError("C and delta_E must share a shape")
    mass = coupling.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    state0 = coupling / mass
    kappa = state0.amax(dim=-1, keepdim=True)
    gamma0 = F.softplus(log_gamma0) + 1e-3
    beta = F.softplus(beta_raw)
    gamma = gamma0 * (1.0 + beta * kappa)
    return gamma * coupling / (gamma.square() + delta_e.square() + 1e-12)
