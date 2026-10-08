"""Most-1 Breita–Wignera.

Szerokość jednego wiersza:

    gamma = gamma0 * (1 + beta * conc)

    conc = kappa                         skala historyczna "kappa"
         = 0 na płaskim wierszu          skala "excess"

    kappa = max(C) / sum(C)
    A = gamma * C / (gamma^2 + delta_E^2)

beta_exact=0 wyłącza zależność od wiersza: gamma = gamma0.
Domyślne beta = softplus(beta_raw); beta_raw=0 daje beta ≈ ln 2, nie zero.

Znak delta_E nie wchodzi do wagi. C ma być już wyzerowane na zabronionych kluczach,
żeby padding i maska nie ustawiały kappa.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def breit_wigner_rows(
    coupling: torch.Tensor,
    delta_e: torch.Tensor,
    log_gamma0: torch.Tensor,
    beta_raw: torch.Tensor,
    *,
    beta_exact: float | None = None,
    scale: str = "kappa",
) -> torch.Tensor:
    """Surowe A, ten sam kształt co coupling. Ostatnia oś to klucze."""
    if coupling.shape != delta_e.shape:
        raise ValueError("C and delta_E must share a shape")
    gamma = _row_gamma(
        coupling, log_gamma0, beta_raw, beta_exact=beta_exact, scale=scale
    )
    return gamma * coupling / (gamma.square() + delta_e.square() + 1e-12)


def breit_wigner_weights(
    coupling: torch.Tensor,
    delta_e: torch.Tensor,
    log_gamma0: torch.Tensor,
    beta_raw: torch.Tensor,
    *,
    beta_exact: float | None = None,
    scale: str = "kappa",
) -> torch.Tensor:
    """A znormalizowane po kluczach. Wiersz bez żadnego klucza zostaje zerem.

    gamma w liczniku jest stała w wierszu i skraca się przy normalizacji,
    więc tu go nie ma.
    """
    if coupling.shape != delta_e.shape:
        raise ValueError("C and delta_E must share a shape")
    gamma = _row_gamma(
        coupling, log_gamma0, beta_raw, beta_exact=beta_exact, scale=scale
    )
    A = coupling / (gamma.square() + delta_e.square() + 1e-12)
    return A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _row_gamma(
    coupling: torch.Tensor,
    log_gamma0: torch.Tensor,
    beta_raw: torch.Tensor,
    *,
    beta_exact: float | None = None,
    scale: str = "kappa",
) -> torch.Tensor:
    # max(C / sum C) == max(C) / sum C, bez dzielenia całej macierzy
    mass = coupling.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    kappa = coupling.amax(dim=-1, keepdim=True) / mass
    conc = _concentration(kappa, coupling, scale)
    gamma0 = F.softplus(log_gamma0) + 1e-3
    if beta_exact is None:
        beta = F.softplus(beta_raw)
    else:
        beta_value = float(beta_exact)
        if not math.isfinite(beta_value) or beta_value < 0.0:
            raise ValueError("beta_exact must be >= 0")
        beta = beta_value
    return gamma0 * (1.0 + beta * conc)


def _concentration(
    kappa: torch.Tensor,
    coupling: torch.Tensor,
    scale: str,
) -> torch.Tensor:
    """kappa: historyczna skala, płaski wiersz ma 1/n.

    excess: 0 na płaskim wierszu i na jednym żywym kluczu, 1 na szczycie.
    Dzięki temu γ startuje od γ0 niezależnie od długości wiersza.
    """
    if scale == "kappa":
        return kappa
    if scale != "excess":
        raise ValueError("scale must be 'kappa' or 'excess'")
    n = (coupling > 0).sum(dim=-1, keepdim=True).to(dtype=coupling.dtype)
    alpha = (kappa * n - 1.0) / (n - 1.0).clamp_min(1.0)
    alpha = torch.where(n > 1, alpha, torch.zeros_like(alpha))
    return alpha.clamp_min(0.0)
