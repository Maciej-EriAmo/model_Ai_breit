"""
Kanoniczna konfiguracja PhiMost — zsynchronizowana z wynikami A–E + lab Most.

NIE zmieniać bez nowego bencha Softmax|Most|ZERO.
Tracer (n_dim) = WARUNEK Mostu (DeltaE), nie handicap vs Softmax.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhiMostCanon:
    # wygrywający most (lab + Faza B)
    coupling: str = "softplus"
    num_layers: int = 2
    d_model: int = 32
    n_heads: int = 2
    d_x: int = 8
    # n_dim=6: depth OH(3) + T + state_scalar + mazur  (patrz phi_schema.atom_tracer_vec)
    n_dim: int = 6
    n_classes: int = 4
    # trening lab
    lr: float = 1e-3
    epochs_classify: int = 6
    epochs_recall: int = 8
    # bramki sukcesu
    gate_classify_delta: float = 0.03  # Most−SM na ≥2 n
    gate_recall_r1: float = 0.02  # Most−SM r@1
    # zakazy
    allow_ema_superposition: bool = False
    # Faza E — product OFF
    default_live: bool = False
    default_gen: bool = False
    default_neural_lambda: bool = False


CANON = PhiMostCanon()


def assert_canon_model_kwargs(coupling: str, num_layers: int) -> None:
    if CANON.allow_ema_superposition:
        raise RuntimeError("EMA-S zabronione w kanonie")
    if coupling != CANON.coupling:
        raise ValueError(f"coupling={coupling!r} ≠ canon {CANON.coupling!r}")
    if num_layers < CANON.num_layers:
        raise ValueError(f"num_layers={num_layers} < canon {CANON.num_layers}")
