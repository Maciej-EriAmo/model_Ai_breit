"""
Szew λ_neural: Λ' = λ_R·R + λ_M·M + λ_N·N

N = neural cosine score z encodera Most (checkpoint lub set_encoder).
Flaga PHIMOST_NEURAL_LAMBDA domyślnie OFF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from checkpoint import DEFAULT_MOST, load_encoder
from phimost_flags import lambda_neural_weight, neural_lambda_enabled
from phi_schema import PhiAtom, PhiGraph, PhiState, atom_content_vec, atom_tracer_vec
from recall_model import PhiBridgeEncoder


def blend_lambda(
    R: float,
    M: float,
    N: float,
    *,
    lambda_R: float = 0.2,
    lambda_M: float = 0.8,
    lambda_N: Optional[float] = None,
) -> float:
    """Gdy neural wyłączony → klasyczne Λ = λ_R R + λ_M M."""
    if not neural_lambda_enabled():
        return float(lambda_R) * float(R) + float(lambda_M) * float(M)
    ln = lambda_neural_weight() if lambda_N is None else float(lambda_N)
    s = float(lambda_R) + float(lambda_M) + ln
    if s <= 1e-12:
        return float(R)
    return (float(lambda_R) * float(R) + float(lambda_M) * float(M) + ln * float(N)) / s


def graph_tensors(g: PhiGraph, d_x: int = 8, n_dim: int = 6):
    N = len(g.atoms)
    content = torch.zeros(1, N, d_x)
    tracer = torch.zeros(1, N, n_dim)
    t_max = max((a.T for a in g.atoms), default=1.0) or 1.0
    for i, a in enumerate(g.atoms):
        aa = PhiAtom(
            atom_id=a.atom_id,
            S=a.S,
            E=a.E,
            T=a.T / t_max,
            state=a.state,
            bubble_id=a.bubble_id,
            depth=a.depth,
            payload=a.payload,
            mazur_tracer=a.mazur_tracer,
        )
        content[0, i] = atom_content_vec(aa, d_x)
        tracer[0, i] = atom_tracer_vec(aa, n_dim)
    return content, tracer


def _align_tracer(encoder: PhiBridgeEncoder, tracer: torch.Tensor) -> torch.Tensor:
    n_model = encoder.layers[0]["e"].in_features
    if tracer.shape[-1] == n_model:
        return tracer
    if tracer.shape[-1] > n_model:
        return tracer[..., :n_model]
    pad = torch.zeros(*tracer.shape[:-1], n_model - tracer.shape[-1])
    return torch.cat([tracer, pad], dim=-1)


@torch.no_grad()
def neural_scores(
    encoder: PhiBridgeEncoder,
    g: PhiGraph,
    query_idx: int,
    d_x: int = 8,
    n_dim: int = 6,
) -> list[float]:
    """Cosine(q, cand) z Most — N∈[0,1] per atom."""
    if len(g.atoms) == 0:
        return []
    content, tracer = graph_tensors(g, d_x=d_x, n_dim=n_dim)
    tracer = _align_tracer(encoder, tracer)
    mask = torch.ones(1, content.shape[1])
    tok, _ = encoder(content, tracer, mask)
    q = F.normalize(tok[0, query_idx], dim=-1)
    h = F.normalize(tok[0], dim=-1)
    sims = (h @ q).clamp(0, 1).tolist()
    return [float(x) for x in sims]


def rank_with_neural(
    encoder: PhiBridgeEncoder,
    g: PhiGraph,
    query_idx: int,
    scalar_R: list[float],
    memory_M: Optional[list[float]] = None,
    *,
    lambda_R: float = 0.2,
    lambda_M: float = 0.8,
) -> list[tuple[float, int]]:
    """
    Zwraca listę (score, idx) malejąco.
    Bez flagi = λ_R R + λ_M M (N ignorowane).
    """
    N = len(g.atoms)
    M = memory_M if memory_M is not None else [0.0] * N
    if neural_lambda_enabled():
        Ns = neural_scores(encoder, g, query_idx)
    else:
        Ns = [0.0] * N
    scored = []
    for i in range(N):
        if i == query_idx:
            continue
        sc = blend_lambda(
            scalar_R[i], M[i], Ns[i], lambda_R=lambda_R, lambda_M=lambda_M
        )
        scored.append((sc, i))
    scored.sort(key=lambda x: -x[0])
    return scored


# --- API dla mazur_crystal (import po ścieżce) ---

_ENCODER: Optional[PhiBridgeEncoder] = None
_CKPT_TRIED: bool = False


def set_encoder(enc: Optional[PhiBridgeEncoder]) -> None:
    """
    Ustaw encoder ręcznie.
    `set_encoder(None)` = jawne wyłączenie (bez autoload ckpt), aż do `reset_encoder()`.
    """
    global _ENCODER, _CKPT_TRIED
    _ENCODER = enc
    if enc is None:
        _CKPT_TRIED = True  # nie ładuj ckpt po jawnym clear


def reset_encoder() -> None:
    """Wyczyść encoder i pozwól ponownie na autoload z checkpointu."""
    global _ENCODER, _CKPT_TRIED
    _ENCODER = None
    _CKPT_TRIED = False


def ensure_encoder(
    path: str | Path | None = None,
) -> Optional[PhiBridgeEncoder]:
    """
    Zwraca encoder Most gdy flaga ON.
    Kolejność: set_encoder → load checkpoints/most_dump_recall.pt → None.
    """
    global _ENCODER, _CKPT_TRIED
    if not neural_lambda_enabled():
        return None
    if _ENCODER is not None:
        return _ENCODER
    if _CKPT_TRIED:
        return None
    _CKPT_TRIED = True
    ckpt = Path(path) if path else DEFAULT_MOST
    if not ckpt.is_file():
        return None
    try:
        enc, _meta = load_encoder(ckpt)
        _ENCODER = enc  # type: ignore[assignment]
        return _ENCODER
    except Exception:
        return None


def _coerce_phi_atom(obj: Any, default_id: str) -> PhiAtom:
    if isinstance(obj, PhiAtom):
        return obj
    st = getattr(obj, "state", PhiState.HOT)
    if not isinstance(st, PhiState):
        try:
            st = PhiState[str(st).upper()]
        except Exception:
            st = PhiState.HOT
    return PhiAtom(
        atom_id=str(getattr(obj, "id", default_id)),
        S=str(getattr(obj, "S", "") or ""),
        E=str(getattr(obj, "E", "") or ""),
        T=float(getattr(obj, "T", 1.0) or 1.0),
        state=st,
        bubble_id=str(getattr(obj, "bubble_id", "live") or "live"),
        depth=int(getattr(obj, "depth", 0) or 0),
        payload=getattr(obj, "payload", None),
        mazur_tracer=(
            float(getattr(obj, "mazur_tracer"))
            if getattr(obj, "mazur_tracer", None) is not None
            else None
        ),
    )


def neural_component_for_pair(atom: Any, probe: Any) -> Optional[float]:
    """
    Wołane z mazur hooka. None = brak neural (użyj czystego Λ).
    Gdy encoder ustawiony → Most cosine(probe, atom); inaczej None.
    """
    enc = ensure_encoder()
    if enc is None:
        return None
    try:
        a = _coerce_phi_atom(atom, "atom")
        p = _coerce_phi_atom(probe, "probe")
        g = PhiGraph(atoms=[a, p], edges=[])
        # query = probe (idx 1), score for atom (idx 0)
        scores = neural_scores(enc, g, query_idx=1)
        return float(scores[0]) if scores else None
    except Exception:
        return None
