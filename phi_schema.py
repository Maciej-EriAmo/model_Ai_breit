"""
Kontrakt danych Φ dla PhiMost.

Kanoniczne: atomy (S, E, T, state) + bańki + głębokość.
Atomstore / Store Karmazyn — opcjonalny adapter później, nie kanon.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import torch


class PhiState(IntEnum):
    HOT = 0
    WARM = 1
    COLD = 2
    TOMB = 3


@dataclass
class PhiAtom:
    """Jeden atom w przestrzeni — nie bajty pliku."""

    atom_id: str
    S: str
    E: str
    T: float
    state: PhiState = PhiState.HOT
    bubble_id: str = "root"
    depth: int = 0
    payload: str | None = None  # tylko HOT text; opcjonalne
    mazur_tracer: float | None = None


@dataclass
class PhiEdge:
    kind: str  # bind | parent | root | env_of
    src: str
    dst: str
    name: str = ""


@dataclass
class PhiGraph:
    """Jeden „nie-płaski plik” / wycinek przestrzeni."""

    atoms: list[PhiAtom]
    edges: list[PhiEdge] = field(default_factory=list)
    label: int | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.atoms)


@dataclass
class PhiGraphBatch:
    """Tensory gotowe pod Most: content + tracer (osie rozdzielone)."""

    content: torch.Tensor  # [B, N, d_x]
    tracer: torch.Tensor  # [B, N, n]
    labels: torch.Tensor  # [B]
    depths: torch.Tensor  # [B, N]
    states: torch.Tensor  # [B, N]
    mask: torch.Tensor  # [B, N] 1=atom obecny

    @property
    def batch_size(self) -> int:
        return int(self.content.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.content.shape[1])


# --- Mapowanie osi → kanały ---

STATE_DIM = 4  # one-hot PhiState
MAX_DEPTH_BINS = 3  # jak warstwy drzewa w labie Most


def hash_embed(text: str, dim: int, seed: int = 0) -> torch.Tensor:
    """Stabilny pseudo-embedding (SHA256 → seed); niezależny od PYTHONHASHSEED."""
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    h = int.from_bytes(digest[:8], "little") % (2**31 - 1)
    g = torch.Generator()
    g.manual_seed(int(h))
    v = torch.randn(dim, generator=g)
    return v / v.norm().clamp_min(1e-6)


def atom_content_vec(atom: PhiAtom, d_x: int) -> torch.Tensor:
    """Content = S ⊕ E ⊕ (payload jeśli HOT). Nie wrzuca T do content."""
    half = max(d_x // 2, 1)
    s = hash_embed(atom.S, half, seed=1)
    e = hash_embed(atom.E, d_x - half, seed=2)
    v = torch.cat([s, e], dim=0)
    if atom.payload and atom.state == PhiState.HOT and d_x >= 4:
        p = hash_embed(atom.payload, d_x, seed=3)
        v = 0.7 * v + 0.3 * p
        v = v / v.norm().clamp_min(1e-6)
    return v


def atom_tracer_vec(atom: PhiAtom, n_dim: int) -> torch.Tensor:
    """
    Tracer z osi Φ (obserwator) — packing zgodny z kanonem n_dim=6:

      n=1     — skalar depth
      0..2    — one-hot depth bins (warstwy)
      3       — T ∈ [0,1]
      4       — state jako skalar st/3  (HOT=0 … TOMB=1)  [n_dim≥5]
      5       — mazur_tracer ∈ [0,1]                     [n_dim≥6]
      6..9    — one-hot state (gdy n_dim≥10; legacy pełny)
      8/9     — mazur / PE gdy n_dim>8 (legacy lab)

    Przy n_dim=6 mazur NIE jest gubiony (wcześniej wymagało n_dim>8).
    """
    t = torch.zeros(n_dim)
    depth_bin = int(max(0, min(atom.depth, MAX_DEPTH_BINS - 1)))
    if n_dim == 1:
        t[0] = float(depth_bin)
        return t
    for i in range(min(MAX_DEPTH_BINS, n_dim)):
        t[i] = 1.0 if i == depth_bin else 0.0
    if n_dim > 3:
        t[3] = float(max(0.0, min(atom.T, 1.0)))
    if n_dim >= 5:
        t[4] = float(int(atom.state)) / float(max(STATE_DIM - 1, 1))
    if n_dim >= 6:
        mz = 0.0 if atom.mazur_tracer is None else float(atom.mazur_tracer)
        t[5] = float(max(0.0, min(mz, 1.0)))
    # legacy pełny one-hot state + mazur@8 gdy duży tracer
    if n_dim >= 10:
        st = int(atom.state)
        for i in range(STATE_DIM):
            idx = 6 + i
            if idx >= n_dim:
                break
            t[idx] = 1.0 if i == st else 0.0
        if n_dim > 10 and atom.mazur_tracer is not None:
            t[10] = float(max(0.0, min(atom.mazur_tracer, 1.0)))
    return t


def graphs_to_batch(
    graphs: Sequence[PhiGraph],
    d_x: int = 8,
    n_dim: int = 6,
    max_n: int | None = None,
    use_tracer: bool = True,
) -> PhiGraphBatch:
    """Pakuje listę PhiGraph → tensory; padding do max długości."""
    if not graphs:
        raise ValueError("pusty batch")
    lengths = [len(g) for g in graphs]
    N = max_n or max(lengths)
    B = len(graphs)
    content = torch.zeros(B, N, d_x)
    tracer = torch.zeros(B, N, n_dim)
    depths = torch.zeros(B, N)
    states = torch.zeros(B, N, dtype=torch.long)
    mask = torch.zeros(B, N)
    labels = torch.zeros(B, dtype=torch.long)

    # normalizacja T w batchu
    all_T = [a.T for g in graphs for a in g.atoms] or [1.0]
    t_max = max(all_T) if max(all_T) > 0 else 1.0

    for bi, g in enumerate(graphs):
        labels[bi] = int(g.label or 0)
        for ai, atom in enumerate(g.atoms[:N]):
            a = PhiAtom(
                atom_id=atom.atom_id,
                S=atom.S,
                E=atom.E,
                T=atom.T / t_max,
                state=atom.state,
                bubble_id=atom.bubble_id,
                depth=atom.depth,
                payload=atom.payload,
                mazur_tracer=atom.mazur_tracer,
            )
            content[bi, ai] = atom_content_vec(a, d_x)
            if use_tracer:
                tracer[bi, ai] = atom_tracer_vec(a, n_dim)
                # dim pozycji w sekwencji (jeśli jest miejsce)
                if n_dim > 9:
                    tracer[bi, ai, 9] = ai / max(N - 1, 1)
            depths[bi, ai] = float(atom.depth)
            states[bi, ai] = int(atom.state)
            mask[bi, ai] = 1.0

    return PhiGraphBatch(
        content=content,
        tracer=tracer,
        labels=labels,
        depths=depths,
        states=states,
        mask=mask,
    )


def flat_text_content(text: str, n_tokens: int, d_x: int) -> torch.Tensor:
    """Kontrola: spłaszczony tekst jak „plik Unix” — tylko hash chunków, bez Φ."""
    if n_tokens <= 0:
        raise ValueError("n_tokens")
    chunk = max(1, len(text) // n_tokens)
    rows = []
    for i in range(n_tokens):
        part = text[i * chunk : (i + 1) * chunk] or f"pad-{i}"
        rows.append(hash_embed(part, d_x, seed=100 + i))
    return torch.stack(rows, dim=0)
