"""
Syntetyczne Φ-drzewo bańek — odpowiednik tree_data, ale z osiami Karmazyn.

Warstwy depth 0/1/2 (jak warstwy energii w labie Most).
Etykieta = interakcje wewnątrz warstw depth % 4.
"""

from __future__ import annotations

import torch

from phi_schema import (
    PhiAtom,
    PhiGraph,
    PhiGraphBatch,
    PhiState,
    graphs_to_batch,
    layer_parent_edges,
)


def generate_phi_tree(
    batch_size: int,
    seq_len: int = 16,
    d_x: int = 8,
    n_dim: int = 6,
    use_tracer: bool = True,
) -> PhiGraphBatch:
    """
    Layout (seq_len=16):
      0–3   depth 0 (korzeń bańki)
      4–11  depth 1
      12–15 depth 2 (liście)

    Content = surowe wektory (jak tree_data), żeby etykieta korelowała z x.
    Tracer = osie Φ (depth / T / state / mazur) — obserwator, nie treść.
    """
    assert seq_len == 16, "layout warstw na sztywno dla seq_len=16"
    graphs: list[PhiGraph] = []
    raw_batch = torch.randn(batch_size, seq_len, d_x)
    depths = [0] * 4 + [1] * 8 + [2] * 4

    for b in range(batch_size):
        atoms: list[PhiAtom] = []
        raw = raw_batch[b]
        y = 0
        for layer in (0, 1, 2):
            idxs = [i for i, d in enumerate(depths) if d == layer]
            layer_sum = raw[idxs].sum()
            y = y + (1 if layer_sum > 0 else 0) * (layer + 1)
        label = y % 4

        for i in range(seq_len):
            depth = depths[i]
            T = 1.0 - 0.25 * depth + 0.05 * torch.rand(1).item()
            state = PhiState.HOT if depth < 2 else PhiState.WARM
            payload = None
            if state == PhiState.HOT and i % 3 == 0:
                payload = f"chunk-{b}-{i}"
            atoms.append(
                PhiAtom(
                    atom_id=f"a{b}_{i}",
                    S=f"struct/layer{depth}/slot{i}",
                    E=f"sem/{b}/{i}",
                    T=T,
                    state=state,
                    bubble_id=f"bub-d{depth}",
                    depth=depth,
                    payload=payload,
                    mazur_tracer=0.5 + 0.5 * (1.0 - depth / 2.0),
                )
            )

        edges = layer_parent_edges(atoms)
        graphs.append(PhiGraph(atoms=atoms, edges=edges, label=label))

    batch = graphs_to_batch(graphs, d_x=d_x, n_dim=n_dim, use_tracer=use_tracer)
    # nadpisz content surowym x (etykieta z niego); tracer zostaje z Φ
    return PhiGraphBatch(
        content=raw_batch,
        tracer=batch.tracer,
        labels=batch.labels,
        depths=batch.depths,
        states=batch.states,
        mask=batch.mask,
    )

def same_text_two_views(text: str, n_atoms: int = 8, d_x: int = 8, n_dim: int = 6):
    """
    Ten sam tekst jako:
      1) płaski Unix (chunki)
      2) Φ-graph (atomy z depth/T/state + payload na HOT)
    Do testu ontologii „nie płaski”.
    """
    from phi_schema import flat_text_content

    flat = flat_text_content(text, n_atoms, d_x)

    atoms = []
    chunk = max(1, len(text) // n_atoms)
    for i in range(n_atoms):
        part = text[i * chunk : (i + 1) * chunk] or f"pad-{i}"
        depth = 0 if i < n_atoms // 3 else (1 if i < 2 * n_atoms // 3 else 2)
        state = PhiState.HOT if depth == 0 else (PhiState.WARM if depth == 1 else PhiState.COLD)
        atoms.append(
            PhiAtom(
                atom_id=f"t{i}",
                S=f"text/struct/{i}",
                E=f"text/sem/{i}",
                T=1.0 - 0.3 * depth,
                state=state,
                bubble_id="doc",
                depth=depth,
                payload=part if state == PhiState.HOT else None,
                mazur_tracer=1.0 - 0.2 * depth,
            )
        )
    g = PhiGraph(atoms=atoms, label=0, meta={"source_text": text})
    phi = graphs_to_batch([g], d_x=d_x, n_dim=n_dim, use_tracer=True)
    return flat, phi
