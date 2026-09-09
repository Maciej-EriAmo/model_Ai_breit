"""
Faza C — dataset recall/ranking nad Φ.

Query atom → kandydaci w bańce.
Pozytyw = bind: ten sam topic + ta sama depth (sąsiad w warstwie).
Hard neg = ten sam topic, inna depth.
Easy neg = inny topic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phi_schema import (
    PhiAtom,
    PhiEdge,
    PhiGraph,
    PhiGraphBatch,
    PhiState,
    graphs_to_batch,
    hash_embed,
)


N_TOPICS = 4
# 24 atomy: 8 na depth → każdy topic 2× w warstwie + klony cross-depth (hard neg)
SEQ_LEN = 24
DEPTHS = [0] * 8 + [1] * 8 + [2] * 8


@dataclass
class RecallBatch:
    content: torch.Tensor  # [B, N, d_x]
    tracer: torch.Tensor  # [B, N, n]
    depths: torch.Tensor  # [B, N]
    states: torch.Tensor  # [B, N]
    mask: torch.Tensor  # [B, N]
    query_idx: torch.Tensor  # [B]
    pos_idx: torch.Tensor  # [B]
    topics: torch.Tensor  # [B, N]

    def ablate(
        self,
        *,
        zero_tracer: bool = False,
        drop_depth: bool = False,
        drop_T: bool = False,
        drop_state: bool = False,
        drop_mazur: bool = False,
        flat_payload: bool = False,
    ) -> "RecallBatch":
        content = self.content.clone()
        tracer = self.tracer.clone()
        n = tracer.shape[-1]

        if zero_tracer:
            tracer = torch.zeros_like(tracer)
        else:
            if drop_depth and n >= 1:
                if n == 1:
                    tracer[:, :, 0] = 0.0
                else:
                    tracer[:, :, :3] = 0.0
            if drop_T and n > 3:
                tracer[:, :, 3] = 0.0
            if drop_state and n >= 5:
                tracer[:, :, 4] = 0.0
            if drop_mazur and n >= 6:
                tracer[:, :, 5] = 0.0

        if flat_payload:
            # treść tylko z „bajtów” pozycji — bez S/E topic; tracer ZERO
            B, N, d = content.shape
            flat = torch.zeros_like(content)
            for b in range(B):
                for i in range(N):
                    flat[b, i] = hash_embed(f"byte-{b}-{i}", d, seed=200 + i)
            content = flat
            tracer = torch.zeros_like(tracer)

        return RecallBatch(
            content=content,
            tracer=tracer,
            depths=self.depths,
            states=self.states,
            mask=self.mask,
            query_idx=self.query_idx,
            pos_idx=self.pos_idx,
            topics=self.topics,
        )


def _topic_content(topic: int, slot: int, d_x: int, noise: torch.Tensor) -> torch.Tensor:
    """Content = topic prototype + slot noise (semantyka w x, nie w tracerze)."""
    proto = hash_embed(f"topic/{topic}", d_x, seed=7)
    slot_v = hash_embed(f"slot/{slot}", d_x, seed=11)
    v = 0.75 * proto + 0.15 * slot_v + 0.10 * noise
    return v / v.norm().clamp_min(1e-6)


def _topics_for_layout(seq_len: int = SEQ_LEN) -> list[int]:
    """W każdej depth: [0,0,1,1,2,2,3,3] — para same-depth + hard negs cross-depth."""
    assert seq_len == 24
    per_depth = []
    for t in range(N_TOPICS):
        per_depth.extend([t, t])
    assert len(per_depth) == 8
    return per_depth + per_depth + per_depth


def generate_recall_batch(
    batch_size: int,
    d_x: int = 8,
    n_dim: int = 6,
    seq_len: int = SEQ_LEN,
    *,
    decorrelate_axes: bool = True,
) -> RecallBatch:
    """
    domyślnie `decorrelate_axes=True` (P2.13):
      T, state, mazur **nie** są funkcją depth — ablacje −depth/−T mają sens.
    `decorrelate_axes=False` = legacy (T↓ z depth) do porównania.
    """
    assert seq_len == SEQ_LEN
    graphs: list[PhiGraph] = []
    raw = torch.zeros(batch_size, seq_len, d_x)
    topics_t = torch.zeros(batch_size, seq_len, dtype=torch.long)
    query_idx = torch.zeros(batch_size, dtype=torch.long)
    pos_idx = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        topic_of = _topics_for_layout(seq_len)
        topics_t[b] = torch.tensor(topic_of, dtype=torch.long)
        atoms: list[PhiAtom] = []
        noise = torch.randn(seq_len, d_x)
        # osie niezależne od depth (permutacja / hash topic+slot)
        if decorrelate_axes:
            t_axis = torch.rand(seq_len)
            # state: cyklicznie po slocie, nie po depth
            state_of = [
                PhiState.HOT if (i + topic_of[i]) % 3 != 2 else PhiState.WARM
                for i in range(seq_len)
            ]
            mazur_of = (0.15 + 0.7 * ((torch.arange(seq_len) * 5 + 3) % 7).float() / 6.0)
        for i in range(seq_len):
            depth = DEPTHS[i]
            topic = topic_of[i]
            raw[b, i] = _topic_content(topic, i, d_x, noise[i])
            if decorrelate_axes:
                T = float(t_axis[i].item())
                state = state_of[i]
                mazur = float(mazur_of[i].item())
            else:
                T = 1.0 - 0.25 * depth + 0.03 * torch.rand(1).item()
                state = PhiState.HOT if depth < 2 else PhiState.WARM
                mazur = 0.5 + 0.5 * (1.0 - depth / 2.0)
            atoms.append(
                PhiAtom(
                    atom_id=f"a{b}_{i}",
                    S=f"struct/d{depth}/t{topic}",
                    E=f"sem/topic{topic}/slot{i}",
                    T=T,
                    state=state,
                    bubble_id=f"bub-d{depth}",
                    depth=depth,
                    payload=f"payload-t{topic}-{i}" if state == PhiState.HOT else None,
                    mazur_tracer=mazur,
                )
            )

        # wybierz query i pozytyw: ten sam topic + ta sama depth, inny slot
        # zbierz pary
        pairs = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                if topic_of[i] == topic_of[j] and DEPTHS[i] == DEPTHS[j]:
                    pairs.append((i, j))
        if not pairs:
            # fallback — nie powinno się zdarzyć przy layoutcie
            pairs = [(0, 1)]
        q, p = pairs[torch.randint(0, len(pairs), (1,)).item()]
        query_idx[b] = q
        pos_idx[b] = p

        edges = [
            PhiEdge("bind", atoms[q].atom_id, atoms[p].atom_id, name="same_layer_topic"),
            PhiEdge("parent", "bub-d1", "bub-d0"),
            PhiEdge("parent", "bub-d2", "bub-d1"),
        ]
        graphs.append(PhiGraph(atoms=atoms, edges=edges, label=int(topic_of[q])))

    packed = graphs_to_batch(graphs, d_x=d_x, n_dim=n_dim, use_tracer=True)
    return RecallBatch(
        content=raw,  # topic geometry w content
        tracer=packed.tracer,  # Φ observer
        depths=packed.depths,
        states=packed.states,
        mask=packed.mask,
        query_idx=query_idx,
        pos_idx=pos_idx,
        topics=topics_t,
    )


def make_loader(n_batches: int, batch_size: int, **kw) -> list[RecallBatch]:
    return [generate_recall_batch(batch_size, **kw) for _ in range(n_batches)]


def axis_correlation_report(batch: RecallBatch) -> dict[str, float]:
    """Pearson |r| depth↔T, depth↔mazur (n_dim≥6) — im niżej, tym lepsza dekorelacja."""
    d = batch.depths[0].float()
    t = batch.tracer[0]
    out: dict[str, float] = {}

    def _abs_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a - a.mean()
        b = b - b.mean()
        den = a.norm() * b.norm()
        if float(den) < 1e-8:
            return 0.0
        return float((a * b).sum().abs() / den)

    if t.shape[-1] > 3:
        out["|corr|(depth,T)"] = _abs_corr(d, t[:, 3])
    if t.shape[-1] >= 6:
        out["|corr|(depth,mazur)"] = _abs_corr(d, t[:, 5])
    if t.shape[-1] >= 5:
        out["|corr|(depth,state)"] = _abs_corr(d, t[:, 4])
    return out
