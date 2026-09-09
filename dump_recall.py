"""
Recall-taski z dumpa Φ: pozytyw = krawędź bind (prefer same-depth).

Content = atom_content_vec(S/E/payload) — strip payload ma efekt.
Tracer = osie Φ (depth/T/state/mazur).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from phi_dump import load_dump, warm_without_payload, _parse_state
from phi_schema import (
    PhiAtom,
    PhiGraph,
    PhiState,
    atom_content_vec,
    atom_tracer_vec,
    hash_embed,
)
from recall_data import RecallBatch


def load_raw_records(dump_dir: str | Path) -> list[dict]:
    path = Path(dump_dir) / "atoms.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def records_to_graph(rows: list[dict], edges: list) -> PhiGraph:
    atoms = []
    for o in rows:
        atoms.append(
            PhiAtom(
                atom_id=str(o["id"]),
                S=str(o.get("S", "")),
                E=str(o.get("E", "")),
                T=float(o.get("T", 1.0)),
                state=_parse_state(o.get("state", "HOT")),
                bubble_id=str(o.get("bubble", "root")),
                depth=int(o.get("depth", 0)),
                payload=o.get("payload") if _parse_state(o.get("state", "HOT")) == PhiState.HOT else None,
                mazur_tracer=(
                    float(o["mazur_tracer"]) if o.get("mazur_tracer") is not None else None
                ),
            )
        )
    return PhiGraph(atoms=atoms, edges=list(edges), meta={"n": len(atoms)})


def _index(g: PhiGraph) -> dict[str, int]:
    return {a.atom_id: i for i, a in enumerate(g.atoms)}


def bind_pairs_idx(g: PhiGraph) -> list[tuple[int, int]]:
    ix = _index(g)
    pairs = []
    seen = set()
    for e in g.edges:
        if e.kind != "bind":
            continue
        if e.src not in ix or e.dst not in ix:
            continue
        a, b = ix[e.src], ix[e.dst]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def bind_pairs_same_depth(g: PhiGraph) -> list[tuple[int, int]]:
    return [
        (a, b)
        for a, b in bind_pairs_idx(g)
        if g.atoms[a].depth == g.atoms[b].depth
    ]


def graph_to_tensors(
    g: PhiGraph,
    topics: list[str] | None = None,  # legacy unused — content ≠ topic prototype
    d_x: int = 8,
    n_dim: int = 6,
    noise_scale: float = 0.0,
    *,
    blend_holo_proto: float = 0.0,
    holo_bins: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Content z S/E/payload (`atom_content_vec`).

    `blend_holo_proto`: domieszka prototypu klastra z `mazur_tracer` (hologram).
    Jak topic w Fazie C — semantyka bańki/holo w x; geometria nadal w τ.
    """
    _ = topics
    N = len(g.atoms)
    content = torch.zeros(N, d_x)
    tracer = torch.zeros(N, n_dim)
    depths = torch.zeros(N)
    states = torch.zeros(N, dtype=torch.long)
    t_max = max((a.T for a in g.atoms), default=1.0) or 1.0
    bins = max(int(holo_bins), 1)
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
        vec = atom_content_vec(aa, d_x)
        if blend_holo_proto > 0 and a.mazur_tracer is not None:
            bin_id = int(min(bins - 1, max(0, float(a.mazur_tracer) * bins)))
            proto = hash_embed(f"holo-bin/{bin_id}", d_x, seed=77)
            w = float(blend_holo_proto)
            vec = (1.0 - w) * vec + w * proto
            vec = vec / vec.norm().clamp_min(1e-6)
        if noise_scale > 0:
            vec = vec + noise_scale * torch.randn(d_x)
            vec = vec / vec.norm().clamp_min(1e-6)
        content[i] = vec
        tracer[i] = atom_tracer_vec(aa, n_dim)
        depths[i] = float(a.depth)
        states[i] = int(a.state)
    return content, tracer, depths, states


def make_recall_from_graph(
    g: PhiGraph,
    topics: list[str],
    batch_size: int,
    d_x: int = 8,
    n_dim: int = 6,
    seed: int = 0,
    *,
    pad_to: int | None = None,
    noise_scale: float | None = None,
    pair_mode: str = "same_depth",
    t_jitter: float = 0.05,
    blend_holo_proto: float = 0.0,
    restrict_same_prefix: bool = False,
) -> RecallBatch:
    """
    pair_mode:
      same_depth — tylko bind same-depth (domyślnie)
      all — wszystkie bindy
      mixed — 50/50 same-depth / cross-depth (gdy są); daje Mostowi ΔE
    """
    rng = random.Random(seed)
    pairs_sd = bind_pairs_same_depth(g)
    pairs_all = bind_pairs_idx(g)
    pairs_xd = [
        (a, b)
        for a, b in pairs_all
        if g.atoms[a].depth != g.atoms[b].depth
    ]
    if pair_mode == "all":
        pairs = pairs_all
    elif pair_mode == "mixed":
        pairs = pairs_sd or pairs_all
    else:
        pairs = pairs_sd if pairs_sd else pairs_all
    if not pairs:
        raise ValueError("dump bez krawędzi bind — brak recall positives")

    N0 = len(g.atoms)
    N = pad_to or N0
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.zeros(batch_size, N)
    topics_t = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    topic_ids = {t: i for i, t in enumerate(sorted(set(topics)))}
    # clean batch (seed=0,b=0): zero noise; train: domyślnie łagodniejszy szum
    default_noise = 0.08

    for b in range(batch_size):
        if noise_scale is not None:
            ns = float(noise_scale)
        else:
            ns = 0.0 if (seed == 0 and b == 0) else default_noise
        c, t, d, s = graph_to_tensors(
            g,
            topics,
            d_x=d_x,
            n_dim=n_dim,
            noise_scale=ns,
            blend_holo_proto=blend_holo_proto,
        )
        if ns > 0 and t_jitter > 0 and n_dim > 3:
            t[:, 3] = (t[:, 3] + t_jitter * torch.randn(N0)).clamp(0, 1)
        content[b, :N0] = c
        tracer[b, :N0] = t
        depths[b, :N0] = d
        states[b, :N0] = s
        mask[b, :N0] = 1.0
        for i, top in enumerate(topics):
            topics_t[b, i] = topic_ids.get(top, 0)
        if pair_mode == "mixed" and pairs_xd and rng.random() < 0.5:
            qi, pi = pairs_xd[rng.randrange(len(pairs_xd))]
        else:
            qi, pi = pairs[rng.randrange(len(pairs))]
        if rng.random() < 0.5:
            qi, pi = pi, qi
        # QM / multi-graf: kandydaci tylko z tym samym prefiksem (np. pierwiastek)
        if restrict_same_prefix:
            pref = g.atoms[qi].atom_id.split("/", 1)[0]
            for j, atom in enumerate(g.atoms):
                if not atom.atom_id.startswith(pref):
                    mask[b, j] = 0.0
            mask[b, qi] = 1.0
            mask[b, pi] = 1.0
        q_idx[b] = qi
        p_idx[b] = pi

    return RecallBatch(
        content=content,
        tracer=tracer,
        depths=depths,
        states=states,
        mask=mask,
        query_idx=q_idx,
        pos_idx=p_idx,
        topics=topics_t,
    )


def make_dump_loaders(
    dump_dir: str,
    n_train: int = 80,
    n_val: int = 25,
    batch_size: int = 16,
    d_x: int = 8,
    n_dim: int = 6,
    *,
    pair_mode: str = "same_depth",
    noise_scale: float | None = None,
    seed0: int = 1000,
    blend_holo_proto: float = 0.0,
    restrict_same_prefix: bool = False,
):
    from phi_dump import load_edges_jsonl

    rows = load_raw_records(dump_dir)
    edges = load_edges_jsonl(Path(dump_dir) / "edges.jsonl")
    g = records_to_graph(rows, edges)
    g = warm_without_payload(g)
    topics = [str(r.get("topic", r.get("id", "x"))) for r in rows]
    N = len(g.atoms)
    train = [
        make_recall_from_graph(
            g,
            topics,
            batch_size,
            d_x,
            n_dim,
            seed=seed0 + i,
            pad_to=N,
            pair_mode=pair_mode,
            noise_scale=noise_scale,
            blend_holo_proto=blend_holo_proto,
            restrict_same_prefix=restrict_same_prefix,
        )
        for i in range(n_train)
    ]
    val = [
        make_recall_from_graph(
            g,
            topics,
            batch_size,
            d_x,
            n_dim,
            seed=seed0 + 8000 + i,
            pad_to=N,
            pair_mode=pair_mode,
            noise_scale=noise_scale,
            blend_holo_proto=blend_holo_proto,
            restrict_same_prefix=restrict_same_prefix,
        )
        for i in range(n_val)
    ]
    val_clean = make_recall_from_graph(
        g,
        topics,
        batch_size,
        d_x,
        n_dim,
        seed=0,
        pad_to=N,
        pair_mode=pair_mode,
        noise_scale=0.0,
        t_jitter=0.0,
        blend_holo_proto=blend_holo_proto,
        restrict_same_prefix=restrict_same_prefix,
    )
    return g, train, val, val_clean
