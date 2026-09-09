"""
Loader dumpów Φ dla PhiMost.

Format kanoniczny (ten repo):
  dumps/<name>/manifest.json
  dumps/<name>/atoms.jsonl
  dumps/<name>/edges.jsonl

BubbleFS (*.bbl) — opcjonalny, nie wymagany.
Atomstore — nie kanoniczne.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState, graphs_to_batch


STATE_MAP = {
    "HOT": PhiState.HOT,
    "WARM": PhiState.WARM,
    "COLD": PhiState.COLD,
    "TOMB": PhiState.TOMB,
    0: PhiState.HOT,
    1: PhiState.WARM,
    2: PhiState.COLD,
    3: PhiState.TOMB,
}


def _parse_state(v) -> PhiState:
    if isinstance(v, PhiState):
        return v
    if v in STATE_MAP:
        return STATE_MAP[v]
    if isinstance(v, str) and v.upper() in STATE_MAP:
        return STATE_MAP[v.upper()]
    return PhiState.HOT


def load_atoms_jsonl(path: Path) -> list[PhiAtom]:
    atoms = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            # E bez mutacji topic — topic zostaje w JSON dla callerów (dump_recall)
            atoms.append(
                PhiAtom(
                    atom_id=str(o["id"]),
                    S=str(o.get("S", "")),
                    E=str(o.get("E", "")),
                    T=float(o.get("T", 1.0)),
                    state=_parse_state(o.get("state", "HOT")),
                    bubble_id=str(o.get("bubble", "root")),
                    depth=int(o.get("depth", 0)),
                    payload=o.get("payload"),
                    mazur_tracer=(
                        float(o["mazur_tracer"])
                        if o.get("mazur_tracer") is not None
                        else None
                    ),
                )
            )
    return atoms


def load_edges_jsonl(path: Path) -> list[PhiEdge]:
    edges = []
    if not path.exists():
        return edges
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            edges.append(
                PhiEdge(
                    kind=str(o.get("kind", "bind")),
                    src=str(o["src"]),
                    dst=str(o["dst"]),
                    name=str(o.get("name", "")),
                )
            )
    return edges


def load_dump(dump_dir: str | Path) -> tuple[dict, PhiGraph]:
    """Wczytaj dump → (manifest, PhiGraph)."""
    root = Path(dump_dir)
    man_path = root / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    atoms = load_atoms_jsonl(root / "atoms.jsonl")
    edges = load_edges_jsonl(root / "edges.jsonl")
    g = PhiGraph(atoms=atoms, edges=edges, meta={"dump": str(root), **manifest})
    return manifest, g


def graph_to_content_tracer(g: PhiGraph, d_x: int = 8, n_dim: int = 6, use_tracer: bool = True):
    """Jeden graf → batch B=1 (content z hash S/E/payload, tracer z osi Φ)."""
    return graphs_to_batch([g], d_x=d_x, n_dim=n_dim, use_tracer=use_tracer)


def iter_bind_pairs(g: PhiGraph) -> Iterator[tuple[str, str]]:
    idset = {a.atom_id for a in g.atoms}
    for e in g.edges:
        if e.kind == "bind" and e.src in idset and e.dst in idset:
            yield e.src, e.dst


def warm_without_payload(g: PhiGraph) -> PhiGraph:
    """Sanity HOT/WARM: zdejmij payload z WARM/COLD (i tak nie powinno być)."""
    atoms = []
    for a in g.atoms:
        payload = a.payload if a.state == PhiState.HOT else None
        atoms.append(
            PhiAtom(
                atom_id=a.atom_id,
                S=a.S,
                E=a.E,
                T=a.T,
                state=a.state,
                bubble_id=a.bubble_id,
                depth=a.depth,
                payload=payload,
                mazur_tracer=a.mazur_tracer,
            )
        )
    return PhiGraph(atoms=atoms, edges=list(g.edges), label=g.label, meta=dict(g.meta))


def strip_all_payloads(g: PhiGraph) -> PhiGraph:
    """Kontrola: zero bajtów — tylko S/E/T/state/depth."""
    atoms = [
        PhiAtom(
            atom_id=a.atom_id,
            S=a.S,
            E=a.E,
            T=a.T,
            state=a.state,
            bubble_id=a.bubble_id,
            depth=a.depth,
            payload=None,
            mazur_tracer=a.mazur_tracer,
        )
        for a in g.atoms
    ]
    return PhiGraph(atoms=atoms, edges=list(g.edges), label=g.label, meta=dict(g.meta))
