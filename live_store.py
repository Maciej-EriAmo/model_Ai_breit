"""
Live read-only: Store / dump → PhiGraph.

- NIGDY nie woła tick/heat/create/settle (GC poza modelem).
- PHIMOST_LIVE=0 → tylko dump / mock (bezpieczny domyślny tor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from phimost_flags import live_enabled, phimost_root
from phi_dump import load_dump
from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState


STATE_MAP = {
    "HOT": PhiState.HOT,
    "WARM": PhiState.WARM,
    "COLD": PhiState.COLD,
    "TOMB": PhiState.TOMB,
}


@dataclass
class LiveSnapshot:
    graph: PhiGraph
    source: str  # dump | mock | host
    read_only: bool = True


def _state_of(atom: Any) -> PhiState:
    st = getattr(atom, "state", None)
    if st is None:
        return PhiState.HOT
    if isinstance(st, PhiState):
        return st
    key = str(st).upper()
    return STATE_MAP.get(key, PhiState.HOT)


def atoms_from_host_objects(atoms: Iterable[Any], *, bubble: str = "live") -> list[PhiAtom]:
    out = []
    for i, a in enumerate(atoms):
        aid = str(getattr(a, "id", i))
        S = str(getattr(a, "S", "") or "")
        E = str(getattr(a, "E", "") or "")
        T = float(getattr(a, "T", 1.0) or 0.0)
        depth = int(getattr(a, "depth", 0) or 0)
        payload = None
        # host zwykle nie eksponuje payload osobno — E jako treść HOT
        st = _state_of(a)
        if st == PhiState.HOT and E:
            payload = E
        mazur = None
        try:
            from mazur_crystal import tracer_energy  # type: ignore

            mazur = float(tracer_energy(a))
        except Exception:
            md = getattr(a, "metadata", None) or {}
            if isinstance(md, dict) and "tracer" in md:
                try:
                    mazur = float(md["tracer"])
                except Exception:
                    mazur = None
        out.append(
            PhiAtom(
                atom_id=aid,
                S=S,
                E=E,
                T=T,
                state=st,
                bubble_id=str(getattr(a, "bubble_id", bubble)),
                depth=depth,
                payload=payload,
                mazur_tracer=mazur,
            )
        )
    return out


def snapshot_from_dump(dump_dir: str | Path) -> LiveSnapshot:
    man, g = load_dump(dump_dir)
    g.meta = {**g.meta, "manifest": man}
    return LiveSnapshot(graph=g, source="dump", read_only=True)


def snapshot_mock() -> LiveSnapshot:
    """Mini Store bez Karmazyn — do testów flag / CI."""
    atoms = [
        PhiAtom("q", "query/s", "query about most", 1.0, PhiState.HOT, "mock", 0, "query about most", 1.0),
        PhiAtom("a1", "lab/most", "bridge softplus", 0.9, PhiState.WARM, "mock", 0, None, 0.9),
        PhiAtom("a2", "lab/tracer", "observer deltaE", 0.85, PhiState.WARM, "mock", 0, None, 0.8),
        PhiAtom("b1", "other", "unrelated weather", 0.5, PhiState.COLD, "mock", 1, None, 0.2),
    ]
    edges = [
        PhiEdge("bind", "q", "a1", "rel"),
        PhiEdge("bind", "a1", "a2", "same_depth"),
        PhiEdge("bind", "a2", "a1", "same_depth"),
    ]
    return LiveSnapshot(
        graph=PhiGraph(atoms=atoms, edges=edges, meta={"mock": True}),
        source="mock",
        read_only=True,
    )


def snapshot_from_host(host: Any) -> LiveSnapshot:
    """
    Read-only: tylko list_atoms / get_atom / atoms().
    Wymaga PHIMOST_LIVE=1.
    """
    if not live_enabled():
        raise RuntimeError("PHIMOST_LIVE=0 — odmawiam kontaktu z HostAPI")

    raw: list = []
    errors: list[BaseException] = []

    if hasattr(host, "list_atoms"):
        try:
            arr = host.list_atoms("ALL")
            if hasattr(arr, "__iter__"):
                raw = list(arr)
        except Exception as ex:
            errors.append(ex)

    if not raw and hasattr(host, "store") and hasattr(host.store, "atoms"):
        try:
            raw = list(host.store.atoms())
        except Exception as ex:
            errors.append(ex)
            raise RuntimeError(
                f"Host store.atoms() failed after list_atoms error(s): {errors}"
            ) from ex

    if not raw:
        if errors:
            raise RuntimeError(
                f"Host list_atoms failed and no fallback atoms: {errors[0]!r}"
            ) from errors[0]
        raise RuntimeError(
            "Host returned zero atoms (empty Store or wrong host object)"
        )

    atoms = atoms_from_host_objects(raw)
    # bez zgadywania krawędzi z hosta — pusty graf relacji (ranking i tak po content/tracer)
    g = PhiGraph(atoms=atoms, edges=[], meta={"host": True, "n": len(atoms)})
    return LiveSnapshot(graph=g, source="host", read_only=True)


def export_snapshot(snap: LiveSnapshot, out_dir: str | Path) -> Path:
    """Zapis PhiJSON (read-only export)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = {
        "name": out.name,
        "source": snap.source,
        "read_only": True,
        "n_atoms": len(snap.graph.atoms),
        "n_edges": len(snap.graph.edges),
        "phimost_root": phimost_root(),
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in snap.graph.atoms:
            f.write(
                json.dumps(
                    {
                        "id": a.atom_id,
                        "S": a.S,
                        "E": a.E,
                        "T": a.T,
                        "state": a.state.name,
                        "bubble": a.bubble_id,
                        "depth": a.depth,
                        "payload": a.payload,
                        "mazur_tracer": a.mazur_tracer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in snap.graph.edges:
            f.write(
                json.dumps(
                    {"kind": e.kind, "src": e.src, "dst": e.dst, "name": e.name},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out


def open_readonly_view(
    *,
    dump_dir: Optional[str | Path] = None,
    host: Any = None,
    prefer: str = "auto",
) -> LiveSnapshot:
    """
    prefer: auto|dump|mock|host
    auto: host jeśli LIVE+host, else dump, else mock
    """
    root = Path(phimost_root())
    dump_dir = Path(dump_dir) if dump_dir else root / "dumps" / "textfile_phi"

    if prefer == "mock":
        return snapshot_mock()
    if prefer == "dump":
        return snapshot_from_dump(dump_dir)
    if prefer == "host":
        if host is None:
            raise ValueError("host wymagany dla prefer=host")
        return snapshot_from_host(host)

    # auto
    if live_enabled() and host is not None:
        return snapshot_from_host(host)
    if dump_dir.exists():
        return snapshot_from_dump(dump_dir)
    return snapshot_mock()
