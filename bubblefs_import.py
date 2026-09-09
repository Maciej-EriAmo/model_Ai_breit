"""
Importer BubbleFS (*.bbl) → PhiGraph / PhiJSON.

Czyta layout Karmazyn extras/bubblefs.py **bez** uruchamiania runtime Store:
  <root>/manifest.json
  <root>/bubbles/<id>.bbl
  <root>/holograms/<id>.hgm   (opcjonalnie → krawędzie bind)

Atomstore nie jest kanonem — to adapter offline pod MemoryBus / dump.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState

ROOT = Path(__file__).resolve().parent


def _ub64(s: str) -> bytes:
    return base64.b64decode(s)


def _vec_digest(vec: list | None, n: int = 4) -> str:
    if not vec:
        return "0"
    # skrót stabilny bez numpy
    parts = []
    for x in vec[:n]:
        parts.append(f"{float(x):.4f}")
    return ",".join(parts)


def _state_from_bbl(bbl: dict) -> PhiState:
    if bbl.get("revoked"):
        return PhiState.TOMB
    meta = bbl.get("metadata") or {}
    st = str(meta.get("state", "")).upper()
    if st in PhiState.__members__:
        return PhiState[st]
    # decay rozpoczęty → WARM/COLD
    if bbl.get("decay_start_epoch") is not None:
        rate = float(bbl.get("decay_rate") or 0.0)
        return PhiState.COLD if rate > 0.5 else PhiState.WARM
    content_b64 = bbl.get("content_b64") or ""
    if content_b64 and not bbl.get("content_encrypted"):
        return PhiState.HOT
    if content_b64:
        return PhiState.WARM  # zaszyfrowane — bez plaintext
    return PhiState.WARM


def _decode_payload(bbl: dict) -> str | None:
    if bbl.get("content_encrypted"):
        return None  # nie zgadujemy shared_secret
    raw_b64 = bbl.get("content_b64") or ""
    if not raw_b64:
        return None
    try:
        raw = _ub64(raw_b64)
    except Exception:
        return None
    # UTF-8 tekst gdy się da; inaczej hex-skrót
    try:
        text = raw.decode("utf-8")
        if text.strip():
            return text[:4000]
    except UnicodeDecodeError:
        pass
    return f"bin:{raw[:32].hex()}"


def _T_from_bbl(bbl: dict, vacuum: float = 0.1) -> float:
    meta = bbl.get("metadata") or {}
    if "T" in meta:
        return float(max(0.0, min(float(meta["T"]), 1.0)))
    # recall_count ↑ → cieplejsze; decay → chłodniejsze
    rc = float(bbl.get("recall_count") or 0)
    t = 0.35 + 0.5 * (1.0 - math.exp(-rc / 5.0))
    if bbl.get("decay_start_epoch") is not None:
        t *= max(0.15, 1.0 - float(bbl.get("decay_rate") or 0.0))
    return float(max(vacuum, min(t, 1.0)))


def _depth_from_bbl(bbl: dict, inode_rank: dict[int, int] | None = None) -> int:
    meta = bbl.get("metadata") or {}
    if "depth" in meta:
        return int(meta["depth"])
    if bbl.get("consolidated_from"):
        return 1
    inode = bbl.get("inode")
    if inode_rank is not None and inode in inode_rank:
        return min(2, inode_rank[inode])
    return 0


def bbl_to_atom(
    bbl: dict,
    *,
    vacuum: float = 0.1,
    inode_rank: dict[int, int] | None = None,
) -> PhiAtom:
    label = str(bbl.get("label") or bbl.get("id") or "bubble")
    bid = str(bbl.get("id") or label)
    s_dig = _vec_digest(bbl.get("S_struct"))
    e_dig = _vec_digest(bbl.get("S_sem"))
    state = _state_from_bbl(bbl)
    # payload tylko HOT; E woli plaintext content (czytelny recall)
    raw_payload = _decode_payload(bbl)
    payload = raw_payload if state == PhiState.HOT else None
    if raw_payload:
        E = raw_payload[:240]
    else:
        E = f"bbl/E/{label}/{e_dig}"
    mz = None
    meta = bbl.get("metadata") or {}
    if "mazur_tracer" in meta:
        mz = float(meta["mazur_tracer"])
    elif bbl.get("fingerprint_b64"):
        # sonda z fingerprintu (0..1)
        fp = _ub64(bbl["fingerprint_b64"])
        mz = (fp[0] / 255.0) if fp else 0.5
    return PhiAtom(
        atom_id=bid,
        S=f"bbl/S/{label}/{s_dig}",
        E=E,
        T=_T_from_bbl(bbl, vacuum=vacuum),
        state=state,
        bubble_id=label,
        depth=_depth_from_bbl(bbl, inode_rank=inode_rank),
        payload=payload,
        mazur_tracer=mz,
    )


def load_bbl_file(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_hgm_file(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _inode_ranks(bbls: list[dict]) -> dict[int, int]:
    inodes = sorted({int(b["inode"]) for b in bbls if b.get("inode") is not None})
    return {ino: i % 3 for i, ino in enumerate(inodes)}


def bubblefs_to_graph(
    root: str | Path,
    *,
    shared_secret: bytes | None = None,
) -> tuple[dict, PhiGraph]:
    """
    Wczytaj katalog BubbleFS → (manifest, PhiGraph).
    `shared_secret` zarezerwowane (szyfrowane content → bez payload).
    """
    _ = shared_secret  # nie deszyfrujemy tu — RO adapter
    root = Path(root)
    man_path = root / "manifest.json"
    manifest = (
        json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    )
    bdir = root / "bubbles"
    if not bdir.is_dir():
        # pojedynczy .bbl albo katalog z plikami .bbl
        if root.is_file() and root.suffix == ".bbl":
            bbls = [load_bbl_file(root)]
            root_meta = {"single_bbl": str(root)}
            manifest = {**manifest, **root_meta}
        else:
            raise FileNotFoundError(f"brak bubbles/ w {root}")
    else:
        bbls = [
            load_bbl_file(p)
            for p in sorted(bdir.glob("*.bbl"))
        ]
    if not bbls:
        raise ValueError(f"BubbleFS bez plików .bbl: {root}")

    vacuum = float(manifest.get("t_vacuum()", 0.1) or 0.1)
    ranks = _inode_ranks(bbls)
    atoms = [bbl_to_atom(b, vacuum=vacuum, inode_rank=ranks) for b in bbls]
    id_set = {a.atom_id for a in atoms}
    label_to_id = {str(b.get("label") or b.get("id")): str(b.get("id") or b.get("label")) for b in bbls}

    edges: list[PhiEdge] = []
    # parent z consolidated_from / metadata.parent
    for b in bbls:
        src = str(b.get("id") or b.get("label"))
        parent = (b.get("metadata") or {}).get("parent")
        if parent and str(parent) in id_set:
            edges.append(PhiEdge("parent", src, str(parent)))
        cf = b.get("consolidated_from")
        if isinstance(cf, list):
            for p in cf:
                if str(p) in id_set:
                    edges.append(PhiEdge("parent", src, str(p), name="consolidated"))
        elif cf and str(cf) in id_set:
            edges.append(PhiEdge("parent", src, str(cf), name="consolidated"))

    # hologramy → bind między etykietami
    hdir = root / "holograms"
    if hdir.is_dir():
        for hp in sorted(hdir.glob("*.hgm")):
            h = load_hgm_file(hp)
            labels = list(h.get("bubble_labels") or [])
            ids = []
            for lab in labels:
                i = label_to_id.get(str(lab)) or (str(lab) if str(lab) in id_set else None)
                if i:
                    ids.append(i)
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    edges.append(
                        PhiEdge("bind", a, b, name=str(h.get("topic") or hp.stem))
                    )

    # jeśli brak bindów — połącz sekwencyjnie w bańce (słaby graf, żeby recall działał)
    if not any(e.kind == "bind" for e in edges) and len(atoms) >= 2:
        for i in range(len(atoms) - 1):
            edges.append(
                PhiEdge("bind", atoms[i].atom_id, atoms[i + 1].atom_id, name="bbl-seq")
            )

    g = PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={
            "source": "bubblefs",
            "path": str(root),
            "bubblefs_version": manifest.get("bubblefs_version"),
            "n_bbl": len(bbls),
        },
    )
    return manifest, g


def export_graph_as_dump(g: PhiGraph, out_dir: str | Path, *, description: str = "") -> Path:
    """Zapisz PhiJSON kompatybilny z phi_dump / MemoryBus."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in g.atoms:
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
    with (out_dir / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in g.edges:
            f.write(
                json.dumps(
                    {"kind": e.kind, "src": e.src, "dst": e.dst, "name": e.name},
                    ensure_ascii=False,
                )
                + "\n"
            )
    man = {
        "format": "phijson-v1",
        "description": description or "imported from BubbleFS",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "source": g.meta.get("source"),
        "bubblefs_path": g.meta.get("path"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


def write_fixture_bubblefs(path: str | Path | None = None) -> Path:
    """Minimalny BubbleFS do testów (plaintext, bez Karmazyn runtime)."""
    path = Path(path) if path else ROOT / "dumps" / "bubblefs_fixture"
    bdir = path / "bubbles"
    hdir = path / "holograms"
    bdir.mkdir(parents=True, exist_ok=True)
    hdir.mkdir(parents=True, exist_ok=True)

    def bbl(
        bid: str,
        label: str,
        inode: int,
        text: str,
        *,
        depth: int = 0,
        parent: str | None = None,
        revoked: bool = False,
        recall: int = 1,
    ) -> dict:
        content = text.encode("utf-8")
        meta: dict[str, Any] = {"depth": depth, "state": "HOT"}
        if parent:
            meta["parent"] = parent
        return {
            "id": bid,
            "label": label,
            "inode": inode,
            "epoch_born": 1,
            "recall_count": recall,
            "consolidated_from": [],
            "metadata": meta,
            "revoked": revoked,
            "content_encrypted": False,
            "content_b64": base64.b64encode(content).decode(),
            "fingerprint_b64": base64.b64encode(bid.encode()).decode(),
            "S_struct": [0.1 * inode, 0.2, 0.3],
            "S_sem": [0.5, 0.1 * len(label), 0.2],
        }

    items = [
        bbl("b_root", "notes", 1, "Notatka Most i Φ", depth=0, recall=5),
        bbl("b_intro", "intro", 2, "Wstęp: Most Lorentza", depth=0, recall=3),
        bbl("b_most", "most", 3, "softplus×2 + tracer", depth=1, parent="b_root", recall=4),
        bbl("b_recall", "recall", 4, "InfoNCE bind same depth", depth=1, parent="b_root", recall=2),
        bbl("b_cold", "archive", 5, "stary fragment", depth=2, parent="b_most", recall=0),
    ]
    # cold → decay
    items[-1]["decay_start_epoch"] = 10
    items[-1]["decay_rate"] = 0.4
    items[-1]["metadata"]["state"] = "WARM"

    for b in items:
        (bdir / f"{b['id']}.bbl").write_text(
            json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    hgm = {
        "id": "h1",
        "topic": "most-lab",
        "proto": [0.1, 0.2],
        "generators": [[0.1], [0.2]],
        "weights": [0.5, 0.5],
        "bubble_labels": ["notes", "most", "recall"],
        "epoch_created": 1,
        "decay_rate": 0.0,
        "metadata": {},
    }
    (hdir / "h1.hgm").write_text(
        json.dumps(hgm, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    man = {
        "bubblefs_version": "1.0.0",
        "karmazyn_version": "fixture",
        "epoch": 1,
        "dim": 8,
        "t_vacuum()": 0.1,
        "n_bubbles": len(items),
        "n_holograms": 1,
        "encrypted": False,
        "include_phi_vectors": False,
    }
    (path / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="BubbleFS → PhiJSON")
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="katalog BubbleFS (domyślnie: zbuduj fixture)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dumps" / "from_bubblefs",
        help="docelowy dump PhiJSON",
    )
    ap.add_argument("--fixture", action="store_true", help="zapisz dumps/bubblefs_fixture")
    args = ap.parse_args()

    src = args.src
    if args.fixture or src is None:
        src = write_fixture_bubblefs()
        print(f"fixture → {src}")

    man, g = bubblefs_to_graph(src)
    out = export_graph_as_dump(
        g, args.out, description=f"BubbleFS import ({man.get('bubblefs_version')})"
    )
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    print(f"atoms={len(g.atoms)} edges={len(g.edges)} bind={n_bind} → {out}")


if __name__ == "__main__":
    main()
