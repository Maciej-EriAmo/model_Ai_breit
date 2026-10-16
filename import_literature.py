"""
Literatura / notatki → PhiJSON.

Plik .md / .txt (albo katalog takich plików):
  - dokument = bańka + atom __main__ (HOT, payload = treść)
  - nagłówki # ## ### = sekcje (WARM/HOT), depth = poziom
  - akapity bez nagłówka = chunki pod najbliższą sekcją
  - bind: main ↔ sekcje; sekcja ↔ chunki; sąsiednie sekcje

  python import_literature.py --src moja_ksiazka.md
  python phimost_learn.py --src moja_ksiazka.md --no-warm --no-promote
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from phi_schema import PhiAtom, PhiEdge, PhiGraph, PhiState

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "dumps" / "from_literature"

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def _slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip(), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return (s or "sec")[:n]


def _split_paragraphs(body: str) -> list[str]:
    parts = re.split(r"\n\s*\n", body.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_markdown(text: str, doc_id: str = "doc") -> list[dict]:
    """
    Zwraca listę bloków:
      {kind: main|section|chunk, level, title, body, id}
    """
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[dict] = []
    cur_level = 0
    cur_title = doc_id
    cur_id = f"{doc_id}/__main__"
    buf: list[str] = []
    sec_i = 0

    def flush():
        nonlocal buf
        body = "\n".join(buf).strip()
        buf = []
        if not body and cur_id.endswith("__main__"):
            return
        if cur_id.endswith("__main__"):
            blocks.append(
                {
                    "kind": "main",
                    "level": 0,
                    "title": cur_title,
                    "body": body or text[:2000],
                    "id": cur_id,
                }
            )
        else:
            # sekcja: tytuł + body; chunki z akapitów
            blocks.append(
                {
                    "kind": "section",
                    "level": cur_level,
                    "title": cur_title,
                    "body": body,
                    "id": cur_id,
                }
            )
            for j, para in enumerate(_split_paragraphs(body)):
                if len(para) < 40:
                    continue
                blocks.append(
                    {
                        "kind": "chunk",
                        "level": cur_level + 1,
                        "title": cur_title,
                        "body": para,
                        "id": f"{cur_id}/p{j}",
                        "parent": cur_id,
                    }
                )

    for line in lines:
        m = _HEADING.match(line)
        if m:
            flush()
            cur_level = len(m.group(1))
            cur_title = m.group(2).strip()
            sec_i += 1
            cur_id = f"{doc_id}/sec{sec_i}_{_slug(cur_title)}"
            buf = []
        else:
            buf.append(line)
    flush()

    if not any(b["kind"] == "main" for b in blocks):
        blocks.insert(
            0,
            {
                "kind": "main",
                "level": 0,
                "title": doc_id,
                "body": text[:8000],
                "id": f"{doc_id}/__main__",
            },
        )
    return blocks


def file_to_graph(path: Path, *, doc_id: str | None = None) -> PhiGraph:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    doc_id = doc_id or _slug(path.stem, 24)
    blocks = parse_markdown(text, doc_id=doc_id)

    atoms: list[PhiAtom] = []
    edges: list[PhiEdge] = []
    main_id = f"{doc_id}/__main__"
    section_ids: list[str] = []

    for b in blocks:
        kind = b["kind"]
        level = int(b["level"])
        depth = min(2, max(0, level if kind != "main" else 0))
        if kind == "chunk":
            depth = min(2, depth + 1)
        title = b["title"]
        body = b["body"]
        aid = b["id"]
        if kind == "main":
            state = PhiState.HOT
            T = 1.0
            payload = body
            E = f"doc:{title}"
            mazur = 0.9
        elif kind == "section":
            state = PhiState.HOT if level <= 1 else PhiState.WARM
            T = max(0.35, 1.0 - 0.2 * level)
            payload = body if body else None
            E = f"section:{title}"
            mazur = 0.5 + 0.1 * level
            section_ids.append(aid)
        else:
            # chunk niesie treść w E i w payload, więc stan jest HOT
            state = PhiState.HOT
            T = 0.55
            E = body[:240]
            payload = body
            mazur = 0.35

        atoms.append(
            PhiAtom(
                atom_id=aid,
                S=f"lit/{doc_id}/{kind}/L{level}",
                E=E,
                T=T,
                state=state,
                bubble_id=f"lit/{doc_id}",
                depth=depth,
                payload=payload,
                mazur_tracer=min(1.0, mazur),
            )
        )

    # edges
    for b in blocks:
        if b["kind"] == "section":
            edges.append(PhiEdge("bind", main_id, b["id"], name="outline"))
            edges.append(PhiEdge("parent", b["id"], main_id, name="in-doc"))
        if b["kind"] == "chunk":
            parent = b.get("parent") or main_id
            edges.append(PhiEdge("bind", parent, b["id"], name="para"))
            edges.append(PhiEdge("parent", b["id"], parent, name="in-sec"))
    for i in range(len(section_ids) - 1):
        edges.append(
            PhiEdge("bind", section_ids[i], section_ids[i + 1], name="next-sec")
        )

    return PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={
            "source": "literature",
            "path": str(path),
            "doc_id": doc_id,
            "n_sections": len(section_ids),
        },
    )


def paths_to_graph(paths: list[Path]) -> PhiGraph:
    atoms: list[PhiAtom] = []
    edges: list[PhiEdge] = []
    docs = []
    for p in paths:
        g = file_to_graph(p)
        atoms.extend(g.atoms)
        edges.extend(g.edges)
        docs.append(g.meta.get("doc_id"))
        # słaby bind między dokumentami (korpus)
    if len(docs) >= 2:
        mains = [f"{d}/__main__" for d in docs]
        for i in range(len(mains) - 1):
            edges.append(PhiEdge("bind", mains[i], mains[i + 1], name="corpus"))
    return PhiGraph(
        atoms=atoms,
        edges=edges,
        meta={"source": "literature", "docs": docs},
    )


def collect_paths(src: Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    exts = {".md", ".txt", ".markdown", ".rst"}
    files = sorted(
        p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise FileNotFoundError(f"brak .md/.txt w {src}")
    return files


def export_dump(src: Path, out_dir: Path) -> Path:
    paths = collect_paths(src)
    g = paths_to_graph(paths)
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
                        "topic": a.bubble_id,
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
    n_bind = sum(1 for e in g.edges if e.kind == "bind")
    man = {
        "format": "phijson-v1",
        "description": "literature .md/.txt → Φ",
        "n_atoms": len(g.atoms),
        "n_edges": len(g.edges),
        "n_bind": n_bind,
        "docs": g.meta.get("docs"),
        "source": "import_literature",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Literatura → PhiJSON")
    ap.add_argument("--src", type=Path, required=True, help="plik .md/.txt lub katalog")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = export_dump(args.src, args.out)
    # summary
    n_atoms = sum(1 for _ in open(out / "atoms.jsonl", encoding="utf-8"))
    print(f"literature {args.src} → {out}  atoms={n_atoms}")


if __name__ == "__main__":
    main()
