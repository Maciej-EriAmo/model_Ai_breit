"""
Dump Holo-Φ → dumps/from_holo/ (lub wariant P3)

  python make_dump_holo.py
  python make_dump_holo.py --no-root-phase --out dumps/from_holo_nophase
  python make_dump_holo.py --no-root-binds --out dumps/from_holo_nobinds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holo_phi import build_holo_graph

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_holo"


def write_dump(
    out: Path,
    *,
    root_shared_phase: bool = True,
    root_binds: bool = True,
) -> tuple[int, int, dict]:
    out.mkdir(parents=True, exist_ok=True)
    atoms, edges, meta = build_holo_graph(
        root_shared_phase=root_shared_phase,
        root_binds=root_binds,
    )
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    man = {
        "name": out.name,
        "version": "0.2",
        "method": "holo_phi",
        "description": "Holo-Φ: char=wave; word/root=interference → Most",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "source": "make_dump_holo.py",
        "canon": "PhiJSON; T≠XY; most_seal; A=freq; blend_holo on mazur phase",
        **{k: v for k, v in meta.items() if k != "family_glyph_ids"},
        "n_family_glyphs": len(meta.get("family_glyph_ids") or []),
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "family_ids.json").write_text(
        json.dumps(meta.get("family_glyph_ids") or [], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"OK {out} atoms={len(atoms)} edges={len(edges)} HOT={n_hot} "
        f"phase={root_shared_phase} binds={root_binds}"
    )
    return len(atoms), len(edges), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DUMP)
    ap.add_argument("--no-root-phase", action="store_true")
    ap.add_argument("--no-root-binds", action="store_true")
    args = ap.parse_args()
    write_dump(
        args.out,
        root_shared_phase=not args.no_root_phase,
        root_binds=not args.no_root_binds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
