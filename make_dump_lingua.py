"""
Dump karmazyn_lingua (warstwa lingwistyczna: morfem hologram/HRR -> słowo)
-> dumps/from_lingua/

  python make_dump_lingua.py
  python make_dump_lingua.py --encoding symbolic --out dumps/from_lingua_symbolic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmazyn_lingua import AtomEncoding, przyklad_pisac_rodzina, to_phi_atoms

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_lingua"


def write_dump(out: Path, *, encoding: AtomEncoding = AtomEncoding.HOLOGRAM) -> tuple[int, int, dict]:
    out.mkdir(parents=True, exist_ok=True)
    lingua, slowa = przyklad_pisac_rodzina(encoding)
    atoms, edges, meta = to_phi_atoms(lingua, slowa)
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    man = {
        "name": out.name,
        "version": "0.1",
        "method": "karmazyn_lingua",
        "description": "Karmazyn Lingua: morfem hologram/symbolic + HRR bind -> słowo -> Most",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "source": "make_dump_lingua.py",
        "encoding": encoding.value,
        **meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"OK {out} atoms={len(atoms)} edges={len(edges)} HOT={n_hot} encoding={encoding.value}")
    return len(atoms), len(edges), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DUMP)
    ap.add_argument("--encoding", choices=["hologram", "symbolic"], default="hologram")
    args = ap.parse_args()
    write_dump(args.out, encoding=AtomEncoding(args.encoding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
