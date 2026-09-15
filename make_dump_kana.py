"""
Dump kana_corpus (71 tożsamości dźwiękowych hiragana/katakana/romaji,
bez kanji) -> dumps/from_kana_p1/<encoding>/. Analog make_dump_lingua.py,
generalizowany na kana_corpus.przyklad_kana_korpus zamiast
przyklad_pisac_rodzina.

  python make_dump_kana.py
  python make_dump_kana.py --encoding symbolic --out dumps/from_kana_symbolic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmazyn_lingua import AtomEncoding, to_phi_atoms
from kana_corpus import przyklad_kana_korpus

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_kana_p1"


def write_dump(out: Path, *, encoding: AtomEncoding = AtomEncoding.HOLOGRAM) -> tuple[int, int, dict]:
    out.mkdir(parents=True, exist_ok=True)
    lingua, slowa = przyklad_kana_korpus(encoding)
    atoms, edges, meta = to_phi_atoms(
        lingua, slowa,
        page_id="kana_page",
        corpus_note="kana_corpus: 71 tozsamosci dzwiekowych x hiragana/katakana/romaji (bez kanji)",
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
        "version": "0.1",
        "method": "kana_corpus",
        "description": "Kana corpus: 71 sound identities x hiragana/katakana/romaji Hepburn -> Most",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "source": "make_dump_kana.py",
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
