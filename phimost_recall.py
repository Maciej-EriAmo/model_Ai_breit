"""
CLI: recall na dumpie Φ przez MemoryBus (checkpoint Most).

  python phimost_recall.py --dump dumps/textfile_phi --id __main__ -k 5
  python phimost_recall.py --dump dumps/textfile_phi --text "InfoNCE bind" -k 5
  python phimost_recall.py --dump dumps/textfile_phi --id __main__ --log-episode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from checkpoint import DEFAULT_MOST
from memory_bus import MemoryBus, append_episode, hits_to_dicts

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="PhiMost recall (MemoryBus)")
    ap.add_argument(
        "--dump",
        type=Path,
        default=ROOT / "dumps" / "textfile_phi",
        help="katalog PhiJSON",
    )
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_MOST, help="checkpoint Most")
    ap.add_argument("--id", type=str, default=None, help="atom_id query")
    ap.add_argument("--text", type=str, default=None, help="query tekst → kotwica hash")
    ap.add_argument("--index", type=int, default=None, help="query index w dumpie")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--log-episode",
        action="store_true",
        help="dopisz epizod do dumps/episodes.jsonl",
    )
    ap.add_argument("--json", action="store_true", help="sam JSON na stdout")
    args = ap.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(
            f"brak checkpointu {args.ckpt} — uruchom: python smoke_checkpoint.py"
        )
    if not (args.dump / "atoms.jsonl").is_file():
        raise SystemExit(f"brak dumpa {args.dump}")

    bus = MemoryBus(args.ckpt)
    n = bus.ingest_dump(args.dump)
    bus.encode()

    if args.id:
        hits = bus.recall_by_id(args.id, k=args.k)
        q = {"kind": "id", "value": args.id}
    elif args.text:
        hits = bus.recall_by_text(args.text, k=args.k)
        q = {"kind": "text", "value": args.text}
    elif args.index is not None:
        hits = bus.recall_by_index(args.index, k=args.k)
        q = {"kind": "index", "value": args.index}
    else:
        # domyślnie __main__ albo 0
        ids = [a.atom_id for a in bus.graph.atoms]  # type: ignore[union-attr]
        qid = "__main__" if "__main__" in ids else ids[0]
        hits = bus.recall_by_id(qid, k=args.k)
        q = {"kind": "id", "value": qid}

    payload = {
        "dump": str(args.dump),
        "atoms": n,
        "query": q,
        "hits": hits_to_dicts(hits),
    }
    if args.log_episode:
        ep = append_episode(
            {
                "kind": "recall",
                "dump": str(args.dump),
                "query": q,
                "k": args.k,
                "hits": payload["hits"],
                "ckpt": str(args.ckpt),
            }
        )
        payload["episode"] = str(ep)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"dump={args.dump} atoms={n} query={q}")
        for i, h in enumerate(hits, 1):
            print(
                f"  {i}. {h.atom_id:20s}  score={h.score:+.4f}  "
                f"d={h.depth} {h.state:4s}  E={h.E[:50]}"
            )
        if args.log_episode:
            print(f"episode → {payload['episode']}")


if __name__ == "__main__":
    main()
