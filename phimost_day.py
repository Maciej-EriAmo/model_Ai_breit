"""
U3 — jedna komenda dnia: eksport → ckpt → encode → top-k → epizod.

  python phimost_day.py
  python phimost_day.py --query main -k 5
  python phimost_day.py --skip-export
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE_BBL = ROOT / "dumps" / "karmazyn_live_bbl"
LIVE_PHI = ROOT / "dumps" / "from_karmazyn_live"
FALLBACK_PHI = ROOT / "dumps" / "textfile_phi"


def ensure_checkpoint():
    from checkpoint import DEFAULT_MOST

    if DEFAULT_MOST.is_file():
        return DEFAULT_MOST
    import smoke_checkpoint

    return smoke_checkpoint.train_and_save(epochs=8)


def ensure_live_dump(skip_export: bool) -> Path:
    if skip_export and (LIVE_PHI / "atoms.jsonl").is_file():
        return LIVE_PHI
    if skip_export and (FALLBACK_PHI / "atoms.jsonl").is_file():
        return FALLBACK_PHI

    scripts = ROOT / "archiwum" / "skrypty"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import export_karmazyn_live as ex
    from bubblefs_import import bubblefs_to_graph, export_graph_as_dump

    ex._setup_path()
    s, ids, bubbles, holo = ex.seed_store(rich=True)
    ex.store_to_bubblefs(s, ids, bubbles, LIVE_BBL, holo_groups=holo)
    _man, g = bubblefs_to_graph(LIVE_BBL)
    export_graph_as_dump(
        g, LIVE_PHI, description="phimost_day live Karmazyn→BubbleFS→Φ"
    )
    return LIVE_PHI


def resolve_query(bus, query: str, k: int):
    ids = [a.atom_id for a in bus.graph.atoms]  # type: ignore[union-attr]
    # live export: label bywany jako atom_id (main, ctx, …)
    for cand in (query, "main", "__main__", "ctx"):
        if cand in ids:
            return {"kind": "id", "value": cand}, bus.recall_by_id(cand, k=k)
    return {"kind": "text", "value": query}, bus.recall_by_text(query, k=k)


def main():
    ap = argparse.ArgumentParser(description="PhiMost day ritual")
    ap.add_argument("--query", default="main")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ckpt = ensure_checkpoint()
    dump = ensure_live_dump(args.skip_export)

    from memory_bus import MemoryBus, append_episode, hits_to_dicts

    bus = MemoryBus(ckpt)
    n = bus.ingest_dump(dump)
    bus.encode()
    qmeta, hits = resolve_query(bus, args.query, args.k)

    out = {
        "ok": True,
        "ckpt": str(ckpt),
        "dump": str(dump),
        "atoms": n,
        "query": qmeta,
        "hits": hits_to_dicts(hits),
        "seconds": round(time.time() - t0, 3),
    }
    ep = append_episode(
        {
            "kind": "day_ritual",
            "dump": str(dump),
            "query": out["query"],
            "hits": out["hits"],
            "ckpt": str(ckpt),
        }
    )
    out["episode"] = str(ep)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=" * 60)
        print("PHIMOST DAY")
        print(f"  dump   {out['dump']}  (atoms={out['atoms']})")
        print(f"  ckpt   {out['ckpt']}")
        print(f"  query {out['query']}")
        for i, h in enumerate(out["hits"], 1):
            print(
                f"  {i}. {h['atom_id']:16s}  {h['score']:+.4f}  "
                f"d={h['depth']} {h['state']:4s}  {h['E'][:48]}"
            )
        print(f"  episode {out['episode']}")
        print(f"  time    {out['seconds']}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
