"""
CLI produktowe: train / propose E (gen head + Most).

  set PHIMOST_GEN=1
  python phimost_gen.py --train --dump dumps/textfile_phi
  python phimost_gen.py --id __main__ --dump dumps/textfile_phi
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from config import CANON
from dump_recall import make_dump_loaders
from gen_head import (
    DEFAULT_GEN,
    PhiGenHead,
    load_gen_bundle,
    propose_E,
    save_gen_bundle,
    train_gen,
)
from memory_bus import append_episode
from recall_model import PhiBridgeEncoder

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "textfile_phi"


def train_on_dump(dump: Path, epochs: int = 12) -> Path:
    dump = Path(dump)
    if not (dump / "atoms.jsonl").is_file():
        # Fixture lab jest tylko dla kanonicznego dumpa textfile. Inna ścieżka
        # nie może cicho zapisać dumps/textfile_phi i dalej czytać pusty --dump.
        if dump.resolve() == DUMP.resolve():
            import make_dump_textfile

            make_dump_textfile.main()
        if not (dump / "atoms.jsonl").is_file():
            raise FileNotFoundError(f"brak atoms.jsonl w {dump}")
    _, train, val, _ = make_dump_loaders(
        dump,
        n_train=60,
        n_val=20,
        batch_size=16,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
    )
    torch.manual_seed(42)
    enc = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )
    head = PhiGenHead(CANON.d_model, CANON.d_x, CANON.n_dim)
    metrics = train_gen(enc, head, train, val, epochs=epochs)
    path = save_gen_bundle(
        enc,
        head,
        DEFAULT_GEN,
        meta={
            "task": "gen_E",
            "dump": str(dump),
            "mse": metrics["mse"],
            "mse_zero": metrics["mse_zero"],
            "delta_zero": metrics["delta_zero"],
            "epochs": epochs,
        },
    )
    print(
        f"gen trained mse={metrics['mse']:.4f} zero={metrics['mse_zero']:.4f} "
        f"Δ={metrics['delta_zero']:+.4f} → {path}"
    )
    return path


def propose_for_id(dump: Path, atom_id: str, ckpt: Path) -> dict:
    from dump_recall import graph_to_tensors, load_raw_records, records_to_graph
    from phi_dump import load_edges_jsonl, warm_without_payload

    enc, head, meta = load_gen_bundle(ckpt)
    rows = load_raw_records(dump)
    edges = load_edges_jsonl(dump / "edges.jsonl")
    g = warm_without_payload(records_to_graph(rows, edges))
    ids = [a.atom_id for a in g.atoms]
    if atom_id not in ids:
        raise SystemExit(f"brak atomu {atom_id!r} w {dump}")
    idx = ids.index(atom_id)
    c, t, _, _ = graph_to_tensors(g, d_x=CANON.d_x, n_dim=CANON.n_dim)
    mask = torch.ones(c.shape[0])
    os.environ["PHIMOST_GEN"] = "1"
    vec = propose_E(enc, head, c, t, idx, mask=mask)
    target = c[idx]
    cos = float(
        torch.nn.functional.cosine_similarity(
            vec.unsqueeze(0), target.unsqueeze(0)
        ).item()
    )
    return {
        "atom_id": atom_id,
        "index": idx,
        "vec": [float(x) for x in vec.tolist()],
        "cosine_vs_content": cos,
        "meta_mse": meta.get("mse"),
        "delta_zero": meta.get("delta_zero"),
    }


def main():
    ap = argparse.ArgumentParser(description="PhiMost gen head (produkt)")
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_GEN)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--id", type=str, default="__main__")
    ap.add_argument("--log-episode", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.train or not args.ckpt.is_file():
        train_on_dump(args.dump, epochs=args.epochs)

    # propose wymaga flagi produktowej
    prev = os.environ.get("PHIMOST_GEN", "0")
    os.environ["PHIMOST_GEN"] = "1"
    try:
        out = propose_for_id(args.dump, args.id, args.ckpt)
    finally:
        os.environ["PHIMOST_GEN"] = prev

    if args.log_episode:
        ep = append_episode(
            {
                "kind": "gen_propose_E",
                "dump": str(args.dump),
                "atom_id": out["atom_id"],
                "cosine_vs_content": out["cosine_vs_content"],
                "ckpt": str(args.ckpt),
            }
        )
        out["episode"] = str(ep)

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(
            f"propose_E id={out['atom_id']} cos(content)={out['cosine_vs_content']:+.4f} "
            f"dim={len(out['vec'])}"
        )
        if out.get("episode"):
            print(f"episode → {out['episode']}")


if __name__ == "__main__":
    main()
