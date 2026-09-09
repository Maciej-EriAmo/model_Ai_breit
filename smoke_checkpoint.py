"""
P1.3–P1.4 — train (jeśli brak) → save → load → recall na dumpie bez re-treningu.

Użycie:
  python smoke_checkpoint.py           # load lub train+load
  python smoke_checkpoint.py --force   # zawsze przeucz Most i nadpisz ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from checkpoint import DEFAULT_MOST, load_encoder, save_encoder
from config import CANON
from dump_recall import make_dump_loaders
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, recall_metrics, train_encoder

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "textfile_phi"


def ensure_dump(dump: Path = DUMP):
    if not (dump / "atoms.jsonl").exists():
        if dump == DUMP:
            import make_dump_textfile

            make_dump_textfile.main()
        else:
            raise FileNotFoundError(dump)


def train_and_save(epochs: int = 10, dump: Path = DUMP) -> Path:
    ensure_dump(dump)
    n = max(4, min(16, sum(1 for _ in open(dump / "atoms.jsonl", encoding="utf-8"))))
    g, train, val, _ = make_dump_loaders(
        dump,
        n_train=60,
        n_val=20,
        batch_size=n,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
    )
    torch.manual_seed(42)
    most = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )
    metrics = train_encoder(most, train, val, epochs=epochs)
    path = save_encoder(
        most,
        DEFAULT_MOST,
        kind="most",
        meta={
            "task": "dump_recall",
            "source": "smoke_checkpoint",
            "dump": str(dump),
            "atoms": len(g.atoms),
            "r@1": metrics["r@1"],
            "epochs": epochs,
        },
    )
    print(f"trained Most r@1={metrics['r@1']:.3f} → {path}")
    return path


def eval_loaded(path: Path, dump: Path = DUMP) -> dict:
    ensure_dump(dump)
    model, meta = load_encoder(path)
    n = max(4, min(16, sum(1 for _ in open(dump / "atoms.jsonl", encoding="utf-8"))))
    _, train, val, val_clean = make_dump_loaders(
        dump,
        n_train=40,
        n_val=15,
        batch_size=n,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
    )
    # Softmax twin = świadek (wolno trenować); Most tylko load — bez re-treningu
    torch.manual_seed(0)
    sm = PhiSoftmaxEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
    )
    sm_m = train_encoder(sm, train, val, epochs=8)

    metrics = []
    with torch.no_grad():
        for batch in val:
            tok, _ = model(batch.content, batch.tracer, batch.mask)
            metrics.append(
                recall_metrics(
                    tok, batch.query_idx, batch.pos_idx, mask=batch.mask
                )
            )
        tok_c, _ = model(val_clean.content, val_clean.tracer, val_clean.mask)
        clean = recall_metrics(
            tok_c, val_clean.query_idx, val_clean.pos_idx, mask=val_clean.mask
        )

    keys = metrics[0].keys()
    avg = {k: float(sum(m[k] for m in metrics) / len(metrics)) for k in keys}
    avg["clean_r@1"] = clean["r@1"]
    avg["softmax_r@1"] = sm_m["r@1"]
    avg["delta_sm"] = avg["r@1"] - sm_m["r@1"]
    avg["meta_r@1"] = float(meta.get("r@1", float("nan")))
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="przeucz i nadpisz ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument(
        "--dump",
        type=Path,
        default=DUMP,
        help="PhiJSON do train/eval (lab textfile_phi albo from_karmazyn_live)",
    )
    args = ap.parse_args()

    if args.force or not DEFAULT_MOST.is_file():
        train_and_save(epochs=args.epochs, dump=args.dump)
    else:
        print(f"używam istniejącego {DEFAULT_MOST}")

    m = eval_loaded(DEFAULT_MOST, dump=args.dump)
    print("=" * 60)
    print("SMOKE checkpoint → recall (bez re-treningu Most)")
    print(f"  Most load  r@1={m['r@1']:.3f}  clean={m['clean_r@1']:.3f}")
    print(f"  Softmax    r@1={m['softmax_r@1']:.3f}  (świadek wytrenowany teraz)")
    print(f"  Δ Most−SM  {m['delta_sm']:+.3f}")
    print(f"  meta ckpt  r@1={m['meta_r@1']}")
    # Kryterium smoke: load działa + Most nie jest martwy (r@1 > Softmax losowego poziomu)
    ok_load = m["r@1"] == m["r@1"]  # finite
    ok_signal = m["r@1"] >= 0.15
    print(f"WERDYKT: {'PASS' if ok_load and ok_signal else 'FAIL'}")
    if not (ok_load and ok_signal):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
