"""
Fine-tune Most na dumps/from_struktura (z Twojego CSV/JSON).

  python import_struktura_atomow.py
  python finetune_struktura_atomow.py --no-warm --no-promote
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_struktura"
CKPT = ROOT / "checkpoints" / "most_struktura_atomow.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,7,42")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    if not args.skip_import or not (DUMP / "atoms.jsonl").is_file():
        import import_struktura_atomow as imp

        src = args.src
        if src is None:
            src = (
                imp.DEFAULT_JSON
                if imp.DEFAULT_JSON.is_file()
                else imp.DEFAULT_CSV
            )
        rows = imp.load_rows(src)
        imp.export_dump(rows, DUMP)
        print(f"imported → {DUMP}")

    import finetune_karmazyn_live as ft

    ft.LIVE_PHI = DUMP
    ft.DEFAULT_OUT = CKPT
    _orig = ft.make_dump_loaders

    def _loaders(*a, **kw):
        kw.setdefault("restrict_same_prefix", True)
        kw.setdefault("blend_holo_proto", 0.35)
        kw.setdefault("pair_mode", "mixed")
        return _orig(*a, **kw)

    ft.make_dump_loaders = _loaders  # type: ignore[assignment]

    argv = [
        "finetune_karmazyn_live.py",
        "--skip-export",
        "--epochs",
        str(args.epochs),
        "--seeds",
        args.seeds,
        "--pair-mode",
        "mixed",
        "--blend-holo",
        "0.35",
        "--out",
        str(CKPT),
    ]
    if args.no_warm:
        argv.append("--no-warm")
    if args.no_promote:
        argv.append("--no-promote")
    sys.argv = argv
    ft.main()


if __name__ == "__main__":
    main()
