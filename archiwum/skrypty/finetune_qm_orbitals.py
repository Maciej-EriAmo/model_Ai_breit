"""
Fine-tune Most na dumpie QM (powłoki / podpowłoki / orbitale).

  python qm_orbitals_data.py
  python finetune_qm_orbitals.py --no-warm
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QM_DUMP = ROOT / "dumps" / "qm_orbitals"
QM_CKPT = ROOT / "checkpoints" / "most_qm_orbitals.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,7,42")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--symbols", default="H,He,C,N,O,Ne,Na,Si,Fe,Cu")
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    if not args.skip_export or not (QM_DUMP / "atoms.jsonl").is_file():
        import qm_orbitals_data as qm

        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        qm.export_phi_dump(QM_DUMP, syms)
        print(f"exported {QM_DUMP}")

    # reuse live finetune machinery with QM dump paths
    import finetune_karmazyn_live as ft

    # monkeypatch paths for this run via argv
    # point LIVE_PHI to QM dump for this process
    ft.LIVE_PHI = QM_DUMP
    ft.DEFAULT_OUT = QM_CKPT

    # wstrzyknij restrict_same_prefix do make_dump_loaders przez wrapper
    _orig_loaders = ft.make_dump_loaders

    def _qm_loaders(*a, **kw):
        kw.setdefault("restrict_same_prefix", True)
        kw.setdefault("blend_holo_proto", 0.35)
        kw.setdefault("pair_mode", "mixed")
        return _orig_loaders(*a, **kw)

    ft.make_dump_loaders = _qm_loaders  # type: ignore[assignment]

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
        str(QM_CKPT),
    ]
    if args.no_warm:
        argv.append("--no-warm")
    if args.no_promote:
        argv.append("--no-promote")

    sys.argv = argv
    ft.main()


if __name__ == "__main__":
    main()
