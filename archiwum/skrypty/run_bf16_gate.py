"""Gate kryterium #2 — BF16 Most bez NaN (smoke + dłuższy tor).

  python archiwum/skrypty/run_bf16_gate.py
  python archiwum/skrypty/run_bf16_gate.py --epochs 16 --dump dumps/from_mendelejew_full
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from config import CANON
from dump_recall import make_dump_loaders
from phimost_learn import build_model
from precision import bf16_supported
from recall_model import train_encoder
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"
DEFAULT_DUMP = ROOT / "dumps" / "from_mendelejew_full"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-val", type=int, default=20)
    args = ap.parse_args(argv)

    if not bf16_supported():
        print("FAIL: bf16 nieobsługiwane na tym hoście")
        return 2
    if not (args.dump / "atoms.jsonl").is_file():
        print(f"FAIL: brak dumpa {args.dump}")
        return 2

    t0 = time.time()
    torch.manual_seed(args.seed)
    n_atoms = sum(1 for _ in (args.dump / "atoms.jsonl").open(encoding="utf-8"))
    bs = min(16, max(4, n_atoms))
    _, train, val, _ = make_dump_loaders(
        args.dump,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + args.seed * 17,
    )
    model = build_model(warm=False)
    rundir = RunDir.create(
        "bf16_gate",
        argv=sys.argv,
        hparams={
            "dump": str(args.dump),
            "epochs": args.epochs,
            "seed": args.seed,
            "amp": "bfloat16",
            "canon": asdict(CANON),
        },
        extra_meta={"criterion": "success#2_bf16"},
    )
    print(f"BF16 gate dump={args.dump} epochs={args.epochs} run={rundir.path}")
    try:
        metrics = train_encoder(
            model,
            train,
            val,
            epochs=args.epochs,
            lr=CANON.lr,
            return_curve=True,
            amp_dtype=torch.bfloat16,
            finite_check=True,
        )
    except RuntimeError as e:
        rundir.finalize(
            metrics={"error": str(e)},
            summary=f"FAIL BF16 gate\n\n{e}\n",
            status="fail",
        )
        print(f"FAIL: {e}")
        return 1

    elapsed = time.time() - t0
    curve = metrics.get("curve") or []
    all_finite = all(
        torch.isfinite(torch.tensor([row["loss"], row["r@1"], row["r@3"], row["mrr"]])).all()
        for row in curve
    )
    ok = all_finite and torch.isfinite(torch.tensor(metrics["r@1"]))
    report = {
        "protocol": "BF16 gate — Most autocast bfloat16 + finite-check",
        "dump": str(args.dump),
        "epochs": args.epochs,
        "seed": args.seed,
        "r@1": metrics["r@1"],
        "r@3": metrics["r@3"],
        "mrr": metrics["mrr"],
        "amp_dtype": metrics.get("amp_dtype"),
        "curve_len": len(curve),
        "all_curve_finite": all_finite,
        "elapsed_s": elapsed,
        "run_dir": str(rundir.path),
        "verdict": "PASS" if ok else "FAIL",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "BF16_GATE_results.json"
    mp = OUT_DIR / "BF16_GATE.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mp.write_text(
        "\n".join(
            [
                "# BF16 gate — kryterium sukcesu #2",
                "",
                f"**Werdykt:** {'**PASS**' if ok else '**FAIL**'} — Most w autocast `bfloat16` "
                f"bez NaN/Inf (finite-check na S/C/ΔE/A/w/loss).",
                "",
                f"dump=`{args.dump}` · epochs={args.epochs} · seed={args.seed}",
                f"r@1={metrics['r@1']:.4f} · r@3={metrics['r@3']:.4f} · mrr={metrics['mrr']:.4f}",
                f"elapsed={elapsed:.1f}s · run-dir=`{rundir.path}`",
                "",
                "Lab nadal domyślnie FP32; gate #2 = ścieżka `--bf16` / ten skrypt.",
                "",
                "JSON: `BF16_GATE_results.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rundir.finalize(
        metrics=report,
        summary=mp.read_text(encoding="utf-8"),
        status="ok" if ok else "fail",
        curve=curve,
    )
    print(report["verdict"], f"r@1={metrics['r@1']:.4f}", f"→ {mp}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
