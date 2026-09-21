"""Most vs Softmax vs ZERO na from_mendelejew_full — surowy zrzut (#1 / efektywność).

Te same seedy i dump; per arm: r@1, r@3, mrr, wall_s, peak_rss_mb.
VRAM: N/A na CPU (mierzony peak RSS procesu).

  python archiwum/skrypty/run_mendelejew_baselines_compare.py
  python archiwum/skrypty/run_mendelejew_baselines_compare.py --seeds 0,1,2,7,42 --epochs 16
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

import psutil
import torch

from config import CANON
from dump_recall import make_dump_loaders
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_mendelejew_full"
OUT_DIR = ROOT / "doswiadczenie" / "glify"
SEEDS_DEFAULT = [0, 1, 2, 7, 42]


def _peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _train_arm(factory, train, val, *, seed: int, epochs: int, ablate: dict | None = None):
    torch.manual_seed(seed)
    model = factory()
    rss0 = _peak_rss_mb()
    t0 = time.perf_counter()
    metrics = train_encoder(
        model,
        train,
        val,
        epochs=epochs,
        lr=CANON.lr,
        ablate_kw=ablate or {},
        return_curve=True,
    )
    wall = time.perf_counter() - t0
    rss1 = _peak_rss_mb()
    n_steps = max(1, epochs * max(1, len(train)))
    return {
        "r@1": metrics["r@1"],
        "r@3": metrics["r@3"],
        "mrr": metrics["mrr"],
        "wall_s": wall,
        "sec_per_epoch": wall / max(1, epochs),
        "sec_per_step_approx": wall / n_steps,
        "peak_rss_mb": max(rss0, rss1),
        "delta_rss_mb": rss1 - rss0,
        "curve": metrics.get("curve"),
        "n_train_batches": len(train),
        "n_val_batches": len(val),
    }


def run_seed(dump: Path, seed: int, epochs: int, n_train: int, n_val: int) -> dict:
    n_atoms = sum(1 for _ in (dump / "atoms.jsonl").open(encoding="utf-8"))
    bs = min(16, max(4, n_atoms))
    g, train, val, _ = make_dump_loaders(
        dump,
        n_train=n_train,
        n_val=n_val,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
    )
    print(f"\n=== seed={seed} N={len(g.atoms)} batches_train={len(train)} ===", flush=True)

    print("  [Most] …", flush=True)
    most = _train_arm(lambda: build_model(warm=False), train, val, seed=seed, epochs=epochs)
    print(
        f"    r@1={most['r@1']:.3f} r@3={most['r@3']:.3f} "
        f"wall={most['wall_s']:.1f}s rss={most['peak_rss_mb']:.0f}MB",
        flush=True,
    )

    print("  [Softmax] …", flush=True)
    sm = _train_arm(
        lambda: PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        ),
        train,
        val,
        seed=seed,
        epochs=epochs,
    )
    print(
        f"    r@1={sm['r@1']:.3f} r@3={sm['r@3']:.3f} "
        f"wall={sm['wall_s']:.1f}s rss={sm['peak_rss_mb']:.0f}MB",
        flush=True,
    )

    print("  [ZERO] …", flush=True)
    zero = _train_arm(
        lambda: PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
        ),
        train,
        val,
        seed=seed,
        epochs=epochs,
        ablate={"zero_tracer": True},
    )
    print(
        f"    r@1={zero['r@1']:.3f} r@3={zero['r@3']:.3f} "
        f"wall={zero['wall_s']:.1f}s rss={zero['peak_rss_mb']:.0f}MB",
        flush=True,
    )

    d_sm = most["r@1"] - sm["r@1"]
    d_r3 = most["r@3"] - sm["r@3"]
    gate = d_sm >= CANON.gate_recall_r1
    return {
        "seed": seed,
        "n_atoms": len(g.atoms),
        "most": {k: v for k, v in most.items() if k != "curve"},
        "softmax": {k: v for k, v in sm.items() if k != "curve"},
        "zero": {k: v for k, v in zero.items() if k != "curve"},
        "curves": {
            "most": most.get("curve"),
            "softmax": sm.get("curve"),
            "zero": zero.get("curve"),
        },
        "delta_sm_r1": d_sm,
        "delta_sm_r3": d_r3,
        "gate_pass": gate,
        "time_ratio_most_over_sm": most["wall_s"] / max(1e-9, sm["wall_s"]),
        "rss_ratio_most_over_sm": most["peak_rss_mb"] / max(1e-9, sm["peak_rss_mb"]),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    args = ap.parse_args(argv)

    if not (args.dump / "atoms.jsonl").is_file():
        raise SystemExit(f"brak dumpa {args.dump}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cuda = torch.cuda.is_available()
    t_all = time.time()
    rundir = RunDir.create(
        "mendelejew_baselines_compare",
        argv=sys.argv,
        hparams={
            "dump": str(args.dump),
            "seeds": seeds,
            "epochs": args.epochs,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "canon": asdict(CANON),
            "cuda": cuda,
        },
        extra_meta={"criterion": "success#1_baselines_compare"},
    )
    print(
        f"Baselines compare dump={args.dump} seeds={seeds} epochs={args.epochs} "
        f"cuda={cuda} run={rundir.path}",
        flush=True,
    )

    rows = []
    for seed in seeds:
        rows.append(
            run_seed(
                args.dump,
                seed,
                args.epochs,
                args.n_train,
                args.n_val,
            )
        )

    n = len(rows)
    mean = {
        "most_r1": sum(r["most"]["r@1"] for r in rows) / n,
        "most_r3": sum(r["most"]["r@3"] for r in rows) / n,
        "sm_r1": sum(r["softmax"]["r@1"] for r in rows) / n,
        "sm_r3": sum(r["softmax"]["r@3"] for r in rows) / n,
        "zero_r1": sum(r["zero"]["r@1"] for r in rows) / n,
        "zero_r3": sum(r["zero"]["r@3"] for r in rows) / n,
        "delta_sm_r1": sum(r["delta_sm_r1"] for r in rows) / n,
        "delta_sm_r3": sum(r["delta_sm_r3"] for r in rows) / n,
        "most_wall_s": sum(r["most"]["wall_s"] for r in rows) / n,
        "sm_wall_s": sum(r["softmax"]["wall_s"] for r in rows) / n,
        "zero_wall_s": sum(r["zero"]["wall_s"] for r in rows) / n,
        "most_peak_rss_mb": sum(r["most"]["peak_rss_mb"] for r in rows) / n,
        "sm_peak_rss_mb": sum(r["softmax"]["peak_rss_mb"] for r in rows) / n,
        "n_gate_pass": sum(1 for r in rows if r["gate_pass"]),
    }
    verdict = (
        f"Most mean r@1={mean['most_r1']:.3f} r@3={mean['most_r3']:.3f} | "
        f"SM r@1={mean['sm_r1']:.3f} r@3={mean['sm_r3']:.3f} | "
        f"ΔSM r@1={mean['delta_sm_r1']:+.3f} r@3={mean['delta_sm_r3']:+.3f} | "
        f"gate {mean['n_gate_pass']}/{n} | "
        f"wall Most/SM={mean['most_wall_s']:.1f}/{mean['sm_wall_s']:.1f}s | "
        f"RSS Most/SM={mean['most_peak_rss_mb']:.0f}/{mean['sm_peak_rss_mb']:.0f}MB"
    )

    report = {
        "protocol": "Mendelejew FULL118 — Most vs Softmax vs ZERO (baselines compare)",
        "dump": str(args.dump),
        "epochs": args.epochs,
        "seeds": seeds,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "cuda": cuda,
        "vram_note": (
            "CUDA niedostępne — VRAM N/A; proxy = peak RSS (MB) procesu + wall time."
            if not cuda
            else "CUDA OK — peak RSS nadal logowane; VRAM peak do dopięcia torch.cuda.max_memory_allocated."
        ),
        "gate_r1": CANON.gate_recall_r1,
        "rows": rows,
        "aggregate": mean,
        "verdict": verdict,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
    }

    # strip heavy curves from main JSON rows already done; keep curves only in run-dir
    curves_only = {str(r["seed"]): r.pop("curves") for r in rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "MENDELEJEW_BASELINES_COMPARE_results.json"
    mp = OUT_DIR / "MENDELEJEW_BASELINES_COMPARE.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (rundir.path / "curves_by_seed.json").write_text(
        json.dumps(curves_only, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Mendelejew FULL118 — Most vs Softmax vs ZERO",
        "",
        f"dump=`{args.dump}` · epochs={args.epochs} · seeds={seeds}",
        f"cuda={cuda} · {report['vram_note']}",
        f"gate Δ(Most−SM) r@1 ≥ {CANON.gate_recall_r1}",
        "",
        f"**Werdykt:** {verdict}",
        "",
        "## Tabela seedów",
        "",
        "| seed | Most r@1 | Most r@3 | SM r@1 | SM r@3 | ZERO r@1 | ΔSM r@1 | gate | Most s | SM s | Most RSS | SM RSS |",
        "|-----:|---------:|---------:|-------:|-------:|---------:|--------:|:----:|-------:|-----:|---------:|-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seed']} | {r['most']['r@1']:.3f} | {r['most']['r@3']:.3f} | "
            f"{r['softmax']['r@1']:.3f} | {r['softmax']['r@3']:.3f} | "
            f"{r['zero']['r@1']:.3f} | {r['delta_sm_r1']:+.3f} | "
            f"{'Y' if r['gate_pass'] else 'N'} | "
            f"{r['most']['wall_s']:.1f} | {r['softmax']['wall_s']:.1f} | "
            f"{r['most']['peak_rss_mb']:.0f} | {r['softmax']['peak_rss_mb']:.0f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- mean Most r@1/r@3 = **{mean['most_r1']:.3f}** / **{mean['most_r3']:.3f}**",
        f"- mean Softmax r@1/r@3 = **{mean['sm_r1']:.3f}** / **{mean['sm_r3']:.3f}**",
        f"- mean ZERO r@1/r@3 = **{mean['zero_r1']:.3f}** / **{mean['zero_r3']:.3f}**",
        f"- mean ΔSM r@1/r@3 = **{mean['delta_sm_r1']:+.3f}** / **{mean['delta_sm_r3']:+.3f}**",
        f"- gate PASS = **{mean['n_gate_pass']}/{n}**",
        f"- mean wall Most/SM/ZERO = {mean['most_wall_s']:.1f} / {mean['sm_wall_s']:.1f} / {mean['zero_wall_s']:.1f} s",
        f"- mean peak RSS Most/SM = {mean['most_peak_rss_mb']:.0f} / {mean['sm_peak_rss_mb']:.0f} MB",
        "",
        f"elapsed_total={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `MENDELEJEW_BASELINES_COMPARE_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    rundir.finalize(
        metrics={**report, "curves_file": "curves_by_seed.json"},
        summary="\n".join(lines),
        status="ok",
    )
    print("\n" + verdict)
    print(f"JSON → {jp}\nMD → {mp}")
    return report


if __name__ == "__main__":
    main()
