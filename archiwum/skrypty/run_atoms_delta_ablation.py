"""Warianty zamka Most (ΔE) na secie atomów FULL118.

dump=from_mendelejew_full · Softmax + square/directed/onesided/ultra_wide/ultra_wide_directed
metryki: r@1, r@3, mrr, wall_s

  python archiwum/skrypty/run_atoms_delta_ablation.py
  python archiwum/skrypty/run_atoms_delta_ablation.py --seeds 0,1 --epochs 16
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
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_mendelejew_full"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

VARIANTS: list[tuple[str, dict]] = [
    ("square", dict(delta_mode="square")),
    ("directed", dict(delta_mode="directed", delta_alpha=1.0, directed_bias=0.25)),
    ("onesided", dict(delta_mode="onesided", delta_alpha=1.0)),
    (
        "ultra_wide",
        dict(gamma_init=2.5, e_scale=0.1, couple_temp=2.0, delta_mode="square"),
    ),
    (
        "ultra_wide_directed",
        dict(
            gamma_init=2.5,
            e_scale=0.1,
            couple_temp=2.0,
            delta_mode="directed",
            delta_alpha=1.0,
            directed_bias=0.25,
        ),
    ),
]


def _train(factory, train, val, *, seed: int, epochs: int) -> dict:
    torch.manual_seed(seed)
    model = factory()
    t0 = time.perf_counter()
    m = train_encoder(model, train, val, epochs=epochs, lr=CANON.lr, return_curve=False)
    wall = time.perf_counter() - t0
    return {
        "r@1": m["r@1"],
        "r@3": m["r@3"],
        "mrr": m["mrr"],
        "wall_s": wall,
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=DUMP)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    args = ap.parse_args(argv)

    if not (args.dump / "atoms.jsonl").is_file():
        raise SystemExit(f"brak dumpa {args.dump}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rundir = RunDir.create(
        "atoms_delta_ablation",
        argv=sys.argv,
        hparams={
            "dump": str(args.dump),
            "seeds": seeds,
            "epochs": args.epochs,
            "variants": [n for n, _ in VARIANTS],
            "canon": asdict(CANON),
        },
        extra_meta={"criterion": "atoms_signed_delta"},
    )
    print(
        f"Atoms ΔE ablation dump={args.dump} seeds={seeds} epochs={args.epochs} "
        f"run={rundir.path}",
        flush=True,
    )

    t_all = time.time()
    rows = []
    for seed in seeds:
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
            seed0=1000 + seed * 17,
        )
        print(f"\n=== seed={seed} N≈{n_atoms} ===", flush=True)

        print("  [Softmax] …", flush=True)
        sm = _train(
            lambda: PhiSoftmaxEncoder(
                input_dim=CANON.d_x,
                dim_tracer=CANON.n_dim,
                num_layers=CANON.num_layers,
                d=CANON.d_model,
            ),
            train,
            val,
            seed=seed,
            epochs=args.epochs,
        )
        print(f"    r@1={sm['r@1']:.3f} r@3={sm['r@3']:.3f} wall={sm['wall_s']:.1f}s", flush=True)

        seed_row = {"seed": seed, "softmax": sm, "most": {}}
        for name, kw in VARIANTS:
            print(f"  [Most:{name}] …", flush=True)

            def factory(kw=kw):
                return PhiBridgeEncoder(
                    input_dim=CANON.d_x,
                    dim_tracer=CANON.n_dim,
                    num_layers=CANON.num_layers,
                    d=CANON.d_model,
                    coupling=CANON.coupling,
                    **kw,
                )

            m = _train(factory, train, val, seed=seed, epochs=args.epochs)
            m["delta_sm_r1"] = m["r@1"] - sm["r@1"]
            m["delta_sm_r3"] = m["r@3"] - sm["r@3"]
            seed_row["most"][name] = m
            print(
                f"    r@1={m['r@1']:.3f} r@3={m['r@3']:.3f} "
                f"ΔSM={m['delta_sm_r1']:+.3f} wall={m['wall_s']:.1f}s",
                flush=True,
            )
        rows.append(seed_row)

    # aggregate
    names = [n for n, _ in VARIANTS]
    agg = {
        "softmax_r1": sum(r["softmax"]["r@1"] for r in rows) / len(rows),
        "softmax_r3": sum(r["softmax"]["r@3"] for r in rows) / len(rows),
        "most": {},
    }
    for name in names:
        agg["most"][name] = {
            "r@1": sum(r["most"][name]["r@1"] for r in rows) / len(rows),
            "r@3": sum(r["most"][name]["r@3"] for r in rows) / len(rows),
            "delta_sm_r1": sum(r["most"][name]["delta_sm_r1"] for r in rows) / len(rows),
            "delta_sm_r3": sum(r["most"][name]["delta_sm_r3"] for r in rows) / len(rows),
            "wall_s": sum(r["most"][name]["wall_s"] for r in rows) / len(rows),
        }

    best_name = max(names, key=lambda n: agg["most"][n]["r@1"])
    best = agg["most"][best_name]
    square = agg["most"]["square"]
    gate = CANON.gate_recall_r1
    n_gate = {
        n: sum(1 for r in rows if r["most"][n]["delta_sm_r1"] >= gate) for n in names
    }

    if best["delta_sm_r1"] >= gate and best["r@1"] > square["r@1"] + 0.005:
        verdict = (
            f"PASS na atomach: **{best_name}** mean r@1={best['r@1']:.3f} "
            f"ΔSM={best['delta_sm_r1']:+.3f} (gate≥{gate}) · bije square "
            f"({square['r@1']:.3f})."
        )
        status = "pass"
    elif best["r@1"] > square["r@1"] + 0.005 or best["delta_sm_r1"] > square["delta_sm_r1"] + 0.005:
        verdict = (
            f"PARTIAL: **{best_name}** lepszy niż square na FULL118 "
            f"(r@1 {best['r@1']:.3f} vs {square['r@1']:.3f}, "
            f"ΔSM {best['delta_sm_r1']:+.3f} vs {square['delta_sm_r1']:+.3f}); "
            f"gate vs SM: {n_gate[best_name]}/{len(rows)}."
        )
        status = "partial"
    else:
        verdict = (
            f"FAIL/neutral: warianty ΔE nie biją wyraźnie square na FULL118 "
            f"(best {best_name} r@1={best['r@1']:.3f}, square={square['r@1']:.3f}, "
            f"SM={agg['softmax_r1']:.3f})."
        )
        status = "neutral"

    report = {
        "protocol": "FULL118 atoms — Most ΔE variants vs Softmax",
        "dump": str(args.dump),
        "epochs": args.epochs,
        "seeds": seeds,
        "variants": names,
        "rows": rows,
        "aggregate": agg,
        "gate_r1": gate,
        "n_gate_pass": n_gate,
        "verdict": verdict,
        "status": status,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "ATOMS_DELTA_ABLATION_results.json"
    mp = OUT_DIR / "ATOMS_DELTA_ABLATION.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Atomy FULL118 — ablacja ΔE (square / directed / onesided / ultra_wide)",
        "",
        f"dump=`{args.dump}` · epochs={args.epochs} · seeds={seeds}",
        "",
        f"**Werdykt:** {verdict}",
        "",
        "## Aggregate",
        "",
        f"Softmax mean r@1/r@3 = **{agg['softmax_r1']:.3f}** / **{agg['softmax_r3']:.3f}**",
        "",
        "| variant | r@1 | r@3 | ΔSM r@1 | ΔSM r@3 | gate | wall s |",
        "|---------|----:|----:|--------:|--------:|-----:|-------:|",
    ]
    lines.append(
        f"| softmax | {agg['softmax_r1']:.3f} | {agg['softmax_r3']:.3f} | — | — | — | — |"
    )
    for n in names:
        a = agg["most"][n]
        lines.append(
            f"| {n} | {a['r@1']:.3f} | {a['r@3']:.3f} | {a['delta_sm_r1']:+.3f} | "
            f"{a['delta_sm_r3']:+.3f} | {n_gate[n]}/{len(rows)} | {a['wall_s']:.1f} |"
        )
    lines += [
        "",
        "## Per seed",
        "",
    ]
    for r in rows:
        lines.append(f"### seed {r['seed']}")
        lines.append(
            f"SM r@1={r['softmax']['r@1']:.3f} r@3={r['softmax']['r@3']:.3f}"
        )
        for n in names:
            m = r["most"][n]
            lines.append(
                f"- {n}: r@1={m['r@1']:.3f} r@3={m['r@3']:.3f} ΔSM={m['delta_sm_r1']:+.3f}"
            )
        lines.append("")
    lines += [
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `ATOMS_DELTA_ABLATION_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    rundir.finalize(metrics=report, summary="\n".join(lines), status=status)
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
