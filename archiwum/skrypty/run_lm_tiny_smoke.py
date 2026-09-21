"""#1b smoke — tiny causal LM Most vs Softmax (≪1M params, CPU).

  python archiwum/skrypty/run_lm_tiny_smoke.py
  python archiwum/skrypty/run_lm_tiny_smoke.py --steps 300 --seeds 0,1
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
from pathlib import Path

import torch

from lm_tiny import (
    DEFAULT_TEXTS,
    TinyLM,
    TinyLMConfig,
    build_char_corpus,
    config_dict,
    count_params,
    make_batches,
    train_lm,
)
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"


def run_arm(
    attn: str,
    cfg: TinyLMConfig,
    batches: list[torch.Tensor],
    *,
    steps: int,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = TinyLM(cfg, attn=attn)
    n_params = count_params(model)
    t0 = time.perf_counter()
    curve = train_lm(model, batches, steps=steps, lr=lr, log_every=max(1, steps // 15))
    wall = time.perf_counter() - t0
    final = curve[-1]["loss"] if curve else float("nan")
    first = curve[0]["loss"] if curve else float("nan")
    return {
        "attn": attn,
        "n_params": n_params,
        "wall_s": wall,
        "steps": steps,
        "loss_first": first,
        "loss_final": final,
        "ppl_final": curve[-1]["ppl"] if curve else float("nan"),
        "curve": curve,
        "sec_per_step": wall / max(1, steps),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cfg = TinyLMConfig(
        vocab_size=128,
        d=args.d,
        h=2,
        num_layers=args.layers,
        n_dim=4,
        max_seq=max(args.seq_len, 128),
    )
    data = build_char_corpus(DEFAULT_TEXTS, cfg.vocab_size)
    # probe params
    probe = TinyLM(cfg, attn="most")
    n_most = count_params(probe)
    del probe
    if n_most >= 1_000_000:
        raise SystemExit(f"FAIL smoke: params={n_most} nie jest ≪1M")

    rundir = RunDir.create(
        "lm_tiny_smoke_1b",
        argv=sys.argv,
        hparams={
            "cfg": config_dict(cfg),
            "steps": args.steps,
            "seeds": seeds,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "lr": args.lr,
            "data_bytes": int(data.numel()),
        },
        extra_meta={"criterion": "success#1b_tiny_lm"},
    )
    print(
        f"#1b tiny LM smoke d={cfg.d} L={cfg.num_layers} params≈{n_most} "
        f"steps={args.steps} seeds={seeds} run={rundir.path}",
        flush=True,
    )

    t_all = time.time()
    rows = []
    for seed in seeds:
        batches = make_batches(
            data,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            seed=1000 + seed,
        )
        print(f"\n=== seed={seed} ===", flush=True)
        most = run_arm("most", cfg, batches, steps=args.steps, lr=args.lr, seed=seed)
        print(
            f"  Most   loss {most['loss_first']:.3f}→{most['loss_final']:.3f} "
            f"ppl={most['ppl_final']:.2f} wall={most['wall_s']:.1f}s params={most['n_params']}",
            flush=True,
        )
        sm = run_arm("softmax", cfg, batches, steps=args.steps, lr=args.lr, seed=seed)
        print(
            f"  Softmax loss {sm['loss_first']:.3f}→{sm['loss_final']:.3f} "
            f"ppl={sm['ppl_final']:.2f} wall={sm['wall_s']:.1f}s params={sm['n_params']}",
            flush=True,
        )
        rows.append(
            {
                "seed": seed,
                "most": {k: v for k, v in most.items() if k != "curve"},
                "softmax": {k: v for k, v in sm.items() if k != "curve"},
                "delta_final_loss_sm_minus_most": sm["loss_final"] - most["loss_final"],
                "time_ratio_most_over_sm": most["wall_s"] / max(1e-9, sm["wall_s"]),
                "curves": {"most": most["curve"], "softmax": sm["curve"]},
            }
        )

    n = len(rows)
    agg = {
        "most_loss_final": sum(r["most"]["loss_final"] for r in rows) / n,
        "sm_loss_final": sum(r["softmax"]["loss_final"] for r in rows) / n,
        "most_ppl": sum(r["most"]["ppl_final"] for r in rows) / n,
        "sm_ppl": sum(r["softmax"]["ppl_final"] for r in rows) / n,
        "delta_sm_minus_most": sum(r["delta_final_loss_sm_minus_most"] for r in rows) / n,
        "time_ratio": sum(r["time_ratio_most_over_sm"] for r in rows) / n,
        "n_params_most": rows[0]["most"]["n_params"],
        "n_params_sm": rows[0]["softmax"]["n_params"],
    }
    # smoke gate: obie krzywe spadają (final < first) — tor działa
    both_descend = all(
        r["most"]["loss_final"] < r["most"]["loss_first"]
        and r["softmax"]["loss_final"] < r["softmax"]["loss_first"]
        for r in rows
    )
    verdict = (
        f"#1b tiny LM: params Most/SM={agg['n_params_most']}/{agg['n_params_sm']} "
        f"(≪1M) | mean loss Most={agg['most_loss_final']:.3f} SM={agg['sm_loss_final']:.3f} "
        f"Δ(SM−Most)={agg['delta_sm_minus_most']:+.3f} | "
        f"time Most/SM≈{agg['time_ratio']:.2f}× | descend={both_descend}"
    )

    curves = {str(r["seed"]): r.pop("curves") for r in rows}
    report = {
        "protocol": "Tiny causal LM Most vs Softmax — kryterium #1b smoke",
        "cfg": config_dict(cfg),
        "steps": args.steps,
        "seeds": seeds,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "data_bytes": int(data.numel()),
        "rows": rows,
        "aggregate": agg,
        "both_curves_descend": both_descend,
        "verdict": verdict,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
        "note": (
            "To NIE jest jeszcze gate 100M–300M vs Llama. "
            "Smoke = ten sam data/pipeline, mierzalna krzywa loss Most vs Softmax na CPU."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "LM_TINY_1B_results.json"
    mp = OUT_DIR / "LM_TINY_1B.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (rundir.path / "curves_by_seed.json").write_text(
        json.dumps(curves, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Tiny LM smoke — kryterium #1b (≪1M params)",
        "",
        f"d={cfg.d} L={cfg.num_layers} h={cfg.h} vocab={cfg.vocab_size} "
        f"seq={args.seq_len} steps={args.steps} seeds={seeds}",
        f"params Most/SM = **{agg['n_params_most']}** / **{agg['n_params_sm']}**",
        "",
        f"**Werdykt:** {verdict}",
        "",
        report["note"],
        "",
        "| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |",
        "|-----:|--------------|------------|-----------:|-------:|-----:|------:|",
    ]
    for r in rows:
        m, s = r["most"], r["softmax"]
        lines.append(
            f"| {r['seed']} | {m['loss_first']:.3f}→{m['loss_final']:.3f} | "
            f"{s['loss_first']:.3f}→{s['loss_final']:.3f} | "
            f"{r['delta_final_loss_sm_minus_most']:+.3f} | "
            f"{m['wall_s']:.1f} | {s['wall_s']:.1f} | {r['time_ratio_most_over_sm']:.2f} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- mean final loss Most/SM = **{agg['most_loss_final']:.3f}** / **{agg['sm_loss_final']:.3f}**",
        f"- mean Δ(SM−Most) = **{agg['delta_sm_minus_most']:+.3f}** (dodatnie → Most niższy loss)",
        f"- mean time ratio Most/SM = **{agg['time_ratio']:.2f}×**",
        f"- both descend = **{both_descend}**",
        "",
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `LM_TINY_1B_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    # loss.csv = Most seed0 for run-dir convenience
    c0 = curves.get(str(seeds[0]), {}).get("most") or []
    rundir.write_curve(
        [{"epoch": row["step"], "loss": row["loss"], "r@1": "", "r@3": "", "mrr": ""} for row in c0]
    )
    rundir.finalize(metrics=report, summary="\n".join(lines), status="ok" if both_descend else "weak")
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
