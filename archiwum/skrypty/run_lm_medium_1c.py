"""#1c — medium causal LM Most vs Softmax (~3.3M params, CPU).

To nadal NIE jest gate 100–300M. To skalowanie toru #1b na medium.

  python archiwum/skrypty/run_lm_medium_1c.py
  python archiwum/skrypty/run_lm_medium_1c.py --steps 600 --seeds 0,1
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
    TinyLM,
    TinyLMConfig,
    build_char_corpus,
    config_dict,
    count_params,
    load_text_corpus,
    make_batches,
    train_lm,
)
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"


def run_arm(attn, cfg, batches, *, steps, lr, seed):
    torch.manual_seed(seed)
    model = TinyLM(cfg, attn=attn)
    n_params = count_params(model)
    t0 = time.perf_counter()
    curve = train_lm(
        model, batches, steps=steps, lr=lr, log_every=max(1, steps // 20)
    )
    wall = time.perf_counter() - t0
    return {
        "attn": attn,
        "n_params": n_params,
        "wall_s": wall,
        "steps": steps,
        "loss_first": curve[0]["loss"] if curve else float("nan"),
        "loss_final": curve[-1]["loss"] if curve else float("nan"),
        "ppl_final": curve[-1]["ppl"] if curve else float("nan"),
        "curve": curve,
        "sec_per_step": wall / max(1, steps),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cfg = TinyLMConfig.preset("medium")
    texts = load_text_corpus(ROOT, repeat=3)
    data = build_char_corpus(texts, cfg.vocab_size)
    n_params = count_params(TinyLM(cfg, attn="most"))
    if not (2_000_000 <= n_params <= 5_000_000):
        print(f"WARN: params={n_params} poza oknem 2–5M (kontynuuję)")

    rundir = RunDir.create(
        "lm_medium_1c",
        argv=sys.argv,
        hparams={
            "preset": "medium",
            "cfg": config_dict(cfg),
            "steps": args.steps,
            "seeds": seeds,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "lr": args.lr,
            "data_bytes": int(data.numel()),
            "n_params": n_params,
        },
        extra_meta={"criterion": "success#1c_medium_lm"},
    )
    print(
        f"#1c medium LM d={cfg.d} L={cfg.num_layers} params={n_params} "
        f"data_bytes={data.numel()} steps={args.steps} seeds={seeds}\n"
        f"run={rundir.path}",
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
            seed=2000 + seed,
        )
        print(f"\n=== seed={seed} ===", flush=True)
        most = run_arm("most", cfg, batches, steps=args.steps, lr=args.lr, seed=seed)
        print(
            f"  Most   {most['loss_first']:.3f}→{most['loss_final']:.3f} "
            f"ppl={most['ppl_final']:.2f} wall={most['wall_s']:.1f}s",
            flush=True,
        )
        sm = run_arm("softmax", cfg, batches, steps=args.steps, lr=args.lr, seed=seed)
        print(
            f"  Softmax {sm['loss_first']:.3f}→{sm['loss_final']:.3f} "
            f"ppl={sm['ppl_final']:.2f} wall={sm['wall_s']:.1f}s",
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
        "delta_sm_minus_most": sum(r["delta_final_loss_sm_minus_most"] for r in rows) / n,
        "time_ratio": sum(r["time_ratio_most_over_sm"] for r in rows) / n,
        "n_params_most": rows[0]["most"]["n_params"],
        "n_params_sm": rows[0]["softmax"]["n_params"],
    }
    both_descend = all(
        r["most"]["loss_final"] < r["most"]["loss_first"]
        and r["softmax"]["loss_final"] < r["softmax"]["loss_first"]
        for r in rows
    )
    verdict = (
        f"#1c medium LM: params={agg['n_params_most']} | "
        f"mean loss Most={agg['most_loss_final']:.3f} SM={agg['sm_loss_final']:.3f} "
        f"Δ(SM−Most)={agg['delta_sm_minus_most']:+.3f} | "
        f"time≈{agg['time_ratio']:.2f}× | descend={both_descend}"
    )

    curves = {str(r["seed"]): r.pop("curves") for r in rows}
    report = {
        "protocol": "Medium causal LM Most vs Softmax — kryterium #1c",
        "preset": "medium",
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
            "Skala medium (~3M), nie gate 100–300M vs Llama. "
            "Ten sam korpus/pipeline co #1b; #3 VRAM nadal wymaga NVIDIA."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "LM_MEDIUM_1C_results.json"
    mp = OUT_DIR / "LM_MEDIUM_1C.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (rundir.path / "curves_by_seed.json").write_text(
        json.dumps(curves, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Medium LM — kryterium #1c (~2–5M params)",
        "",
        f"preset=medium · d={cfg.d} L={cfg.num_layers} h={cfg.h} vocab={cfg.vocab_size}",
        f"seq={args.seq_len} batch={args.batch_size} steps={args.steps} seeds={seeds}",
        f"params Most/SM = **{agg['n_params_most']}** / **{agg['n_params_sm']}**",
        f"data_bytes = {data.numel()}",
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
        f"- mean Δ(SM−Most) = **{agg['delta_sm_minus_most']:+.3f}**",
        f"- mean time ratio = **{agg['time_ratio']:.2f}×**",
        f"- both descend = **{both_descend}**",
        "",
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `LM_MEDIUM_1C_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    c0 = curves.get(str(seeds[0]), {}).get("most") or []
    rundir.write_curve(
        [{"epoch": row["step"], "loss": row["loss"], "r@1": "", "r@3": "", "mrr": ""} for row in c0]
    )
    rundir.finalize(
        metrics=report,
        summary="\n".join(lines),
        status="ok" if both_descend else "weak",
    )
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
