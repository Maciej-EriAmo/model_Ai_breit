"""#1c — medium causal LM Most vs Softmax (~3.3M params, CPU).

To nadal NIE jest gate 100–300M. To skalowanie toru #1b na medium.

  python archiwum/skrypty/run_lm_medium_1c.py
  python archiwum/skrypty/run_lm_medium_1c.py --steps 600 --seeds 0,1
  python archiwum/skrypty/run_lm_medium_1c.py --arms most,softmax,beta0,excess

beta0: γ = γ0, bez szerokości zależnej od wiersza.
excess: płaski wiersz startuje od γ0. Para most,softmax zostaje protokołem #1c.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


def _parse_arms(text: str) -> list[str]:
    arms = [part.strip() for part in text.split(",") if part.strip()]
    known = ("most", "softmax", "beta0", "excess")
    bad = [name for name in arms if name not in known]
    if bad or not arms:
        raise SystemExit(f"ramiona {arms!r}; dozwolone: {', '.join(known)}")
    if "most" not in arms or "softmax" not in arms:
        raise SystemExit("para most,softmax zostaje; beta0 i excess są dodatkowe")
    return arms


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
    }, model


def load_byte_file(path: Path, vocab_size: int) -> torch.Tensor:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return build_char_corpus([raw], vocab_size)


@torch.no_grad()
def eval_lm(model: TinyLM, data: torch.Tensor, *, seq_len: int, batch_size: int) -> dict:
    """Średnia strata na całym pliku: okna seq_len bez zachodzenia na siebie."""
    model.eval()
    starts = list(range(0, data.numel() - seq_len + 1, seq_len))
    if not starts:
        raise ValueError(f"plik krótszy niż seq_len={seq_len}")
    total = 0.0
    tokens = 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        rows = torch.stack([data[s : s + seq_len] for s in chunk])
        logits = model(rows)
        total += float(
            F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                rows[:, 1:].reshape(-1),
                reduction="sum",
            )
        )
        tokens += rows[:, 1:].numel()
    loss = total / tokens
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "tokens": tokens,
        "windows": len(starts),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="plik tekstu zamiast notatek z repo (np. WikiText train)",
    )
    ap.add_argument("--valid", type=Path, default=None, help="plik valid, średnia strata po treningu")
    ap.add_argument("--test", type=Path, default=None, help="plik test, średnia strata po treningu")
    ap.add_argument(
        "--arms",
        default="most,softmax",
        help="most,softmax oraz opcjonalnie beta0,excess",
    )
    args = ap.parse_args(argv)
    arms = _parse_arms(args.arms)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    cfg = TinyLMConfig.preset("medium")
    if args.corpus is not None:
        raw = args.corpus.read_text(encoding="utf-8", errors="replace")
        data = build_char_corpus([raw], cfg.vocab_size)
        corpus_name = str(args.corpus)
        out_stem = "LM_MEDIUM_WIKITEXT_HOLDOUT" if (args.valid or args.test) else "LM_MEDIUM_WIKITEXT"
        run_name = "lm_medium_wikitext_holdout" if (args.valid or args.test) else "lm_medium_wikitext"
    else:
        texts = load_text_corpus(ROOT, repeat=3)
        data = build_char_corpus(texts, cfg.vocab_size)
        corpus_name = "docs+glify"
        out_stem = "LM_MEDIUM_1C"
        run_name = "lm_medium_1c"
    valid_data = load_byte_file(args.valid, cfg.vocab_size) if args.valid else None
    test_data = load_byte_file(args.test, cfg.vocab_size) if args.test else None
    n_params = count_params(TinyLM(cfg, attn="most"))
    if not (2_000_000 <= n_params <= 5_000_000):
        print(f"WARN: params={n_params} poza oknem 2–5M (kontynuuję)")

    rundir = RunDir.create(
        run_name,
        argv=sys.argv,
        hparams={
            "preset": "medium",
            "arms": arms,
            "cfg": config_dict(cfg),
            "corpus": corpus_name,
            "steps": args.steps,
            "seeds": seeds,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_batches": args.n_batches,
            "lr": args.lr,
            "data_bytes": int(data.numel()),
            "valid": str(args.valid) if args.valid else None,
            "test": str(args.test) if args.test else None,
            "valid_bytes": int(valid_data.numel()) if valid_data is not None else None,
            "test_bytes": int(test_data.numel()) if test_data is not None else None,
            "n_params": n_params,
        },
        extra_meta={"criterion": "success#1c_medium_lm"},
    )
    print(
        f"#1c medium LM d={cfg.d} L={cfg.num_layers} params={n_params} "
        f"data_bytes={data.numel()} steps={args.steps} seeds={seeds}\n"
        f"corpus={corpus_name}\n"
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
        ran: dict[str, tuple] = {}
        for name in arms:
            info, model = run_arm(name, cfg, batches, steps=args.steps, lr=args.lr, seed=seed)
            print(
                f"  {name:7} {info['loss_first']:.3f}→{info['loss_final']:.3f} "
                f"ppl={info['ppl_final']:.2f} wall={info['wall_s']:.1f}s",
                flush=True,
            )
            ran[name] = (info, model)
        most, _most_model = ran["most"]
        sm, _sm_model = ran["softmax"]
        for name, (arm, model) in ran.items():
            torch.save(model.state_dict(), rundir.path / f"{name}_seed{seed}.pt")
            if valid_data is not None:
                arm["valid"] = eval_lm(
                    model, valid_data, seq_len=args.seq_len, batch_size=args.batch_size
                )
            if test_data is not None:
                arm["test"] = eval_lm(
                    model, test_data, seq_len=args.seq_len, batch_size=args.batch_size
                )
        if "valid" in most:
            print(
                f"  valid Most={most['valid']['loss']:.3f} SM={sm['valid']['loss']:.3f} "
                f"Δ={sm['valid']['loss'] - most['valid']['loss']:+.3f}",
                flush=True,
            )
        if "test" in most:
            print(
                f"  test  Most={most['test']['loss']:.3f} SM={sm['test']['loss']:.3f} "
                f"Δ={sm['test']['loss'] - most['test']['loss']:+.3f}",
                flush=True,
            )
        extra = {
            name: {k: v for k, v in info.items() if k != "curve"}
            for name, (info, _model) in ran.items()
            if name not in ("most", "softmax")
        }
        rows.append(
            {
                "seed": seed,
                "most": {k: v for k, v in most.items() if k != "curve"},
                "softmax": {k: v for k, v in sm.items() if k != "curve"},
                "extra": extra,
                "delta_final_loss_sm_minus_most": sm["loss_final"] - most["loss_final"],
                "time_ratio_most_over_sm": most["wall_s"] / max(1e-9, sm["wall_s"]),
                "curves": {
                    name: info["curve"] for name, (info, _model) in ran.items()
                },
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
        "corpus": corpus_name,
        "data_bytes": int(data.numel()),
        "rows": rows,
        "aggregate": agg,
        "both_curves_descend": both_descend,
        "verdict": verdict,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
        "note": (
            "Skala medium (~3M), nie gate 100–300M vs Llama. "
            + (
                "Korpus z --corpus. Valid i test to średnia strata na całym pliku, okna bez zachodzenia."
                if (args.valid or args.test)
                else (
                    "Korpus z --corpus. Strata jest treningowa, nie na pliku valid/test."
                    if args.corpus is not None
                    else "Ten sam korpus/pipeline co #1b; #3 VRAM nadal wymaga NVIDIA."
                )
            )
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / f"{out_stem}_results.json"
    mp = OUT_DIR / f"{out_stem}.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (rundir.path / "curves_by_seed.json").write_text(
        json.dumps(curves, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Medium LM — kryterium #1c (~2–5M params)",
        "",
        f"preset=medium · d={cfg.d} L={cfg.num_layers} h={cfg.h} vocab={cfg.vocab_size}",
        f"seq={args.seq_len} batch={args.batch_size} steps={args.steps} seeds={seeds}",
        f"corpus={corpus_name}",
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
    ]
    if rows and "valid" in rows[0]["most"]:
        mv = sum(r["most"]["valid"]["loss"] for r in rows) / n
        sv = sum(r["softmax"]["valid"]["loss"] for r in rows) / n
        lines += [
            "## Valid",
            "",
            f"- mean loss Most/SM = **{mv:.3f}** / **{sv:.3f}**",
            f"- mean Δ(SM−Most) = **{sv - mv:+.3f}**",
            "",
            "| seed | Most | SM | Δ(SM−Most) | ppl Most | ppl SM |",
            "|-----:|-----:|---:|-----------:|---------:|-------:|",
        ]
        for r in rows:
            m, s = r["most"]["valid"], r["softmax"]["valid"]
            lines.append(
                f"| {r['seed']} | {m['loss']:.3f} | {s['loss']:.3f} | "
                f"{s['loss'] - m['loss']:+.3f} | {m['ppl']:.2f} | {s['ppl']:.2f} |"
            )
        lines.append("")
    if rows and "test" in rows[0]["most"]:
        mt = sum(r["most"]["test"]["loss"] for r in rows) / n
        st = sum(r["softmax"]["test"]["loss"] for r in rows) / n
        lines += [
            "## Test",
            "",
            f"- mean loss Most/SM = **{mt:.3f}** / **{st:.3f}**",
            f"- mean Δ(SM−Most) = **{st - mt:+.3f}**",
            "",
            "| seed | Most | SM | Δ(SM−Most) | ppl Most | ppl SM |",
            "|-----:|-----:|---:|-----------:|---------:|-------:|",
        ]
        for r in rows:
            m, s = r["most"]["test"], r["softmax"]["test"]
            lines.append(
                f"| {r['seed']} | {m['loss']:.3f} | {s['loss']:.3f} | "
                f"{s['loss'] - m['loss']:+.3f} | {m['ppl']:.2f} | {s['ppl']:.2f} |"
            )
        lines.append("")
    extra_names = [name for name in arms if name not in ("most", "softmax")]
    if extra_names and rows:
        lines += [
            "## Szerokość wiersza",
            "",
            "beta0: γ = γ0 na każdym wierszu. excess: płaski wiersz startuje od γ0.",
            "To nie jest werdykt pary Most/Softmax.",
            "",
            "| seed | " + " | ".join(extra_names) + " |",
            "|-----:|" + "|".join("------:" for _ in extra_names) + "|",
        ]
        for r in rows:
            cells = " | ".join(
                f"{r['extra'][name]['loss_final']:.3f}" for name in extra_names
            )
            lines.append(f"| {r['seed']} | {cells} |")
        lines.append("")
    lines += [
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        f"JSON: `{jp.name}`",
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
