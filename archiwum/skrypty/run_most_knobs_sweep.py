"""Strojenie gałek zamka Most → ponowny mid-probe (odpowiedź na FAIL hipotezy).

Gałki: gamma_init, e_scale, couple_temp.
Cel: wyższa entropia mid / mniejszy mid_dip vs Softmax i vs baseline Most.

  python archiwum/skrypty/run_most_knobs_sweep.py
  python archiwum/skrypty/run_most_knobs_sweep.py --seq-len 2048 --train-steps 300
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))
# import probe helpers
sys.path.insert(0, str(_LAB / "archiwum" / "skrypty"))

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from lm_tiny import (
    MostKnobs,
    TinyLM,
    TinyLMConfig,
    build_char_corpus,
    count_params,
    load_text_corpus,
    with_most_knobs,
)
from run_lost_in_middle_probe import (
    mid_dip,
    probe_entropy,
    probe_needle_mass,
    train_arm,
)
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"

# baseline = to co FAIL-owało; potem warianty „grubszy ogon”
SWEEP: list[tuple[str, MostKnobs]] = [
    ("baseline", MostKnobs(gamma_init=0.0, e_scale=1.0, couple_temp=1.0)),
    ("wide_gamma", MostKnobs(gamma_init=1.5, e_scale=1.0, couple_temp=1.0)),  # większe g
    ("soft_energy", MostKnobs(gamma_init=0.0, e_scale=0.25, couple_temp=1.0)),  # słabsze ΔE
    ("fat_tail", MostKnobs(gamma_init=1.5, e_scale=0.25, couple_temp=1.5)),  # g↑ e↓ τ↑
    ("ultra_wide", MostKnobs(gamma_init=2.5, e_scale=0.1, couple_temp=2.0)),
]


def eval_arm(model: TinyLM, seq_len: int, seed: int, n_rep: int = 4) -> dict:
    model.eval()
    rels = [0.1, 0.25, 0.5, 0.75, 0.9]
    ent = probe_entropy(model, seq_len, n_samples=3, seed=seed + 7)
    needle = probe_needle_mass(model, seq_len, rels, n_repeat=n_rep)
    return {
        "entropy": ent,
        "needle": needle,
        "mid_dip_attn_mass": mid_dip(needle, "attn_mass_mean"),
        "mid_dip_secret_acc": mid_dip(needle, "secret_acc"),
        "secret_acc_mean": sum(r["secret_acc"] for r in needle) / len(needle),
        "attn_mass_mid": next(r["attn_mass_mean"] for r in needle if r["needle_rel"] == 0.5),
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base_cfg = TinyLMConfig.preset("medium_long")
    texts = load_text_corpus(ROOT, repeat=2)
    data = build_char_corpus(texts, base_cfg.vocab_size)

    rundir = RunDir.create(
        "most_knobs_sweep",
        argv=sys.argv,
        hparams={
            "seq_len": args.seq_len,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "sweep": [n for n, _ in SWEEP],
        },
        extra_meta={"criterion": "most_knobs_tune"},
    )
    print(
        f"Most knobs sweep seq={args.seq_len} steps={args.train_steps} "
        f"variants={len(SWEEP)} run={rundir.path}",
        flush=True,
    )

    t_all = time.time()
    rows = []

    # Softmax reference (raz)
    print("\n=== softmax ref ===", flush=True)
    sm_model, sm_train = train_arm(
        "softmax", base_cfg, data, steps=args.train_steps, seq_len=args.seq_len, seed=args.seed
    )
    sm_eval = eval_arm(sm_model, args.seq_len, args.seed)
    del sm_model
    rows.append(
        {
            "name": "softmax",
            "knobs": None,
            "train": {k: v for k, v in sm_train.items() if k != "train_curve"},
            **sm_eval,
        }
    )
    print(
        f"  SM ent_mid={sm_eval['entropy']['entropy_mid']:.3f} "
        f"dip={sm_eval['mid_dip_attn_mass']:+.4f} mass_mid={sm_eval['attn_mass_mid']:.4f}",
        flush=True,
    )

    for name, knobs in SWEEP:
        cfg = with_most_knobs(base_cfg, knobs)
        print(f"\n=== most:{name} {knobs} ===", flush=True)
        model, tr = train_arm(
            "most", cfg, data, steps=args.train_steps, seq_len=args.seq_len, seed=args.seed
        )
        ev = eval_arm(model, args.seq_len, args.seed)
        del model
        row = {
            "name": f"most:{name}",
            "knobs": asdict(knobs),
            "n_params": count_params(TinyLM(cfg, attn="most")),
            "train": {k: v for k, v in tr.items() if k != "train_curve"},
            **ev,
        }
        rows.append(row)
        print(
            f"  ent_mid={ev['entropy']['entropy_mid']:.3f} "
            f"dip={ev['mid_dip_attn_mass']:+.4f} mass_mid={ev['attn_mass_mid']:.4f} "
            f"vs SM entΔ={ev['entropy']['entropy_mid']-sm_eval['entropy']['entropy_mid']:+.3f}",
            flush=True,
        )

    sm = next(r for r in rows if r["name"] == "softmax")
    most_rows = [r for r in rows if r["name"].startswith("most:")]
    # sukces strojenia: któryś Most ma ent_mid >= SM oraz mid_dip <= SM
    winners = [
        r
        for r in most_rows
        if r["entropy"]["entropy_mid"] >= sm["entropy"]["entropy_mid"]
        and r["mid_dip_attn_mass"] <= sm["mid_dip_attn_mass"] + 1e-6
    ]
    # albo poprawa vs własny baseline
    base = next(r for r in most_rows if r["name"] == "most:baseline")
    improved = [
        r
        for r in most_rows
        if r["name"] != "most:baseline"
        and (
            r["entropy"]["entropy_mid"] > base["entropy"]["entropy_mid"] + 0.05
            or r["mid_dip_attn_mass"] < base["mid_dip_attn_mass"] - 1e-4
        )
    ]

    if winners:
        best = max(winners, key=lambda r: r["entropy"]["entropy_mid"])
        verdict = (
            f"PASS strojenia vs Softmax: **{best['name']}** "
            f"ent_mid={best['entropy']['entropy_mid']:.3f}≥SM {sm['entropy']['entropy_mid']:.3f}, "
            f"dip={best['mid_dip_attn_mass']:+.4f}≤SM {sm['mid_dip_attn_mass']:+.4f}."
        )
        status = "pass"
    elif improved:
        best = max(improved, key=lambda r: r["entropy"]["entropy_mid"])
        verdict = (
            f"PARTIAL: gałki ruszają mechanikę vs baseline Most "
            f"(best **{best['name']}** ent_mid={best['entropy']['entropy_mid']:.3f} "
            f"dip={best['mid_dip_attn_mass']:+.4f}), ale nadal nie bije Softmax mid."
        )
        status = "partial"
    else:
        verdict = (
            "FAIL strojenia: żaden wariant γ/e_scale/τ nie poprawia mid vs Softmax "
            "ani wyraźnie vs baseline Most — problem nie jest „złym init γ”."
        )
        status = "fail"

    report = {
        "protocol": "Most knobs sweep → lost-in-middle metrics",
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "rows": [
            {
                "name": r["name"],
                "knobs": r["knobs"],
                "entropy_mid": r["entropy"]["entropy_mid"],
                "entropy_head": r["entropy"]["entropy_head"],
                "entropy_tail": r["entropy"]["entropy_tail"],
                "mid_dip_attn_mass": r["mid_dip_attn_mass"],
                "attn_mass_mid": r["attn_mass_mid"],
                "secret_acc_mean": r["secret_acc_mean"],
                "train_wall_s": r["train"].get("train_wall_s"),
            }
            for r in rows
        ],
        "verdict": verdict,
        "status": status,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "MOST_KNOBS_SWEEP_results.json"
    mp = OUT_DIR / "MOST_KNOBS_SWEEP.md"
    # pełne wiersze też
    (rundir.path / "full_rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Strojenie gałek zamka Most (γ / e_scale / τ)",
        "",
        f"seq={args.seq_len} · steps={args.train_steps} · seed={args.seed}",
        "",
        f"**Werdykt:** {verdict}",
        "",
        "A = g · softplus(S/τ)² / (g² + (e_scale·ΔE)²)",
        "",
        "| name | γ_init | e_scale | τ | ent_mid | mid_dip | mass@0.5 |",
        "|------|-------:|--------:|--:|--------:|--------:|---------:|",
    ]
    for r in report["rows"]:
        k = r["knobs"] or {}
        lines.append(
            f"| {r['name']} | {k.get('gamma_init', '—')} | {k.get('e_scale', '—')} | "
            f"{k.get('couple_temp', '—')} | {r['entropy_mid']:.3f} | "
            f"{r['mid_dip_attn_mass']:+.4f} | {r['attn_mass_mid']:.4f} |"
        )
    lines += [
        "",
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `MOST_KNOBS_SWEEP_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    rundir.finalize(metrics=report, summary="\n".join(lines), status=status)
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
