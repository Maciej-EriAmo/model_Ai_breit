"""Ablacja: dE² (square) vs skierowane ΔE (directed / onesided).

E i tak może być ujemne z Linear; dE² gubi znak. Tu znak wraca do A (≥0).

  python archiwum/skrypty/run_signed_delta_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))
sys.path.insert(0, str(_LAB / "archiwum" / "skrypty"))

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from lm_tiny import (
    MostKnobs,
    TinyLMConfig,
    build_char_corpus,
    load_text_corpus,
    with_most_knobs,
)
from run_lost_in_middle_probe import train_arm
from run_most_knobs_sweep import eval_arm
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"

VARIANTS: list[tuple[str, MostKnobs]] = [
    ("square", MostKnobs(delta_mode="square")),
    ("directed", MostKnobs(delta_mode="directed", delta_alpha=1.0, directed_bias=0.25)),
    ("onesided", MostKnobs(delta_mode="onesided", delta_alpha=1.0)),
    # skierowane + wcześniej wygrane „szerokie” gałki
    (
        "ultra_wide_directed",
        MostKnobs(
            gamma_init=2.5,
            e_scale=0.1,
            couple_temp=2.0,
            delta_mode="directed",
            delta_alpha=1.0,
            directed_bias=0.25,
        ),
    ),
]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    base = TinyLMConfig.preset("medium_long")
    data = build_char_corpus(load_text_corpus(ROOT, repeat=2), base.vocab_size)
    rundir = RunDir.create(
        "signed_delta_ablation",
        argv=sys.argv,
        hparams={"seq_len": args.seq_len, "train_steps": args.train_steps, "seed": args.seed},
        extra_meta={"criterion": "signed_delta_E"},
    )
    print(f"Signed ΔE ablation seq={args.seq_len} steps={args.train_steps} run={rundir.path}", flush=True)

    t0 = time.time()
    rows = []

    print("\n=== softmax ===", flush=True)
    sm_m, sm_tr = train_arm(
        "softmax", base, data, steps=args.train_steps, seq_len=args.seq_len, seed=args.seed
    )
    sm_ev = eval_arm(sm_m, args.seq_len, args.seed)
    del sm_m
    rows.append({"name": "softmax", "knobs": None, **sm_ev, "train_wall_s": sm_tr["train_wall_s"]})
    print(
        f"  ent_mid={sm_ev['entropy']['entropy_mid']:.3f} dip={sm_ev['mid_dip_attn_mass']:+.4f}",
        flush=True,
    )

    for name, knobs in VARIANTS:
        cfg = with_most_knobs(base, knobs)
        print(f"\n=== most:{name} ===", flush=True)
        model, tr = train_arm(
            "most", cfg, data, steps=args.train_steps, seq_len=args.seq_len, seed=args.seed
        )
        ev = eval_arm(model, args.seq_len, args.seed)
        del model
        rows.append(
            {
                "name": f"most:{name}",
                "knobs": asdict(knobs),
                **ev,
                "train_wall_s": tr["train_wall_s"],
            }
        )
        print(
            f"  ent_mid={ev['entropy']['entropy_mid']:.3f} "
            f"dip={ev['mid_dip_attn_mass']:+.4f} mass_mid={ev['attn_mass_mid']:.4f}",
            flush=True,
        )

    sm = rows[0]
    mosts = rows[1:]
    # czy signed bije square na ent_mid lub mid_dip
    square = next(r for r in mosts if r["name"] == "most:square")
    signed = [r for r in mosts if r["name"] != "most:square"]
    better = [
        r
        for r in signed
        if r["entropy"]["entropy_mid"] > square["entropy"]["entropy_mid"] + 0.05
        or r["mid_dip_attn_mass"] < square["mid_dip_attn_mass"] - 1e-4
    ]
    beat_sm = [
        r
        for r in mosts
        if r["entropy"]["entropy_mid"] >= sm["entropy"]["entropy_mid"]
        and r["mid_dip_attn_mass"] <= sm["mid_dip_attn_mass"] + 1e-6
    ]

    if beat_sm:
        best = max(beat_sm, key=lambda r: r["entropy"]["entropy_mid"])
        verdict = (
            f"PASS vs Softmax: **{best['name']}** "
            f"ent_mid={best['entropy']['entropy_mid']:.3f} dip={best['mid_dip_attn_mass']:+.4f}."
        )
        status = "pass"
    elif better:
        best = max(better, key=lambda r: r["entropy"]["entropy_mid"])
        verdict = (
            f"PARTIAL: signed ΔE poprawia vs square (**{best['name']}** "
            f"ent_mid={best['entropy']['entropy_mid']:.3f} vs square "
            f"{square['entropy']['entropy_mid']:.3f}), ale nie bije Softmax."
        )
        status = "partial"
    else:
        verdict = (
            "FAIL: skierowane ΔE nie poprawia mid/entropii vs square na tej skali — "
            "sam znak energii nie wystarcza (albo α/protokół za słabe)."
        )
        status = "fail"

    report = {
        "protocol": "Signed / directed ΔE ablation in Most",
        "note": (
            "E z Linear może być ujemne już dziś; dE² gubi znak. "
            "directed/onesided wstawiają znak z powrotem przy A≥0."
        ),
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "rows": [
            {
                "name": r["name"],
                "knobs": r["knobs"],
                "entropy_mid": r["entropy"]["entropy_mid"],
                "mid_dip_attn_mass": r["mid_dip_attn_mass"],
                "attn_mass_mid": r["attn_mass_mid"],
                "secret_acc_mean": r["secret_acc_mean"],
            }
            for r in rows
        ],
        "verdict": verdict,
        "status": status,
        "elapsed_s": time.time() - t0,
        "run_dir": str(rundir.path),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "SIGNED_DELTA_ABLATION_results.json"
    mp = OUT_DIR / "SIGNED_DELTA_ABLATION.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Skierowane ΔE w Moście (energie ze znakiem)",
        "",
        report["note"],
        "",
        f"**Werdykt:** {verdict}",
        "",
        f"seq={args.seq_len} · steps={args.train_steps} · seed={args.seed}",
        "",
        "| name | mode | ent_mid | mid_dip | mass@0.5 |",
        "|------|------|--------:|--------:|---------:|",
    ]
    for r in report["rows"]:
        mode = (r["knobs"] or {}).get("delta_mode", "—")
        lines.append(
            f"| {r['name']} | {mode} | {r['entropy_mid']:.3f} | "
            f"{r['mid_dip_attn_mass']:+.4f} | {r['attn_mass_mid']:.4f} |"
        )
    lines += ["", f"elapsed={report['elapsed_s']:.0f}s · `{rundir.path}`", ""]
    mp.write_text("\n".join(lines), encoding="utf-8")
    rundir.finalize(metrics=report, summary="\n".join(lines), status=status)
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
