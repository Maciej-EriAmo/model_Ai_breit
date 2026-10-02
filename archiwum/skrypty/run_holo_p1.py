"""
Holo-Φ P1 — kontrola eksperymentalna:
  - ziarnistość r@1 (N_queries → minimalny krok metryki)
  - 8 seedów Softmax|Most|ZERO
  - krzywe loss/r@1 (Most; seed0 także SM/ZERO)

  python make_dump_holo.py
  python run_holo_p1.py
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
from pathlib import Path

import torch

# Determinizm run-to-run: manual_seed(seed) sam nie wystarcza — wielowątkowy
# CPU matmul/attention ma niedeterministyczną kolejność redukcji fp (patrz
# archiwum/holo/HOLO_WNIOSKIab.md §5: ten sam seed dawał różne r@1
# między przebiegami identycznego kodu).
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import load_raw_records, make_dump_loaders
from make_dump_holo import DUMP, main as build_dump
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder
from checkpoint import save_encoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "archiwum" / "holo"
SEEDS = [0, 1, 2, 7, 11, 17, 23, 42]  # 8 seedów (P1: 5–10)
EPOCHS = 16
BLEND = 0.4
N_TRAIN = 80
N_VAL = 25


def r1_granularity(*, n_val: int, batch_size: int) -> dict:
    """r@1 = średnia hitów; minimalny skok = 1/N_queries."""
    n_queries = n_val * batch_size
    step = 1.0 / float(n_queries) if n_queries else float("nan")
    gate = float(CANON.gate_recall_r1)
    return {
        "n_val_batches": n_val,
        "batch_size": batch_size,
        "n_queries": n_queries,
        "r1_min_step": step,
        "gate_delta_sm": gate,
        "gate_in_steps": (gate / step) if step else None,
        "note": (
            "ΔSM gate +0.02 jest rozróżnialny tylko jeśli |Δ| ≳ step; "
            f"tu step={step:.4f} → gate ≈ {gate/step:.1f} kroków"
            if step
            else "n/a"
        ),
    }


def run() -> dict:
    build_dump()
    n_atoms = len(load_raw_records(DUMP))
    # Duże grafy (N≳400): mniejszy batch — uwaga O(N²) na próbkę
    bs = 4 if n_atoms >= 400 else min(16, max(4, n_atoms))
    gran = r1_granularity(n_val=N_VAL, batch_size=bs)
    print("[P1] granularity", json.dumps(gran, ensure_ascii=False))

    rows = []
    curves: dict = {}

    for seed in SEEDS:
        print(f"------------------------------------------------------------\nSEED {seed}")
        torch.manual_seed(seed)
        g, train, val, val_clean = make_dump_loaders(
            DUMP,
            n_train=N_TRAIN,
            n_val=N_VAL,
            batch_size=bs,
            d_x=CANON.d_x,
            n_dim=CANON.n_dim,
            pair_mode="mixed",
            noise_scale=0.02,
            seed0=1000 + seed * 17,
            blend_holo_proto=BLEND,
        )
        # Most + curve
        torch.manual_seed(seed)
        most = build_model(warm=False)
        torch.manual_seed(seed)
        most_m = train_encoder(
            most, train, val, epochs=EPOCHS, lr=CANON.lr, return_curve=True
        )
        curves[f"most_seed{seed}"] = most_m.get("curve", [])

        # Softmax / ZERO — pełne metryki; krzywe tylko seed 0 (czas)
        torch.manual_seed(seed)
        sm = PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        )
        sm_kw = dict(epochs=EPOCHS, lr=CANON.lr)
        if seed == 0:
            sm_m = train_encoder(sm, train, val, return_curve=True, **sm_kw)
            curves["softmax_seed0"] = sm_m.get("curve", [])
        else:
            sm_m = train_encoder(sm, train, val, **sm_kw)

        torch.manual_seed(seed)
        most_z = PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
        )
        if seed == 0:
            z_m = train_encoder(
                most_z,
                train,
                val,
                return_curve=True,
                ablate_kw={"zero_tracer": True},
                **sm_kw,
            )
            curves["zero_seed0"] = z_m.get("curve", [])
        else:
            z_m = train_encoder(
                most_z, train, val, ablate_kw={"zero_tracer": True}, **sm_kw
            )

        d_sm = most_m["r@1"] - sm_m["r@1"]
        ok = d_sm >= CANON.gate_recall_r1
        tag = "PASS" if ok else "FAIL"
        print(
            f"  Most={most_m['r@1']:.3f}  SM={sm_m['r@1']:.3f}  "
            f"ZERO={z_m['r@1']:.3f}  ΔSM={d_sm:+.4f}  {tag}"
        )
        rows.append(
            {
                "seed": seed,
                "most_r1": most_m["r@1"],
                "sm_r1": sm_m["r@1"],
                "zero_r1": z_m["r@1"],
                "delta_sm": d_sm,
                "gate_pass": ok,
                "most_curve_last": (most_m.get("curve") or [{}])[-1],
            }
        )
        if seed == 0:
            save_encoder(
                most,
                ROOT / "checkpoints" / "most_holo_p1.pt",
                kind="most",
                meta={"p1": True, "seed": 0, "r@1": most_m["r@1"]},
            )

    n_pass = sum(1 for r in rows if r["gate_pass"])
    best = max(rows, key=lambda r: r["delta_sm"])
    report = {
        "protocol": "Holo-Φ P1",
        "dump": str(DUMP),
        "n_atoms": n_atoms,
        "epochs": EPOCHS,
        "blend_holo": BLEND,
        "seeds": SEEDS,
        "granularity": gran,
        "rows": rows,
        "n_pass": n_pass,
        "n_seeds": len(SEEDS),
        "best_delta_sm": best["delta_sm"],
        "best_seed": best["seed"],
        "verdict": (
            f"gate PASS {n_pass}/{len(SEEDS)} "
            f"(kryterium docelowe ≥4/5 wymaga P4; tu kontrola P1)"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holo_p1_results.json").write_text(
        json.dumps({"report": report, "curves": curves}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    _write_md(report, curves)
    print(report["verdict"])
    return report


def _write_md(report: dict, curves: dict) -> None:
    gran = report["granularity"]
    lines = [
        "# Holo-Φ P1 — kontrola eksperymentalna",
        "",
        f"**Data:** dump `{report['dump']}` · atoms={report['n_atoms']} · "
        f"epochs={report['epochs']} · blend_holo={report['blend_holo']}",
        "",
        "## Ziarnistość r@1",
        "",
        f"| | |",
        f"|--|--|",
        f"| n_queries (val) | **{gran['n_queries']}** (= {gran['n_val_batches']}×{gran['batch_size']}) |",
        f"| minimalny krok r@1 | **{gran['r1_min_step']:.4f}** |",
        f"| gate ΔSM | {gran['gate_delta_sm']} ≈ **{gran['gate_in_steps']:.1f}** kroków |",
        "",
        gran["note"],
        "",
        "## Softmax | Most | ZERO (wszystkie seedy)",
        "",
        "| seed | Most | SM | ZERO | ΔSM | gate |",
        "|------|------|----|------|-----|------|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | {tag} |"
        )
    lines += [
        "",
        f"**PASS:** {report['n_pass']}/{report['n_seeds']} · "
        f"best ΔSM={report['best_delta_sm']:+.4f} (seed {report['best_seed']})",
        "",
        f"**Werdykt P1:** {report['verdict']}",
        "",
        "## Krzywe (skrót)",
        "",
        "Pełne JSON: `holo_p1_results.json` → `curves`.",
        "",
    ]
    # Most seed0 curve table
    c0 = curves.get("most_seed0") or []
    if c0:
        lines += [
            "### Most seed=0",
            "",
            "| epoch | loss | r@1 |",
            "|------:|-----:|----:|",
        ]
        for pt in c0:
            lines.append(
                f"| {pt['epoch']} | {pt['loss']:.4f} | {pt['r@1']:.3f} |"
            )
        lines.append("")
        # plateau check: last 3 epochs r@1 range
        last = c0[-3:]
        r1s = [p["r@1"] for p in last]
        lines.append(
            f"Ostatnie 3 epoki r@1: {r1s} · rozrzut={max(r1s)-min(r1s):.4f} "
            f"({'plateau-ish' if max(r1s)-min(r1s) < 2*gran['r1_min_step'] else 'jeszcze się rusza'})"
        )
        lines.append("")
    sm0 = curves.get("softmax_seed0") or []
    if sm0:
        lines += [
            "### Softmax seed=0 (końcówka)",
            "",
            f"ep16 loss={sm0[-1]['loss']:.4f} r@1={sm0[-1]['r@1']:.3f}",
            "",
        ]
    (OUT_DIR / "HOLO_P1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P1] wrote {OUT_DIR / 'HOLO_P1.md'}")


if __name__ == "__main__":
    run()
