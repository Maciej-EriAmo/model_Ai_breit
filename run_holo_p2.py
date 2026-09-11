"""
Holo-Φ P2 — mini-sweep blend_holo ∈ {0.0, 0.2, 0.4, 0.6, 0.8}

Seedy z P1 (w tym PASS 23 + FAIL): czy wynik zależy mocno od hiperparametru?

  python make_dump_holo.py
  python run_holo_p2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Determinizm run-to-run: manual_seed(seed) sam nie wystarcza — wielowątkowy
# CPU matmul/attention ma niedeterministyczną kolejność redukcji fp (patrz
# doswiadczenie/glify/HOLO_WNIOSKIab.md §5: ten sam seed dawał różne r@1
# między przebiegami identycznego kodu).
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import load_raw_records, make_dump_loaders
from make_dump_holo import DUMP, main as build_dump
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "doswiadczenie" / "glify"

BLENDS = [0.0, 0.2, 0.4, 0.6, 0.8]
SEEDS = [0, 1, 17, 23, 42]  # 5 seedów: PASS(23) + typowe FAIL
EPOCHS = 16
N_TRAIN = 80
N_VAL = 25


def run_one(seed: int, blend: float, *, bs: int) -> dict:
    torch.manual_seed(seed)
    _g, train, val, _vc = make_dump_loaders(
        DUMP,
        n_train=N_TRAIN,
        n_val=N_VAL,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
        blend_holo_proto=blend,
    )
    torch.manual_seed(seed)
    most = build_model(warm=False)
    torch.manual_seed(seed)
    most_m = train_encoder(most, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    sm = PhiSoftmaxEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
    )
    sm_m = train_encoder(sm, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    most_z = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )
    z_m = train_encoder(
        most_z,
        train,
        val,
        epochs=EPOCHS,
        lr=CANON.lr,
        ablate_kw={"zero_tracer": True},
    )
    d_sm = most_m["r@1"] - sm_m["r@1"]
    return {
        "seed": seed,
        "blend": blend,
        "most_r1": most_m["r@1"],
        "sm_r1": sm_m["r@1"],
        "zero_r1": z_m["r@1"],
        "delta_sm": d_sm,
        "gate_pass": d_sm >= CANON.gate_recall_r1,
    }


def run() -> dict:
    build_dump()
    n_atoms = len(load_raw_records(DUMP))
    # Duże grafy (N≳400): mniejszy batch — uwaga O(N²) na próbkę
    bs = 4 if n_atoms >= 400 else min(16, max(4, n_atoms))
    rows: list[dict] = []

    for blend in BLENDS:
        print(f"========== blend_holo={blend} ==========")
        for seed in SEEDS:
            r = run_one(seed, blend, bs=bs)
            tag = "PASS" if r["gate_pass"] else "FAIL"
            print(
                f"  seed={seed}  Most={r['most_r1']:.3f} SM={r['sm_r1']:.3f} "
                f"ZERO={r['zero_r1']:.3f} ΔSM={r['delta_sm']:+.4f} {tag}"
            )
            rows.append(r)

    # agregaty per blend
    by_blend = []
    for blend in BLENDS:
        sub = [r for r in rows if r["blend"] == blend]
        n_pass = sum(1 for r in sub if r["gate_pass"])
        deltas = [r["delta_sm"] for r in sub]
        by_blend.append(
            {
                "blend": blend,
                "n_pass": n_pass,
                "n_seeds": len(sub),
                "mean_delta_sm": sum(deltas) / len(deltas),
                "best_delta_sm": max(deltas),
                "worst_delta_sm": min(deltas),
                "mean_most": sum(r["most_r1"] for r in sub) / len(sub),
                "mean_sm": sum(r["sm_r1"] for r in sub) / len(sub),
            }
        )

    best_cfg = max(by_blend, key=lambda x: (x["n_pass"], x["mean_delta_sm"]))
    report = {
        "protocol": "Holo-Φ P2 blend sweep",
        "seeds": SEEDS,
        "blends": BLENDS,
        "epochs": EPOCHS,
        "n_atoms": n_atoms,
        "rows": rows,
        "by_blend": by_blend,
        "best_blend": best_cfg["blend"],
        "verdict": (
            f"best blend={best_cfg['blend']} PASS {best_cfg['n_pass']}/{best_cfg['n_seeds']} "
            f"meanΔSM={best_cfg['mean_delta_sm']:+.4f}; "
            "czy FAIL P1 to metoda czy hiperparametr — patrz rozrzut by_blend"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holo_p2_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(report["verdict"])
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Holo-Φ P2 — sweep `blend_holo`",
        "",
        f"Seedy: {report['seeds']} · epochs={report['epochs']} · atoms={report['n_atoms']}",
        "",
        "## Agregat per blend",
        "",
        "| blend | PASS | mean ΔSM | best ΔSM | worst ΔSM | mean Most | mean SM |",
        "|------:|-----:|---------:|---------:|----------:|----------:|--------:|",
    ]
    for b in report["by_blend"]:
        lines.append(
            f"| {b['blend']} | {b['n_pass']}/{b['n_seeds']} | {b['mean_delta_sm']:+.4f} | "
            f"{b['best_delta_sm']:+.4f} | {b['worst_delta_sm']:+.4f} | "
            f"{b['mean_most']:.3f} | {b['mean_sm']:.3f} |"
        )
    lines += [
        "",
        f"**Najlepszy blend (PASS, potem meanΔ):** **{report['best_blend']}**",
        "",
        "## Pełna tabela",
        "",
        "| blend | seed | Most | SM | ZERO | ΔSM | gate |",
        "|------:|-----:|-----:|---:|-----:|----:|:----:|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['blend']} | {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | {tag} |"
        )
    lines += [
        "",
        f"**Werdykt:** {report['verdict']}",
        "",
        "JSON: `holo_p2_results.json`",
        "",
    ]
    (OUT_DIR / "HOLO_P2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P2] wrote {OUT_DIR / 'HOLO_P2.md'}")


if __name__ == "__main__":
    run()
