"""
Holo-Φ P4a — gate lokalny (pre-registered)

  blend_holo = 0.6   (lab default z P2 — BEZ dalszego dostrajania)
  seeds = 8 (jak P1)
  kryterium PASS: n_pass / 8 >= 4/5  →  potrzeba ≥ 7/8
  (równoważnie: udział ≥ 80%)

  python run_holo_p4a.py
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
from make_dump_holo import DUMP, write_dump
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder
from checkpoint import save_encoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "archiwum" / "holo"

BLEND = 0.6
SEEDS = [0, 1, 2, 7, 11, 17, 23, 42]
EPOCHS = 16
N_TRAIN = 80
N_VAL = 25
# ≥ 4/5 udziału → przy 8 seedach minimum 7 PASS
MIN_PASS_RATIO = 4 / 5


def run() -> dict:
    write_dump(DUMP, root_shared_phase=True, root_binds=True)
    n_atoms = len(load_raw_records(DUMP))
    # Duże grafy (N≳400): mniejszy batch — uwaga O(N²) na próbkę
    bs = 4 if n_atoms >= 400 else min(16, max(4, n_atoms))
    rows = []
    best_model = None
    best_row: dict | None = None

    print(
        f"[P4a] blend={BLEND} seeds={SEEDS} epochs={EPOCHS} "
        f"gate≥{MIN_PASS_RATIO:.0%} (≥{int(MIN_PASS_RATIO * len(SEEDS) + 0.999)}/{len(SEEDS)})"
    )

    for seed in SEEDS:
        print(f"------------------------------------------------------------\nSEED {seed}")
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
            blend_holo_proto=BLEND,
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
        # float-safe: traktuj Δ >= gate - 0.5*step jako pass (r@1 kwantyzowane co step = 1/(N_VAL*bs))
        step = 1.0 / float(N_VAL * bs)
        ok = d_sm >= CANON.gate_recall_r1 - 0.5 * step
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
            }
        )
        if best_row is None or d_sm > best_row["delta_sm"]:
            best_row = rows[-1]
            best_model = most

    n_pass = sum(1 for r in rows if r["gate_pass"])
    need = int(MIN_PASS_RATIO * len(SEEDS) + 0.999)  # ceil
    p4a_pass = n_pass >= need
    best = best_row
    mean_d = sum(r["delta_sm"] for r in rows) / len(rows)

    save_encoder(
        best_model,
        ROOT / "checkpoints" / "most_holo_p4a.pt",
        kind="most",
        meta={
            "p4a": True,
            "blend": BLEND,
            "seed": best["seed"],
            "delta_sm": best["delta_sm"],
            "n_pass": n_pass,
        },
    )

    verdict = (
        f"P4a {'PASS' if p4a_pass else 'FAIL'}: {n_pass}/{len(SEEDS)} "
        f"(wymagane ≥{need}/{len(SEEDS)} = {MIN_PASS_RATIO:.0%}); "
        f"meanΔSM={mean_d:+.4f} best={best['delta_sm']:+.4f}@seed{best['seed']}"
    )
    report = {
        "protocol": "Holo-Φ P4a local gate",
        "blend_holo": BLEND,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "min_pass_ratio": MIN_PASS_RATIO,
        "min_pass_count": need,
        "n_pass": n_pass,
        "p4a_pass": p4a_pass,
        "mean_delta_sm": mean_d,
        "best": best,
        "rows": rows,
        "verdict": verdict,
        "note": (
            "blend=0.6 pre-registered z P2; bez fine-tune. "
            "To gate lokalny (P4a), nie gate skalowania alfabetów (P4b)."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holo_p4a_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(verdict)
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Holo-Φ P4a — gate lokalny",
        "",
        f"**Pre-registered:** `blend_holo={report['blend_holo']}` · "
        f"seeds={report['seeds']} · epochs={report['epochs']}",
        f"**Kryterium:** PASS ≥ {report['min_pass_ratio']:.0%} seedów "
        f"(≥{report['min_pass_count']}/{len(report['seeds'])})",
        "",
        report["note"],
        "",
        "| seed | Most | SM | ZERO | ΔSM | gate |",
        "|-----:|-----:|---:|-----:|----:|:----:|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | {tag} |"
        )
    lines += [
        "",
        f"**Wynik:** {report['n_pass']}/{len(report['seeds'])} PASS · "
        f"mean ΔSM={report['mean_delta_sm']:+.4f}",
        "",
        f"**Werdykt P4a:** {report['verdict']}",
        "",
        "JSON: `holo_p4a_results.json` · ckpt: `checkpoints/most_holo_p4a.pt`",
        "",
    ]
    (OUT_DIR / "HOLO_P4a.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[P4a] wrote {OUT_DIR / 'HOLO_P4a.md'}")


if __name__ == "__main__":
    run()
