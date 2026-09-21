"""
Holo-Φ P5.1 — kontrola eksperymentalna na τ=Unihan (stroke_count), holdout=radykał.

  python bench_holo_v2.py

Gate główny (P5 §2, plan): Most vs ZERO NA HOLDOUT (nie r@1 w znanym grafie).
Gate drugorzędny: Most vs Softmax na holdout.
Determinizm: torch.set_num_threads(1) + use_deterministic_algorithms — patrz
  doswiadczenie/testy/test_holo_v2_determinism.py (musi PASS zanim jakikolwiek seed liczy się jako dowód).
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

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from make_dump_holo_v2 import (
    HOLDOUT_RADICAL,
    TRAIN_RADICALS,
    build_holdout_eval_batch,
    build_mixed_train_batch,
    write_manifest,
)
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, recall_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"
SEEDS = [0, 1, 2, 7, 11, 17, 23, 42]
EPOCHS = 60  # train_r1 (in-distribution) plateauje ~ep45 przy 100% — patrz HOLO_P5_1.md
TRAIN_BATCH_SIZE = 16
# r@1 kwantyzowane co 1/EVAL_BATCH_SIZE — gate to +0.02, więc potrzeba drobnej
# ziarnistości (P5.0: "krok 0.0025 (400 queries)"); 20 znaków holdout + noise
# per-row daje różne query/pos losowania nawet przy dużym batchu.
EVAL_BATCH_SIZE = 200


def _make_batches(seed: int) -> tuple[list, list, dict]:
    # reset globalnego RNG torch PRZED szumem w build_*_batch — inaczej wynik
    # zależy od tego, ile losowań zużyło to, co działo się wcześniej w procesie
    # (dokładnie ten sam efekt co niedeterminizm z replikacji P4a, §2.5).
    torch.manual_seed(seed)
    # mieszany graf (wszystkie radykały treningowe razem) — patrz
    # build_mixed_train_batch: build_train_batch (jeden-radykał-per-graf) nie
    # dawał żadnych negatywów podczas treningu, więc τ nie miało nic do
    # odróżniania (HOLO_P5_1.md diagnoza). Kilka wywołań z różnym seedem =
    # tyle samo kroków gradientu na epokę co poprzednio (4).
    train = [
        build_mixed_train_batch(
            TRAIN_RADICALS, d_x=CANON.d_x, n_dim=CANON.n_dim,
            batch_size=TRAIN_BATCH_SIZE, seed=seed * 97 + k,
        )
        for k in range(len(TRAIN_RADICALS))
    ]
    holdout, meta = build_holdout_eval_batch(
        d_x=CANON.d_x, n_dim=CANON.n_dim, batch_size=EVAL_BATCH_SIZE, seed=seed * 97 + HOLDOUT_RADICAL
    )
    return train, [holdout], meta


def train_eval_holo(model_ctor, seed: int, *, zero_tracer: bool = False) -> dict:
    """model_ctor() -> nn.Module świeży (żeby seed kontrolował też init wag)."""
    train, val, _meta = _make_batches(seed)
    torch.manual_seed(seed)
    model = model_ctor()
    opt = torch.optim.Adam(model.parameters(), lr=CANON.lr)
    ablate_kw = {"zero_tracer": True} if zero_tracer else {}

    for _ in range(EPOCHS):
        model.train()
        for batch in train:
            b = batch.ablate(**ablate_kw) if hasattr(batch, "ablate") else batch
            opt.zero_grad()
            tok, _ = model(b.content, b.tracer, b.mask)
            from recall_model import info_nce_loss

            loss = info_nce_loss(tok, b.query_idx, b.pos_idx, mask=b.mask)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        b = val[0].ablate(**ablate_kw) if hasattr(val[0], "ablate") else val[0]
        tok, _ = model(b.content, b.tracer, b.mask)
        m = recall_metrics(tok, b.query_idx, b.pos_idx, mask=b.mask)
        # sanity in-distribution: recall na ZNANYM (train) radykale — odróżnia
        # "τ nie pomaga" od "model w ogóle się tu niczego nie uczy" (P5.1 §krzywa
        # do plateau). Jeśli to też ~chance, wynik na holdout jest bez znaczenia.
        tb = train[0]
        tb2 = tb.ablate(**ablate_kw) if hasattr(tb, "ablate") else tb
        tok_tr, _ = model(tb2.content, tb2.tracer, tb2.mask)
        m_train = recall_metrics(tok_tr, tb2.query_idx, tb2.pos_idx, mask=tb2.mask)
    m["train_r1_sanity"] = m_train["r@1"]
    return m


def _most_ctor():
    return PhiBridgeEncoder(
        input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
        dim_tracer=CANON.n_dim, num_layers=CANON.num_layers, coupling=CANON.coupling,
    )


def _sm_ctor():
    return PhiSoftmaxEncoder(
        input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
        dim_tracer=CANON.n_dim, num_layers=CANON.num_layers,
    )


def run() -> dict:
    write_manifest()
    print(f"[P5.1] holdout_radical={HOLDOUT_RADICAL} train={TRAIN_RADICALS} "
          f"seeds={SEEDS} epochs={EPOCHS}")
    rows = []
    for seed in SEEDS:
        most_m = train_eval_holo(_most_ctor, seed)
        sm_m = train_eval_holo(_sm_ctor, seed)
        zero_m = train_eval_holo(_most_ctor, seed, zero_tracer=True)
        d_zero = most_m["r@1"] - zero_m["r@1"]
        d_sm = most_m["r@1"] - sm_m["r@1"]
        gate_pass = d_zero >= CANON.gate_recall_r1
        print(
            f"  seed={seed:<3} Most={most_m['r@1']:.3f} SM={sm_m['r@1']:.3f} "
            f"ZERO={zero_m['r@1']:.3f} ΔZERO={d_zero:+.4f} ΔSM={d_sm:+.4f} "
            f"train_sanity={most_m['train_r1_sanity']:.3f} "
            f"{'PASS' if gate_pass else 'FAIL'}"
        )
        rows.append({
            "seed": seed, "most_r1": most_m["r@1"], "sm_r1": sm_m["r@1"],
            "zero_r1": zero_m["r@1"], "delta_zero": d_zero, "delta_sm": d_sm,
            "gate_pass": gate_pass, "most_train_r1_sanity": most_m["train_r1_sanity"],
        })

    n_pass = sum(1 for r in rows if r["gate_pass"])
    need = int(0.875 * len(SEEDS) + 0.999)  # ≥7/8 per plan P5.4 (stosujemy tu jako referencję)
    verdict = f"P5.1 {'PASS' if n_pass >= need else 'FAIL'}: {n_pass}/{len(SEEDS)} (Most vs ZERO na holdout)"
    report = {
        "protocol": "Holo-Phi P5.1 control (unihan stroke tracer, holdout radical)",
        "holdout_radical": HOLDOUT_RADICAL,
        "train_radicals": list(TRAIN_RADICALS),
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "n_pass": n_pass,
        "verdict": verdict,
        "rows": rows,
        "mean_delta_zero": sum(r["delta_zero"] for r in rows) / len(rows),
        "mean_delta_sm": sum(r["delta_sm"] for r in rows) / len(rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "holo_p5_1_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(verdict, "mean ΔZERO=", report["mean_delta_zero"], "mean ΔSM=", report["mean_delta_sm"])
    return report


if __name__ == "__main__":
    run()
