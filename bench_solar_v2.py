"""
Solar Names v2 — kontrola eksperymentalna na prawdziwym τ (log-dystans orbitalny,
Wikidata), holdout = 7 ciał. 18x większy korpus niż Holo P5.1 (1815 vs 100 atomów),
budowany od razu z lekcjami P5.1: prawdziwe negatywy w treningu, t_seal=False
(orbital distance ≠ semantyka świeżości Karmazyn HOT/WARM/COLD/TOMB).

  python bench_solar_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from make_dump_solar_v2 import (
    HOLDOUT_BODIES,
    TRAIN_BODIES,
    build_holdout_eval_batch,
    build_mixed_train_batch,
    write_manifest,
)
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, info_nce_loss, recall_metrics

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "doswiadczenie" / "glify"
SEEDS = [0, 1, 2, 7, 11, 17, 23, 42]
EPOCHS = 60
N_TRAIN_BATCHES_PER_EPOCH = 6
TRAIN_BODIES_PER_BATCH = 8
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 100
EVAL_DISTRACTOR_BODIES = 8


def _make_train_batches(seed: int) -> list:
    torch.manual_seed(seed)
    return [
        build_mixed_train_batch(
            d_x=CANON.d_x, n_dim=CANON.n_dim, batch_size=TRAIN_BATCH_SIZE,
            seed=seed * 97 + k, n_bodies=TRAIN_BODIES_PER_BATCH,
        )
        for k in range(N_TRAIN_BATCHES_PER_EPOCH)
    ]


def _make_eval_batch(seed: int):
    return build_holdout_eval_batch(
        d_x=CANON.d_x, n_dim=CANON.n_dim, batch_size=EVAL_BATCH_SIZE,
        seed=seed * 97 + 999, n_distractor_bodies=EVAL_DISTRACTOR_BODIES,
    )


def _make_batches(seed: int):
    """Zachowane dla determinism test (całość naraz, jak wcześniej)."""
    train = _make_train_batches(seed)
    holdout, meta = _make_eval_batch(seed)
    return train, [holdout], meta


def _most_ctor():
    return PhiBridgeEncoder(
        input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
        dim_tracer=CANON.n_dim, num_layers=CANON.num_layers, coupling=CANON.coupling,
        t_seal=False,
    )


def _sm_ctor():
    return PhiSoftmaxEncoder(
        input_dim=CANON.d_x, d=CANON.d_model, h=CANON.n_heads,
        dim_tracer=CANON.n_dim, num_layers=CANON.num_layers,
    )


def train_eval_solar(model_ctor, seed: int, *, zero_tracer: bool = False) -> dict:
    # Batche ŚWIEŻE co epokę (nowy losowy podzbiór ciał + nowy szum + nowe
    # query/pos), nie raz-na-cały-trening — inaczej sieć może zapamiętać
    # zaszumiony "odcisk palca" konkretnego (query,pos) zamiast nauczyć się
    # reguły "dopasuj po τ w obrębie kategorii" (zob. SOLAR_NAMES_V2.md).
    holdout, _meta = _make_eval_batch(seed)
    val = [holdout]
    torch.manual_seed(seed)
    model = model_ctor()
    opt = torch.optim.Adam(model.parameters(), lr=CANON.lr)
    ablate_kw = {"zero_tracer": True} if zero_tracer else {}

    for ep in range(EPOCHS):
        train = _make_train_batches(seed * 1_000_003 + ep)
        model.train()
        for batch in train:
            b = batch.ablate(**ablate_kw) if ablate_kw else batch
            opt.zero_grad()
            tok, _ = model(b.content, b.tracer, b.mask)
            loss = info_nce_loss(tok, b.query_idx, b.pos_idx, mask=b.mask)
            loss.backward()
            opt.step()

    model.eval()
    train_sanity = _make_train_batches(seed * 1_000_003 + EPOCHS)
    with torch.no_grad():
        bv = val[0].ablate(**ablate_kw) if ablate_kw else val[0]
        tok, _ = model(bv.content, bv.tracer, bv.mask)
        m = recall_metrics(tok, bv.query_idx, bv.pos_idx, mask=bv.mask)
        bt = train_sanity[0].ablate(**ablate_kw) if ablate_kw else train_sanity[0]
        tok_tr, _ = model(bt.content, bt.tracer, bt.mask)
        m_train = recall_metrics(tok_tr, bt.query_idx, bt.pos_idx, mask=bt.mask)
    m["train_r1_sanity"] = m_train["r@1"]
    return m


def run() -> dict:
    write_manifest()
    print(f"[Solar v2] holdout={HOLDOUT_BODIES} train_n={len(TRAIN_BODIES)} "
          f"seeds={SEEDS} epochs={EPOCHS}")
    rows = []
    for seed in SEEDS:
        most_m = train_eval_solar(_most_ctor, seed)
        sm_m = train_eval_solar(_sm_ctor, seed)
        zero_m = train_eval_solar(_most_ctor, seed, zero_tracer=True)
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
    verdict = f"Solar v2 {'PASS' if n_pass >= 7 else 'FAIL'}: {n_pass}/{len(SEEDS)} (Most vs ZERO na holdout)"
    report = {
        "protocol": "Solar Names v2 control (orbital log-distance tracer, holdout=7 bodies)",
        "holdout_bodies": list(HOLDOUT_BODIES), "train_bodies": list(TRAIN_BODIES),
        "seeds": SEEDS, "epochs": EPOCHS, "n_pass": n_pass, "verdict": verdict, "rows": rows,
        "mean_delta_zero": sum(r["delta_zero"] for r in rows) / len(rows),
        "mean_delta_sm": sum(r["delta_sm"] for r in rows) / len(rows),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "solar_v2_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(verdict, "mean ΔZERO=", report["mean_delta_zero"], "mean ΔSM=", report["mean_delta_sm"])
    return report


if __name__ == "__main__":
    run()
