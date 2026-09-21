"""Diagnostyka poziomu nauki Kana P1: krzywa loss/r@1 vs poziom przypadku."""
from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import make_dump_loaders
from phimost_learn import build_model
from recall_model import (
    PhiBridgeEncoder,
    PhiSoftmaxEncoder,
    _avg_val_metrics,
    train_encoder,
)

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_kana_p1" / "hologram"
OUT = ROOT / "doswiadczenie" / "glify"
SEED = 0
EPOCHS = 16
N_TRAIN = 80
N_VAL = 25


def main() -> None:
    g, train, val, _vc = make_dump_loaders(
        DUMP,
        n_train=N_TRAIN,
        n_val=N_VAL,
        batch_size=min(16, max(4, 286)),
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + SEED * 17,
    )
    N = len(g.atoms)
    chance = 1.0 / max(1, N - 1)
    print(f"N={N} chance_r@1={chance:.5f} ({chance*100:.3f}%) epochs={EPOCHS}")

    results = {"N": N, "chance_r1": chance, "epochs": EPOCHS, "seed": SEED, "models": {}}

    for name, factory, ablate in (
        ("Most", lambda: build_model(warm=False), {}),
        ("Softmax", lambda: PhiSoftmaxEncoder(
            input_dim=CANON.d_x, dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers, d=CANON.d_model,
        ), {}),
        ("ZERO", lambda: PhiBridgeEncoder(
            input_dim=CANON.d_x, dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers, d=CANON.d_model, coupling=CANON.coupling,
        ), {"zero_tracer": True}),
    ):
        torch.manual_seed(SEED)
        model = factory()
        cold = _avg_val_metrics(model, val, ablate)
        print(f"\n=== {name} cold  r@1={cold['r@1']:.4f} r@3={cold['r@3']:.4f} "
              f"mrr={cold['mrr']:.4f}  (chance={chance:.4f})")
        t0 = time.time()
        torch.manual_seed(SEED)
        model = factory()  # fresh, same init as train path
        trained = train_encoder(
            model, train, val, epochs=EPOCHS, lr=CANON.lr,
            ablate_kw=ablate, return_curve=True,
        )
        dt = time.time() - t0
        curve = trained["curve"]
        print(f"=== {name} trained ({dt:.0f}s) final r@1={trained['r@1']:.4f} "
              f"r@3={trained['r@3']:.4f} mrr={trained['mrr']:.4f}")
        print(f"    loss: ep1={curve[0]['loss']:.3f} → ep{EPOCHS}={curve[-1]['loss']:.3f}")
        print(f"    r@1:  ep1={curve[0]['r@1']:.4f} → ep{EPOCHS}={curve[-1]['r@1']:.4f}")
        # show every 4 epochs
        for row in curve:
            if row["epoch"] in (1, 4, 8, 12, 16):
                ratio = row["r@1"] / chance if chance > 0 else float("nan")
                print(
                    f"    ep{row['epoch']:02d}  loss={row['loss']:.3f}  "
                    f"r@1={row['r@1']:.4f} ({ratio:.1f}×chance)  "
                    f"r@3={row['r@3']:.4f}  mrr={row['mrr']:.4f}"
                )
        results["models"][name] = {
            "cold": {k: cold[k] for k in ("r@1", "r@3", "mrr")},
            "final": {k: trained[k] for k in ("r@1", "r@3", "mrr")},
            "curve": curve,
            "seconds": dt,
            "final_over_chance": trained["r@1"] / chance,
            "learned": trained["r@1"] > cold["r@1"] + 0.5 * chance,
            "loss_drop": curve[0]["loss"] - curve[-1]["loss"],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "kana_learning_diag.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Kana P1 — diagnostyka poziomu nauki",
        "",
        f"Seed={SEED}, encoding=hologram, epochs={EPOCHS}, N={N}, "
        f"**chance r@1 = 1/(N-1) = {chance:.5f} ({100*chance:.3f}%)**.",
        "",
        "InfoNCE recall po całym grafie (286 węzłów) — Kana P1 raportowało "
        "końcowe r@1≈0.00–0.02 bez krzywej. Tu: cold start vs krzywa vs ×chance.",
        "",
        "## Podsumowanie",
        "",
        "| model | cold r@1 | final r@1 | ×chance | Δloss | uczy się? |",
        "|-------|----------|-----------|---------|-------|-----------|",
    ]
    for name, m in results["models"].items():
        lines.append(
            f"| {name} | {m['cold']['r@1']:.4f} | {m['final']['r@1']:.4f} | "
            f"{m['final_over_chance']:.1f}× | {m['loss_drop']:+.3f} | "
            f"{'tak' if m['learned'] else 'słabo/nie'} |"
        )
    lines += [
        "",
        "## Interpretacja",
        "",
        f"- Poziom przypadku ≈ **{100*chance:.2f}%**. "
        "Kana P1 mean Most≈0.5% ≈ **~1–2×chance** — praktycznie nie nauczył się rankingu.",
        "- Holo-Φ P2 (dla porównania) miało mean Most≈0.14–0.16 (~40×chance przy mniejszym N) — inna skala nauki.",
        "- Jeśli loss spada, a r@1 zostaje przy podłodze: model minimalizuje CE, ale nie generalizuje rankingu na val.",
        "",
        f"JSON: `{out_json.name}`",
        "",
    ]
    out_md = OUT / "KANA_LEARNING_DIAG.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
