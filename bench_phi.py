"""
Faza B — Softmax | Most | Most ZERO na syntetycznym Φ-drzewie.

Kryterium: Most − Softmax ≥ +3 pkt na ≥2 wartościach n; ZERO ≈ Softmax.
"""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from bridge import BridgeTransformer, SoftmaxTransformer, train_eval
from synthetic_phi import generate_phi_tree

ROOT = Path(__file__).resolve().parent
HISTORIA = ROOT / "archiwum" / "historia"


def main():
    HISTORIA.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    d_x = 8
    n_dims = [1, 3, 6, 8]
    # dla n=1 tracer w schema ma tylko depth one-hot partial — i tak OK
    train_batches = [
        generate_phi_tree(32, d_x=d_x, n_dim=max(n_dims)) for _ in range(80)
    ]
    val_batches = [
        generate_phi_tree(64, d_x=d_x, n_dim=max(n_dims)) for _ in range(15)
    ]

    print("SOFTMAX | MOST | MOST ZERO — Φ-drzewo (softplus×2)")
    print("kryterium: Most-SM ≥ +0.03 na ≥2 n; ZERO ≈ Softmax")
    print("=" * 72)

    results = {}
    for n_dim in n_dims:
        # przytnij tracer do n_dim (pierwsze n współrzędnych)
        def slice_batches(batches, n):
            out = []
            for b in batches:
                from phi_schema import PhiGraphBatch

                out.append(
                    PhiGraphBatch(
                        content=b.content,
                        tracer=b.tracer[:, :, :n],
                        labels=b.labels,
                        depths=b.depths,
                        states=b.states,
                        mask=b.mask,
                    )
                )
            return out

        tr = slice_batches(train_batches, n_dim)
        va = slice_batches(val_batches, n_dim)

        torch.manual_seed(42)
        sm = SoftmaxTransformer(
            input_dim=d_x, dim_tracer=n_dim, n_classes=4, num_layers=2
        )
        r_sm = train_eval(sm, tr, va, zero_tracer=False)

        # t_seal=False: ten bench (Faza B, zamknięty przed powstaniem seala)
        # ma T liści (depth=2) po normalizacji batchowej blisko progu COLD=0.45
        # — seal je przygasza i psuje zamknięty, audytowany wynik (zob. AUDYT.md).
        torch.manual_seed(42)
        most = BridgeTransformer(
            input_dim=d_x,
            dim_tracer=n_dim,
            n_classes=4,
            num_layers=2,
            coupling="softplus",
            t_seal=False,
        )
        r_most = train_eval(most, tr, va, zero_tracer=False)

        torch.manual_seed(42)
        most0 = BridgeTransformer(
            input_dim=d_x,
            dim_tracer=n_dim,
            n_classes=4,
            num_layers=2,
            coupling="softplus",
            t_seal=False,
        )
        r_zero = train_eval(most0, tr, va, zero_tracer=True)

        delta = r_most["acc"] - r_sm["acc"]
        d_zero = r_zero["acc"] - r_sm["acc"]
        tag = "PASS" if delta >= 0.03 else ""
        print(
            f"n={n_dim}  SM={r_sm['acc']:.4f}  Most={r_most['acc']:.4f}  "
            f"Δ={delta:+.4f}  ZERO={r_zero['acc']:.4f}  "
            f"ΔZ-SM={d_zero:+.4f}  |dE|={r_most['mean_dE']:.4f}  {tag}"
        )
        results[n_dim] = {
            "sm": r_sm,
            "most": r_most,
            "zero": r_zero,
            "delta": delta,
        }

    wins = sum(1 for r in results.values() if r["delta"] >= 0.03)
    print("=" * 72)
    print(f"KRYTERIUM +3pkt na ≥2 n: {'PASS' if wins >= 2 else 'FAIL'} ({wins}/4)")

    # tabela + wykres
    lines = [
        "# Wyniki Faza B — Φ-drzewo",
        "",
        "| n | Softmax | Most | ZERO | Most−SM |",
        "|---|---------|------|------|---------|",
    ]
    for n, r in results.items():
        lines.append(
            f"| {n} | {r['sm']['acc']:.4f} | {r['most']['acc']:.4f} | "
            f"{r['zero']['acc']:.4f} | {r['delta']:+.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Kryterium:** Most−SM ≥ +0.03 na ≥2 n → "
        f"**{'PASS' if wins >= 2 else 'FAIL'}** ({wins}/4)."
    )
    out_md = HISTORIA / "WYNIKI_FAZA_B.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zapisano: {out_md}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ns = list(results.keys())
    ax.plot(ns, [results[n]["sm"]["acc"] for n in ns], "o-", label="Softmax")
    ax.plot(ns, [results[n]["most"]["acc"] for n in ns], "s-", label="Most")
    ax.plot(ns, [results[n]["zero"]["acc"] for n in ns], "x--", label="Most ZERO")
    ax.set_xlabel("n_dim tracera Φ")
    ax.set_ylabel("val acc")
    ax.set_title("PhiMost Faza B — softplus×2 na Φ-drzewie")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = HISTORIA / "phi_most_faza_b.png"
    plt.savefig(out_png, dpi=120)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
