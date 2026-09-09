"""
Faza C — Softmax | Most | skalarne R + ablacje.

Kryterium (metryka główna = r@1; r@3 raportowane):
  Most r@1 ≥ Softmax r@1 + 0.02
  Most nie gorszy od skalarnego R o > 0.02 na r@1 (albo bije R = sukces mocny)

Uwaga: przy wielu hard-negach same-topic Softmax często trzyma pozytyw w top-3
bez geometrii depth — stąd r@1, nie r@3, jako gate.
"""
from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from recall_data import axis_correlation_report, make_loader
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder
from scalar_r import eval_scalar_R, eval_cosine_only

ROOT = Path(__file__).resolve().parent
HISTORIA = ROOT / "historia"


def fmt(m: dict) -> str:
    return (
        f"r@1={m['r@1']:.3f} r@3={m['r@3']:.3f} r@5={m['r@5']:.3f} "
        f"MRR={m['mrr']:.3f}"
    )


def main():
    HISTORIA.mkdir(exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)

    d_x, n_dim = 8, 6
    train = make_loader(
        60, batch_size=32, d_x=d_x, n_dim=n_dim, decorrelate_axes=True
    )
    val = make_loader(
        20, batch_size=64, d_x=d_x, n_dim=n_dim, decorrelate_axes=True
    )
    corr = axis_correlation_report(val[0])

    print("FAZA C — recall/ranking nad Φ (N=24, hard negs cross-depth)")
    print("pozytyw = ten sam topic + ta sama depth")
    print("osie T/state/mazur dekorelowane od depth (P2.13)")
    print("kryterium: Most r@1 ≥ Softmax +0.02; Most ≱ R−0.02 (r@1)")
    print("corr:", ", ".join(f"{k}={v:.3f}" for k, v in corr.items()))
    print("=" * 72)

    results = {}

    # --- neural ---
    torch.manual_seed(42)
    sm = PhiSoftmaxEncoder(input_dim=d_x, dim_tracer=n_dim, num_layers=2)
    results["Softmax"] = train_encoder(sm, train, val, epochs=8)
    print(f"Softmax     {fmt(results['Softmax'])}")

    torch.manual_seed(42)
    most = PhiBridgeEncoder(
        input_dim=d_x, dim_tracer=n_dim, num_layers=2, coupling="softplus"
    )
    results["Most"] = train_encoder(most, train, val, epochs=8)
    print(
        f"Most        {fmt(results['Most'])}  |dE|={results['Most']['mean_dE']:.4f}"
    )

    torch.manual_seed(42)
    most_z = PhiBridgeEncoder(
        input_dim=d_x, dim_tracer=n_dim, num_layers=2, coupling="softplus"
    )
    results["Most ZERO"] = train_encoder(
        most_z, train, val, epochs=8, ablate_kw={"zero_tracer": True}
    )
    print(f"Most ZERO   {fmt(results['Most ZERO'])}")

    # ablacje Most (po dekorelacji −depth/−T/−mazur są informatywne)
    for name, kw in [
        ("Most −depth", {"drop_depth": True}),
        ("Most −T", {"drop_T": True}),
        ("Most −mazur", {"drop_mazur": True}),
        ("flat payload", {"flat_payload": True}),
    ]:
        torch.manual_seed(42)
        m = PhiBridgeEncoder(
            input_dim=d_x, dim_tracer=n_dim, num_layers=2, coupling="softplus"
        )
        results[name] = train_encoder(m, train, val, epochs=8, ablate_kw=kw)
        print(f"{name:<12} {fmt(results[name])}")

    # --- skalarne R ---
    results["R (depth dE)"] = eval_scalar_R(val, g=0.3, use_tracer_l2=False)
    results["R (tracer L2)"] = eval_scalar_R(val, g=0.3, use_tracer_l2=True)
    results["cosine only"] = eval_cosine_only(val)
    print(f"R depth     {fmt(results['R (depth dE)'])}")
    print(f"R tracerL2  {fmt(results['R (tracer L2)'])}")
    print(f"cosine      {fmt(results['cosine only'])}")

    # --- kryterium (r@1) ---
    d_sm = results["Most"]["r@1"] - results["Softmax"]["r@1"]
    d_r = results["Most"]["r@1"] - results["R (depth dE)"]["r@1"]
    d_sm3 = results["Most"]["r@3"] - results["Softmax"]["r@3"]
    beat_sm = d_sm >= 0.02
    not_worse_r = d_r >= -0.02
    strong = d_r >= 0.02
    ok = beat_sm and not_worse_r

    print("=" * 72)
    print(f"Δ Most−Softmax r@1 = {d_sm:+.4f}  ({'PASS' if beat_sm else 'FAIL'} ≥+0.02)")
    print(f"Δ Most−R       r@1 = {d_r:+.4f}  ({'PASS' if not_worse_r else 'FAIL'} ≥−0.02)")
    print(f"(info) Δ Most−Softmax r@3 = {d_sm3:+.4f}")
    if strong:
        print("SUKCES MOCNY: Most bije skalarne R o ≥+2 pkt (r@1)")
    print(f"WERDYKT FAZY C: {'PASS' if ok else 'FAIL'}")

    # markdown
    order = [
        "Softmax",
        "Most",
        "Most ZERO",
        "Most −depth",
        "Most −T",
        "Most −mazur",
        "flat payload",
        "R (depth dE)",
        "R (tracer L2)",
        "cosine only",
    ]
    lines = [
        "# Wyniki Faza C — recall / ranking Φ",
        "",
        "Pozytyw: ten sam **topic** + ta sama **depth** (bind w warstwie bańki).",
        "N=24; hard negs = ten sam topic, inna depth.",
        "Loss neural: InfoNCE; skalarne R = g·V²/(g²+dE²).",
        "Gate: **r@1** (r@3 raportowane — Softmax często saturuje top-3 same-topic).",
        "P2.13: T/state/mazur **dekorelowane** od depth.",
        "",
        "Korelacja osi (val[0]): "
        + ", ".join(f"{k}={v:.3f}" for k, v in corr.items()),
        "",
        "| wariant | r@1 | r@3 | r@5 | MRR |",
        "|---------|-----|-----|-----|-----|",
    ]
    for k in order:
        m = results[k]
        lines.append(
            f"| {k} | {m['r@1']:.3f} | {m['r@3']:.3f} | {m['r@5']:.3f} | {m['mrr']:.3f} |"
        )
    lines += [
        "",
        f"**Δ Most−Softmax r@1** = {d_sm:+.4f} → {'PASS' if beat_sm else 'FAIL'}",
        f"**Δ Most−R r@1** = {d_r:+.4f} → {'PASS' if not_worse_r else 'FAIL'}"
        + (" (mocny)" if strong else ""),
        f"(info) Δ Most−Softmax r@3 = {d_sm3:+.4f}",
        f"**Werdykt:** **{'PASS' if ok else 'FAIL'}**",
        "",
        "## Ablacje",
        "- Most ZERO / flat payload: kontrola obserwatora i „płaskiego pliku”.",
        "- Most −depth / −T / −mazur: osie Φ niezależne (po dekorelacji).",
        "",
        "## Go / no-go live Store",
        (
            "GO na Faza D (offline dump), potem live read-only — Most wnosi geometrię Φ."
            if ok
            else "NO-GO live: najpierw poprawić wyrównanie tracera do zadania recall."
        ),
    ]
    out_md = HISTORIA / "WYNIKI_RECALL.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zapisano: {out_md}")

    # wykres r@1
    fig, ax = plt.subplots(figsize=(9, 4))
    names = order
    vals = [results[k]["r@1"] for k in names]
    colors = [
        "#888",
        "#c44",
        "#c44",
        "#c44",
        "#c44",
        "#c44",
        "#c44",
        "#48a",
        "#48a",
        "#48a",
    ]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("recall@1")
    ax.set_title("PhiMost Faza C — recall@1")
    ax.axhline(results["Softmax"]["r@1"] + 0.02, color="gray", ls="--", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = HISTORIA / "phi_most_faza_c.png"
    plt.savefig(out_png, dpi=120)
    print(f"Zapisano: {out_png}")


if __name__ == "__main__":
    main()
