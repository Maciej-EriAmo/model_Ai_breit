"""
Walidacja fizyczna PhiMost: Softmax | Most | ZERO

Zadania:
  - solar  : Układ Słoneczny (star/rocky/giant + Moon offset), label warstw
  - three  : 3 ciała + Verlet, label = interakcje wewnątrz warstw masy

Kanon: softplus×2, n_dim=6, gate Most−SM ≥ +3 pkt (classify).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bridge import BridgeTransformer, SoftmaxTransformer
from config import CANON, assert_canon_model_kwargs
from nbody_data import make_nbody_loaders, NBodyBatch

ROOT = Path(__file__).resolve().parent
HISTORIA = ROOT / "historia"


def train_eval_nbody(
    model,
    train: list[NBodyBatch],
    val: list[NBodyBatch],
    *,
    zero_tracer: bool = False,
    epochs: int = 10,
    lr: float = CANON.lr,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch in train:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            opt.zero_grad()
            logits, _, _ = model(x, tr)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    dE_vals = []
    with torch.no_grad():
        for batch in val:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            logits, _, dE = model(x, tr)
            loss = crit(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * x.shape[0]
            total_acc += (pred == y).float().sum().item()
            count += x.shape[0]
            if dE is not None:
                dE_vals.append(dE.abs().mean().item())
    return {
        "loss": total_loss / count,
        "acc": total_acc / count,
        "mean_dE": float(np.mean(dE_vals)) if dE_vals else 0.0,
    }


def run_kind(kind: str) -> dict:
    assert_canon_model_kwargs(CANON.coupling, CANON.num_layers)
    d_x, n_dim = CANON.d_x, CANON.n_dim
    train, val = make_nbody_loaders(
        kind=kind,
        n_train=80,
        n_val=15,
        batch_size=32,
        d_x=d_x,
        n_dim=n_dim,
        include_moon=(kind == "solar"),
    )

    print(f"\n=== {kind.upper()} === softplus×{CANON.num_layers} n_dim={n_dim}")
    torch.manual_seed(42)
    sm = SoftmaxTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=CANON.n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
    )
    r_sm = train_eval_nbody(sm, train, val)

    # t_seal=False: tracer[...,3] tu koduje znormalizowaną energię Kepler
    # (patrz nbody_data.py), nie T/świeżość Karmazyn (HOT/WARM/COLD/TOMB) —
    # bramka most_seal.apply_t_key_seal zakłada tę drugą semantykę i przy
    # tej pierwszej łamie fizyczny bench (zob. docs/AUDYT.md, seal audit).
    torch.manual_seed(42)
    most = BridgeTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=CANON.n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
        t_seal=False,
    )
    r_most = train_eval_nbody(most, train, val)

    torch.manual_seed(42)
    most0 = BridgeTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=CANON.n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
        t_seal=False,
    )
    r_zero = train_eval_nbody(most0, train, val, zero_tracer=True)

    delta = r_most["acc"] - r_sm["acc"]
    d_z = r_zero["acc"] - r_sm["acc"]
    tag = "PASS" if delta >= CANON.gate_classify_delta else "FAIL"
    print(
        f"SM={r_sm['acc']:.4f}  Most={r_most['acc']:.4f}  "
        f"Δ={delta:+.4f}  ZERO={r_zero['acc']:.4f}  ΔZ={d_z:+.4f}  "
        f"|dE|={r_most['mean_dE']:.4f}  {tag}"
    )
    return {
        "sm": r_sm,
        "most": r_most,
        "zero": r_zero,
        "delta": delta,
        "pass": delta >= CANON.gate_classify_delta,
    }


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    print("PhiMost AUDYT FIZYCZNY — kanon", CANON)
    print("Zakaz EMA-S:", not CANON.allow_ema_superposition)
    results = {}
    for kind in ("solar", "three"):
        results[kind] = run_kind(kind)

    wins = sum(1 for r in results.values() if r["pass"])
    ok = wins >= 1  # ≥1 zadanie fizyczne z +3 pkt
    strong = wins == 2

    lines = [
        "# Wyniki — walidacja n-ciał / Układ Słoneczny",
        "",
        f"Kanon: `{CANON.coupling}×{CANON.num_layers}`, d={CANON.d_model}, "
        f"n_dim={CANON.n_dim}, gate Δ≥{CANON.gate_classify_delta}.",
        "",
        "| zadanie | Softmax | Most | ZERO | Most−SM | |",
        "|---------|---------|------|------|---------|---|",
    ]
    for kind, r in results.items():
        lines.append(
            f"| {kind} | {r['sm']['acc']:.4f} | {r['most']['acc']:.4f} | "
            f"{r['zero']['acc']:.4f} | {r['delta']:+.4f} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**Werdykt:** **{'PASS' if ok else 'FAIL'}** "
        f"({wins}/2 zadań ≥+{CANON.gate_classify_delta*100:.0f} pkt)"
        + (" — mocny" if strong else ""),
        "",
        "## Interpretacja",
        "- **solar**: warstwy star / rocky / giant (+Księżyc w rocky); energia Kepler w tracerze; content bez masy/hierarchii.",
        "- **three**: 3 ciała + Verlet; warstwy = rank masy; label = interakcje wewnątrz warstw (jak drzewo lab).",
        "- ZERO ≈ Softmax ⇒ bez τ most nie powstaje (τ = warunek, nie handicap).",
        "",
        "## Config po audycie",
        "- `config.CANON` — jedyne źródło domyślnych hiperparametrów.",
        "- `hash_embed` stabilny (SHA256).",
        "- tracer n_dim=6 zawiera mazur (dim5).",
        "",
    ]
    HISTORIA.mkdir(exist_ok=True)
    out = HISTORIA / "WYNIKI_NBODY.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("Zapisano:", out)

    fig, ax = plt.subplots(figsize=(7, 4))
    kinds = list(results.keys())
    x = np.arange(len(kinds))
    w = 0.25
    ax.bar(x - w, [results[k]["sm"]["acc"] for k in kinds], w, label="Softmax")
    ax.bar(x, [results[k]["most"]["acc"] for k in kinds], w, label="Most")
    ax.bar(x + w, [results[k]["zero"]["acc"] for k in kinds], w, label="ZERO")
    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("val acc")
    ax.set_title("PhiMost — solar / three-body")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = HISTORIA / "phi_most_nbody.png"
    plt.savefig(out_png, dpi=120)
    print(f"Zapisano: {out_png}")

    # update AUDYT
    aud = ROOT / "docs" / "AUDYT.md"
    prev = aud.read_text(encoding="utf-8") if aud.exists() else ""
    if "## Walidacja n-ciał" not in prev:
        aud.write_text(
            prev
            + "\n## Walidacja n-ciał\n\n"
            + f"Zob. `historia/WYNIKI_NBODY.md` — werdykt **{'PASS' if ok else 'FAIL'}** "
            + f"({wins}/2).\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
