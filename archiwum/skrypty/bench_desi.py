"""
Softmax | Most | ZERO na cutoutach DESI (patchy, T≠XY).

Kanon: tracer = obserwator gwarantujący prawidłowy most.
Główny warunek: Most vs ZERO (Δacc ≥ CANON.gate_classify_delta).
Softmax = twin bez obserwatora (osobna metryka).

  python bench_desi.py --demo-only --force-import --epochs 24 --augment 6
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
from pathlib import Path

import numpy as np
import torch

from bridge import BridgeTransformer, SoftmaxTransformer
from config import CANON, assert_canon_model_kwargs
from image_batch import ImageBatch, load_desi_train_val

ROOT = Path(__file__).resolve().parents[2]
DOSWIADCZENIE_DESI = ROOT / "archiwum" / "desi"


def train_eval_image(
    model,
    train: list[ImageBatch],
    val: list[ImageBatch],
    *,
    zero_tracer: bool = False,
    epochs: int = 12,
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
            logits, _, _ = model(x, tr, mask=batch.mask)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

    model.eval()
    total_loss, total_acc, count = 0.0, 0.0, 0
    dE_vals: list[float] = []
    with torch.no_grad():
        for batch in val:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            logits, _, dE = model(x, tr, mask=batch.mask)
            loss = crit(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * x.shape[0]
            total_acc += (pred == y).float().sum().item()
            count += x.shape[0]
            if dE is not None:
                dE_vals.append(dE.abs().mean().item())
    return {
        "loss": total_loss / max(count, 1),
        "acc": total_acc / max(count, 1),
        "mean_dE": float(np.mean(dE_vals)) if dE_vals else 0.0,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--geometry-only",
        action="store_true",
        help="zaszum content — Softmax bez fotometrii; Most musi użyć τ",
    )
    ap.add_argument(
        "--from-demo",
        action="store_true",
        help="zaimportuj ZdjęcieW/demo/desi_*.fits do cache",
    )
    ap.add_argument(
        "--demo-only",
        action="store_true",
        help="trenuj tylko na demo_*.npz (zdjęcia z projektu demo/)",
    )
    ap.add_argument(
        "--force-import",
        action="store_true",
        help="przebuduj demo_*.npz z FITS nawet gdy cache istnieje",
    )
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--augment", type=int, default=6)
    ap.add_argument(
        "--size",
        type=int,
        default=128,
        help="jednolity bok cutoutu (domyślnie 128 — bez mieszania N)",
    )
    ap.add_argument("--val-frac", type=float, default=0.34)
    args = ap.parse_args()

    assert_canon_model_kwargs(CANON.coupling, CANON.num_layers)
    d_x, n_dim = CANON.d_x, CANON.n_dim
    train, val, meta = load_desi_train_val(
        patch=16,
        batch_size=4,
        d_x=d_x,
        n_dim=n_dim,
        augment=args.augment,
        geometry_only=args.geometry_only,
        from_demo=args.from_demo or args.demo_only,
        demo_only=args.demo_only,
        force_import=args.force_import,
        size=args.size,
        val_frac=args.val_frac,
    )
    ys = torch.cat([b.labels for b in train + val])
    n_classes = int(ys.max().item()) + 1
    print(
        f"DESI bench: train_batches={len(train)} val_batches={len(val)} "
        f"files={meta['n_train_files']}/{meta['n_val_files']} "
        f"items={meta['n_train_items']}/{meta['n_val_items']} "
        f"N={meta['seq_len']} size={meta['size']} "
        f"classes={n_classes} geometry_only={args.geometry_only}"
    )
    # assert no file leak
    assert not (set(meta["train_files"]) & set(meta["val_files"])) or meta[
        "n_files"
    ] == 1

    torch.manual_seed(42)
    sm = SoftmaxTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
    )
    r_sm = train_eval_image(sm, train, val, epochs=args.epochs)

    # t_seal=False: oś 3 to barwa (percentyl g−r), nie temperatura archiwum.
    torch.manual_seed(42)
    most = BridgeTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
        t_seal=False,
    )
    r_most = train_eval_image(most, train, val, epochs=args.epochs)

    torch.manual_seed(42)
    most0 = BridgeTransformer(
        input_dim=d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=n_classes,
        dim_tracer=n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
        t_seal=False,
    )
    r_zero = train_eval_image(
        most0, train, val, zero_tracer=True, epochs=args.epochs
    )

    d_sm = r_most["acc"] - r_sm["acc"]
    d_z = r_most["acc"] - r_zero["acc"]
    gate = CANON.gate_classify_delta
    bridge_ok = d_z >= gate
    vs_sm = d_sm >= gate
    tag_bridge = "PASS" if bridge_ok else "FAIL"
    tag_sm = "PASS" if vs_sm else "FAIL"

    lines = [
        "# WYNIKI DESI (patchy cutouts) — po audycie",
        "",
        "**Kanon:** tracer = obserwator gwarantujący prawidłowy most.",
        "Główny warunek = **Most vs ZERO**. Softmax = twin bez obserwatora.",
        "",
        f"- Softmax acc={r_sm['acc']:.4f} loss={r_sm['loss']:.4f}  (brak τ)",
        f"- Most     acc={r_most['acc']:.4f} loss={r_most['loss']:.4f} |dE|={r_most['mean_dE']:.4f}",
        f"- ZERO     acc={r_zero['acc']:.4f} loss={r_zero['loss']:.4f}  (Most z τ=0)",
        f"- **Δ(Most−ZERO)={d_z:+.4f}**  gate={gate} → most prawidłowy? **{tag_bridge}**",
        f"- Δ(Most−SM)={d_sm:+.4f}  gate={gate} → vs Softmax **{tag_sm}**",
        "",
        f"- size={meta['size']} N={meta['seq_len']} "
        f"train_files={meta['n_train_files']} val_files={meta['n_val_files']} "
        f"(augment tylko train; maska w uwadze)",
        f"- geometry_only={args.geometry_only} demo_only={args.demo_only}",
        "",
        "Most<SM nie znaczy „tracer zbędny” — Softmax nie formuje mostu.",
    ]
    DOSWIADCZENIE_DESI.mkdir(parents=True, exist_ok=True)
    out_md = DOSWIADCZENIE_DESI / (
        "WYNIKI_DESI_DEMO.md" if args.demo_only else "WYNIKI_DESI.md"
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DOSWIADCZENIE_DESI / "desi_last.json").write_text(
        json.dumps(
            {
                "softmax": r_sm,
                "most": r_most,
                "zero": r_zero,
                "delta_sm": d_sm,
                "delta_zero": d_z,
                "gate": gate,
                "bridge_ok_most_vs_zero": bridge_ok,
                "vs_softmax": vs_sm,
                "tag_bridge": tag_bridge,
                "tag_sm": tag_sm,
                "meta": meta,
                "canon": "tracer=observer; primary=Most-ZERO; split-by-file; size-uniform; mask",
                "audit_fix": "2026-09-10",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for line in lines:
        print(line)
    print(f"wrote {out_md}")
    return 0 if bridge_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
