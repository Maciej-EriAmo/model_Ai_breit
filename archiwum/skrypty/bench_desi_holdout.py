"""
Duży trening DESI (barwa→T) + test na NOWYCH obiektach (hold-out never in train).

  python bench_desi_holdout.py --force-import --epochs 40 --augment 3
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
from desi_cutouts import DEMO_DIR, import_demo_fits, label_from_demo_name, load_cached
from image_batch import (
    ImageBatch,
    _items_from_files,
    _pack_batches,
    cutout_to_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
DOSWIADCZENIE_DESI = ROOT / "doswiadczenie" / "desi"
CKPT_DIR = ROOT / "checkpoints"


def _eval(model, batches, *, zero_tracer: bool = False) -> dict:
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    total_loss, total_acc, count = 0.0, 0.0, 0
    dE_vals: list[float] = []
    with torch.no_grad():
        for batch in batches:
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
        "n": count,
    }


def _train(model, train, *, zero_tracer: bool = False, epochs: int = 40):
    opt = torch.optim.Adam(model.parameters(), lr=CANON.lr)
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
    return model


def _predict_files(model, files: list[Path], *, zero_tracer: bool = False) -> list[dict]:
    model.eval()
    rows = []
    with torch.no_grad():
        for fp in files:
            d = load_cached(fp)
            c, tr = cutout_to_tokens(
                np.asarray(d["image"], dtype=np.float32),
                np.asarray(d["T"], dtype=np.float32),
                patch=16,
            )
            x = torch.from_numpy(c).unsqueeze(0)
            t = torch.from_numpy(tr).unsqueeze(0)
            if zero_tracer:
                t = torch.zeros_like(t)
            mask = torch.ones(1, x.shape[1])
            logits, _, dE = model(x, t, mask=mask)
            prob = torch.softmax(logits, dim=-1)[0]
            pred = int(logits.argmax(1).item())
            y = int(d["label"])
            rows.append(
                {
                    "file": fp.name,
                    "y": y,
                    "pred": pred,
                    "ok": pred == y,
                    "p_field": float(prob[0]),
                    "p_cluster": float(prob[1]) if prob.numel() > 1 else 0.0,
                    "dE": float(dE.abs().mean()) if dE is not None else 0.0,
                    "hint": str(d.get("hint", "")),
                }
            )
    return rows


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force-import", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--augment", type=int, default=3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert_canon_model_kwargs(CANON.coupling, CANON.num_layers)
    import_demo_fits(DEMO_DIR, force=args.force_import, prefer_size=args.size)

    cache = ROOT / "dumps" / "desi_cache"
    all_files = sorted(cache.glob("demo_*_s128.npz"))
    hold_path = DEMO_DIR / "holdout_stems.json"
    hold_stems: set[str] = set()
    if hold_path.is_file():
        hold_stems = set(json.loads(hold_path.read_text(encoding="utf-8")))

    hold_files = [
        p
        for p in all_files
        if "new_" in p.name or any(h in p.stem for h in hold_stems)
    ]
    train_pool = [p for p in all_files if p not in hold_files]

    rng = np.random.default_rng(args.seed)
    pool = list(train_pool)
    rng.shuffle(pool)
    n_val = max(1, int(round(len(pool) * args.val_frac))) if len(pool) > 2 else 1
    val_files = pool[:n_val]
    train_files = pool[n_val:]
    if not train_files:
        train_files, val_files = pool[:-1], pool[-1:]

    print(
        f"files train={len(train_files)} val={len(val_files)} "
        f"holdout_new={len(hold_files)}"
    )
    if len(hold_files) < 3:
        print("WARN: mało holdout — uruchom pobierz_duzy_set.py")

    d_x, n_dim = CANON.d_x, CANON.n_dim
    train_items = _items_from_files(
        train_files,
        patch=16,
        d_x=d_x,
        n_dim=n_dim,
        augment=args.augment,
        rng=rng,
        geometry_only=False,
    )
    val_items = _items_from_files(
        val_files, patch=16, d_x=d_x, n_dim=n_dim, augment=0, rng=rng, geometry_only=False
    )
    hold_items = _items_from_files(
        hold_files, patch=16, d_x=d_x, n_dim=n_dim, augment=0, rng=rng, geometry_only=False
    )
    rng.shuffle(train_items)
    train_b = _pack_batches(train_items, batch_size=8, d_x=d_x, n_dim=n_dim)
    val_b = _pack_batches(val_items, batch_size=8, d_x=d_x, n_dim=n_dim)
    hold_b = _pack_batches(hold_items, batch_size=8, d_x=d_x, n_dim=n_dim)

    ys = [int(load_cached(p)["label"]) for p in train_files + val_files + hold_files]
    n_classes = max(ys) + 1 if ys else 2

    torch.manual_seed(args.seed)
    sm = SoftmaxTransformer(
        input_dim=d_x, d=CANON.d_model, h=CANON.n_heads, n_classes=n_classes,
        dim_tracer=n_dim, num_layers=CANON.num_layers,
    )
    _train(sm, train_b, epochs=args.epochs)
    r_sm_val = _eval(sm, val_b)
    r_sm_new = _eval(sm, hold_b) if hold_b else {"acc": 0, "loss": 0, "mean_dE": 0, "n": 0}

    torch.manual_seed(args.seed)
    most = BridgeTransformer(
        input_dim=d_x, d=CANON.d_model, h=CANON.n_heads, n_classes=n_classes,
        dim_tracer=n_dim, num_layers=CANON.num_layers, coupling=CANON.coupling,
    )
    _train(most, train_b, epochs=args.epochs)
    r_most_val = _eval(most, val_b)
    r_most_new = _eval(most, hold_b) if hold_b else {"acc": 0, "loss": 0, "mean_dE": 0, "n": 0}

    torch.manual_seed(args.seed)
    zero = BridgeTransformer(
        input_dim=d_x, d=CANON.d_model, h=CANON.n_heads, n_classes=n_classes,
        dim_tracer=n_dim, num_layers=CANON.num_layers, coupling=CANON.coupling,
    )
    _train(zero, train_b, zero_tracer=True, epochs=args.epochs)
    r_z_val = _eval(zero, val_b, zero_tracer=True)
    r_z_new = _eval(zero, hold_b, zero_tracer=True) if hold_b else {"acc": 0, "loss": 0, "mean_dE": 0, "n": 0}

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / "most_desi_color.pt"
    torch.save(
        {"model": most.state_dict(), "n_classes": n_classes, "canon": "color_T"},
        ckpt,
    )

    gate = CANON.gate_classify_delta
    d_sm_val = r_most_val["acc"] - r_sm_val["acc"]
    d_z_val = r_most_val["acc"] - r_z_val["acc"]
    d_sm_new = r_most_new["acc"] - r_sm_new["acc"]
    d_z_new = r_most_new["acc"] - r_z_new["acc"]

    per_obj = _predict_files(most, hold_files)
    per_obj_sm = _predict_files(sm, hold_files)

    lines = [
        "# WYNIKI DESI — duży set + NOWE obiekty (hold-out)",
        "",
        f"- train={len(train_files)} val={len(val_files)} **new_holdout={len(hold_files)}**",
        f"- epochs={args.epochs} augment={args.augment} T=color(g−r) content=luminancja",
        f"- ckpt=`{ckpt}`",
        "",
        "## Val (widziane rozkładem, nie plikami holdout)",
        f"| | Softmax | Most | ZERO |",
        f"|--|---------|------|------|",
        f"| acc | {r_sm_val['acc']:.4f} | {r_most_val['acc']:.4f} | {r_z_val['acc']:.4f} |",
        f"| Δ Most−SM={d_sm_val:+.4f}  Δ Most−ZERO={d_z_val:+.4f}  gate={gate}",
        "",
        "## Nowe obiekty (nigdy w train)",
        f"| | Softmax | Most | ZERO |",
        f"|--|---------|------|------|",
        f"| acc | {r_sm_new['acc']:.4f} | {r_most_new['acc']:.4f} | {r_z_new['acc']:.4f} | n={r_most_new['n']} |",
        f"| **Δ Most−SM={d_sm_new:+.4f}**  **Δ Most−ZERO={d_z_new:+.4f}**",
        f"| Most−SM new → {'**PASS**' if d_sm_new >= gate else 'FAIL'} · "
        f"most vs ZERO → {'**PASS**' if d_z_new >= gate else 'FAIL'}",
        "",
        "## Per obiekt (Most)",
        "| plik | y | pred | ok | p_cluster | |dE| |",
        "|------|---|------|----|-----------|------|",
    ]
    for r in per_obj:
        lines.append(
            f"| `{r['file'][:40]}` | {r['y']} | {r['pred']} | "
            f"{'✓' if r['ok'] else '✗'} | {r['p_cluster']:.2f} | {r['dE']:.3f} |"
        )

    DOSWIADCZENIE_DESI.mkdir(parents=True, exist_ok=True)
    out = DOSWIADCZENIE_DESI / "WYNIKI_DESI_HOLDOUT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DOSWIADCZENIE_DESI / "desi_holdout_last.json").write_text(
        json.dumps(
            {
                "val": {"sm": r_sm_val, "most": r_most_val, "zero": r_z_val},
                "new": {"sm": r_sm_new, "most": r_most_new, "zero": r_z_new},
                "delta_sm_new": d_sm_new,
                "delta_zero_new": d_z_new,
                "per_object_most": per_obj,
                "per_object_sm": per_obj_sm,
                "n_train": len(train_files),
                "n_val": len(val_files),
                "n_hold": len(hold_files),
                "ckpt": str(ckpt),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for line in lines:
        print(line)
    print(f"wrote {out}")
    return 0 if d_sm_new >= gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
