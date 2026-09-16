"""Leave-one-out CV bh_* vs reszta — Softmax | Most | ZERO.

Filtr jakości cutoutów (domyślnie ON):
  - pusty obraz (max≈0) → invalid (np. cena poza footprint)
  - snr_core_max < --min-snr → za słaby (np. ton618)

Użycie:
  python check_bh_loo.py
  python check_bh_loo.py --no-filter          # stary n=11
  python check_bh_loo.py --min-snr 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from bridge import BridgeTransformer, SoftmaxTransformer
from config import CANON
from desi_cutouts import CACHE, load_cached
from image_batch import _augment_flip, _pack_batches, cutout_to_tokens

DOSWIADCZENIE = ROOT / "doswiadczenie" / "desi"
SEED = 0
VAL_FRAC_NON = 0.3
EPOCHS = 40
AUGMENT = 4


def _center_mask(h: int, w: int, frac: float = 0.08) -> np.ndarray:
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = frac * min(h, w)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def cutout_quality(path: Path) -> dict:
    """Metryki jakości + powód odrzucenia (lub None)."""
    d = load_cached(path)
    img = np.asarray(d["image"], dtype=np.float32)
    snr = np.asarray(d["snr"], dtype=np.float32)
    H, W = snr.shape
    core = _center_mask(H, W, 0.08)
    snr_core_max = float(snr[core].max()) if core.any() else float(snr.max())
    img_max = float(img.max())
    img_mean = float(img.mean())
    empty = img_max < 1e-6
    return {
        "path": str(path),
        "stem": path.stem,
        "img_max": img_max,
        "img_mean": img_mean,
        "snr_core_max": snr_core_max,
        "empty": empty,
        "shape_ok": img.shape[-1] == 128 and img.shape[-2] == 128,
    }


def is_valid(q: dict, *, min_snr: float, filter_on: bool) -> tuple[bool, str | None]:
    if not q["shape_ok"]:
        return False, "bad_shape"
    if not filter_on:
        return True, None
    if q["empty"]:
        return False, "empty"
    if q["snr_core_max"] < min_snr:
        return False, f"snr_core_max<{min_snr:g}"
    return True, None


def _build_items(files, label, augment, rng):
    items = []
    for p in files:
        d = load_cached(p)
        base_img = np.asarray(d["image"], dtype=np.float32)
        base_T = np.asarray(d["T"], dtype=np.float32)
        variants = [(base_img, base_T)]
        for _ in range(augment):
            variants.append(_augment_flip(base_img, base_T, rng))
        for img, Tm in variants:
            c, t = cutout_to_tokens(
                img, Tm, patch=16, d_x=CANON.d_x, n_dim=CANON.n_dim
            )
            items.append((c, t, label))
    return items


def train_eval_weighted(
    model,
    train,
    val,
    class_weights,
    *,
    zero_tracer=False,
    epochs=EPOCHS,
    lr=CANON.lr,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)
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
    all_pred, all_y = [], []
    with torch.no_grad():
        for batch in val:
            x, y = batch.content, batch.labels
            tr = torch.zeros_like(batch.tracer) if zero_tracer else batch.tracer
            logits, _, _ = model(x, tr, mask=batch.mask)
            all_pred.append(logits.argmax(1))
            all_y.append(y)
    pred = torch.cat(all_pred)
    y = torch.cat(all_y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def short_name(stem: str) -> str:
    # demo_desi_bh_3c345_ra... → 3c345
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p == "bh" and i + 1 < len(parts):
            return parts[i + 1]
    return stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-snr", type=float, default=20.0,
                    help="odrzuć cutout gdy snr_core_max < próg (default 20)")
    ap.add_argument("--no-filter", action="store_true",
                    help="wyłącz filtr jakości (stary n=11)")
    ap.add_argument("--out-json", type=Path,
                    default=DOSWIADCZENIE / "loo_bh_filtered.json")
    ap.add_argument("--out-md", type=Path,
                    default=DOSWIADCZENIE / "WYNIKI_LOO_BH_FILTERED.md")
    args = ap.parse_args(argv)
    filter_on = not args.no_filter

    rng_master = np.random.default_rng(args.seed)
    all_files = sorted(p for p in CACHE.glob("demo_*.npz"))

    files_bh: list[Path] = []
    files_non: list[Path] = []
    rejected: list[dict] = []

    for p in all_files:
        q = cutout_quality(p)
        ok, reason = is_valid(q, min_snr=args.min_snr, filter_on=filter_on)
        # filtr SNR/empty tylko dla bh_*; nie-bh: tylko empty/shape
        is_bh = "bh_" in p.stem
        if is_bh:
            if not ok:
                rejected.append({**q, "reason": reason, "role": "bh"})
                continue
            files_bh.append(p)
        else:
            if not q["shape_ok"] or q["empty"]:
                rejected.append({
                    **q,
                    "reason": "bad_shape" if not q["shape_ok"] else "empty",
                    "role": "non",
                })
                continue
            # nie-bh: nie stosuj min-snr (puste pola mogą być ciemne)
            files_non.append(p)

    print(f"filtr={'ON' if filter_on else 'OFF'} min_snr={args.min_snr}")
    print(f"pliki: bh={len(files_bh)} nie-bh={len(files_non)} rejected={len(rejected)}")
    for r in rejected:
        print(f"  REJECT [{r['role']}] {short_name(r['stem'])}: {r['reason']} "
              f"(snr_core_max={r['snr_core_max']:.2f} img_max={r['img_max']:.4g})")
    print("bh kept:", ", ".join(short_name(p.stem) for p in files_bh))

    if len(files_bh) < 2:
        print("ERROR: za mało bh po filtrze")
        return 1

    idx_non = list(files_non)
    rng_master.shuffle(idx_non)
    n_val_non = max(1, int(round(len(idx_non) * VAL_FRAC_NON)))
    val_non = idx_non[:n_val_non]
    train_non = idx_non[n_val_non:]
    print(f"nie-bh: train={len(train_non)} val={len(val_non)}")

    agg = {
        name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "caught": [], "folds": []}
        for name in ("Softmax", "Most", "ZERO")
    }
    fold_meta = []
    t0 = time.time()

    for fold, held_out in enumerate(files_bh):
        train_bh = [p for p in files_bh if p != held_out]
        val_bh = [held_out]
        held = short_name(held_out.stem)
        print(f"\n--- FOLD {fold+1}/{len(files_bh)}: held_out={held} ---", flush=True)

        fold_rng = np.random.default_rng(1000 + fold)
        train_items = (
            _build_items(train_bh, 1, AUGMENT, fold_rng)
            + _build_items(train_non, 0, AUGMENT, fold_rng)
        )
        val_items = (
            _build_items(val_bh, 1, 0, fold_rng)
            + _build_items(val_non, 0, 0, fold_rng)
        )
        fold_rng.shuffle(train_items)

        n_bh_items = sum(1 for *_, y in train_items if y == 1)
        n_non_items = len(train_items) - n_bh_items
        class_weights = torch.tensor(
            [1.0, n_non_items / max(1, n_bh_items)], dtype=torch.float32
        )

        train_b = _pack_batches(
            train_items, batch_size=8, d_x=CANON.d_x, n_dim=CANON.n_dim
        )
        val_b = _pack_batches(
            val_items, batch_size=8, d_x=CANON.d_x, n_dim=CANON.n_dim
        )

        fold_row = {"held_out": held, "stem": held_out.stem, "models": {}}
        for name, kwargs in (
            ("Softmax", dict(cls=SoftmaxTransformer, zero_tracer=False)),
            ("Most", dict(cls=BridgeTransformer, zero_tracer=False)),
            ("ZERO", dict(cls=BridgeTransformer, zero_tracer=True)),
        ):
            torch.manual_seed(1000 + fold)
            if kwargs["cls"] is SoftmaxTransformer:
                model = SoftmaxTransformer(
                    input_dim=CANON.d_x,
                    d=CANON.d_model,
                    h=CANON.n_heads,
                    n_classes=2,
                    dim_tracer=CANON.n_dim,
                    num_layers=CANON.num_layers,
                )
            else:
                model = BridgeTransformer(
                    input_dim=CANON.d_x,
                    d=CANON.d_model,
                    h=CANON.n_heads,
                    n_classes=2,
                    dim_tracer=CANON.n_dim,
                    num_layers=CANON.num_layers,
                    coupling=CANON.coupling,
                )
            tp, fp, fn, tn = train_eval_weighted(
                model,
                train_b,
                val_b,
                class_weights,
                zero_tracer=kwargs["zero_tracer"],
                epochs=args.epochs,
            )
            agg[name]["tp"] += tp
            agg[name]["fp"] += fp
            agg[name]["fn"] += fn
            agg[name]["tn"] += tn
            agg[name]["caught"].append(tp)
            fold_row["models"][name] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "caught": bool(tp == 1),
            }
            print(
                f"  {name:8s} tp={tp} fp={fp} fn={fn} tn={tn}  "
                f"{'CAUGHT' if tp == 1 else 'MISSED'}",
                flush=True,
            )
        fold_meta.append(fold_row)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(
        f"WYNIKI LOO-CV (n={len(files_bh)} foldow, val_non={len(val_non)}, "
        f"filter={'ON' if filter_on else 'OFF'}, {elapsed:.0f}s)"
    )
    print("=" * 70)

    summary = {}
    for name, a in agg.items():
        tp, fp, fn, tn = a["tp"], a["fp"], a["fn"], a["tn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        caught_str = "".join("1" if c else "0" for c in a["caught"])
        per_obj = {
            short_name(files_bh[i].stem): bool(a["caught"][i])
            for i in range(len(files_bh))
        }
        summary[name] = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "pattern": caught_str,
            "per_object": per_obj,
        }
        print(
            f"{name:8s} recall={recall:.3f} ({tp}/{tp+fn})  "
            f"precision={precision:.4f} ({tp}/{tp+fp})  "
            f"F1={f1:.3f}  pattern={caught_str}"
        )
        missed = [k for k, v in per_obj.items() if not v]
        caught = [k for k, v in per_obj.items() if v]
        print(f"         caught={caught}  missed={missed}")

    payload = {
        "protocol": "loo-bh-v2-filtered",
        "filter_on": filter_on,
        "min_snr": args.min_snr,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_bh": len(files_bh),
        "n_non_train": len(train_non),
        "n_non_val": len(val_non),
        "bh_kept": [short_name(p.stem) for p in files_bh],
        "rejected": [
            {
                "name": short_name(r["stem"]),
                "reason": r["reason"],
                "snr_core_max": r["snr_core_max"],
                "img_max": r["img_max"],
                "role": r["role"],
            }
            for r in rejected
        ],
        "elapsed_s": elapsed,
        "baseline_unfiltered": {
            "note": "n=11 z poprzedniej sesji (bez filtra)",
            "Softmax": {"recall": 0.545, "precision": 0.462, "f1": 0.500},
            "Most": {"recall": 0.636, "precision": 0.304, "f1": 0.412},
            "ZERO": {"recall": 0.545, "precision": 0.333, "f1": 0.414},
        },
        "summary": summary,
        "folds": fold_meta,
    }

    DOSWIADCZENIE.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nJSON → {args.out_json}")

    lines = [
        "# LOO-CV bh-detekcja — po filtrze jakości",
        "",
        f"**Filtr:** `{'ON' if filter_on else 'OFF'}` · `min_snr={args.min_snr}` · "
        f"epochs={args.epochs} · n_bh={len(files_bh)} · {elapsed:.0f}s",
        "",
        "## Odrzucone",
        "",
    ]
    if rejected:
        lines.append("| obiekt | powód | snr_core_max | img_max |")
        lines.append("|--------|-------|--------------|---------|")
        for r in rejected:
            if r["role"] != "bh":
                continue
            lines.append(
                f"| `{short_name(r['stem'])}` | {r['reason']} | "
                f"{r['snr_core_max']:.2f} | {r['img_max']:.4g} |"
            )
    else:
        lines.append("_(brak)_")
    lines += [
        "",
        f"**Kept:** {', '.join('`'+short_name(p.stem)+'`' for p in files_bh)}",
        "",
        "## Wyniki vs baseline (n=11 bez filtra)",
        "",
        "| model | recall | precision | F1 | recall n=11 |",
        "|-------|--------|-----------|----|-------------|",
    ]
    base = payload["baseline_unfiltered"]
    for name in ("Softmax", "Most", "ZERO"):
        s = summary[name]
        b = base[name]
        lines.append(
            f"| {name} | {s['recall']:.3f} ({s['tp']}/{s['tp']+s['fn']}) | "
            f"{s['precision']:.3f} | {s['f1']:.3f} | {b['recall']:.3f} |"
        )
    lines += ["", "## Per obiekt (1=caught)", "", "| obiekt | Softmax | Most | ZERO |",
              "|--------|---------|------|------|"]
    for p in files_bh:
        n = short_name(p.stem)
        lines.append(
            f"| `{n}` | "
            f"{'1' if summary['Softmax']['per_object'][n] else '0'} | "
            f"{'1' if summary['Most']['per_object'][n] else '0'} | "
            f"{'1' if summary['ZERO']['per_object'][n] else '0'} |"
        )
    lines += [
        "",
        f"JSON: `{args.out_json.name}`",
        "",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD   → {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
