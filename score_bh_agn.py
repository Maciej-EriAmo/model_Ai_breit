"""
Score cutoutów bh_* / ctrl_* modelem most_desi_color.pt

  python score_bh_agn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bridge import BridgeTransformer
from config import CANON
from desi_cutouts import (
    CACHE,
    label_from_demo_name,
    load_fits_bytes_or_path,
    pack_photometry,
)
from image_batch import cutout_to_tokens

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "ZdjęcieW" / "demo"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
HISTORIA = ROOT / "historia"


def _ensure_npz(fits_path: Path, labels_map: dict) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"demo_{fits_path.stem}.npz"
    img, inv = load_fits_bytes_or_path(fits_path)
    photo = pack_photometry(img, inv)
    label = label_from_demo_name(fits_path.stem, labels_map)
    hint = "cluster" if label == 1 else "field"
    np.savez_compressed(
        out,
        image=photo["image"],
        snr=photo["snr"],
        color_gr=photo["color_gr"],
        color_rz=photo["color_rz"],
        T=photo["T"],
        T_mode=np.asarray(photo["T_mode"]),
        label=np.int64(label),
        name=np.asarray(fits_path.stem),
        hint=np.asarray(hint),
        has_invvar=np.bool_(inv is not None),
        source=np.asarray(str(fits_path)),
    )
    return out


def main() -> int:
    if not CKPT.is_file():
        raise SystemExit(f"brak {CKPT} — najpierw bench_desi_holdout.py")

    labels_map = {}
    if (DEMO / "labels.json").is_file():
        labels_map = json.loads((DEMO / "labels.json").read_text(encoding="utf-8"))

    fits_list = sorted(
        p
        for p in DEMO.glob("desi_*_s128.fits")
        if "_bh_" in p.name or "_ctrl_" in p.name
    )
    if not fits_list:
        raise SystemExit("brak bh_/ctrl_ FITS — uruchom ZdjęcieW/pobierz_bh_agn.py")

    files = [_ensure_npz(fp, labels_map) for fp in fits_list]

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    n_classes = int(blob.get("n_classes", 2))
    model = BridgeTransformer(
        input_dim=CANON.d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=n_classes,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
    )
    model.load_state_dict(blob["model"])
    model.eval()

    kinds = {}
    lab_path = DEMO / "labels.json"
    if lab_path.is_file():
        raw = json.loads(lab_path.read_text(encoding="utf-8"))
        for k, v in raw.items():
            if k.startswith("kind:"):
                kinds[k[5:]] = v

    rows = []
    with torch.no_grad():
        for fp in files:
            d = {k: np.load(fp, allow_pickle=True)[k] for k in np.load(fp, allow_pickle=True).files}
            c, tr = cutout_to_tokens(
                np.asarray(d["image"], dtype=np.float32),
                np.asarray(d["T"], dtype=np.float32),
                patch=16,
            )
            x = torch.from_numpy(c).unsqueeze(0)
            t = torch.from_numpy(tr).unsqueeze(0)
            mask = torch.ones(1, x.shape[1])
            logits, _, dE = model(x, t, mask=mask)
            prob = torch.softmax(logits, dim=-1)[0]
            pred = int(logits.argmax(1).item())
            stem = str(d.get("name", fp.stem))
            # name w npz to FITS stem
            kind = kinds.get(stem, "agn" if "_bh_" in fp.name else "ctrl")
            T_mean = float(np.asarray(d["T"]).mean())
            rows.append(
                {
                    "file": fp.name,
                    "stem": stem,
                    "kind": kind,
                    "pred": pred,
                    "label_name": "cluster" if pred == 1 else "field",
                    "p_cluster": float(prob[1]) if prob.numel() > 1 else 0.0,
                    "p_field": float(prob[0]),
                    "T_mean": T_mean,
                    "dE": float(dE.abs().mean()) if dE is not None else 0.0,
                    "is_bh_target": "_bh_" in fp.name,
                }
            )

    rows.sort(key=lambda r: -r["p_cluster"])
    lines = [
        "# Poszukiwanie śladów czarnych dziur (AGN/kwazar) — score Most",
        "",
        "**Uwaga naukowa:** model nie widzi horyzontu. Widzi cutout Legacy; "
        "trening był cluster/field. Tu: czy znane AGN wyglądają „gorąco/klastrowo” "
        "w barwie→T vs kontrolne pola.",
        "",
        f"ckpt=`{CKPT.name}` · n={len(rows)}",
        "",
        "| obiekt | kind | pred | p_cluster | T_mean | |dE| |",
        "|--------|------|------|-----------|--------|------|",
    ]
    for r in rows:
        tag = "BH/AGN" if r["is_bh_target"] else "CTRL"
        short = r["stem"].replace("desi_", "")[:42]
        lines.append(
            f"| `{short}` | {r['kind']}/{tag} | {r['label_name']} | "
            f"{r['p_cluster']:.3f} | {r['T_mean']:.3f} | {r['dE']:.3f} |"
        )

    bh = [r for r in rows if r["is_bh_target"]]
    ctrl = [r for r in rows if not r["is_bh_target"]]
    if bh and ctrl:
        lines += [
            "",
            "## Skrót",
            f"- AGN/BH targets: mean p_cluster={np.mean([r['p_cluster'] for r in bh]):.3f} "
            f"(n={len(bh)})",
            f"- CTRL fields: mean p_cluster={np.mean([r['p_cluster'] for r in ctrl]):.3f} "
            f"(n={len(ctrl)})",
        ]

    HISTORIA.mkdir(parents=True, exist_ok=True)
    out = HISTORIA / "WYNIKI_BH_AGN_SCORE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HISTORIA / "bh_agn_score.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    for line in lines:
        print(line)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
