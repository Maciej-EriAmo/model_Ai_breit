"""
Skan pól/grid (nie znane AGN) — wysokie p_cluster = kandydat „coś nowego”.
Potencjalna nazwa użytkownika: JoannaM (dopiero po świadomym wyborze).
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

from bridge import BridgeTransformer
from config import CANON
from desi_cutouts import load_fits_bytes_or_path, pack_photometry
from image_batch import cutout_to_tokens

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "archiwum" / "ZdjęcieW" / "demo"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
OUT = ROOT / "doswiadczenie" / "joannam" / "JOANNAM_CANDIDATES.json"

SKIP_SUB = (
    "_bh_",
    "_ctrl_",
    "coma",
    "virgo",
    "hercules",
    "leo",
    "hydra",
    "fornax",
    "centaurus",
    "abell",
    "cluster",
    "perseus",
)


def main() -> int:
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = BridgeTransformer(
        input_dim=CANON.d_x,
        d=CANON.d_model,
        h=CANON.n_heads,
        n_classes=int(blob.get("n_classes", 2)),
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        coupling=CANON.coupling,
    )
    model.load_state_dict(blob["model"])
    model.eval()

    cands: list[Path] = []
    for p in sorted(DEMO.glob("desi_*_s128.fits")):
        n = p.name.lower()
        if any(x in n for x in SKIP_SUB):
            continue
        if "grid_" in n or "field_" in n or "new_field" in n:
            cands.append(p)

    rows = []
    with torch.no_grad():
        for fp in cands:
            img, inv = load_fits_bytes_or_path(fp)
            photo = pack_photometry(img, inv)
            c, tr = cutout_to_tokens(photo["image"], photo["T"], patch=16)
            x = torch.from_numpy(c).unsqueeze(0)
            t = torch.from_numpy(tr).unsqueeze(0)
            mask = torch.ones(1, x.shape[1])
            logits, _, dE = model(x, t, mask=mask)
            prob = torch.softmax(logits, dim=-1)[0]
            pc = float(prob[1]) if prob.numel() > 1 else 0.0
            rows.append(
                {
                    "file": fp.name,
                    "p_cluster": pc,
                    "T_mean": float(photo["T"].mean()),
                    "dE": float(dE.abs().mean()) if dE is not None else 0.0,
                    "pred": int(logits.argmax(1).item()),
                }
            )

    rows.sort(key=lambda r: -r["p_cluster"])
    top = [r for r in rows if r["p_cluster"] >= 0.85][:20]
    print(f"scored {len(rows)} field/grid candidates")
    print("TOP p_cluster>=0.85 (possible novel interest):")
    if top:
        for r in top:
            print(
                f"  {r['p_cluster']:.4f}  T={r['T_mean']:.3f}  "
                f"dE={r['dE']:.3f}  {r['file']}"
            )
    else:
        print("  (brak >=0.85) top5:")
        for r in rows[:5]:
            print(f"  {r['p_cluster']:.4f}  T={r['T_mean']:.3f}  {r['file']}")

    best = top[0] if top else (rows[0] if rows else None)
    note = (
        "NOT confirmed BH/discovery. Optical Legacy score only. "
        "If user accepts a candidate, propose name JoannaM."
    )
    out = {
        "n_scored": len(rows),
        "threshold": 0.85,
        "top": top,
        "top5": rows[:5],
        "best_candidate": best,
        "joannam_ready": bool(top),
        "note": note,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    if best and best["p_cluster"] >= 0.85:
        print(
            f"\n*** KANDYDAT do nazwy JoannaM (do Twojej decyzji): "
            f"{best['file']} p={best['p_cluster']:.3f} ***"
        )
    else:
        print("\n(Brak silnego nowego kandydata — nie nadajemy jeszcze JoannaM.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
