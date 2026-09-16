"""Ablacja JoannyM: zamaskuj gwiazdę Gaia @~8.8\" i sprawdź Δ p_cluster.

  cd "C:\\Users\\drwis\\model Ai"
  python doswiadczenie/joannam/ablate_star_mask.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridge import BridgeTransformer
from config import CANON
from desi_cutouts import load_fits_bytes_or_path, pack_photometry
from image_batch import cutout_to_tokens
FITS = ROOT / "ZdjęcieW" / "demo" / "desi_grid_004_ra120.0_dec5.0_s128.fits"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
OUT_JSON = Path(__file__).with_name("ABLATION_STAR_MASK.json")
OUT_MD = Path(__file__).with_name("ABLATION_STAR_MASK.md")

RA0, DEC0 = 120.0, 5.0
PIXSCALE = 0.262  # arcsec/px (Legacy cutout)

# LS DR10 tractor bright Gaia stars inside cutout
STARS = [
    {
        "name": "gaia_8p8",
        "ra": 120.0022588903326,
        "dec": 4.999042584188773,
        "dist_arcsec": 8.804,
        "mag_r": 17.676,
        "ref": "GE:3095364135451155712",
    },
    {
        "name": "gaia_17",
        "ra": 119.9955724982025,
        "dec": 4.998323810555948,
        "dist_arcsec": 16.986,
        "mag_r": 19.209,
        "ref": "GE:3095364101091415936",
    },
]


def sky_to_xy(
    ra: float,
    dec: float,
    *,
    h: int,
    w: int,
    ra0: float = RA0,
    dec0: float = DEC0,
    pixscale: float = PIXSCALE,
) -> tuple[float, float]:
    """Legacy cutout: center = (ra0,dec0), North up, East left (FITS row0 = South)."""
    cosd = math.cos(math.radians(dec0))
    dra_as = (ra - ra0) * cosd * 3600.0
    ddec_as = (dec - dec0) * 3600.0
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    # East left → −x with +RA; North up with FITS origin bottom-left → +y with +Dec
    x = cx - dra_as / pixscale
    y = cy + ddec_as / pixscale
    return x, y


def refine_peak(lum: np.ndarray, x0: float, y0: float, search_r: int = 4) -> tuple[int, int, float]:
    """Snap to local luminance max near predicted star pixel."""
    H, W = lum.shape
    xi, yi = int(round(x0)), int(round(y0))
    best = (xi, yi, float(lum[np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]))
    for dy in range(-search_r, search_r + 1):
        for dx in range(-search_r, search_r + 1):
            x, y = xi + dx, yi + dy
            if 0 <= x < W and 0 <= y < H:
                v = float(lum[y, x])
                if v > best[2]:
                    best = (x, y, v)
    return best


def circular_mask(h: int, w: int, x: float, y: float, radius_px: float) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - x) ** 2 + (yy - y) ** 2) <= radius_px**2


def apply_mask(img: np.ndarray, mask: np.ndarray, mode: str = "median") -> np.ndarray:
    """img [C,H,W]; mask [H,W] True = replace."""
    out = img.copy()
    C, H, W = out.shape
    if mode == "zero":
        out[:, mask] = 0.0
        return out
    # fill with global median of unmasked (robust, no ring artifact from local empty)
    for c in range(C):
        band = out[c]
        fill = float(np.median(band[~mask])) if np.any(~mask) else 0.0
        band = band.copy()
        band[mask] = fill
        out[c] = band
    return out


def score(model: BridgeTransformer, img: np.ndarray, inv) -> dict:
    photo = pack_photometry(img, inv)
    c, tr = cutout_to_tokens(photo["image"], photo["T"], patch=16)
    x = torch.from_numpy(c).unsqueeze(0)
    t = torch.from_numpy(tr).unsqueeze(0)
    mask = torch.ones(1, x.shape[1])
    with torch.no_grad():
        logits, _, dE = model(x, t, mask=mask)
        prob = torch.softmax(logits, dim=-1)[0]
    return {
        "p_cluster": float(prob[1]) if prob.numel() > 1 else 0.0,
        "p_field": float(prob[0]),
        "pred": int(logits.argmax(1).item()),
        "T_mean": float(photo["T"].mean()),
        "dE": float(dE.abs().mean()) if dE is not None else 0.0,
        "lum_mean": float(np.nan_to_num(img, nan=0.0).mean()),
    }


def main() -> int:
    if not FITS.is_file():
        raise SystemExit(f"brak FITS: {FITS}")
    if not CKPT.is_file():
        raise SystemExit(f"brak ckpt: {CKPT}")

    img0, inv0 = load_fits_bytes_or_path(FITS)
    C, H, W = img0.shape
    lum = np.nan_to_num(img0, nan=0.0).mean(axis=0)

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

    base = score(model, img0, inv0)
    print(f"BASELINE  p_cluster={base['p_cluster']:.4f}  T={base['T_mean']:.3f}  dE={base['dE']:.3f}")

    star_pix = []
    for s in STARS:
        x_pred, y_pred = sky_to_xy(s["ra"], s["dec"], h=H, w=W)
        x, y, peak = refine_peak(lum, x_pred, y_pred, search_r=5)
        star_pix.append(
            {
                **s,
                "x_pred": round(x_pred, 2),
                "y_pred": round(y_pred, 2),
                "x": x,
                "y": y,
                "peak_lum": peak,
                "snap_shift_px": round(math.hypot(x - x_pred, y - y_pred), 2),
            }
        )
        print(
            f"STAR {s['name']}: pred=({x_pred:.1f},{y_pred:.1f}) → "
            f"peak=({x},{y}) lum={peak:.3g} shift={star_pix[-1]['snap_shift_px']}px"
        )

    radii = [2.0, 3.0, 4.0, 6.0]  # px; 3px ≈ 0.8″
    conditions = []

    # 1) mask primary Gaia star only
    for r_px in radii:
        s0 = star_pix[0]
        m = circular_mask(H, W, s0["x"], s0["y"], r_px)
        img_m = apply_mask(img0, m, mode="median")
        inv_m = None
        if inv0 is not None:
            inv_m = inv0.copy()
            # zero invvar in mask → ignored / low weight if used
            inv_m[:, m] = 0.0
        sc = score(model, img_m, inv_m)
        conditions.append(
            {
                "name": f"mask_{s0['name']}_r{r_px:g}px",
                "mask": s0["name"],
                "radius_px": r_px,
                "radius_arcsec": round(r_px * PIXSCALE, 3),
                "n_pix_masked": int(m.sum()),
                **sc,
                "delta_p": sc["p_cluster"] - base["p_cluster"],
            }
        )
        print(
            f"  mask {s0['name']} r={r_px:g}px  p={sc['p_cluster']:.4f}  "
            f"Δ={sc['p_cluster']-base['p_cluster']:+.4f}"
        )

    # 2) mask both bright Gaia stars
    for r_px in (3.0, 4.0):
        m = np.zeros((H, W), dtype=bool)
        for s in star_pix:
            m |= circular_mask(H, W, s["x"], s["y"], r_px)
        img_m = apply_mask(img0, m, mode="median")
        inv_m = inv0.copy() if inv0 is not None else None
        if inv_m is not None:
            inv_m[:, m] = 0.0
        sc = score(model, img_m, inv_m)
        conditions.append(
            {
                "name": f"mask_both_gaia_r{r_px:g}px",
                "mask": "both_gaia",
                "radius_px": r_px,
                "radius_arcsec": round(r_px * PIXSCALE, 3),
                "n_pix_masked": int(m.sum()),
                **sc,
                "delta_p": sc["p_cluster"] - base["p_cluster"],
            }
        )
        print(
            f"  mask BOTH r={r_px:g}px  p={sc['p_cluster']:.4f}  "
            f"Δ={sc['p_cluster']-base['p_cluster']:+.4f}"
        )

    # 3) controls: mask at same radius from center, opposite / random angles
    s0 = star_pix[0]
    ctrl_angles = [0.0, 90.0, 180.0, 270.0]  # deg from +x
    r_star = math.hypot(s0["x"] - (W - 1) / 2.0, s0["y"] - (H - 1) / 2.0)
    for ang in ctrl_angles:
        rad = math.radians(ang)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        x = cx + r_star * math.cos(rad)
        y = cy + r_star * math.sin(rad)
        # snap to local peak (may be empty field)
        x, y, peak = refine_peak(lum, x, y, search_r=2)
        m = circular_mask(H, W, x, y, 3.0)
        img_m = apply_mask(img0, m, mode="median")
        sc = score(model, img_m, inv0)
        conditions.append(
            {
                "name": f"ctrl_angle{ang:g}_r3px",
                "mask": "control",
                "angle_deg": ang,
                "x": x,
                "y": y,
                "peak_lum": peak,
                "radius_px": 3.0,
                "n_pix_masked": int(m.sum()),
                **sc,
                "delta_p": sc["p_cluster"] - base["p_cluster"],
            }
        )
        print(
            f"  CTRL ang={ang:g} ({x},{y}) p={sc['p_cluster']:.4f}  "
            f"Δ={sc['p_cluster']-base['p_cluster']:+.4f}"
        )

    # 4) zero-mask primary (harsher)
    m = circular_mask(H, W, s0["x"], s0["y"], 4.0)
    img_z = apply_mask(img0, m, mode="zero")
    sc_z = score(model, img_z, inv0)
    conditions.append(
        {
            "name": "mask_gaia_8p8_r4px_ZERO",
            "mask": "gaia_8p8_zero",
            "radius_px": 4.0,
            "n_pix_masked": int(m.sum()),
            **sc_z,
            "delta_p": sc_z["p_cluster"] - base["p_cluster"],
        }
    )
    print(
        f"  ZERO mask r=4px  p={sc_z['p_cluster']:.4f}  "
        f"Δ={sc_z['p_cluster']-base['p_cluster']:+.4f}"
    )

    # Verdict
    star_conds = [c for c in conditions if c.get("mask") in ("gaia_8p8", "both_gaia", "gaia_8p8_zero")]
    ctrl_conds = [c for c in conditions if c.get("mask") == "control"]
    max_drop_star = min((c["delta_p"] for c in star_conds), default=0.0)
    max_drop_ctrl = min((c["delta_p"] for c in ctrl_conds), default=0.0)
    # "responsible" if star mask drops p by >=0.05 AND more than controls
    responsible = (max_drop_star <= -0.05) and (max_drop_star < max_drop_ctrl - 0.02)

    verdict = {
        "baseline_p_cluster": base["p_cluster"],
        "max_delta_star_mask": max_drop_star,
        "max_delta_control": max_drop_ctrl,
        "star_responsible_for_flag": responsible,
        "note": (
            "If star_responsible: flaga Most silnie zależy od jasnej gwiazdy Gaia. "
            "If not: p_cluster trzyma się na strukturze/barwie pola (nie na jednej gwieździe)."
        ),
    }
    print("\nVERDICT:", json.dumps(verdict, indent=2))

    payload = {
        "target": {
            "name": "JoannaM",
            "fits": str(FITS),
            "ckpt": str(CKPT.name),
            "ra": RA0,
            "dec": DEC0,
            "shape": [C, H, W],
            "pixscale": PIXSCALE,
        },
        "baseline": base,
        "stars": star_pix,
        "conditions": conditions,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# JoannaM — ablacja maski gwiazdy Gaia (2026-09-16)",
        "",
        f"**FITS:** `{FITS.name}` · **ckpt:** `{CKPT.name}`",
        "",
        "## Baseline",
        "",
        f"| p_cluster | T_mean | \\|dE\\| | pred |",
        f"|----------:|-------:|------:|------|",
        f"| **{base['p_cluster']:.4f}** | {base['T_mean']:.3f} | {base['dE']:.3f} | "
        f"{'cluster' if base['pred']==1 else 'field'} |",
        "",
        "## Gwiazdy (pozycja)",
        "",
        "| name | dist″ | mag_r | pix (x,y) | peak_lum | snap_shift |",
        "|------|------:|------:|-----------|----------:|-----------:|",
    ]
    for s in star_pix:
        lines.append(
            f"| {s['name']} | {s['dist_arcsec']:.2f} | {s['mag_r']:.2f} | "
            f"({s['x']},{s['y']}) | {s['peak_lum']:.3g} | {s['snap_shift_px']} |"
        )
    lines += [
        "",
        "## Warunki",
        "",
        "| warunek | p_cluster | Δp | T_mean | n_pix |",
        "|---------|----------:|---:|-------:|------:|",
    ]
    for c in conditions:
        lines.append(
            f"| `{c['name']}` | {c['p_cluster']:.4f} | {c['delta_p']:+.4f} | "
            f"{c['T_mean']:.3f} | {c.get('n_pix_masked', 0)} |"
        )
    lines += [
        "",
        "## Werdykt",
        "",
        f"- max Δ (maska gwiazdy): **{max_drop_star:+.4f}**",
        f"- max Δ (kontrole kątowe): **{max_drop_ctrl:+.4f}**",
        f"- gwiazda odpowiedzialna za flagę? **{'TAK' if responsible else 'NIE'}**",
        "",
        verdict["note"],
        "",
        "JSON: `ABLATION_STAR_MASK.json`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
