"""Świeży cutout JoannaM (niezależny download) → score + ablacja vs stary FITS.

  cd "C:\\Users\\drwis\\model Ai"
  python doswiadczenie/joannam/reproduce_fresh_cutout.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ZDIR = ROOT / "ZdjęcieW"
if str(ZDIR) not in sys.path:
    sys.path.insert(0, str(ZDIR))

from bridge import BridgeTransformer  # noqa: E402
from config import CANON  # noqa: E402
from desi_cutouts import load_fits_bytes_or_path, pack_photometry  # noqa: E402
from image_batch import cutout_to_tokens  # noqa: E402
from pobieranie import fetch_fits, fits_to_rgb_preview, save_cutout  # noqa: E402

DEMO = ROOT / "ZdjęcieW" / "demo"
OLD_FITS = DEMO / "desi_grid_004_ra120.0_dec5.0_s128.fits"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
OUT_JSON = Path(__file__).with_name("REPRODUCE_FRESH_CUTOUT.json")
OUT_MD = Path(__file__).with_name("REPRODUCE_FRESH_CUTOUT.md")

RA0, DEC0 = 120.0, 5.0
PIXSCALE = 0.262
STAR = {
    "name": "gaia_8p8",
    "ra": 120.0022588903326,
    "dec": 4.999042584188773,
    "mag_r": 17.676,
}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def sky_to_xy(ra: float, dec: float, h: int, w: int) -> tuple[float, float]:
    cosd = math.cos(math.radians(DEC0))
    dra_as = (ra - RA0) * cosd * 3600.0
    ddec_as = (dec - DEC0) * 3600.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    x = cx - dra_as / PIXSCALE
    y = cy + ddec_as / PIXSCALE
    return x, y


def refine_peak(lum: np.ndarray, x0: float, y0: float, search_r: int = 5):
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


def apply_mask_median(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    for c in range(out.shape[0]):
        band = out[c].copy()
        fill = float(np.median(band[~mask])) if np.any(~mask) else 0.0
        band[mask] = fill
        out[c] = band
    return out


def load_model():
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
    return model


def score(model, img, inv) -> dict:
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
    }


def eval_fits(model, fits_path: Path) -> dict:
    img, inv = load_fits_bytes_or_path(fits_path)
    C, H, W = img.shape
    lum = np.nan_to_num(img, nan=0.0).mean(axis=0)
    base = score(model, img, inv)

    x_pred, y_pred = sky_to_xy(STAR["ra"], STAR["dec"], H, W)
    x, y, peak = refine_peak(lum, x_pred, y_pred)
    abl = {}
    for r_px in (3.0, 4.0):
        m = circular_mask(H, W, x, y, r_px)
        img_m = apply_mask_median(img, m)
        sc = score(model, img_m, inv)
        abl[f"r{r_px:g}px"] = {
            "p_cluster": sc["p_cluster"],
            "delta_p": sc["p_cluster"] - base["p_cluster"],
            "T_mean": sc["T_mean"],
            "n_pix": int(m.sum()),
        }

    # control opposite angle
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    r_star = math.hypot(x - cx, y - cy)
    xc, yc = cx - (x - cx), cy - (y - cy)  # 180°
    xc, yc, _ = refine_peak(lum, xc, yc, search_r=2)
    m_c = circular_mask(H, W, xc, yc, 3.0)
    sc_c = score(model, apply_mask_median(img, m_c), inv)

    return {
        "file": fits_path.name,
        "path": str(fits_path),
        "sha256_16": sha16(fits_path),
        "size_bytes": fits_path.stat().st_size,
        "shape": [C, H, W],
        "baseline": base,
        "star_peak": {
            "x": x,
            "y": y,
            "x_pred": round(x_pred, 2),
            "y_pred": round(y_pred, 2),
            "peak_lum": peak,
            "snap_shift_px": round(math.hypot(x - x_pred, y - y_pred), 2),
        },
        "ablation_star": abl,
        "control_180_r3px": {
            "p_cluster": sc_c["p_cluster"],
            "delta_p": sc_c["p_cluster"] - base["p_cluster"],
            "x": xc,
            "y": yc,
        },
    }


def main() -> int:
    if not OLD_FITS.is_file():
        raise SystemExit(f"brak starego FITS: {OLD_FITS}")
    if not CKPT.is_file():
        raise SystemExit(f"brak ckpt: {CKPT}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"joannam_fresh_{stamp}"
    print(f"Downloading fresh cutout ra={RA0} dec={DEC0} name={name} ...")
    fresh_fits, fresh_png = save_cutout(
        ra=RA0,
        dec=DEC0,
        name=name,
        size=128,
        out_dir=DEMO,
        skip_existing=False,
    )
    print(f"fresh FITS: {fresh_fits.name} sha={sha16(fresh_fits)}")
    print(f"old   FITS: {OLD_FITS.name} sha={sha16(OLD_FITS)}")
    identical_bytes = sha16(fresh_fits) == sha16(OLD_FITS)

    # Also keep a stable alias copy for docs
    alias = DEMO / "desi_joannam_fresh_ra120.0_dec5.0_s128.fits"
    alias.write_bytes(fresh_fits.read_bytes())
    alias_png = DEMO / "desi_joannam_fresh_ra120.0_dec5.0_s128.png"
    if fresh_png and fresh_png.is_file():
        alias_png.write_bytes(fresh_png.read_bytes())
    else:
        fits_to_rgb_preview(fresh_fits.read_bytes()).save(alias_png)

    model = load_model()
    old_res = eval_fits(model, OLD_FITS)
    fresh_res = eval_fits(model, fresh_fits)
    alias_res = eval_fits(model, alias)

    dp = fresh_res["baseline"]["p_cluster"] - old_res["baseline"]["p_cluster"]
    repro_ok = abs(dp) < 0.05  # soft gate: score within 5 pp
    star_still = fresh_res["ablation_star"]["r4px"]["delta_p"] <= -0.05

    verdict = {
        "bytes_identical_to_old": identical_bytes,
        "delta_p_fresh_minus_old": dp,
        "score_reproduced_within_5pp": repro_ok,
        "star_ablation_replicates_on_fresh": star_still,
        "note": (
            "Fresh Legacy download scored with same most_desi_color.pt. "
            "Ablation of Gaia@8.8\" should still drop p_cluster if star-driven."
        ),
    }
    print("\nOLD   p=", old_res["baseline"]["p_cluster"])
    print("FRESH p=", fresh_res["baseline"]["p_cluster"], " Δ=", f"{dp:+.4f}")
    print("FRESH abl r4 Δ=", fresh_res["ablation_star"]["r4px"]["delta_p"])
    print("VERDICT", json.dumps(verdict, indent=2))

    payload = {
        "when_utc": stamp,
        "ra": RA0,
        "dec": DEC0,
        "ckpt": CKPT.name,
        "old": old_res,
        "fresh": fresh_res,
        "alias": alias_res,
        "alias_file": alias.name,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# JoannaM — reprodukcja na świeżym cutoucie (2026-09-16)",
        "",
        f"**UTC:** `{stamp}` · **ckpt:** `{CKPT.name}` · RA {RA0} Dec {DEC0}",
        "",
        "## Pliki",
        "",
        f"| | plik | sha256[:16] | bajty |",
        f"|--|------|-------------|------:|",
        f"| stary | `{OLD_FITS.name}` | `{old_res['sha256_16']}` | {old_res['size_bytes']} |",
        f"| świeży | `{fresh_fits.name}` | `{fresh_res['sha256_16']}` | {fresh_res['size_bytes']} |",
        f"| alias | `{alias.name}` | `{alias_res['sha256_16']}` | {alias_res['size_bytes']} |",
        "",
        f"Bajty identyczne ze starym? **{'TAK' if identical_bytes else 'NIE'}**",
        "",
        "## Score",
        "",
        "| cutout | p_cluster | T_mean | \\|dE\\| | abl r=4px Δp | ctrl Δp |",
        "|--------|----------:|-------:|------:|-------------:|--------:|",
        f"| stary | {old_res['baseline']['p_cluster']:.4f} | {old_res['baseline']['T_mean']:.3f} | "
        f"{old_res['baseline']['dE']:.3f} | {old_res['ablation_star']['r4px']['delta_p']:+.4f} | "
        f"{old_res['control_180_r3px']['delta_p']:+.4f} |",
        f"| **świeży** | **{fresh_res['baseline']['p_cluster']:.4f}** | {fresh_res['baseline']['T_mean']:.3f} | "
        f"{fresh_res['baseline']['dE']:.3f} | **{fresh_res['ablation_star']['r4px']['delta_p']:+.4f}** | "
        f"{fresh_res['control_180_r3px']['delta_p']:+.4f} |",
        "",
        f"Δ(fresh−old) p_cluster = **{dp:+.4f}**",
        "",
        "## Werdykt",
        "",
        f"- Score reprodukowalny (±5 pp)? **{'TAK' if repro_ok else 'NIE'}**",
        f"- Ablacja gwiazdy nadal działa na świeżym? **{'TAK' if star_still else 'NIE'}**",
        "",
        verdict["note"],
        "",
        "JSON: `REPRODUCE_FRESH_CUTOUT.json`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
