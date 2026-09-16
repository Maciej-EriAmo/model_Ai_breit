"""Szybki screening kolejnego kandydata (po JoannaM).

  python doswiadczenie/joannam/screen_next_candidate.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bridge import BridgeTransformer
from config import CANON
from desi_cutouts import load_fits_bytes_or_path, pack_photometry
from image_batch import cutout_to_tokens

DEMO = ROOT / "ZdjęcieW" / "demo"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
FITS = DEMO / "desi_new_field_01_ra142.5_dec6.5_s128.fits"
OUT = Path(__file__).with_name("NEXT_CANDIDATE_SCREEN.json")
OUT_MD = Path(__file__).with_name("NEXT_CANDIDATE_SCREEN.md")

RA0, DEC0 = 142.5, 6.5
PIXSCALE = 0.262
CTX = ssl._create_unverified_context()


def sep_arcsec(ra2, dec2):
    dra = math.radians((ra2 - RA0) * math.cos(math.radians(0.5 * (DEC0 + dec2))))
    ddec = math.radians(dec2 - DEC0)
    return math.degrees(math.hypot(dra, ddec)) * 3600.0


def ls_tractor(radius_arcsec: float = 36.0):
    q = f"""SELECT TOP 80 ra, dec, type, mag_g, mag_r, mag_z, parallax, ref_cat, ref_id
FROM ls_dr10.tractor
WHERE ra BETWEEN {RA0 - 0.02} AND {RA0 + 0.02}
AND dec BETWEEN {DEC0 - 0.02} AND {DEC0 + 0.02}
"""
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}
    ).encode()
    req = urllib.request.Request("https://datalab.noirlab.edu/tap/sync", data=body)
    with urllib.request.urlopen(req, context=CTX, timeout=120) as r:
        text = r.read().decode("utf-8", errors="replace")
    if text.startswith("<?xml") or "ERROR" in text[:200]:
        return [], text[:500]
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            d = sep_arcsec(float(row["ra"]), float(row["dec"]))
        except Exception:
            continue
        if d <= radius_arcsec:
            row["dist_arcsec"] = round(d, 3)
            rows.append(row)
    rows.sort(key=lambda r: r["dist_arcsec"])
    return rows, None


def sky_to_xy(ra, dec, h, w):
    cosd = math.cos(math.radians(DEC0))
    dra_as = (ra - RA0) * cosd * 3600.0
    ddec_as = (dec - DEC0) * 3600.0
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return cx - dra_as / PIXSCALE, cy + ddec_as / PIXSCALE


def refine_peak(lum, x0, y0, search_r=5):
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


def circular_mask(h, w, x, y, r):
    yy, xx = np.ogrid[:h, :w]
    return ((xx - x) ** 2 + (yy - y) ** 2) <= r**2


def apply_mask(img, mask):
    out = img.copy()
    for c in range(out.shape[0]):
        band = out[c].copy()
        fill = float(np.median(band[~mask])) if np.any(~mask) else 0.0
        band[mask] = fill
        out[c] = band
    return out


def score(model, img, inv):
    photo = pack_photometry(img, inv)
    c, tr = cutout_to_tokens(photo["image"], photo["T"], patch=16)
    x = torch.from_numpy(c).unsqueeze(0)
    t = torch.from_numpy(tr).unsqueeze(0)
    with torch.no_grad():
        logits, _, dE = model(x, t, mask=torch.ones(1, x.shape[1]))
        prob = torch.softmax(logits, dim=-1)[0]
    return {
        "p_cluster": float(prob[1]) if prob.numel() > 1 else 0.0,
        "T_mean": float(photo["T"].mean()),
        "dE": float(dE.abs().mean()) if dE is not None else 0.0,
        "pred": int(logits.argmax(1).item()),
    }


def main():
    if not FITS.is_file():
        raise SystemExit(f"brak {FITS}")

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

    img, inv = load_fits_bytes_or_path(FITS)
    C, H, W = img.shape
    lum = np.nan_to_num(img, nan=0.0).mean(axis=0)
    base = score(model, img, inv)
    print(f"BASE p_cluster={base['p_cluster']:.4f} T={base['T_mean']:.3f}")

    print("LS tractor...")
    sources, err = ls_tractor(36.0)
    bright = []
    for r in sources:
        try:
            mr = float(r["mag_r"])
        except Exception:
            continue
        if mr == mr and mr < 20:  # not NaN
            bright.append(r)
    print(f"  LS n={len(sources)} bright_r<20={len(bright)}")
    for r in bright[:8]:
        print(
            f"  {r['dist_arcsec']:5.1f}\" type={r['type']} r={r['mag_r']} "
            f"plx={r['parallax']} ref={r['ref_cat']}:{r['ref_id']}"
        )

    # Ablate each bright source + global brightest pixel
    conditions = []
    targets = []
    for r in bright:
        x_pred, y_pred = sky_to_xy(float(r["ra"]), float(r["dec"]), H, W)
        x, y, peak = refine_peak(lum, x_pred, y_pred)
        targets.append(
            {
                "kind": "ls_bright",
                "dist_arcsec": r["dist_arcsec"],
                "mag_r": float(r["mag_r"]),
                "type": r["type"],
                "ref": f"{r['ref_cat']}:{r['ref_id']}",
                "x": x,
                "y": y,
                "peak": peak,
            }
        )

    # also top luminance peak in cutout (whatever it is)
    yi, xi = np.unravel_index(int(np.argmax(lum)), lum.shape)
    targets.append(
        {
            "kind": "lum_peak",
            "dist_arcsec": None,
            "mag_r": None,
            "type": "peak",
            "ref": "argmax_lum",
            "x": int(xi),
            "y": int(yi),
            "peak": float(lum[yi, xi]),
        }
    )

    for t in targets:
        for r_px in (3.0, 4.0):
            m = circular_mask(H, W, t["x"], t["y"], r_px)
            sc = score(model, apply_mask(img, m), inv)
            cond = {
                "name": f"mask_{t['kind']}_{t['ref']}_r{r_px:g}",
                "target": t,
                "radius_px": r_px,
                **sc,
                "delta_p": sc["p_cluster"] - base["p_cluster"],
            }
            conditions.append(cond)
            print(
                f"  {cond['name'][:60]:60s} p={sc['p_cluster']:.4f} Δ={cond['delta_p']:+.4f}"
            )

    # control: mask empty-ish corner
    m = circular_mask(H, W, 10, 10, 3.0)
    sc = score(model, apply_mask(img, m), inv)
    conditions.append(
        {
            "name": "ctrl_corner_r3",
            "delta_p": sc["p_cluster"] - base["p_cluster"],
            **sc,
        }
    )
    print(f"  ctrl_corner p={sc['p_cluster']:.4f} Δ={conditions[-1]['delta_p']:+.4f}")

    star_like = [c for c in conditions if c.get("target", {}).get("kind") == "ls_bright"]
    max_drop = min((c["delta_p"] for c in star_like), default=0.0) if star_like else min(
        (c["delta_p"] for c in conditions if "lum_peak" in c["name"]), default=0.0
    )
    # Also consider lum peak
    lum_drop = min(
        (c["delta_p"] for c in conditions if "lum_peak" in c["name"]), default=0.0
    )
    worst = min(max_drop, lum_drop)
    star_driven = worst <= -0.15

    verdict = {
        "candidate_file": FITS.name,
        "ra": RA0,
        "dec": DEC0,
        "baseline_p_cluster": base["p_cluster"],
        "n_ls_sources_36as": len(sources),
        "n_bright_r_lt_20": len(bright),
        "max_delta_bright_or_peak": worst,
        "likely_star_driven_like_joannam": star_driven,
        "interesting_for_followup": (base["p_cluster"] >= 0.85) and (not star_driven),
    }
    print("\nVERDICT", json.dumps(verdict, indent=2))

    payload = {
        "baseline": base,
        "ls_sources": sources[:40],
        "bright": bright,
        "targets": targets,
        "conditions": conditions,
        "verdict": verdict,
        "ls_error": err,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Next candidate screen — `{FITS.name}`",
        "",
        f"**RA {RA0} · Dec {DEC0}** · p_cluster baseline **{base['p_cluster']:.4f}**",
        "",
        f"LS DR10 ≤36″: n={len(sources)}, bright r<20: **{len(bright)}**",
        "",
        "| źródło | dist″ | type | r | Δp (r=4px) |",
        "|--------|------:|------|--:|-----------:|",
    ]
    for t in targets:
        d4 = next(
            (
                c["delta_p"]
                for c in conditions
                if c.get("target") is t and c.get("radius_px") == 4.0
            ),
            None,
        )
        # match by name fragment
        d4 = None
        for c in conditions:
            if c.get("radius_px") == 4.0 and c.get("target", {}).get("ref") == t["ref"]:
                d4 = c["delta_p"]
                break
        lines.append(
            f"| {t['ref']} | {t['dist_arcsec'] if t['dist_arcsec'] is not None else '—'} | "
            f"{t['type']} | {t['mag_r'] if t['mag_r'] is not None else '—'} | "
            f"{d4:+.4f}" if d4 is not None else f"| {t['ref']} | … |"
        )
    # fix broken f-string lines - rewrite simpler
    lines = [
        f"# Next candidate screen — `{FITS.name}`",
        "",
        f"**RA {RA0} · Dec {DEC0}** · p_cluster baseline **{base['p_cluster']:.4f}**",
        "",
        f"LS DR10 ≤36″: n={len(sources)}, bright r<20: **{len(bright)}**",
        "",
        f"- max Δ (bright/peak): **{worst:+.4f}**",
        f"- star-driven jak JoannaM? **{'TAK' if star_driven else 'NIE'}**",
        f"- wart follow-upu? **{'TAK' if verdict['interesting_for_followup'] else 'NIE / ostrożnie'}**",
        "",
        "JSON: `NEXT_CANDIDATE_SCREEN.json`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
