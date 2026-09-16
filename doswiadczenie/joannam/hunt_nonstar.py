"""Poluj na kandydatów high-p_cluster, którzy NIE padają po masce jasnego peaka.

  python doswiadczenie/joannam/hunt_nonstar.py
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
OUT = Path(__file__).with_name("HUNT_NONSTAR.json")
OUT_MD = Path(__file__).with_name("HUNT_NONSTAR.md")

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
    "joannam_fresh",  # alias reprodukcji
)
CTX = ssl._create_unverified_context()
PIXSCALE = 0.262


def parse_radec(name: str) -> tuple[float, float] | None:
    # desi_*_ra142.5_dec6.5_s128
    import re

    m = re.search(r"_ra(-?[\d.]+)_dec(-?[\d.]+)_", name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def score(model, img, inv):
    photo = pack_photometry(img, inv)
    c, tr = cutout_to_tokens(photo["image"], photo["T"], patch=16)
    x = torch.from_numpy(c).unsqueeze(0)
    t = torch.from_numpy(tr).unsqueeze(0)
    with torch.no_grad():
        logits, _, dE = model(x, t, mask=torch.ones(1, x.shape[1]))
        prob = torch.softmax(logits, dim=-1)[0]
    return float(prob[1]) if prob.numel() > 1 else 0.0, float(photo["T"].mean()), float(
        dE.abs().mean()
    ) if dE is not None else 0.0


def mask_peak(img, r_px=4.0):
    lum = np.nan_to_num(img, nan=0.0).mean(axis=0)
    yi, xi = np.unravel_index(int(np.argmax(lum)), lum.shape)
    H, W = lum.shape
    yy, xx = np.ogrid[:H, :W]
    m = (xx - xi) ** 2 + (yy - yi) ** 2 <= r_px**2
    out = img.copy()
    for c in range(out.shape[0]):
        band = out[c].copy()
        fill = float(np.median(band[~m])) if np.any(~m) else 0.0
        band[m] = fill
        out[c] = band
    return out, int(xi), int(yi), float(lum[yi, xi])


def ls_bright(ra0, dec0, rmax=36.0, maglim=18.5):
    q = f"""SELECT TOP 40 ra, dec, type, mag_g, mag_r, mag_z, parallax, ref_cat, ref_id
FROM ls_dr10.tractor
WHERE ra BETWEEN {ra0 - 0.015} AND {ra0 + 0.015}
AND dec BETWEEN {dec0 - 0.015} AND {dec0 + 0.015}
"""
    body = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q}
    ).encode()
    req = urllib.request.Request("https://datalab.noirlab.edu/tap/sync", data=body)
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=90) as r:
            text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return [], str(e)
    if text.startswith("<?xml"):
        return [], "TAP error"
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            ra, dec = float(row["ra"]), float(row["dec"])
            mr = float(row["mag_r"])
            dra = math.radians((ra - ra0) * math.cos(math.radians(0.5 * (dec0 + dec))))
            ddec = math.radians(dec - dec0)
            d = math.degrees(math.hypot(dra, ddec)) * 3600.0
        except Exception:
            continue
        if d <= rmax and mr == mr and mr < maglim:
            row["dist_arcsec"] = round(d, 2)
            row["mag_r_f"] = mr
            out.append(row)
    out.sort(key=lambda r: r["mag_r_f"])
    return out, None


def main():
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

    cands = []
    for p in sorted(DEMO.glob("desi_*_s128.fits")):
        n = p.name.lower()
        if any(x in n for x in SKIP_SUB):
            continue
        if not ("grid_" in n or "field_" in n or "new_field" in n):
            continue
        cands.append(p)

    rows = []
    print(f"scoring {len(cands)} ...")
    for fp in cands:
        img, inv = load_fits_bytes_or_path(fp)
        p0, Tm, dE = score(model, img, inv)
        img_m, x, y, peak = mask_peak(img, 4.0)
        p1, _, _ = score(model, img_m, inv)
        rows.append(
            {
                "file": fp.name,
                "p0": p0,
                "p_mask_peak": p1,
                "delta_peak": p1 - p0,
                "T_mean": Tm,
                "dE": dE,
                "peak_xy": [x, y],
                "peak_lum": peak,
            }
        )
        if p0 >= 0.5:
            print(f"  {p0:.3f} → {p1:.3f} (Δ={p1-p0:+.3f})  {fp.name}")

    rows.sort(key=lambda r: -r["p0"])

    # shortlist: high baseline AND survives peak mask (Δ > -0.15) OR still p_mask>=0.7
    short = [
        r
        for r in rows
        if r["p0"] >= 0.55
        and (r["delta_peak"] > -0.15 or r["p_mask_peak"] >= 0.70)
        and r["file"] != "desi_grid_004_ra120.0_dec5.0_s128.fits"  # JoannaM
    ]
    # also keep top survivors by p_mask_peak among p0>=0.4
    by_surv = sorted(
        [r for r in rows if r["p0"] >= 0.40 and r["file"] != "desi_grid_004_ra120.0_dec5.0_s128.fits"],
        key=lambda r: -r["p_mask_peak"],
    )[:8]

    print("\n=== shortlist (not JoannaM) ===")
    for r in short[:10]:
        print(f"  p0={r['p0']:.3f} pmask={r['p_mask_peak']:.3f} Δ={r['delta_peak']:+.3f} {r['file']}")

    print("\n=== best survivors by p_mask_peak (p0>=0.4) ===")
    for r in by_surv:
        print(f"  p0={r['p0']:.3f} pmask={r['p_mask_peak']:.3f} Δ={r['delta_peak']:+.3f} {r['file']}")

    # LS bright check for top survivors
    follow = []
    seen = set()
    for r in by_surv + short:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        radec = parse_radec(r["file"])
        if not radec:
            continue
        bright, err = ls_bright(*radec)
        item = {
            **r,
            "ra": radec[0],
            "dec": radec[1],
            "n_bright_r_lt_18_5": len(bright),
            "brightest": (
                {
                    "mag_r": bright[0]["mag_r_f"],
                    "dist_arcsec": bright[0]["dist_arcsec"],
                    "type": bright[0]["type"],
                    "ref": f"{bright[0]['ref_cat']}:{bright[0]['ref_id']}",
                    "plx": bright[0].get("parallax"),
                }
                if bright
                else None
            ),
            "ls_err": err,
        }
        # prefer: survives mask AND no very bright star (r<15) within 15"
        harsh_star = False
        if bright:
            b0 = bright[0]
            if b0["mag_r_f"] < 15.5 and b0["dist_arcsec"] < 20:
                harsh_star = True
        item["harsh_star"] = harsh_star
        item["recommend"] = (
            r["p_mask_peak"] >= 0.55
            and r["delta_peak"] > -0.20
            and not harsh_star
        ) or (r["p_mask_peak"] >= 0.75 and r["delta_peak"] > -0.12)
        follow.append(item)
        if len(follow) >= 10:
            break

    recommend = [f for f in follow if f.get("recommend")]
    print("\n=== RECOMMEND ===")
    for f in recommend:
        print(
            f"  ★ p0={f['p0']:.3f} pmask={f['p_mask_peak']:.3f} Δ={f['delta_peak']:+.3f} "
            f"bright={f['n_bright_r_lt_18_5']} harsh={f['harsh_star']}  "
            f"RA={f['ra']} Dec={f['dec']}  {f['file']}"
        )
    if not recommend and follow:
        print("  (brak idealnych — top survivor):")
        f = follow[0]
        print(
            f"  ? p0={f['p0']:.3f} pmask={f['p_mask_peak']:.3f} Δ={f['delta_peak']:+.3f} "
            f"{f['file']}"
        )

    payload = {
        "n_scored": len(rows),
        "top15_by_p0": rows[:15],
        "best_survivors": by_surv,
        "follow": follow,
        "recommend": recommend,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Hunt non-star candidates (po JoannaM)",
        "",
        f"Scored **{len(rows)}** field/grid · mask = peak luminance r=4 px",
        "",
        "## Top by baseline p0",
        "",
        "| p0 | p_mask | Δ | file |",
        "|---:|-------:|--:|------|",
    ]
    for r in rows[:12]:
        lines.append(
            f"| {r['p0']:.3f} | {r['p_mask_peak']:.3f} | {r['delta_peak']:+.3f} | `{r['file']}` |"
        )
    lines += ["", "## Recommend (nie JoannaM)", ""]
    if recommend:
        for f in recommend:
            b = f.get("brightest")
            bs = (
                f"brightest r={b['mag_r']:.1f} @{b['dist_arcsec']}\" {b['ref']}"
                if b
                else "no bright r<18.5"
            )
            lines.append(
                f"- **`{f['file']}`** RA {f['ra']} Dec {f['dec']} · "
                f"p0={f['p0']:.3f} → pmask={f['p_mask_peak']:.3f} ({f['delta_peak']:+.3f}) · {bs}"
            )
    else:
        lines.append("_Brak mocnego recommend — patrz survivors w JSON._")
        if follow:
            f = follow[0]
            lines.append(
                f"Najbliższy: `{f['file']}` p0={f['p0']:.3f} pmask={f['p_mask_peak']:.3f}"
            )
    lines += ["", "JSON: `HUNT_NONSTAR.json`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
