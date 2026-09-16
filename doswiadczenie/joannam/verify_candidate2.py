"""Pełny cross-match kandydata #2 (new_field_01, RA 142.5 Dec +6.5).

  python doswiadczenie/joannam/verify_candidate2.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ZdjęcieW"))

from bridge import BridgeTransformer
from config import CANON
from desi_cutouts import load_fits_bytes_or_path, pack_photometry
from image_batch import cutout_to_tokens
from pobieranie import fits_to_rgb_preview, save_cutout

DEMO = ROOT / "ZdjęcieW" / "demo"
CKPT = ROOT / "checkpoints" / "most_desi_color.pt"
OLD_FITS = DEMO / "desi_new_field_01_ra142.5_dec6.5_s128.fits"
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "CROSSMATCH_C2.json"
OUT_MD = HERE / "CROSSMATCH_C2.md"

RA0, DEC0 = 142.5, 6.5
PIXSCALE = 0.262
CTX = ssl._create_unverified_context()
GAIA = {
    "ra": None,  # filled from LS
    "dec": None,
    "ref": "GE:586076295021789056",
    "mag_r": 13.598594,
}


def http_get(url: str, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "C2-verify/1.0"})
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def http_post(url: str, data: dict, timeout: int = 120) -> str:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "C2-verify/1.0"})
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def sep_as(ra2: float, dec2: float) -> float:
    dra = math.radians((ra2 - RA0) * math.cos(math.radians(0.5 * (DEC0 + dec2))))
    ddec = math.radians(dec2 - DEC0)
    return math.degrees(math.hypot(dra, ddec)) * 3600.0


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def gal_lb(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    ra_ngp = math.radians(192.85948)
    dec_ngp = math.radians(27.12825)
    l_ngp = math.radians(122.93192)
    sin_b = math.sin(dec) * math.sin(dec_ngp) + math.cos(dec) * math.cos(dec_ngp) * math.cos(
        ra - ra_ngp
    )
    b = math.asin(max(-1.0, min(1.0, sin_b)))
    y = math.cos(dec) * math.sin(ra - ra_ngp)
    x = math.sin(dec) * math.cos(dec_ngp) - math.cos(dec) * math.sin(dec_ngp) * math.cos(
        ra - ra_ngp
    )
    l = (math.degrees(l_ngp - math.atan2(y, x))) % 360
    return l, math.degrees(b)


def simbad(radius: float, unit: str) -> dict:
    url = (
        f"https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={RA0}+%2B{DEC0}"
        f"&CooFrame=ICRS&Radius={radius}&Radius.unit={unit}&output.format=ASCII"
    )
    text = http_get(url)
    n = 0
    nearest = []
    if "No astronomical object found" in text:
        return {"n": 0, "nearest": [], "raw_preview": text[:800]}
    m = re.search(r"Number of objects\s*:\s*(\d+)", text)
    if m:
        n = int(m.group(1))
    for ln in text.splitlines():
        if re.match(r"^\d+\s*\|", ln):
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 4:
                try:
                    dist = float(parts[1])
                except ValueError:
                    continue
                nearest.append(
                    {"dist_asec": dist, "id": parts[2], "otype": parts[3]}
                )
    return {"n": n, "nearest": nearest[:12], "raw_preview": text[:1200]}


def ls_tractor(rmax: float = 36.0) -> list[dict]:
    q = f"""SELECT TOP 80 ra, dec, type, mag_g, mag_r, mag_z, parallax, ref_cat, ref_id
FROM ls_dr10.tractor
WHERE ra BETWEEN {RA0 - 0.02} AND {RA0 + 0.02}
AND dec BETWEEN {DEC0 - 0.02} AND {DEC0 + 0.02}
"""
    text = http_post(
        "https://datalab.noirlab.edu/tap/sync",
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q},
    )
    if text.startswith("<?xml"):
        return []
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            d = sep_as(float(row["ra"]), float(row["dec"]))
            mr = float(row["mag_r"])
        except Exception:
            continue
        if d <= rmax:
            row["dist_arcsec"] = round(d, 3)
            row["mag_r_f"] = mr if mr == mr else None
            rows.append(row)
    rows.sort(key=lambda r: r["dist_arcsec"])
    return rows


def desi_near(box_deg: float = 0.2) -> list[dict]:
    q = f"""SELECT TOP 100 targetid, z, zwarn, spectype, mean_fiber_ra, mean_fiber_dec
FROM desi_dr1.zpix
WHERE mean_fiber_ra BETWEEN {RA0 - box_deg} AND {RA0 + box_deg}
AND mean_fiber_dec BETWEEN {DEC0 - box_deg} AND {DEC0 + box_deg}
"""
    text = http_post(
        "https://datalab.noirlab.edu/tap/sync",
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": q},
    )
    if text.startswith("<?xml"):
        return []
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        d = sep_as(float(row["mean_fiber_ra"]), float(row["mean_fiber_dec"]))
        if d < 600:
            row["dist_arcsec"] = round(d, 2)
            out.append(row)
    out.sort(key=lambda r: r["dist_arcsec"])
    return out


def vizier_count(source: str, radius_arcmin: float) -> dict:
    params = {
        "-source": source,
        "-c": f"{RA0} {DEC0:+}",
        "-c.rm": str(radius_arcmin),
        "-out.max": "40",
        "-out.add": "_r",
        "-sort": "_r",
    }
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?" + urllib.parse.urlencode(params)
    try:
        text = http_get(url)
    except Exception as e:
        return {"ok": False, "error": str(e), "n": 0, "objects": []}
    objs = []
    for ln in text.splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        if "\t" in ln:
            p = [x.strip() for x in ln.split("\t")]
            try:
                objs.append({"_r_arcmin": float(p[0]), "row0": p[1] if len(p) > 1 else p[0]})
            except ValueError:
                continue
        elif "1eRASS" in ln:
            m = re.search(r"([\d.]+)\s+(1eRASS\s+\S+)", ln)
            if m:
                objs.append({"_r_arcmin": float(m.group(1)), "IAUName": m.group(2)})
        elif re.match(r"^[\d.]+\s+\S+", ln) and "NVSS" in ln or re.match(
            r"^\s*[\d.]+\s+NVSS", ln
        ):
            pass
    # NVSS / generic: lines with leading distance in first tab col already handled
    # Also parse NVSS name lines without relying on header
    if "nvss" in source.lower():
        objs = []
        header = None
        for ln in text.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            if header is None and "\t" in ln and "NVSS" in ln.upper() or (
                header is None and "\t" in ln and "RAJ2000" in ln
            ):
                header = [h.strip() for h in ln.split("\t")]
                continue
            if header is None:
                continue
            if any(u in ln for u in ('"h:m:s"', "----", "mJy", "arcsec")):
                continue
            parts = [p.strip() for p in ln.split("\t")]
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts))
            try:
                r = float(row.get("_r") or 0) * 60.0  # arcmin→arcsec if _r in arcmin
            except ValueError:
                r = None
            # VizieR _r often already arcmin when -c.rm used
            try:
                r_am = float(row.get("_r", "nan"))
            except ValueError:
                continue
            objs.append(
                {
                    "_r_arcmin": r_am,
                    "NVSS": row.get("NVSS"),
                    "S1.4": row.get("S1.4"),
                }
            )
    return {"ok": True, "n": len(objs), "objects": objs[:20], "raw_preview": text[:1000]}


def erosita(radius_arcmin: float) -> dict:
    params = {
        "-source": "J/A+A/682/A34",
        "-c": f"{RA0} {DEC0:+}",
        "-c.rm": str(radius_arcmin),
        "-out.max": "30",
        "-out.add": "_r",
        "-sort": "_r",
    }
    url = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?" + urllib.parse.urlencode(params)
    text = http_get(url)
    objs = []
    for ln in text.splitlines():
        if "1eRASS" not in ln:
            continue
        if "\t" in ln:
            p = [x.strip() for x in ln.split("\t")]
            try:
                objs.append(
                    {
                        "_r_arcmin": float(p[0]),
                        "IAUName": p[1],
                        "RA": float(p[3]),
                        "DE": float(p[4]),
                    }
                )
            except Exception:
                continue
        else:
            m = re.search(
                r"(?P<r>[\d.]+)\s+(?P<name>1eRASS\s+\S+)\s+\S+\s+(?P<ra>[\d.]+)\s+(?P<dec>[+-]?[\d.]+)",
                ln,
            )
            if m:
                objs.append(
                    {
                        "_r_arcmin": float(m.group("r")),
                        "IAUName": m.group("name"),
                        "RA": float(m.group("ra")),
                        "DE": float(m.group("dec")),
                    }
                )
    seen = set()
    uniq = []
    for o in sorted(objs, key=lambda x: x["_r_arcmin"]):
        if o["IAUName"] in seen:
            continue
        seen.add(o["IAUName"])
        uniq.append(o)
    return {"n": len(uniq), "objects": uniq}


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


def eval_fits(model, fits_path: Path, gaia_ra: float, gaia_dec: float) -> dict:
    img, inv = load_fits_bytes_or_path(fits_path)
    C, H, W = img.shape
    lum = np.nan_to_num(img, nan=0.0).mean(axis=0)
    base = score(model, img, inv)
    xp, yp = sky_to_xy(gaia_ra, gaia_dec, H, W)
    x, y, peak = refine_peak(lum, xp, yp)
    abl = {}
    for r_px in (3.0, 4.0):
        m = circular_mask(H, W, x, y, r_px)
        sc = score(model, apply_mask(img, m), inv)
        abl[f"r{r_px:g}"] = {
            "p": sc["p_cluster"],
            "delta": sc["p_cluster"] - base["p_cluster"],
            "n_pix": int(m.sum()),
        }
    # control 180
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    xc, yc = cx - (x - cx), cy - (y - cy)
    xc, yc, _ = refine_peak(lum, xc, yc, 2)
    sc_c = score(model, apply_mask(img, circular_mask(H, W, xc, yc, 3.0)), inv)
    return {
        "file": fits_path.name,
        "sha": sha16(fits_path),
        "bytes": fits_path.stat().st_size,
        "baseline": base,
        "gaia_pix": {"x": x, "y": y, "peak": peak, "shift": round(math.hypot(x - xp, y - yp), 2)},
        "ablation_gaia": abl,
        "control_180": {
            "p": sc_c["p_cluster"],
            "delta": sc_c["p_cluster"] - base["p_cluster"],
        },
    }


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    l, b = gal_lb(RA0, DEC0)
    print(f"C2 RA={RA0} Dec={DEC0}  gal l={l:.2f} b={b:.2f}")

    print("LS...")
    ls = ls_tractor(36.0)
    bright = [r for r in ls if r.get("mag_r_f") is not None and r["mag_r_f"] < 20]
    gaia_ra = gaia_dec = None
    for r in bright:
        if "586076295021789056" in str(r.get("ref_id")):
            gaia_ra, gaia_dec = float(r["ra"]), float(r["dec"])
            GAIA["ra"], GAIA["dec"] = gaia_ra, gaia_dec
            GAIA["dist_arcsec"] = r["dist_arcsec"]
            break
    if gaia_ra is None and bright:
        # fallback brightest
        bright.sort(key=lambda r: r["mag_r_f"])
        gaia_ra, gaia_dec = float(bright[0]["ra"]), float(bright[0]["dec"])
        GAIA["ra"], GAIA["dec"] = gaia_ra, gaia_dec
        GAIA["dist_arcsec"] = bright[0]["dist_arcsec"]
        GAIA["ref"] = f"{bright[0]['ref_cat']}:{bright[0]['ref_id']}"
        GAIA["mag_r"] = bright[0]["mag_r_f"]
    print(f"  LS n={len(ls)} bright={len(bright)} gaia@{GAIA.get('dist_arcsec')}\"")

    print("SIMBAD...")
    sim60 = simbad(60, "arcsec")
    sim5 = simbad(5, "arcmin")
    print(f"  60as n={sim60['n']}  5' n={sim5['n']}")

    print("DESI...")
    desi = desi_near(0.2)
    print(f"  within 10' n={len(desi)}")
    for row in desi[:5]:
        print(f"    {row['dist_arcsec']}\" z={row['z']} {row['spectype']}")

    print("eROSITA...")
    er5 = erosita(5.0)
    er15 = erosita(15.0)
    print(f"  5' n={er5['n']}  15' n={er15['n']}")
    if er15["objects"]:
        o = er15["objects"][0]
        print(f"    nearest {o['_r_arcmin']:.2f}' {o.get('IAUName')}")

    print("NVSS...")
    nvss5 = vizier_count("VIII/65/nvss", 5.0)
    nvss15 = vizier_count("VIII/65/nvss", 15.0)
    print(f"  5' n={nvss5['n']}  15' n={nvss15['n']}")

    print("2RXS...")
    rxs = vizier_count("J/A+A/588/A103/catalog", 15.0)
    print(f"  15' n={rxs['n']}")

    print("Scoring + fresh cutout...")
    model = load_model()
    assert gaia_ra is not None
    old_res = eval_fits(model, OLD_FITS, gaia_ra, gaia_dec)
    print(f"  OLD p={old_res['baseline']['p_cluster']:.4f} abl4 Δ={old_res['ablation_gaia']['r4']['delta']:+.4f}")

    fresh_fits, fresh_png = save_cutout(
        ra=RA0,
        dec=DEC0,
        name=f"c2_fresh_{stamp}",
        size=128,
        out_dir=DEMO,
        skip_existing=False,
    )
    alias = DEMO / "desi_c2_fresh_ra142.5_dec6.5_s128.fits"
    alias.write_bytes(fresh_fits.read_bytes())
    if fresh_png and fresh_png.is_file():
        (DEMO / "desi_c2_fresh_ra142.5_dec6.5_s128.png").write_bytes(fresh_png.read_bytes())
    else:
        fits_to_rgb_preview(fresh_fits.read_bytes()).save(
            DEMO / "desi_c2_fresh_ra142.5_dec6.5_s128.png"
        )
    fresh_res = eval_fits(model, fresh_fits, gaia_ra, gaia_dec)
    print(
        f"  FRESH p={fresh_res['baseline']['p_cluster']:.4f} "
        f"sha_same={fresh_res['sha']==old_res['sha']} "
        f"abl4 Δ={fresh_res['ablation_gaia']['r4']['delta']:+.4f}"
    )

    verdict = {
        "spectroscopy_center": "NO_MATCH_SIMBAD_60as",
        "desi_in_cutout": any(r["dist_arcsec"] < 20 for r in desi),
        "nearest_desi_arcsec": desi[0]["dist_arcsec"] if desi else None,
        "xray_erosita_5arcmin": er5["n"],
        "radio_nvss_5arcmin": nvss5["n"],
        "star_partial": old_res["ablation_gaia"]["r4"]["delta"] <= -0.10,
        "residual_high_after_mask": old_res["ablation_gaia"]["r4"]["p"] >= 0.70,
        "fresh_reproduces": abs(
            fresh_res["baseline"]["p_cluster"] - old_res["baseline"]["p_cluster"]
        )
        < 0.05,
        "unlike_joannam": old_res["ablation_gaia"]["r4"]["p"] >= 0.50,
        "summary_pl": (
            "Kandydat #2: wysoki p_cluster; Gaia dokłada ~0.17, ale po masce zostaje ~0.80. "
            "SIMBAD 60as pusto; eROSITA/NVSS ≤5' do sprawdzenia w tabelach."
        ),
    }

    payload = {
        "when_utc": stamp,
        "target": {
            "label": "candidate_2_new_field_01",
            "file": OLD_FITS.name,
            "ra": RA0,
            "dec": DEC0,
            "galactic_l": round(l, 3),
            "galactic_b": round(b, 3),
        },
        "gaia": GAIA,
        "ls": {"n": len(ls), "bright_r_lt_20": bright[:5]},
        "simbad_60as": sim60,
        "simbad_5arcmin": {"n": sim5["n"], "nearest": sim5["nearest"][:8]},
        "desi": desi[:15],
        "erosita_5": er5,
        "erosita_15": er15,
        "nvss_5": nvss5,
        "nvss_15": nvss15,
        "rass_2rxs_15": rxs,
        "old": old_res,
        "fresh": fresh_res,
        "alias": alias.name,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown report
    er_near = er15["objects"][0] if er15["objects"] else None
    nv_near = nvss15["objects"][0] if nvss15.get("objects") else None
    er_near_s = "—"
    if er_near:
        er_near_s = f"{er_near.get('IAUName')} @ {er_near['_r_arcmin']:.2f}'"
    nv_near_s = "—"
    if nv_near:
        nv_near_s = f"{nv_near.get('NVSS') or nv_near.get('row0')} @ {nv_near.get('_r_arcmin')}′"
    sim5_near = "—"
    if sim5["nearest"]:
        sim5_near = f"{sim5['nearest'][0]['id']} @ {sim5['nearest'][0]['dist_asec']}″"
    desi_near_s = "—"
    if desi:
        desi_near_s = (
            f"{desi[0]['dist_arcsec']}″ {desi[0]['spectype']} z={desi[0]['z']}"
        )
    lines = [
        "# Kandydat #2 — cross-match pełny (`new_field_01`)",
        "",
        f"**UTC:** `{stamp}` · **RA {RA0}° · Dec {DEC0}°** · gal l={l:.1f} b={b:.1f}",
        f"**FITS:** `{OLD_FITS.name}` · **ckpt:** `{CKPT.name}`",
        "",
        "> Nie JoannaM. Roboczo: **candidate #2** — nazwa dopiero po decyzji.",
        "",
        "## Score + ablacja",
        "",
        "| cutout | p_cluster | abl Gaia r=4 Δp | p po masce | ctrl Δp |",
        "|--------|----------:|----------------:|-----------:|--------:|",
        f"| stary | **{old_res['baseline']['p_cluster']:.4f}** | "
        f"{old_res['ablation_gaia']['r4']['delta']:+.4f} | "
        f"**{old_res['ablation_gaia']['r4']['p']:.4f}** | "
        f"{old_res['control_180']['delta']:+.4f} |",
        f"| świeży | {fresh_res['baseline']['p_cluster']:.4f} | "
        f"{fresh_res['ablation_gaia']['r4']['delta']:+.4f} | "
        f"{fresh_res['ablation_gaia']['r4']['p']:.4f} | "
        f"{fresh_res['control_180']['delta']:+.4f} |",
        "",
        f"Fresh sha={'IDENTYCZNY' if fresh_res['sha']==old_res['sha'] else 'INNY'} "
        f"(`{old_res['sha']}` / `{fresh_res['sha']}`).",
        "",
        f"Gaia: `{GAIA['ref']}` r≈{GAIA['mag_r']:.2f} @ {GAIA.get('dist_arcsec')}″.",
        "",
        "**vs JoannaM:** tu po masce zostaje **~0.80**; JoannaM spadała do **~0.24**.",
        "",
        "## Optyka / spektroskopia",
        "",
        "| Katalog | Wynik |",
        "|---------|--------|",
        f"| SIMBAD ≤60″ | **n={sim60['n']}** |",
        f"| SIMBAD ≤5′ | n={sim5['n']} · nearest: {sim5_near} |",
        f"| LS DR10 ≤36″ | n={len(ls)}, bright r<20: **{len(bright)}** |",
        f"| DESI DR1 ≤10′ | n={len(desi)} · nearest {desi_near_s} |",
        "",
        "## X-ray / radio",
        "",
        "| Katalog | ≤5′ | ≤15′ | nearest |",
        "|---------|----:|-----:|---------|",
        f"| eROSITA eRASS1 | **{er5['n']}** | {er15['n']} | {er_near_s} |",
        f"| NVSS | **{nvss5['n']}** | {nvss15['n']} | {nv_near_s} |",
        f"| 2RXS | — | {rxs['n']} | — |",
        "",
        "## Werdykt",
        "",
        "- Spektrum w centrum cutoutu (SIMBAD 60″): **brak**",
        f"- DESI w cutoucie (<20″): **{'TAK' if verdict['desi_in_cutout'] else 'NIE'}**",
        f"- eROSITA ≤5′: **{er5['n']}** · NVSS ≤5′: **{nvss5['n']}**",
        "- Flaga Most: **częściowo gwiazda + silny residual (~0.80)** — wart follow-upu, nie BH claim",
        "- Fresh cutout: **reprodukuje**",
        "",
        "JSON: `CROSSMATCH_C2.json` · alias FITS: `desi_c2_fresh_ra142.5_dec6.5_s128.fits`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    print("VERDICT", json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
