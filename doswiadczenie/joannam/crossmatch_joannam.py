"""Cone cross-match for JoannaM (RA 120.0, Dec +5.0).

Queries: SIMBAD, NED, SDSS DR17 SpecObj, Astro Data Lab DESI (if available).
Writes JSON summary next to this script.
"""
from __future__ import annotations

import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RA = 120.0
DEC = 5.0
OUT = Path(__file__).with_name("CROSSMATCH_RA120_DEC5.json")

# Lab / Windows Python often lacks CDS/SDSS CA chain — use unverified for catalog TAP.
CTX = ssl._create_unverified_context()


def http_get(url: str, timeout: int = 90) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "JoannaM-crossmatch/1.0"})
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def http_post(url: str, data: dict, timeout: int = 90) -> str:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"User-Agent": "JoannaM-crossmatch/1.0"}
    )
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def angsep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = math.radians((ra2 - ra1) * math.cos(math.radians(0.5 * (dec1 + dec2))))
    ddec = math.radians(dec2 - dec1)
    return math.degrees(math.hypot(dra, ddec)) * 3600.0


def query_simbad(radius_arcmin: float = 5.0) -> dict:
    script = f"""
output console=off script=off
format object "%MAIN_ID|%COO(A)|%COO(D)|%OTYPE|%SP(S)|%RV_VALUE|%PLX_VALUE|%DIST"
query coo {RA:.5f} {DEC:+.5f} radius={radius_arcmin}m
"""
    try:
        text = http_post(
            "https://simbad.cds.unistra.fr/simbad/sim-script",
            {"script": script},
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": []}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("::")]
    objects = []
    for ln in lines:
        if "|" not in ln:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 4:
            continue
        objects.append(
            {
                "id": parts[0],
                "ra": parts[1],
                "dec": parts[2],
                "otype": parts[3],
                "sp": parts[4] if len(parts) > 4 else "",
                "rv": parts[5] if len(parts) > 5 else "",
                "plx": parts[6] if len(parts) > 6 else "",
                "dist": parts[7] if len(parts) > 7 else "",
                "raw": ln,
            }
        )
    return {
        "ok": True,
        "radius_arcmin": radius_arcmin,
        "n": len(objects),
        "objects": objects,
        "raw_preview": text[:1500],
    }


def query_ned(radius_arcmin: float = 5.0) -> dict:
    # NED Object Lookup by coordinates (OFName)
    params = {
        "search_type": "Near Position Search",
        "coordinates": f"{RA}d {DEC}d",
        "radius": str(radius_arcmin),
        "in_csys": "Equatorial",
        "in_equinox": "J2000.0",
        "out_csys": "Equatorial",
        "out_equinox": "J2000.0",
        "obj_sort": "Distance to search center",
        "of": "xml_main",
        "zv_breaker": "30000.0",
        "list_limit": "50",
        "img_stamp": "NO",
    }
    url = "https://ned.ipac.caltech.edu/cgi-bin/objsearch?" + urllib.parse.urlencode(params)
    try:
        text = http_get(url, timeout=120)
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": [], "url": url}
    # crude XML extract of Object_Name / RA / DEC / Redshift
    objects = []
    chunks = text.split("<NEDREFObject")
    for ch in chunks[1:]:
        def tag(name: str) -> str:
            a = ch.find(f"<{name}>")
            b = ch.find(f"</{name}>")
            if a < 0 or b < 0:
                return ""
            return ch[a + len(name) + 2 : b].strip()

        objects.append(
            {
                "name": tag("Object_Name") or tag("NED_Object_Name"),
                "ra": tag("RA"),
                "dec": tag("DEC"),
                "otype": tag("Type") or tag("Object_Type"),
                "z": tag("Redshift"),
                "z_flag": tag("Redshift_Flag"),
                "dist_arcmin": tag("Distance_arcmin") or tag("Distance"),
            }
        )
    return {
        "ok": True,
        "radius_arcmin": radius_arcmin,
        "n": len(objects),
        "objects": objects,
        "url": url,
        "raw_preview": text[:2000],
    }


def query_sdss_spec(radius_arcmin: float = 5.0) -> dict:
    # SkyServer SQL cone around RA/Dec using SpecObj + PhotoObj
    # radius in arcmin for dbo.fGetNearbySpecObjEq
    sql = f"""
SELECT TOP 50
  s.specObjID, s.ra, s.dec, s.z, s.zErr, s.class, s.subclass,
  s.plate, s.mjd, s.fiberid, s.snMedian,
  dbo.fDistanceArcMinEq({RA},{DEC},s.ra,s.dec) AS dist_arcmin
FROM SpecObjAll AS s
WHERE s.ra BETWEEN {RA - 0.2} AND {RA + 0.2}
  AND s.dec BETWEEN {DEC - 0.2} AND {DEC + 0.2}
  AND dbo.fDistanceArcMinEq({RA},{DEC},s.ra,s.dec) < {radius_arcmin}
ORDER BY dist_arcmin
"""
    url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
    try:
        text = http_post(url, {"cmd": sql, "format": "json"})
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": []}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "bad json", "raw_preview": text[:1500], "objects": []}
    rows = []
    # DR17 returns list of Tables
    if isinstance(data, list):
        for block in data:
            if isinstance(block, dict) and "Rows" in block:
                rows.extend(block["Rows"])
    elif isinstance(data, dict):
        rows = data.get("Rows") or data.get("rows") or []
    return {"ok": True, "radius_arcmin": radius_arcmin, "n": len(rows), "objects": rows}


def _parse_datalab_csv(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("<?")]
    if not lines:
        return []
    # VOTable error?
    if "<VOTABLE" in text and "QUERY_STATUS" in text:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def query_datalab_desi(radius_arcmin: float = 5.0) -> dict:
    # Astro Data Lab TAP — desi_dr1.zpix uses mean_fiber_ra/dec
    rad_deg = radius_arcmin / 60.0
    adql = (
        "SELECT TOP 50 targetid, z, zwarn, spectype, subtype, survey, program, "
        "mean_fiber_ra, mean_fiber_dec, zcat_primary, "
        f"DISTANCE(POINT('ICRS', mean_fiber_ra, mean_fiber_dec), POINT('ICRS', {RA}, {DEC}))*60 "
        "AS dist_arcmin "
        "FROM desi_dr1.zpix WHERE 1=CONTAINS("
        f"POINT('ICRS', mean_fiber_ra, mean_fiber_dec), CIRCLE('ICRS', {RA}, {DEC}, {rad_deg})) "
        "ORDER BY dist_arcmin"
    )
    url = "https://datalab.noirlab.edu/tap/sync"
    try:
        text = http_post(
            url,
            {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "QUERY": adql,
            },
            timeout=180,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": [], "query": adql}
    if "QUERY_STATUS" in text and "ERROR" in text:
        return {
            "ok": False,
            "error": "TAP error",
            "raw_preview": text[:1500],
            "objects": [],
            "query": adql,
        }
    objects = _parse_datalab_csv(text)
    return {
        "ok": True,
        "radius_arcmin": radius_arcmin,
        "n": len(objects),
        "objects": objects,
        "query": adql,
        "raw_preview": text[:1500],
    }


def query_datalab_ls_tractor(radius_arcsec: float = 36.0) -> dict:
    """Legacy Surveys DR10 tractor sources inside ~cutout half-diagonal."""
    rad_deg = radius_arcsec / 3600.0
    adql = (
        "SELECT TOP 40 ra, dec, type, mag_g, mag_r, mag_z, parallax, ref_cat, ref_id, "
        f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {RA}, {DEC}))*3600 AS dist_arcsec "
        "FROM ls_dr10.tractor WHERE 1=CONTAINS("
        f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {RA}, {DEC}, {rad_deg})) "
        "ORDER BY dist_arcsec"
    )
    try:
        text = http_post(
            "https://datalab.noirlab.edu/tap/sync",
            {
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "QUERY": adql,
            },
            timeout=180,
        )
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": [], "query": adql}
    if "QUERY_STATUS" in text and "ERROR" in text:
        return {
            "ok": False,
            "error": "TAP error",
            "raw_preview": text[:1500],
            "objects": [],
            "query": adql,
        }
    objects = _parse_datalab_csv(text)
    return {
        "ok": True,
        "radius_arcsec": radius_arcsec,
        "n": len(objects),
        "objects": objects,
        "query": adql,
        "raw_preview": text[:2000],
    }


def query_vizier_gaia(radius_arcmin: float = 2.0) -> dict:
    # Gaia DR3 via VizieR cone — photometric/astrometric context
    rad_deg = radius_arcmin / 60.0
    url = (
        "https://vizier.cds.unistra.fr/viz-bin/votable?-source=I/355/gaiadr3"
        f"&-c={RA}+{DEC}&-c.rd={rad_deg}&-out.max=30"
        "&-out=Source,RA_ICRS,DE_ICRS,Plx,Gmag,BPmag,RPmag,RV,Teff"
    )
    try:
        text = http_get(url, timeout=90)
    except Exception as e:
        return {"ok": False, "error": str(e), "objects": []}
    # count TR rows roughly
    n_tr = text.count("<TR>")
    return {
        "ok": True,
        "radius_arcmin": radius_arcmin,
        "n_tr_rows": n_tr,
        "raw_preview": text[:2500],
        "url": url,
    }


def main() -> None:
    summary = {
        "target": {
            "name": "JoannaM",
            "ra_deg": RA,
            "dec_deg": DEC,
            "cutout": "desi_grid_004_ra120.0_dec5.0_s128",
            "note": "Most p_cluster~0.980 optical anomaly — not confirmed BH",
        },
        "queries": {},
    }
    print("SIMBAD 5' ...")
    summary["queries"]["simbad_5arcmin"] = query_simbad(5.0)
    print("  n=", summary["queries"]["simbad_5arcmin"].get("n"))

    print("NED 5' ...")
    summary["queries"]["ned_5arcmin"] = query_ned(5.0)
    print("  n=", summary["queries"]["ned_5arcmin"].get("n"), "ok=", summary["queries"]["ned_5arcmin"].get("ok"))

    print("SDSS SpecObj 5' ...")
    summary["queries"]["sdss_spec_5arcmin"] = query_sdss_spec(5.0)
    print("  n=", summary["queries"]["sdss_spec_5arcmin"].get("n"), "ok=", summary["queries"]["sdss_spec_5arcmin"].get("ok"))

    print("DESI Data Lab zpix 5' ...")
    summary["queries"]["desi_zpix_5arcmin"] = query_datalab_desi(5.0)
    print(
        "  n=",
        summary["queries"]["desi_zpix_5arcmin"].get("n"),
        "ok=",
        summary["queries"]["desi_zpix_5arcmin"].get("ok"),
        "err=",
        summary["queries"]["desi_zpix_5arcmin"].get("error"),
    )

    print("Legacy Surveys DR10 tractor 36\" ...")
    summary["queries"]["ls_dr10_tractor_36arcsec"] = query_datalab_ls_tractor(36.0)
    print(
        "  n=",
        summary["queries"]["ls_dr10_tractor_36arcsec"].get("n"),
        "ok=",
        summary["queries"]["ls_dr10_tractor_36arcsec"].get("ok"),
        "err=",
        summary["queries"]["ls_dr10_tractor_36arcsec"].get("error"),
    )

    print("Gaia DR3 Vizier 2' ...")
    summary["queries"]["gaia_dr3_2arcmin"] = query_vizier_gaia(2.0)
    print("  rows~", summary["queries"]["gaia_dr3_2arcmin"].get("n_tr_rows"))

    # Also wider SIMBAD 15' for context
    print("SIMBAD 15' ...")
    summary["queries"]["simbad_15arcmin"] = query_simbad(15.0)
    print("  n=", summary["queries"]["simbad_15arcmin"].get("n"))

    # Manual SIMBAD summary from browser (authoritative when SSL blocks script)
    summary["manual_browser"] = {
        "simbad_60arcsec": {"n": 0, "note": "No SIMBAD object within 60\""},
        "simbad_10arcmin": {
            "n": 43,
            "nearest": [
                {"id": "Zel 0757+051", "otype": "Rad", "dist_asec": 174.44},
                {"id": "SDSS J075947.23+045938.4", "otype": "G", "dist_asec": 191.86},
                {"id": "CRTS J075959.7+050400", "otype": "EB*", "dist_asec": 240.34},
                {"id": "Gaia DR3 3095367640144517632", "otype": "AGN", "dist_asec": 249.19},
            ],
        },
        "sdss_specobj_5arcmin": {
            "n": 0,
            "note": "SkyServer DR17 SpecObjAll: No entries within 5'",
        },
    }

    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
