"""
DESI Legacy Survey cutouts → cache NPZ (tor 1: nauka Most na niebie).

FITS + invvar → SNR → T (hot/cold). XY zostaje osobno w image_batch.
Nie mylić z screenshotem Lunety (płaski RGB bez τ).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "dumps" / "desi_cache"
DEMO_DIR = ROOT / "ZdjęcieW" / "demo"
CUTOUT_URL = "https://www.legacysurvey.org/viewer/fits-cutout"

# Pola w footprintcie Legacy (unikaj Cygnus A — FITS 500 poza pokryciem).
SEED_FIELDS = [
    {"name": "coma", "ra": 194.953, "dec": 27.981, "hint": "cluster"},
    {"name": "virgo_core", "ra": 187.7, "dec": 12.3, "hint": "cluster"},
    {"name": "field_a", "ra": 150.0, "dec": 2.0, "hint": "field"},
    {"name": "field_b", "ra": 30.0, "dec": -5.0, "hint": "field"},
    {"name": "field_c", "ra": 210.0, "dec": 40.0, "hint": "field"},
    {"name": "field_d", "ra": 5.0, "dec": 15.0, "hint": "field"},
]


def fetch_legacy_cutout(
    ra: float,
    dec: float,
    *,
    size: int = 128,
    bands: str = "grz",
    layer: str = "ls-dr10",
    pixscale: float = 0.262,
    with_invvar: bool = True,
    timeout: float = 90.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pobiera FITS cutout. Zwraca (image [C,H,W], invvar [C,H,W]|None)."""
    from astropy.io import fits

    params: dict = {
        "ra": ra,
        "dec": dec,
        "layer": layer,
        "pixscale": pixscale,
        "bands": bands,
        "size": size,
    }
    # Pusty invvar= powoduje 500; wartość "1" dokłada HDU z inverse-variance.
    if with_invvar:
        params["invvar"] = "1"
    r = requests.get(CUTOUT_URL, params=params, timeout=timeout)
    r.raise_for_status()
    hdul = fits.open(io.BytesIO(r.content))
    try:
        img = np.asarray(hdul[0].data, dtype=np.float32)
        if img.ndim == 2:
            img = img[None, ...]
        inv = None
        if len(hdul) > 1 and hdul[1].data is not None:
            inv = np.asarray(hdul[1].data, dtype=np.float32)
            if inv.ndim == 2:
                inv = inv[None, ...]
    finally:
        hdul.close()
    return img, inv


def normalize_log(img: np.ndarray) -> np.ndarray:
    """Log1p + per-band median subtract; clip outliers."""
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        band = np.nan_to_num(img[c], nan=0.0, posinf=0.0, neginf=0.0)
        band = np.log1p(np.clip(band, 0.0, None))
        med = float(np.median(band))
        band = band - med
        lo, hi = np.percentile(band, [1.0, 99.0])
        if hi > lo:
            band = np.clip(band, lo, hi)
            band = (band - lo) / (hi - lo + 1e-8)
        out[c] = band.astype(np.float32)
    return out


def snr_map(img: np.ndarray, invvar: np.ndarray | None) -> np.ndarray:
    """Mapa SNR [H,W] — pod T. Bez invvar: lokalny kontrast jako proxy."""
    if invvar is not None and invvar.shape == img.shape:
        flux = np.nan_to_num(img, nan=0.0).mean(axis=0)
        iv = np.nan_to_num(invvar, nan=0.0).mean(axis=0)
        iv = np.clip(iv, 0.0, None)
        return (flux * np.sqrt(iv + 1e-12)).astype(np.float32)
    # proxy: |flux - local median|
    flux = np.nan_to_num(img, nan=0.0).mean(axis=0)
    # crude box blur via stride mean
    k = 5
    pad = k // 2
    padded = np.pad(flux, pad, mode="edge")
    H, W = flux.shape
    local = np.zeros_like(flux)
    for i in range(H):
        for j in range(W):
            local[i, j] = padded[i : i + k, j : j + k].mean()
    return np.abs(flux - local).astype(np.float32)


def snr_to_T(snr: np.ndarray) -> np.ndarray:
    """Legacy: T z SNR. Preferuj color_to_T (barwa = ciepłota)."""
    return _percentile01(snr)


def color_index_map(img: np.ndarray) -> np.ndarray:
    """
    Mapa barwy g−r z log-flux (PRZED per-band stretch).
    Dodatnia / ujemna = różnica „ciepłoty” populacji / widma.
    """
    x = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.log1p(np.clip(x, 0.0, None))
    if x.shape[0] >= 2:
        return (x[0] - x[1]).astype(np.float32)
    return np.zeros(x.shape[1:], dtype=np.float32)


def color_rz_map(img: np.ndarray) -> np.ndarray:
    """Drugi indeks barwy r−z (gdy są ≥3 pasma)."""
    x = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.log1p(np.clip(x, 0.0, None))
    if x.shape[0] >= 3:
        return (x[1] - x[2]).astype(np.float32)
    return np.zeros(x.shape[1:], dtype=np.float32)


def _percentile01(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    lo, hi = np.percentile(flat, [10.0, 90.0])
    if hi <= lo:
        return np.full_like(arr, 0.5, dtype=np.float32)
    t = (arr - lo) / (hi - lo + 1e-8)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def color_to_T(color: np.ndarray) -> np.ndarray:
    """T ∈ [0,1] z barwy (ciepłota fotometryczna → oś obserwatora)."""
    return _percentile01(color)


def pack_photometry(img: np.ndarray, inv: np.ndarray | None) -> dict:
    """
    Jasność/morfologia vs barwa→T (nie mylić).
    image (norm) → content; T z g−r; snr osobno (pewność).
    """
    color = color_index_map(img)
    color_rz = color_rz_map(img)
    # T: mieszanka g−r (główna) + lekko r−z — nadal barwa, nie SNR
    T = color_to_T(0.7 * color + 0.3 * color_rz)
    snr = snr_map(img, inv)
    norm = normalize_log(img)
    return {
        "image": norm,
        "snr": snr,
        "color_gr": color,
        "color_rz": color_rz,
        "T": T,
        "T_mode": "color_temperature",
    }


def weak_label_density(
    img: np.ndarray,
    *,
    radius_frac: float = 0.25,
    thr_lo: float = 0.33,
    thr_hi: float = 0.66,
) -> int:
    """
    Słaba etykieta z gęstości jasnych pikseli w centrum.
    0=pustka, 1=średnie, 2=klaster.
    """
    flux = np.nan_to_num(img, nan=0.0).mean(axis=0)
    H, W = flux.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = radius_frac * min(H, W)
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    # źródła ≈ powyżej lokalnego 85. percentyla
    thr = float(np.percentile(flux, 85.0))
    dens = float((flux[mask] > thr).mean()) if mask.any() else 0.0
    if dens < thr_lo * 0.15:
        return 0
    if dens > thr_hi * 0.25:
        return 2
    return 1


def cache_path(name: str, size: int, layer: str) -> Path:
    return CACHE / f"{name}_s{size}_{layer}.npz"


def fetch_and_cache(
    fields: Iterable[dict],
    *,
    size: int = 128,
    layer: str = "ls-dr10",
    force: bool = False,
) -> list[Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for f in fields:
        name = str(f["name"])
        path = cache_path(name, size, layer)
        paths.append(path)
        if path.exists() and not force:
            print(f"  cache hit {path.name}")
            continue
        print(f"  fetch {name} ra={f['ra']} dec={f['dec']} …")
        try:
            img, inv = fetch_legacy_cutout(
                float(f["ra"]), float(f["dec"]), size=size, layer=layer
            )
        except requests.HTTPError as e:
            print(f"    SKIP {name}: {e}")
            paths.pop()
            continue
        photo = pack_photometry(img, inv)
        hint = str(f.get("hint", "field"))
        # Smoke: binary hint cluster=1 / field=0 (stabilniejsze niż surowa gęstość).
        if hint == "cluster":
            label = 1
        elif hint == "field":
            label = 0
        else:
            label = 1 if weak_label_density(photo["image"]) >= 2 else 0
        dens_label = weak_label_density(photo["image"])
        np.savez_compressed(
            path,
            image=photo["image"],
            snr=photo["snr"],
            color_gr=photo["color_gr"],
            color_rz=photo["color_rz"],
            T=photo["T"],
            T_mode=np.asarray(photo["T_mode"]),
            label=np.int64(label),
            dens_label=np.int64(dens_label),
            ra=np.float64(f["ra"]),
            dec=np.float64(f["dec"]),
            name=np.asarray(name),
            hint=np.asarray(hint),
            has_invvar=np.bool_(inv is not None),
        )
        print(
            f"    → {path.name} shape={norm.shape} label={label} "
            f"(dens={dens_label}) invvar={inv is not None}"
        )
    meta = {
        "n": len(paths),
        "size": size,
        "layer": layer,
        "files": [p.name for p in paths],
    }
    (CACHE / "manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return paths


def load_cached(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_fits_bytes_or_path(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """FITS z dysku → (image [C,H,W], invvar|None)."""
    from astropy.io import fits

    hdul = fits.open(path)
    try:
        img = np.asarray(hdul[0].data, dtype=np.float32)
        if img.ndim == 2:
            img = img[None, ...]
        if img.shape[0] > 8 and img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        inv = None
        if len(hdul) > 1 and hdul[1].data is not None:
            inv = np.asarray(hdul[1].data, dtype=np.float32)
            if inv.ndim == 2:
                inv = inv[None, ...]
            if inv is not None and inv.shape[0] > 8 and inv.ndim == 3:
                inv = np.transpose(inv, (2, 0, 1))
    finally:
        hdul.close()
    return img, inv


_CLUSTER_KEYS = frozenset(
    {
        "coma",
        "virgo",
        "cluster",
        "cygnus",
        "perseus",
        "hercules",
        "leo",
        "hydra",
        "centaurus",
        "fornax",
        "abell",
        "abell85",
    }
)


def label_from_demo_name(name: str, labels_map: dict | None = None) -> int:
    """
    cluster=1 / field=0.
    Priorytet: labels.json → token `field` → exact token klastra (nie substring!).
    """
    import re

    stem = Path(name).stem if ("/" in name or "\\" in name) else name
    stem = stem.lower()
    tokens = [t for t in re.split(r"[_\-]+", stem) if t]
    if labels_map:
        if stem in labels_map:
            return 1 if str(labels_map[stem]).lower() in ("cluster", "1", "true") else 0
        for k, v in labels_map.items():
            kl = k.lower()
            if kl in tokens or stem.startswith(f"desi_{kl}_"):
                return 1 if str(v).lower() in ("cluster", "1", "true") else 0

    # field_* / field185 → zawsze pole (token exact / prefix)
    for t in tokens:
        if t == "field" or t.startswith("field"):
            return 0
        if t == "cluster":
            return 1
    for t in tokens:
        if t in _CLUSTER_KEYS:
            return 1
        # abell1689 / abell2744 …
        if t.startswith("abell"):
            return 1
    return 0


def _parse_size_from_stem(stem: str) -> int | None:
    import re

    m = re.search(r"_s(\d+)(?:_|$)", stem)
    return int(m.group(1)) if m else None


def _ra_dec_key(stem: str) -> str | None:
    """Klucz dedupe po (ra,dec,size) — ta sama pozycja nieba = jeden sample."""
    import re

    m = re.search(
        r"_ra(-?\d+(?:\.\d+)?)_dec(-?\d+(?:\.\d+)?)_s(\d+)$",
        stem,
        re.I,
    )
    if m:
        ra, dec, sz = float(m.group(1)), float(m.group(2)), m.group(3)
        return f"ra{ra:.4f}_dec{dec:.4f}_s{sz}"
    m2 = re.match(r"(desi_.+_ra.+_dec.+)_s\d+$", stem)
    if m2:
        return m2.group(1).replace("field185", "field")
    return stem.replace("field185", "field")


def import_demo_fits(
    demo_dir: Path | None = None,
    *,
    force: bool = False,
    prefer_size: int | None = 128,
) -> list[Path]:
    """
    ZdjęcieW/demo/desi_*.fits → dumps/desi_cache/demo_*.npz.
    prefer_size: tylko ten bok (domyślnie 128); dedupe po (name,ra,dec).
    """
    demo_dir = Path(demo_dir or DEMO_DIR)
    CACHE.mkdir(parents=True, exist_ok=True)
    fits_files = sorted(demo_dir.glob("desi_*.fits"))
    if not fits_files:
        raise FileNotFoundError(f"brak desi_*.fits w {demo_dir}")

    labels_map: dict = {}
    lab_path = demo_dir / "labels.json"
    if lab_path.is_file():
        labels_map = json.loads(lab_path.read_text(encoding="utf-8"))

    # filtr size + dedupe (ra,dec) — preferuj prefer_size
    ranked: list[Path] = []
    for fp in fits_files:
        sz = _parse_size_from_stem(fp.stem)
        if prefer_size is not None and sz is not None and sz != prefer_size:
            continue
        ranked.append(fp)

    out_paths: list[Path] = []
    seen: set[str] = set()
    for fp in ranked:
        key = _ra_dec_key(fp.stem)
        if key in seen:
            print(f"  skip dup {fp.name}")
            continue
        seen.add(key)
        out = CACHE / f"demo_{fp.stem}.npz"
        out_paths.append(out)
        if out.exists() and not force:
            print(f"  cache hit {out.name}")
            continue
        print(f"  import {fp.name} …")
        img, inv = load_fits_bytes_or_path(fp)
        if prefer_size is not None and (
            img.shape[-1] != prefer_size or img.shape[-2] != prefer_size
        ):
            print(f"    skip shape {img.shape} ≠ s{prefer_size}")
            out_paths.pop()
            continue
        photo = pack_photometry(img, inv)
        label = label_from_demo_name(fp.stem, labels_map)
        dens = weak_label_density(photo["image"])
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
            dens_label=np.int64(dens),
            name=np.asarray(fp.stem),
            hint=np.asarray(hint),
            has_invvar=np.bool_(inv is not None),
            source=np.asarray(str(fp)),
        )
        print(
            f"    → {out.name} shape={photo['image'].shape} label={label} "
            f"hint={hint} T=color"
        )
    return out_paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DESI Legacy cutouts → dumps/desi_cache")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--layer", default="ls-dr10")
    p.add_argument("--force", action="store_true")
    p.add_argument("--n", type=int, default=len(SEED_FIELDS), help="ile pól z SEED_FIELDS")
    args = p.parse_args(argv)
    fields = SEED_FIELDS[: max(1, min(args.n, len(SEED_FIELDS)))]
    print(f"DESI fetch n={len(fields)} size={args.size} → {CACHE}")
    paths = fetch_and_cache(fields, size=args.size, layer=args.layer, force=args.force)
    # smoke shapes
    for path in paths:
        d = load_cached(path)
        assert d["image"].ndim == 3, d["image"].shape
        assert d["T"].shape == d["image"].shape[1:], (d["T"].shape, d["image"].shape)
        print(
            f"OK {path.name}: img={d['image'].shape} T=[{float(d['T'].min()):.2f},"
            f"{float(d['T'].max()):.2f}] label={int(d['label'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
