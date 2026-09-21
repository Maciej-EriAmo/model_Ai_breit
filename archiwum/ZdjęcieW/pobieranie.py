"""
Pobiera cutout DESI Legacy Survey i zapisuje do ZdjęcieW/demo/

  python pobieranie.py
  python pobieranie.py --ra 194.953 --dec 27.981 --name coma
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "demo"
CUTOUT_URL = "https://www.legacysurvey.org/viewer/fits-cutout"


def fetch_fits(
    ra: float,
    dec: float,
    *,
    size: int = 256,
    bands: str = "grz",
    layer: str = "ls-dr10",
    pixscale: float = 0.262,
    with_invvar: bool = True,
    timeout: float = 90.0,
) -> bytes:
    params: dict = {
        "ra": ra,
        "dec": dec,
        "layer": layer,
        "pixscale": pixscale,
        "bands": bands,
        "size": size,
    }
    # pusty invvar= → HTTP 500; "1" dokłada HDU inverse-variance
    if with_invvar:
        params["invvar"] = "1"
    r = requests.get(CUTOUT_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.content


def fits_to_rgb_preview(fits_bytes: bytes) -> Image.Image:
    """Prosty podgląd: pasma g,r,z → RGB (log + percentyle)."""
    hdul = fits.open(io.BytesIO(fits_bytes))
    try:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=0)
        # oczekiwane [C,H,W]; Legacy bywa [C,H,W]
        if data.shape[0] > 8 and data.ndim == 3:
            # czasem [H,W,C]
            data = np.transpose(data, (2, 0, 1))
        bands = []
        for c in range(min(3, data.shape[0])):
            band = np.nan_to_num(data[c], nan=0.0, posinf=0.0, neginf=0.0)
            band = np.log1p(np.clip(band, 0.0, None))
            lo, hi = np.percentile(band, [2, 98])
            if hi <= lo:
                hi = lo + 1e-6
            band = np.clip((band - lo) / (hi - lo), 0, 1)
            bands.append((band * 255).astype(np.uint8))
        while len(bands) < 3:
            bands.append(bands[-1])
        rgb = np.stack(bands[:3], axis=-1)  # H,W,3
        return Image.fromarray(rgb, mode="RGB")
    finally:
        hdul.close()


def save_cutout(
    *,
    ra: float,
    dec: float,
    name: str,
    size: int = 256,
    out_dir: Path = OUT_DIR,
    no_png: bool = False,
    skip_existing: bool = True,
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"desi_{name}_ra{ra}_dec{dec}_s{size}"
    fits_path = out_dir / f"{stem}.fits"
    png_path = out_dir / f"{stem}.png"

    if skip_existing and fits_path.is_file() and fits_path.stat().st_size > 1000:
        print(f"  exists {fits_path.name}")
        return fits_path, png_path if png_path.is_file() else None

    print(f"fetch ra={ra} dec={dec} size={size} → {out_dir}")
    raw = fetch_fits(ra, dec, size=size)
    fits_path.write_bytes(raw)
    print(f"  FITS {fits_path} ({fits_path.stat().st_size} B)")

    if no_png:
        return fits_path, None
    preview = fits_to_rgb_preview(raw)
    preview.save(png_path)
    print(f"  PNG  {png_path} ({preview.size[0]}x{preview.size[1]})")
    return fits_path, png_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DESI cutout → ZdjęcieW/demo/")
    ap.add_argument("--ra", type=float, default=185.0)
    ap.add_argument("--dec", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=128, help="bok cutoutu (domyślnie 128)")
    ap.add_argument("--name", default="field", help="krótka nazwa w pliku")
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"katalog docelowy (domyślnie {OUT_DIR})",
    )
    args = ap.parse_args(argv)
    save_cutout(
        ra=args.ra,
        dec=args.dec,
        name=args.name,
        size=args.size,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
