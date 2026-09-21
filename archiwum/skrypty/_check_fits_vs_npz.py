"""FITS vs NPZ + morfologia martwych punktów."""
from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "archiwum" / "ZdjęcieW" / "demo"
CACHE = ROOT / "dumps" / "desi_cache"

STEMS = [
    "desi_bh_cena_ra201.3651_dec-43.0191_s128",
    "desi_bh_ton618_ra186.3542_dec31.5261_s128",
    "desi_bh_3c345_ra250.745_dec39.81_s128",
    "desi_bh_mrk421_ra166.1138_dec38.2088_s128",
    "desi_bh_3c48_ra24.4221_dec33.1598_s128",
    "desi_bh_cta102_ra338.1517_dec11.7308_s128",
    "desi_bh_oj287_ra133.7036_dec20.1085_s128",
]


def morph(img: np.ndarray) -> dict:
    """img [C,H,W] normalized. concentration + core dominance."""
    flux = np.nan_to_num(img, nan=0.0).mean(axis=0)
    H, W = flux.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    r = np.sqrt(r2)
    # half-light radius proxy: radius enclosing 50% of positive flux
    pos = np.clip(flux, 0, None)
    total = pos.sum() + 1e-12
    order = np.argsort(r.ravel())
    csum = np.cumsum(pos.ravel()[order])
    r_sorted = r.ravel()[order]
    i50 = int(np.searchsorted(csum, 0.5 * total))
    i80 = int(np.searchsorted(csum, 0.8 * total))
    r50 = float(r_sorted[min(i50, len(r_sorted) - 1)])
    r80 = float(r_sorted[min(i80, len(r_sorted) - 1)])
    core = r <= 4
    ring = (r > 8) & (r <= 24)
    core_m = float(pos[core].mean()) if core.any() else 0.0
    ring_m = float(pos[ring].mean()) if ring.any() else 0.0
    # saturation proxy: fraction of core at hard 1.0 after normalize
    sat = float((flux[core] >= 0.99).mean()) if core.any() else 0.0
    return {
        "r50": r50,
        "r80": r80,
        "conc_r80_r50": r80 / max(r50, 1e-6),
        "core_ring_ratio": core_m / max(ring_m, 1e-8),
        "sat_frac_core": sat,
        "total_flux": float(total),
    }


def main() -> None:
    print("=" * 72)
    print("FITS vs NPZ")
    print("=" * 72)
    for stem in STEMS:
        fits_p = DEMO / f"{stem}.fits"
        npz_p = CACHE / f"demo_{stem}.npz"
        print(f"\n## {stem}")
        if fits_p.exists():
            with fits.open(fits_p) as hdul:
                for i, h in enumerate(hdul):
                    d = h.data
                    if d is None:
                        print(f"  HDU{i}: None")
                        continue
                    a = np.asarray(d, dtype=np.float64)
                    finite = np.isfinite(a)
                    print(
                        f"  HDU{i}: shape={a.shape} "
                        f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} "
                        f"mean={np.nanmean(a):.4g} "
                        f"nan={(~finite).mean():.3f} zero={(a == 0).mean():.3f}"
                    )
        else:
            print("  FITS missing")
        if npz_p.exists():
            z = np.load(npz_p, allow_pickle=True)
            img = z["image"]
            src = str(z["source"]) if "source" in z.files else "?"
            print(
                f"  NPZ: min={img.min():.4g} max={img.max():.4g} "
                f"mean={img.mean():.4g} nonzero={(img != 0).mean():.3f}"
            )
            print(f"  source={src}")
            m = morph(img)
            print(
                f"  morph: r50={m['r50']:.1f} r80={m['r80']:.1f} "
                f"conc={m['conc_r80_r50']:.2f} core/ring={m['core_ring_ratio']:.1f} "
                f"sat={m['sat_frac_core']:.3f} tot={m['total_flux']:.1f}"
            )


if __name__ == "__main__":
    main()
