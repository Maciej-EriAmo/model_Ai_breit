"""Porównanie surowych cech cutoutów bh: martwe punkty LOO vs zawsze złapane."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "dumps" / "desi_cache"

DEAD = ("3c345", "mrk421", "ton618")
CAUGHT = ("3c48", "cena", "cta102", "oj287")
# pozostałe bh z LOO (częściowo łapane)
OTHER = ("ngc4151", "m87", "3c273", "3c279")


def find_bh(token: str) -> Path:
    hits = sorted(CACHE.glob(f"*bh*{token}*.npz"))
    if not hits:
        raise FileNotFoundError(token)
    return hits[0]


def center_mask(h: int, w: int, frac: float = 0.25) -> np.ndarray:
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = frac * min(h, w)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def feats(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    img = np.asarray(z["image"], dtype=np.float32)  # [C,H,W] already log-normed
    snr = np.asarray(z["snr"], dtype=np.float32)
    cgr = np.asarray(z["color_gr"], dtype=np.float32)
    crz = np.asarray(z["color_rz"], dtype=np.float32)
    T = np.asarray(z["T"], dtype=np.float32)
    H, W = snr.shape
    m = center_mask(H, W, 0.25)
    m5 = center_mask(H, W, 0.08)  # rdzeń
    flux = img.mean(axis=0)
    # kontrast: std / mean w centrum; p95-p5
    f_c = flux[m]
    f_core = flux[m5]
    return {
        "file": path.name,
        "token": path.stem,
        "hint": str(z["hint"]) if "hint" in z.files else "?",
        "has_invvar": bool(z["has_invvar"]) if "has_invvar" in z.files else None,
        "flux_mean": float(flux.mean()),
        "flux_center_mean": float(f_c.mean()),
        "flux_core_mean": float(f_core.mean()),
        "flux_core_max": float(f_core.max()),
        "contrast_std": float(f_c.std()),
        "contrast_p95_p5": float(np.percentile(f_c, 95) - np.percentile(f_c, 5)),
        "peak_over_med": float(f_core.max() / (np.median(flux) + 1e-8)),
        "snr_mean": float(snr.mean()),
        "snr_center": float(snr[m].mean()),
        "snr_core_max": float(snr[m5].max()),
        "snr_p95": float(np.percentile(snr, 95)),
        "T_mean": float(T.mean()),
        "T_center": float(T[m].mean()),
        "color_gr_center": float(cgr[m].mean()),
        "color_rz_center": float(crz[m].mean()),
        "band_means": [float(img[c].mean()) for c in range(img.shape[0])],
        "band_center": [float(img[c][m].mean()) for c in range(img.shape[0])],
        "frac_bright_center": float((f_c > np.percentile(flux, 85)).mean()),
    }


def group_stats(rows: list[dict], keys: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(vals.mean()), float(vals.std(ddof=0)))
    return out


def main() -> None:
    groups = {
        "DEAD": [feats(find_bh(t)) for t in DEAD],
        "CAUGHT": [feats(find_bh(t)) for t in CAUGHT],
        "OTHER": [feats(find_bh(t)) for t in OTHER],
    }

    metric_keys = [
        "flux_center_mean",
        "flux_core_mean",
        "flux_core_max",
        "contrast_std",
        "contrast_p95_p5",
        "peak_over_med",
        "snr_center",
        "snr_core_max",
        "snr_p95",
        "T_center",
        "color_gr_center",
        "color_rz_center",
        "frac_bright_center",
    ]

    print("=" * 72)
    print("PER-OBJECT")
    print("=" * 72)
    for gname, rows in groups.items():
        print(f"\n## {gname}")
        for r in rows:
            print(
                f"  {r['file']}\n"
                f"    hint={r['hint']} invvar={r['has_invvar']}\n"
                f"    flux_c={r['flux_center_mean']:.4f} core_max={r['flux_core_max']:.4f} "
                f"contrast={r['contrast_std']:.4f} peak/med={r['peak_over_med']:.2f}\n"
                f"    snr_c={r['snr_center']:.4f} snr_core_max={r['snr_core_max']:.4f} "
                f"snr_p95={r['snr_p95']:.4f}\n"
                f"    T_c={r['T_center']:.3f} g-r={r['color_gr_center']:.4f} "
                f"r-z={r['color_rz_center']:.4f} bright_frac={r['frac_bright_center']:.3f}\n"
                f"    bands_c={['%.3f'%x for x in r['band_center']]}"
            )

    print("\n" + "=" * 72)
    print("GROUP MEANS ± STD  (DEAD vs CAUGHT)")
    print("=" * 72)
    sd = group_stats(groups["DEAD"], metric_keys)
    sc = group_stats(groups["CAUGHT"], metric_keys)
    print(f"{'metric':22s} {'DEAD mean':>10s} {'CAUGHT mean':>12s} {'Δ(D-C)':>10s} {'ratio D/C':>10s}")
    for k in metric_keys:
        dm, _ = sd[k]
        cm, _ = sc[k]
        ratio = dm / cm if abs(cm) > 1e-12 else float("nan")
        print(f"{k:22s} {dm:10.4f} {cm:12.4f} {dm-cm:10.4f} {ratio:10.3f}")

    # ranking all 11 by snr_core_max and peak_over_med
    all_rows = []
    for g, rows in groups.items():
        for r in rows:
            r2 = dict(r)
            r2["group"] = g
            all_rows.append(r2)
    print("\n" + "=" * 72)
    print("RANK by snr_core_max (ASC = najsłabsze)")
    print("=" * 72)
    for r in sorted(all_rows, key=lambda x: x["snr_core_max"]):
        tok = next(t for t in DEAD + CAUGHT + OTHER if t in r["file"])
        print(
            f"  {r['group']:6s} {tok:10s} snr_core_max={r['snr_core_max']:8.3f} "
            f"peak/med={r['peak_over_med']:6.2f} flux_core={r['flux_core_mean']:.4f} "
            f"T_c={r['T_center']:.3f}"
        )


if __name__ == "__main__":
    main()
