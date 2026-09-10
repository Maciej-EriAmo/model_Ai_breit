"""
Duży set DESI s128 → demo/ (+ osobne holdout „nowe obiekty”).

  python pobierz_duzy_set.py --target 120 --holdout 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pobieranie import save_cutout  # noqa: E402
from pobierz_paczke import PACK  # noqa: E402

# Gęsta siatka w typowym footprintcie Legacy (unikaj skrajnych DEC/RA-dziur)
def field_grid() -> list[tuple[str, float, float, str]]:
    rows: list[tuple[str, float, float, str]] = []
    n = 0
    for ra in range(120, 241, 5):  # 120..240
        for dec in range(-15, 31, 5):  # -15..30
            rows.append((f"grid_{n:03d}", float(ra), float(dec), "field"))
            n += 1
    for ra in range(0, 61, 5):
        for dec in range(-10, 21, 5):
            rows.append((f"gridb_{n:03d}", float(ra), float(dec), "field"))
            n += 1
    return rows


# Hold-out „nowe obiekty” — nie w PACK / nie w typowej siatce treningowej
HOLDOUT_OBJECTS: list[tuple[str, float, float, str]] = [
    ("new_abell2219", 250.08, 46.71, "cluster"),  # może 500 — spróbuj
    ("new_abell1835", 210.26, 2.88, "cluster"),
    ("new_abell2390", 328.40, 17.70, "cluster"),
    ("new_coma_far", 196.5, 26.5, "cluster"),
    ("new_virgo_far", 189.5, 11.0, "cluster"),
    ("new_leo_far", 171.8, 13.5, "cluster"),
    ("new_field_01", 142.5, 6.5, "field"),
    ("new_field_02", 157.5, -6.5, "field"),
    ("new_field_03", 167.5, 14.5, "field"),
    ("new_field_04", 177.5, -4.5, "field"),
    ("new_field_05", 187.5, 19.5, "field"),
    ("new_field_06", 197.5, -9.5, "field"),
    ("new_field_07", 207.5, 8.5, "field"),
    ("new_field_08", 217.5, -2.5, "field"),
    ("new_field_09", 227.5, 11.5, "field"),
    ("new_field_10", 132.5, 3.5, "field"),
    ("new_field_11", 162.5, 21.5, "field"),
    ("new_field_12", 192.5, -12.5, "field"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--target", type=int, default=120, help="ile OK cutoutów treningowych")
    ap.add_argument("--holdout", type=int, default=16, help="max holdout new_*")
    ap.add_argument("--no-png", action="store_true", default=True)
    args = ap.parse_args(argv)

    out = HERE / "demo"
    labels: dict[str, str] = {}
    if (out / "labels.json").is_file():
        labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))

    train_pool = list(PACK) + field_grid()
    ok = 0
    fail = 0
    for name, ra, dec, hint in train_pool:
        if ok >= args.target:
            break
        try:
            fp, _ = save_cutout(
                ra=ra,
                dec=dec,
                name=name,
                size=args.size,
                out_dir=out,
                no_png=args.no_png,
                skip_existing=True,
            )
            labels[fp.stem] = hint
            labels[name] = hint
            ok += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            fail += 1

    hold_ok = 0
    hold_names: list[str] = []
    for name, ra, dec, hint in HOLDOUT_OBJECTS[: args.holdout]:
        try:
            fp, _ = save_cutout(
                ra=ra,
                dec=dec,
                name=name,
                size=args.size,
                out_dir=out,
                no_png=args.no_png,
                skip_existing=True,
            )
            labels[fp.stem] = hint
            labels[name] = hint
            hold_ok += 1
            hold_names.append(fp.stem)
        except Exception as e:
            print(f"  FAIL holdout {name}: {e}")

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "holdout_stems.json").write_text(
        json.dumps(hold_names, indent=2) + "\n", encoding="utf-8"
    )
    n_fits = len(list(out.glob(f"desi_*_s{args.size}.fits")))
    print(
        f"DONE train_ok≈{ok} fail={fail} holdout_ok={hold_ok} "
        f"fits_s{args.size}={n_fits} → {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
