"""
Paczka cutoutów DESI → ZdjęcieW/demo/ (cluster + field).

  python pobierz_paczke.py
  python pobierz_paczke.py --size 128 --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ten sam katalog co pobieranie.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pobieranie import save_cutout  # noqa: E402

# (name, ra, dec, hint) — footprint Legacy, mieszanka klastrów / pól
PACK: list[tuple[str, float, float, str]] = [
    # klastry / gęste
    ("coma", 194.953, 27.981, "cluster"),
    ("virgo", 187.7, 12.3, "cluster"),
    ("perseus", 49.95, 41.51, "cluster"),
    ("hercules", 241.3, 17.7, "cluster"),
    ("leo", 169.95, 12.58, "cluster"),
    ("hydra", 159.17, -27.53, "cluster"),
    ("centaurus", 192.2, -41.3, "cluster"),
    ("fornax", 54.6, -35.45, "cluster"),
    ("coma_b", 195.5, 28.2, "cluster"),
    ("virgo_b", 186.5, 13.0, "cluster"),
    # pola / losowe
    ("field_a", 150.0, 2.0, "field"),
    ("field_b", 30.0, -5.0, "field"),
    ("field_c", 210.0, 40.0, "field"),
    ("field_d", 5.0, 15.0, "field"),
    ("field_e", 120.0, 20.0, "field"),
    ("field_f", 330.0, 10.0, "field"),
    ("field_g", 60.0, 0.0, "field"),
    ("field_h", 250.0, -10.0, "field"),
    ("field_i", 90.0, 35.0, "field"),
    ("field_j", 180.0, -15.0, "field"),
    ("field_k", 15.0, 45.0, "field"),
    ("field_l", 270.0, 5.0, "field"),
    ("field_m", 135.0, -20.0, "field"),
    ("field_n", 300.0, 25.0, "field"),
    # --- rozszerzenie (większy set; footprint-friendly) ---
    ("coma_c", 194.2, 27.5, "cluster"),
    ("coma_d", 195.8, 28.5, "cluster"),
    ("virgo_c", 188.2, 12.0, "cluster"),
    ("virgo_d", 186.0, 12.8, "cluster"),
    ("hercules_b", 241.8, 17.3, "cluster"),
    ("leo_b", 170.5, 12.2, "cluster"),
    ("fornax_b", 54.0, -35.0, "cluster"),
    ("abell1689", 197.87, -1.34, "cluster"),
    ("abell2744", 3.58, -30.39, "cluster"),
    ("abell370", 39.97, -1.58, "cluster"),
    ("cluster_ngc", 185.0, 14.0, "cluster"),
    ("field_o", 145.0, 5.0, "field"),
    ("field_p", 155.0, 8.0, "field"),
    ("field_q", 200.0, 30.0, "field"),
    ("field_r", 170.0, -8.0, "field"),
    ("field_s", 220.0, 15.0, "field"),
    ("field_t", 140.0, 10.0, "field"),
    ("field_u", 160.0, -5.0, "field"),
    ("field_v", 190.0, 5.0, "field"),
    ("field_w", 175.0, 20.0, "field"),
    ("field_x", 205.0, -5.0, "field"),
    ("field_y", 125.0, 15.0, "field"),
    ("field_z", 230.0, 20.0, "field"),
    ("field_aa", 148.0, -2.0, "field"),
    ("field_ab", 152.0, 4.0, "field"),
    ("field_ac", 158.0, 12.0, "field"),
    ("field_ad", 165.0, 18.0, "field"),
    ("field_ae", 172.0, 8.0, "field"),
    ("field_af", 178.0, -3.0, "field"),
    ("field_ag", 182.0, 22.0, "field"),
    ("field_ah", 188.0, -10.0, "field"),
    ("field_ai", 193.0, 16.0, "field"),
    ("field_aj", 198.0, 3.0, "field"),
    ("field_ak", 203.0, 12.0, "field"),
    ("field_al", 208.0, -7.0, "field"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="0 = cała paczka")
    ap.add_argument("--clusters-only", action="store_true")
    ap.add_argument("--fields-only", action="store_true")
    args = ap.parse_args(argv)

    rows = PACK
    if args.clusters_only:
        rows = [r for r in rows if r[3] == "cluster"]
    if args.fields_only:
        rows = [r for r in rows if r[3] == "field"]
    if args.limit > 0:
        rows = rows[: args.limit]

    ok, fail = 0, 0
    labels: dict[str, str] = {}
    out_dir = HERE / "demo"
    for name, ra, dec, hint in rows:
        try:
            fits_p, _png = save_cutout(
                ra=ra, dec=dec, name=name, size=args.size, out_dir=out_dir
            )
            labels[fits_p.stem] = hint
            labels[name] = hint
            ok += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            fail += 1
    lab_path = out_dir / "labels.json"
    if lab_path.is_file():
        old = json.loads(lab_path.read_text(encoding="utf-8"))
        old.update(labels)
        labels = old
    lab_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"DONE ok={ok} fail={fail} → {out_dir} labels={lab_path.name}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
