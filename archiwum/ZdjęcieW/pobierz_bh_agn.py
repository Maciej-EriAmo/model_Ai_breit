"""
Znane AGN / kwazary (optyczny ślad SMBH) → ZdjęcieW/demo/

Uwaga: DESI nie obrazuje horyzontu zdarzeń — widać jądro aktywne / kwazar / host.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pobieranie import save_cutout  # noqa: E402

# (name, ra_deg, dec_deg, kind)
BH_AGN = [
    ("bh_3c273", 187.2779, 2.0524, "quasar"),       # pierwszy zidentyfikowany kwazar
    ("bh_m87", 187.7059, 12.3911, "smbh_host"),     # EHT M87*
    ("bh_cena", 201.3651, -43.0191, "radio_gal"),   # Cen A — może poza footprint
    ("bh_cyga", 299.8682, 40.7339, "radio_gal"),    # Cyg A — często 500
    ("bh_3c279", 194.0465, -5.7893, "blazar"),
    ("bh_cta102", 338.1517, 11.7308, "quasar"),
    ("bh_3c345", 250.7450, 39.8100, "quasar"),
    ("bh_ngc4151", 182.6358, 39.4057, "seyfert"),
    ("bh_mrk421", 166.1138, 38.2088, "blazar"),
    ("bh_3c48", 24.4221, 33.1598, "quasar"),
    ("bh_ton618", 186.3542, 31.5261, "quasar_ultramassive"),
    ("bh_oj287", 133.7036, 20.1085, "blazar_binary_cand"),
    # kontrole: „puste” pola obok (nie AGN)
    ("ctrl_near_3c273", 188.5, 3.0, "field"),
    ("ctrl_near_m87", 186.0, 11.0, "field"),
    ("ctrl_near_4151", 181.0, 38.0, "field"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--no-png", action="store_true", default=True)
    args = ap.parse_args(argv)
    out = HERE / "demo"
    labels = {}
    if (out / "labels.json").is_file():
        labels.update(json.loads((out / "labels.json").read_text(encoding="utf-8")))

    ok, fail = [], []
    for name, ra, dec, kind in BH_AGN:
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
            # do scores: nie mylić z cluster/field treningu — osobna etykieta meta
            labels[fp.stem] = "field" if kind == "field" else "cluster"
            labels[name] = labels[fp.stem]
            labels[f"kind:{fp.stem}"] = kind
            ok.append((name, kind, fp.name))
            print(f"  OK {name} ({kind})")
        except Exception as e:
            fail.append((name, str(e)))
            print(f"  FAIL {name}: {e}")

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "bh_agn_list.json").write_text(
        json.dumps({"ok": ok, "fail": [(n, e[:80]) for n, e in fail]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"DONE ok={len(ok)} fail={len(fail)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
