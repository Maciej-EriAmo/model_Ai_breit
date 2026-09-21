"""Wygeneruj pełną tablicę Mendelejewa (Z=1..118) → CSV/JSON pod import_struktura.

Konfiguracje: Aufbau + znane wyjątki (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, …).
Nazwy PL dla 1–118 (skrócone tam, gdzie brak ugruntowanej formy).

  python make_periodic_full.py
  python make_periodic_full.py --out struktura_atomow_full
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (symbol, name_pl, group, period) — group=0 → lantanowce/aktynowce (III)
# group IUPAC 1–18; f-blok: 3
ELEMENTS: list[tuple[str, str, int, int]] = [
    ("H", "Wodór", 1, 1), ("He", "Hel", 18, 1),
    ("Li", "Lit", 1, 2), ("Be", "Beryl", 2, 2), ("B", "Bor", 13, 2),
    ("C", "Węgiel", 14, 2), ("N", "Azot", 15, 2), ("O", "Tlen", 16, 2),
    ("F", "Fluor", 17, 2), ("Ne", "Neon", 18, 2),
    ("Na", "Sód", 1, 3), ("Mg", "Magnez", 2, 3), ("Al", "Glin", 13, 3),
    ("Si", "Krzem", 14, 3), ("P", "Fosfor", 15, 3), ("S", "Siarka", 16, 3),
    ("Cl", "Chlor", 17, 3), ("Ar", "Argon", 18, 3),
    ("K", "Potas", 1, 4), ("Ca", "Wapń", 2, 4),
    ("Sc", "Skand", 3, 4), ("Ti", "Tytan", 4, 4), ("V", "Wanad", 5, 4),
    ("Cr", "Chrom", 6, 4), ("Mn", "Mangan", 7, 4), ("Fe", "Żelazo", 8, 4),
    ("Co", "Kobalt", 9, 4), ("Ni", "Nikiel", 10, 4), ("Cu", "Miedź", 11, 4),
    ("Zn", "Cynk", 12, 4), ("Ga", "Gal", 13, 4), ("Ge", "German", 14, 4),
    ("As", "Arsen", 15, 4), ("Se", "Selen", 16, 4), ("Br", "Brom", 17, 4),
    ("Kr", "Krypton", 18, 4),
    ("Rb", "Rubid", 1, 5), ("Sr", "Stront", 2, 5),
    ("Y", "Itr", 3, 5), ("Zr", "Cyrkon", 4, 5), ("Nb", "Niob", 5, 5),
    ("Mo", "Molibden", 6, 5), ("Tc", "Technet", 7, 5), ("Ru", "Ruten", 8, 5),
    ("Rh", "Rod", 9, 5), ("Pd", "Pallad", 10, 5), ("Ag", "Srebro", 11, 5),
    ("Cd", "Kadm", 12, 5), ("In", "Ind", 13, 5), ("Sn", "Cyna", 14, 5),
    ("Sb", "Antymon", 15, 5), ("Te", "Tellur", 16, 5), ("I", "Jod", 17, 5),
    ("Xe", "Ksenon", 18, 5),
    ("Cs", "Cez", 1, 6), ("Ba", "Bar", 2, 6),
    ("La", "Lantan", 3, 6),
    ("Ce", "Cer", 3, 6), ("Pr", "Prazeodym", 3, 6), ("Nd", "Neodym", 3, 6),
    ("Pm", "Promet", 3, 6), ("Sm", "Samar", 3, 6), ("Eu", "Europ", 3, 6),
    ("Gd", "Gadolin", 3, 6), ("Tb", "Terb", 3, 6), ("Dy", "Dysproz", 3, 6),
    ("Ho", "Holm", 3, 6), ("Er", "Erb", 3, 6), ("Tm", "Tul", 3, 6),
    ("Yb", "Iterb", 3, 6), ("Lu", "Lutet", 3, 6),
    ("Hf", "Hafn", 4, 6), ("Ta", "Tantal", 5, 6), ("W", "Wolfram", 6, 6),
    ("Re", "Ren", 7, 6), ("Os", "Osm", 8, 6), ("Ir", "Iryd", 9, 6),
    ("Pt", "Platyna", 10, 6), ("Au", "Złoto", 11, 6), ("Hg", "Rtęć", 12, 6),
    ("Tl", "Tal", 13, 6), ("Pb", "Ołów", 14, 6), ("Bi", "Bizmut", 15, 6),
    ("Po", "Polon", 16, 6), ("At", "Astat", 17, 6), ("Rn", "Radon", 18, 6),
    ("Fr", "Frans", 1, 7), ("Ra", "Rad", 2, 7),
    ("Ac", "Aktyn", 3, 7),
    ("Th", "Tor", 3, 7), ("Pa", "Protaktyn", 3, 7), ("U", "Uran", 3, 7),
    ("Np", "Neptun", 3, 7), ("Pu", "Pluton", 3, 7), ("Am", "Ameryk", 3, 7),
    ("Cm", "Kiur", 3, 7), ("Bk", "Berkel", 3, 7), ("Cf", "Kaliforn", 3, 7),
    ("Es", "Einstein", 3, 7), ("Fm", "Ferm", 3, 7), ("Md", "Mendelew", 3, 7),
    ("No", "Nobel", 3, 7), ("Lr", "Lorens", 3, 7),
    ("Rf", "Rutherford", 4, 7), ("Db", "Dubn", 5, 7), ("Sg", "Seaborg", 6, 7),
    ("Bh", "Bohr", 7, 7), ("Hs", "Has", 8, 7), ("Mt", "Meitner", 9, 7),
    ("Ds", "Darmstadt", 10, 7), ("Rg", "Roentgen", 11, 7), ("Cn", "Kopernik", 12, 7),
    ("Nh", "Nihon", 13, 7), ("Fl", "Flerow", 14, 7), ("Mc", "Moskow", 15, 7),
    ("Lv", "Liwermor", 16, 7), ("Ts", "Tenness", 17, 7), ("Og", "Oganeson", 18, 7),
]

assert len(ELEMENTS) == 118

# Madelung order: (n, ℓ) filling sequence
AUFBAU = [
    (1, 0),
    (2, 0), (2, 1),
    (3, 0), (3, 1),
    (4, 0), (3, 2), (4, 1),
    (5, 0), (4, 2), (5, 1),
    (6, 0), (4, 3), (5, 2), (6, 1),
    (7, 0), (5, 3), (6, 2), (7, 1),
    (8, 0), (6, 3), (7, 2),
]
ELL = "spdf"
CAP = {0: 2, 1: 6, 2: 10, 3: 14}

# Wyjątki vs czysty Aufbau: Z → lista (n,ℓ,count) nadpisująca całość valence-ish
# Używamy pełnych wyjątków jako gotowych stringów konfiguracji.
EXCEPTIONS: dict[int, str] = {
    24: "[Ar] 4s1 3d5",   # Cr
    29: "[Ar] 4s1 3d10",  # Cu
    41: "[Kr] 5s1 4d4",   # Nb
    42: "[Kr] 5s1 4d5",   # Mo
    44: "[Kr] 5s1 4d7",   # Ru
    45: "[Kr] 5s1 4d8",   # Rh
    46: "[Kr] 4d10",      # Pd
    47: "[Kr] 5s1 4d10",  # Ag
    57: "[Xe] 6s2 5d1",   # La (często tak w tabelach podręcznikowych)
    58: "[Xe] 6s2 4f1 5d1",  # Ce
    64: "[Xe] 6s2 4f7 5d1",  # Gd
    78: "[Xe] 6s1 4f14 5d9",  # Pt
    79: "[Xe] 6s1 4f14 5d10",  # Au
    89: "[Rn] 7s2 6d1",   # Ac
    90: "[Rn] 7s2 6d2",   # Th
    91: "[Rn] 7s2 5f2 6d1",  # Pa
    92: "[Rn] 7s2 5f3 6d1",  # U
    93: "[Rn] 7s2 5f4 6d1",  # Np
    96: "[Rn] 7s2 5f7 6d1",  # Cm
}


def aufbau_terms(Z: int) -> list[tuple[int, int, int]]:
    left = Z
    terms: list[tuple[int, int, int]] = []
    for n, ell in AUFBAU:
        if left <= 0:
            break
        take = min(CAP[ell], left)
        terms.append((n, ell, take))
        left -= take
    if left > 0:
        raise RuntimeError(f"Aufbau nie zapełnił Z={Z}, left={left}")
    return terms


def terms_to_cfg(terms: list[tuple[int, int, int]]) -> str:
    # kompakt: z rdzeniem gazu szlachetnego gdy pasuje
    noble_Z = {2: "He", 10: "Ne", 18: "Ar", 36: "Kr", 54: "Xe", 86: "Rn"}
    filled = 0
    core = None
    for n, ell, cnt in terms:
        filled += cnt
        if filled in noble_Z and ell == (0 if filled == 2 else 1):
            # po zamknięciu p (lub 1s dla He)
            if filled == 2 or (ell == 1 and cnt == CAP[1]):
                core = (filled, noble_Z[filled])
    parts = [f"{n}{ELL[ell]}{cnt}" for n, ell, cnt in terms]
    if core is None:
        return " ".join(parts)
    # obetnij termy wchodzące w rdzeń
    z_core, name = core
    consumed = 0
    i = 0
    for n, ell, cnt in terms:
        consumed += cnt
        i += 1
        if consumed >= z_core:
            break
    rest = terms[i:]
    if not rest:
        return f"[{name}]" if name != "He" or Z_of_terms(terms) == 2 else "1s2"
    rest_s = " ".join(f"{n}{ELL[ell]}{cnt}" for n, ell, cnt in rest)
    return f"[{name}] {rest_s}"


def Z_of_terms(terms: list[tuple[int, int, int]]) -> int:
    return sum(c for _, _, c in terms)


def electrons_per_shell(terms: list[tuple[int, int, int]]) -> str:
    by_n: dict[int, int] = {}
    for n, _ell, cnt in terms:
        by_n[n] = by_n.get(n, 0) + cnt
    return ", ".join(str(by_n[n]) for n in sorted(by_n))


def build_rows() -> list[dict]:
    rows = []
    for Z, (sym, name, group, period) in enumerate(ELEMENTS, start=1):
        if Z in EXCEPTIONS:
            cfg = EXCEPTIONS[Z]
            # shells: policz z expand przez import — tu uproszczenie z Aufbau Z
            terms = aufbau_terms(Z)  # approx shells
        else:
            terms = aufbau_terms(Z)
            cfg = terms_to_cfg(terms)
        rows.append({
            "atom_number": Z,
            "symbol": sym,
            "name": name,
            "group": group,
            "period": period,
            "electron_configuration": cfg,
            "electrons_per_shell": electrons_per_shell(terms),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "struktura_atomow_full")
    args = ap.parse_args()
    rows = build_rows()
    assert len(rows) == 118
    csv_path = args.out.with_suffix(".csv") if args.out.suffix == "" else args.out
    if csv_path.suffix != ".csv":
        csv_path = Path(str(args.out) + ".csv")
    json_path = csv_path.with_suffix(".json")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "atom_number", "symbol", "name", "group", "period",
                "electron_configuration", "electrons_per_shell",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK {len(rows)} → {csv_path} / {json_path}")
    print("sample", rows[0]["symbol"], rows[0]["electron_configuration"])
    print("sample", rows[25]["symbol"], rows[25]["electron_configuration"])
    print("sample", rows[-1]["symbol"], rows[-1]["electron_configuration"])


if __name__ == "__main__":
    main()
