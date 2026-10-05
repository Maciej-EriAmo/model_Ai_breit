"""T0 — węższy margines równoważności, bez nowego treningu.

Te same 30 porównań co TOST z 2 października dla τ w tokenach
(bw·tok − sm·tok oraz bw·tok+ΔE − sm·tok+ΔE, protokół transduktywny,
3 warianty × 5 miar). 90% CI bootstrapem jak w analyze_kernel_equivalence.py.

Próg T0, ustalony w JADRO_OGOLNE.md przed tym rachunkiem: ±0.02 hit@1.
Równoważne = cały przedział w ±0.02. Przy ±0.05 te same pary dały 26/30.

  python archiwum/skrypty/analyze_jadro_t0.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_LAB = _HERE.parents[1]
for _p in (_LAB, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import math

import analyze_kernel_equivalence as eq

OUT = eq.OUT
REPORT = OUT / "JADRO_T0.md"
PAIRS = (("bw·tok", "sm·tok"), ("bw·tok+ΔE", "sm·tok+ΔE"))
MARGINS = (0.05, 0.02)


def verdict(lo, hi, margin):
    if -margin < lo and hi < margin:
        return "równoważne"
    if lo >= margin:
        return "Most lepszy (> margines)"
    if hi <= -margin:
        return "Softmax lepszy (> margines)"
    if lo > 0:
        return "Most lepszy, niepewne czy > margines"
    if hi < 0:
        return "Softmax lepszy, niepewne czy > margines"
    return "nierozstrzygnięte"


def bucket(v):
    if v == "równoważne":
        return "w marginesie"
    if v.endswith("(> margines)"):
        return "wychodzi"
    return "nierozstrzygnięte"


def main():
    val = eq.load(OUT / "kernel_tok_runs.jsonl")
    rows = []
    for a, b in PAIRS:
        for v in eq.VARIANTS:
            for k in eq.METRICS:
                d = []
                for r in range(eq.N):
                    x, y = val.get(("periodic-transd", a, v, k, r)), val.get(("periodic-transd", b, v, k, r))
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        d.append(x - y)
                if len(d) < 5:
                    raise SystemExit(f"za mało przebiegów: {a} {v} {k} ({len(d)})")
                lo, hi = eq.ci90(d)
                rows.append((f"{a} − {b}", v, k, sum(d) / len(d), lo, hi, len(d)))

    def counts(margin):
        c = {"w marginesie": 0, "wychodzi": 0, "nierozstrzygnięte": 0}
        for *_, lo, hi, _n in rows:
            c[bucket(verdict(lo, hi, margin))] += 1
        return c

    c05, c02 = counts(0.05), counts(0.02)
    if c05["w marginesie"] != 26 or sum(c05.values()) != 30:
        raise SystemExit(f"kontrola ±5 nie odtwarza 26/30: {c05}")

    L = [
        "# T0 — margines ±2 punkty na porównaniach z tracerem w tokenach",
        "",
        "Bez nowego treningu. Te same 30 porównań co TOST z 2 października "
        "(`bw·tok − sm·tok` i `bw·tok+ΔE − sm·tok+ΔE`, układ okresowy, protokół transduktywny, "
        "3 warianty × 5 miar, 20 przebiegów). 90% CI różnicy hit@1, bootstrap 10 000, ziarno 4242, "
        "jak w `archiwum/skrypty/analyze_kernel_equivalence.py`.",
        "",
        "Próg **±2 punkty** (±0.02) jest progim T0 z `JADRO_OGOLNE.md`, ustalonym przed tym rachunkiem. "
        "Równoważne = cały przedział mieści się w progu. Kontrola: ten sam CI przy ±5 punktach daje 26/30, "
        "zgodnie z `MENDELEJEW_KERNEL_EQUIV.md`.",
        "",
        "| próg | w marginesie | wychodzi poza próg | nierozstrzygnięte | razem |",
        "|---|---:|---:|---:|---:|",
        f"| ±5 punktów (kontrola) | {c05['w marginesie']} | {c05['wychodzi']} | {c05['nierozstrzygnięte']} | 30 |",
        f"| ±2 punkty (T0) | {c02['w marginesie']} | {c02['wychodzi']} | {c02['nierozstrzygnięte']} | 30 |",
        "",
        "„Wychodzi” = cały przedział leży poza progiem (różnica większa niż próg). "
        "„Nierozstrzygnięte” = przedział nie mieści się w progu i nie leży cały poza nim.",
        "",
        "## Trzydzieści porównań",
        "",
        "| porównanie | wariant | miara | Δ | 90% CI | ±5 | ±2 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for comp, v, k, m, lo, hi, _n in rows:
        L.append(
            f"| {comp} | {v} | {k} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{verdict(lo, hi, 0.05)} | {verdict(lo, hi, 0.02)} |"
        )
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"±5 {c05}")
    print(f"±2 {c02}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
