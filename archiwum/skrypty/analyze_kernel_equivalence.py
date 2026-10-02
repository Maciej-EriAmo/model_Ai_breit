"""Test równoważności (TOST) jądra Most (Breit–Wigner) i Softmax — bez nowego treningu.

Pytanie: czy Most może zastąpić Softmax? Brak istotnej różnicy ≠ równoważność, więc dla każdej
pary ramion różniących się WYŁĄCZNIE jądrem (ta sama informacja, ta sama procedura) sprawdzamy,
czy 90% przedział ufności różnicy (bootstrap po przebiegach) mieści się w marginesie ±MARGIN
(TOST, α = 0.05 na stronę). Margines ±0.05 hit@1 (5 punktów procentowych) przyjęty przed analizą.

Dane:
  A. układ okresowy, protokół transduktywny (kernel_ablation_runs + kernel_tok_runs, 20 przebiegów):
     bw·0−sm·0, bw·ΔE−sm·ΔE, bw·tok−sm·tok, bw·tok+ΔE−sm·tok+ΔE
  B. trzy dziedziny, ścisłe rozdzielenie (crossdomain_runs, 20 przebiegów/dziedzinę):
     bw·0−sm·0, bw·ΔE−sm·ΔE
Miary: A_dół, B_te_hop, B_te_dd (nowe elementy), B_tr_op, B_tr_dd (znane); warianty comp, comp+mix, comp+rel·PoE.

  python archiwum/skrypty/analyze_kernel_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_LAB = _HERE.parents[1]
for _p in (_LAB, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import math

import torch

import run_mendelejew_comprel_frozen as fz

OUT = _LAB / "doswiadczenie" / "glify"
REPORT = OUT / "MENDELEJEW_KERNEL_EQUIV.md"
MARGIN = 0.05
METRICS = ["A_dół", "B_te_hop", "B_te_dd", "B_tr_op", "B_tr_dd"]
VARIANTS = ["comp", "comp+mix", "comp+rel·PoE"]
N = 20


def load(path, domain_field=False):
    val = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dom = r.get("domain", "periodic-transd") if domain_field else "periodic-transd"
        for v, ranks in r["variants"].items():
            for k in METRICS:
                val[(dom, r["arm"], v, k, r["run"])] = fz.metric(ranks, k)[0]
    return val


def ci90(d, n_boot=10000):
    t = torch.tensor(d, dtype=torch.float64)
    g = torch.Generator().manual_seed(4242)
    b = t[torch.randint(len(d), (n_boot, len(d)), generator=g)].mean(1).sort().values
    return b[int(0.05 * n_boot)].item(), b[int(0.95 * n_boot) - 1].item()


def main():
    val = load(OUT / "kernel_ablation_runs.jsonl")
    val.update(load(OUT / "kernel_tok_runs.jsonl"))
    val.update(load(OUT / "crossdomain_runs.jsonl", domain_field=True))
    groups = [("periodic-transd", p) for p in (("bw·0", "sm·0"), ("bw·ΔE", "sm·ΔE"), ("bw·tok", "sm·tok"), ("bw·tok+ΔE", "sm·tok+ΔE"))]
    groups += [(d, p) for d in ("periodic", "kana", "music") for p in (("bw·0", "sm·0"), ("bw·ΔE", "sm·ΔE"))]
    rows = []
    for dom, (a, b) in groups:
        for v in VARIANTS:
            for k in METRICS:
                d = []
                for r in range(N):
                    x, y = val.get((dom, a, v, k, r)), val.get((dom, b, v, k, r))
                    if x is not None and y is not None and not (math.isnan(x) or math.isnan(y)):
                        d.append(x - y)
                if len(d) < 5:
                    continue
                lo, hi = ci90(d)
                m = sum(d) / len(d)
                if -MARGIN < lo and hi < MARGIN:
                    verdict = "równoważne"
                elif lo >= MARGIN:
                    verdict = "Most lepszy (> margines)"
                elif hi <= -MARGIN:
                    verdict = "Softmax lepszy (> margines)"
                elif lo > 0:
                    verdict = "Most lepszy, niepewne czy > margines"
                elif hi < 0:
                    verdict = "Softmax lepszy, niepewne czy > margines"
                else:
                    verdict = "nierozstrzygnięte"
                rows.append((dom, f"{a} − {b}", v, k, m, lo, hi, verdict))

    def summary(sel):
        c = {}
        for r in sel:
            c[r[7]] = c.get(r[7], 0) + 1
        return c

    L = [
        "# Czy Most może zastąpić Softmax? Test równoważności (TOST)",
        "",
        f"Bez nowego treningu. Pary ramion różniące się **wyłącznie jądrem** (ta sama informacja, parametry, procedura). "
        f"Równoważne = 90% CI różnicy hit@1 (bootstrap po {N} przebiegach) w całości w ±{MARGIN} (5 punktów procentowych; "
        "TOST, α = 0.05 na stronę; margines ustalony przed analizą). Skrypt `archiwum/skrypty/analyze_kernel_equivalence.py`.",
        "",
        "Kategorie: **równoważne** · **nierozstrzygnięte** (CI obejmuje 0 i wychodzi poza margines — za mało danych) · "
        "**X lepszy, niepewne czy > margines** (CI po jednej stronie zera, ale sięga marginesu) · **X lepszy (> margines)**.",
        "",
        "## Podsumowanie",
        "",
        "| dane | porównanie | równoważne | nierozstrz. | Most lepszy* | Softmax lepszy* | razem |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    keys = []
    for r in rows:
        if (r[0], r[1]) not in keys:
            keys.append((r[0], r[1]))
    tot = {"równ": 0, "nier": 0, "most": 0, "sm": 0, "all": 0}
    for dom, comp in keys:
        c = summary([r for r in rows if r[0] == dom and r[1] == comp])
        e = c.get("równoważne", 0)
        n = c.get("nierozstrzygnięte", 0)
        mo = sum(v for k2, v in c.items() if k2.startswith("Most"))
        so = sum(v for k2, v in c.items() if k2.startswith("Softmax"))
        a = sum(c.values())
        for k2, v2 in (("równ", e), ("nier", n), ("most", mo), ("sm", so), ("all", a)):
            tot[k2] += v2
        L.append(f"| {dom} | {comp} | {e} | {n} | {mo} | {so} | {a} |")
    L.append(f"| **razem** | | **{tot['równ']}** | {tot['nier']} | {tot['most']} | {tot['sm']} | {tot['all']} |")
    L += ["", "\\* łącznie z „niepewne czy > margines”.", "", "## Porównania, które NIE są równoważne", "",
          "| dane | porównanie | wariant | miara | Δ | 90% CI | werdykt |", "|---|---|---|---|---:|---|---|"]
    for dom, comp, v, k, m, lo, hi, verdict in rows:
        if verdict != "równoważne":
            L.append(f"| {dom} | {comp} | {v} | {k} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
