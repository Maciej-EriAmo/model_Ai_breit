"""Mendelejew REL — nauka relacyjna: osobne głowice relacji nad wspólnym enkoderem.

Cechy jak ZC (tylko Z + konfiguracja). Protokół jak SEMI (split 70/30, Sc/Ga/Ge/Tc
zawsze w teście, wszystkie pary relacji WYŁĄCZNIE spośród pierwiastków treningowych).

Relacje:
  grupa  — ta sama grupa (pionowa; K par round-robin po grupach jak SEMI)
  Z      — sąsiad w Z (|ΔZ| = 1) — darmowa, wynika z samego Z
  okres  — ten sam okres (pozioma; dodatkowe etykiety)
Reżimy:
  single  — tylko grupa, jedna przestrzeń (≡ SEMI sup)
  mix_Z   — grupa + Z w JEDNEJ przestrzeni (relacje zlane w jedno podobieństwo)
  rel_Z   — grupa + Z, osobne głowice (Linear d→d na relację, cos w jej przestrzeni)
  mix_ZP  — grupa + Z + okres w jednej przestrzeni
  rel_ZP  — grupa + Z + okres, osobne głowice
Ocena w przestrzeni głowicy „grupa” (dla single/mix: wspólnej):
  hit@1 ¬f, hit@1, NMI grupy; dla rel_ZP dodatkowo hit@1 okresu w głowicy „okres”.

  python archiwum/skrypty/run_mendelejew_rel.py [--smoke]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_LAB = _HERE.parents[1]
for _p in (_LAB, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_mendelejew_ml3_zc as zc  # ustawia ml3.D_X / T_SEAL dla ZC
from run_mendelejew_ml3 import ARMS, encode, fmt, make_encoder, ms, noisy
from run_mendelejew_semi import GAPS, _nce, analogs, evaluate, labeled_pairs, random_hit, split
from config import CANON
from import_struktura_atomow import load_rows

OUT_DIR = _LAB / "doswiadczenie" / "glify"
BUDGETS = [18, 54, "all"]
REGIMES = {
    "single": (["grupa"], False),
    "mix_Z": (["grupa", "Z"], False),
    "rel_Z": (["grupa", "Z"], True),
    "mix_ZP": (["grupa", "Z", "okres"], False),
    "rel_ZP": (["grupa", "Z", "okres"], True),
}


def relation_pairs(meta, tr):
    byZ = {meta[i][2]: i for i in tr}
    zpairs = [(byZ[z], byZ[z + 1]) for z in sorted(byZ) if z + 1 in byZ]
    per = [(a, b) for i, a in enumerate(tr) for b in tr[i + 1:] if meta[a][1] == meta[b][1]]
    return {"Z": zpairs, "okres": per}


def train(arm, regime, content, tracer, rels, seed, steps):
    names, separate = REGIMES[regime]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    heads = nn.ModuleDict({r: nn.Linear(CANON.d_model, CANON.d_model) for r in (names if separate else ["wspólna"])})
    opt = torch.optim.Adam(list(enc.parameters()) + list(heads.parameters()), lr=CANON.lr)
    for _ in range(steps):
        enc.train()
        h = encode(enc, arm, noisy(content, gen), tracer)
        loss = torch.zeros(())
        for r in names:
            if not rels[r]:
                continue
            head = heads[r] if separate else heads["wspólna"]
            loss = loss + _nce(F.normalize(head(h), dim=-1), rels[r], gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        h = encode(enc, arm, content, tracer)
        return {r: F.normalize(heads[r](h), dim=-1) for r in heads}


def period_hit(h, meta, te):
    s = (h @ h.T).fill_diagonal_(-2.0)
    nn1 = s.argmax(-1)
    return sum(meta[q][1] == meta[int(nn1[q])][1] for q in te) / len(te)


def write_report(res, meta_run, md, js):
    seeds = meta_run["seeds"]
    L = [
        "# Mendelejew REL — nauka relacyjna (osobne głowice relacji)",
        "",
        "Cechy jak ZC (tylko Z + konfiguracja). Pary wszystkich relacji tylko spośród pierwiastków treningowych; "
        "ocena na nigdy nieoznaczonych (Sc/Ga/Ge/Tc zawsze w teście). "
        f"Seedy={seeds} · kroki={meta_run['steps']} · czas={meta_run['elapsed']:.0f}s · "
        "skrypt `archiwum/skrypty/run_mendelejew_rel.py`",
        "",
        f"Pary w treningu (średnio): Z-sąsiedzi={meta_run['n_Z']:.0f}, okres={meta_run['n_okres']:.0f}, "
        f"grupa all={meta_run['n_all']:.0f}. Losowe hit@1 ¬f = {meta_run['random_nonf']:.3f}.",
        "",
        "`mix_*` = relacje zlane w jedno podobieństwo; `rel_*` = osobna głowica (przestrzeń) na relację.",
        "",
    ]
    for metric, title in (("hit1_nonf", "Grupa: hit@1 bez bloku f (główna miara)"),
                          ("nmi_group", "Grupa: NMI (k-means, wszystkie 118)")):
        L += [f"## {title}", "", "| K grupa | reżim | " + " | ".join(ARMS) + " |", "|---|---|" + "---:|" * len(ARMS)]
        for K in BUDGETS:
            for rg in REGIMES:
                r = res["runs"][f"{K}|{rg}"]
                L.append(f"| {K} | {rg} | " + " | ".join(fmt([x[metric] for x in r[a]]) for a in ARMS) + " |")
        L.append("")
    L += ["## Efekt relacji względem `single` (Δ hit@1 ¬f, wygrane seedy) — Most i średnia po armiach", "",
          "| K | reżim | Most Δ (wygrane) | średnia Δ po armiach |", "|---|---|---:|---:|"]
    for K in BUDGETS:
        base = res["runs"][f"{K}|single"]
        for rg in REGIMES:
            if rg == "single":
                continue
            r = res["runs"][f"{K}|{rg}"]
            m = [x["hit1_nonf"] for x in r["most"]]
            mb = [x["hit1_nonf"] for x in base["most"]]
            allv = [x["hit1_nonf"] for a in ARMS for x in r[a]]
            allb = [x["hit1_nonf"] for a in ARMS for x in base[a]]
            L.append(f"| {K} | {rg} | {ms(m)[0] - ms(mb)[0]:+.3f} ({sum(x > y + 1e-9 for x, y in zip(m, mb))}/{len(seeds)}) | "
                     f"{ms(allv)[0] - ms(allb)[0]:+.3f} |")
    L += ["", "## Most vs reszta (hit@1 ¬f, wygrane seedy)", "", "| K | reżim | vs softmax | vs zero | vs mlp |",
          "|---|---|---:|---:|---:|"]
    for K in BUDGETS:
        for rg in REGIMES:
            r = res["runs"][f"{K}|{rg}"]
            m = [x["hit1_nonf"] for x in r["most"]]
            L.append(f"| {K} | {rg} | " + " | ".join(
                f"{sum(x > y + 1e-9 for x, y in zip(m, [z['hit1_nonf'] for z in r[o]]))}/{len(seeds)}"
                for o in ("softmax", "zero", "mlp")) + " |")
    L += ["", "## Kontrola: głowica „okres” w rel_ZP (hit@1 okresu na teście)", "",
          "| K | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for K in BUDGETS:
        r = res["runs"][f"{K}|rel_ZP"]
        L.append(f"| {K} | " + " | ".join(fmt([x["period_hit"] for x in r[a]]) for a in ARMS) + " |")
    L += ["", "## Analogi luk w głowicy „grupa” (Most, seed 0, K=54)", "",
          "Oczekiwane: Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re", "",
          "| reżim | Sc | Ga | Ge | Tc |", "|---|---|---|---|---|"]
    for rg in REGIMES:
        an = res["analogs"].get(rg)
        if an:
            L.append(f"| {rg} | " + " | ".join(", ".join(an[g]) for g in GAPS) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta_run, "results": res}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = 60 if args.smoke else 600
    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    t0 = time.time()
    res = {"runs": {}, "analogs": {}}
    nz, npr, nall, rnd = [], [], [], []
    for seed in seeds:
        tr, te = split(meta, seed)
        rels = relation_pairs(meta, tr)
        nz.append(len(rels["Z"]))
        npr.append(len(rels["okres"]))
        nall.append(len(labeled_pairs(meta, tr, "all", seed)))
        rnd.append(random_hit(meta, te, True))
        for K in BUDGETS:
            rels["grupa"] = labeled_pairs(meta, tr, K, seed)
            for rg in REGIMES:
                key = f"{K}|{rg}"
                res["runs"].setdefault(key, {a: [] for a in ARMS})
                for a in ARMS:
                    hs = train(a, rg, content, tracer, rels, seed, steps)
                    hg = hs.get("grupa", hs.get("wspólna"))
                    ev = evaluate(hg, meta, te, seed)
                    if "okres" in hs:
                        ev["period_hit"] = period_hit(hs["okres"], meta, te)
                    res["runs"][key][a].append(ev)
                    if seed == seeds[0] and K == 54 and a == "most":
                        res["analogs"][rg] = analogs(hg, meta)
                print(f"  seed={seed} K={K:>3} {rg:7s} ¬f: " + " ".join(
                    f"{a}={res['runs'][key][a][-1]['hit1_nonf']:.2f}" for a in ARMS)
                    + f" ({time.time() - t0:.0f}s)", flush=True)
    meta_run = {"seeds": seeds, "steps": steps, "elapsed": time.time() - t0, "smoke": args.smoke,
                "n_Z": sum(nz) / len(nz), "n_okres": sum(npr) / len(npr), "n_all": sum(nall) / len(nall),
                "random_nonf": sum(rnd) / len(rnd)}
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_REL{tag}.md"
    write_report(res, meta_run, md, OUT_DIR / f"mendelejew_rel{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
