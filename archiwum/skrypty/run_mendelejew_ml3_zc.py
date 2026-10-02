"""Mendelejew ML3 ZC — pełna tablica BEZ okresu/grupy w cechach: tylko Z + konfiguracja.

Model nie widzi okresu, grupy ani bloku — nigdzie (treść, tracer, krawędzie).
Etykiety okres/grupa/blok służą wyłącznie do oceny i jako nagroda środowiska w RL.

Cechy pierwiastka:
  treść x (24) — obsadzenie podpowłok w kolejności Madelunga (1s … 7p, 8s 5g 6f 7d 8p),
                 ułamek pojemności, wektor znormalizowany
  tracer τ (6) — [Z/118, obsadzenie zewnętrznej s, p, d, f (najwyższe n danego typu),
                 liczba obsadzonych podpowłok/24]
  T-seal Mostu wyłączony: w tym wariancie nie ma kanału T (τ3 = zewnętrzna d).

Krawędzie samonadzoru: k=3 najbliższych sąsiadów w surowej przestrzeni [x ⊕ τ] —
graf zbudowany bez etykiet (zamiast krawędzi grupa/okres z FULL118).

Część A — SL (okres/grupa/blok), UL (k-means NMI), RL (nagroda: wybrany pierwiastek
          z tej samej grupy co zapytanie).
Część B — odkrycie: Sc, Ga, Ge, Tc usunięte. Z luki jest znane (ciąg Z),
          szukamy jej KONFIGURACJI: kandydaci = konfiguracja prawdziwa + warianty
          z 1–2 elektronami przesuniętymi w oknie zewnętrznych podpowłok.
          Wygrany kandydat → analogi (kNN wśród znanych) → przewidziana grupa/blok.
          Eksploracja: Z=119, 120.

  python archiwum/skrypty/run_mendelejew_ml3_zc.py [--smoke]
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
import re
import time
from collections import Counter

import torch
import torch.nn.functional as F

import run_mendelejew_ml3 as ml3

SUB = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s",
       "4f", "5d", "6p", "7s", "5f", "6d", "7p", "8s", "5g", "6f", "7d", "8p"]
ml3.D_X = len(SUB)
ml3.T_SEAL = False

from run_mendelejew_ml3 import ARMS, fmt, ms, wins  # noqa: E402
from run_mendelejew_ml3_full import BLOCKS, KNN, auc, probe_scores, rank_of, train_ssl  # noqa: E402
from export_mendelejew_full import _block  # noqa: E402
from import_struktura_atomow import load_rows  # noqa: E402

ROOT = _LAB
SRC = ROOT / "archiwum" / "dane" / "struktura_atomow_full.csv"
OUT_DIR = ROOT / "doswiadczenie" / "glify"
HELD_OUT = {"Sc": "eka-bor", "Ga": "eka-glin", "Ge": "eka-krzem", "Tc": "eka-mangan"}
CAP = {"s": 2, "p": 6, "d": 10, "f": 14, "g": 18}
NOBLE = {"He": "1s2", "Ne": "[He] 2s2 2p6", "Ar": "[Ne] 3s2 3p6", "Kr": "[Ar] 3d10 4s2 4p6",
         "Xe": "[Kr] 4d10 5s2 5p6", "Rn": "[Xe] 4f14 5d10 6s2 6p6", "Og": "[Rn] 5f14 6d10 7s2 7p6"}
K_SSL = 3


# ---------------------------------------------------------------- konfiguracja


def parse_cfg(cfg: str) -> list[int]:
    occ = [0] * len(SUB)
    cfg = cfg.strip()
    m = re.match(r"\[(\w+)\]\s*(.*)", cfg)
    if m:
        occ = parse_cfg(NOBLE[m.group(1)])
        cfg = m.group(2)
    for n, l, c in re.findall(r"(\d)([spdfg])(\d+)", cfg):
        occ[SUB.index(n + l)] = int(c)
    return occ


def features(Z: int, occ: list[int]):
    cap = torch.tensor([CAP[s[-1]] for s in SUB], dtype=torch.float)
    o = torch.tensor(occ, dtype=torch.float)
    x = o / cap
    x = x / x.norm().clamp_min(1e-6)
    tau = torch.zeros(6)
    tau[0] = Z / 118.0
    for j, typ in enumerate("spdf"):
        idx = [i for i, s in enumerate(SUB) if s[-1] == typ and occ[i] > 0]
        if idx:
            top = max(idx, key=lambda i: int(SUB[i][:-1]))
            tau[1 + j] = occ[top] / CAP[typ]
    tau[5] = sum(1 for c in occ if c > 0) / len(SUB)
    return x, tau


def knn_partners(content, tracer, k=K_SSL):
    h = F.normalize(torch.cat([content, tracer], -1), dim=-1)
    s = (h @ h.T).fill_diagonal_(-2.0)
    nn_idx = s.topk(k, dim=-1).indices
    partners: dict[int, set[int]] = {}
    for i, row in enumerate(nn_idx.tolist()):
        for j in row:
            partners.setdefault(i, set()).add(j)
            partners.setdefault(j, set()).add(i)
    return partners


def build(rows):
    occs = [parse_cfg(r["electron_configuration"]) for r in rows]
    bad = [(r["symbol"], sum(o), int(r["atom_number"])) for r, o in zip(rows, occs)
           if sum(o) != int(r["atom_number"])]
    if bad:
        raise ValueError(f"liczba elektronów ≠ Z: {bad}")
    feats = [features(int(r["atom_number"]), o) for r, o in zip(rows, occs)]
    content = torch.stack([f[0] for f in feats])
    tracer = torch.stack([f[1] for f in feats])
    meta = [(int(r["group"]), int(r["period"]), int(r["atom_number"]), r["symbol"]) for r in rows]
    labels = {
        "period": torch.tensor([p - 1 for _, p, _, _ in meta]),
        "group": torch.tensor([g - 1 for g, _, _, _ in meta]),
        "block": torch.tensor([BLOCKS.index(_block(g, p, Z)) for g, p, Z, _ in meta]),
    }
    return content, tracer, labels, meta, occs


def same_group_partners(meta):
    by_g: dict[int, list[int]] = {}
    for i, (g, _, _, _) in enumerate(meta):
        by_g.setdefault(g, []).append(i)
    return {i: set(by_g[g]) - {i} for i, (g, _, _, _) in enumerate(meta) if len(by_g[g]) > 1}


def pair_purity(partners, meta):
    pairs = {(min(a, b), max(a, b)) for a, ps in partners.items() for b in ps}
    sg = sum(meta[a][0] == meta[b][0] for a, b in pairs) / len(pairs)
    sp = sum(meta[a][1] == meta[b][1] for a, b in pairs) / len(pairs)
    adj = sum(abs(meta[a][2] - meta[b][2]) == 1 for a, b in pairs) / len(pairs)
    return {"n_pairs": len(pairs), "same_group": sg, "same_period": sp, "z_adjacent": adj}


# ---------------------------------------------------------------- kandydaci konfiguracji


def cfg_str(occ):
    return " ".join(f"{SUB[i]}{c}" for i, c in enumerate(occ) if c > 0)


def short_cfg(occ, ref):
    """Tylko podpowłoki różniące się od / najbliżej końca — do czytelnego raportu."""
    last = max(i for i, c in enumerate(occ) if c > 0)
    lo = max(0, min(last, max(i for i, c in enumerate(ref) if c > 0)) - 4)
    return " ".join(f"{SUB[i]}{occ[i]}" for i in range(lo, len(SUB)) if occ[i] > 0 or ref[i] > 0)


def variants(occ, window_lo, window_hi, depth=2):
    def moves(o):
        for i in range(window_lo, window_hi + 1):
            if o[i] == 0:
                continue
            for j in range(window_lo, window_hi + 1):
                if j != i and o[j] < CAP[SUB[j][-1]]:
                    n = list(o)
                    n[i] -= 1
                    n[j] += 1
                    yield tuple(n)

    seen = {tuple(occ)}
    frontier = [tuple(occ)]
    for _ in range(depth):
        nxt = []
        for o in frontier:
            for n in moves(o):
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    return [list(o) for o in seen]


def aufbau_cost(occ):
    """Reguła Madelunga (bez ML): suma indeksów kolejności zapełniania — mniej = bardziej Aufbau."""
    return sum(i * c for i, c in enumerate(occ))


# ---------------------------------------------------------------- część B


def run_discovery(rows_all, seeds, steps):
    rows = [r for r in rows_all if r["symbol"] not in HELD_OUT]
    content, tracer, _, meta, _ = build(rows)
    partners = knn_partners(content, tracer)
    gaps = {}
    for r in rows_all:
        if r["symbol"] in HELD_OUT:
            occ = parse_cfg(r["electron_configuration"])
            last = max(i for i, c in enumerate(occ) if c > 0)
            cands = variants(occ, max(0, last - 4), min(len(SUB) - 1, last + 2))
            cands.sort(key=lambda o: (o != occ, cfg_str(o)))
            gaps[r["symbol"]] = {"Z": int(r["atom_number"]), "true": occ, "cands": cands,
                                 "group": int(r["group"]), "period": int(r["period"])}

    og = parse_cfg("[Og]")
    extra = [SUB.index(s) for s in ("8s", "5g", "6f", "7d", "8p")]
    frontier = {}
    for Z, ne in ((119, 1), (120, 2)):
        cands = set()
        for i in extra:
            for j in extra:
                if ne == 1 and j != i:
                    continue
                o = list(og)
                o[i] += 1
                if ne == 2:
                    o[j] += 1
                if all(o[k] <= CAP[SUB[k][-1]] for k in range(len(SUB))):
                    cands.add(tuple(o))
        frontier[Z] = sorted((list(c) for c in cands), key=cfg_str)

    def probes(Z, cands):
        f = [features(Z, o) for o in cands]
        return torch.stack([a for a, _ in f]), torch.stack([b for _, b in f])

    def knn_info(knn):
        groups = Counter(meta[i][0] for i in knn)
        blocks = Counter(_block(meta[i][0], meta[i][1], meta[i][2]) for i in knn)
        return (", ".join(meta[i][3] for i in knn), groups.most_common(1)[0][0], blocks.most_common(1)[0][0])

    res = {"gaps": {}, "frontier": {}, "pair_purity": pair_purity(partners, meta)}
    methods = ["reguła Aufbau (bez ML)", "surowe cechy kNN"] + list(ARMS)
    for sym, gp in gaps.items():
        res["gaps"][sym] = {"Z": gp["Z"], "n_cands": len(gp["cands"]), "true_cfg": short_cfg(gp["true"], gp["true"]),
                            "group": gp["group"], "methods": {m: [] for m in methods}}
        costs = [-aufbau_cost(o) for o in gp["cands"]]
        best = max(range(len(costs)), key=lambda i: costs[i])
        res["gaps"][sym]["methods"]["reguła Aufbau (bez ML)"].append(
            {"rank_true": rank_of(costs, [0])[0], "best": short_cfg(gp["cands"][best], gp["true"])})
        pc, pt = probes(gp["Z"], gp["cands"])
        raw = probe_scores(None, None, content, tracer, pc, pt)
        cs = [c for _, c, _ in raw]
        best = max(range(len(cs)), key=lambda i: cs[i])
        nn, g_, b_ = knn_info(raw[0][2])
        res["gaps"][sym]["methods"]["surowe cechy kNN"].append(
            {"rank_true": rank_of(cs, [0])[0], "best": short_cfg(gp["cands"][best], gp["true"]),
             "knn_true": nn, "group_pred": g_, "block_pred": b_})

    for arm in ARMS:
        for seed in seeds:
            enc = train_ssl(arm, content, tracer, partners, seed, steps)
            line = []
            for sym, gp in gaps.items():
                pc, pt = probes(gp["Z"], gp["cands"])
                sc = probe_scores(enc, arm, content, tracer, pc, pt)
                cs = [c for _, c, _ in sc]
                best = max(range(len(cs)), key=lambda i: cs[i])
                nn, g_, b_ = knn_info(sc[0][2])
                res["gaps"][sym]["methods"][arm].append(
                    {"rank_true": rank_of(cs, [0])[0], "best": short_cfg(gp["cands"][best], gp["true"]),
                     "knn_true": nn, "group_pred": g_, "block_pred": b_})
                line.append(f"{sym}:{rank_of(cs, [0])[0]}/{len(cs)}")
            if seed == seeds[0]:
                for Z, cands in frontier.items():
                    pc, pt = probes(Z, cands)
                    sc = probe_scores(enc, arm, content, tracer, pc, pt)
                    order = sorted(range(len(cands)), key=lambda i: -sc[i][1])
                    res["frontier"].setdefault(str(Z), {})[arm] = [
                        (" ".join(f"+{SUB[k]}{cands[i][k] - og[k]}" for k in range(len(SUB)) if cands[i][k] > og[k]), round(sc[i][1], 3), knn_info(sc[i][2])[0]) for i in order[:5]]
            print(f"  [B] {arm:8s} seed={seed} rank prawdziwej konfiguracji: " + " ".join(line), flush=True)
    return res


# ---------------------------------------------------------------- raport


def write_report(resA, resB, meta, md, js):
    seeds = meta["seeds"]
    pp = resB["pair_purity"]
    L = [
        "# Mendelejew ML3 ZC — pełna tablica, cechy tylko z Z i konfiguracji elektronowej",
        "",
        "Model **nie widzi** okresu, grupy ani bloku (ani w treści, ani w tracerze, ani w krawędziach). "
        "Etykiety służą tylko do oceny i jako nagroda środowiska w RL.",
        "",
        f"Treść x = obsadzenie {len(SUB)} podpowłok (kolejność Madelunga) · tracer τ = [Z, zewn. s/p/d/f, liczba podpowłok] · "
        f"T-seal wyłączony · krawędzie samonadzoru = {K_SSL}-NN w surowych cechach "
        f"({meta['purity']['n_pairs']} par: ta sama grupa {meta['purity']['same_group']:.0%}, "
        f"ten sam okres {meta['purity']['same_period']:.0%}, sąsiedzi Z {meta['purity']['z_adjacent']:.0%})",
        "",
        f"Seedy={seeds} · kroki SL={meta['steps'][0]} UL={meta['steps'][1]} RL={meta['steps'][2]} · czas={meta['elapsed']:.0f}s · "
        "skrypt `archiwum/skrypty/run_mendelejew_ml3_zc.py`",
        "",
        "## A1. Nadzorowane — czy okres/grupę/blok da się przewidzieć z konfiguracji (test 30%)",
        "",
        "| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, k in (("period", 7), ("group", 18), ("block", 4)):
        r = resA["sl"][task]
        acc = {a: [x["test_acc"] for x in r[a]] for a in ARMS}
        L.append(f"| {task} | {k} | {ms([x['majority_acc'] for x in r['most']])[0]:.3f} | "
                 + " | ".join(fmt(acc[a]) for a in ARMS) + f" | {wins(acc['most'], acc['softmax'])}/{len(seeds)} |")
    L += ["", "## A2. Nienadzorowane — czy okresowość wyłania się bez etykiet (k-means → NMI)", "",
          "| cechy / arm | stan | NMI okres | NMI grupa | NMI blok |", "|---|---|---:|---:|---:|"]
    for feat in ("content", "tracer", "concat"):
        L.append(f"| surowe {feat} | — | " + " | ".join(
            fmt([x[feat][t]["nmi"] for x in resA["ul_raw"]]) for t in ("period", "group", "block")) + " |")
    for a in ARMS:
        for st in ("untrained", "ssl"):
            L.append(f"| {a} | {st} | " + " | ".join(
                fmt([x[st][t]["nmi"] for x in resA["ul"][a]]) for t in ("period", "group", "block")) + " |")
    L += ["", "## A3. Ze wzmocnieniem — nagroda: wybrany pierwiastek z tej samej grupy", "",
          "| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        rl, ref = resA["rl"][a], resA["rl_ref"][a]
        L.append(f"| {a} | {fmt([x['test_hit1'] for x in rl])} | {fmt([x['train_hit1'] for x in rl])} | "
                 f"{fmt([x['test_hit1'] for x in ref])} | {ms([x['random_hit1'] for x in rl])[0]:.3f} |")
    m, s = [x["test_hit1"] for x in resA["rl"]["most"]], [x["test_hit1"] for x in resA["rl"]["softmax"]]
    L += ["", f"Most > Softmax (RL test) na {wins(m, s)}/{len(seeds)} seedach."]

    L += ["", "## B. Odkrycie — jaka jest konfiguracja brakującego pierwiastka?", "",
          "Usunięte: " + ", ".join(f"**{k}** (Z={v['Z']}, {HELD_OUT[k]})" for k, v in resB["gaps"].items()) + ". "
          "Z luki wynika z ciągu Z; model ma wybrać jej konfigurację spośród kandydatów "
          "(prawdziwa + warianty z 1–2 elektronami przesuniętymi w zewnętrznych podpowłokach). "
          "Wynik kandydata = CSLS w przestrzeni embeddingów (uczonej bez luk). Pozycja 1 = trafienie.",
          "",
          "| luka | kandydatów | prawdziwa konfiguracja (koniec) | Aufbau | surowe kNN | most | softmax | zero | mlp |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for sym, g in resB["gaps"].items():
        cells = []
        for mth in ["reguła Aufbau (bez ML)", "surowe cechy kNN"] + list(ARMS):
            rk = [x["rank_true"] for x in g["methods"][mth]]
            cells.append(f"{rk[0]}" if len(rk) == 1 else f"{ms(rk)[0]:.1f} ({sum(r == 1 for r in rk)}/{len(rk)}×1.)")
        L.append(f"| {sym} | {g['n_cands']} | {g['true_cfg']} | " + " | ".join(cells) + " |")
    L += ["", "Średnia pozycja (w nawiasie: ile seedów trafiło na 1. miejsce). Losowo ≈ (kandydaci+1)/2.", "",
          "### Analogi i przewidziana grupa/blok (sonda z prawdziwą konfiguracją, 4-NN wśród znanych, seed 0)", "",
          "| luka | grupa prawdz. | metoda | 4-NN | grupa przewid. | blok przewid. |", "|---|---:|---|---|---:|:-:|"]
    for sym, g in resB["gaps"].items():
        for mth in ["surowe cechy kNN"] + list(ARMS):
            x = g["methods"][mth][0]
            ok = "✓" if x["group_pred"] == g["group"] else "✗"
            L.append(f"| {sym} | {g['group']} | {mth} | {x['knn_true']} | {x['group_pred']} {ok} | {x['block_pred']} |")
    L += ["", "Trafność grupy z 4-NN po wszystkich seedach:", "", "| metoda | Sc | Ga | Ge | Tc | razem |", "|---|---:|---:|---:|---:|---:|"]
    for mth in ["surowe cechy kNN"] + list(ARMS):
        hits = [sum(x["group_pred"] == g["group"] for x in g["methods"][mth]) / len(g["methods"][mth])
                for g in resB["gaps"].values()]
        L.append(f"| {mth} | " + " | ".join(f"{h:.1f}" for h in hits) + f" | {sum(hits) / len(hits):.2f} |")
    L += ["", "### Eksploracja: Z=119 i 120 (seed 0, top-3 wg CSLS)", "",
          "Teoria (ekstrapolacja Aufbau / obliczenia relatywistyczne): 119 = [Og] 8s1, 120 = [Og] 8s2. "
          "Uwaga: podpowłoki 8s/5g/6f/7d/8p nie są obsadzone u żadnego znanego pierwiastka — "
          "ich wagi w treści są nieuczone; o wyborze może decydować tylko τ.", "",
          "| Z | arm | 1. | 2. | 3. | 4-NN zwycięzcy |", "|---|---|---|---|---|---|"]
    for Z, d in resB["frontier"].items():
        for a in ARMS:
            top = d[a]
            L.append(f"| {Z} | {a} | " + " | ".join(f"{c} ({s})" for c, s, _ in top[:3]) + f" | {top[0][2]} |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta, "A": resA, "B": resB}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = (40, 40, 60) if args.smoke else (400, 400, 600)
    rows_all = load_rows(SRC)
    t0 = time.time()

    content, tracer, labels, meta_rows, _ = build(rows_all)
    ssl_pairs = knn_partners(content, tracer)
    purity = pair_purity(ssl_pairs, meta_rows)
    rl_partners = same_group_partners(meta_rows)
    print(f"ZC: N={content.shape[0]} d_x={content.shape[1]} ssl_pairs={purity}", flush=True)
    resA = {"sl": {t: {a: [] for a in ARMS} for t in labels}, "ul": {a: [] for a in ARMS}, "ul_raw": [],
            "rl": {a: [] for a in ARMS}, "rl_ref": {a: [] for a in ARMS}}
    for seed in seeds:
        print(f"\n=== A seed {seed} ===", flush=True)
        resA["ul_raw"].append(ml3.run_ul_raw(content, tracer, labels, seed))
        for a in ARMS:
            for t in labels:
                resA["sl"][t][a].append(ml3.run_sl(a, t, content, tracer, labels, seed, steps[0]))
            resA["ul"][a].append(ml3.run_ul(a, content, tracer, labels, ssl_pairs, seed, steps[1]))
            resA["rl"][a].append(ml3.run_rl(a, content, tracer, rl_partners, seed, steps[2]))
            resA["rl_ref"][a].append(ml3.run_rl(a, content, tracer, rl_partners, seed, steps[2], "sl_ref"))
            ul = resA["ul"][a][-1]["ssl"]
            print(f"  {a:8s} SL[" + " ".join(f"{t}={resA['sl'][t][a][-1]['test_acc']:.2f}" for t in labels)
                  + f"] UL-NMI[p={ul['period']['nmi']:.2f} g={ul['group']['nmi']:.2f} b={ul['block']['nmi']:.2f}]"
                  + f" RL={resA['rl'][a][-1]['test_hit1']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== B odkrycie ===", flush=True)
    resB = run_discovery(rows_all, seeds, steps[1])
    meta = {"seeds": seeds, "steps": steps, "elapsed": time.time() - t0, "smoke": args.smoke, "purity": purity}
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_ML3_ZC{tag}.md"
    write_report(resA, resB, meta, md, OUT_DIR / f"mendelejew_ml3_zc{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
