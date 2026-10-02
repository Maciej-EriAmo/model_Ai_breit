"""Mendelejew ML3 FULL — SL / UL / RL na pełnej tablicy (118) + szukanie nowego pierwiastka.

Graf: FULL118 jak w `export_mendelejew_full.py` — 118 jąder, krawędzie bind
Z-sąsiedzi / ta sama grupa / ten sam okres. Tracer: T ~ okres, mazur ~ grupa.

Część A — powtórka ML3 (armie most / softmax / zero / mlp, funkcje z run_mendelejew_ml3):
  SL: klasyfikacja węzła → okres (7), grupa (18), blok (s/p/d/f)
  UL: k-means na surowych cechach / embeddingach po InfoNCE → NMI/ARI
  RL: bandyta REINFORCE na krawędziach bind

Część B — „odkrycie” (tylko uczenie nienadzorowane):
  1. Z grafu usuwamy pierwiastki przewidziane przez Mendelejewa:
     Sc (eka-bor), Ga (eka-glin), Ge (eka-krzem), Tc (eka-mangan).
  2. Enkoder uczy się samonadzorem (InfoNCE na krawędziach) — bez etykiet.
  3. Kandydaci = każda wolna komórka (grupa, okres) siatki 18×7:
     4 prawdziwe luki + 36 komórek nieistniejących (np. g5/p2).
     Sonda = nieznany pierwiastek (treść = hash „?”, tracer z pozycji).
  4. Wynik komórki = gęstość: średni cos do 4 najbliższych znanych pierwiastków.
     Czy prawdziwe luki są na szczycie rankingu? Jakie mają sąsiedztwo / blok?
  5. Eksploracja: okres 8 (pierwiastki 119+) — ranking 18 komórek.

  python archiwum/skrypty/run_mendelejew_ml3_full.py [--smoke]
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
from pathlib import Path

import torch
import torch.nn.functional as F

from config import CANON
from dump_recall import bind_pairs_idx
from export_mendelejew_full import _block, rows_to_table_graph
from import_struktura_atomow import load_rows
from phi_schema import PhiAtom, PhiState, atom_content_vec, atom_tracer_vec
import run_mendelejew_ml3 as ml3
from run_mendelejew_ml3 import ARMS, encode, fmt, make_encoder, ms, noisy, wins

ROOT = _LAB
SRC = ROOT / "archiwum" / "dane" / "struktura_atomow_full.csv"
OUT_DIR = ROOT / "doswiadczenie" / "glify"
HELD_OUT = {"Sc": "eka-bor", "Ga": "eka-glin", "Ge": "eka-krzem", "Tc": "eka-mangan"}
BLOCKS = ["s", "p", "d", "f"]
KNN = 4


# ---------------------------------------------------------------- dane


def _atom_for(g: int, p: int, name: str, sym: str, Z: int | None) -> PhiAtom:
    """Ta sama reguła co rows_to_table_graph; Z=None → nieznany pierwiastek."""
    T = min(1.0, 0.15 + 0.12 * p)
    if Z is None:
        S, E, payload = f"element/Z?/g{g}/p{p}/block:?", f"? group={g} period={p}", "?"
    else:
        b = _block(g, p, Z)
        S = f"element/Z{Z}/g{g}/p{p}/block:{b}"
        E = f"{name} ({sym}) Z={Z} group={g} period={p} block={b}"
        payload = f"{sym};Z={Z};g={g};p={p};block={b}"
    return PhiAtom(atom_id=f"{sym}/nucleus", S=S, E=E, T=T, state=PhiState.HOT,
                   bubble_id=f"period/{p}", depth=min(2, max(0, p - 1) // 3),
                   payload=payload, mazur_tracer=g / 18.0)


def tensors(atoms: list[PhiAtom], t_max: float):
    c, t = [], []
    for a in atoms:
        aa = PhiAtom(**{**a.__dict__, "T": a.T / t_max})
        c.append(atom_content_vec(aa, CANON.d_x))
        t.append(atom_tracer_vec(aa, CANON.n_dim))
    return torch.stack(c), torch.stack(t)


def build(rows):
    g = rows_to_table_graph(rows)
    t_max = max(a.T for a in g.atoms)
    content, tracer = tensors(g.atoms, t_max)
    meta = [(int(r["group"]), int(r["period"]), int(r["atom_number"]), r["symbol"]) for r in rows]
    labels = {
        "period": torch.tensor([p - 1 for _, p, _, _ in meta]),
        "group": torch.tensor([gr - 1 for gr, _, _, _ in meta]),
        "block": torch.tensor([BLOCKS.index(_block(gr, p, Z)) for gr, p, Z, _ in meta]),
    }
    partners: dict[int, set[int]] = {}
    for a, b in bind_pairs_idx(g):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return g, content, tracer, labels, partners, meta, t_max


# ---------------------------------------------------------------- odkrycie


def train_ssl(arm, content, tracer, partners, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    pairs = [(a, b) for a, ps in partners.items() for b in ps]
    q_all = torch.tensor([p[0] for p in pairs])
    p_all = torch.tensor([p[1] for p in pairs])
    N = content.shape[0]
    opt = torch.optim.Adam(enc.parameters(), lr=CANON.lr)
    for _ in range(steps):
        enc.train()
        sel = torch.randint(len(pairs), (64,), generator=gen)
        q, p = q_all[sel], p_all[sel]
        h = F.normalize(encode(enc, arm, noisy(content, gen), tracer), dim=-1)
        s = (h[q] @ h.T / ml3.TEMP).masked_fill(F.one_hot(q, N).bool(), -1e9)
        loss = F.cross_entropy(s, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    return enc


@torch.no_grad()
def probe_scores(enc, arm, content, tracer, probes_c, probes_t):
    """Każda sonda osobno doklejona do znanych węzłów → (gęstość, csls, indeksy kNN).

    csls (post-hoc, korekta hubness): 2·gęstość(sonda) − średnia gęstość jej kNN
    wśród znanych — zdejmuje premię za trafienie w ciasno upakowany region.
    """
    out = []
    N = content.shape[0]
    if enc is None:
        h0 = F.normalize(torch.cat([content, tracer], -1), dim=-1)
    else:
        h0 = F.normalize(encode(enc, arm, content, tracer), dim=-1)
    s0 = (h0 @ h0.T).fill_diagonal_(-2.0)
    r_known = s0.topk(KNN, dim=-1).values.mean(-1)  # [N]
    for pc, pt in zip(probes_c, probes_t):
        if enc is None:  # surowe cechy
            h = F.normalize(torch.cat([torch.cat([content, pc[None]]), torch.cat([tracer, pt[None]])], -1), dim=-1)
        else:
            h = F.normalize(encode(enc, arm, torch.cat([content, pc[None]]), torch.cat([tracer, pt[None]])), dim=-1)
        sim = h[:N] @ h[N]
        top = sim.topk(KNN)
        dens = top.values.mean().item()
        csls = 2 * dens - r_known[top.indices].mean().item()
        out.append((dens, csls, top.indices.tolist()))
    return out


def grid_support(cell, occupied):
    g, p = cell
    return sum((g + dg, p + dp) in occupied for dg, dp in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def rank_of(scores, idx_targets):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    pos = {i: r + 1 for r, i in enumerate(order)}
    return [pos[i] for i in idx_targets]


def auc(scores, pos_idx):
    pos = [scores[i] for i in pos_idx]
    neg = [s for i, s in enumerate(scores) if i not in pos_idx]
    tot = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return tot / (len(pos) * len(neg))


def run_discovery(rows_all, seeds, steps):
    rows = [r for r in rows_all if r["symbol"] not in HELD_OUT]
    g, content, tracer, _, partners, meta, t_max = build(rows)
    occupied = {(gr, p) for gr, p, _, _ in meta}
    held = {r["symbol"]: (int(r["group"]), int(r["period"]), int(r["atom_number"])) for r in rows_all
            if r["symbol"] in HELD_OUT}
    cand = sorted({(gr, p) for gr in range(1, 19) for p in range(1, 8)} - occupied, key=lambda c: (c[1], c[0]))
    true_idx = [cand.index((v[0], v[1])) for v in held.values()]
    frontier = [(gr, 8) for gr in range(1, 19)]
    pc, pt = tensors([_atom_for(gr, p, "?", f"?{gr}_{p}", None) for gr, p in cand + frontier], t_max)
    nc = len(cand)

    def neigh_desc(knn):
        return ", ".join(meta[i][3] for i in knn)

    def blk_vote(knn):
        votes = [_block(meta[i][0], meta[i][1], meta[i][2]) for i in knn]
        return max(set(votes), key=votes.count)

    res = {"candidates": [f"g{gr}/p{p}" for gr, p in cand], "held_out": {k: list(v) for k, v in held.items()},
           "methods": {}}
    sup = [float(grid_support(c, occupied)) for c in cand]
    res["methods"]["reguła siatki (bez ML)"] = [{"ranks": rank_of(sup, true_idx), "auc": auc(sup, true_idx)}]
    raw = probe_scores(None, None, content, tracer, pc, pt)
    raw_s = [s for s, _, _ in raw[:nc]]
    raw_c = [c for _, c, _ in raw[:nc]]
    res["methods"]["surowe cechy kNN"] = [{"ranks": rank_of(raw_s, true_idx), "auc": auc(raw_s, true_idx)}]
    res["methods_csls"] = {"surowe cechy kNN": [{"ranks": rank_of(raw_c, true_idx), "auc": auc(raw_c, true_idx)}]}
    detail = {}
    for arm in ARMS:
        res["methods"][arm] = []
        res["methods_csls"][arm] = []
        for seed in seeds:
            enc = train_ssl(arm, content, tracer, partners, seed, steps)
            sc = probe_scores(enc, arm, content, tracer, pc, pt)
            s = [x for x, _, _ in sc[:nc]]
            cs = [c for _, c, _ in sc[:nc]]
            row = {"ranks": rank_of(s, true_idx), "auc": auc(s, true_idx)}
            res["methods_csls"][arm].append({"ranks": rank_of(cs, true_idx), "auc": auc(cs, true_idx)})
            if seed == seeds[0]:
                order = sorted(range(nc), key=lambda i: -cs[i])
                row["top8"] = [(f"g{cand[i][0]}/p{cand[i][1]}", round(cs[i], 3), neigh_desc(sc[i][2])) for i in order[:8]]
                row["gaps"] = {sym: {"rank": rank_of(s, [cand.index((v[0], v[1]))])[0],
                                     "rank_csls": rank_of(cs, [cand.index((v[0], v[1]))])[0],
                                     "knn": neigh_desc(sc[cand.index((v[0], v[1]))][2]),
                                     "block_pred": blk_vote(sc[cand.index((v[0], v[1]))][2]),
                                     "block_true": _block(v[0], v[1], v[2])} for sym, v in held.items()}
                fs = [c for _, c, _ in sc[nc:]]
                fo = sorted(range(len(frontier)), key=lambda i: -fs[i])
                row["frontier"] = [(f"g{frontier[i][0]}/p8", round(fs[i], 3), neigh_desc(sc[nc + i][2])) for i in fo]
            res["methods"][arm].append(row)
            c_ = res["methods_csls"][arm][-1]
            print(f"  [B] {arm:8s} seed={seed} dens ranks={row['ranks']} auc={row['auc']:.3f} | "
                  f"csls ranks={c_['ranks']} auc={c_['auc']:.3f}", flush=True)
    res["n_candidates"] = nc
    return res


# ---------------------------------------------------------------- raport


def write_report(resA, resB, meta, md, js):
    seeds = meta["seeds"]
    L = [
        "# Mendelejew ML3 FULL — pełna tablica (118) + szukanie nowego pierwiastka",
        "",
        f"Graf FULL118 (jądra; bind: Z-sąsiedzi / grupa / okres) · krawędzie bind={meta['n_bind']} · "
        f"seedy={seeds} · kroki SL={meta['steps'][0]} UL={meta['steps'][1]} RL={meta['steps'][2]} · "
        f"czas={meta['elapsed']:.0f}s · skrypt `archiwum/skrypty/run_mendelejew_ml3_full.py`",
        "",
        "Tracer węzła = (głębokość z okresu, T ~ okres, mazur = grupa/18); treść = hash napisu (tożsamość, nie fizyka).",
        "",
        "## A1. Nadzorowane — klasyfikacja węzłów (test = 30% wstrzymanych pierwiastków)",
        "",
        "| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, k in (("period", 7), ("group", 18), ("block", 4)):
        r = resA["sl"][task]
        acc = {a: [x["test_acc"] for x in r[a]] for a in ARMS}
        L.append(f"| {task} | {k} | {ms([x['majority_acc'] for x in r['most']])[0]:.3f} | "
                 + " | ".join(fmt(acc[a]) for a in ARMS) + f" | {wins(acc['most'], acc['softmax'])}/{len(seeds)} |")
    L += ["", "Train acc:", "", "| zadanie | most | softmax | zero | mlp |", "|---|---:|---:|---:|---:|"]
    for task in ("period", "group", "block"):
        r = resA["sl"][task]
        L.append(f"| {task} | " + " | ".join(fmt([x["train_acc"] for x in r[a]]) for a in ARMS) + " |")
    L += ["", "## A2. Nienadzorowane — k-means → NMI", "",
          "| cechy / arm | stan | NMI okres | NMI grupa | NMI blok |", "|---|---|---:|---:|---:|"]
    for feat in ("content", "tracer", "concat"):
        L.append(f"| surowe {feat} | — | " + " | ".join(
            fmt([x[feat][t]["nmi"] for x in resA["ul_raw"]]) for t in ("period", "group", "block")) + " |")
    for a in ARMS:
        for st in ("untrained", "ssl"):
            L.append(f"| {a} | {st} | " + " | ".join(
                fmt([x[st][t]["nmi"] for x in resA["ul"][a]]) for t in ("period", "group", "block")) + " |")
    L += ["", "## A3. Ze wzmocnieniem — bandyta REINFORCE (partner bind)", "",
          "| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |", "|---|---:|---:|---:|---:|"]
    for a in ARMS:
        rl, ref = resA["rl"][a], resA["rl_ref"][a]
        L.append(f"| {a} | {fmt([x['test_hit1'] for x in rl])} | {fmt([x['train_hit1'] for x in rl])} | "
                 f"{fmt([x['test_hit1'] for x in ref])} | {ms([x['random_hit1'] for x in rl])[0]:.3f} |")
    m, s = [x["test_hit1"] for x in resA["rl"]["most"]], [x["test_hit1"] for x in resA["rl"]["softmax"]]
    L += ["", f"Most > Softmax (RL test) na {wins(m, s)}/{len(seeds)} seedach."]

    nc = resB["n_candidates"]
    L += [
        "",
        "## B. Szukanie nowego pierwiastka (nienadzorowane)",
        "",
        "Usunięte z danych: " + ", ".join(f"**{k}** ({v}, g{resB['held_out'][k][0]}/p{resB['held_out'][k][1]})"
                                          for k, v in HELD_OUT.items()) + ".",
        f"Kandydaci: {nc} wolnych komórek siatki 18×7 = 4 prawdziwe luki + {nc - 4} komórek nieistniejących.",
        f"Wynik komórki = średni cos sondy do {KNN} najbliższych znanych pierwiastków (gęstość). "
        "Idealnie: 4 luki na miejscach 1–4, AUC = 1. Losowo: średnia pozycja ≈ "
        f"{(nc + 1) / 2:.1f}, AUC = 0.5.",
        "",
        "| wynik | metoda | średnia pozycja luk | luki w top-4 | luki w top-8 | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for score, name, runs in [("gęstość", n, r) for n, r in resB["methods"].items()] + [
            ("CSLS (post-hoc)", n, r) for n, r in resB["methods_csls"].items()]:
        ranks = [r for x in runs for r in x["ranks"]]
        n = len(runs)
        L.append(f"| {score} | {name} | {sum(ranks) / len(ranks):.1f} | {sum(r <= 4 for r in ranks) / n:.1f}/4 | "
                 f"{sum(r <= 8 for r in ranks) / n:.1f}/4 | "
                 + (fmt([x["auc"] for x in runs]) if n > 1 else f"{runs[0]['auc']:.3f}") + " |")
    L += ["", "### Szczegóły luk (seed 0): pozycja, najbliżsi sąsiedzi, przewidziany blok", "",
          "| luka | arm | pozycja (gęstość) | pozycja (CSLS) | 4-NN (znane) | blok przewid. | blok prawdz. |",
          "|---|---|---:|---:|---|:-:|:-:|"]
    for sym in HELD_OUT:
        for a in ARMS:
            d = resB["methods"][a][0]["gaps"][sym]
            ok = "✓" if d["block_pred"] == d["block_true"] else "✗"
            L.append(f"| {sym} | {a} | {d['rank']} | {d['rank_csls']} | {d['knn']} | {d['block_pred']} {ok} | {d['block_true']} |")
    L += ["", "### Top-8 komórek wg Mostu, wynik CSLS (seed 0)", "", "| # | komórka | CSLS | 4-NN |", "|---:|---|---:|---|"]
    gap_cells = {f"g{v[0]}/p{v[1]}": k for k, v in resB["held_out"].items()}
    for i, (c, sc, nn) in enumerate(resB["methods"]["most"][0]["top8"], 1):
        tag = f" ← **{gap_cells[c]}**" if c in gap_cells else ""
        L.append(f"| {i} | {c}{tag} | {sc} | {nn} |")
    L += ["", "### Eksploracja: okres 8 (pierwiastki 119+), ranking CSLS wg Mostu i MLP (seed 0)", "",
          "| # | most: komórka | 4-NN | mlp: komórka | 4-NN |", "|---:|---|---|---|---|"]
    fm, fl = resB["methods"]["most"][0]["frontier"], resB["methods"]["mlp"][0]["frontier"]
    for i in range(6):
        L.append(f"| {i + 1} | {fm[i][0]} ({fm[i][1]}) | {fm[i][2]} | {fl[i][0]} ({fl[i][1]}) | {fl[i][2]} |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta, "A": resA, "B": resB}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-b", action="store_true", help="część A wczytaj z istniejącego JSON")
    args = ap.parse_args(argv)
    tag = "_smoke" if args.smoke else ""
    js = OUT_DIR / f"mendelejew_ml3_full{tag}_results.json"
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = (40, 40, 60) if args.smoke else (400, 400, 600)
    rows_all = load_rows(SRC)
    t0 = time.time()

    _, content, tracer, labels, partners, _, _ = build(rows_all)
    print(f"FULL: N={content.shape[0]} bind_nodes={len(partners)}", flush=True)
    prev = json.loads(js.read_text(encoding="utf-8")) if args.only_b else None
    resA = prev["A"] if prev else {"sl": {t: {a: [] for a in ARMS} for t in labels}, "ul": {a: [] for a in ARMS}, "ul_raw": [],
            "rl": {a: [] for a in ARMS}, "rl_ref": {a: [] for a in ARMS}}
    for seed in ([] if prev else seeds):
        print(f"\n=== A seed {seed} ===", flush=True)
        resA["ul_raw"].append(ml3.run_ul_raw(content, tracer, labels, seed))
        for a in ARMS:
            for t in labels:
                resA["sl"][t][a].append(ml3.run_sl(a, t, content, tracer, labels, seed, steps[0]))
            resA["ul"][a].append(ml3.run_ul(a, content, tracer, labels, partners, seed, steps[1]))
            resA["rl"][a].append(ml3.run_rl(a, content, tracer, partners, seed, steps[2]))
            resA["rl_ref"][a].append(ml3.run_rl(a, content, tracer, partners, seed, steps[2], "sl_ref"))
            print(f"  {a:8s} SL[" + " ".join(f"{t}={resA['sl'][t][a][-1]['test_acc']:.2f}" for t in labels)
                  + f"] RL={resA['rl'][a][-1]['test_hit1']:.2f} ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== B odkrycie ===", flush=True)
    resB = run_discovery(rows_all, seeds, steps[1])
    elapsed = time.time() - t0 + (prev["meta"]["elapsed"] if prev else 0)
    meta = {"seeds": seeds, "steps": steps, "elapsed": elapsed, "smoke": args.smoke,
            "n_bind": sum(len(p) for p in partners.values()) // 2}
    md = OUT_DIR / f"MENDELEJEW_ML3_FULL{tag}.md"
    write_report(resA, resB, meta, md, js)
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
