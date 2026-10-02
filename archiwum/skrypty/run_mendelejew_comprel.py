"""Mendelejew COMPREL — relacje kompozycyjne + nauka relacyjna (osobne głowice).

Cechy jak ZC (tylko Z + konfiguracja); split jak SEMI; operatory liniowe jak COMP.
Wszystkie fakty/pary WYŁĄCZNIE spośród pierwiastków treningowych.

Warianty:
  comp          — tylko operatory „dół”/„prawo” (≡ COMP lin)
  comp+rel      — operatory + osobne głowice: „grupa” (K=all par, InfoNCE) i „Z” (|ΔZ|=1);
                  odpowiedź z samego operatora
  comp+rel·PoE  — ten sam model; dla łańcuchów złożonych WYŁĄCZNIE z „dół”
                  wynik = cos(operator) + cos(głowica grupy) — operator wskazuje okres/okolicę,
                  głowica grupy kolumnę (cel „dół” jest zawsze w tej samej grupie)
  comp+mix      — pary grupowe uczone w TEJ SAMEJ przestrzeni co operatory (kontrola zlania)

Ocena jak COMP: A (jeden krok z pierwiastkiem testowym), B (złożenia nieuczone:
ścieżka treningowa / dotyka testu; op i hop), pytania Mendelejewa.

  python archiwum/skrypty/run_mendelejew_comprel.py [--smoke]
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
import run_mendelejew_ml3 as ml3
from run_mendelejew_ml3 import ARMS, encode, fmt, make_encoder, ms, noisy
from run_mendelejew_comp import COMPS, MENDELEJEW_Q, Ops, comp_queries, facts, summarize
from run_mendelejew_rel import relation_pairs
from run_mendelejew_semi import _nce, labeled_pairs, split
from config import CANON
from import_struktura_atomow import load_rows

OUT_DIR = _LAB / "doswiadczenie" / "glify"
VARIANTS = ["comp", "comp+rel", "comp+rel·PoE", "comp+mix"]
TRAINED = {"comp": "comp", "comp+rel": "rel", "comp+rel·PoE": "rel", "comp+mix": "mix"}


def train(arm, mode, content, tracer, fr, gpairs, zpairs, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    ops = Ops("lin", CANON.d_model)
    heads = nn.ModuleDict({r: nn.Linear(CANON.d_model, CANON.d_model) for r in ("grupa", "Z")}) if mode == "rel" else nn.ModuleDict()
    opt = torch.optim.Adam(list(enc.parameters()) + list(ops.parameters()) + list(heads.parameters()), lr=CANON.lr)
    N = content.shape[0]
    for _ in range(steps):
        enc.train()
        raw = encode(enc, arm, noisy(content, gen), tracer)
        h = F.normalize(raw, dim=-1)
        loss = torch.zeros(())
        for r, fs in fr.items():
            sel = torch.randint(len(fs), (32,), generator=gen).tolist()
            a = torch.tensor([fs[i][0] for i in sel])
            b = torch.tensor([fs[i][1] for i in sel])
            s = (ops.apply(h[a], [r]) @ h.T / ml3.TEMP).masked_fill(F.one_hot(a, N).bool(), -1e9)
            loss = loss + F.cross_entropy(s, b)
        if mode == "rel":
            loss = loss + _nce(F.normalize(heads["grupa"](raw), dim=-1), gpairs, gen)
            loss = loss + _nce(F.normalize(heads["Z"](raw), dim=-1), zpairs, gen)
        elif mode == "mix":
            loss = loss + _nce(h, gpairs, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        raw = encode(enc, arm, content, tracer)
        g = F.normalize(heads["grupa"](raw), dim=-1) if mode == "rel" else None
        return F.normalize(raw, dim=-1), ops, g


@torch.no_grad()
def scores(h, ops, g, x, r, poe):
    s = ops.apply(h[x:x + 1], [r])[0] @ h.T
    if poe and r == "dół" and g is not None:
        s = s + g[x] @ g.T
    return s.clone()


@torch.no_grad()
def query(h, ops, g, a, rels, b, mode, poe):
    if mode == "op":
        all_down = all(r == "dół" for r in rels)
        s = ops.apply(h[a:a + 1], list(rels))[0] @ h.T
        if poe and all_down and g is not None:
            s = s + g[a] @ g.T
        s = s.clone()
        s[a] = -1e9
        return int((s > s[b]).sum()) + 1
    x = a  # hop
    for r in rels[:-1]:
        s = scores(h, ops, g, x, r, poe)
        s[x] = -1e9
        x = int(s.argmax())
    if x == b:
        return 10**6
    s = scores(h, ops, g, x, rels[-1], poe)
    s[a] = -1e9
    s[x] = -1e9
    return int((s > s[b]).sum()) + 1


def evaluate(h, ops, g, poe, meta, trs, fr_all):
    out = {}
    atomic = [(r, a, b) for r, fs in fr_all.items() for a, b in fs if a not in trs or b not in trs]
    for rel in ("dół", "prawo"):
        out[f"A_{rel}"] = summarize([query(h, ops, g, a, (r,), b, "op", poe) for r, a, b in atomic if r == rel])
    cq = comp_queries(fr_all)
    for sp, flt in (("tr", lambda a, x, b: {a, x, b} <= trs), ("te", lambda a, x, b: not {a, x, b} <= trs)):
        qs = [(rels, a, b) for rels, a, x, b in cq if flt(a, x, b)]
        for mode in ("op", "hop"):
            out[f"B_{sp}_{mode}"] = summarize([query(h, ops, g, a, rels, b, mode, poe) for rels, a, b in qs])
        dd = [(rels, a, b) for rels, a, b in qs if rels == ("dół", "dół")]
        out[f"B_{sp}_dd"] = summarize([query(h, ops, g, a, rels, b, "op", poe) for rels, a, b in dd])
    syms = [m[3] for m in meta]
    out["mendelejew"] = {}
    for hs, rels, ts in MENDELEJEW_Q:
        a, b = syms.index(hs), syms.index(ts)
        s = ops.apply(h[a:a + 1], list(rels))[0] @ h.T
        if poe and all(r == "dół" for r in rels) and g is not None:
            s = s + g[a] @ g.T
        s = s.clone()
        s[a] = -1e9
        out["mendelejew"][f"{'∘'.join(reversed(rels))}({hs})={ts}"] = {
            "rank": int((s > s[b]).sum()) + 1, "top3": [syms[i] for i in s.topk(3).indices.tolist()]}
    return out


def write_report(res, meta_run, md, js):
    seeds = meta_run["seeds"]
    L = [
        "# Mendelejew COMPREL — relacje kompozycyjne + nauka relacyjna",
        "",
        "Cechy jak ZC (tylko Z + konfiguracja); operatory liniowe „dół”/„prawo” jak COMP; głowice „grupa” (K=all par) "
        "i „Z” jak REL. Wszystkie fakty i pary tylko spośród pierwiastków treningowych. "
        f"Seedy={seeds} · kroki={meta_run['steps']} · czas={meta_run['elapsed']:.0f}s · skrypt `archiwum/skrypty/run_mendelejew_comprel.py`",
        "",
        "`comp+rel·PoE` = ten sam model co `comp+rel`, ale dla łańcuchów samych „dół” wynik = cos(operator) + cos(głowica grupy). "
        "Dla łańcuchów z „prawo” PoE = sam operator (cel nie jest w tej samej grupie).",
        "",
    ]
    hdr = "| test | " + " | ".join(f"{v}:{a}" for v in VARIANTS for a in ARMS) + " |"
    for title, keys in (
        ("A. Jeden krok z pierwiastkiem testowym — hit@1", [("A_dół", "dół"), ("A_prawo", "prawo")]),
        ("B. Złożenia nieuczone — hit@1",
         [("B_tr_op", "treningowa · op"), ("B_tr_hop", "treningowa · hop"), ("B_tr_dd", "treningowa · dół∘dół"),
          ("B_te_op", "dotyka testu · op"), ("B_te_hop", "dotyka testu · hop"), ("B_te_dd", "dotyka testu · dół∘dół")]),
    ):
        L += [f"## {title}", "", "Kolumny: wariant × arm.", ""]
        for a in ARMS:
            pass
        L += ["| test | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
        for a in ARMS:
            for key, name in keys:
                L.append(f"| {a} · {name} | " + " | ".join(fmt([x[key]["hit1"] for x in res[f"{v}|{a}"]]) for v in VARIANTS) + " |")
        L.append("")
    L += ["## Efekt połączenia dla Mostu (Δ względem `comp`, wygrane seedy)", "",
          "| test | " + " | ".join(v for v in VARIANTS if v != "comp") + " |", "|---|" + "---:|" * (len(VARIANTS) - 1)]
    for key in ("A_dół", "A_prawo", "B_tr_op", "B_te_op", "B_te_hop", "B_te_dd"):
        base = [x[key]["hit1"] for x in res["comp|most"]]
        cells = []
        for v in VARIANTS:
            if v == "comp":
                continue
            m = [x[key]["hit1"] for x in res[f"{v}|most"]]
            cells.append(f"{ms(m)[0] - ms(base)[0]:+.3f} ({sum(p > q + 1e-9 for p, q in zip(m, base))}/{len(seeds)})")
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    L += ["", "## Most vs reszta w `comp+rel·PoE` (wygrane seedy)", "", "| test | vs softmax | vs zero | vs mlp |", "|---|---:|---:|---:|"]
    for key in ("A_dół", "B_tr_op", "B_te_op", "B_te_hop"):
        m = [x[key]["hit1"] for x in res["comp+rel·PoE|most"]]
        L.append(f"| {key} | " + " | ".join(
            f"{sum(p > q + 1e-9 for p, q in zip(m, [x[key]['hit1'] for x in res[f'comp+rel·PoE|{o}']]))}/{len(seeds)}"
            for o in ("softmax", "zero", "mlp")) + " |")
    L += ["", "## Pytania Mendelejewa (Most) — średnia pozycja (trafień 1.) · top-3 seed 0", "",
          "| pytanie | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for q in res["comp|most"][0]["mendelejew"]:
        cells = []
        for v in VARIANTS:
            rk = [x["mendelejew"][q]["rank"] for x in res[f"{v}|most"]]
            cells.append(f"{ms(rk)[0]:.1f} ({sum(r == 1 for r in rk)}) · {', '.join(res[f'{v}|most'][0]['mendelejew'][q]['top3'])}")
        L.append(f"| {q} | " + " | ".join(cells) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta_run, "results": res}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = 60 if args.smoke else 800
    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    fr_all = facts(meta)
    t0 = time.time()
    res = {}
    for seed in seeds:
        tr, te = split(meta, seed)
        trs = set(tr)
        fr = {r: [(a, b) for a, b in fs if a in trs and b in trs] for r, fs in fr_all.items()}
        gpairs = labeled_pairs(meta, tr, "all", seed)
        zpairs = relation_pairs(meta, tr)["Z"]
        for a in ARMS:
            cache = {}
            for v in VARIANTS:
                mode = TRAINED[v]
                if mode not in cache:
                    cache[mode] = train(a, mode, content, tracer, fr, gpairs, zpairs, seed, steps)
                h, ops, g = cache[mode]
                res.setdefault(f"{v}|{a}", []).append(evaluate(h, ops, g, v.endswith("PoE"), meta, trs, fr_all))
        print(f"  seed={seed} A_dół most: " + " ".join(f"{v}={res[f'{v}|most'][-1]['A_dół']['hit1']:.2f}" for v in VARIANTS)
              + " | B_te_op most: " + " ".join(f"{v}={res[f'{v}|most'][-1]['B_te_op']['hit1']:.2f}" for v in VARIANTS)
              + f" ({time.time() - t0:.0f}s)", flush=True)
    meta_run = {"seeds": seeds, "steps": steps, "elapsed": time.time() - t0, "smoke": args.smoke}
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_COMPREL{tag}.md"
    write_report(res, meta_run, md, OUT_DIR / f"mendelejew_comprel{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
