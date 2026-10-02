"""Mendelejew COMP — relacje kompozycyjne: operatory „w dół” i „w prawo” po tablicy.

Cechy jak ZC (tylko Z + konfiguracja). Split jak SEMI (70/30, Sc/Ga/Ge/Tc zawsze w teście).
Relacje (fakty uczone WYŁĄCZNIE gdy oba końce są treningowe):
  dół   — a → b: ta sama grupa, okres+1 (bez węzłów bloku f — tam „dół” niejednoznaczny)
  prawo — a → b: Z+1
Operatory na h̃ = normalize(enkoder):
  lin   — q = normalize(M_r h̃_a)              złożenie: M_r2 · M_r1
  trans — q = normalize(h̃_a + t_r)  (TransE)   złożenie: t_r1 + t_r2
Strata: InfoNCE po wszystkich 118 kandydatach (bez głowy), temp 0.1.

Ocena (kandydaci = wszystkie pierwiastki ≠ głowa; hit@1, MRR):
  A  pojedynczy krok, fakty z ≥1 końcem testowym (indukcja)
  B  złożenia nigdy nieuczone: dół∘dół, prawo∘dół, dół∘prawo
     B-op    — złożony operator, jedno zapytanie
     B-hop   — dwa kroki: argmax po 1. operatorze, potem 2. operator z tego węzła
     podział: ścieżka w całości treningowa / dotyka testu
  C  przemienność ‖M_d M_p − M_p M_d‖ / ‖M_d M_p‖ (lin)
  D  pytania Mendelejewa: co jest pod Al / Si / Mn; dół∘dół(B); prawo(Zn); dół(Ca)→Sr …
Baseline „bez operatora”: najbliższy sąsiad h̃_a (samo podobieństwo).

  python archiwum/skrypty/run_mendelejew_comp.py [--smoke]
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
from run_mendelejew_semi import split
from config import CANON
from export_mendelejew_full import _block
from import_struktura_atomow import load_rows

OUT_DIR = _LAB / "doswiadczenie" / "glify"
OPS = ["lin", "trans"]
COMPS = [("dół", "dół"), ("dół", "prawo"), ("prawo", "dół")]  # (pierwszy, drugi)
MENDELEJEW_Q = [("Al", ("dół",), "Ga"), ("Si", ("dół",), "Ge"), ("Mn", ("dół",), "Tc"),
                ("B", ("dół", "dół"), "Ga"), ("C", ("dół", "dół"), "Ge"), ("Zn", ("prawo",), "Ga"),
                ("Ga", ("prawo",), "Ge"), ("Ca", ("prawo",), "Sc"), ("Mg", ("dół", "prawo"), "Sc"),
                ("Al", ("prawo", "dół"), "Ge")]


def facts(meta):
    nonf = [i for i, m in enumerate(meta) if _block(*m[:3]) != "f"]
    pos = {(meta[i][0], meta[i][1]): i for i in nonf}
    down = [(i, pos[(meta[i][0], meta[i][1] + 1)]) for i in nonf if (meta[i][0], meta[i][1] + 1) in pos]
    byZ = {m[2]: i for i, m in enumerate(meta)}
    right = [(byZ[z], byZ[z + 1]) for z in sorted(byZ) if z + 1 in byZ]
    return {"dół": down, "prawo": right}


class Ops(nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        if kind == "lin":
            self.M = nn.ParameterDict({r: nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d)) for r in ("dół", "prawo")})
        else:
            self.t = nn.ParameterDict({r: nn.Parameter(0.01 * torch.randn(d)) for r in ("dół", "prawo")})

    def apply(self, x, rels):
        for r in rels:
            x = F.normalize(x @ self.M[r].T if self.kind == "lin" else x + self.t[r], dim=-1)
        return x


def train(arm, kind, content, tracer, fr, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    ops = Ops(kind, CANON.d_model)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ops.parameters()), lr=CANON.lr)
    N = content.shape[0]
    for _ in range(steps):
        enc.train()
        h = F.normalize(encode(enc, arm, noisy(content, gen), tracer), dim=-1)
        loss = torch.zeros(())
        for r, fs in fr.items():
            sel = torch.randint(len(fs), (32,), generator=gen).tolist()
            a = torch.tensor([fs[i][0] for i in sel])
            b = torch.tensor([fs[i][1] for i in sel])
            s = (ops.apply(h[a], [r]) @ h.T / ml3.TEMP).masked_fill(F.one_hot(a, N).bool(), -1e9)
            loss = loss + F.cross_entropy(s, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        return F.normalize(encode(enc, arm, content, tracer), dim=-1), ops


@torch.no_grad()
def rank(h, q, a, b):
    s = (q @ h.T).clone()
    s[a] = -1e9
    return int((s > s[b]).sum()) + 1


@torch.no_grad()
def query(h, ops, a, rels, b, mode):
    if mode == "op":
        return rank(h, ops.apply(h[a:a + 1], list(rels))[0], a, b)
    if mode == "nn":  # bez operatora
        return rank(h, h[a], a, b)
    x = a  # hop: dyskretne kroki
    for r in rels[:-1]:
        s = (ops.apply(h[x:x + 1], [r])[0] @ h.T).clone()
        s[x] = -1e9
        x = int(s.argmax())
    if x == b:
        return 10**6  # pośrednik trafił w cel — liczymy jako chybienie
    s = (ops.apply(h[x:x + 1], [rels[-1]])[0] @ h.T).clone()
    s[a] = -1e9
    s[x] = -1e9
    return int((s > s[b]).sum()) + 1


def comp_queries(fr):
    nxt = {r: dict(fs) for r, fs in fr.items()}
    out = []
    for r1, r2 in COMPS:
        for a, x in nxt[r1].items():
            if x in nxt[r2]:
                out.append(((r1, r2), a, x, nxt[r2][x]))
    return out


def summarize(ranks):
    return {"hit1": sum(r == 1 for r in ranks) / len(ranks), "mrr": sum(1 / r for r in ranks) / len(ranks), "n": len(ranks)}


def run_one(arm, kind, content, tracer, meta, seed, steps):
    tr, te = split(meta, seed)
    trs = set(tr)
    fr_all = facts(meta)
    fr_tr = {r: [(a, b) for a, b in fs if a in trs and b in trs] for r, fs in fr_all.items()}
    h, ops = train(arm, kind, content, tracer, fr_tr, seed, steps)
    out = {}
    atomic = [(r, a, b) for r, fs in fr_all.items() for a, b in fs if a not in trs or b not in trs]
    out["A_op"] = summarize([query(h, ops, a, (r,), b, "op") for r, a, b in atomic])
    out["A_nn"] = summarize([query(h, ops, a, (r,), b, "nn") for r, a, b in atomic])
    for rel in ("dół", "prawo"):
        out[f"A_op_{rel}"] = summarize([query(h, ops, a, (r,), b, "op") for r, a, b in atomic if r == rel])
        out[f"A_nn_{rel}"] = summarize([query(h, ops, a, (r,), b, "nn") for r, a, b in atomic if r == rel])
    out["A_train_op"] = summarize([query(h, ops, a, (r,), b, "op") for r, fs in fr_tr.items() for a, b in fs])
    cq = comp_queries(fr_all)
    for split_name, flt in (("tr", lambda a, x, b: {a, x, b} <= trs), ("te", lambda a, x, b: not {a, x, b} <= trs)):
        qs = [(rels, a, b) for rels, a, x, b in cq if flt(a, x, b)]
        for mode in ("op", "hop", "nn"):
            out[f"B_{split_name}_{mode}"] = summarize([query(h, ops, a, rels, b, mode) for rels, a, b in qs])
        for rels in COMPS:
            q2 = [(r_, a, b) for r_, a, b in qs if r_ == rels]
            if q2:
                out[f"B_{split_name}_op_{'∘'.join(reversed(rels))}"] = summarize([query(h, ops, a, r_, b, "op") for r_, a, b in q2])
    if kind == "lin":
        Md, Mp = ops.M["dół"], ops.M["prawo"]
        out["comm"] = float((Md @ Mp - Mp @ Md).norm() / (Md @ Mp).norm())
    syms = [m[3] for m in meta]
    out["mendelejew"] = {}
    for hs, rels, ts in MENDELEJEW_Q:
        a, b = syms.index(hs), syms.index(ts)
        s = (ops.apply(h[a:a + 1], list(rels))[0] @ h.T).clone()
        s[a] = -1e9
        top = [syms[i] for i in s.topk(3).indices.tolist()]
        out["mendelejew"][f"{'∘'.join(reversed(rels))}({hs})={ts}"] = {"rank": rank(h, ops.apply(h[a:a + 1], list(rels))[0], a, b),
                                                                          "top3": top, "head_train": a in trs}
    return out


def write_report(res, meta_run, md, js):
    seeds = meta_run["seeds"]
    L = [
        "# Mendelejew COMP — relacje kompozycyjne („w dół”, „w prawo”)",
        "",
        "Cechy jak ZC (tylko Z + konfiguracja). Fakty uczone tylko między pierwiastkami treningowymi; "
        "„dół” = ta sama grupa, okres+1 (bez bloku f); „prawo” = Z+1. "
        f"Seedy={seeds} · kroki={meta_run['steps']} · czas={meta_run['elapsed']:.0f}s · skrypt `archiwum/skrypty/run_mendelejew_comp.py`",
        "",
        f"Fakty (wszystkie): dół={meta_run['n_down']}, prawo={meta_run['n_right']}. Losowe hit@1 ≈ {1 / 117:.3f}. "
        "`op` = złożony operator; `hop` = dwa dyskretne kroki; `nn` = bez operatora (najbliższy sąsiad).",
        "",
    ]

    def row(key, metric="hit1"):
        return " | ".join(fmt([x[key][metric] for x in res[f"{k}|{a}"]]) for k in OPS for a in ARMS)

    hdr = "| test | " + " | ".join(f"{k}:{a}" for k in OPS for a in ARMS) + " |"
    sep = "|---|" + "---:|" * (len(OPS) * len(ARMS))
    L += ["## A. Jeden krok — fakty z pierwiastkiem testowym (indukcja), hit@1", "", hdr, sep,
          "| operator | " + row("A_op") + " |", "| bez operatora | " + row("A_nn") + " |",
          "| (kontrola: fakty treningowe) | " + row("A_train_op") + " |",
          "| operator · tylko dół | " + row("A_op_dół") + " |", "| bez operatora · tylko dół | " + row("A_nn_dół") + " |",
          "| operator · tylko prawo | " + row("A_op_prawo") + " |", "| bez operatora · tylko prawo | " + row("A_nn_prawo") + " |", ""]
    L += ["## B. Złożenia nigdy nieuczone, hit@1", "", hdr, sep]
    for sp, name in (("tr", "ścieżka treningowa"), ("te", "ścieżka dotyka testu")):
        for mode in ("op", "hop", "nn"):
            L.append(f"| {name} · {mode} | " + row(f"B_{sp}_{mode}") + " |")
    L += ["", "Złożenia osobno (ścieżka treningowa, operator, hit@1):", "", hdr, sep]
    for rels in COMPS:
        key = f"B_tr_op_{'∘'.join(reversed(rels))}"
        if key in res[f"lin|{ARMS[0]}"][0]:
            L.append(f"| {'∘'.join(reversed(rels))} | " + row(key) + " |")
    L += ["", f"Liczba zapytań B (seed 0): treningowe={res[f'lin|most'][0]['B_tr_op']['n']}, "
          f"dotykające testu={res[f'lin|most'][0]['B_te_op']['n']}.", ""]
    L += ["## B — MRR (złożony operator, ścieżka dotyka testu)", "", hdr, sep, "| MRR | " + row("B_te_op", "mrr") + " |", ""]
    L += ["## C. Przemienność operatorów liniowych ‖M_d M_p − M_p M_d‖ / ‖M_d M_p‖ (0 = siatka)", "",
          "| " + " | ".join(ARMS) + " |", "|" + "---:|" * len(ARMS),
          "| " + " | ".join(fmt([x["comm"] for x in res[f"lin|{a}"]]) for a in ARMS) + " |", ""]
    L += ["## Most vs reszta (B dotyka testu, operator, hit@1 — wygrane seedy)", "",
          "| operatory | vs softmax | vs zero | vs mlp |", "|---|---:|---:|---:|"]
    for k in OPS:
        m = [x["B_te_op"]["hit1"] for x in res[f"{k}|most"]]
        L.append(f"| {k} | " + " | ".join(
            f"{sum(p > q + 1e-9 for p, q in zip(m, [x['B_te_op']['hit1'] for x in res[f'{k}|{o}']]))}/{len(seeds)}"
            for o in ("softmax", "zero", "mlp")) + " |")
    L += ["", "## D. Pytania Mendelejewa (Most, operator liniowy) — pozycja prawdziwej odpowiedzi", "",
          "Średnia pozycja po seedach (1 = trafienie; losowo ≈ 59) · top-3 dla seed 0 · czy głowa była treningowa (seed 0)", "",
          "| pytanie | średnia pozycja | trafień 1. | top-3 (seed 0) | głowa w treningu |", "|---|---:|---:|---|:-:|"]
    for q in res["lin|most"][0]["mendelejew"]:
        rk = [x["mendelejew"][q]["rank"] for x in res["lin|most"]]
        d0 = res["lin|most"][0]["mendelejew"][q]
        L.append(f"| {q} | {ms(rk)[0]:.1f} | {sum(r == 1 for r in rk)}/{len(seeds)} | {', '.join(d0['top3'])} | "
                 f"{'tak' if d0['head_train'] else 'nie'} |")
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
    fr = facts(meta)
    t0 = time.time()
    res = {}
    for seed in seeds:
        for k in OPS:
            for a in ARMS:
                r = run_one(a, k, content, tracer, meta, seed, steps)
                res.setdefault(f"{k}|{a}", []).append(r)
            print(f"  seed={seed} {k:5s} A_op: " + " ".join(f"{a}={res[f'{k}|{a}'][-1]['A_op']['hit1']:.2f}" for a in ARMS)
                  + " | B_te_op: " + " ".join(f"{a}={res[f'{k}|{a}'][-1]['B_te_op']['hit1']:.2f}" for a in ARMS)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    meta_run = {"seeds": seeds, "steps": steps, "elapsed": time.time() - t0, "smoke": args.smoke,
                "n_down": len(fr["dół"]), "n_right": len(fr["prawo"])}
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_COMP{tag}.md"
    write_report(res, meta_run, md, OUT_DIR / f"mendelejew_comp{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
