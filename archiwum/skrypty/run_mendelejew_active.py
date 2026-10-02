"""Mendelejew ACTIVE — aktywny dobór ~20 par „ta sama grupa?” (wariant Z + konfiguracja).

Protokół jak SEMI (split 70/30, Sc/Ga/Ge/Tc zawsze w teście, ocena na nigdy nieoznaczonych).
Wyrocznia: zapytanie = para pierwiastków TRENINGOWYCH → odpowiedź tak/nie (ta sama grupa).
Budżet 20 zapytań, 4 rundy po 5; po każdej rundzie model trenowany od zera na odpowiedziach:
  „tak” → InfoNCE (pozytyw), „nie” → kara relu(cos − 0.2) (jawny negatyw).
Runda 0 wybiera na surowych cechach (brak modelu).

Strategie (bez znajomości etykiet):
  losowe      — losowe pary z puli
  najbliżsi   — pary o najwyższym cos w przestrzeni modelu (eksploatacja)
  niepewność  — pary z p(ta sama) najbliżej 0.5; p = sigmoid((cos − próg)/0.05),
                próg = środek między średnim cos odpowiedzi tak / nie (bez „tak”: jak najbliżsi)
  pokrycie    — k-center: pierwiastek najdalej od już odpytanych + jego najbliższy partner
  kontrast    — (POST-HOC, dodana po porażce strategii opartych na podobieństwie)
                pary o podobnych zewnętrznych podpowłokach (τ1..4) i RÓŻNEJ treści (rdzeń):
                score = cos(τ) − cos(x), surowe cechy, bez modelu; każdy pierwiastek max raz
                — „podobna walencja, inna masa” (myśl Mendelejewa), bez etykiet
Odniesienie (ZNA etykiety przy doborze): wyrocznia — 20 par „tak” round-robin po grupach.

  python archiwum/skrypty/run_mendelejew_active.py [--smoke]
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
import torch.nn.functional as F

import run_mendelejew_ml3_zc as zc  # ustawia ml3.D_X / T_SEAL dla ZC
import run_mendelejew_ml3 as ml3
from run_mendelejew_ml3 import ARMS, encode, fmt, make_encoder, ms, noisy
from run_mendelejew_semi import GAPS, _nce, analogs, evaluate, labeled_pairs, split
from config import CANON
from import_struktura_atomow import load_rows

OUT_DIR = _LAB / "doswiadczenie" / "glify"
STRATEGIES = ["losowe", "najbliżsi", "niepewność", "pokrycie", "kontrast", "wyrocznia"]
BUDGET = 20
ROUND = 5
NEG_MARGIN = 0.2


def train(arm, content, tracer, pos, neg, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    opt = torch.optim.Adam(enc.parameters(), lr=CANON.lr)
    if pos or neg:
        na = torch.tensor([a for a, _ in neg]) if neg else None
        nb = torch.tensor([b for _, b in neg]) if neg else None
        for _ in range(steps):
            enc.train()
            h = F.normalize(encode(enc, arm, noisy(content, gen), tracer), dim=-1)
            loss = torch.zeros(())
            if pos:
                loss = loss + _nce(h, pos, gen)
            if neg:
                loss = loss + F.relu((h[na] * h[nb]).sum(-1) - NEG_MARGIN).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    with torch.no_grad():
        return F.normalize(encode(enc, arm, content, tracer), dim=-1)


def select(strategy, h, pool, asked, pos, neg, k, gen, raw=None):
    cand = [p for p in pool if p not in asked]
    if strategy == "kontrast":
        x, tau = raw
        tn = F.normalize(tau[:, 1:5], dim=-1)
        score = {p: float(tn[p[0]] @ tn[p[1]] - x[p[0]] @ x[p[1]]) for p in cand}
        used = {e for p in asked for e in p}
        out = []
        for p in sorted(cand, key=lambda q: -score[q]):
            if p[0] not in used and p[1] not in used:
                out.append(p)
                used.update(p)
            if len(out) == k:
                break
        return out
    if strategy == "losowe":
        return [cand[i] for i in torch.randperm(len(cand), generator=gen)[:k].tolist()]
    sim = {p: float(h[p[0]] @ h[p[1]]) for p in cand}
    if strategy == "najbliżsi" or (strategy == "niepewność" and not pos):
        return sorted(cand, key=lambda p: -sim[p])[:k]
    if strategy == "niepewność":
        cp = sum(float(h[a] @ h[b]) for a, b in pos) / len(pos)
        cn = sum(float(h[a] @ h[b]) for a, b in neg) / len(neg) if neg else 0.0
        thr = 0.5 * (cp + cn)
        return sorted(cand, key=lambda p: abs(torch.sigmoid(torch.tensor((sim[p] - thr) / 0.05)).item() - 0.5))[:k]
    if strategy == "pokrycie":
        out = []
        touched = {x for p in asked for x in p}
        elems = sorted({x for p in pool for x in p})
        for _ in range(k):
            if touched:
                tt = torch.tensor(sorted(touched))
                far = max(elems, key=lambda e: -float((h[e] @ h[tt].T).max()))
            else:
                far = elems[int(torch.randint(len(elems), (1,), generator=gen))]
            partners = [p for p in cand if far in p and p not in out]
            if not partners:
                touched.add(far)
                continue
            out.append(max(partners, key=lambda p: sim[p]))
            touched.update(out[-1])
        return out
    raise ValueError(strategy)


def run_strategy(strategy, arm, content, tracer, meta, tr, te, seed, steps):
    same = lambda p: meta[p[0]][0] == meta[p[1]][0]
    pool = [(a, b) for i, a in enumerate(tr) for b in tr[i + 1:]]
    curve = []
    if strategy == "wyrocznia":
        allpos = labeled_pairs(meta, tr, BUDGET, seed)
        for r in range(ROUND, BUDGET + 1, ROUND):
            h = train(arm, content, tracer, allpos[:r], [], seed, steps)
            curve.append({"asked": r, "n_pos": r, **evaluate(h, meta, te, seed)})
        groups = {meta[a][0] for a, _ in allpos}
        return curve, h, allpos, len(groups)
    gen = torch.Generator().manual_seed(20_000 + seed)
    h = F.normalize(torch.cat([content, tracer], -1), dim=-1)  # runda 0: surowe cechy
    asked, pos, neg = [], [], []
    while len(asked) < BUDGET:
        new = select(strategy, h, pool, set(asked), pos, neg, ROUND, gen, raw=(content, tracer))
        asked += new
        pos += [p for p in new if same(p)]
        neg += [p for p in new if not same(p)]
        h = train(arm, content, tracer, pos, neg, seed, steps)
        curve.append({"asked": len(asked), "n_pos": len(pos), **evaluate(h, meta, te, seed)})
    return curve, h, asked, len({meta[a][0] for a, _ in pos})


def write_report(res, meta_run, md, js):
    seeds = meta_run["seeds"]
    L = [
        "# Mendelejew ACTIVE — aktywny dobór ~20 par „ta sama grupa?”",
        "",
        "Cechy jak ZC (tylko Z + konfiguracja). Wyrocznia odpowiada tak/nie na pary pierwiastków treningowych; "
        f"budżet {BUDGET} zapytań w rundach po {ROUND}; ocena na pierwiastkach nigdy nieodpytanych. "
        f"Seedy={seeds} · kroki={meta_run['steps']} · czas={meta_run['elapsed']:.0f}s · "
        "skrypt `archiwum/skrypty/run_mendelejew_active.py`",
        "",
        f"Udział par „tak” w puli: {meta_run['pos_rate']:.1%}. Losowe hit@1 ¬f: {meta_run['random_nonf']:.3f}. "
        "„wyrocznia” zna etykiety przy doborze (20 par „tak” po grupach) — to odniesienie, nie strategia aktywna.",
        "",
        f"## Po {BUDGET} zapytaniach — hit@1 bez bloku f (główna miara)",
        "",
        "| strategia | " + " | ".join(ARMS) + " |",
        "|---|" + "---:|" * len(ARMS),
    ]

    def final(s, a, key):
        return [c[-1][key] for c in res[s][a]["curve"]]

    for s in STRATEGIES:
        L.append(f"| {s} | " + " | ".join(fmt(final(s, a, "hit1_nonf")) for a in ARMS) + " |")
    L += ["", "## Ile „tak” zebrano / ile grup pokryto (średnio)", "",
          "| strategia | " + " | ".join(f"{a}: tak / grupy" for a in ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    for s in STRATEGIES:
        L.append(f"| {s} | " + " | ".join(
            f"{ms(final(s, a, 'n_pos'))[0]:.1f} / {ms(res[s][a]['groups'])[0]:.1f}" for a in ARMS) + " |")
    L += ["", f"## NMI grupy po {BUDGET} zapytaniach", "", "| strategia | " + " | ".join(ARMS) + " |",
          "|---|" + "---:|" * len(ARMS)]
    for s in STRATEGIES:
        L.append(f"| {s} | " + " | ".join(fmt(final(s, a, "nmi_group")) for a in ARMS) + " |")
    L += ["", "## Krzywa uczenia Mostu (hit@1 ¬f po 5/10/15/20 zapytaniach)", "",
          "| strategia | 5 | 10 | 15 | 20 |", "|---|---:|---:|---:|---:|"]
    for s in STRATEGIES:
        cur = res[s]["most"]["curve"]
        L.append(f"| {s} | " + " | ".join(f"{ms([c[i]['hit1_nonf'] for c in cur])[0]:.3f}" for i in range(len(cur[0]))) + " |")
    L += ["", "## Najlepsza strategia aktywna vs losowe (hit@1 ¬f, wygrane seedy)", "",
          "| arm | " + " | ".join(s for s in STRATEGIES if s != "losowe") + " |", "|---|" + "---:|" * (len(STRATEGIES) - 1)]
    for a in ARMS:
        base = final("losowe", a, "hit1_nonf")
        cells = []
        for s in STRATEGIES:
            if s == "losowe":
                continue
            v = final(s, a, "hit1_nonf")
            cells.append(f"{sum(x > y + 1e-9 for x, y in zip(v, base))}/{len(seeds)} ({ms(v)[0] - ms(base)[0]:+.3f})")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L += ["", "## Most vs reszta przy tej samej strategii (hit@1 ¬f, wygrane seedy)", "",
          "| strategia | vs softmax | vs zero | vs mlp |", "|---|---:|---:|---:|"]
    for s in STRATEGIES:
        m = final(s, "most", "hit1_nonf")
        L.append(f"| {s} | " + " | ".join(
            f"{sum(x > y + 1e-9 for x, y in zip(m, final(s, o, 'hit1_nonf')))}/{len(seeds)}" for o in ("softmax", "zero", "mlp")) + " |")
    L += ["", "## Odpytane pary „tak” — Most, seed 0", "", "| strategia | pary „tak” (grupa) |", "|---|---|"]
    for s in STRATEGIES:
        L.append(f"| {s} | {res[s]['most']['pairs_seed0'] or '—'} |")
    L += ["", "## Analogi luk (Most, seed 0, 4-NN po 20 zapytaniach)", "",
          "Oczekiwane: Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re", "",
          "| strategia | Sc | Ga | Ge | Tc |", "|---|---|---|---|---|"]
    for s in STRATEGIES:
        an = res[s]["most"]["analogs_seed0"]
        L.append(f"| {s} | " + " | ".join(", ".join(an[g]) for g in GAPS) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta_run, "results": res}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None, help="np. kontrast — dolicz tylko te strategie i scal z istniejącym JSON")
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = 60 if args.smoke else 600

    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    syms = [m[3] for m in meta]
    t0 = time.time()
    tag = "_smoke" if args.smoke else ""
    js = OUT_DIR / f"mendelejew_active{tag}_results.json"
    run_s = args.only.split(",") if args.only else STRATEGIES
    prev = json.loads(js.read_text(encoding="utf-8")) if args.only else None
    res = prev["results"] if prev else {}
    for s in run_s:
        res[s] = {a: {"curve": [], "groups": [], "pairs_seed0": "", "analogs_seed0": {}} for a in ARMS}
    pos_rate, rnd_nf = [], []
    for seed in seeds:
        tr, te = split(meta, seed)
        pool = [(a, b) for i, a in enumerate(tr) for b in tr[i + 1:]]
        pos_rate.append(sum(meta[a][0] == meta[b][0] for a, b in pool) / len(pool))
        from run_mendelejew_semi import random_hit
        rnd_nf.append(random_hit(meta, te, True))
        for s in run_s:
            for a in ARMS:
                curve, h, asked, n_groups = run_strategy(s, a, content, tracer, meta, tr, te, seed, steps)
                r = res[s][a]
                r["curve"].append(curve)
                r["groups"].append(n_groups)
                if seed == seeds[0]:
                    yes = [p for p in asked if meta[p[0]][0] == meta[p[1]][0]]
                    r["pairs_seed0"] = ", ".join(f"{syms[x]}–{syms[y]} ({meta[x][0]})" for x, y in yes)
                    r["analogs_seed0"] = analogs(h, meta)
            print(f"  seed={seed} {s:10s} ¬f@20: " + " ".join(
                f"{a}={res[s][a]['curve'][-1][-1]['hit1_nonf']:.2f}(tak={res[s][a]['curve'][-1][-1]['n_pos']})" for a in ARMS)
                + f" ({time.time() - t0:.0f}s)", flush=True)
    elapsed = time.time() - t0 + (prev["meta"]["elapsed"] if prev else 0)
    meta_run = {"seeds": seeds, "steps": steps, "elapsed": elapsed, "smoke": args.smoke,
                "pos_rate": sum(pos_rate) / len(pos_rate), "random_nonf": sum(rnd_nf) / len(rnd_nf)}
    md = OUT_DIR / f"MENDELEJEW_ACTIVE{tag}.md"
    write_report(res, meta_run, md, js)
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
