"""Mendelejew SEMI — częściowo nadzorowane: kilka par „ta sama grupa” + reszta bez etykiet.

Cechy jak w wariancie ZC (tylko Z + konfiguracja; okres/grupa/blok nie wchodzą do modelu).
Split pierwiastków 70/30; Sc, Ga, Ge, Tc zawsze w teście (luki Mendelejewa).

Etykiety: K par (a, b) z tej samej grupy, losowane TYLKO spośród pierwiastków treningowych.
Reżimy:
  sup        — tylko K par
  semi_knn   — K par + pary bez etykiet: 3-NN w surowych cechach (jak ZC; ~90% poziome)
  semi_pl    — K par + pseudo-pary: wzajemni 1-NN we własnej przestrzeni modelu,
               odświeżane co 100 kroków po rozgrzewce na K parach (self-training)
Budżety K: 0 (tylko bez etykiet), 6, 18, 54, all (sufit: wszystkie pary w treningu).
Pary K rozkładane round-robin po grupach; batch losowany po pierwiastkach.

Ocena (pierwiastki testowe = nigdy nieoznaczone):
  hit@1      — najbliższy (cos) spośród pozostałych 117 jest z tej samej grupy
  hit@1 ¬f   — j.w. bez zapytań z bloku f (grupa 3 = 32 lantanowce/aktynowce, łatwe trafienia)
  NMI grupy  — k-means (k=18) na embeddingach wszystkich 118
  analogi luk Sc/Ga/Ge/Tc (seed 0, K=18)

  python archiwum/skrypty/run_mendelejew_semi.py [--smoke]
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
from config import CANON
from export_mendelejew_full import _block
from import_struktura_atomow import load_rows

OUT_DIR = _LAB / "doswiadczenie" / "glify"
GAPS = list(zc.HELD_OUT)
BUDGETS = [0, 6, 18, 54, "all"]
REGIMES = ["sup", "semi_knn", "semi_pl"]
WARMUP = 150
PL_EVERY = 100


def split(meta, seed):
    gen = torch.Generator().manual_seed(seed)
    syms = [m[3] for m in meta]
    rest = [i for i, s in enumerate(syms) if s not in GAPS]
    rest = [rest[i] for i in torch.randperm(len(rest), generator=gen).tolist()]
    n_test = round(0.3 * len(meta)) - len(GAPS)
    test = sorted(rest[:n_test] + [syms.index(s) for s in GAPS])
    train = sorted(rest[n_test:])
    return train, test


def labeled_pairs(meta, train, K, seed):
    """K par rozłożonych po grupach (round-robin), nie równomiernie po parach —
    inaczej ~75% to pary lantanowców/aktynowców z grupy 3 (32 członków)."""
    pairs = [(a, b) for i, a in enumerate(train) for b in train[i + 1:] if meta[a][0] == meta[b][0]]
    if K == "all" or K >= len(pairs):
        return pairs
    gen = torch.Generator().manual_seed(10_000 + seed)
    by_g: dict[int, list] = {}
    for pr in pairs:
        by_g.setdefault(meta[pr[0]][0], []).append(pr)
    for g in by_g:
        by_g[g] = [by_g[g][i] for i in torch.randperm(len(by_g[g]), generator=gen).tolist()]
    order = [sorted(by_g)[i] for i in torch.randperm(len(by_g), generator=gen).tolist()]
    out = []
    while len(out) < K:
        for g in order:
            if by_g[g] and len(out) < K:
                out.append(by_g[g].pop())
    return out


def _both(pairs):
    return [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]


def _nce(h, pairs, gen, bs=32):
    """Batch losowany po pierwiastkach (nie po parach): zapytanie ~ U(węzły z partnerami),
    pozytyw ~ U(jego partnerzy) — duże grupy nie dominują straty."""
    adj: dict[int, list[int]] = {}
    for a, b in _both(pairs):
        adj.setdefault(a, []).append(b)
    nodes = sorted(adj)
    qi = torch.randint(len(nodes), (bs,), generator=gen).tolist()
    q = torch.tensor([nodes[i] for i in qi])
    p = torch.tensor([adj[nodes[i]][int(torch.randint(len(adj[nodes[i]]), (1,), generator=gen))] for i in qi])
    s = (h[q] @ h.T / ml3.TEMP).masked_fill(F.one_hot(q, h.shape[0]).bool(), -1e9)
    return F.cross_entropy(s, p)


@torch.no_grad()
def mutual_nn(h):
    s = (h @ h.T).fill_diagonal_(-2.0)
    nn1 = s.argmax(-1)
    return [(i, int(j)) for i, j in enumerate(nn1.tolist()) if int(nn1[j]) == i and i < j]


def train(arm, regime, content, tracer, lab, knn_pairs, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    opt = torch.optim.Adam(enc.parameters(), lr=CANON.lr)
    pl: list = []
    n_pl = []
    for step in range(steps):
        enc.train()
        h = F.normalize(encode(enc, arm, noisy(content, gen), tracer), dim=-1)
        loss = torch.zeros(())
        if lab:
            loss = loss + _nce(h, lab, gen)
        if regime == "semi_knn":
            loss = loss + _nce(h, knn_pairs, gen)
        elif regime == "semi_pl" and step >= WARMUP:
            if (step - WARMUP) % PL_EVERY == 0:
                enc.eval()
                pl = mutual_nn(F.normalize(encode(enc, arm, content, tracer), dim=-1))
                n_pl.append(len(pl))
                enc.train()
            if pl:
                loss = loss + _nce(h, pl, gen)
        if not loss.requires_grad:
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        h = F.normalize(encode(enc, arm, content, tracer), dim=-1)
    return h, n_pl


def evaluate(h, meta, test, seed):
    s = (h @ h.T).fill_diagonal_(-2.0)
    nn1 = s.argmax(-1)
    grp = torch.tensor([m[0] for m in meta])
    hit = [(grp[q] == grp[nn1[q]]).item() for q in test]
    nonf = [q for q in test if _block(*meta[q][:3]) != "f"]
    hit_nf = [(grp[q] == grp[nn1[q]]).item() for q in nonf]
    lab = ml3.kmeans(h, 18, torch.Generator().manual_seed(seed))
    return {"hit1": sum(hit) / len(hit), "hit1_nonf": sum(hit_nf) / len(hit_nf),
            "nmi_group": ml3.nmi(lab, grp)}


def random_hit(meta, test, nonf_only=False):
    qs = [q for q in test if not nonf_only or _block(*meta[q][:3]) != "f"]
    return sum((sum(m[0] == meta[q][0] for m in meta) - 1) / (len(meta) - 1) for q in qs) / len(qs)


def analogs(h, meta, k=4):
    s = (h @ h.T).fill_diagonal_(-2.0)
    syms = [m[3] for m in meta]
    out = {}
    for g in GAPS:
        i = syms.index(g)
        out[g] = [syms[j] for j in s[i].topk(k).indices.tolist()]
    return out


def write_report(res, meta_run, md, js):
    seeds = meta_run["seeds"]
    L = [
        "# Mendelejew SEMI — kilka par „ta sama grupa” + reszta bez etykiet",
        "",
        "Cechy jak w ZC: tylko Z + konfiguracja elektronowa (okres/grupa/blok nie wchodzą do modelu). "
        f"Split pierwiastków 70/30; luki Sc/Ga/Ge/Tc zawsze w teście. Seedy={seeds} · kroki={meta_run['steps']} · "
        f"czas={meta_run['elapsed']:.0f}s · skrypt `archiwum/skrypty/run_mendelejew_semi.py`",
        "",
        f"Pary bez etykiet (semi_knn): {meta_run['n_knn']} par 3-NN w surowych cechach. "
        f"Wszystkich par grupowych w treningu (K=all): ~{meta_run['n_all']:.0f}. "
        f"Losowe hit@1: {meta_run['random']:.3f} (¬f: {meta_run['random_nonf']:.3f}).",
        "",
        "Bez uczenia: surowe cechy hit@1 = "
        f"{fmt(res['raw']['hit1'])}, ¬f = {fmt(res['raw']['hit1_nonf'])}, NMI grupy = {fmt(res['raw']['nmi_group'])}.",
        "",
    ]
    for metric, title in (("hit1_nonf", "hit@1 bez bloku f (główna miara)"), ("hit1", "hit@1 wszystkie zapytania testowe"),
                          ("nmi_group", "NMI grupy (k-means, wszystkie 118)")):
        L += [f"## {title}", "", "| K par | reżim | " + " | ".join(ARMS) + " |", "|---|---|" + "---:|" * len(ARMS)]
        for K in BUDGETS:
            for rg in REGIMES:
                key = f"{K}|{rg}"
                if key not in res["runs"]:
                    continue
                L.append(f"| {K} | {rg} | " + " | ".join(fmt([x[metric] for x in res["runs"][key][a]]) for a in ARMS) + " |")
        L.append("")
    L += ["## Most vs reszta (hit@1 ¬f, ile seedów Most wygrywa)", "",
          "| K | reżim | vs softmax | vs zero | vs mlp |", "|---|---|---:|---:|---:|"]
    for K in BUDGETS:
        for rg in REGIMES:
            key = f"{K}|{rg}"
            if key not in res["runs"]:
                continue
            r = res["runs"][key]
            m = [x["hit1_nonf"] for x in r["most"]]
            cells = []
            for o in ("softmax", "zero", "mlp"):
                s = [x["hit1_nonf"] for x in r[o]]
                cells.append(f"{sum(a > b + 1e-9 for a, b in zip(m, s))}/{len(seeds)} ({ms(m)[0] - ms(s)[0]:+.3f})")
            L.append(f"| {K} | {rg} | " + " | ".join(cells) + " |")
    L += ["", "## Czy bez etykiet pomaga? (Δ hit@1 ¬f względem sup przy tym samym K, średnia po armiach i seedach)", "",
          "| K | semi_knn − sup | semi_pl − sup |", "|---|---:|---:|"]
    for K in BUDGETS:
        if f"{K}|sup" not in res["runs"]:
            continue
        base = [x["hit1_nonf"] for a in ARMS for x in res["runs"][f"{K}|sup"][a]]
        row = []
        for rg in ("semi_knn", "semi_pl"):
            v = [x["hit1_nonf"] for a in ARMS for x in res["runs"][f"{K}|{rg}"][a]]
            row.append(f"{ms(v)[0] - ms(base)[0]:+.3f}")
        L.append(f"| {K} | " + " | ".join(row) + " |")
    L += ["", "## Analogi luk Mendelejewa (4-NN, seed 0, K=18 ≈ jedna para na grupę)", "",
          "Oczekiwane (ta sama grupa): Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re", "",
          "| reżim | arm | Sc | Ga | Ge | Tc |", "|---|---|---|---|---|---|"]
    L.append("| surowe | — | " + " | ".join(", ".join(res["analogs_raw"][g]) for g in GAPS) + " |")
    for rg in REGIMES:
        for a in ARMS:
            an = res["analogs"].get(f"{rg}|{a}")
            if an:
                L.append(f"| {rg} | {a} | " + " | ".join(", ".join(an[g]) for g in GAPS) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    js.write_text(json.dumps({"meta": meta_run, "results": res}, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else [0, 1, 2, 3, 4]
    steps = 60 if args.smoke else 600
    budgets = [0, 12] if args.smoke else BUDGETS

    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    knn_pairs = sorted({(min(a, b), max(a, b)) for a, ps in zc.knn_partners(content, tracer).items() for b in ps})
    t0 = time.time()
    res = {"runs": {}, "raw": {"hit1": [], "hit1_nonf": [], "nmi_group": []}, "analogs": {}}
    h_raw = F.normalize(torch.cat([content, tracer], -1), dim=-1)
    res["analogs_raw"] = analogs(h_raw, meta)
    n_all, rnd, rnd_nf = [], [], []
    for seed in seeds:
        tr, te = split(meta, seed)
        n_all.append(len(labeled_pairs(meta, tr, "all", seed)))
        rnd.append(random_hit(meta, te))
        rnd_nf.append(random_hit(meta, te, True))
        ev = evaluate(h_raw, meta, te, seed)
        for k in ev:
            res["raw"][k].append(ev[k])
        for K in budgets:
            lab = labeled_pairs(meta, tr, K, seed) if K != 0 else []
            for rg in REGIMES:
                if K == 0 and rg != "semi_knn":
                    continue  # bez etykiet sensowny jest tylko samonadzór kNN
                key = f"{K}|{rg}"
                res["runs"].setdefault(key, {a: [] for a in ARMS})
                for a in ARMS:
                    h, n_pl = train(a, rg, content, tracer, lab, knn_pairs, seed, steps)
                    ev = evaluate(h, meta, te, seed)
                    ev["n_pl"] = n_pl
                    res["runs"][key][a].append(ev)
                    if seed == seeds[0] and K == (18 if not args.smoke else 12):
                        res["analogs"][f"{rg}|{a}"] = analogs(h, meta)
                print(f"  seed={seed} K={K:>3} {rg:8s} ¬f: " + " ".join(
                    f"{a}={res['runs'][key][a][-1]['hit1_nonf']:.2f}" for a in ARMS)
                    + f" ({time.time() - t0:.0f}s)", flush=True)
    meta_run = {"seeds": seeds, "steps": steps, "elapsed": time.time() - t0, "n_knn": len(knn_pairs),
                "n_all": sum(n_all) / len(n_all), "random": sum(rnd) / len(rnd),
                "random_nonf": sum(rnd_nf) / len(rnd_nf), "smoke": args.smoke}
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_SEMI{tag}.md"
    write_report(res, meta_run, md, OUT_DIR / f"mendelejew_semi{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
