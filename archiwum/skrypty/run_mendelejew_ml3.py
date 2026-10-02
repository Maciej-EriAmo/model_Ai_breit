"""Mendelejew ML3 — nadzorowane / nienadzorowane / ze wzmocnieniem na małym secie.

Set: archiwum/dane/struktura_atomow.csv (H He Li C N O Na Fe) → graf Φ, 72 węzły
(8 jąder + 64 elektrony-orbitale). Cały graf = jedna sekwencja dla enkodera.

Armie (te same w każdym torze):
  most    — PhiBridgeEncoder (Most-1 Breita–Wignera, kanon)
  softmax — PhiSoftmaxEncoder (twin, tracer ignorowany)
  zero    — Most z wyzerowanym tracerem (ΔE=0)
  mlp     — per-węzeł MLP na [content ⊕ tracer], bez uwagi (baseline)

Tory:
  SL  — klasyfikacja węzłów (transduktywnie, split węzłów 70/30):
        element (8 kl.), powłoka n (1–4), podpowłoka ℓ (s/p/d)
  UL  — k-means na surowych cechach i na embeddingach po samonadzorze
        (InfoNCE na krawędziach bind, bez etykiet) → NMI / ARI
  RL  — kontekstowy bandyta: stan = węzeł-zapytanie, akcja = wybór węzła,
        nagroda 1 gdy wybrany jest partnerem bind; REINFORCE; ocena greedy
        hit@1 na zapytaniach wstrzymanych (30%)

  python archiwum/skrypty/run_mendelejew_ml3.py
  python archiwum/skrypty/run_mendelejew_ml3.py --smoke   # 1 seed, krótko
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import _graph_clean_tensors, bind_pairs_idx
from import_struktura_atomow import DEFAULT_CSV, load_rows, rows_to_graph
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"

ARMS = ("most", "softmax", "zero", "mlp")
SEEDS_FULL = [0, 1, 2, 3, 4]
NOISE = 0.02
TEMP = 0.1
# warianty mogą nadpisać (np. ZC: treść = obsadzenie podpowłok, brak kanału T)
D_X = CANON.d_x
T_SEAL: bool | None = None


# ---------------------------------------------------------------- dane


def load_graph():
    g = rows_to_graph(load_rows(DEFAULT_CSV))
    content, tracer, _, _ = _graph_clean_tensors(g, CANON.d_x, CANON.n_dim)
    symbols = g.meta["symbols"]
    elem, shell, sub = [], [], []
    for a in g.atoms:
        sym = a.atom_id.split("/")[0]
        elem.append(symbols.index(sym))
        m = re.search(r"/n(\d+)/l(\d+)/", a.S)
        shell.append(int(m.group(1)) - 1 if m else -1)
        sub.append(int(m.group(2)) if m else -1)
    labels = {
        "element": torch.tensor(elem),
        "n": torch.tensor(shell),
        "l": torch.tensor(sub),
    }
    partners: dict[int, set[int]] = {}
    for a, b in bind_pairs_idx(g):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return g, content, tracer, labels, partners


def noisy(content: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    x = content + NOISE * torch.randn(content.shape, generator=gen)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------- modele


class NodeMLP(nn.Module):
    """Baseline bez uwagi: każdy węzeł osobno z [content ⊕ tracer]."""

    def __init__(self, d_x: int, n_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + n_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, x, tracer, mask=None):
        return self.net(torch.cat([x, tracer], dim=-1)), None


def make_encoder(arm: str) -> nn.Module:
    kw = dict(
        input_dim=D_X,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
    )
    if arm in ("most", "zero"):
        return PhiBridgeEncoder(coupling=CANON.coupling, t_seal=T_SEAL, **kw)
    if arm == "softmax":
        return PhiSoftmaxEncoder(**kw)
    if arm == "mlp":
        return NodeMLP(D_X, CANON.n_dim, CANON.d_model)
    raise ValueError(arm)


def encode(model, arm, content, tracer) -> torch.Tensor:
    tr = torch.zeros_like(tracer) if arm == "zero" else tracer
    tok, _ = model(content.unsqueeze(0), tr.unsqueeze(0))
    return tok[0]  # [N,d]


# ---------------------------------------------------------------- metryki


def stratified_split(y: torch.Tensor, idx: torch.Tensor, frac: float, gen):
    tr, te = [], []
    for c in y[idx].unique().tolist():
        members = idx[y[idx] == c]
        members = members[torch.randperm(len(members), generator=gen)]
        k = int(round(frac * len(members))) if len(members) > 1 else 0
        te += members[:k].tolist()
        tr += members[k:].tolist()
    return torch.tensor(sorted(tr)), torch.tensor(sorted(te))


def kmeans(x: torch.Tensor, k: int, gen, restarts: int = 10, iters: int = 50):
    best, best_in = None, float("inf")
    for _ in range(restarts):
        # k-means++
        c = [x[torch.randint(len(x), (1,), generator=gen)].squeeze(0)]
        for _ in range(1, k):
            d2 = torch.cdist(x, torch.stack(c)).min(dim=1).values.square()
            p = d2 / d2.sum().clamp_min(1e-12)
            c.append(x[torch.multinomial(p, 1, generator=gen)].squeeze(0))
        cent = torch.stack(c)
        for _ in range(iters):
            lab = torch.cdist(x, cent).argmin(dim=1)
            new = torch.stack(
                [x[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)]
            )
            if torch.allclose(new, cent):
                break
            cent = new
        inertia = torch.cdist(x, cent).min(dim=1).values.square().sum().item()
        if inertia < best_in:
            best_in, best = inertia, lab
    return best


def _contingency(a, b):
    ua, ia = a.unique(return_inverse=True)
    ub, ib = b.unique(return_inverse=True)
    m = torch.zeros(len(ua), len(ub), dtype=torch.float64)
    m.index_put_((ia, ib), torch.ones(len(a), dtype=torch.float64), accumulate=True)
    return m


def nmi(a, b) -> float:
    m = _contingency(a, b)
    n = m.sum()
    pa, pb, pab = m.sum(1) / n, m.sum(0) / n, m / n
    nz = pab > 0
    mi = (pab[nz] * (pab[nz] / (pa[:, None] * pb[None, :])[nz]).log()).sum()
    ha = -(pa[pa > 0] * pa[pa > 0].log()).sum()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum()
    den = 0.5 * (ha + hb)
    return float(mi / den) if den > 0 else 1.0


def ari(a, b) -> float:
    m = _contingency(a, b)
    c2 = lambda t: t * (t - 1) / 2
    n = m.sum()
    sij, sa, sb = c2(m).sum(), c2(m.sum(1)).sum(), c2(m.sum(0)).sum()
    exp = sa * sb / c2(n)
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0


def macro_f1(pred, y) -> float:
    f = []
    for c in y.unique():
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(f) / len(f)


# ---------------------------------------------------------------- SL


def run_sl(arm, task, content, tracer, labels, seed, steps):
    y = labels[task]
    idx = torch.nonzero(y >= 0).squeeze(1)
    gen = torch.Generator().manual_seed(seed)
    tr, te = stratified_split(y, idx, 0.3, gen)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    head = nn.Linear(CANON.d_model, int(y.max()) + 1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=CANON.lr * 3)
    for _ in range(steps):
        enc.train()
        logits = head(encode(enc, arm, noisy(content, gen), tracer))
        loss = F.cross_entropy(logits[tr], y[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        pred = head(encode(enc, arm, content, tracer)).argmax(-1)
    maj = y[tr].mode().values
    return {
        "test_acc": (pred[te] == y[te]).float().mean().item(),
        "test_f1": macro_f1(pred[te], y[te]),
        "train_acc": (pred[tr] == y[tr]).float().mean().item(),
        "majority_acc": (y[te] == maj).float().mean().item(),
        "n_train": len(tr),
        "n_test": len(te),
    }


# ---------------------------------------------------------------- UL


def cluster_scores(emb, labels, seed):
    out = {}
    gen = torch.Generator().manual_seed(seed)
    for task, y in labels.items():
        idx = torch.nonzero(y >= 0).squeeze(1)
        k = len(y[idx].unique())
        lab = kmeans(emb[idx], k, gen)
        out[task] = {"nmi": nmi(lab, y[idx]), "ari": ari(lab, y[idx])}
    return out


def run_ul(arm, content, tracer, labels, partners, seed, steps):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    with torch.no_grad():
        enc.eval()
        e0 = F.normalize(encode(enc, arm, content, tracer), dim=-1)
    untrained = cluster_scores(e0, labels, seed)

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
        s = h[q] @ h.T / TEMP
        s = s.masked_fill(F.one_hot(q, N).bool(), -1e9)
        loss = F.cross_entropy(s, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        e1 = F.normalize(encode(enc, arm, content, tracer), dim=-1)
    return {"untrained": untrained, "ssl": cluster_scores(e1, labels, seed)}


def run_ul_raw(content, tracer, labels, seed):
    return {
        "content": cluster_scores(content, labels, seed),
        "tracer": cluster_scores(tracer, labels, seed),
        "concat": cluster_scores(torch.cat([content, tracer], -1), labels, seed),
    }


# ---------------------------------------------------------------- RL


def _policy_scores(enc, arm, x, tracer, q):
    h = F.normalize(encode(enc, arm, x, tracer), dim=-1)
    s = h[q] @ h.T / TEMP
    return s.masked_fill(F.one_hot(q, x.shape[0]).bool(), -1e9)


def _hit_greedy(enc, arm, content, tracer, qs, P):
    enc.eval()
    with torch.no_grad():
        a = _policy_scores(enc, arm, content, tracer, qs).argmax(-1)
    return P[qs, a].float().mean().item()


def run_rl(arm, content, tracer, partners, seed, steps, mode="reinforce"):
    """mode=reinforce: tylko skalarna nagroda; mode=sl_ref: CE z pełną wiedzą o partnerach."""
    N = content.shape[0]
    P = torch.zeros(N, N, dtype=torch.bool)
    for a, ps in partners.items():
        P[a, list(ps)] = True
    gen = torch.Generator().manual_seed(seed)
    qs = torch.tensor(sorted(partners))
    qs = qs[torch.randperm(len(qs), generator=gen)]
    k = int(round(0.3 * len(qs)))
    q_te, q_tr = qs[:k], qs[k:]
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    opt = torch.optim.Adam(enc.parameters(), lr=CANON.lr)
    baseline = 0.0
    rewards = []
    for step in range(steps):
        enc.train()
        q = q_tr[torch.randint(len(q_tr), (32,), generator=gen)]
        s = _policy_scores(enc, arm, noisy(content, gen), tracer, q)
        logp = F.log_softmax(s, dim=-1)
        if mode == "reinforce":
            a = torch.multinomial(logp.exp().detach(), 1, generator=gen).squeeze(1)
            r = P[q, a].float()
            adv = r - baseline
            ent = -(logp.exp() * logp.clamp_min(-1e4)).sum(-1).mean()
            loss = -(adv * logp[torch.arange(len(q)), a]).mean() - 0.01 * ent
            baseline = 0.9 * baseline + 0.1 * r.mean().item()
            rewards.append(r.mean().item())
        else:
            # multi-pozytyw: -log Σ_{a∈partnerzy} π(a|q)
            loss = -(logp.masked_fill(~P[q], -1e9).logsumexp(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = {
        "test_hit1": _hit_greedy(enc, arm, content, tracer, q_te, P),
        "train_hit1": _hit_greedy(enc, arm, content, tracer, q_tr, P),
        "random_hit1": (P[q_te].sum(1).float() / (N - 1)).mean().item(),
        "n_q_train": len(q_tr),
        "n_q_test": len(q_te),
    }
    if rewards:
        w = max(1, len(rewards) // 10)
        out["reward_curve"] = [
            sum(rewards[i : i + w]) / len(rewards[i : i + w]) for i in range(0, len(rewards), w)
        ]
    return out


# ---------------------------------------------------------------- raport


def ms(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
    return m, s


def fmt(v):
    m, s = ms(v)
    return f"{m:.3f} ± {s:.3f}"


def wins(a, b):
    return sum(1 for x, y in zip(a, b) if x > y + 1e-9)


def write_report(res, meta, path_md, path_json):
    seeds = meta["seeds"]
    L = [
        "# Mendelejew ML3 — nadzorowane · nienadzorowane · ze wzmocnieniem",
        "",
        f"Set: `archiwum/dane/struktura_atomow.csv` · pierwiastki={meta['symbols']} · "
        f"węzły Φ={meta['N']} (8 jąder + {meta['N'] - 8} elektronów) · krawędzie bind={meta['n_bind']}",
        f"Seedy={seeds} · kroki SL={meta['steps_sl']} UL={meta['steps_ul']} RL={meta['steps_rl']} · "
        f"czas={meta['elapsed']:.0f}s · skrypt `archiwum/skrypty/run_mendelejew_ml3.py`",
        "",
        "Armie: **most** (Most-1 Breita–Wignera) · **softmax** (twin) · **zero** (Most, tracer=0) · "
        "**mlp** (per-węzeł, bez uwagi). Wartości: średnia ± odch. std. po seedach.",
        "",
        "## 1. Nadzorowane — klasyfikacja węzłów (test = 30% wstrzymanych węzłów)",
        "",
        "| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, ncls in (("element", 8), ("n", 4), ("l", 3)):
        r = res["sl"][task]
        maj = [x["majority_acc"] for x in r["most"]]
        accs = {a: [x["test_acc"] for x in r[a]] for a in ARMS}
        L.append(
            f"| {task} | {ncls} | {ms(maj)[0]:.3f} | "
            + " | ".join(fmt(accs[a]) for a in ARMS)
            + f" | {wins(accs['most'], accs['softmax'])}/{len(seeds)} |"
        )
    L += ["", "Macro-F1 (test):", "", "| zadanie | most | softmax | zero | mlp |", "|---|---:|---:|---:|---:|"]
    for task in ("element", "n", "l"):
        r = res["sl"][task]
        L.append(f"| {task} | " + " | ".join(fmt([x["test_f1"] for x in r[a]]) for a in ARMS) + " |")
    L += ["", "Train acc (czy model w ogóle dopasował zbiór uczący):", "",
          "| zadanie | most | softmax | zero | mlp |", "|---|---:|---:|---:|---:|"]
    for task in ("element", "n", "l"):
        r = res["sl"][task]
        L.append(f"| {task} | " + " | ".join(fmt([x["train_acc"] for x in r[a]]) for a in ARMS) + " |")

    L += [
        "",
        "## 2. Nienadzorowane — k-means (k = liczba klas) → NMI",
        "",
        "### 2a. Surowe cechy (bez uczenia)",
        "",
        "| cechy | NMI element | NMI n | NMI ℓ | ARI element | ARI n | ARI ℓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for feat in ("content", "tracer", "concat"):
        rr = res["ul_raw"]
        L.append(
            f"| {feat} | "
            + " | ".join(fmt([x[feat][t]["nmi"] for x in rr]) for t in ("element", "n", "l"))
            + " | "
            + " | ".join(fmt([x[feat][t]["ari"] for x in rr]) for t in ("element", "n", "l"))
            + " |"
        )
    L += [
        "",
        "### 2b. Embeddingi enkodera: losowa inicjalizacja vs po samonadzorze (InfoNCE na krawędziach bind)",
        "",
        "| arm | stan | NMI element | NMI n | NMI ℓ | ARI element | ARI n | ARI ℓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in ARMS:
        for st in ("untrained", "ssl"):
            rr = res["ul"][a]
            L.append(
                f"| {a} | {st} | "
                + " | ".join(fmt([x[st][t]["nmi"] for x in rr]) for t in ("element", "n", "l"))
                + " | "
                + " | ".join(fmt([x[st][t]["ari"] for x in rr]) for t in ("element", "n", "l"))
                + " |"
            )

    L += [
        "",
        "## 3. Ze wzmocnieniem — kontekstowy bandyta (REINFORCE)",
        "",
        "Stan = węzeł-zapytanie; akcja = wybór jednego z pozostałych węzłów; nagroda 1, gdy to partner bind.",
        "Polityka = softmax(cos(h_q, h_a)/0.1) z enkodera. Ocena: greedy hit@1 na 30% zapytań wstrzymanych.",
        "`sl_ref` = ten sam enkoder uczony nadzorem (CE po zbiorze partnerów) — górna referencja.",
        "",
        "| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |",
        "|---|---:|---:|---:|---:|",
    ]
    for a in ARMS:
        rl, ref = res["rl"][a], res["rl_ref"][a]
        L.append(
            f"| {a} | {fmt([x['test_hit1'] for x in rl])} | {fmt([x['train_hit1'] for x in rl])} | "
            f"{fmt([x['test_hit1'] for x in ref])} | {ms([x['random_hit1'] for x in rl])[0]:.3f} |"
        )
    rl_m = [x["test_hit1"] for x in res["rl"]["most"]]
    rl_s = [x["test_hit1"] for x in res["rl"]["softmax"]]
    L += ["", f"Most > Softmax (RL test) na {wins(rl_m, rl_s)}/{len(seeds)} seedach.", ""]
    L += ["Krzywa nagrody treningowej (seed 0, 10 okien):", "", "| arm | nagroda |", "|---|---|"]
    for a in ARMS:
        c = res["rl"][a][0].get("reward_curve", [])
        L.append(f"| {a} | " + " → ".join(f"{v:.2f}" for v in c) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    path_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    path_json.write_text(json.dumps({"meta": meta, "results": res}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args(argv)
    seeds = [0] if args.smoke else (
        [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS_FULL
    )
    steps_sl, steps_ul, steps_rl = (40, 40, 60) if args.smoke else (400, 400, 600)

    g, content, tracer, labels, partners = load_graph()
    N = content.shape[0]
    print(f"graf: N={N} bind_nodes={len(partners)} symbols={g.meta['symbols']}", flush=True)
    t0 = time.time()
    res = {"sl": {t: {a: [] for a in ARMS} for t in labels}, "ul": {a: [] for a in ARMS},
           "ul_raw": [], "rl": {a: [] for a in ARMS}, "rl_ref": {a: [] for a in ARMS}}
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        res["ul_raw"].append(run_ul_raw(content, tracer, labels, seed))
        for a in ARMS:
            for t in labels:
                res["sl"][t][a].append(run_sl(a, t, content, tracer, labels, seed, steps_sl))
            res["ul"][a].append(run_ul(a, content, tracer, labels, partners, seed, steps_ul))
            res["rl"][a].append(run_rl(a, content, tracer, partners, seed, steps_rl))
            res["rl_ref"][a].append(run_rl(a, content, tracer, partners, seed, steps_rl, "sl_ref"))
            sl = " ".join(f"{t}={res['sl'][t][a][-1]['test_acc']:.2f}" for t in labels)
            ul = res["ul"][a][-1]["ssl"]
            print(
                f"  {a:8s} SL[{sl}] UL-NMI[el={ul['element']['nmi']:.2f} n={ul['n']['nmi']:.2f} "
                f"l={ul['l']['nmi']:.2f}] RL={res['rl'][a][-1]['test_hit1']:.2f} "
                f"ref={res['rl_ref'][a][-1]['test_hit1']:.2f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    meta = {
        "seeds": seeds, "N": N, "symbols": g.meta["symbols"],
        "n_bind": sum(len(p) for p in partners.values()) // 2,
        "steps_sl": steps_sl, "steps_ul": steps_ul, "steps_rl": steps_rl,
        "elapsed": time.time() - t0, "smoke": args.smoke,
    }
    tag = "_smoke" if args.smoke else ""
    md = OUT_DIR / f"MENDELEJEW_ML3{tag}.md"
    write_report(res, meta, md, OUT_DIR / f"mendelejew_ml3{tag}_results.json")
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
