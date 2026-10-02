"""Mendelejew COMPREL FROZEN — zamrożony protokół, 20 przebiegów, statystyka parami.

1. Specyfikacja (`doswiadczenie/glify/comprel_frozen_spec.json`, SHA-256 w każdym rekordzie):
   dla przebiegu i = 0..19: podział pierwiastków (symbole; ten sam generator co SEMI → przebiegi
   0–4 = dotychczasowe), fakty treningowe „dół”/„prawo”, pary grupowe (K=all) i Z, oraz
   WSZYSTKIE zapytania testowe jako (łańcuch, głowa, cel) w symbolach + pytania Mendelejewa.
   Istniejąca specyfikacja jest wczytywana, nigdy nadpisywana (--freeze tylko gdy jej brak).
2. Warianty na identycznych zapytaniach: comp · comp+rel · comp+rel·PoE · comp+mix
   (comp+rel i PoE = ten sam wytrenowany model). Armie: most, softmax, zero, mlp.
3. Każde uruchomienie (przebieg × tryb treningu × arm) = jedna linia w
   `doswiadczenie/glify/comprel_frozen_runs.jsonl` (dopisywanie; pozycja KAŻDEGO zapytania).
   Ponowne uruchomienie pomija gotowe linie (wznawianie).
4. Raport `MENDELEJEW_COMPREL_FROZEN.md`: różnice parami po przebiegach, 95% CI (bootstrap),
   p (permutacja znaków, dokładna), korekta Holma w rodzinach zadeklarowanych z góry.

Miary główne (zadeklarowane przed uruchomieniem): A_dół (krok „dół” z pierwiastkiem testowym),
B_te_hop (2 kroki, ścieżka dotyka testu), B_te_dd (dół∘dół dotyka testu); hit@1.

  python archiwum/skrypty/run_mendelejew_comprel_frozen.py [--runs 20] [--report-only]
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
import hashlib
import json
import math
import time

import torch

import run_mendelejew_ml3_zc as zc  # ustawia ml3.D_X / T_SEAL dla ZC
from run_mendelejew_ml3 import ARMS
from run_mendelejew_comp import COMPS, MENDELEJEW_Q, comp_queries, facts
from run_mendelejew_comprel import query, train
from run_mendelejew_rel import relation_pairs
from run_mendelejew_semi import labeled_pairs, split
from import_struktura_atomow import load_rows

OUT = _LAB / "doswiadczenie" / "glify"
SPEC = OUT / "comprel_frozen_spec.json"
RUNS = OUT / "comprel_frozen_runs.jsonl"
REPORT = OUT / "MENDELEJEW_COMPREL_FROZEN.md"
STEPS = 800
MODES = ["comp", "rel", "mix"]
VARIANTS = {"comp": ("comp", False), "comp+rel": ("rel", False), "comp+rel·PoE": ("rel", True), "comp+mix": ("mix", False)}
PRIMARY = ["A_dół", "B_te_hop", "B_te_dd"]
SECONDARY = ["A_prawo", "B_tr_op", "B_te_op"]


# ---------------------------------------------------------------- specyfikacja


def build_spec(meta, n_runs):
    syms = [m[3] for m in meta]
    S = lambda i: syms[i]
    fr_all = facts(meta)
    cq = comp_queries(fr_all)
    runs = []
    for i in range(n_runs):
        tr, te = split(meta, i)
        trs = set(tr)
        fr = {r: [(S(a), S(b)) for a, b in fs if a in trs and b in trs] for r, fs in fr_all.items()}
        q = {}
        for rel in ("dół", "prawo"):
            q[f"A_{rel}"] = [[[rel], S(a), S(b)] for a, b in fr_all[rel] if a not in trs or b not in trs]
        for sp, flt in (("tr", lambda a, x, b: {a, x, b} <= trs), ("te", lambda a, x, b: not {a, x, b} <= trs)):
            qs = [[list(rels), S(a), S(b)] for rels, a, x, b in cq if flt(a, x, b)]
            q[f"B_{sp}"] = qs
            q[f"B_{sp}_dd"] = [x for x in qs if x[0] == ["dół", "dół"]]
        runs.append({
            "run": i, "seed": i, "train": sorted(S(x) for x in tr), "test": sorted(S(x) for x in te),
            "facts": {r: [list(p) for p in v] for r, v in fr.items()},
            "group_pairs": [[S(a), S(b)] for a, b in labeled_pairs(meta, tr, "all", i)],
            "z_pairs": [[S(a), S(b)] for a, b in relation_pairs(meta, tr)["Z"]],
            "queries": q,
        })
    spec = {
        "version": 1, "features": "ZC: Z + konfiguracja (24 podpowłoki) · τ6 · T-seal off",
        "steps": STEPS, "operators": "lin", "temp": 0.1,
        "relations": {"dół": "ta sama grupa, okres+1, bez bloku f", "prawo": "Z+1"},
        "mendelejew": [[list(r), h, t] for h, r, t in MENDELEJEW_Q],
        "runs": runs,
    }
    return spec


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- uruchomienia


def run_one(spec_run, mode, arm, content, tracer, idx, mendel):
    I = lambda s: idx[s]
    fr = {r: [(I(a), I(b)) for a, b in v] for r, v in spec_run["facts"].items()}
    gp = [(I(a), I(b)) for a, b in spec_run["group_pairs"]]
    zp = [(I(a), I(b)) for a, b in spec_run["z_pairs"]]
    h, ops, g = train(arm, mode, content, tracer, fr, gp, zp, spec_run["seed"], STEPS)
    out = {}
    for vname, (vmode, poe) in VARIANTS.items():
        if vmode != mode:
            continue
        ranks = {}
        for key, qs in spec_run["queries"].items():
            ranks[key + "|op"] = [query(h, ops, g, I(a), tuple(rels), I(b), "op", poe) for rels, a, b in qs]
            if key in ("B_tr", "B_te"):
                ranks[key + "|hop"] = [query(h, ops, g, I(a), tuple(rels), I(b), "hop", poe) for rels, a, b in qs]
        ranks["mendelejew"] = [query(h, ops, g, I(a), tuple(rels), I(b), "op", poe) for rels, a, b in mendel]
        out[vname] = ranks
    return out


def load_done():
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["run"], r["mode"], r["arm"])] = r
    return done


# ---------------------------------------------------------------- metryki i statystyka


def metric(ranks, key):
    """key: A_dół, A_prawo, B_tr_op, B_te_op, B_te_hop, B_te_dd → hit@1 (i MRR)."""
    m = {"A_dół": "A_dół|op", "A_prawo": "A_prawo|op", "B_tr_op": "B_tr|op", "B_te_op": "B_te|op",
         "B_tr_hop": "B_tr|hop", "B_te_hop": "B_te|hop", "B_te_dd": "B_te_dd|op", "B_tr_dd": "B_tr_dd|op"}[key]
    r = ranks[m]
    if not r:
        return float("nan"), float("nan")
    return sum(x == 1 for x in r) / len(r), sum(1 / x for x in r) / len(r)


def paired_stats(d, n_boot=10000):
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    t = torch.tensor(d, dtype=torch.float64)
    gen = torch.Generator().manual_seed(12345)
    idx = torch.randint(n, (n_boot, n), generator=gen)
    boots = t[idx].mean(1).sort().values
    lo, hi = boots[int(0.025 * n_boot)].item(), boots[int(0.975 * n_boot) - 1].item()
    # dokładny test permutacyjny znaków (n ≤ 20 → 2^n kombinacji)
    if n <= 20:
        signs = ((torch.arange(2 ** n).unsqueeze(1) >> torch.arange(n)) & 1).to(torch.float64) * 2 - 1
        null = (signs * t.abs()).mean(1)
        p = (null.abs() >= abs(mean) - 1e-12).double().mean().item()
    else:
        signs = torch.randint(0, 2, (100000, n), generator=gen).double() * 2 - 1
        p = ((signs * t.abs()).mean(1).abs() >= abs(mean) - 1e-12).double().mean().item()
    wins = sum(x > 1e-12 for x in d)
    losses = sum(x < -1e-12 for x in d)
    return {"mean": mean, "sd": sd, "ci": (lo, hi), "p": p, "wins": wins, "losses": losses, "n": n}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------- raport


def report(spec, h, done, n_runs):
    runs = list(range(n_runs))
    val = {}  # (variant, arm, key) -> [hit1 per run]
    mrr = {}
    for (run, mode, arm), rec in done.items():
        if run >= n_runs:
            continue
        for vname, ranks in rec["variants"].items():
            for key in PRIMARY + SECONDARY + ["B_tr_hop", "B_tr_dd"]:
                h1, m1 = metric(ranks, key)
                val.setdefault((vname, arm, key), {})[run] = h1
                mrr.setdefault((vname, arm, key), {})[run] = m1
    complete = all(len(val.get((v, a, PRIMARY[0]), {})) == n_runs for v in VARIANTS for a in ARMS)
    V = list(VARIANTS)

    def series(v, a, k, src=val):
        return [src[(v, a, k)][r] for r in runs]

    L = [
        "# Mendelejew COMPREL FROZEN — zamrożony protokół, 20 przebiegów, statystyka parami",
        "",
        f"Specyfikacja: `comprel_frozen_spec.json` · SHA-256 `{h[:16]}…` · przebiegi={n_runs} (podział + ziarno zamrożone per przebieg; "
        "0–4 = dotychczasowe) · kroki=800 · każde uruchomienie w `comprel_frozen_runs.jsonl` (pozycja każdego zapytania) · "
        "skrypt `archiwum/skrypty/run_mendelejew_comprel_frozen.py`",
        "",
        "Wszystkie warianty i armie oceniane na **identycznych zapytaniach** w danym przebiegu; porównania parami po przebiegach. "
        "Miary główne (zadeklarowane z góry): **A_dół**, **B_te_hop**, **B_te_dd** (hit@1). "
        "CI = 95% bootstrap średniej różnicy; p = dokładny test permutacyjny znaków; p_Holm = korekta w obrębie rodziny.",
        "",
        f"Liczba zapytań (średnio na przebieg): A_dół={sum(len(r['queries']['A_dół']) for r in spec['runs'][:n_runs]) / n_runs:.1f}, "
        f"B_te={sum(len(r['queries']['B_te']) for r in spec['runs'][:n_runs]) / n_runs:.1f}, "
        f"B_te_dd={sum(len(r['queries']['B_te_dd']) for r in spec['runs'][:n_runs]) / n_runs:.1f}.",
        "",
    ]
    if not complete:
        L += ["**UWAGA: niekompletne — brakuje części uruchomień.**", ""]
    L += ["## 1. Średnie hit@1 (± odch. std. po przebiegach)", "", "| miara | arm | " + " | ".join(V) + " |",
          "|---|---|" + "---:|" * len(V)]
    for k in PRIMARY + SECONDARY:
        for a in ARMS:
            cells = []
            for v in V:
                s = series(v, a, k)
                m = sum(s) / len(s)
                sd = math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1))
                cells.append(f"{m:.3f} ± {sd:.3f}")
            L.append(f"| {k} | {a} | " + " | ".join(cells) + " |")
    # rodzina A: warianty (Most), miary główne
    fam_a = []
    for k in PRIMARY:
        for v1, v2 in (("comp+mix", "comp"), ("comp+rel·PoE", "comp"), ("comp+mix", "comp+rel·PoE")):
            d = [x - y for x, y in zip(series(v1, "most", k), series(v2, "most", k))]
            fam_a.append((k, f"{v1} − {v2}", paired_stats(d)))
    adj = holm([x[2]["p"] for x in fam_a])
    L += ["", "## 2. Rodzina A — warianty dla Mostu (miary główne, 9 testów, Holm)", "",
          "| miara | porównanie | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |",
          "|---|---|---:|---|---:|---:|---:|:-:|"]
    for (k, name, s), pa in zip(fam_a, adj):
        ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
        L.append(f"| {k} | {name} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | "
                 f"{s['p']:.4f} | {pa:.4f} | {ok} |")
    # rodzina B: Most vs reszta w każdym wariancie, miary główne
    fam_b = []
    for v in ("comp", "comp+mix", "comp+rel·PoE"):
        for k in PRIMARY:
            for o in ("softmax", "zero", "mlp"):
                d = [x - y for x, y in zip(series(v, "most", k), series(v, o, k))]
                fam_b.append((v, k, o, paired_stats(d)))
    adj_b = holm([x[3]["p"] for x in fam_b])
    L += ["", f"## 3. Rodzina B — Most vs reszta ({len(fam_b)} testów, Holm)", "",
          "| wariant | miara | vs | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |",
          "|---|---|---|---:|---|---:|---:|:-:|"]
    for (v, k, o, s), pa in zip(fam_b, adj_b):
        ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
        L.append(f"| {v} | {k} | {o} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | "
                 f"{s['wins']}/{s['losses']} | {pa:.4f} | {ok} |")
    # odporność: MRR dla rodziny A
    L += ["", "## 4. Odporność — te same porównania na MRR (bez korekty)", "",
          "| miara | porównanie | Δ MRR | 95% CI | p |", "|---|---|---:|---|---:|"]
    for k in PRIMARY:
        for v1, v2 in (("comp+mix", "comp"), ("comp+rel·PoE", "comp"), ("comp+mix", "comp+rel·PoE")):
            d = [x - y for x, y in zip(series(v1, "most", k, mrr), series(v2, "most", k, mrr))]
            s = paired_stats(d)
            L.append(f"| {k} | {v1} − {v2} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['p']:.4f} |")
    # replikacja przebiegów 0–4 vs 5–19
    if n_runs > 5:
        L += ["", "## 5. Replikacja: przebiegi 0–4 (dotychczasowe) vs 5–19 (nowe), Most, Δ względem comp", "",
              "| miara | wariant | Δ 0–4 | Δ 5–19 |", "|---|---|---:|---:|"]
        for k in PRIMARY:
            for v in ("comp+mix", "comp+rel·PoE"):
                d = [x - y for x, y in zip(series(v, "most", k), series("comp", "most", k))]
                L.append(f"| {k} | {v} | {sum(d[:5]) / 5:+.3f} | {sum(d[5:]) / len(d[5:]):+.3f} |")
    # kompozycja na znanych pierwiastkach (twierdzenie z COMP; Most vs reszta, wariant comp)
    fam_c = []
    for k in ("B_tr_op", "B_tr_hop", "B_tr_dd"):
        for o in ("softmax", "zero", "mlp"):
            d = [x - y for x, y in zip(series("comp", "most", k), series("comp", o, k))]
            fam_c.append((k, o, paired_stats(d)))
    adj_c = holm([x[2]["p"] for x in fam_c])
    L += ["", "## 5b. Kompozycja na znanych pierwiastkach — Most vs reszta (`comp`, 9 testów, Holm)", "",
          "| miara | Most | vs | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |",
          "|---|---:|---|---:|---|---:|---:|:-:|"]
    for (k, o, s), pa in zip(fam_c, adj_c):
        m = sum(series("comp", "most", k)) / n_runs
        ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
        L.append(f"| {k} | {m:.3f} | {o} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | "
                 f"{s['wins']}/{s['losses']} | {pa:.4f} | {ok} |")
    # Mendelejew
    L += ["", "## 6. Pytania Mendelejewa (Most) — mediana pozycji [IQR], trafień na 1. miejscu", "",
          "| pytanie | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for qi, (rels, a, b) in enumerate(spec["mendelejew"]):
        cells = []
        for v in V:
            rk = sorted(done[(r, VARIANTS[v][0], "most")]["variants"][v]["mendelejew"][qi] for r in runs)
            q1, med, q3 = rk[len(rk) // 4], rk[len(rk) // 2], rk[(3 * len(rk)) // 4]
            cells.append(f"{med} [{q1}–{q3}] · {sum(x == 1 for x in rk)}/{len(rk)}")
        L.append(f"| {'∘'.join(reversed(rels))}({a})={b} | " + " | ".join(cells) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    return fam_a, adj, fam_b, adj_b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    idx = {m[3]: i for i, m in enumerate(meta)}

    if SPEC.is_file():
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
        if len(spec["runs"]) < args.runs:
            raise SystemExit(f"specyfikacja ma {len(spec['runs'])} przebiegów < {args.runs}; nie nadpisuję zamrożonej")
    else:
        spec = build_spec(meta, args.runs)
        SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"zamrożono specyfikację → {SPEC}")
    h = spec_hash(spec)
    print(f"spec sha256={h}")

    done = load_done()
    bad = [k for k, r in done.items() if r["spec_sha256"] != h]
    if bad:
        raise SystemExit(f"{len(bad)} rekordów z innym hashem specyfikacji — przerwano")
    t0 = time.time()
    if not args.report_only:
        mendel = spec["mendelejew"]
        with RUNS.open("a", encoding="utf-8") as f:
            for run in range(args.runs):
                for mode in MODES:
                    for arm in ARMS:
                        if (run, mode, arm) in done:
                            continue
                        t1 = time.time()
                        variants = run_one(spec["runs"][run], mode, arm, content, tracer, idx, mendel)
                        rec = {"run": run, "mode": mode, "arm": arm, "spec_sha256": h, "steps": STEPS,
                               "elapsed_s": round(time.time() - t1, 2),
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "variants": variants}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        done[(run, mode, arm)] = rec
                print(f"  przebieg {run} gotowy ({time.time() - t0:.0f}s)", flush=True)
    report(spec, h, done, args.runs)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
