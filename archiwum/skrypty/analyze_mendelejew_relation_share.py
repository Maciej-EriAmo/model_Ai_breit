"""Mendelejew — udział struktury relacyjnej w poprawie (bez nowego treningu).

Dane: `kernel_ablation_runs.jsonl` (4 ramiona: bw·ΔE, bw·0, sm·ΔE, sm·0 — ta sama klasa
enkodera, zamrożona specyfikacja, 20 przebiegów) + `comprel_frozen_runs.jsonl` (ramię mlp,
bez uwagi, jako odniesienie). Wszystkie warianty w danym przebiegu na identycznych zapytaniach.

1. Efekt relacji przy STAŁYM jądrze i informacji: (wariant z relacją grupy) − comp,
   osobno w każdym ramieniu. Rodzina zadeklarowana z góry: 4 ramiona × {comp+rel, comp+rel·PoE,
   comp+mix} × {A_dół, B_te_hop, B_te_dd} = 36 testów, Holm. Ramię mlp i miary na znanych
   pierwiastkach (B_tr_op, B_tr_dd) — dodatkowo, bez korekty.
2. Rozkład poprawy od (sm·0, comp) do (bw·ΔE, relacja) na trzy czynniki:
   informacja ΔE {0, ΔE} × jądro {sm, bw} × relacja {comp, wariant} — wartości Shapleya
   (średnia z 6 kolejności dodawania czynników), per przebieg → średnia, 95% CI, p.

  python archiwum/skrypty/analyze_mendelejew_relation_share.py
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
from itertools import permutations

import run_mendelejew_comprel_frozen as fz

OUT = _LAB / "doswiadczenie" / "glify"
KRUNS = OUT / "kernel_ablation_runs.jsonl"
REPORT = OUT / "MENDELEJEW_RELATION_SHARE.md"
ARMS = ["bw·ΔE", "bw·0", "sm·ΔE", "sm·0"]
REL_VARIANTS = ["comp+rel", "comp+rel·PoE", "comp+mix"]
NEW = ["A_dół", "B_te_hop", "B_te_dd"]
KNOWN = ["B_tr_op", "B_tr_dd"]
N_RUNS = 20


def load(path, arms):
    val = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] not in arms or r["run"] >= N_RUNS:
            continue
        for v, ranks in r["variants"].items():
            for k in NEW + KNOWN:
                val[(r["arm"], v, k, r["run"])] = fz.metric(ranks, k)[0]
    return val


def main():
    spec = json.loads(fz.SPEC.read_text(encoding="utf-8"))
    h = fz.spec_hash(spec)
    for p in (KRUNS, fz.RUNS):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["spec_sha256"] != h:
                raise SystemExit(f"{p.name}: inny hash specyfikacji")
    val = load(KRUNS, ARMS)
    val.update(load(fz.RUNS, ["mlp"]))
    runs = range(N_RUNS)
    ser = lambda a, v, k: [val[(a, v, k, r)] for r in runs]
    mean = lambda xs: sum(xs) / len(xs)

    L = [
        "# Mendelejew — jaka część poprawy wynika ze struktury relacyjnej",
        "",
        f"Bez nowego treningu: dane z ablacji jądra (`kernel_ablation_runs.jsonl`, 4 ramiona, ta sama klasa enkodera) "
        f"i z COMPREL FROZEN (ramię mlp). Specyfikacja SHA-256 `{h[:16]}…`, {N_RUNS} przebiegów, identyczne zapytania. "
        "Skrypt `archiwum/skrypty/analyze_mendelejew_relation_share.py`.",
        "",
        "Relacja grupy = wariant `comp+rel` (osobne głowice, odpowiedź z operatora), `comp+rel·PoE` (operator + głowica grupy), "
        "`comp+mix` (grupa w przestrzeni operatorów). Bez relacji = `comp`.",
        "",
        "## 1. Efekt relacji przy stałym jądrze: wariant − comp (36 testów, Holm)",
        "",
        "| ramię | wariant | miara | comp | z relacją | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |",
        "|---|---|---|---:|---:|---:|---|---:|---:|:-:|",
    ]
    fam = []
    for a in ARMS:
        for v in REL_VARIANTS:
            for k in NEW:
                d = [x - y for x, y in zip(ser(a, v, k), ser(a, "comp", k))]
                fam.append((a, v, k, fz.paired_stats(d)))
    adj = fz.holm([x[3]["p"] for x in fam])
    for (a, v, k, s), pa in zip(fam, adj):
        ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
        L.append(f"| {a} | {v} | {k} | {mean(ser(a, 'comp', k)):.3f} | {mean(ser(a, v, k)):.3f} | {s['mean']:+.3f} | "
                 f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | {pa:.4f} | {ok} |")

    L += ["", "## 1b. Dodatkowo (bez korekty): ramię mlp (bez uwagi) i miary na znanych pierwiastkach", "",
          "| ramię | wariant | miara | comp | z relacją | Δ | 95% CI | p |", "|---|---|---|---:|---:|---:|---|---:|"]
    for a, ks in [("mlp", NEW + KNOWN)] + [(a, KNOWN) for a in ARMS]:
        for v in REL_VARIANTS:
            for k in ks:
                d = [x - y for x, y in zip(ser(a, v, k), ser(a, "comp", k))]
                s = fz.paired_stats(d)
                L.append(f"| {a} | {v} | {k} | {mean(ser(a, 'comp', k)):.3f} | {mean(ser(a, v, k)):.3f} | {s['mean']:+.3f} | "
                         f"[{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['p']:.4f} |")

    # 2. Shapley: czynniki info (ΔE), jądro (bw), relacja (wariant)
    def f(info, kern, rel, relv, k, r):
        arm = f"{'bw' if kern else 'sm'}·{'ΔE' if info else '0'}"
        return val[(arm, relv if rel else "comp", k, r)]

    names = ["informacja ΔE", "jądro BW", "relacja grupy"]
    for relv in ("comp+rel·PoE", "comp+mix"):
        L += ["", f"## 2. Rozkład poprawy (sm·0, comp) → (bw·ΔE, {relv}) — wartości Shapleya", "",
              "Udział = φ / całkowita poprawa (średnie po przebiegach). φ: 95% CI i p po przebiegach.", "",
              "| miara | start | koniec | całość | czynnik | φ | 95% CI | p | udział |", "|---|---:|---:|---:|---|---:|---|---:|---:|"]
        for k in NEW + KNOWN:
            phis = {n: [] for n in names}
            tot = []
            for r in runs:
                for i, n in enumerate(names):
                    acc = 0.0
                    for order in permutations(range(3)):
                        state = [0, 0, 0]
                        for j in order:
                            if j == i:
                                before = f(*state, relv, k, r)
                                state[j] = 1
                                acc += f(*state, relv, k, r) - before
                                break
                            state[j] = 1
                    phis[n].append(acc / 6)
                tot.append(f(1, 1, 1, relv, k, r) - f(0, 0, 0, relv, k, r))
            T = mean(tot)
            start = mean([f(0, 0, 0, relv, k, r) for r in runs])
            for j, n in enumerate(names):
                s = fz.paired_stats(phis[n])
                share = f"{s['mean'] / T:.0%}" if abs(T) > 1e-9 else "—"
                head = f"| {k} | {start:.3f} | {start + T:.3f} | {T:+.3f} |" if j == 0 else "| | | | |"
                L.append(f"{head} {n} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['p']:.4f} | {share} |")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
