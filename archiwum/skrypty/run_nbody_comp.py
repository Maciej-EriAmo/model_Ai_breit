"""Zadanie kompozycyjne tam, gdzie relacja jest składową stanu.

Ten sam protokół co `run_crossdomain_comp.py`: ścisła indukcja (enkoder w treningu
widzi tylko elementy treningowe), 20 przebiegów, 800 kroków, ramiona
bw·ΔE / sm·ΔE / bw·0 / sm·0 / mlp, warianty comp / rel / mix.
Testy zadeklarowane z góry, 27 na dziedzinę, Holm w obrębie dziedziny:

  T1 kompozycja na znanych (comp; B_tr_op, B_tr_dd): bw·ΔE − sm·0, bw·ΔE − sm·ΔE, sm·ΔE − sm·0
  T2 relacja kolumny: PoE − comp, A_dół / B_te_hop / B_te_dd, każde z 5 ramion
  T3 jądro w PoE: bw·ΔE − sm·ΔE, 3 miary na nowych
  T4 mix − comp dla bw·ΔE, 3 miary na nowych

solar — 80 ciał, orbity kołowe wokół masy 1. Kolumna = klasa masy (8), wiersz = powłoka
  odległości (10; a = 0.3·2^wiersz). Dół = ta sama masa, następna dalsza powłoka.
  Prawo = następny w kolejności (powłoka, masa), także przez granicę powłoki.
  Cechy (składowe, nie indeks komórki): ε, v, ℓ, log m, log a, log r_Hilla.

three — 84 ciała, 28 trójek hierarchicznych (para wewnętrzna + ciało zewnętrzne,
  konfiguracja współliniowa, a_out = 4+k). Kolumna = rola (pierwotny / wtórny / zewnętrzny).
  Dół = ta sama rola, następna szersza trójka. Prawo = następna rola w tej samej trójce
  (nie przechodzi do następnej). Cechy: log m, r od środka masy, v, ε, Lz, potencjał,
  odległość do najbliższego.

Stary `bench_nbody.py` to inne zadanie (etykieta warstwy, one-hot hierarchii w tracerze).
Tu go nie powtarzamy.

  python archiwum/skrypty/run_nbody_comp.py [--domains solar,three] [--smoke] [--report-only]
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
import math
import time

import torch
import torch.nn.functional as F

import run_crossdomain_comp as xd
import run_mendelejew_comprel_frozen as fz
import run_mendelejew_ml3 as ml3

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "nbody_comp_runs.jsonl"
REPORT = OUT / "NBODY_COMP.md"
STATUS = OUT / "nbody_comp_status.txt"
G = 1.0

FEATURES = {
    "solar": "ε, v, ℓ, log m, log a, log r_Hilla (z-score, L2); τ = te same 6, min-max",
    "three": "log m, r_CM, v, ε, Lz, potencjał, r_min (z-score, L2); τ = pierwsze 6, min-max",
}


def _pack(raw: torch.Tensor, n_tracer: int = 6):
    mu = raw.mean(0, keepdim=True)
    sd = raw.std(0, unbiased=False).clamp_min(1e-8)
    content = F.normalize((raw - mu) / sd, dim=-1)
    t = raw[:, :n_tracer]
    lo, hi = t.min(0).values, t.max(0).values
    tracer = (t - lo) / (hi - lo).clamp_min(1e-8)
    return content.float(), tracer.float()


def domain_solar():
    n_mass, n_shell = 8, 10
    masses = [10 ** (-7 + 4 * i / (n_mass - 1)) for i in range(n_mass)]
    rows = []
    meta = []
    for shell in range(n_shell):
        a = 0.3 * (2 ** shell)
        eps = -0.5 / a
        v = a ** -0.5
        ell = a ** 0.5
        log_a = math.log10(a)
        for mi, m in enumerate(masses):
            r_h = a * (m / 3.0) ** (1.0 / 3.0)
            rows.append([eps, v, ell, math.log10(m), log_a, math.log10(r_h)])
            meta.append((mi, shell, shell * n_mass + mi, f"m{mi}-s{shell}"))
    content, tracer = _pack(torch.tensor(rows, dtype=torch.float64))
    return meta, content, tracer, xd.grid_facts(meta), []


def _triple(k: int):
    """Współliniowa trójka hierarchiczna. Zwraca 3 wiersze cech, rola 0/1/2."""
    m1, m2, m3 = 1.0, 0.4, 0.2
    m12 = m1 + m2
    total = m12 + m3
    a_in = 1.0
    a_out = 4.0 + k
    x_icm = -a_out * m3 / total
    r1 = a_in * m2 / m12
    r2 = a_in * m1 / m12
    xs = [x_icm - r1, x_icm + r2, a_out * m12 / total]
    ms = [m1, m2, m3]
    w_out = math.sqrt(G * total / a_out ** 3)
    w_in = math.sqrt(G * m12 / a_in ** 3)
    vy = [w_out * x_icm + w_in * (xs[0] - x_icm), w_out * x_icm + w_in * (xs[1] - x_icm), w_out * xs[2]]
    out = []
    for i in range(3):
        pot = 0.0
        near = float("inf")
        for j in range(3):
            if i == j:
                continue
            d = abs(xs[i] - xs[j])
            pot -= G * ms[j] / d
            near = min(near, d)
        speed = abs(vy[i])
        eps = 0.5 * speed * speed + pot
        lz = xs[i] * vy[i]
        out.append([math.log10(ms[i]), abs(xs[i]), speed, eps, lz, pot, near])
    return out


def domain_three():
    n = 28
    rows, meta = [], []
    for k in range(n):
        for role, feat in enumerate(_triple(k)):
            rows.append(feat)
            # przerwa w numeracji: „prawo” nie przeskakuje do następnej trójki
            meta.append((role, k, k * 10 + role, f"t{k:02d}{role}"))
    content, tracer = _pack(torch.tensor(rows, dtype=torch.float64))
    return meta, content, tracer, xd.grid_facts(meta), []


DOMAINS = {"solar": domain_solar, "three": domain_three}


def check_domain(name, meta, content, tracer, fr):
    symbols = [m[3] for m in meta]
    if len(symbols) != len(set(symbols)):
        raise SystemExit(f"{name}: symbole nie są unikalne")
    if not torch.isfinite(content).all() or not torch.isfinite(tracer).all():
        raise SystemExit(f"{name}: cechy niefinitywne")
    if tracer.min() < -1e-6 or tracer.max() > 1 + 1e-6:
        raise SystemExit(f"{name}: tracer poza [0, 1]")
    by = {m[3]: m for m in meta}
    for a, b in fr["dół"]:
        if meta[a][0] != meta[b][0] or meta[b][1] <= meta[a][1]:
            raise SystemExit(f"{name}: zły fakt dół {symbols[a]} → {symbols[b]}")
    for a, b in fr["prawo"]:
        if meta[b][2] != meta[a][2] + 1:
            raise SystemExit(f"{name}: prawo nie jest kolejnością+1")
        if name == "three" and meta[a][1] != meta[b][1]:
            raise SystemExit(f"{name}: prawo wychodzi z trójki")
    if name == "three":
        # energie specyficzne w surowej skali sprawdzamy na nowo, bo pack je z-scoruje
        for k in range(28):
            for row in _triple(k):
                if row[3] >= 0:
                    raise SystemExit(f"{name}: ciało niezwiązane w trójce {k}")
    _ = by


def build_spec(name, meta, fr_all):
    spec = xd.build_spec(name, meta, fr_all, [])
    spec["axes"] = {
        "solar": "kolumna=klasa masy, wiersz=powłoka a, dół=ta sama masa dalej, prawo=kolejność (a, masa)",
        "three": "kolumna=rola w trójce, wiersz=a_out, dół=ta sama rola szersza trójka, prawo=następna rola wewnątrz trójki",
    }[name]
    spec["features"] = FEATURES[name]
    return spec


def report(specs, done):
    L = [
        "# Zadanie kompozycyjne — Układ Słoneczny i trójka hierarchiczna",
        "",
        "Ten sam protokół co przy pierwiastkach, kanie i muzyce (`run_crossdomain_comp.py`): "
        "relacje „dół” i „prawo”, ramiona bw·ΔE / sm·ΔE / bw·0 / sm·0 / mlp, warianty comp / PoE / mix, "
        "800 kroków, 20 przebiegów. **Ścisłe rozdzielenie:** w treningu enkoder widzi tylko elementy treningowe. "
        "Testy T1–T4 zadeklarowane przed uruchomieniem, Holm w obrębie dziedziny. "
        "Skrypt `archiwum/skrypty/run_nbody_comp.py`; przebiegi w `nbody_comp_runs.jsonl`; "
        "specyfikacje `nbody_comp_spec_<dziedzina>.json`.",
        "",
        "Stary bench n-ciał (etykieta warstwy, one-hot hierarchii w tracerze) nie jest tym pomiarem.",
        "",
        "| dziedzina | elementy | kolumny | osie | cechy | podobieństwo w surowych cechach |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, spec in specs.items():
        L.append(
            f"| {name} | {len(spec['elements'])} | {spec['n_cols']} | {spec['axes']} | {spec['features']} | {spec['raw_sim']} |"
        )
    summary = []
    for name, spec in specs.items():
        val = {}
        for (dom, run, mode, arm), rec in done.items():
            if dom != name:
                continue
            for v, ranks in rec["variants"].items():
                for k in xd.NEW + xd.KNOWN:
                    val[(arm, v, k, run)] = fz.metric(ranks, k)[0]
        runs = range(xd.N_RUNS)

        def ser(a, v, k, _val=val):
            return [_val[(a, v, k, r)] for r in runs]

        def diff(a1, v1, a2, v2, k):
            d = [x - y for x, y in zip(ser(a1, v1, k), ser(a2, v2, k)) if not (math.isnan(x) or math.isnan(y))]
            return fz.paired_stats(d) if len(d) >= 5 else None

        L += ["", f"## {name}", "", spec["axes"], "", "Zapytań na przebieg (średnio): " + ", ".join(
            f"{k}={sum(len(r['queries'][q]) for r in spec['runs']) / xd.N_RUNS:.1f}"
            for k, q in (("A_dół", "A_dół"), ("B_te", "B_te"), ("B_te_dd", "B_te_dd"), ("B_tr", "B_tr"), ("B_tr_dd", "B_tr_dd"))
        ) + ".", "", "Średnie hit@1:", "",
              "| miara | wariant | " + " | ".join(xd.ARMS) + " |",
              "|---|---|" + "---:|" * len(xd.ARMS)]
        for k in xd.NEW + xd.KNOWN:
            for v in ("comp", "comp+rel·PoE", "comp+mix"):
                cells = []
                for a in xd.ARMS:
                    s = [x for x in ser(a, v, k) if not math.isnan(x)]
                    cells.append(f"{sum(s) / len(s):.3f}" if s else "—")
                L.append(f"| {k} | {v} | " + " | ".join(cells) + " |")
        tests = []
        for k in xd.KNOWN:
            for a1, a2, lab in (
                ("bw·ΔE", "sm·0", "T1 Most − dawny Softmax"),
                ("bw·ΔE", "sm·ΔE", "T1 jądro"),
                ("sm·ΔE", "sm·0", "T1 informacja ΔE"),
            ):
                tests.append((lab, "comp", k, diff(a1, "comp", a2, "comp", k)))
        for a in xd.ARMS:
            for k in xd.NEW:
                tests.append((f"T2 relacja (PoE−comp) · {a}", "PoE", k, diff(a, "comp+rel·PoE", a, "comp", k)))
        for k in xd.NEW:
            tests.append(("T3 jądro w PoE", "PoE", k, diff("bw·ΔE", "comp+rel·PoE", "sm·ΔE", "comp+rel·PoE", k)))
        for k in xd.NEW:
            tests.append(("T4 mix−comp · bw·ΔE", "mix", k, diff("bw·ΔE", "comp+mix", "bw·ΔE", "comp", k)))
        ok_tests = [t for t in tests if t[3] is not None]
        fam = xd.stats_family(ok_tests)
        L += ["", f"Testy zadeklarowane z góry ({len(ok_tests)}, Holm):", "",
              "| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |",
              "|---|---|---:|---|---:|---:|:-:|"]
        for (lab, v, k, s), pa in fam:
            ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
            L.append(
                f"| {lab} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | {pa:.4f} | {ok} |"
            )
            summary.append((name, lab, k, s, pa))
    L += ["", "## Zestawienie (Δ, w nawiasie: utrzymuje się po Holmie?)", "",
          "| test | miara | " + " | ".join(specs) + " |",
          "|---|---|" + "---:|" * len(specs)]
    keys = []
    for name, lab, k, s, pa in summary:
        if (lab, k) not in keys:
            keys.append((lab, k))
    for lab, k in keys:
        cells = []
        for name in specs:
            hit = [(s, pa) for n2, l2, k2, s, pa in summary if n2 == name and l2 == lab and k2 == k]
            if hit:
                s, pa = hit[0]
                ok = pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0)
                cells.append(f"{s['mean']:+.3f} ({'tak' if ok else 'nie'})")
            else:
                cells.append("—")
        L.append(f"| {lab} | {k} | " + " | ".join(cells) + " |")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def _status(text: str):
    STATUS.write_text(text + "\n", encoding="utf-8")


def smoke():
    saved = xd.STEPS
    xd.STEPS = 2
    try:
        for name in ("solar", "three"):
            meta, content, tracer, fr, _ = DOMAINS[name]()
            check_domain(name, meta, content, tracer, fr)
            ml3.D_X = content.shape[1]
            h, ops, _ = xd.train_strict("bw·ΔE", "comp", content, tracer, list(range(len(meta))), fr, [], [], 0)
            if not torch.isfinite(h).all() or h.shape != (len(meta), 32):
                raise SystemExit(f"smoke {name}: zły embedding {tuple(h.shape)}")
            print(
                f"[smoke {name}] N={len(meta)} d_x={tuple(content.shape)} τ={tuple(tracer.shape)} "
                f"dół={len(fr['dół'])} prawo={len(fr['prawo'])} {xd.raw_similarity(meta, content)} "
                f"embedding={tuple(h.shape)}",
                flush=True,
            )
            _ = ops
    finally:
        xd.STEPS = saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default="solar,three")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke()
        return
    names = args.domains.split(",")
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = __import__("json").loads(line)
                done[(r["domain"], r["run"], r["mode"], r["arm"])] = r
    specs = {}
    t0 = time.time()
    json = __import__("json")
    try:
        for name in names:
            meta, content, tracer, fr_all, _ = DOMAINS[name]()
            check_domain(name, meta, content, tracer, fr_all)
            ml3.D_X = content.shape[1]
            sp_path = OUT / f"nbody_comp_spec_{name}.json"
            if sp_path.is_file():
                spec = json.loads(sp_path.read_text(encoding="utf-8"))
            else:
                spec = build_spec(name, meta, fr_all)
                sp_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            hsh = fz.spec_hash(spec)
            idx = {m[3]: i for i, m in enumerate(meta)}
            if [m[3] for m in meta] != spec["elements"]:
                raise SystemExit(f"{name}: elementy różne od zamrożonej specyfikacji")
            print(
                f"[{name}] N={len(meta)} d_x={content.shape[1]} dół={len(fr_all['dół'])} prawo={len(fr_all['prawo'])} "
                f"spec={hsh[:12]} · {xd.raw_similarity(meta, content)}",
                flush=True,
            )
            for (d, *_), r in done.items():
                if d == name and r["spec_sha256"] != hsh:
                    raise SystemExit(f"{name}: rekord z innym hashem — przerwano")
            if not args.report_only:
                with RUNS.open("a", encoding="utf-8") as f:
                    for run in range(xd.N_RUNS):
                        for mode in xd.MODES:
                            for arm in xd.ARMS:
                                if (name, run, mode, arm) in done:
                                    continue
                                t1 = time.time()
                                variants = xd.run_one(spec["runs"][run], mode, arm, content, tracer, idx, spec["mendelejew"])
                                rec = {
                                    "domain": name, "run": run, "mode": mode, "arm": arm, "spec_sha256": hsh,
                                    "elapsed_s": round(time.time() - t1, 2),
                                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "variants": variants,
                                }
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                f.flush()
                                done[(name, run, mode, arm)] = rec
                        _status(f"RUNNING {name} przebieg {run} ({time.time() - t0:.0f}s)")
                        print(f"  [{name}] przebieg {run} gotowy ({time.time() - t0:.0f}s)", flush=True)
            spec["n_cols"] = len({m[0] for m in meta})
            spec["features"] = FEATURES[name]
            spec["axes"] = spec.get("axes") or build_spec(name, meta, fr_all)["axes"]
            spec["raw_sim"] = xd.raw_similarity(meta, content)
            specs[name] = spec
        report(specs, done)
        _status(f"DONE {time.time() - t0:.0f}s")
        print(f"wrote {REPORT}", flush=True)
    except Exception as e:
        _status(f"FAILED {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
