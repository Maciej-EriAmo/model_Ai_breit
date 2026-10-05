"""T1 — notacja muzyczna jako dodatkowe zmienne. Protokół z JADRO_OGOLNE.md.

Katalog syntetyczny, 80 zdarzeń (rząd dotychczasowych 84 nut). Widmo ośmiu harmonicznych
jak w domain_music. Cis i Des (i cztery dalsze pary) mają to samo widmo. Kolumna relacji,
ustalona przed treningiem, to pisownia: oś, której najbliższy sąsiad na samym widmie nie
odtwarza. Dół = ta sama pisownia, następna oktawa. Wartość rytmiczna, miejsce w takcie,
dynamika, artykulacja i głos są w tracerze i nie wynikają z widma.

Protokół jak run_nbody_comp.py: ścisła indukcja, 20 przebiegów, 800 kroków, warianty
comp / PoE / mix. Ramiona: bw·ΔE, sm·ΔE, bw·0, sm·0, mlp oraz bw·tok, sm·tok
(τ w tokenie, jak KERNEL·TOK).

Rodzina Holma ustalona przed uruchomieniem (jedno zadanie, 11 testów):
  N1  PoE − comp, miara A_dół, ramiona bw·ΔE, sm·ΔE, bw·0, sm·0, mlp
  N2  bw·tok − sm·tok, wariant comp, miary A_dół, B_te_hop, B_te_dd
  N3  sm·tok − sm·ΔE, wariant comp, te same trzy miary
Odczyt pozytywny: N1 i N3 — 95% CI nad zerem i p_Holm < 0.05.
N2 — różnica nie przeżywa korekty, a 90% CI mieści się w ±0.05.

T3 przy okazji: czas ramienia i czy strata schodzi (ostatnia < pierwsza) oraz jest skończona.

  python archiwum/skrypty/run_jadro_t1.py [--smoke] [--report-only]
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
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_crossdomain_comp as xd
import run_mendelejew_comprel_frozen as fz
import run_mendelejew_kernel_ablation as ka
import run_mendelejew_ml3 as ml3
import analyze_kernel_equivalence as eq
from run_mendelejew_comp import Ops
from run_mendelejew_kernel_tok import TokKernelEncoder
from run_mendelejew_semi import _nce
from config import CANON

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "jadro_t1_runs.jsonl"
SPEC = OUT / "jadro_t1_spec.json"
REPORT = OUT / "JADRO_T1.md"
STATUS = OUT / "jadro_t1_status.txt"

ARMS = ["bw·ΔE", "sm·ΔE", "bw·0", "sm·0", "mlp", "bw·tok", "sm·tok"]
N1_ARMS = ["bw·ΔE", "sm·ΔE", "bw·0", "sm·0", "mlp"]
NEW = ["A_dół", "B_te_hop", "B_te_dd"]
# pary enharmoniczne: ta sama klasa wysokości, dwie pisownie
PAIRS = (("Cs", "Db", 1), ("Ds", "Eb", 3), ("Fs", "Gb", 6), ("Gs", "Ab", 8), ("As", "Bb", 10))
N_OCT = 8
LOSS = {}


def spectrum(midi: int) -> torch.Tensor:
    bins = torch.arange(24, 144, dtype=torch.float)
    spec = torch.zeros_like(bins)
    for k in range(1, 9):
        spec = spec + (1 / k) * torch.exp(-0.5 * ((bins - (midi + 12 * math.log2(k))) / 0.5) ** 2)
    return F.normalize(spec, dim=0)


def notation(idx: int, col: int) -> torch.Tensor:
    """Sześć składowych notacji w [0, 1]. Nie są funkcją widma: bliźniak enharmoniczny ma inny idx."""
    return torch.tensor([
        col / 9,
        ((idx * 3) % 5) / 4,
        ((idx * 2) % 4) / 3,
        ((idx * 5) % 6) / 5,
        ((idx * 7) % 4) / 3,
        ((idx * 11) % 4) / 3,
    ], dtype=torch.float)


def domain():
    meta, cs, ts = [], [], []
    col = 0
    spellings = []
    for sharp, flat, pc in PAIRS:
        spellings.append((sharp, pc))
        spellings.append((flat, pc))
    for o in range(N_OCT):
        for c, (name, pc) in enumerate(spellings):
            midi = 24 + o * 12 + pc
            idx = o * len(spellings) + c
            cs.append(spectrum(midi))
            ts.append(notation(idx, c))
            meta.append((c, o, idx, f"{name}{o + 1}"))
            _ = col
    content, tracer = torch.stack(cs), torch.stack(ts)
    # bliźniaki enharmoniczne: to samo widmo, inna pisownia
    by = {m[3]: i for i, m in enumerate(meta)}
    for o in range(N_OCT):
        for sharp, flat, _pc in PAIRS:
            a, b = by[f"{sharp}{o + 1}"], by[f"{flat}{o + 1}"]
            if not torch.equal(content[a], content[b]):
                raise SystemExit("bliźniak enharmoniczny nie ma tego samego widma")
            if torch.equal(tracer[a], tracer[b]):
                raise SystemExit("krotka notacji jest funkcją widma")
    return meta, content, tracer, xd.grid_facts(meta)


def make_encoder(arm):
    if arm == "mlp":
        return ml3.make_encoder("mlp")
    if "tok" in arm:
        kern, rest = arm.split("·")
        return TokKernelEncoder(kern, "ΔE" in rest, True, input_dim=ml3.D_X, dim_tracer=CANON.n_dim,
                                 num_layers=CANON.num_layers, d=CANON.d_model, coupling=CANON.coupling, t_seal=False)
    return ka.make_kernel_encoder(arm)


def train_logged(arm, mode, content, tracer, tr_idx, fr, gp, zp, seed):
    loc = {g: j for j, g in enumerate(tr_idx)}
    L = lambda pairs: [(loc[a], loc[b]) for a, b in pairs]
    fr_l = {r: L(v) for r, v in fr.items()}
    gp_l, zp_l = L(gp), L(zp)
    c_tr, t_tr = content[tr_idx], tracer[tr_idx]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = xd.make_encoder(arm)
    ops = Ops("lin", CANON.d_model)
    heads = nn.ModuleDict({r: nn.Linear(CANON.d_model, CANON.d_model) for r in ("grupa", "Z")}) if mode == "rel" else nn.ModuleDict()
    opt = torch.optim.Adam(list(enc.parameters()) + list(ops.parameters()) + list(heads.parameters()), lr=CANON.lr)
    n = len(tr_idx)
    loss0 = lossT = None
    for step in range(xd.STEPS):
        enc.train()
        raw = ml3.encode(enc, arm, ml3.noisy(c_tr, gen), t_tr)
        h = F.normalize(raw, dim=-1)
        loss = torch.zeros(())
        for r, fs in fr_l.items():
            if not fs:
                continue
            sel = torch.randint(len(fs), (32,), generator=gen).tolist()
            a = torch.tensor([fs[i][0] for i in sel])
            b = torch.tensor([fs[i][1] for i in sel])
            s = (ops.apply(h[a], [r]) @ h.T / ml3.TEMP).masked_fill(F.one_hot(a, n).bool(), -1e9)
            loss = loss + F.cross_entropy(s, b)
        if mode == "rel" and gp_l:
            loss = loss + _nce(F.normalize(heads["grupa"](raw), dim=-1), gp_l, gen)
            if zp_l:
                loss = loss + _nce(F.normalize(heads["Z"](raw), dim=-1), zp_l, gen)
        elif mode == "mix" and gp_l:
            loss = loss + _nce(h, gp_l, gen)
        if step == 0:
            loss0 = float(loss.detach())
        if step == xd.STEPS - 1:
            lossT = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        raw = ml3.encode(enc, arm, content, tracer)
        g = F.normalize(heads["grupa"](raw), dim=-1) if mode == "rel" else None
        h = F.normalize(raw, dim=-1)
    LOSS["last"] = {
        "loss0": loss0, "lossT": lossT,
        "finite": bool(math.isfinite(loss0) and math.isfinite(lossT) and torch.isfinite(h).all()),
        "descend": bool(math.isfinite(loss0) and math.isfinite(lossT) and lossT < loss0),
    }
    return h, ops, g


def nn_rate(meta, content, field):
    h = F.normalize(content, dim=-1)
    s = (h @ h.T).fill_diagonal_(-2)
    nn1 = s.argmax(-1).tolist()
    return sum(meta[i][field] == meta[j][field] for i, j in enumerate(nn1)) / len(meta)


def _status(text):
    STATUS.write_text(text + "\n", encoding="utf-8")


def smoke():
    saved = xd.STEPS
    xd.STEPS = 2
    xd.make_encoder = make_encoder
    xd.train_strict = train_logged
    try:
        meta, content, tracer, fr = domain()
        ml3.D_X = content.shape[1]
        col, row = nn_rate(meta, content, 0), nn_rate(meta, content, 1)
        print(f"[smoke] N={len(meta)} d_x={tuple(content.shape)} τ={tuple(tracer.shape)} dół={len(fr['dół'])} prawo={len(fr['prawo'])} 1-NN kolumna={col:.0%} wiersz={row:.0%}", flush=True)
        if col > 0.25:
            raise SystemExit("widmo odtwarza pisownię — katalog zły")
        h, _, _ = train_logged("bw·tok", "comp", content, tracer, list(range(len(meta))), fr, [], [], 0)
        if not LOSS["last"]["finite"] or h.shape != (len(meta), 32):
            raise SystemExit(f"smoke: {LOSS['last']} {tuple(h.shape)}")
        print(f"[smoke] bw·tok {LOSS['last']}", flush=True)
    finally:
        xd.STEPS = saved


def series(spec, done):
    """(arm, variant, key, run) -> hit@1."""
    val = {}
    for (run, mode, arm), rec in done.items():
        for v, ranks in rec["variants"].items():
            for k in NEW + ["B_tr_op", "B_tr_dd"]:
                val[(arm, v, k, run)] = fz.metric(ranks, k)[0]
    return val


def contrast(val, a1, v1, a2, v2, k):
    d = []
    for r in range(xd.N_RUNS):
        x, y = val.get((a1, v1, k, r)), val.get((a2, v2, k, r))
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        d.append(x - y)
    if len(d) < 5:
        return None
    return d


def report(spec, done):
    val = series(spec, done)
    tests = []
    for arm in N1_ARMS:
        d = contrast(val, arm, "comp+rel·PoE", arm, "comp", "A_dół")
        tests.append(("N1", arm, "A_dół", d))
    for k in NEW:
        tests.append(("N2", "bw·tok − sm·tok", k, contrast(val, "bw·tok", "comp", "sm·tok", "comp", k)))
        tests.append(("N3", "sm·tok − sm·ΔE", k, contrast(val, "sm·tok", "comp", "sm·ΔE", "comp", k)))
    packed = []
    for lab, who, k, d in tests:
        if d is None:
            raise SystemExit(f"brak kontrastu {lab} {who} {k}")
        packed.append((lab, who, k, fz.paired_stats(d), eq.ci90(d)))
    adj = fz.holm([s["p"] for *_, s, _ci in packed])
    L = [
        "# T1 — notacja muzyczna jako dodatkowe zmienne",
        "",
        "Katalog syntetyczny, 80 zdarzeń. Widmo ośmiu harmonicznych jak dotychczasowa muzyka. "
        "Pięć par enharmonicznych (Cs/Db, Ds/Eb, Fs/Gb, Gs/Ab, As/Bb) w ośmiu oktawach: bliźniak ma to samo widmo. "
        "Kolumna = pisownia, wiersz = oktawa, dół = ta sama pisownia i następna oktawa. "
        "W tracerze (plugin, 6 składowych): pisownia, wartość rytmiczna, miejsce w takcie, dynamika, artykulacja, głos. "
        "Nie wynikają z widma.",
        "",
        f"Ścisła indukcja, 800 kroków, 20 przebiegów. Specyfikacja `{SPEC.name}`, przebiegi `jadro_t1_runs.jsonl`. "
        f"Skrypt `archiwum/skrypty/run_jadro_t1.py`. Na samym widmie: {spec['raw_sim']}.",
        "",
        "Rodzina Holma (11 testów) ustalona przed uruchomieniem. N1: PoE − comp na A_dół, pięć ramion protokołu. "
        "N2: bw·tok − sm·tok, wariant comp, miary nowych zdarzeń. N3: sm·tok − sm·ΔE, te same miary. "
        "Hop i dwa kroki w dół są w N2 i N3, bo program mówi o miarach liczby mnogiej; N1 zostaje przy jednym kroku.",
        "",
        "## Średnie hit@1",
        "",
        "| miara | wariant | " + " | ".join(ARMS) + " |",
        "|---|---|" + "---:|" * len(ARMS),
    ]
    for k in ["A_dół", "B_te_hop", "B_te_dd", "B_tr_op"]:
        for v in ("comp", "comp+rel·PoE", "comp+mix"):
            cells = []
            for a in ARMS:
                s = [val[(a, v, k, r)] for r in range(xd.N_RUNS)]
                s = [x for x in s if not math.isnan(x)]
                cells.append(f"{sum(s) / len(s):.3f}" if s else "—")
            L.append(f"| {k} | {v} | " + " | ".join(cells) + " |")
    L += ["", "## Testy zadeklarowane", "",
          "| test | kontrast | miara | Δ | 95% CI | 90% CI | wygr./przegr. | p_Holm | utrzymuje się? | ±5? |",
          "|---|---|---|---:|---|---|---:|---:|:-:|:-:|"]
    for (lab, who, k, s, (lo90, hi90)), pa in zip(packed, adj):
        holds = pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0)
        inside = -0.05 < lo90 and hi90 < 0.05
        L.append(
            f"| {lab} | {who} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | "
            f"[{lo90:+.3f}, {hi90:+.3f}] | {s['wins']}/{s['losses']} | {pa:.4f} | "
            f"{'tak' if holds else 'nie'} | {'tak' if inside else 'nie'} |"
        )
    # T3
    by = {}
    bad = []
    for (run, mode, arm), rec in done.items():
        by.setdefault(arm, []).append(rec["elapsed_s"])
        info = rec.get("loss") or {}
        if not info.get("finite", False) or not info.get("descend", False):
            bad.append(f"{arm} {mode} przebieg {run}: {info}")
    L += ["", "## T3 — czas i stabilność", "",
          "| ramię | suma czasu | mediana zadania |", "|---|---:|---:|"]
    for a in ARMS:
        xs = by.get(a, [])
        if not xs:
            continue
        xs_s = sorted(xs)
        L.append(f"| {a} | {sum(xs):.0f} s | {xs_s[len(xs_s) // 2]:.2f} s |")
    def tot(prefix):
        return sum(t for a, xs in by.items() if a.startswith(prefix) for t in xs)
    bw, sm = tot("bw"), tot("sm")
    ratio = bw / sm if sm else float("nan")
    L.append("")
    L.append(f"Suma czasu Most / Softmax (ramiona bw* wobec sm*, bez mlp) = {bw:.0f} s / {sm:.0f} s = {ratio:.2f}×.")
    if bad:
        L.append("")
        L.append(f"Porażka T3: {len(bad)} przebiegów bez skończonej, schodzącej straty.")
        L += [f"- {b}" for b in bad[:20]]
    else:
        L.append("")
        L.append("Każdy przebieg ma skończoną stratę i ostatni krok niższy niż pierwszy.")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke()
        return
    xd.make_encoder = make_encoder
    xd.train_strict = train_logged
    meta, content, tracer, fr = domain()
    ml3.D_X = content.shape[1]
    col, row = nn_rate(meta, content, 0), nn_rate(meta, content, 1)
    if col > 0.25:
        raise SystemExit(f"widmo odtwarza pisownię ({col:.0%}) — przerwano przed treningiem")
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["run"], r["mode"], r["arm"])] = r
    if SPEC.is_file():
        spec = json.loads(SPEC.read_text(encoding="utf-8"))
    else:
        spec = xd.build_spec("music-notation", meta, fr, [])
        spec["axes"] = "kolumna=pisownia enharmoniczna, wiersz=oktawa, dół=ta sama pisownia wyżej"
        spec["features"] = "widmo 8 harmonicznych (treść); τ = pisownia, rytm, takt, dynamika, artykulacja, głos"
        spec["raw_sim"] = f"1-NN na samym widmie: ta sama pisownia {col:.0%}, ta sama oktawa {row:.0%}"
        SPEC.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    hsh = fz.spec_hash(spec)
    idx = {m[3]: i for i, m in enumerate(meta)}
    if [m[3] for m in meta] != spec["elements"]:
        raise SystemExit("elementy różne od zamrożonej specyfikacji")
    print(f"[T1] N={len(meta)} spec={hsh[:12]} · {spec['raw_sim']}", flush=True)
    t0 = time.time()
    try:
        if not args.report_only:
            with RUNS.open("a", encoding="utf-8") as f:
                for run in range(xd.N_RUNS):
                    for mode in xd.MODES:
                        for arm in ARMS:
                            if (run, mode, arm) in done:
                                continue
                            t1 = time.time()
                            variants = xd.run_one(spec["runs"][run], mode, arm, content, tracer, idx, spec["mendelejew"])
                            info = dict(LOSS["last"])
                            rec = {
                                "domain": "music-notation", "run": run, "mode": mode, "arm": arm,
                                "spec_sha256": hsh, "elapsed_s": round(time.time() - t1, 2),
                                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "loss": info, "variants": variants,
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            done[(run, mode, arm)] = rec
                    _status(f"RUNNING przebieg {run} ({time.time() - t0:.0f}s)")
                    print(f"  przebieg {run} gotowy ({time.time() - t0:.0f}s)", flush=True)
        report(spec, done)
        _status(f"DONE {time.time() - t0:.0f}s")
        print(f"wrote {REPORT}", flush=True)
    except Exception as e:
        _status(f"FAILED {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
