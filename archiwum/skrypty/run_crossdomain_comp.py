"""Zadanie kompozycyjne na innych zbiorach relacji — czy efekty są specyficzne dla układu okresowego?

Trzy dziedziny z siatką (kolumna = „grupa”, wiersz, kolejność), te same relacje:
  dół   — ta sama kolumna, następny istniejący wiersz
  prawo — następny element w kolejności (przekracza granice wierszy, jak Z+1)
  grupa — ta sama kolumna (głowica relacji / PoE / mix)

  periodic — 118 pierwiastków, cechy ZC (konfiguracja; „dół” bez bloku f), luki Sc/Ga/Ge/Tc w teście
  kana     — 45 znaków gojūon (bez ん; luki yi/ye/wi/wu/we), cechy = bitmapy glifów hiragany+katakany
             (Yu Gothic), τ = statystyki glifu + kolejność; kolumna = samogłoska, wiersz = rząd spółgłoskowy
  music    — 84 nuty MIDI 24–107, cechy = widmo 8 harmonicznych (amplituda 1/k) na skali półtonów,
             τ = cechy widma + kolejność; kolumna = klasa wysokości, wiersz = oktawa

ŚCISŁE ROZDZIELENIE: podczas treningu enkoder przetwarza WYŁĄCZNIE elementy treningowe
(cechy elementów testowych nie wchodzą do żadnego przebiegu w przód); elementy testowe są
dokładane dopiero przy ocenie. Fakty i pary — tylko między elementami treningowymi.

Ramiona: bw·ΔE (Most) · sm·ΔE · bw·0 · sm·0 (jedna klasa enkodera, jak KERNEL) · mlp.
Tryby/warianty jak COMPREL FROZEN; 20 przebiegów; specyfikacja per dziedzina zamrożona (SHA-256).
Testy zadeklarowane z góry (27 na dziedzinę, Holm):
  T1 kompozycja na znanych (comp; B_tr_op, B_tr_dd): bw·ΔE − sm·0, bw·ΔE − sm·ΔE, sm·ΔE − sm·0
  T2 relacja grupy: PoE − comp, A_dół/B_te_hop/B_te_dd, każde z 5 ramion
  T3 jądro w PoE: bw·ΔE − sm·ΔE, 3 miary nowe
  T4 mix − comp dla bw·ΔE, 3 miary nowe

  python archiwum/skrypty/run_crossdomain_comp.py [--domains periodic,kana,music] [--report-only]
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

import run_mendelejew_ml3_zc as zc  # ustawia T_SEAL=False
import run_mendelejew_ml3 as ml3
import run_mendelejew_kernel_ablation as ka
import run_mendelejew_comprel_frozen as fz
from run_mendelejew_comp import MENDELEJEW_Q, Ops, comp_queries, facts as periodic_facts
from run_mendelejew_comprel import query
from run_mendelejew_semi import _nce, split as periodic_split
from config import CANON
from import_struktura_atomow import load_rows

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "crossdomain_runs.jsonl"
REPORT = OUT / "CROSSDOMAIN_COMP.md"
ARMS = ["bw·ΔE", "sm·ΔE", "bw·0", "sm·0", "mlp"]
MODES = ["comp", "rel", "mix"]
VARIANTS = {"comp": ("comp", False), "comp+rel": ("rel", False), "comp+rel·PoE": ("rel", True), "comp+mix": ("mix", False)}
N_RUNS = 20
STEPS = 800
NEW = ["A_dół", "B_te_hop", "B_te_dd"]
KNOWN = ["B_tr_op", "B_tr_dd"]
FONT = "C:/Windows/Fonts/YuGothM.ttc"


# ---------------------------------------------------------------- dziedziny
# meta: lista (kolumna, wiersz, kolejność, symbol)


def domain_periodic():
    content, tracer, _, meta, _ = zc.build(load_rows(zc.SRC))
    return meta, content, tracer, periodic_facts(meta), [[list(r), h, t] for h, r, t in MENDELEJEW_Q]


KANA = [("あ", "ア", "a", "a"), ("い", "イ", "a", "i"), ("う", "ウ", "a", "u"), ("え", "エ", "a", "e"), ("お", "オ", "a", "o"),
        ("か", "カ", "k", "a"), ("き", "キ", "k", "i"), ("く", "ク", "k", "u"), ("け", "ケ", "k", "e"), ("こ", "コ", "k", "o"),
        ("さ", "サ", "s", "a"), ("し", "シ", "s", "i"), ("す", "ス", "s", "u"), ("せ", "セ", "s", "e"), ("そ", "ソ", "s", "o"),
        ("た", "タ", "t", "a"), ("ち", "チ", "t", "i"), ("つ", "ツ", "t", "u"), ("て", "テ", "t", "e"), ("と", "ト", "t", "o"),
        ("な", "ナ", "n", "a"), ("に", "ニ", "n", "i"), ("ぬ", "ヌ", "n", "u"), ("ね", "ネ", "n", "e"), ("の", "ノ", "n", "o"),
        ("は", "ハ", "h", "a"), ("ひ", "ヒ", "h", "i"), ("ふ", "フ", "h", "u"), ("へ", "ヘ", "h", "e"), ("ほ", "ホ", "h", "o"),
        ("ま", "マ", "m", "a"), ("み", "ミ", "m", "i"), ("む", "ム", "m", "u"), ("め", "メ", "m", "e"), ("も", "モ", "m", "o"),
        ("や", "ヤ", "y", "a"), ("ゆ", "ユ", "y", "u"), ("よ", "ヨ", "y", "o"),
        ("ら", "ラ", "r", "a"), ("り", "リ", "r", "i"), ("る", "ル", "r", "u"), ("れ", "レ", "r", "e"), ("ろ", "ロ", "r", "o"),
        ("わ", "ワ", "w", "a"), ("を", "ヲ", "w", "o")]
ROWS = "aksntymrhw"  # placeholder, nadpisane niżej
ROWS = ["a", "k", "s", "t", "n", "h", "m", "y", "r", "w"]
VOWELS = ["a", "i", "u", "e", "o"]


def _glyph(ch, font, size=64, out=12):
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=font)
    d.text(((size - (x1 - x0)) / 2 - x0, (size - (y1 - y0)) / 2 - y0), ch, fill=255, font=font)
    big = torch.tensor(list(img.getdata()), dtype=torch.float).view(size, size) / 255.0
    small = torch.tensor(list(img.resize((out, out), Image.BOX).getdata()), dtype=torch.float) / 255.0
    return big, small


def _components(big):
    m = (big > 0.5).tolist()
    H, W = len(m), len(m[0])
    seen = [[False] * W for _ in range(H)]
    n = 0
    for i in range(H):
        for j in range(W):
            if m[i][j] and not seen[i][j]:
                n += 1
                stack = [(i, j)]
                seen[i][j] = True
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and m[x][y] and not seen[x][y]:
                            seen[x][y] = True
                            stack.append((x, y))
    return n


def domain_kana():
    from PIL import ImageFont
    font = ImageFont.truetype(FONT, 48)
    meta, cs, ts = [], [], []
    ys, xs = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    for order, (h, k, row, vow) in enumerate(KANA):
        hb, hs = _glyph(h, font)
        kb, ks = _glyph(k, font)
        cs.append(F.normalize(torch.cat([hs, ks]), dim=0))
        ink = hb.sum().clamp_min(1e-6)
        tau = torch.tensor([order / len(KANA), float(hb.mean()) * 4, float(kb.mean()) * 4,
                            float((hb * xs).sum() / ink / 64), float((hb * ys).sum() / ink / 64), _components(hb) / 5])
        ts.append(tau)
        meta.append((VOWELS.index(vow) + 1, ROWS.index(row) + 1, order, h))
    return meta, torch.stack(cs), torch.stack(ts), grid_facts(meta), []


def domain_music():
    lo, hi = 24, 107
    bins = torch.arange(lo, hi + 37, dtype=torch.float)  # półtony 24..143
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    meta, cs, ts = [], [], []
    for m in range(lo, hi + 1):
        spec = torch.zeros_like(bins)
        for k in range(1, 9):
            spec += (1 / k) * torch.exp(-0.5 * ((bins - (m + 12 * math.log2(k))) / 0.5) ** 2)
        p = spec / spec.sum()
        cen = float((p * bins).sum())
        spread = float(((p * (bins - cen) ** 2).sum()).sqrt())
        cs.append(F.normalize(spec, dim=0))
        ts.append(torch.tensor([(m - lo) / (hi - lo + 1), (cen - lo) / 120, spread / 24,
                                float(p[bins < m + 12].sum()), float(p[bins >= m + 24].sum()),
                                float((p > p.max() * 0.1).float().mean())]))
        meta.append((m % 12 + 1, m // 12 - 1, m - lo, f"{names[m % 12]}{m // 12 - 1}"))
    return meta, torch.stack(cs), torch.stack(ts), grid_facts(meta), []


def grid_facts(meta):
    """dół: ta sama kolumna, następny istniejący wiersz; prawo: kolejność+1."""
    by_col = {}
    for i, (c, r, o, s) in enumerate(meta):
        by_col.setdefault(c, []).append((r, i))
    down = []
    for c, lst in by_col.items():
        lst.sort()
        down += [(lst[j][1], lst[j + 1][1]) for j in range(len(lst) - 1)]
    by_o = {o: i for i, (c, r, o, s) in enumerate(meta)}
    right = [(by_o[o], by_o[o + 1]) for o in sorted(by_o) if o + 1 in by_o]
    return {"dół": down, "prawo": right}


DOMAINS = {"periodic": domain_periodic, "kana": domain_kana, "music": domain_music}


def generic_split(n, seed):
    g = torch.Generator().manual_seed(50_000 + seed)
    perm = torch.randperm(n, generator=g).tolist()
    k = round(0.3 * n)
    return sorted(perm[k:]), sorted(perm[:k])


# ---------------------------------------------------------------- specyfikacja


def build_spec(name, meta, fr_all, mendel):
    S = lambda i: meta[i][3]
    cq = comp_queries(fr_all)
    runs = []
    for i in range(N_RUNS):
        tr, te = periodic_split(meta, i) if name == "periodic" else generic_split(len(meta), i)
        trs = set(tr)
        q = {}
        for rel in ("dół", "prawo"):
            q[f"A_{rel}"] = [[[rel], S(a), S(b)] for a, b in fr_all[rel] if a not in trs or b not in trs]
        for sp, flt in (("tr", lambda a, x, b: {a, x, b} <= trs), ("te", lambda a, x, b: not {a, x, b} <= trs)):
            qs = [[list(rels), S(a), S(b)] for rels, a, x, b in cq if flt(a, x, b)]
            q[f"B_{sp}"] = qs
            q[f"B_{sp}_dd"] = [x for x in qs if x[0] == ["dół", "dół"]]
        gp = [[S(a), S(b)] for j, a in enumerate(tr) for b in tr[j + 1:] if meta[a][0] == meta[b][0]]
        byo = {meta[x][2]: x for x in tr}
        zp = [[S(byo[o]), S(byo[o + 1])] for o in sorted(byo) if o + 1 in byo]
        runs.append({"run": i, "seed": i, "train": sorted(S(x) for x in tr), "test": sorted(S(x) for x in te),
                     "facts": {r: [[S(a), S(b)] for a, b in v if a in trs and b in trs] for r, v in fr_all.items()},
                     "group_pairs": gp, "z_pairs": zp, "queries": q})
    return {"version": 1, "domain": name, "protocol": "strict-inductive", "steps": STEPS, "mendelejew": mendel,
            "elements": [m[3] for m in meta], "runs": runs}


# ---------------------------------------------------------------- trening ściśle indukcyjny


def make_encoder(arm):
    return ml3.make_encoder("mlp") if arm == "mlp" else ka.make_kernel_encoder(arm)


def train_strict(arm, mode, content, tracer, tr_idx, fr, gp, zp, seed):
    """Przebieg w przód tylko po elementach treningowych; indeksy faktów przemapowane na podzbiór."""
    loc = {g: j for j, g in enumerate(tr_idx)}
    L = lambda pairs: [(loc[a], loc[b]) for a, b in pairs]
    fr_l = {r: L(v) for r, v in fr.items()}
    gp_l, zp_l = L(gp), L(zp)
    c_tr, t_tr = content[tr_idx], tracer[tr_idx]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = make_encoder(arm)
    ops = Ops("lin", CANON.d_model)
    heads = nn.ModuleDict({r: nn.Linear(CANON.d_model, CANON.d_model) for r in ("grupa", "Z")}) if mode == "rel" else nn.ModuleDict()
    opt = torch.optim.Adam(list(enc.parameters()) + list(ops.parameters()) + list(heads.parameters()), lr=CANON.lr)
    N = len(tr_idx)
    for _ in range(STEPS):
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
            s = (ops.apply(h[a], [r]) @ h.T / ml3.TEMP).masked_fill(F.one_hot(a, N).bool(), -1e9)
            loss = loss + F.cross_entropy(s, b)
        if mode == "rel" and gp_l:
            loss = loss + _nce(F.normalize(heads["grupa"](raw), dim=-1), gp_l, gen)
            if zp_l:
                loss = loss + _nce(F.normalize(heads["Z"](raw), dim=-1), zp_l, gen)
        elif mode == "mix" and gp_l:
            loss = loss + _nce(h, gp_l, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():  # ocena: wszystkie elementy (testowe dopiero teraz)
        raw = ml3.encode(enc, arm, content, tracer)
        g = F.normalize(heads["grupa"](raw), dim=-1) if mode == "rel" else None
        return F.normalize(raw, dim=-1), ops, g


def run_one(spec_run, mode, arm, content, tracer, idx, mendel):
    I = lambda s: idx[s]
    tr_idx = [I(s) for s in spec_run["train"]]
    fr = {r: [(I(a), I(b)) for a, b in v] for r, v in spec_run["facts"].items()}
    h, ops, g = train_strict(arm, mode, content, tracer, tr_idx, fr,
                             [(I(a), I(b)) for a, b in spec_run["group_pairs"]],
                             [(I(a), I(b)) for a, b in spec_run["z_pairs"]], spec_run["seed"])
    out = {}
    for vname, (vmode, poe) in VARIANTS.items():
        if vmode != mode:
            continue
        ranks = {}
        for key, qs in spec_run["queries"].items():
            ranks[key + "|op"] = [query(h, ops, g, I(a), tuple(r), I(b), "op", poe) for r, a, b in qs]
            if key in ("B_tr", "B_te"):
                ranks[key + "|hop"] = [query(h, ops, g, I(a), tuple(r), I(b), "hop", poe) for r, a, b in qs]
        ranks["mendelejew"] = [query(h, ops, g, I(a), tuple(r), I(b), "op", poe) for r, a, b in mendel]
        out[vname] = ranks
    return out


# ---------------------------------------------------------------- raport


def stats_family(tests):
    adj = fz.holm([s["p"] for *_, s in tests])
    return list(zip(tests, adj))


def report(specs, done):
    L = [
        "# Zadanie kompozycyjne na innych zbiorach relacji — układ okresowy vs kana vs muzyka",
        "",
        "Te same relacje („dół” = ta sama kolumna, następny wiersz; „prawo” = następny element; „grupa” = ta sama kolumna), "
        "te same ramiona (jedna klasa enkodera z ablacji jądra + MLP), warianty i 800 kroków; 20 przebiegów na dziedzinę; "
        "specyfikacje zamrożone (`crossdomain_spec_<dziedzina>.json`). **Ścisłe rozdzielenie:** w treningu enkoder przetwarza "
        "tylko elementy treningowe — cechy testowych pojawiają się dopiero przy ocenie. "
        "Skrypt `archiwum/skrypty/run_crossdomain_comp.py`; każde uruchomienie w `crossdomain_runs.jsonl`.",
        "",
        "| dziedzina | elementy | kolumny | cechy | podobieństwo w surowych cechach |",
        "|---|---:|---:|---|---|",
    ]
    for name, spec in specs.items():
        n = len(spec["elements"])
        L.append(f"| {name} | {n} | {spec['n_cols']} | {spec['features']} | {spec['raw_sim']} |")
    summary = []
    for name, spec in specs.items():
        val = {}
        for (dom, run, mode, arm), rec in done.items():
            if dom != name:
                continue
            for v, ranks in rec["variants"].items():
                for k in NEW + KNOWN:
                    val[(arm, v, k, run)] = fz.metric(ranks, k)[0]
        runs = range(N_RUNS)
        ser = lambda a, v, k: [val[(a, v, k, r)] for r in runs]

        def diff(a1, v1, a2, v2, k):
            d = [x - y for x, y in zip(ser(a1, v1, k), ser(a2, v2, k)) if not (math.isnan(x) or math.isnan(y))]
            return fz.paired_stats(d) if len(d) >= 5 else None

        L += ["", f"## {name}", "", f"Zapytań na przebieg (średnio): " + ", ".join(
            f"{k}={sum(len(r['queries'][q]) for r in spec['runs']) / N_RUNS:.1f}"
            for k, q in (("A_dół", "A_dół"), ("B_te", "B_te"), ("B_te_dd", "B_te_dd"), ("B_tr", "B_tr"), ("B_tr_dd", "B_tr_dd"))) + ".",
              "", "Średnie hit@1:", "", "| miara | wariant | " + " | ".join(ARMS) + " |", "|---|---|" + "---:|" * len(ARMS)]
        for k in NEW + KNOWN:
            for v in ("comp", "comp+rel·PoE", "comp+mix"):
                cells = []
                for a in ARMS:
                    s = [x for x in ser(a, v, k) if not math.isnan(x)]
                    cells.append(f"{sum(s) / len(s):.3f}" if s else "—")
                L.append(f"| {k} | {v} | " + " | ".join(cells) + " |")
        tests = []
        for k in KNOWN:
            for a1, a2, lab in (("bw·ΔE", "sm·0", "T1 Most − dawny Softmax"), ("bw·ΔE", "sm·ΔE", "T1 jądro"), ("sm·ΔE", "sm·0", "T1 informacja ΔE")):
                tests.append((lab, "comp", k, diff(a1, "comp", a2, "comp", k)))
        for a in ARMS:
            for k in NEW:
                tests.append((f"T2 relacja (PoE−comp) · {a}", "PoE", k, diff(a, "comp+rel·PoE", a, "comp", k)))
        for k in NEW:
            tests.append(("T3 jądro w PoE", "PoE", k, diff("bw·ΔE", "comp+rel·PoE", "sm·ΔE", "comp+rel·PoE", k)))
        for k in NEW:
            tests.append(("T4 mix−comp · bw·ΔE", "mix", k, diff("bw·ΔE", "comp+mix", "bw·ΔE", "comp", k)))
        ok_tests = [t for t in tests if t[3] is not None]
        fam = stats_family(ok_tests)
        L += ["", f"Testy zadeklarowane z góry ({len(ok_tests)}, Holm):", "",
              "| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |", "|---|---|---:|---|---:|---:|:-:|"]
        for (lab, v, k, s), pa in fam:
            ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
            L.append(f"| {lab} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | {pa:.4f} | {ok} |")
            summary.append((name, lab, k, s, pa))
    # zestawienie
    L += ["", "## Zestawienie między dziedzinami (Δ, w nawiasie: utrzymuje się po Holmie?)", "",
          "| test | miara | " + " | ".join(specs) + " |", "|---|---|" + "---:|" * len(specs)]
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


def raw_similarity(meta, content):
    """Czy najbliższy sąsiad w surowej treści jest z tej samej kolumny, czy z tego samego wiersza?"""
    h = F.normalize(content, dim=-1)
    s = (h @ h.T).fill_diagonal_(-2)
    nn1 = s.argmax(-1).tolist()
    col = sum(meta[i][0] == meta[j][0] for i, j in enumerate(nn1)) / len(meta)
    row = sum(meta[i][1] == meta[j][1] for i, j in enumerate(nn1)) / len(meta)
    return f"1-NN: ta sama kolumna {col:.0%}, ten sam wiersz {row:.0%}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", default="periodic,kana,music")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    names = args.domains.split(",")
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["domain"], r["run"], r["mode"], r["arm"])] = r
    specs = {}
    t0 = time.time()
    for name in names:
        meta, content, tracer, fr_all, mendel = DOMAINS[name]()
        ml3.D_X = content.shape[1]
        sp_path = OUT / f"crossdomain_spec_{name}.json"
        if sp_path.is_file():
            spec = json.loads(sp_path.read_text(encoding="utf-8"))
        else:
            spec = build_spec(name, meta, fr_all, mendel)
            sp_path.write_text(json.dumps(spec, ensure_ascii=False, indent=0), encoding="utf-8")
        h = fz.spec_hash(spec)
        idx = {m[3]: i for i, m in enumerate(meta)}
        if [m[3] for m in meta] != spec["elements"]:
            raise SystemExit(f"{name}: elementy różne od zamrożonej specyfikacji")
        print(f"[{name}] N={len(meta)} d_x={content.shape[1]} fakty dół={len(fr_all['dół'])} prawo={len(fr_all['prawo'])} "
              f"spec={h[:12]} · {raw_similarity(meta, content)}", flush=True)
        for (d, *_), r in done.items():
            if d == name and r["spec_sha256"] != h:
                raise SystemExit(f"{name}: rekord z innym hashem — przerwano")
        if not args.report_only:
            with RUNS.open("a", encoding="utf-8") as f:
                for run in range(N_RUNS):
                    for mode in MODES:
                        for arm in ARMS:
                            if (name, run, mode, arm) in done:
                                continue
                            t1 = time.time()
                            variants = run_one(spec["runs"][run], mode, arm, content, tracer, idx, spec["mendelejew"])
                            rec = {"domain": name, "run": run, "mode": mode, "arm": arm, "spec_sha256": h,
                                   "elapsed_s": round(time.time() - t1, 2),
                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "variants": variants}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            f.flush()
                            done[(name, run, mode, arm)] = rec
                    print(f"  [{name}] przebieg {run} gotowy ({time.time() - t0:.0f}s)", flush=True)
        spec["n_cols"] = len({m[0] for m in meta})
        spec["features"] = {"periodic": "konfiguracja elektronowa (24) + τ6", "kana": "bitmapy glifów 2×12×12 + τ6",
                            "music": "widmo 8 harmonicznych (120 półtonów) + τ6"}[name]
        spec["raw_sim"] = raw_similarity(meta, content)
        specs[name] = spec
    report(specs, done)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
