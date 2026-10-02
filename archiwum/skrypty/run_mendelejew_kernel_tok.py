"""Mendelejew KERNEL·TOK — Softmax dostaje informację z tracera W TOKENACH; powtórka ablacji jądra.

Rozszerza `run_mendelejew_kernel_ablation.py`: ta sama klasa enkodera z dodatkowym rzutem
tracera do embeddingu tokenu (x = embed(treść) + W_τ·τ), obecnym we WSZYSTKICH ramionach
(nieużywanym tam, gdzie tok wyłączone) → identyczne parametry i inicjalizacja. W_τ jest
inicjalizowany własnym generatorem, więc nie zmienia losowania pozostałych parametrów:
dawne ramiona odtwarzają się co do cyfry (sprawdzane na przebiegu 0).

Ramiona (jądro · dostęp do tracera):
  dawne:  bw·ΔE (Most) · sm·ΔE · bw·0 · sm·0              — wyniki z kernel_ablation_runs.jsonl
  nowe:   sm·tok · bw·tok · sm·tok+ΔE · bw·tok+ΔE         — kernel_tok_runs.jsonl
  tok = τ w tokenach; ΔE = ΔE w wagach uwagi (BW: lorentzowsko, softmax: −ΔE²/γ0²).

Kontrasty zadeklarowane przed uruchomieniem (każdy: 5 miar × 3 warianty = 15 testów, Holm):
  P1  bw·ΔE − sm·tok        Most vs Softmax z tymi samymi danymi w standardowy sposób (w tokenach)
  P2  bw·tok − sm·tok       samo jądro, gdy informacja jest w tokenach
Dodatkowe: sm·tok − sm·ΔE (jak lepiej dać informację Softmaxowi), bw·tok+ΔE − sm·tok+ΔE,
           bw·tok+ΔE − bw·ΔE (czy Mostowi pomaga τ w tokenach).

  python archiwum/skrypty/run_mendelejew_kernel_tok.py [--report-only]
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

import run_mendelejew_kernel_ablation as ka  # ustawia ZC i patch comprel.make_encoder
import run_mendelejew_ml3 as ml3
import run_mendelejew_comprel as comprel
import run_mendelejew_comprel_frozen as fz
from breit_kernel import breit_wigner_rows
from config import CANON
from import_struktura_atomow import load_rows
from precision import assert_finite
from recall_model import _key_padding
import run_mendelejew_ml3_zc as zc

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "kernel_tok_runs.jsonl"
REPORT = OUT / "MENDELEJEW_KERNEL_TOK.md"
OLD_ARMS = ["bw·ΔE", "sm·ΔE", "bw·0", "sm·0"]
NEW_ARMS = ["sm·tok", "bw·tok", "sm·tok+ΔE", "bw·tok+ΔE"]
ALL_ARMS = OLD_ARMS + NEW_ARMS


class TokKernelEncoder(ka.KernelEncoder):
    def __init__(self, kernel, use_de, use_tok, **kw):
        super().__init__(kernel, use_de, **kw)
        self.use_tok = use_tok
        rng = torch.get_rng_state()  # konstruktor Linear losuje z globalnego RNG — przywracamy stan
        self.tok_proj = nn.Linear(CANON.n_dim, CANON.d_model)
        torch.set_rng_state(rng)
        g = torch.Generator().manual_seed(777)  # własny generator: globalny RNG nietknięty
        bound = 1 / math.sqrt(CANON.n_dim)
        with torch.no_grad():
            self.tok_proj.weight.uniform_(-bound, bound, generator=g)
            self.tok_proj.bias.uniform_(-bound, bound, generator=g)

    def forward(self, x_raw, tracer, mask=None):
        B, N0, _ = x_raw.shape
        x = self.embed(x_raw)
        if self.use_tok:
            x = x + self.tok_proj(tracer)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)
        tracer = torch.cat([torch.zeros(B, 1, tracer.shape[-1], device=x.device), tracer], dim=1)
        N = x.shape[1]
        key_pad = _key_padding(mask, B, N0, x.device)
        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        for li, layer in enumerate(self.layers):
            q, k, v = layer["qkv"](layer["n1"](x)).chunk(3, dim=-1)
            q, k, v = [t.view(B, N, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
            S = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
            E = layer["e"](tracer).transpose(1, 2).unsqueeze(-1)
            dE = self.e_scale * (E - E.transpose(-2, -1))
            if not self.use_de:
                dE = torch.zeros_like(dE)
            if self.kernel == "bw":
                C = self._couple(S).masked_fill(eye, 0.0)
                if key_pad is not None:
                    C = C.masked_fill(key_pad, 0.0)
                A = breit_wigner_rows(C, dE, self.log_gammas[li], self.beta_raw[li])
                w = A / A.sum(-1, keepdim=True).clamp_min(1e-12)
            else:
                gamma0 = F.softplus(self.log_gammas[li]) + 1e-3
                logits = (S - dE.square() / gamma0.square()).masked_fill(eye, -1e9)
                if key_pad is not None:
                    logits = logits.masked_fill(key_pad, -1e9)
                w = F.softmax(logits, dim=-1)
            assert_finite(w, f"kernel.w[L{li}]")
            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
        return self.out_norm(x[:, 1:, :]), None


def make_tok_encoder(arm: str):
    kern, rest = arm.split("·")
    return TokKernelEncoder(kern, "ΔE" in rest, "tok" in rest, input_dim=ml3.D_X, dim_tracer=CANON.n_dim,
                            num_layers=CANON.num_layers, d=CANON.d_model, coupling=CANON.coupling, t_seal=False)


comprel.make_encoder = make_tok_encoder


def init_identity_check():
    states = []
    for a in ALL_ARMS:
        torch.manual_seed(0)
        states.append(make_tok_encoder(a).state_dict())
    ref = states[0]
    same = all(set(s) == set(ref) and all(torch.equal(s[n], ref[n]) for n in ref) for s in states[1:])
    torch.manual_seed(0)
    old = ka.make_kernel_encoder("bw·ΔE").state_dict()
    torch.manual_seed(0)
    new = make_tok_encoder("bw·ΔE").state_dict()
    old_same = all(torch.equal(old[n], new[n]) for n in old)
    rng_after = []
    for f in (lambda: ka.make_kernel_encoder("bw·ΔE"), lambda: make_tok_encoder("bw·ΔE")):
        torch.manual_seed(0)
        f()
        rng_after.append(torch.randn(3))
    return same, old_same and torch.equal(*rng_after), sum(p.numel() for p in make_tok_encoder("sm·tok").parameters())


def load(path):
    done = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["run"], r["mode"], r["arm"])] = r
    return done


def report(h, done, n_runs, init_ok, compat_ok, n_par, repro):
    runs = list(range(n_runs))
    keys = ka.MAIN_METRICS
    val = {}
    for (run, mode, arm), rec in done.items():
        for v, ranks in rec["variants"].items():
            for k in keys:
                val.setdefault((v, arm, k), {})[run] = fz.metric(ranks, k)[0]
    ser = lambda v, a, k: [val[(v, a, k)][r] for r in runs]
    L = [
        "# Mendelejew KERNEL·TOK — Softmax z informacją tracera w tokenach; powtórka ablacji jądra",
        "",
        f"Ta sama klasa enkodera dla wszystkich 8 ramion ({n_par} parametrów; identyczna inicjalizacja: "
        f"**{'tak' if init_ok else 'NIE'}**; nowy rzut W_τ nie zmienia losowania reszty: **{'tak' if compat_ok else 'NIE'}**). "
        f"Dane: zamrożona specyfikacja (SHA-256 `{h[:16]}…`), 800 kroków, {n_runs} przebiegów. "
        "Skrypt `archiwum/skrypty/run_mendelejew_kernel_tok.py`; nowe uruchomienia w `kernel_tok_runs.jsonl`, "
        "dawne ramiona z `kernel_ablation_runs.jsonl`.",
        "",
        f"**Kontrola odtworzenia dawnych ramion nową klasą (przebieg 0, bw·ΔE i sm·ΔE, 3 tryby):** {repro}.",
        "",
        "`tok` = τ w embeddingu tokenu (x = embed(treść) + W_τ·τ); `ΔE` = ΔE w wagach uwagi.",
        "",
        "## 1. Średnie hit@1 (20 przebiegów)",
        "",
        "| miara | wariant | " + " | ".join(ALL_ARMS) + " |",
        "|---|---|" + "---:|" * len(ALL_ARMS),
    ]
    for k in keys:
        for v in ka.MAIN_VARIANTS:
            L.append(f"| {k} | {v} | " + " | ".join(f"{sum(ser(v, a, k)) / n_runs:.3f}" for a in ALL_ARMS) + " |")

    def table(title, a1, a2):
        out = []
        for v in ka.MAIN_VARIANTS:
            for k in keys:
                out.append((v, k, fz.paired_stats([x - y for x, y in zip(ser(v, a1, k), ser(v, a2, k))])))
        adj = fz.holm([x[2]["p"] for x in out])
        L.extend(["", f"## {title}", "", "| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |",
                  "|---|---|---:|---|---:|---:|:-:|"])
        for (v, k, s), pa in zip(out, adj):
            ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
            L.append(f"| {v} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | {pa:.4f} | {ok} |")

    table("2. P1 (główny) — Most vs Softmax z τ w tokenach: bw·ΔE − sm·tok", "bw·ΔE", "sm·tok")
    table("3. P2 (główny) — samo jądro przy τ w tokenach: bw·tok − sm·tok", "bw·tok", "sm·tok")
    table("4. Jak dać informację Softmaxowi: sm·tok − sm·ΔE", "sm·tok", "sm·ΔE")
    table("5. Jądro przy τ w tokenach + ΔE w uwadze: bw·tok+ΔE − sm·tok+ΔE", "bw·tok+ΔE", "sm·tok+ΔE")
    table("6. Czy Mostowi pomaga τ w tokenach: bw·tok+ΔE − bw·ΔE", "bw·tok+ΔE", "bw·ΔE")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    content, tracer, _, meta, _ = zc.build(load_rows(zc.SRC))
    idx = {m[3]: i for i, m in enumerate(meta)}
    spec = json.loads(fz.SPEC.read_text(encoding="utf-8"))
    h = fz.spec_hash(spec)
    init_ok, compat_ok, n_par = init_identity_check()
    print(f"spec={h[:16]} init identyczna (8 ramion): {init_ok} · zgodność z dawną klasą: {compat_ok} · param={n_par}", flush=True)
    if not (init_ok and compat_ok):
        raise SystemExit("inicjalizacja niezgodna — przerwano")
    old = load(ka.RUNS)
    done = load(RUNS)
    if any(r["spec_sha256"] != h for r in list(old.values()) + list(done.values())):
        raise SystemExit("inny hash specyfikacji — przerwano")

    # kontrola odtworzenia dawnych ramion nową klasą (przebieg 0)
    eq = tot = 0
    for mode in fz.MODES:
        for arm in ("bw·ΔE", "sm·ΔE"):
            v = fz.run_one(spec["runs"][0], mode, arm, content, tracer, idx, spec["mendelejew"])
            for vn, ranks in v.items():
                for qk, rk in ranks.items():
                    tot += 1
                    eq += rk == old[(0, mode, arm)]["variants"][vn][qk]
    repro = f"{eq}/{tot} list pozycji identycznych"
    print(f"odtworzenie: {repro}", flush=True)
    if eq != tot:
        raise SystemExit("dawne ramiona nie odtwarzają się — przerwano")

    t0 = time.time()
    if not args.report_only:
        with RUNS.open("a", encoding="utf-8") as f:
            for run in range(args.runs):
                for mode in fz.MODES:
                    for arm in NEW_ARMS:
                        if (run, mode, arm) in done:
                            continue
                        t1 = time.time()
                        variants = fz.run_one(spec["runs"][run], mode, arm, content, tracer, idx, spec["mendelejew"])
                        rec = {"run": run, "mode": mode, "arm": arm, "spec_sha256": h, "steps": fz.STEPS,
                               "elapsed_s": round(time.time() - t1, 2),
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "variants": variants}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        done[(run, mode, arm)] = rec
                print(f"  przebieg {run} gotowy ({time.time() - t0:.0f}s)", flush=True)
    merged = {k: v for k, v in old.items() if k[2] in OLD_ARMS}
    merged.update(done)
    report(h, merged, args.runs, init_ok, compat_ok, n_par, repro)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
