"""Mendelejew KERNEL — czysta ablacja jądra uwagi: zmienia się WYŁĄCZNIE funkcja wag.

Jedna klasa enkodera (`KernelEncoder` = PhiBridgeEncoder z przełącznikiem jądra):
te same moduły, parametry, inicjalizacja (sprawdzane: identyczny state_dict przy tym samym
ziarnie), te same wejścia (treść w tokenach, tracer tylko przez ΔE), te same dane
(zamrożona specyfikacja COMPREL FROZEN, weryfikacja SHA-256) i procedura (800 kroków,
warianty comp / comp+rel / comp+rel·PoE / comp+mix, 20 przebiegów).

Projekt 2×2 (jądro × dostęp do ΔE):
  bw·ΔE  — A = γC/(γ²+ΔE²), C = softplus(S)²         (≡ dawny „most”)
  bw·0   — j.w., ΔE = 0                               (≡ dawny „zero”)
  sm·ΔE  — w = softmax(S − ΔE²/γ0²)  (ta sama informacja ΔE i ta sama szerokość γ0)
  sm·0   — w = softmax(S)
Brak samo-uwagi w każdym jądrze (przekątna wyłączona), T-seal wyłączony (wariant ZC).

Kontrast główny (zadeklarowany przed uruchomieniem): bw·ΔE − sm·ΔE
  na A_dół, B_te_hop, B_te_dd, B_tr_op, B_tr_dd × {comp, comp+mix, comp+rel·PoE} = 15 testów, Holm.
Dodatkowe: bw·0 − sm·0 (jądro bez informacji ΔE), efekt ΔE w każdym jądrze, interakcja.
Kontrola: bw·ΔE i bw·0 mają odtworzyć dawne most/zero z comprel_frozen_runs.jsonl co do cyfry.

  python archiwum/skrypty/run_mendelejew_kernel_ablation.py [--report-only]
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
import torch.nn.functional as F

import run_mendelejew_ml3_zc as zc  # ustawia ml3.D_X / T_SEAL dla ZC
import run_mendelejew_ml3 as ml3
import run_mendelejew_comprel as comprel
import run_mendelejew_comprel_frozen as fz
from breit_kernel import breit_wigner_rows
from config import CANON
from import_struktura_atomow import load_rows
from precision import assert_finite
from recall_model import PhiBridgeEncoder, _key_padding

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "kernel_ablation_runs.jsonl"
REPORT = OUT / "MENDELEJEW_KERNEL_ABLATION.md"
KERNELS = ["bw·ΔE", "bw·0", "sm·ΔE", "sm·0"]
VARIANTS = ["comp", "comp+rel", "comp+rel·PoE", "comp+mix"]
MAIN_VARIANTS = ["comp", "comp+mix", "comp+rel·PoE"]
MAIN_METRICS = ["A_dół", "B_te_hop", "B_te_dd", "B_tr_op", "B_tr_dd"]
OLD_ARM = {"bw·ΔE": "most", "bw·0": "zero"}


class KernelEncoder(PhiBridgeEncoder):
    """PhiBridgeEncoder z przełącznikiem jądra; moduły i parametry identyczne dla każdego jądra."""

    def __init__(self, kernel: str, use_de: bool, **kw):
        super().__init__(**kw)
        self.kernel = kernel
        self.use_de = use_de

    def forward(self, x_raw, tracer, mask=None):
        B, N0, _ = x_raw.shape
        x = self.embed(x_raw)
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
                logits = S - dE.square() / gamma0.square()
                logits = logits.masked_fill(eye, -1e9)
                if key_pad is not None:
                    logits = logits.masked_fill(key_pad, -1e9)
                w = F.softmax(logits, dim=-1)
            assert_finite(w, f"kernel.w[L{li}]")
            y = (w @ v).transpose(1, 2).contiguous().view(B, N, -1)
            x = x + layer["o"](y)
            x = x + layer["ff"](x)
        return self.out_norm(x[:, 1:, :]), None


def make_kernel_encoder(arm: str):
    kern, de = arm.split("·")
    return KernelEncoder(kern, de == "ΔE", input_dim=ml3.D_X, dim_tracer=CANON.n_dim, num_layers=CANON.num_layers,
                         d=CANON.d_model, coupling=CANON.coupling, t_seal=False)


comprel.make_encoder = make_kernel_encoder  # train() w COMPREL buduje enkoder przez tę nazwę


def init_identity_check():
    states = []
    for k in KERNELS:
        torch.manual_seed(0)
        states.append(make_kernel_encoder(k).state_dict())
    ref = states[0]
    same = all(set(s) == set(ref) and all(torch.equal(s[n], ref[n]) for n in ref) for s in states[1:])
    n_par = sum(p.numel() for p in make_kernel_encoder(KERNELS[0]).parameters())
    return same, n_par


def load_done():
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["run"], r["mode"], r["arm"])] = r
    return done


def report(spec, h, done, n_runs, init_ok, n_par):
    runs = list(range(n_runs))
    keys = sorted(set(fz.PRIMARY + fz.SECONDARY + ["B_tr_hop", "B_tr_dd"]))
    val = {}
    for (run, mode, arm), rec in done.items():
        for v, ranks in rec["variants"].items():
            for k in keys:
                val.setdefault((v, arm, k), {})[run] = fz.metric(ranks, k)[0]
    ser = lambda v, a, k: [val[(v, a, k)][r] for r in runs]

    # kontrola odtworzenia dawnych armii
    old = fz.load_done()
    repro_tot = repro_eq = 0
    for (run, mode, arm), rec in done.items():
        oa = OLD_ARM.get(arm)
        if oa and (run, mode, oa) in old:
            for v, ranks in rec["variants"].items():
                for qk, rk in ranks.items():
                    repro_tot += 1
                    repro_eq += rk == old[(run, mode, oa)]["variants"][v][qk]

    def fam(a1, a2, variants, metrics):
        out = []
        for v in variants:
            for k in metrics:
                d = [x - y for x, y in zip(ser(v, a1, k), ser(v, a2, k))]
                out.append((v, k, fz.paired_stats(d)))
        return out, fz.holm([x[2]["p"] for x in out])

    L = [
        "# Mendelejew KERNEL — czysta ablacja jądra uwagi (Softmax ↔ Most)",
        "",
        "Zmienia się **wyłącznie funkcja wag uwagi**. Te same dane (zamrożona specyfikacja COMPREL FROZEN, "
        f"SHA-256 `{h[:16]}…`), te same reprezentacje (treść w tokenach, tracer tylko przez ΔE), te same moduły i parametry "
        f"({n_par} parametrów w każdym jądrze), identyczna inicjalizacja przy tym samym ziarnie "
        f"(sprawdzone: state_dict równy dla wszystkich 4 jąder → **{'tak' if init_ok else 'NIE'}**), ta sama procedura "
        f"(800 kroków, {n_runs} przebiegów). Skrypt `archiwum/skrypty/run_mendelejew_kernel_ablation.py`; każde uruchomienie "
        "w `kernel_ablation_runs.jsonl`.",
        "",
        "| | ΔE wyłączone | ΔE włączone |",
        "|---|---|---|",
        "| **Softmax** | `sm·0`: softmax(S) | `sm·ΔE`: softmax(S − ΔE²/γ0²) |",
        "| **Most (Breit–Wigner)** | `bw·0`: γC/γ², C = softplus(S)² | `bw·ΔE`: γC/(γ²+ΔE²) |",
        "",
        f"**Kontrola odtworzenia:** `bw·ΔE` i `bw·0` vs dawne `most`/`zero` z COMPREL FROZEN — identycznych list pozycji: "
        f"{repro_eq}/{repro_tot}.",
        "",
        "## 1. Średnie hit@1 (± odch. std. po przebiegach)",
        "",
        "| miara | wariant | " + " | ".join(KERNELS) + " |",
        "|---|---|" + "---:|" * len(KERNELS),
    ]
    for k in MAIN_METRICS:
        for v in MAIN_VARIANTS:
            cells = []
            for a in KERNELS:
                s = ser(v, a, k)
                m = sum(s) / len(s)
                sd = math.sqrt(sum((x - m) ** 2 for x in s) / (len(s) - 1))
                cells.append(f"{m:.3f} ± {sd:.3f}")
            L.append(f"| {k} | {v} | " + " | ".join(cells) + " |")

    def table(title, a1, a2, note=""):
        out, adj = fam(a1, a2, MAIN_VARIANTS, MAIN_METRICS)
        L.extend(["", f"## {title}", ""] + ([note, ""] if note else []) + [
            "| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |",
            "|---|---|---:|---|---:|---:|---:|:-:|"])
        for (v, k, s), pa in zip(out, adj):
            ok = "**tak**" if pa < 0.05 and (s["ci"][0] > 0 or s["ci"][1] < 0) else "nie"
            L.append(f"| {v} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['wins']}/{s['losses']} | "
                     f"{s['p']:.4f} | {pa:.4f} | {ok} |")
        return out, adj

    main = table("2. KONTRAST GŁÓWNY — samo jądro przy tej samej informacji: bw·ΔE − sm·ΔE (15 testów, Holm)",
                 "bw·ΔE", "sm·ΔE")
    table("3. Jądro bez informacji ΔE: bw·0 − sm·0 (15 testów, Holm)", "bw·0", "sm·0")
    table("4. Efekt informacji ΔE w jądrze Mostu: bw·ΔE − bw·0 (15 testów, Holm)", "bw·ΔE", "bw·0")
    table("5. Efekt informacji ΔE w jądrze Softmax: sm·ΔE − sm·0 (15 testów, Holm)", "sm·ΔE", "sm·0")
    # interakcja
    L += ["", "## 6. Interakcja: (bw·ΔE − sm·ΔE) − (bw·0 − sm·0) — czy jądro BW korzysta z ΔE lepiej niż Softmax", "",
          "| wariant | miara | Δ średnia | 95% CI | p |", "|---|---|---:|---|---:|"]
    for v in MAIN_VARIANTS:
        for k in MAIN_METRICS:
            d = [(a - b) - (c - e) for a, b, c, e in zip(ser(v, "bw·ΔE", k), ser(v, "sm·ΔE", k), ser(v, "bw·0", k), ser(v, "sm·0", k))]
            s = fz.paired_stats(d)
            L.append(f"| {v} | {k} | {s['mean']:+.3f} | [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['p']:.4f} |")
    # porównanie z dawnym bliźniakiem
    old_sm = {}
    for (run, mode, arm), rec in old.items():
        if arm == "softmax" and run < n_runs:
            for v, ranks in rec["variants"].items():
                for k in MAIN_METRICS:
                    old_sm.setdefault((v, k), {})[run] = fz.metric(ranks, k)[0]
    if old_sm:
        L += ["", "## 7. Dawny bliźniak Softmax (bez tracera, inne parametry) vs dopasowany `sm·0` i `sm·ΔE` — średnie hit@1", "",
              "| wariant | miara | dawny softmax | sm·0 | sm·ΔE | bw·ΔE (Most) |", "|---|---|---:|---:|---:|---:|"]
        for v in MAIN_VARIANTS:
            for k in MAIN_METRICS:
                o = [old_sm[(v, k)][r] for r in runs]
                L.append(f"| {v} | {k} | {sum(o) / n_runs:.3f} | {sum(ser(v, 'sm·0', k)) / n_runs:.3f} | "
                         f"{sum(ser(v, 'sm·ΔE', k)) / n_runs:.3f} | {sum(ser(v, 'bw·ΔE', k)) / n_runs:.3f} |")
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    rows = load_rows(zc.SRC)
    content, tracer, _, meta, _ = zc.build(rows)
    idx = {m[3]: i for i, m in enumerate(meta)}
    spec = json.loads(fz.SPEC.read_text(encoding="utf-8"))
    h = fz.spec_hash(spec)
    print(f"spec sha256={h}")
    init_ok, n_par = init_identity_check()
    print(f"init identyczna dla 4 jąder: {init_ok} · parametrów: {n_par}")
    if not init_ok:
        raise SystemExit("różna inicjalizacja jąder — przerwano")
    done = load_done()
    if any(r["spec_sha256"] != h for r in done.values()):
        raise SystemExit("rekordy z innym hashem specyfikacji — przerwano")
    t0 = time.time()
    if not args.report_only:
        with RUNS.open("a", encoding="utf-8") as f:
            for run in range(args.runs):
                for mode in fz.MODES:
                    for arm in KERNELS:
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
    report(spec, h, done, args.runs, init_ok, n_par)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
