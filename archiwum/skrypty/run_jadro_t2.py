"""T2 — tiny LM, ta sama informacja dla obu jąder.

Zadeklarowane przed seedem 0 (nie dobierane po wyniku):

  Korpus: sześć dokumentów. Wspólny prefiks, ogon zależny od źródła
  (aaa…, bbb…, …). Numer źródła 0..5 nie jest znakiem w tekście.
  Prefiks sam nie rozstrzyga źródła.

  A  bw·0 wobec sm·0 — same tokeny, Most z zerowym ΔE, Softmax bez pluginu
  B  bw·tok wobec sm·tok — ten sam numer źródła w tokenie obu jąder, Most nadal bez ΔE
  C  Most kanoniczny (tracer pozycji przez ΔE) wobec Softmaxu bez pluginu — porównanie mieszane

  Margines straty ±0.05 (skala opublikowanej różnicy średniego LM 0.058, zaokrąglona w dół).
  Równoważność A i B: 90% CI różnicy średniej straty (Most − Softmax) w całości w ±0.05.
  Plugin: Softmax B ma niższą stratę niż Softmax A, a 90% CI różnicy (B − A) leży pod zerem.

  20 seedów, 300 kroków, d=64, L=2, jak smoke #1b. Średniego LM tu nie ma.

  python archiwum/skrypty/run_jadro_t2.py [--smoke]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_LAB = _Path(__file__).resolve().parents[2]
_HERE = _Path(__file__).resolve().parent
for _p in (_LAB, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import math
import time

import torch

import analyze_kernel_equivalence as eq
from lm_tiny import TinyLM, TinyLMConfig, lm_loss

OUT = _LAB / "doswiadczenie" / "glify"
RUNS = OUT / "jadro_t2_runs.jsonl"
SPEC = OUT / "jadro_t2_spec.json"
REPORT = OUT / "JADRO_T2.md"
STATUS = OUT / "jadro_t2_status.txt"

PREFIX = "COMMON-PREFIX-0123456789-ABCDEFGHIJ-"
TAILS = ("aaa", "bbb", "ccc", "ddd", "eee", "fff")
N_SEEDS = 20
STEPS = 300
MARGIN = 0.05
ARMS = ("A-bw", "A-sm", "B-bw", "B-sm", "C-bw")


def build_stream():
    ids, src = [], []
    for s, tail in enumerate(TAILS):
        text = PREFIX + (tail * 40) + "\n"
        row = [min(b, 127) for b in text.encode("utf-8")]
        ids.extend(row * 8)
        src.extend([s] * (len(row) * 8))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(src, dtype=torch.long)


def make_batches(data, source, *, seq_len, batch_size, n_batches, seed):
    g = torch.Generator().manual_seed(seed)
    max_start = max(1, data.numel() - seq_len - 1)
    batches, sides = [], []
    for _ in range(n_batches):
        starts = torch.randint(0, max_start, (batch_size,), generator=g)
        batches.append(torch.stack([data[s : s + seq_len] for s in starts.tolist()]))
        sides.append(torch.stack([source[s : s + seq_len] for s in starts.tolist()]))
    return batches, sides


def make_model(arm: str, seed: int) -> TinyLM:
    side = 6 if arm.startswith("B-") else 0
    attn = "most" if arm.endswith("bw") else "softmax"
    cfg = TinyLMConfig(vocab_size=128, d=64, h=2, num_layers=2, n_dim=4, max_seq=128, side_sources=side)
    torch.manual_seed(seed)
    model = TinyLM(cfg, attn=attn)
    model.zero_de = arm != "C-bw"
    return model


def train_one(model, batches, sides, steps, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    it = 0
    first = last = None
    use_side = model.side is not None
    while it < steps:
        for batch, side in zip(batches, sides):
            if it >= steps:
                break
            opt.zero_grad()
            logits = model(batch, side=side if use_side else None, zero_de=model.zero_de)
            loss = lm_loss(logits, batch)
            val = float(loss.detach())
            if it == 0:
                first = val
            last = val
            loss.backward()
            opt.step()
            it += 1
    finite = first is not None and last is not None and math.isfinite(first) and math.isfinite(last)
    return {"loss_first": first, "loss_final": last, "finite": finite, "descend": bool(finite and last < first)}


def _status(text):
    STATUS.write_text(text + "\n", encoding="utf-8")


def paired(done, a, b):
    d = []
    for s in range(N_SEEDS):
        x, y = done.get((s, a)), done.get((s, b))
        if not x or not y:
            continue
        d.append(x["loss_final"] - y["loss_final"])
    return d


def report(done, spec):
    def line(name, d):
        if len(d) < 5:
            return f"| {name} | — | — | — |"
        lo, hi = eq.ci90(d)
        mean = sum(d) / len(d)
        inside = -MARGIN < lo and hi < MARGIN
        return f"| {name} | {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'tak' if inside else 'nie'} |"

    dA = paired(done, "A-bw", "A-sm")
    dB = paired(done, "B-bw", "B-sm")
    dP = paired(done, "B-sm", "A-sm")
    dC = paired(done, "C-bw", "A-sm")
    times = {}
    bad = []
    for (seed, arm), rec in done.items():
        times.setdefault(arm, []).append(rec["wall_s"])
        if not rec["finite"] or not rec["descend"]:
            bad.append(f"{arm} seed {seed}")
    bw = sum(sum(times[a]) for a in times if a.endswith("bw"))
    sm = sum(sum(times[a]) for a in times if a.endswith("sm"))
    L = [
        "# T2 — tiny LM, ta sama informacja dla obu jąder",
        "",
        "Zadeklarowane przed seedem 0. Sześć dokumentów ze wspólnym prefiksem "
        f"`{PREFIX}` i ogonem `aaa`…`fff`. Numer źródła nie jest znakiem w tekście. "
        "Margines straty **±0.05**. 20 seedów, 300 kroków, d=64, L=2, jak smoke #1b. "
        "Średniego LM nie ma.",
        "",
        f"Specyfikacja `{SPEC.name}`. Przebiegi `jadro_t2_runs.jsonl`.",
        "",
        "| porównanie | Δ straty | 90% CI | w ±0.05? |",
        "|---|---:|---|:-:|",
        line("A  Most·0 − Softmax·0", dA),
        line("B  Most·tok − Softmax·tok", dB),
        line("plugin  Softmax·tok − Softmax·0", dP),
        line("C  Most kanoniczny − Softmax·0 (mieszane)", dC),
        "",
        "Równoważność wzoru czytamy z A i z B. C nie jest werdyktem o jądrze. "
        "Plugin działa, gdy średnia różnica Softmax·tok − Softmax·0 jest ujemna i cały 90% CI leży pod zerem.",
        "",
        "## Strata końcowa (średnia z seedów)",
        "",
        "| ramię | strata | czas sumaryczny |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        xs = [done[(s, arm)]["loss_final"] for s in range(N_SEEDS) if (s, arm) in done]
        tt = times.get(arm, [])
        if xs:
            L.append(f"| {arm} | {sum(xs) / len(xs):.4f} | {sum(tt):.0f} s |")
    L.append("")
    L.append(f"Czas Most / Softmax = {bw:.0f} s / {sm:.0f} s = {bw / sm:.2f}×." if sm else "brak czasu Softmaxu")
    if bad:
        L += ["", f"Porażka T3: {len(bad)} przebiegów."] + [f"- {b}" for b in bad[:20]]
    else:
        L += ["", "Każdy przebieg ma skończoną stratę i ostatni krok niższy niż pierwszy."]
    L += ["", "<!-- WNIOSKI -->", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def smoke():
    data, source = build_stream()
    batches, sides = make_batches(data, source, seq_len=64, batch_size=4, n_batches=2, seed=0)
    # prefiks nie rozstrzyga źródła: pierwsze 36 bajtów wszystkich dokumentów są równe
    doc = len((PREFIX + TAILS[0] * 40 + "\n").encode())
    start1 = doc * 8
    if not torch.equal(data[: len(PREFIX)], data[start1 : start1 + len(PREFIX)]):
        raise SystemExit("prefiksy źródeł się różnią")
    if torch.equal(source[:doc], source[start1 : start1 + doc]):
        raise SystemExit("numer źródła jest taki sam dla dwóch dokumentów")
    for arm in ARMS:
        model = make_model(arm, 0)
        info = train_one(model, batches, sides, steps=2)
        if not info["finite"]:
            raise SystemExit(f"smoke {arm}: {info}")
        print(f"[smoke] {arm} {info}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        smoke()
        return
    data, source = build_stream()
    spec = {
        "version": 1,
        "task": "T2-tiny",
        "seeds": N_SEEDS,
        "steps": STEPS,
        "margin_loss": MARGIN,
        "prefix": PREFIX,
        "tails": list(TAILS),
        "conditions": {
            "A": "bw·0 vs sm·0, bez dodatkowej zmiennej, Most z zerowym ΔE",
            "B": "ten sam numer źródła w tokenie obu jąder, Most bez ΔE",
            "C": "Most kanoniczny (tracer pozycji przez ΔE) vs Softmax bez pluginu — mieszane",
        },
        "equivalence": "90% CI (Most−Softmax) w całości w ±0.05, osobno dla A i dla B",
        "plugin": "90% CI (Softmax B − Softmax A) w całości pod zerem",
    }
    if not SPEC.is_file():
        SPEC.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    done = {}
    if RUNS.is_file():
        for line in RUNS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["seed"], r["arm"])] = r
    print(f"[T2] bajtów={data.numel()} seedów={N_SEEDS} kroków={STEPS} margines=±{MARGIN}", flush=True)
    t0 = time.time()
    try:
        if not args.report_only:
            with RUNS.open("a", encoding="utf-8") as f:
                for seed in range(N_SEEDS):
                    batches, sides = make_batches(data, source, seq_len=64, batch_size=16, n_batches=40, seed=seed)
                    for arm in ARMS:
                        if (seed, arm) in done:
                            continue
                        t1 = time.time()
                        info = train_one(make_model(arm, seed), batches, sides, steps=STEPS)
                        rec = {"seed": seed, "arm": arm, "wall_s": round(time.time() - t1, 2), **info,
                               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        done[(seed, arm)] = rec
                    _status(f"RUNNING seed {seed} ({time.time() - t0:.0f}s)")
                    print(f"  seed {seed} gotowy ({time.time() - t0:.0f}s)", flush=True)
        report(done, spec)
        _status(f"DONE {time.time() - t0:.0f}s")
        print(f"wrote {REPORT}", flush=True)
    except Exception as e:
        _status(f"FAILED {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
