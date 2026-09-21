"""Probe mechanistyczny (#1 / recenzja): lost-in-the-middle + entropia uwagi.

Na CPU, model medium_long (~3.3M), seq=2048:
  - entropia rozkładu uwagi vs pozycja query (Most vs Softmax)
  - masa uwagi na igle @ head/mid/tail
  - retrieval: P(poprawny bajt sekretu) gdy cue na końcu

Hipoteza: grube ogony Lorentza → płaska krzywa mid; Softmax → wklęsła (dip w środku).

  python archiwum/skrypty/run_lost_in_middle_probe.py
  python archiwum/skrypty/run_lost_in_middle_probe.py --seq-len 2048 --train-steps 400
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
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lm_tiny import (
    TinyLM,
    TinyLMConfig,
    build_char_corpus,
    count_params,
    load_text_corpus,
    make_batches,
    train_lm,
)
from runlog import RunDir

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "doswiadczenie" / "glify"

FILLER = ord(".")
# unikalna igła w vocab 256
NEEDLE_PREFIX = list(b"<<<KEY")
NEEDLE_SECRET = list(b"42")
NEEDLE_SUFFIX = list(b">>>")
CUE = list(b"KEY=")


def _entropy_rows(w: torch.Tensor) -> torch.Tensor:
    """w [B,h,N,N] causal — entropia po kluczach dla każdego query. → [N] średnia B,h."""
    # unikaj log0 na zamaskowanych (już 0)
    p = w.clamp_min(1e-12)
    h = -(p * p.log()).sum(dim=-1)  # [B,h,N]
    return h.mean(dim=(0, 1))  # [N]


def _mean_attn_last_layer(attn_ws: list[torch.Tensor]) -> torch.Tensor:
    """Średnia po głowach, ostatnia warstwa: [N,N]."""
    w = attn_ws[-1][0].mean(dim=0)  # [N,N]
    return w


def build_needle_sequence(seq_len: int, needle_rel: float, secret: list[int] | None = None) -> tuple[torch.Tensor, dict]:
    secret = secret or NEEDLE_SECRET
    needle = NEEDLE_PREFIX + secret + NEEDLE_SUFFIX
    n_len = len(needle)
    cue = CUE
    # zostaw miejsce na cue na końcu
    usable = seq_len - len(cue)
    if usable < n_len + 8:
        raise ValueError("seq_len za krótki na igłę+cue")
    start = int(needle_rel * (usable - n_len))
    start = max(0, min(start, usable - n_len))
    ids = [FILLER] * usable + cue
    ids[start : start + n_len] = needle
    meta = {
        "needle_start": start,
        "needle_end": start + n_len,
        "secret": secret,
        "secret_pos": start + len(NEEDLE_PREFIX),
        "cue_start": usable,
        "needle_rel": needle_rel,
    }
    return torch.tensor(ids, dtype=torch.long), meta


@torch.no_grad()
def probe_entropy(model: TinyLM, seq_len: int, n_samples: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ent_acc = None
    for i in range(n_samples):
        idx = torch.randint(0, model.cfg.vocab_size, (1, seq_len), generator=g)
        _, attn_ws = model(idx, return_attn=True)
        ent = _entropy_rows(attn_ws[-1])  # [N]
        ent_acc = ent if ent_acc is None else ent_acc + ent
    ent_mean = ent_acc / n_samples
    # zrób kubełki względnej pozycji
    buckets = [0.1, 0.25, 0.5, 0.75, 0.9]
    by_bucket = {}
    for b in buckets:
        q = min(seq_len - 1, max(0, int(b * (seq_len - 1))))
        # średnia entropii w oknie ±2% wokół b
        lo = max(0, q - max(1, seq_len // 50))
        hi = min(seq_len, q + max(1, seq_len // 50) + 1)
        by_bucket[str(b)] = float(ent_mean[lo:hi].mean())
    return {
        "entropy_mean": float(ent_mean.mean()),
        "entropy_by_rel_pos": by_bucket,
        "entropy_head": float(ent_mean[: max(1, seq_len // 10)].mean()),
        "entropy_mid": float(ent_mean[seq_len // 2 - seq_len // 20 : seq_len // 2 + seq_len // 20].mean()),
        "entropy_tail": float(ent_mean[-max(1, seq_len // 10) :].mean()),
    }


@torch.no_grad()
def probe_needle_mass(model: TinyLM, seq_len: int, rels: list[float], n_repeat: int = 5) -> list[dict]:
    rows = []
    for rel in rels:
        masses = []
        secrets_ok = []
        for rep in range(n_repeat):
            # różny sekret cyfrowy 10–99
            sec = [ord("0") + ((10 + rel * 17 + rep * 3) % 90) // 10, ord("0") + ((10 + rel * 17 + rep * 3) % 10)]
            sec = [int(x) for x in sec]
            idx, meta = build_needle_sequence(seq_len, rel, secret=sec)
            idx = idx.unsqueeze(0)
            logits, attn_ws = model(idx, return_attn=True)
            w = _mean_attn_last_layer(attn_ws)  # [N,N]
            q = seq_len - 1  # ostatni token cue
            mass = float(w[q, meta["needle_start"] : meta["needle_end"]].sum())
            masses.append(mass)
            # retrieval: po "KEY=" przewiduj pierwszy bajt sekretu
            pred_pos = meta["cue_start"] + len(CUE) - 1  # pozycja '='
            pred = int(logits[0, pred_pos].argmax())
            secrets_ok.append(1.0 if pred == sec[0] else 0.0)
        rows.append(
            {
                "needle_rel": rel,
                "attn_mass_mean": sum(masses) / len(masses),
                "attn_mass_std": float(torch.tensor(masses).std(unbiased=False)),
                "secret_acc": sum(secrets_ok) / len(secrets_ok),
            }
        )
    return rows


def mid_dip(curve: list[dict], key: str = "attn_mass_mean") -> float:
    """Dodatnie = środek gorszy niż średnia head/tail (wklęsłość lost-in-middle)."""
    by = {r["needle_rel"]: r[key] for r in curve}
    head = by.get(0.1, by.get(0.25))
    mid = by.get(0.5)
    tail = by.get(0.9, by.get(0.75))
    if head is None or mid is None or tail is None:
        return float("nan")
    return 0.5 * (head + tail) - mid


def train_arm(attn: str, cfg: TinyLMConfig, data: torch.Tensor, *, steps: int, seq_len: int, seed: int):
    torch.manual_seed(seed)
    model = TinyLM(cfg, attn=attn)
    # trening na krótszych oknach (szybciej); pos-emb i tak na 4k
    train_seq = min(256, seq_len)
    batches = make_batches(
        data, seq_len=train_seq, batch_size=4, n_batches=80, seed=3000 + seed
    )
    # dołóż batche z igłą (uczy retrieval)
    needle_batches = []
    g = torch.Generator().manual_seed(4000 + seed)
    for _ in range(40):
        rels = [0.1, 0.5, 0.9]
        rows = []
        for _b in range(4):
            rel = rels[_b % 3]
            sec = [
                48 + int(torch.randint(0, 10, (1,), generator=g)),
                48 + int(torch.randint(0, 10, (1,), generator=g)),
            ]
            ids, _ = build_needle_sequence(train_seq, rel, secret=sec)
            rows.append(ids)
        needle_batches.append(torch.stack(rows, dim=0))
    all_batches = batches + needle_batches
    t0 = time.perf_counter()
    curve = train_lm(model, all_batches, steps=steps, lr=1e-3, log_every=max(1, steps // 10))
    wall = time.perf_counter() - t0
    return model, {"train_wall_s": wall, "train_curve": curve, "n_params": count_params(model)}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-needle-repeat", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = TinyLMConfig.preset("medium_long")
    if args.seq_len > cfg.max_seq:
        raise SystemExit(f"seq_len {args.seq_len} > max_seq {cfg.max_seq}")

    texts = load_text_corpus(ROOT, repeat=2)
    data = build_char_corpus(texts, cfg.vocab_size)
    rundir = RunDir.create(
        "lost_in_middle_probe",
        argv=sys.argv,
        hparams={
            "preset": "medium_long",
            "seq_len": args.seq_len,
            "train_steps": args.train_steps,
            "seed": args.seed,
            "n_params": count_params(TinyLM(cfg, attn="most")),
        },
        extra_meta={"criterion": "mechanistic_lost_in_middle"},
    )
    print(
        f"Lost-in-middle probe seq={args.seq_len} steps={args.train_steps} "
        f"params≈{count_params(TinyLM(cfg, attn='most'))} run={rundir.path}",
        flush=True,
    )

    t_all = time.time()
    results = {"arms": {}}
    rels = [0.1, 0.25, 0.5, 0.75, 0.9]

    for attn in ("most", "softmax"):
        print(f"\n=== train {attn} ===", flush=True)
        model, train_meta = train_arm(
            attn, cfg, data, steps=args.train_steps, seq_len=args.seq_len, seed=args.seed
        )
        model.eval()
        print(f"  probe entropy @ {args.seq_len}…", flush=True)
        ent = probe_entropy(model, args.seq_len, n_samples=3, seed=args.seed + 7)
        print(f"  probe needle mass…", flush=True)
        needle = probe_needle_mass(
            model, args.seq_len, rels, n_repeat=args.n_needle_repeat
        )
        dip_mass = mid_dip(needle, "attn_mass_mean")
        dip_acc = mid_dip(needle, "secret_acc")
        results["arms"][attn] = {
            **train_meta,
            "entropy": ent,
            "needle": needle,
            "mid_dip_attn_mass": dip_mass,
            "mid_dip_secret_acc": dip_acc,
        }
        print(
            f"  ent mid/head/tail={ent['entropy_mid']:.3f}/{ent['entropy_head']:.3f}/{ent['entropy_tail']:.3f} "
            f"dip_mass={dip_mass:+.4f} dip_acc={dip_acc:+.4f}",
            flush=True,
        )
        del model

    most = results["arms"]["most"]
    sm = results["arms"]["softmax"]
    # Werdykt mechanistyczny
    # Hipoteza: Most ma mniejszy (bliższy 0 lub ujemny) mid_dip niż Softmax
    hyp_mass = most["mid_dip_attn_mass"] < sm["mid_dip_attn_mass"]
    hyp_acc = most["mid_dip_secret_acc"] <= sm["mid_dip_secret_acc"] + 0.05
    # entropia mid: Most wyższa = mniej zapadnięta
    hyp_ent = most["entropy"]["entropy_mid"] >= sm["entropy"]["entropy_mid"]

    if hyp_mass and hyp_ent:
        verdict = (
            "PASS mechanistyczny (na tej skali): Most ma mniejszy dip mid w masie uwagi "
            f"({most['mid_dip_attn_mass']:+.4f} vs SM {sm['mid_dip_attn_mass']:+.4f}) "
            f"i ≥ entropię mid ({most['entropy']['entropy_mid']:.3f} vs {sm['entropy']['entropy_mid']:.3f}). "
            "GPU = potwierdzenie trendu, nie pierwszy pomiar."
        )
        status = "pass"
    elif hyp_mass or hyp_ent:
        verdict = (
            "PARTIAL: sygnał mieszany Most vs Softmax na lost-in-middle / entropii. "
            f"dip_mass Most/SM={most['mid_dip_attn_mass']:+.4f}/{sm['mid_dip_attn_mass']:+.4f}; "
            f"ent_mid Most/SM={most['entropy']['entropy_mid']:.3f}/{sm['entropy']['entropy_mid']:.3f}."
        )
        status = "partial"
    else:
        verdict = (
            "FAIL hipotezy na tej skali: Softmax nie pokazuje głębszego mid-dipu niż Most "
            "(albo Most nie trzyma wyższej entropii mid). Historia „grube ogony → płaski mid” "
            "nie potwierdza się przy 3M/2k — albo potrzeba innego protokołu/treningu."
        )
        status = "fail"

    report = {
        "protocol": "Lost-in-the-middle + attn entropy — mechanistic probe (CPU, ~3M, seq=2048)",
        "review_claim": (
            "Grube ogony Lorentza → masa uwagi nie zapada się; lost-in-middle płaski dla Most, "
            "wklęsły dla Softmax — mierzalne na CPU przed 100M."
        ),
        "seq_len": args.seq_len,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "rels": rels,
        "arms": {
            k: {
                **{kk: vv for kk, vv in v.items() if kk != "train_curve"},
            }
            for k, v in results["arms"].items()
        },
        "hypothesis": {
            "most_smaller_mid_dip_mass": hyp_mass,
            "most_higher_entropy_mid": hyp_ent,
            "most_secret_acc_mid_ok": hyp_acc,
        },
        "verdict": verdict,
        "status": status,
        "elapsed_s": time.time() - t_all,
        "run_dir": str(rundir.path),
        "note": (
            "mid_dip = 0.5*(head+tail) - mid; dodatnie = wklęsłość (gorszy środek). "
            "To nie jest jeszcze Needle 16k (#4); to mechanistyczny proxy na 2k."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jp = OUT_DIR / "LOST_IN_MIDDLE_PROBE_results.json"
    mp = OUT_DIR / "LOST_IN_MIDDLE_PROBE.md"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _needle_table(arm: str) -> list[str]:
        lines = [
            f"### {arm}",
            "",
            "| rel | attn_mass | secret_acc |",
            "|----:|----------:|-----------:|",
        ]
        for r in results["arms"][arm]["needle"]:
            lines.append(
                f"| {r['needle_rel']:.2f} | {r['attn_mass_mean']:.4f} | {r['secret_acc']:.2f} |"
            )
        lines += [
            "",
            f"mid_dip mass={results['arms'][arm]['mid_dip_attn_mass']:+.4f} · "
            f"mid_dip acc={results['arms'][arm]['mid_dip_secret_acc']:+.4f}",
            f"entropy head/mid/tail="
            f"{results['arms'][arm]['entropy']['entropy_head']:.3f}/"
            f"{results['arms'][arm]['entropy']['entropy_mid']:.3f}/"
            f"{results['arms'][arm]['entropy']['entropy_tail']:.3f}",
            "",
        ]
        return lines

    lines = [
        "# Lost-in-the-middle probe — odpowiedź na recenzję",
        "",
        f"Model medium_long (~3.3M) · seq=**{args.seq_len}** · train_steps={args.train_steps} · seed={args.seed}",
        "",
        f"**Werdykt:** {verdict}",
        "",
        report["review_claim"],
        "",
        report["note"],
        "",
        "## Softmax",
        "",
    ]
    lines += _needle_table("softmax")
    lines += ["## Most", ""]
    lines += _needle_table("most")
    lines += [
        "## Hipoteza",
        "",
        f"- Most mniejszy mid_dip (masa): **{hyp_mass}**",
        f"- Most wyższa entropia mid: **{hyp_ent}**",
        f"- secret_acc mid OK: **{hyp_acc}**",
        "",
        f"elapsed={report['elapsed_s']:.0f}s · run-dir=`{rundir.path}`",
        "",
        "JSON: `LOST_IN_MIDDLE_PROBE_results.json`",
        "",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    rundir.finalize(metrics=report, summary="\n".join(lines), status=status)
    print("\n" + verdict)
    print(f"MD → {mp}")
    return report


if __name__ == "__main__":
    main()
