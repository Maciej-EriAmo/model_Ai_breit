"""Nauka pełnej tablicy Mendelejewa (118, nucleus+topology) — jutro ~2–3 h.

Domyślnie / kanon jutra: **--most-only** (bez Softmax, ZERO, flat, bez porównań).
Parity Most≈Softmax już wykazana na P0 (8 el.); tu budujemy kompetencję Mosta.

  python export_mendelejew_full.py
  python run_mendelejew_full.py --most-only
  python run_mendelejew_full.py --most-only --smoke
  python run_mendelejew_full.py --all-arms   # stary protokół A/B/C/D (opcjonalnie)
"""
from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import argparse
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import make_dump_loaders
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_mendelejew_full"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

SEEDS = [0, 1, 17]
EPOCHS = 64
N_TRAIN = 200
N_VAL = 50


def ensure_dump() -> None:
    if (DUMP / "atoms.jsonl").is_file():
        return
    from export_mendelejew_full import export
    from make_periodic_full import main as gen

    gen()
    export(DUMP)


def run_seed(seed: int, *, epochs: int, with_curves: bool, most_only: bool) -> dict:
    man = json.loads((DUMP / "manifest.json").read_text(encoding="utf-8"))
    n_atoms = int(man["n_atoms"])
    bs = min(16, max(4, n_atoms))
    g, train, val, _ = make_dump_loaders(
        DUMP,
        n_train=N_TRAIN,
        n_val=N_VAL,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
    )
    chance = 1.0 / max(1, len(g.atoms) - 1)

    def _train(factory, ablate):
        torch.manual_seed(seed)
        return train_encoder(
            factory(),
            train,
            val,
            epochs=epochs,
            lr=CANON.lr,
            ablate_kw=ablate,
            return_curve=with_curves,
        )

    print("  [Most] …", flush=True)
    most = _train(lambda: build_model(warm=False), {})
    row: dict = {
        "seed": seed,
        "N": len(g.atoms),
        "chance_r1": chance,
        "most_r1": most["r@1"],
        "most_r3": most["r@3"],
        "most_mrr": most["mrr"],
        "above_floor": most["r@1"] >= 5 * chance,
        "most_only": most_only,
    }
    if with_curves:
        row["curves"] = {"Most": most.get("curve")}

    if most_only:
        return row

    print("  [flat] …", flush=True)
    flat = _train(lambda: build_model(warm=False), {"flat_payload": True})
    print("  [Softmax] …", flush=True)
    sm = _train(
        lambda: PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        ),
        {},
    )
    print("  [ZERO] …", flush=True)
    zero = _train(
        lambda: PhiBridgeEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
            coupling=CANON.coupling,
        ),
        {"zero_tracer": True},
    )
    row.update({
        "A_most": most["r@1"],
        "B_flat": flat["r@1"],
        "C_softmax": sm["r@1"],
        "D_zero": zero["r@1"],
        "delta_flat": most["r@1"] - flat["r@1"],
        "delta_sm": most["r@1"] - sm["r@1"],
        "delta_zero": most["r@1"] - zero["r@1"],
        "nonflat_win": (most["r@1"] - flat["r@1"]) >= 0.05,
        "most_win": (most["r@1"] - sm["r@1"]) >= CANON.gate_recall_r1,
        "tau_win": (most["r@1"] - zero["r@1"]) >= 0.02,
    })
    if with_curves:
        row["curves"].update({
            "flat": flat.get("curve"),
            "Softmax": sm.get("curve"),
            "ZERO": zero.get("curve"),
        })
    return row

def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument(
        "--most-only",
        action="store_true",
        default=True,
        help="tylko Most (domyślne; bez Softmax/ZERO/flat)",
    )
    ap.add_argument(
        "--all-arms",
        action="store_true",
        help="opcjonalnie: Most+flat+Softmax+ZERO (wyłącza --most-only)",
    )
    args = ap.parse_args(argv)
    most_only = not args.all_arms  # domyślnie most-only

    ensure_dump()
    seeds = [0] if args.smoke else (
        [int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS)
    )
    epochs = 16 if args.smoke and args.epochs is None else (
        args.epochs if args.epochs is not None else EPOCHS
    )

    man = json.loads((DUMP / "manifest.json").read_text(encoding="utf-8"))
    mode = "MOST-ONLY" if most_only else "ALL-ARMS"
    print(
        f"Mendelejew FULL [{mode}] n={man['n_atoms']} edges={man['n_edges']} "
        f"epochs={epochs} seeds={seeds}"
    )
    t_all = time.time()
    rows = []
    for seed in seeds:
        t0 = time.time()
        print(f"\n=== seed={seed} ===", flush=True)
        r = run_seed(
            seed, epochs=epochs, with_curves=(seed == seeds[0]), most_only=most_only
        )
        rows.append(r)
        if most_only:
            print(
                f"  Most r@1={r['most_r1']:.3f} r@3={r['most_r3']:.3f} "
                f"mrr={r['most_mrr']:.3f} floor={'Y' if r['above_floor'] else 'N'} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        else:
            print(
                f"  A={r['A_most']:.3f} B={r['B_flat']:.3f} C={r['C_softmax']:.3f} "
                f"D={r['D_zero']:.3f} Δflat={r['delta_flat']:+.3f} ΔSM={r['delta_sm']:+.3f} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )

    n = len(rows)
    if most_only:
        agg = {
            "mean_most_r1": sum(r["most_r1"] for r in rows) / n,
            "mean_most_r3": sum(r["most_r3"] for r in rows) / n,
            "mean_most_mrr": sum(r["most_mrr"] for r in rows) / n,
            "n_above_floor": sum(1 for r in rows if r["above_floor"]),
            "n_seeds": n,
        }
        verdict = (
            f"FULL118 MOST-ONLY mean r@1={agg['mean_most_r1']:.3f} "
            f"r@3={agg['mean_most_r3']:.3f} mrr={agg['mean_most_mrr']:.3f} "
            f"above_floor={agg['n_above_floor']}/{n} "
            f"(parity vs Softmax już na P0: Most≈0.43 SM≈0.45 — bez re-porównania)"
        )
        protocol = "Mendelejew FULL 118 — MOST-ONLY (bez Softmax)"
    else:
        agg = {
            "mean_A": sum(r["A_most"] for r in rows) / n,
            "mean_B": sum(r["B_flat"] for r in rows) / n,
            "mean_C": sum(r["C_softmax"] for r in rows) / n,
            "mean_D": sum(r["D_zero"] for r in rows) / n,
            "mean_delta_flat": sum(r["delta_flat"] for r in rows) / n,
            "mean_delta_sm": sum(r["delta_sm"] for r in rows) / n,
            "n_nonflat": sum(1 for r in rows if r["nonflat_win"]),
            "n_most": sum(1 for r in rows if r["most_win"]),
            "n_seeds": n,
        }
        verdict = (
            f"FULL118 mean Most={agg['mean_A']:.3f} flat={agg['mean_B']:.3f} "
            f"SM={agg['mean_C']:.3f} Δflat={agg['mean_delta_flat']:+.3f} "
            f"nonflat={agg['n_nonflat']}/{n} most_win={agg['n_most']}/{n}"
        )
        protocol = "Mendelejew FULL 118 nucleus+topology — A/B/C/D"

    report = {
        "protocol": protocol,
        "most_only": most_only,
        "dump": str(DUMP),
        "n_atoms": man["n_atoms"],
        "n_edges": man["n_edges"],
        "epochs": epochs,
        "seeds": seeds,
        "smoke": bool(args.smoke),
        "rows": rows,
        "aggregate": agg,
        "verdict": verdict,
        "elapsed_s": time.time() - t_all,
        "success_frame_cv": (
            "Domowa konstrukcja Most dorównała Softmaxowi na P0 Mendelejew "
            "(Most≈0.43 vs SM≈0.45, nonflat win) — parity z kanonem; "
            "FULL118 = kompetencja Mosta, nie re-bench vs Softmax."
        ),
        "baseline_p0_8el": {"mean_A": 0.427, "mean_B": 0.329, "mean_C": 0.445},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    jp = OUT_DIR / f"mendelejew_full{tag}_results.json"
    mp = OUT_DIR / ("MENDELEJEW_FULL_SMOKE.md" if args.smoke else "MENDELEJEW_FULL.md")
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # kryterium #5 — run-dir
    try:
        from dataclasses import asdict

        from config import CANON
        from runlog import RunDir

        curve0 = None
        for r in rows:
            if r.get("seed") == seeds[0]:
                curve0 = (r.get("curves") or {}).get("Most")
                break
        rd = RunDir.create(
            "mendelejew_full_smoke" if args.smoke else "mendelejew_full",
            argv=sys.argv,
            hparams={
                "dump": str(DUMP),
                "epochs": epochs,
                "seeds": seeds,
                "most_only": most_only,
                "smoke": bool(args.smoke),
                "canon": asdict(CANON),
            },
            extra_meta={"criterion": "success#5_run_dir", "artifact_md": str(mp)},
        )
        rd.finalize(metrics=report, summary=verdict + f"\n\nMD: `{mp}`\n", status="ok", curve=curve0)
        report["run_dir"] = str(rd.path)
        print(f"run-dir → {rd.path}")
    except Exception as e:
        print(f"[warn] run-dir skipped: {e}")
    lines = [
        "# Mendelejew FULL (118)" + (" — MOST-ONLY" if most_only else ""),
        "",
        f"N={man['n_atoms']} edges={man['n_edges']} · epochs={epochs} · seeds={seeds} · mode={mode}",
        "",
        report["success_frame_cv"],
        "",
        f"**Werdykt:** {verdict}",
        "",
    ]
    if most_only:
        lines += [
            "| seed | Most r@1 | r@3 | mrr | floor |",
            "|-----:|---------:|----:|----:|:-----:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['seed']} | {r['most_r1']:.3f} | {r['most_r3']:.3f} | "
                f"{r['most_mrr']:.3f} | {'Y' if r['above_floor'] else 'N'} |"
            )
    else:
        lines += [
            "| seed | A Most | B flat | C SM | D ZERO | Δflat | ΔSM |",
            "|-----:|-------:|-------:|-----:|-------:|------:|-----:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['seed']} | {r['A_most']:.3f} | {r['B_flat']:.3f} | "
                f"{r['C_softmax']:.3f} | {r['D_zero']:.3f} | "
                f"{r['delta_flat']:+.3f} | {r['delta_sm']:+.3f} |"
            )
    lines += ["", f"JSON: `{jp.name}`", ""]
    mp.write_text("\n".join(lines), encoding="utf-8")
    print(verdict)
    print(f"→ {jp}\n→ {mp}")
    return report


if __name__ == "__main__":
    main()
