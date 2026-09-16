"""Mendelejew P0 — wieloosiowa struktura atomów vs płaski payload.

Armie:
  A Most Φ     — dumps/from_struktura (group/period/shells)
  B Most flat  — flat_payload (bajty, τ=0)
  C Softmax Φ
  D Most ZERO  — zero_tracer

  python run_mendelejew_p0.py
  python run_mendelejew_p0.py --smoke   # 1 seed, 16 epok
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_struktura"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

SEEDS_FULL = [0, 1, 17]
EPOCHS_FULL = 64
N_TRAIN = 160
N_VAL = 40


def ensure_dump() -> None:
    if (DUMP / "atoms.jsonl").is_file() and (DUMP / "manifest.json").is_file():
        return
    import import_struktura_atomow as imp

    src = imp.DEFAULT_JSON if imp.DEFAULT_JSON.is_file() else imp.DEFAULT_CSV
    rows = imp.load_rows(src)
    imp.export_dump(rows, DUMP)
    print(f"imported → {DUMP}")


def run_seed(seed: int, *, epochs: int, bs: int, with_curves: bool) -> dict:
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
    N = len(g.atoms)
    chance = 1.0 / max(1, N - 1)

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

    print(f"  [A] Most Φ …", flush=True)
    most = _train(lambda: build_model(warm=False), {})
    print(f"  [B] Most flat …", flush=True)
    flat = _train(lambda: build_model(warm=False), {"flat_payload": True})
    print(f"  [C] Softmax …", flush=True)
    sm = _train(
        lambda: PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        ),
        {},
    )
    print(f"  [D] Most ZERO …", flush=True)
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

    d_flat = most["r@1"] - flat["r@1"]
    d_sm = most["r@1"] - sm["r@1"]
    d_zero = most["r@1"] - zero["r@1"]
    row = {
        "seed": seed,
        "N": N,
        "chance_r1": chance,
        "A_most": most["r@1"],
        "B_flat": flat["r@1"],
        "C_softmax": sm["r@1"],
        "D_zero": zero["r@1"],
        "A_r3": most["r@3"],
        "B_r3": flat["r@3"],
        "A_mrr": most["mrr"],
        "B_mrr": flat["mrr"],
        "delta_flat": d_flat,
        "delta_sm": d_sm,
        "delta_zero": d_zero,
        "above_floor": most["r@1"] >= 5 * chance,
        "nonflat_win": d_flat >= 0.05,
        "most_win": d_sm >= CANON.gate_recall_r1,
        "tau_win": d_zero >= 0.02,
    }
    if with_curves:
        row["curves"] = {
            "A_most": most.get("curve"),
            "B_flat": flat.get("curve"),
            "C_softmax": sm.get("curve"),
            "D_zero": zero.get("curve"),
        }
    return row


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 16 epok")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="np. 0,1,17")
    args = ap.parse_args(argv)

    ensure_dump()
    seeds = (
        [0]
        if args.smoke
        else (
            [int(x) for x in args.seeds.split(",")]
            if args.seeds
            else list(SEEDS_FULL)
        )
    )
    epochs = (
        16
        if args.smoke and args.epochs is None
        else (args.epochs if args.epochs is not None else EPOCHS_FULL)
    )

    man = json.loads((DUMP / "manifest.json").read_text(encoding="utf-8"))
    n_atoms = int(man.get("n_atoms", 72))
    bs = min(16, max(4, n_atoms))
    chance = 1.0 / max(1, n_atoms - 1)
    print(
        f"Mendelejew P0 dump={DUMP.name} atoms={n_atoms} "
        f"symbols={man.get('symbols')} chance={chance:.4f} "
        f"epochs={epochs} seeds={seeds}"
    )

    rows = []
    t_all = time.time()
    for seed in seeds:
        t0 = time.time()
        print(f"\n=== seed={seed} ===", flush=True)
        r = run_seed(seed, epochs=epochs, bs=bs, with_curves=(seed == seeds[0]))
        rows.append(r)
        print(
            f"  A={r['A_most']:.3f} B_flat={r['B_flat']:.3f} "
            f"C_SM={r['C_softmax']:.3f} D_ZERO={r['D_zero']:.3f} | "
            f"Δflat={r['delta_flat']:+.3f} ΔSM={r['delta_sm']:+.3f} "
            f"Δτ={r['delta_zero']:+.3f} | "
            f"floor={'Y' if r['above_floor'] else 'N'} "
            f"nonflat={'Y' if r['nonflat_win'] else 'N'} "
            f"most={'Y' if r['most_win'] else 'N'} "
            f"tau={'Y' if r['tau_win'] else 'N'} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    n = len(rows)
    agg = {
        "mean_A": sum(r["A_most"] for r in rows) / n,
        "mean_B": sum(r["B_flat"] for r in rows) / n,
        "mean_C": sum(r["C_softmax"] for r in rows) / n,
        "mean_D": sum(r["D_zero"] for r in rows) / n,
        "mean_delta_flat": sum(r["delta_flat"] for r in rows) / n,
        "mean_delta_sm": sum(r["delta_sm"] for r in rows) / n,
        "mean_delta_zero": sum(r["delta_zero"] for r in rows) / n,
        "n_above_floor": sum(1 for r in rows if r["above_floor"]),
        "n_nonflat_win": sum(1 for r in rows if r["nonflat_win"]),
        "n_most_win": sum(1 for r in rows if r["most_win"]),
        "n_tau_win": sum(1 for r in rows if r["tau_win"]),
        "n_seeds": n,
    }

    # werdykt wg planu
    if agg["mean_delta_flat"] >= 0.05 and agg["mean_delta_sm"] >= CANON.gate_recall_r1:
        verdict_kind = 1
        verdict = (
            "Most_A ≫ flat i Most ≳ SM — wielowymiarowe Φ pomaga Mostowi "
            "(metoda aplikacji potwierdzona na Mendelejewie)."
        )
    elif agg["mean_delta_flat"] >= 0.05 and abs(agg["mean_delta_sm"]) < 0.02:
        verdict_kind = 2
        verdict = (
            "Most_A ≫ flat, ale Most≈SM≈ZERO — geometria grafu pomaga wszystkim; "
            "Most nie jest specjalny (jak Kana/PL P1b)."
        )
    elif abs(agg["mean_delta_flat"]) < 0.05:
        verdict_kind = 3
        verdict = (
            "Most_A ≈ flat — hipoteza non-flat pada na tym secie; "
            "rozważyć qm_orbitals / bogatsze bindy."
        )
    else:
        verdict_kind = 0
        verdict = (
            f"Wynik mieszany: mean Δflat={agg['mean_delta_flat']:+.3f} "
            f"ΔSM={agg['mean_delta_sm']:+.3f} Δτ={agg['mean_delta_zero']:+.3f}."
        )

    report = {
        "protocol": "Mendelejew P0 — Φ struktura vs flat_payload vs Softmax vs ZERO",
        "dump": str(DUMP),
        "symbols": man.get("symbols"),
        "n_atoms": n_atoms,
        "chance_r1": chance,
        "epochs": epochs,
        "seeds": seeds,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "smoke": bool(args.smoke),
        "gates": {
            "floor": "Most_A >= 5×chance",
            "nonflat_win": "Most_A - flat >= 0.05",
            "most_win": f"Most_A - Softmax >= {CANON.gate_recall_r1}",
            "tau_win": "Most_A - ZERO >= 0.02",
        },
        "rows": rows,
        "aggregate": agg,
        "verdict_kind": verdict_kind,
        "verdict": verdict,
        "elapsed_s": time.time() - t_all,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    json_path = OUT_DIR / f"mendelejew_p0{suffix}_results.json"
    md_path = OUT_DIR / f"MENDELEJEW_P0{suffix.upper() if args.smoke else ''}.md"
    if args.smoke:
        md_path = OUT_DIR / "MENDELEJEW_P0_SMOKE.md"
    else:
        md_path = OUT_DIR / "MENDELEJEW_P0.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report, md_path)
    print("\n" + "=" * 70)
    print(verdict)
    print(f"JSON → {json_path}")
    print(f"MD   → {md_path}")
    return report


def _write_md(report: dict, path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# Mendelejew P0 — wieloosiowa Φ vs płaski tekst",
        "",
        f"Dump: `{Path(report['dump']).name}` · symbols={report['symbols']} · "
        f"N={report['n_atoms']} · chance={report['chance_r1']:.4f}",
        f"Epoki={report['epochs']} · seedy={report['seeds']} · "
        f"n_train={report['n_train']} · smoke={report['smoke']}",
        "",
        "Hipoteza: pojęcia wielowymiarowe (period×group×shells) są dla Mostu "
        "łatwiejsze niż ta sama wiedza jako płaski payload.",
        "",
        "## Agregat",
        "",
        "| arm | mean r@1 |",
        "|-----|---------:|",
        f"| A Most Φ | {agg['mean_A']:.3f} |",
        f"| B Most flat | {agg['mean_B']:.3f} |",
        f"| C Softmax | {agg['mean_C']:.3f} |",
        f"| D Most ZERO | {agg['mean_D']:.3f} |",
        "",
        f"| Δ(A−B) flat | {agg['mean_delta_flat']:+.3f} | nonflat_win {agg['n_nonflat_win']}/{agg['n_seeds']} |",
        f"| Δ(A−C) SM | {agg['mean_delta_sm']:+.3f} | most_win {agg['n_most_win']}/{agg['n_seeds']} |",
        f"| Δ(A−D) τ | {agg['mean_delta_zero']:+.3f} | tau_win {agg['n_tau_win']}/{agg['n_seeds']} |",
        f"| above floor | {agg['n_above_floor']}/{agg['n_seeds']} |",
        "",
        f"**Werdykt (kind={report['verdict_kind']}):** {report['verdict']}",
        "",
        "## Per seed",
        "",
        "| seed | A Most | B flat | C SM | D ZERO | Δflat | ΔSM | Δτ | flags |",
        "|-----:|-------:|-------:|-----:|-------:|------:|-----:|----:|-------|",
    ]
    for r in report["rows"]:
        flags = []
        if r["above_floor"]:
            flags.append("floor")
        if r["nonflat_win"]:
            flags.append("nonflat")
        if r["most_win"]:
            flags.append("most")
        if r["tau_win"]:
            flags.append("tau")
        lines.append(
            f"| {r['seed']} | {r['A_most']:.3f} | {r['B_flat']:.3f} | "
            f"{r['C_softmax']:.3f} | {r['D_zero']:.3f} | "
            f"{r['delta_flat']:+.3f} | {r['delta_sm']:+.3f} | "
            f"{r['delta_zero']:+.3f} | {','.join(flags) or '—'} |"
        )

    sample = report["rows"][0] if report["rows"] else None
    if sample and sample.get("curves"):
        lines += ["", f"## Krzywe (seed={sample['seed']})", ""]
        for key, title in (
            ("A_most", "A Most Φ"),
            ("B_flat", "B Most flat"),
            ("C_softmax", "C Softmax"),
            ("D_zero", "D Most ZERO"),
        ):
            curve = sample["curves"].get(key) or []
            lines.append(f"### {title}")
            lines.append("| ep | loss | r@1 | r@3 | mrr |")
            lines.append("|---:|-----:|----:|----:|----:|")
            want = {1, 8, 16, 32, 48, 64}
            if report["epochs"] <= 16:
                want = {1, 4, 8, 12, 16}
            for p in curve:
                if p["epoch"] in want or p["epoch"] == report["epochs"]:
                    lines.append(
                        f"| {p['epoch']} | {p['loss']:.3f} | {p['r@1']:.4f} | "
                        f"{p['r@3']:.4f} | {p['mrr']:.4f} |"
                    )
            lines.append("")

    lines += [
        "",
        "Qumran / hologram-E — jutro (poza P0).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
