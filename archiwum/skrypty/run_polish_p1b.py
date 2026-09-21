"""Polish P1b — alfabet 32 litery, metoda jak Kana P1b, 2× epok (128).

  python run_polish_p1b.py
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import make_dump_loaders
from karmazyn_lingua import AtomEncoding, to_phi_atoms
from phimost_learn import build_model
from polish_corpus import POLISH_LETTERS, przyklad_polski_alfabet
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
DUMP_ROOT = ROOT / "dumps" / "from_polish_p1b"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

SEEDS = [0, 1, 17]
EPOCHS = 128  # 2× Kana P1b (64)
N_TRAIN = 160
N_VAL = 40
ENCODINGS = [AtomEncoding.HOLOGRAM, AtomEncoding.SYMBOLIC]


def write_dump(out: Path, *, encoding: AtomEncoding) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    lingua, slowa = przyklad_polski_alfabet(encoding)
    atoms, edges, meta = to_phi_atoms(
        lingua,
        slowa,
        page_id="pl_alpha_page",
        corpus_note="polish_p1b: 32 litery alfabetu PL, grupy~5, metoda jak kana_p1b",
    )
    with (out / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (out / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    man = {
        "name": out.name,
        "version": "0.1",
        "method": "polish_p1b_alphabet",
        "description": "P1b PL: 32 litery alfabetu, epochs=128 (2x kana_p1b)",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "n_letters": len(POLISH_LETTERS),
        "source": "run_polish_p1b.py",
        "encoding": encoding.value,
        **meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"OK {out} atoms={len(atoms)} edges={len(edges)} HOT={n_hot} encoding={encoding.value}")
    return len(atoms), len(edges)


def run_one(dump: Path, seed: int, *, bs: int) -> dict:
    g, train, val, _vc = make_dump_loaders(
        dump,
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
            epochs=EPOCHS,
            lr=CANON.lr,
            ablate_kw=ablate,
            return_curve=True,
        )

    print("[train] cold start (CANON)")
    most = _train(lambda: build_model(warm=False), {})
    sm = _train(
        lambda: PhiSoftmaxEncoder(
            input_dim=CANON.d_x,
            dim_tracer=CANON.n_dim,
            num_layers=CANON.num_layers,
            d=CANON.d_model,
        ),
        {},
    )
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
    dsm = most["r@1"] - sm["r@1"]
    return {
        "seed": seed,
        "N": N,
        "chance_r1": chance,
        "most_r1": most["r@1"],
        "sm_r1": sm["r@1"],
        "zero_r1": zero["r@1"],
        "most_r3": most["r@3"],
        "sm_r3": sm["r@3"],
        "most_mrr": most["mrr"],
        "sm_mrr": sm["mrr"],
        "delta_sm": dsm,
        "gate_pass": dsm >= CANON.gate_recall_r1,
        "above_chance_most": most["r@1"] >= 5 * chance,
        "curves": {
            "Most": most["curve"],
            "Softmax": sm["curve"],
            "ZERO": zero["curve"],
        },
    }


def main() -> dict:
    all_rows: list[dict] = []
    by_encoding: list[dict] = []

    for enc in ENCODINGS:
        out = DUMP_ROOT / enc.value
        n_atoms, n_edges = write_dump(out, encoding=enc)
        bs = min(16, max(4, n_atoms))
        chance = 1.0 / max(1, n_atoms - 1)
        print(
            f"========== encoding={enc.value} (atoms={n_atoms} edges={n_edges} "
            f"bs={bs} chance={chance:.4f} epochs={EPOCHS}) =========="
        )
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            r = run_one(out, seed, bs=bs)
            r["encoding"] = enc.value
            tag = "PASS" if r["gate_pass"] else "FAIL"
            floor = "ABOVE" if r["above_chance_most"] else "FLOOR"
            print(
                f"  seed={seed}  Most={r['most_r1']:.3f} SM={r['sm_r1']:.3f} "
                f"ZERO={r['zero_r1']:.3f} DSM={r['delta_sm']:+.4f} {tag} "
                f"Most={floor} ({time.time()-t0:.1f}s)"
            )
            for name in ("Most", "Softmax"):
                c = r["curves"][name]
                pts = [row for row in c if row["epoch"] in (1, 32, 64, 128)]
                print(
                    f"    {name} loss/r@1: "
                    + " | ".join(
                        f"ep{p['epoch']}={p['loss']:.2f}/{p['r@1']:.3f}" for p in pts
                    )
                )
            rows.append(r)
            all_rows.append(r)

        n_pass = sum(1 for r in rows if r["gate_pass"])
        n_above = sum(1 for r in rows if r["above_chance_most"])
        deltas = [r["delta_sm"] for r in rows]
        by_encoding.append({
            "encoding": enc.value,
            "n_atoms": n_atoms,
            "n_pass": n_pass,
            "n_above_chance": n_above,
            "n_seeds": len(rows),
            "mean_delta_sm": sum(deltas) / len(deltas),
            "best_delta_sm": max(deltas),
            "worst_delta_sm": min(deltas),
            "mean_most": sum(r["most_r1"] for r in rows) / len(rows),
            "mean_sm": sum(r["sm_r1"] for r in rows) / len(rows),
            "mean_zero": sum(r["zero_r1"] for r in rows) / len(rows),
            "chance_r1": chance,
        })

    best = max(
        by_encoding,
        key=lambda x: (x["n_above_chance"], x["n_pass"], x["mean_delta_sm"]),
    )
    report = {
        "protocol": "Polish P1b - alfabet 32, metoda jak kana_p1b, epochs=128 (2x)",
        "letters": "".join(POLISH_LETTERS),
        "seeds": SEEDS,
        "encodings": [e.value for e in ENCODINGS],
        "epochs": EPOCHS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "baseline_kana_p1b": {
            "epochs": 64,
            "n_atoms": 59,
            "mean_most_hologram": 0.293,
            "mean_most_symbolic": 0.297,
        },
        "rows": all_rows,
        "by_encoding": by_encoding,
        "best_encoding": best["encoding"],
        "verdict": (
            f"best={best['encoding']} above_chance={best['n_above_chance']}/{best['n_seeds']} "
            f"PASS={best['n_pass']}/{best['n_seeds']} meanMost={best['mean_most']:.3f} "
            f"meanDSM={best['mean_delta_sm']:+.4f} (chance≈{best['chance_r1']:.4f}; "
            f"epochs={EPOCHS}=2×kana_p1b)"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "polish_p1b_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(report["verdict"])
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Polish P1b — alfabet (32), metoda jak Kana P1b, 128 epok",
        "",
        f"Litery: `{report['letters']}`",
        f"Seedy: {report['seeds']} · epochs={report['epochs']} (=2× Kana P1b) · "
        f"n_train={report['n_train']} n_val={report['n_val']}",
        "",
        "Ta sama metoda: FeatureOverlay(grupa, pozycja) + compose grupy → "
        "dump Φ → Most/Softmax/ZERO. Próg floor: Most r@1 ≥ 5×chance.",
        "",
        "## Agregat",
        "",
        "| encoding | atoms | chance | above_floor | PASS | mean Most | mean SM | mean ZERO | mean DSM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in report["by_encoding"]:
        lines.append(
            f"| {b['encoding']} | {b['n_atoms']} | {b['chance_r1']:.4f} | "
            f"{b['n_above_chance']}/{b['n_seeds']} | {b['n_pass']}/{b['n_seeds']} | "
            f"{b['mean_most']:.3f} | {b['mean_sm']:.3f} | {b['mean_zero']:.3f} | "
            f"{b['mean_delta_sm']:+.4f} |"
        )
    base = report["baseline_kana_p1b"]
    lines += [
        "",
        f"**Werdykt:** {report['verdict']}",
        "",
        f"Baseline Kana P1b (64 ep): mean Most holo={base['mean_most_hologram']:.3f} "
        f"sym={base['mean_most_symbolic']:.3f} (N={base['n_atoms']}).",
        "",
        "## Per seed",
        "",
        "| encoding | seed | Most | SM | ZERO | DSM | floor | gate |",
        "|---|---:|---:|---:|---:|---:|:----:|:----:|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['encoding']} | {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | "
            f"{'ABOVE' if r['above_chance_most'] else 'FLOOR'} | "
            f"{'PASS' if r['gate_pass'] else 'FAIL'} |"
        )
    sample = next(
        (r for r in report["rows"] if r["encoding"] == "hologram" and r["seed"] == 0),
        None,
    )
    if sample:
        lines += ["", "## Krzywa (hologram seed=0)", ""]
        for name in ("Most", "Softmax", "ZERO"):
            lines.append(f"### {name}")
            lines.append("| ep | loss | r@1 | r@3 | mrr |")
            lines.append("|---:|-----:|----:|----:|----:|")
            for p in sample["curves"][name]:
                if p["epoch"] in (1, 16, 32, 64, 96, 128):
                    lines.append(
                        f"| {p['epoch']} | {p['loss']:.3f} | {p['r@1']:.4f} | "
                        f"{p['r@3']:.4f} | {p['mrr']:.4f} |"
                    )
            lines.append("")
    lines += ["JSON: `polish_p1b_results.json`", ""]
    (OUT_DIR / "POLISH_P1B.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
