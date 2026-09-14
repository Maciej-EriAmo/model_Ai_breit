"""
Lingua P1 — kontrola eksperymentalna: czy Most faktycznie uczy się recall
na atomach z karmazyn_lingua (morfem -> HRR bind -> słowo), i czy to
zależy od reprezentacji (hologram / symbolic / onehot)?

Pierwszy P-numerowany eksperyment dla karmazyn_lingua, wzorowany na
run_holo_p1.py/run_holo_p2.py (ta sama metoda: Most vs Softmax vs
zero-tracer ablation, bramka ΔSM = CANON.gate_recall_r1). do_phi_atoms
dawał dotąd tylko ślepy eksport (nikt nie sprawdził, czy Most cokolwiek
się z niego uczy) — to jest ta sprawdzian.

Graf jest MAŁY (11 atomów: corpus+page+3 słowa+6 morfemów) w porównaniu
do glify (setki atomów) — batch_size = cały graf, epoki tanie.

  python run_lingua_p1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Determinizm run-to-run (patrz run_holo_p1.py / HOLO_WNIOSKIab.md §5).
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import load_raw_records, make_dump_loaders
from karmazyn_lingua import AtomEncoding
from make_dump_lingua import write_dump
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parent
DUMP_ROOT = ROOT / "dumps" / "from_lingua_p1"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

ENCODINGS = [AtomEncoding.HOLOGRAM, AtomEncoding.SYMBOLIC, AtomEncoding.SYMBOLIC_ONEHOT]
SEEDS = [0, 1, 17, 23, 42]  # te same 5 co Holo-Phi P2 (PASS(23) + typowe FAIL)
EPOCHS = 16
N_TRAIN = 80
N_VAL = 25


def run_one(dump_dir: Path, seed: int, *, bs: int) -> dict:
    torch.manual_seed(seed)
    _g, train, val, _vc = make_dump_loaders(
        dump_dir,
        n_train=N_TRAIN,
        n_val=N_VAL,
        batch_size=bs,
        d_x=CANON.d_x,
        n_dim=CANON.n_dim,
        pair_mode="mixed",
        noise_scale=0.02,
        seed0=1000 + seed * 17,
    )
    torch.manual_seed(seed)
    most = build_model(warm=False)
    torch.manual_seed(seed)
    most_m = train_encoder(most, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    sm = PhiSoftmaxEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
    )
    sm_m = train_encoder(sm, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    most_z = PhiBridgeEncoder(
        input_dim=CANON.d_x,
        dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers,
        d=CANON.d_model,
        coupling=CANON.coupling,
    )
    z_m = train_encoder(
        most_z, train, val, epochs=EPOCHS, lr=CANON.lr,
        ablate_kw={"zero_tracer": True},
    )
    d_sm = most_m["r@1"] - sm_m["r@1"]
    return {
        "seed": seed,
        "most_r1": most_m["r@1"],
        "sm_r1": sm_m["r@1"],
        "zero_r1": z_m["r@1"],
        "delta_sm": d_sm,
        "gate_pass": d_sm >= CANON.gate_recall_r1,
    }


def run() -> dict:
    by_encoding = []
    all_rows: list[dict] = []

    for enc in ENCODINGS:
        out = DUMP_ROOT / enc.value
        n_atoms, n_edges, _meta = write_dump(out, encoding=enc)
        bs = min(16, max(4, n_atoms))

        print(f"========== encoding={enc.value} (atoms={n_atoms} edges={n_edges} bs={bs}) ==========")
        rows = []
        for seed in SEEDS:
            r = run_one(out, seed, bs=bs)
            r["encoding"] = enc.value
            tag = "PASS" if r["gate_pass"] else "FAIL"
            print(
                f"  seed={seed}  Most={r['most_r1']:.3f} SM={r['sm_r1']:.3f} "
                f"ZERO={r['zero_r1']:.3f} ΔSM={r['delta_sm']:+.4f} {tag}"
            )
            rows.append(r)
            all_rows.append(r)

        n_pass = sum(1 for r in rows if r["gate_pass"])
        deltas = [r["delta_sm"] for r in rows]
        by_encoding.append({
            "encoding": enc.value,
            "n_atoms": n_atoms,
            "n_pass": n_pass,
            "n_seeds": len(rows),
            "mean_delta_sm": sum(deltas) / len(deltas),
            "best_delta_sm": max(deltas),
            "worst_delta_sm": min(deltas),
            "mean_most": sum(r["most_r1"] for r in rows) / len(rows),
            "mean_sm": sum(r["sm_r1"] for r in rows) / len(rows),
            "mean_zero": sum(r["zero_r1"] for r in rows) / len(rows),
        })

    best = max(by_encoding, key=lambda x: (x["n_pass"], x["mean_delta_sm"]))
    report = {
        "protocol": "Lingua P1 — czy Most uczy się recall na karmazyn_lingua, zależność od encoding",
        "seeds": SEEDS,
        "encodings": [e.value for e in ENCODINGS],
        "epochs": EPOCHS,
        "rows": all_rows,
        "by_encoding": by_encoding,
        "best_encoding": best["encoding"],
        "verdict": (
            f"best encoding={best['encoding']} PASS {best['n_pass']}/{best['n_seeds']} "
            f"meanΔSM={best['mean_delta_sm']:+.4f}; "
            "gate = Most r@1 - Softmax r@1 >= CANON.gate_recall_r1 "
            f"({CANON.gate_recall_r1})"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lingua_p1_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(report["verdict"])
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Lingua P1 — czy Most uczy się recall na karmazyn_lingua?",
        "",
        f"Seedy: {report['seeds']} · epochs={report['epochs']} · "
        f"encodings={report['encodings']}",
        "",
        "Pierwszy P-numerowany eksperyment dla `karmazyn_lingua.py`: dotąd "
        "`to_phi_atoms`/`make_dump_lingua.py` był ślepym eksporterem — nikt "
        "nie sprawdził, czy silnik Most (phimost_learn/recall_model) "
        "faktycznie wyciąga coś z tej reprezentacji, ani czy hologram/"
        "symbolic/onehot różnią się z punktu widzenia Mostu (nie tylko "
        "gołego nearest-neighbour r@1 z demo w karmazyn_lingua.py).",
        "",
        "## Agregat per encoding",
        "",
        "| encoding | atoms | PASS | mean ΔSM | best ΔSM | worst ΔSM | mean Most | mean SM | mean ZERO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in report["by_encoding"]:
        lines.append(
            f"| {b['encoding']} | {b['n_atoms']} | {b['n_pass']}/{b['n_seeds']} | "
            f"{b['mean_delta_sm']:+.4f} | {b['best_delta_sm']:+.4f} | {b['worst_delta_sm']:+.4f} | "
            f"{b['mean_most']:.3f} | {b['mean_sm']:.3f} | {b['mean_zero']:.3f} |"
        )
    lines += [
        "",
        f"**Najlepszy encoding (PASS, potem meanΔ):** **{report['best_encoding']}**",
        "",
        "## Pełna tabela",
        "",
        "| encoding | seed | Most | SM | ZERO | ΔSM | gate |",
        "|---|---:|---:|---:|---:|---:|:----:|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['encoding']} | {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | {tag} |"
        )
    lines += [
        "",
        f"**Werdykt:** {report['verdict']}",
        "",
        "JSON: `lingua_p1_results.json`",
        "",
    ]
    (OUT_DIR / "LINGUA_P1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Lingua-P1] wrote {OUT_DIR / 'LINGUA_P1.md'}")


if __name__ == "__main__":
    run()
