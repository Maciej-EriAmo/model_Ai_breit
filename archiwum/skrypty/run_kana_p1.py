"""
Kana P1 — cel pierwotny raportu hologram-znaki (sekcja 1): czy hologram
faktycznie pomaga Mostowi uczyć się PRĘDZEJ (mniej epok/przebiegów do tego
samego wyniku), nie tylko poprawia gołe nearest-neighbour r@1 z
karmazyn_lingua.py. Nigdzie w tym repo (przed tym plikiem) nie było
testu treningowego na TEJ reprezentacji poza Lingua P1 (run_lingua_p1.py),
a ten miał tylko 3 słowa / 11 atomów — "za mało sygnału" (patrz
LINGUA_P1.md, sekcja 'Uwaga: prawdopodobny confound'). Kana P1 to ta sama
metoda (Most vs Softmax vs zero-tracer ablation, bramka ΔSM =
CANON.gate_recall_r1), ale na 71 prawdziwych tożsamościach dźwiękowych
(hiragana/katakana/romaji Hepburn, bez kanji — patrz kana_corpus.py) =
286 atomów, 26x więcej niż Lingua P1.

SYMBOLIC_ONEHOT pominięty: kana_corpus.przyklad_kana_korpus go nie
obsługuje (onehot nie używa fazy, nie ma tu nic do zmierzenia) —
porównanie to HOLOGRAM vs SYMBOLIC, centralne pytanie sekcji 5 raportu.

  python run_kana_p1.py
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

# 4 wątki zamiast 1 (jak Lingua P1): use_deterministic_algorithms wymusza
# deterministyczne jądra CPU nawet wielowątkowo — zweryfikowane empirycznie
# przed tą zmianą (r@1 bit-identyczny na threads=1/4/8, powtarzalny na
# threads=4), ~1.8x przyspieszenie treningu, 0 zysku powyżej 4 na tym
# malutkim modelu (d_model=32).
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True, warn_only=True)

from config import CANON
from dump_recall import load_raw_records, make_dump_loaders
from karmazyn_lingua import AtomEncoding
from make_dump_kana import write_dump
from phimost_learn import build_model
from recall_model import PhiBridgeEncoder, PhiSoftmaxEncoder, train_encoder

ROOT = Path(__file__).resolve().parents[2]
DUMP_ROOT = ROOT / "dumps" / "from_kana_p1"
OUT_DIR = ROOT / "doswiadczenie" / "glify"

ENCODINGS = [AtomEncoding.HOLOGRAM, AtomEncoding.SYMBOLIC]
SEEDS = [0, 1, 17, 23, 42]  # te same 5 co Lingua P1 / Holo-Phi P2
EPOCHS = 16  # te same hiperparametry co Lingua P1 -- ta sama metoda, wiekszy korpus
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
        input_dim=CANON.d_x, dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers, d=CANON.d_model,
    )
    sm_m = train_encoder(sm, train, val, epochs=EPOCHS, lr=CANON.lr)

    torch.manual_seed(seed)
    most_z = PhiBridgeEncoder(
        input_dim=CANON.d_x, dim_tracer=CANON.n_dim,
        num_layers=CANON.num_layers, d=CANON.d_model, coupling=CANON.coupling,
    )
    z_m = train_encoder(most_z, train, val, epochs=EPOCHS, lr=CANON.lr, ablate_kw={"zero_tracer": True})

    d_sm = most_m["r@1"] - sm_m["r@1"]
    return {
        "seed": seed, "most_r1": most_m["r@1"], "sm_r1": sm_m["r@1"], "zero_r1": z_m["r@1"],
        "delta_sm": d_sm, "gate_pass": d_sm >= CANON.gate_recall_r1,
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
            t0 = time.time()
            r = run_one(out, seed, bs=bs)
            r["encoding"] = enc.value
            tag = "PASS" if r["gate_pass"] else "FAIL"
            print(
                f"  seed={seed}  Most={r['most_r1']:.3f} SM={r['sm_r1']:.3f} "
                f"ZERO={r['zero_r1']:.3f} DSM={r['delta_sm']:+.4f} {tag} "
                f"({time.time()-t0:.1f}s)"
            )
            rows.append(r)
            all_rows.append(r)

        n_pass = sum(1 for r in rows if r["gate_pass"])
        deltas = [r["delta_sm"] for r in rows]
        by_encoding.append({
            "encoding": enc.value, "n_atoms": n_atoms, "n_pass": n_pass, "n_seeds": len(rows),
            "mean_delta_sm": sum(deltas) / len(deltas),
            "best_delta_sm": max(deltas), "worst_delta_sm": min(deltas),
            "mean_most": sum(r["most_r1"] for r in rows) / len(rows),
            "mean_sm": sum(r["sm_r1"] for r in rows) / len(rows),
            "mean_zero": sum(r["zero_r1"] for r in rows) / len(rows),
        })

    best = max(by_encoding, key=lambda x: (x["n_pass"], x["mean_delta_sm"]))
    report = {
        "protocol": "Kana P1 - cel pierwotny: czy hologram pomaga Mostowi uczyc sie predzej (71 tozsamosci, bez kanji)",
        "seeds": SEEDS, "encodings": [e.value for e in ENCODINGS], "epochs": EPOCHS,
        "n_train": N_TRAIN, "n_val": N_VAL,
        "rows": all_rows, "by_encoding": by_encoding, "best_encoding": best["encoding"],
        "verdict": (
            f"best encoding={best['encoding']} PASS {best['n_pass']}/{best['n_seeds']} "
            f"meanDSM={best['mean_delta_sm']:+.4f}; gate = Most r@1 - Softmax r@1 >= "
            f"CANON.gate_recall_r1 ({CANON.gate_recall_r1})"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "kana_p1_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    print(report["verdict"])
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Kana P1 - cel pierwotny raportu hologram-znaki: trening, nie retrieval",
        "",
        f"Seedy: {report['seeds']} - epochs={report['epochs']} - "
        f"encodings={report['encodings']} - n_train={report['n_train']} n_val={report['n_val']}",
        "",
        "Pierwszy test CELU PIERWOTNEGO z sekcji 1 raportu hologram-znaki "
        "(oszczednosc przebiegow treningowych) na realnym, duzym korpusie: "
        "71 tozsamosci dzwiekowych (hiragana/katakana/romaji Hepburn, bez "
        "kanji) = 286 atomow, 26x wiecej niz Lingua P1 (3 slowa, 11 atomow), "
        "ktory sam siebie oznaczyl jako 'za malo sygnalu'.",
        "",
        "## Agregat per encoding",
        "",
        "| encoding | atoms | PASS | mean DSM | best DSM | worst DSM | mean Most | mean SM | mean ZERO |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in report["by_encoding"]:
        lines.append(
            f"| {b['encoding']} | {b['n_atoms']} | {b['n_pass']}/{b['n_seeds']} | "
            f"{b['mean_delta_sm']:+.4f} | {b['best_delta_sm']:+.4f} | {b['worst_delta_sm']:+.4f} | "
            f"{b['mean_most']:.3f} | {b['mean_sm']:.3f} | {b['mean_zero']:.3f} |"
        )
    lines += [
        "", f"**Najlepszy encoding (PASS, potem meanD):** **{report['best_encoding']}**", "",
        "## Pelna tabela", "",
        "| encoding | seed | Most | SM | ZERO | DSM | gate |",
        "|---|---:|---:|---:|---:|---:|:----:|",
    ]
    for r in report["rows"]:
        tag = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['encoding']} | {r['seed']} | {r['most_r1']:.3f} | {r['sm_r1']:.3f} | "
            f"{r['zero_r1']:.3f} | {r['delta_sm']:+.4f} | {tag} |"
        )
    lines += ["", f"**Werdykt:** {report['verdict']}", "", "JSON: `kana_p1_results.json`", ""]
    (OUT_DIR / "KANA_P1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Kana-P1] wrote {OUT_DIR / 'KANA_P1.md'}")


if __name__ == "__main__":
    run()
