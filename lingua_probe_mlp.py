"""
Lingua MLP probe — uczenie tradycyjne (trenowany klasyfikator vs analityczny
HRR unbind)
==================================================================================

Inne pytanie niż Lingua P1 (run_lingua_p1.py, Most vs Softmax na CAŁYM
grafie Φ). Tutaj: zamiast analitycznego dekodowania (unbind_position +
nearest_morpheme = najbliższy sąsiad po cosine similarity do prototypu
morfemu — dokładnie to, co robi _run_r1 w karmazyn_lingua.py), trenujemy
zwykły MLP (cross-entropy, Adam, BEZ żadnej wiedzy o HRR/pozycji/bind) na
surowym, złożonym wektorze słowa (ten sam FEATURE_DIM-wymiarowy fragment,
na którym operuje unbind_position) i pytamy: czy TRENOWANY dekoder robi
lepiej niż analityczny najbliższy-sąsiad, zwłaszcza tam gdzie analityczny
FAILuje (hologram, fazy stłoczone 30°-60°, r@1=0.33)?

Cztery warunki — te same co w _demo_retrieval karmazyn_lingua.py:
  symbolic        Im=0 zawsze. PREDYKCJA A PRIORI: nawet trenowany
                   klasyfikator nie może wygrać — trzy prefiksy to
                   DOSŁOWNIE ten sam wektor cech (amplitude=0.8 na tym
                   samym slocie, faza zignorowana), więc accuracy ma
                   twardy sufit 1/3 dla KAŻDEGO dekodera. To test tego,
                   czy uczenie potrafi odróżnić "informacja zniszczona"
                   od "informacja obecna, ale trudna do wydobycia
                   najbliższym sąsiadem" — jeśli MLP też utknie na 1/3,
                   to potwierdza że to pierwsze, nie drugie.
  hologram_close  fazy 30°/45°/60° (analitycznie r@1=0.33 — czy to
                   ograniczenie METODY dekodowania, czy sygnału?)
  hologram_far    fazy 0°/120°/240° (analitycznie r@1=1.0 — sufit)
  onehot          każdy prefiks własny slot (analitycznie r@1=1.0 — sufit)

Dane: N próbek na klasę = czysty złożony wektor słowa + szum gaussowski
(ta sama konwencja augmentacji co dump_recall.graph_to_tensors), żeby
było co dzielić na train/val — jeden "czysty" przykład na klasę nie
daje uczenia.

  python lingua_probe_mlp.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

from karmazyn_lingua import FEATURE_DIM, AtomEncoding, FeatureOverlay, LinguaLayer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "doswiadczenie" / "glify"

SEEDS = [0, 1, 17, 23, 42]
N_PER_CLASS_TRAIN = 80
N_PER_CLASS_VAL = 20
NOISE = 0.08          # ta sama skala co default_noise w dump_recall.py
EPOCHS = 60
HIDDEN = 32
LR = 1e-3

CONDITIONS: dict[str, tuple[AtomEncoding, dict[str, float]]] = {
    "symbolic": (AtomEncoding.SYMBOLIC, {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}),
    "hologram_close": (AtomEncoding.HOLOGRAM, {"na-": math.pi / 4, "prze-": math.pi / 3, "wy-": math.pi / 6}),
    "hologram_far": (AtomEncoding.HOLOGRAM, {"na-": 0.0, "prze-": 2 * math.pi / 3, "wy-": 4 * math.pi / 3}),
    "onehot": (AtomEncoding.SYMBOLIC_ONEHOT, {"na-": 0.0, "prze-": 0.0, "wy-": 0.0}),  # faza ignorowana
}


def _clean_word_vectors(encoding: AtomEncoding, prefiksy_faza: dict[str, float]) -> dict[str, np.ndarray]:
    """Jeden czysty (bez szumu) złożony wektor słowa na prefiks — dokładnie
    to, na czym w _run_r1/_run_onehot_baseline operuje unbind_position."""
    import karmazyn_substrate as sub

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)
    rdzen = lingua.make_morpheme("pisa", [FeatureOverlay("r", 1.0, 0.0, 0)], encoding)

    names = list(prefiksy_faza.keys())
    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        # Twardy baseline wymaga WŁASNEGO slotu na prefiks (patrz karmazyn_lingua.py) —
        # faza jest tu ignorowana, więc słownik faz służy tylko do ustalenia nazw.
        slots = {n: i + 1 for i, n in enumerate(names)}
    else:
        slots = {n: 1 for n in names}  # CELOWA kolizja (ablacja hologram/symbolic)

    out: dict[str, np.ndarray] = {}
    for nazwa, phase in prefiksy_faza.items():
        prefiks = lingua.make_morpheme(
            nazwa, [FeatureOverlay(f"a_{nazwa}", 0.8, phase, slots[nazwa])], encoding
        )
        slowo = lingua.compose(f"{nazwa}pisać_probe", [prefiks, rdzen], kind="word")
        out[nazwa] = np.asarray(slowo.vector[:FEATURE_DIM], dtype=float)
    return out


def _make_split(clean: dict[str, np.ndarray], n_per_class: int, *, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    names = list(clean.keys())
    X = np.zeros((n_per_class * len(names), FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n_per_class * len(names), dtype=np.int64)
    i = 0
    for cls, nazwa in enumerate(names):
        base = clean[nazwa]
        for _ in range(n_per_class):
            X[i] = base + noise * rng.standard_normal(FEATURE_DIM)
            y[i] = cls
            i += 1
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm]), names


class Probe(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_one(condition: str, seed: int) -> dict:
    encoding, prefiksy_faza = CONDITIONS[condition]
    torch.manual_seed(seed)
    clean = _clean_word_vectors(encoding, prefiksy_faza)
    n_classes = len(clean)

    Xtr, ytr, names = _make_split(clean, N_PER_CLASS_TRAIN, noise=NOISE, seed=1000 + seed)
    Xva, yva, _ = _make_split(clean, N_PER_CLASS_VAL, noise=NOISE, seed=9000 + seed)

    torch.manual_seed(seed)
    model = Probe(FEATURE_DIM, HIDDEN, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(Xva).argmax(dim=-1) == yva).float().mean().item()
    return {"condition": condition, "seed": seed, "acc": acc}


def run() -> dict:
    by_condition = []
    all_rows: list[dict] = []
    chance = 1.0 / 3.0

    for cond in CONDITIONS:
        print(f"========== {cond} ==========")
        rows = [run_one(cond, seed) for seed in SEEDS]
        for r in rows:
            print(f"  seed={r['seed']}  acc={r['acc']:.3f}")
        all_rows.extend(rows)
        accs = [r["acc"] for r in rows]
        by_condition.append({
            "condition": cond,
            "mean_acc": sum(accs) / len(accs),
            "min_acc": min(accs),
            "max_acc": max(accs),
            "above_chance": (sum(accs) / len(accs)) - chance,
        })

    report = {
        "protocol": "Lingua MLP probe — trenowany klasyfikator vs analityczny HRR unbind",
        "seeds": SEEDS,
        "conditions": list(CONDITIONS.keys()),
        "epochs": EPOCHS,
        "noise": NOISE,
        "n_per_class_train": N_PER_CLASS_TRAIN,
        "n_per_class_val": N_PER_CLASS_VAL,
        "chance_level": chance,
        "rows": all_rows,
        "by_condition": by_condition,
        "analytic_reference": {
            "symbolic": 0.333, "hologram_close": 0.333, "hologram_far": 1.0, "onehot": 1.0,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lingua_probe_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_md(report)
    for b in by_condition:
        print(f"{b['condition']:16s} mean_acc={b['mean_acc']:.3f}  (analityczne: "
              f"{report['analytic_reference'][b['condition']]:.3f})")
    return report


def _write_md(report: dict) -> None:
    lines = [
        "# Lingua MLP probe — trenowany klasyfikator vs analityczny HRR unbind",
        "",
        f"Seedy: {report['seeds']} · epochs={report['epochs']} · noise={report['noise']} · "
        f"n/klasę train={report['n_per_class_train']} val={report['n_per_class_val']}",
        "",
        "Kontrast z `run_lingua_p1.py` (Most vs Softmax na CAŁYM grafie Φ) i z "
        "analitycznym `unbind_position`+`nearest_morpheme` w `_run_r1`/"
        "`_run_onehot_baseline` (karmazyn_lingua.py): tu zwykły MLP "
        "(cross-entropy, bez żadnej wiedzy o HRR/pozycji) uczy się na surowym "
        "złożonym wektorze słowa, czy odróżni prefiks — pytanie: czy TRENOWANY "
        "dekoder robi lepiej niż analityczny najbliższy-sąsiad tam, gdzie ten "
        "drugi FAILuje (hologram, fazy stłoczone).",
        "",
        "## Wynik vs referencja analityczna",
        "",
        "| warunek | mean acc (MLP) | min | max | powyżej przypadku (1/3) | r@1 analityczny |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b in report["by_condition"]:
        ref = report["analytic_reference"][b["condition"]]
        lines.append(
            f"| {b['condition']} | {b['mean_acc']:.3f} | {b['min_acc']:.3f} | {b['max_acc']:.3f} | "
            f"{b['above_chance']:+.3f} | {ref:.3f} |"
        )
    lines += [
        "",
        "## Pełna tabela",
        "",
        "| warunek | seed | acc |",
        "|---|---:|---:|",
    ]
    for r in report["rows"]:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['acc']:.3f} |")
    lines += [
        "",
        "JSON: `lingua_probe_results.json`",
        "",
    ]
    (OUT_DIR / "LINGUA_PROBE_MLP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Lingua-probe] wrote {OUT_DIR / 'LINGUA_PROBE_MLP.md'}")


if __name__ == "__main__":
    run()
