"""
Holo-Φ P5 — dump v2: graf-per-radykał (opcja A, HOLO_P5_PLAN_IMPLEMENTACJI.md §0/§1).

Różnica vs P0-P4a (`holo_phi.py`/`make_dump_holo.py`):
  - τ (T) = energy_to_T(stroke_count(ch), ref) — prawdziwy pomiar z Unihan,
    NIE hash(etykieta). Kanon `energy_to_T` (holo_phi.py) się nie zmienia,
    zmienia się tylko wejście (P5 §2.1).
  - Klasa/rodzina do recall = **radykał Kangxi** (unihan_tracer.radical_id),
    zewnętrzny fakt słownikowy — nie wstrzyknięta ręcznie etykieta jak φ_root w P0.
  - Train = osobne małe grafy per radykał (izolacja jak cutout w DESI/bench_desi.py).
    Holdout = CAŁY radykał nigdy obecny w żadnym forward passie treningowym —
    dopiero w eval miesza się z dystraktorami z radykałów treningowych.

Uwaga zakresowa: w odróżnieniu od P0-P4a (`make_dump_holo.py` → atoms.jsonl na dysku,
wczytywane przez `dump_recall.py`), ten moduł buduje `RecallBatch` bezpośrednio
(bez rundtripu JSONL) — te same prymitywy (`PhiAtom`/`atom_content_vec`/
`atom_tracer_vec`), mniej maszynerii. Manifest audytowy nadal zapisywany na dysk.
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_LAB = _Path(__file__).resolve().parents[2]
if str(_LAB) not in sys.path:
    sys.path.insert(0, str(_LAB))

import json
import random
from pathlib import Path

import torch

from holo_phi import energy_to_T
from phi_schema import PhiAtom, PhiState, atom_content_vec, atom_tracer_vec
from recall_data import RecallBatch
from unihan_tracer import RADICALS, chars_for_radical, stroke_count

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_holo_v2"

HOLDOUT_RADICAL = 85  # 水 — wybór arbitralny, ustalony przed P5.1 (§0 decyzja)
TRAIN_RADICALS = tuple(r for r in RADICALS if r != HOLDOUT_RADICAL)

# wspólny ref energii dla energy_to_T — max stroke_count w CAŁYM bundlu (P5.0),
# nie tylko w train, żeby holdout nie dostawał τ poza zakresem treningowym.
_ALL_CHARS = [ch for r in RADICALS for ch in chars_for_radical(r)]
_STROKE_REF = float(max(stroke_count(ch) for ch in _ALL_CHARS))


def _atom_for_char(ch: str, radical: int) -> PhiAtom:
    t = energy_to_T(float(stroke_count(ch)), ref=_STROKE_REF)
    return PhiAtom(
        atom_id=f"{radical}/{ch}",
        S=ch,
        E=ch,
        T=t,
        state=PhiState.HOT,
        bubble_id=f"radical/{radical}",
        depth=0,
        payload=ch,
        mazur_tracer=None,
    )


def _pack(atoms: list[PhiAtom], d_x: int, n_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    t_max = max((a.T for a in atoms), default=1.0) or 1.0
    content = torch.zeros(len(atoms), d_x)
    tracer = torch.zeros(len(atoms), n_dim)
    for i, a in enumerate(atoms):
        aa = PhiAtom(
            atom_id=a.atom_id, S=a.S, E=a.E, T=a.T / t_max, state=a.state,
            bubble_id=a.bubble_id, depth=a.depth, payload=a.payload,
            mazur_tracer=a.mazur_tracer,
        )
        content[i] = atom_content_vec(aa, d_x)
        tracer[i] = atom_tracer_vec(aa, n_dim)
    return content, tracer


def build_train_batch(
    radical: int,
    *,
    d_x: int,
    n_dim: int,
    batch_size: int,
    seed: int,
    noise_scale: float = 0.05,
) -> RecallBatch:
    """Jeden radykał = jedna 'komórka izolacji' (jak cutout DESI). Pozytyw = dowolna
    inna postać tego samego radykału — cały graf to jedna rodzina."""
    chars = chars_for_radical(radical)
    atoms = [_atom_for_char(ch, radical) for ch in chars]
    N = len(atoms)
    rng = random.Random(seed)

    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.ones(batch_size, N)
    topics = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    c0, t0 = _pack(atoms, d_x, n_dim)
    for b in range(batch_size):
        ns = 0.0 if b == 0 else noise_scale
        c = c0.clone()
        if ns > 0:
            c = c + ns * torch.randn(N, d_x)
            c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        content[b] = c
        tracer[b] = t0
        qi, pi = rng.sample(range(N), 2)
        q_idx[b], p_idx[b] = qi, pi

    return RecallBatch(
        content=content, tracer=tracer, depths=depths, states=states,
        mask=mask, query_idx=q_idx, pos_idx=p_idx, topics=topics,
    )


def build_mixed_train_batch(
    radicals: tuple[int, ...],
    *,
    d_x: int,
    n_dim: int,
    batch_size: int,
    seed: int,
    noise_scale: float = 0.05,
) -> RecallBatch:
    """Jeden graf = WSZYSTKIE radykały treningowe razem. Pozytyw = ten sam
    radykał co query; kandydaci = cały graf (inne radykały = prawdziwe
    negatywy cross-family podczas treningu).

    `build_train_batch` (jeden-radykał-per-graf) dawał zerowy sygnał
    kontrastowy: w grafie złożonym z samego jednego radykału KAŻDY kandydat
    jest poprawnym pozytywem, więc τ nie ma nic do odróżniania — model uczy
    się dowolnego, samo-spójnego rozmieszczenia 20 elementów (trywialne
    train_r1→100%), co nie ma żadnego powodu przenieść się na holdout
    (zob. HOLO_P5_1.md). Ta wersja daje τ realną robotę: musi pomóc wybrać
    właściwy radykał spośród kandydatów z innych radykałów w tym samym grafie.
    """
    rng = random.Random(seed)
    atoms: list[PhiAtom] = []
    radical_of_idx: list[int] = []
    idx_by_radical: dict[int, list[int]] = {r: [] for r in radicals}
    for r in radicals:
        for ch in chars_for_radical(r):
            idx_by_radical[r].append(len(atoms))
            atoms.append(_atom_for_char(ch, r))
            radical_of_idx.append(r)

    N = len(atoms)
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.ones(batch_size, N)
    topics = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    c0, t0 = _pack(atoms, d_x, n_dim)
    for b in range(batch_size):
        ns = 0.0 if b == 0 else noise_scale
        c = c0.clone()
        if ns > 0:
            c = c + ns * torch.randn(N, d_x)
            c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        content[b] = c
        tracer[b] = t0
        r = rng.choice(radicals)
        qi, pi = rng.sample(idx_by_radical[r], 2)
        q_idx[b], p_idx[b] = qi, pi

    return RecallBatch(
        content=content, tracer=tracer, depths=depths, states=states,
        mask=mask, query_idx=q_idx, pos_idx=p_idx, topics=topics,
    )


def build_holdout_eval_batch(
    *,
    d_x: int,
    n_dim: int,
    batch_size: int,
    seed: int,
    n_distractor_per_radical: int = 8,
    noise_scale: float = 0.05,
) -> RecallBatch:
    """Query/pos zawsze z HOLDOUT_RADICAL (nigdy widzianego w train); kandydaci =
    cały holdout + dystraktory z radykałów treningowych (DESI-styl: generalizacja
    na nowe obiekty w obecności znanych, nie recall w izolowanym grafie)."""
    rng = random.Random(seed)
    holdout_chars = chars_for_radical(HOLDOUT_RADICAL)
    atoms = [_atom_for_char(ch, HOLDOUT_RADICAL) for ch in holdout_chars]
    n_holdout = len(atoms)

    distractor_ids: list[int] = []
    for r in TRAIN_RADICALS:
        pool = chars_for_radical(r)
        pick = rng.sample(pool, min(n_distractor_per_radical, len(pool)))
        for ch in pick:
            atoms.append(_atom_for_char(ch, r))

    N = len(atoms)
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.ones(batch_size, N)
    topics = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    c0, t0 = _pack(atoms, d_x, n_dim)
    for b in range(batch_size):
        ns = 0.0 if b == 0 else noise_scale
        c = c0.clone()
        if ns > 0:
            c = c + ns * torch.randn(N, d_x)
            c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        content[b] = c
        tracer[b] = t0
        qi, pi = rng.sample(range(n_holdout), 2)  # oba z holdout — nigdy z dystraktorów
        q_idx[b], p_idx[b] = qi, pi

    return RecallBatch(
        content=content, tracer=tracer, depths=depths, states=states,
        mask=mask, query_idx=q_idx, pos_idx=p_idx, topics=topics,
    ), {
        "n_holdout": n_holdout,
        "n_distractors": N - n_holdout,
        "holdout_ids": [a.atom_id for a in atoms[:n_holdout]],
        "distractor_ids": [a.atom_id for a in atoms[n_holdout:]],
    }


def write_manifest() -> dict:
    DUMP.mkdir(parents=True, exist_ok=True)
    train_ids = {
        f"{r}/{ch}" for r in TRAIN_RADICALS for ch in chars_for_radical(r)
    }
    holdout_ids = {f"{HOLDOUT_RADICAL}/{ch}" for ch in chars_for_radical(HOLDOUT_RADICAL)}
    assert not (train_ids & holdout_ids), "przeciek train/holdout"
    man = {
        "name": "from_holo_v2",
        "method": "holo_phi_v2_unihan",
        "tracer_source": "unihan_stroke_count",
        "split": "graf-per-radykal (opcja A, P5 §0)",
        "holdout_radical": HOLDOUT_RADICAL,
        "train_radicals": list(TRAIN_RADICALS),
        "n_train_ids": len(train_ids),
        "n_holdout_ids": len(holdout_ids),
        "train_ids": sorted(train_ids),
        "holdout_ids": sorted(holdout_ids),
    }
    (DUMP / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return man


if __name__ == "__main__":
    m = write_manifest()
    print(f"OK {DUMP}/manifest.json holdout_radical={m['holdout_radical']} "
          f"train={m['n_train_ids']} holdout={m['n_holdout_ids']}")
