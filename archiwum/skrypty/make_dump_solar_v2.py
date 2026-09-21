"""
Solar Names v2 — dump/batch builder. Analogiczny do make_dump_holo_v2.py, ale:
  - korpus 18x większy (33 ciała x 55 języków = 1815 atomów vs 100 w P5),
  - τ = prawdziwa fizyka (log dystansu orbitalnego z Wikidata), nie stroke_count,
  - mieszane grafy treningowe (prawdziwe negatywy cross-body od pierwszego batcha),
  - **content = kategoria/rodzeństwo (jak `topic` w Fazie C), NIE nazwa ciała.**

Dlaczego content=kategoria: audyt Fazy C (jedynego benchu, gdzie Most realnie i
przyczynowo wygrywa) pokazał, że tam content koduje GRUPĘ (topic) współdzieloną
przez kilku kandydatów (nierozróżnialnych samą treścią), a τ (depth) jest
JEDYNYM sygnałem rozstrzygającym, który z nich jest prawdziwym pozytywem.
W P5.1/Solar-v1 content = unikalny hash KAŻDEJ instancji — model nie miał
żadnej wspólnej struktury do przeniesienia na holdout. Tu: content = hash
kategorii rodzeństwa (np. "moon_of_Jupiter" — Io/Europa/Ganymede/Callisto
mają PRAWIE identyczny content) + mały szum; τ = dokładny log-dystans
orbitalny (realny, różny dla każdego ciała) — jedyny sposób odróżnienia
KTÓRE konkretne ciało w danej kategorii. Holdout dobrany tak, by każde
trzymane ciało miało ≥1 "rodzeństwo" (ta sama kategoria) w treningu — model
mógł nauczyć się kategorii z rodzeństwa, a musi użyć τ, by rozpoznać
KONKRETNIE nowego członka.

Family/task = ciało (np. "Mars"); instancja = nazwa w jednym z 55 języków.
Holdout = 7 ciał NIGDY nieobecnych w żadnym forward passie treningowym.
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
from phi_schema import PhiAtom, PhiState, atom_tracer_vec, hash_embed
from recall_data import RecallBatch
from solar_names_tracer import (
    BODY_NAMES,
    LANGUAGES,
    body_category,
    body_log_distance,
    body_log_distance_ref,
    body_names,
    category_siblings,
)

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "dumps" / "from_solar_v2"

# Każde holdout ciało ma >=1 rodzeństwo (ta sama kategoria) w treningu —
# zweryfikowane w test_solar_v2_split.py::test_holdout_has_train_sibling.
HOLDOUT_BODIES: tuple[str, ...] = (
    "Neptune",   # planet (7 innych planet zostaje)
    "Ceres",     # dwarf_planet (Pluto zostaje)
    "Deimos",    # moon_of_Mars (Phobos zostaje)
    "Europa",    # moon_of_Jupiter (Io/Ganymede/Callisto zostają)
    "Mimas",     # moon_of_Saturn (6 innych zostaje)
    "Umbriel",   # moon_of_Uranus (4 inne zostają)
    "Nereid",    # moon_of_Neptune (Triton zostaje)
)
TRAIN_BODIES: tuple[str, ...] = tuple(b for b in BODY_NAMES if b not in HOLDOUT_BODIES)

_REF = body_log_distance_ref()


_CAT_PROTO_CACHE: dict[tuple[str, int], torch.Tensor] = {}
_LANG_PROTO_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _cat_proto(category: str, d_x: int) -> torch.Tensor:
    key = (category, d_x)
    if key not in _CAT_PROTO_CACHE:
        _CAT_PROTO_CACHE[key] = hash_embed(f"category/{category}", d_x, seed=7)
    return _CAT_PROTO_CACHE[key]


def _lang_proto(lang: str, d_x: int) -> torch.Tensor:
    key = (lang, d_x)
    if key not in _LANG_PROTO_CACHE:
        _LANG_PROTO_CACHE[key] = hash_embed(f"lang/{lang}", d_x, seed=11)
    return _LANG_PROTO_CACHE[key]


def _atom_for_name(body: str, lang: str, name: str) -> PhiAtom:
    t = energy_to_T(body_log_distance(body), ref=_REF)
    return PhiAtom(
        atom_id=f"{body}/{lang}", S=name, E=name, T=t, state=PhiState.HOT,
        bubble_id=f"body/{body}", depth=0, payload=name, mazur_tracer=None,
    )


def _pack(
    entries: list[tuple[str, str, PhiAtom]], d_x: int, n_dim: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """entries: (body, lang, atom) — content = 0.75*hash(kategoria) +
    0.15*hash(język) + 0.10*szum (wzór `recall_data._topic_content`, Faza C):
    kategoria dominuje (nierozróżnialność w obrębie grupy rodzeństwa), τ z
    tracera jest tym, co ma rozstrzygać KTÓRE konkretne ciało. Proto-wektory
    kategorii/języka cache'owane (nie liczone od nowa per wiersz/epoka)."""
    rng = torch.Generator().manual_seed(seed)
    t_max = max((a.T for _, _, a in entries), default=1.0) or 1.0
    content = torch.zeros(len(entries), d_x)
    tracer = torch.zeros(len(entries), n_dim)
    noise = torch.randn(len(entries), d_x, generator=rng)
    for i, (body, lang, a) in enumerate(entries):
        aa = PhiAtom(
            atom_id=a.atom_id, S=a.S, E=a.E, T=a.T / t_max, state=a.state,
            bubble_id=a.bubble_id, depth=a.depth, payload=a.payload,
            mazur_tracer=a.mazur_tracer,
        )
        v = 0.75 * _cat_proto(body_category(body), d_x) + 0.15 * _lang_proto(lang, d_x) + 0.10 * noise[i]
        content[i] = v / v.norm().clamp_min(1e-6)
        tracer[i] = atom_tracer_vec(aa, n_dim)
    return content, tracer


def build_mixed_train_batch(
    *,
    d_x: int,
    n_dim: int,
    batch_size: int,
    seed: int,
    n_bodies: int = 8,
    noise_scale: float = 0.05,
) -> RecallBatch:
    """Losowy podzbiór `n_bodies` ciał treningowych razem w jednym grafie —
    pozytyw = to samo ciało (inny język), kandydaci = cały graf. Content
    (kategoria) robi kilku kandydatów z tej samej grupy nierozróżnialnymi —
    τ (dokładny dystans) jest tym, co ma rozstrzygać."""
    rng = random.Random(seed)
    bodies = rng.sample(TRAIN_BODIES, min(n_bodies, len(TRAIN_BODIES)))
    entries: list[tuple[str, str, PhiAtom]] = []
    idx_by_body: dict[str, list[int]] = {b: [] for b in bodies}
    for b in bodies:
        for lang, name in body_names(b).items():
            idx_by_body[b].append(len(entries))
            entries.append((b, lang, _atom_for_name(b, lang, name)))

    N = len(entries)
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.ones(batch_size, N)
    topics = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    for b_i in range(batch_size):
        c, t0 = _pack(entries, d_x, n_dim, seed=seed * 100003 + b_i)
        content[b_i] = c
        tracer[b_i] = t0
        body = rng.choice(bodies)
        qi, pi = rng.sample(idx_by_body[body], 2)
        q_idx[b_i], p_idx[b_i] = qi, pi

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
    n_distractor_bodies: int = 8,
    noise_scale: float = 0.05,
) -> tuple[RecallBatch, dict]:
    """Query/pos zawsze z ciała HOLDOUT; kandydaci = wszystkie 7 ciał holdout +
    dystraktory z ciał treningowych (w tym ich rodzeństwo kategorii — hard negs)."""
    rng = random.Random(seed)
    entries: list[tuple[str, str, PhiAtom]] = []
    idx_by_body: dict[str, list[int]] = {}
    for b in HOLDOUT_BODIES:
        idx_by_body[b] = []
        for lang, name in body_names(b).items():
            idx_by_body[b].append(len(entries))
            entries.append((b, lang, _atom_for_name(b, lang, name)))
    n_holdout = len(entries)

    # dystraktory: rodzeństwo holdout ciał (twarde negatywy, ta sama kategoria),
    # ograniczone do n_distractor_bodies (priorytet), żeby N nie eksplodowało
    # (niektóre kategorie, np. "planet", mają dużo rodzeństwa w treningu).
    siblings_in_train = sorted({
        s for b in HOLDOUT_BODIES for s in category_siblings(b) if s in TRAIN_BODIES
    })
    rng.shuffle(siblings_in_train)
    if len(siblings_in_train) > n_distractor_bodies:
        distractor_bodies = siblings_in_train[:n_distractor_bodies]
    else:
        other_train = [b for b in TRAIN_BODIES if b not in siblings_in_train]
        n_extra = n_distractor_bodies - len(siblings_in_train)
        distractor_bodies = siblings_in_train + rng.sample(other_train, min(n_extra, len(other_train)))

    for b in distractor_bodies:
        for lang, name in body_names(b).items():
            entries.append((b, lang, _atom_for_name(b, lang, name)))

    N = len(entries)
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    depths = torch.zeros(batch_size, N)
    states = torch.zeros(batch_size, N, dtype=torch.long)
    mask = torch.ones(batch_size, N)
    topics = torch.zeros(batch_size, N, dtype=torch.long)
    q_idx = torch.zeros(batch_size, dtype=torch.long)
    p_idx = torch.zeros(batch_size, dtype=torch.long)

    for b_i in range(batch_size):
        c, t0 = _pack(entries, d_x, n_dim, seed=seed * 100003 + b_i)
        content[b_i] = c
        tracer[b_i] = t0
        body = rng.choice(HOLDOUT_BODIES)
        qi, pi = rng.sample(idx_by_body[body], 2)
        q_idx[b_i], p_idx[b_i] = qi, pi

    meta = {
        "n_holdout": n_holdout, "n_distractors": N - n_holdout,
        "distractor_bodies": distractor_bodies,
        "holdout_ids": [f"{b}/{l}" for b, l, _ in entries[:n_holdout]],
        "distractor_ids": [f"{b}/{l}" for b, l, _ in entries[n_holdout:]],
    }
    return RecallBatch(
        content=content, tracer=tracer, depths=depths, states=states,
        mask=mask, query_idx=q_idx, pos_idx=p_idx, topics=topics,
    ), meta


def write_manifest() -> dict:
    DUMP.mkdir(parents=True, exist_ok=True)
    train_ids = {f"{b}/{lang}" for b in TRAIN_BODIES for lang in LANGUAGES}
    holdout_ids = {f"{b}/{lang}" for b in HOLDOUT_BODIES for lang in LANGUAGES}
    assert not (train_ids & holdout_ids), "przeciek train/holdout"
    man = {
        "name": "from_solar_v2", "method": "solar_names_v2_category_content",
        "tracer_source": "orbital_log_distance",
        "content_source": "category_hash (topic-style, Faza C)",
        "holdout_bodies": list(HOLDOUT_BODIES), "train_bodies": list(TRAIN_BODIES),
        "n_languages": len(LANGUAGES),
        "n_train_ids": len(train_ids), "n_holdout_ids": len(holdout_ids),
    }
    (DUMP / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return man


if __name__ == "__main__":
    m = write_manifest()
    print(f"OK holdout={m['holdout_bodies']} train_n={len(TRAIN_BODIES)} "
          f"langs={m['n_languages']} train_atoms={m['n_train_ids']} holdout_atoms={m['n_holdout_ids']}")
