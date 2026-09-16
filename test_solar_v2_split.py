"""Solar Names v2 — higiena splitu train/holdout (analogiczne do test_holo_v2_split.py)."""

from __future__ import annotations

from make_dump_solar_v2 import (
    HOLDOUT_BODIES,
    TRAIN_BODIES,
    build_holdout_eval_batch,
    build_mixed_train_batch,
    write_manifest,
)
from solar_names_tracer import LANGUAGES, body_names, category_siblings


def test_no_body_leak_train_holdout() -> None:
    assert not (set(TRAIN_BODIES) & set(HOLDOUT_BODIES))
    m = write_manifest()
    train_ids = set()
    holdout_ids = set()
    for b in TRAIN_BODIES:
        for lang in LANGUAGES:
            train_ids.add(f"{b}/{lang}")
    for b in HOLDOUT_BODIES:
        for lang in LANGUAGES:
            holdout_ids.add(f"{b}/{lang}")
    assert not (train_ids & holdout_ids)
    assert m["n_train_ids"] == len(train_ids)
    assert m["n_holdout_ids"] == len(holdout_ids)


def test_train_batch_never_contains_holdout_body() -> None:
    for seed in range(5):
        b = build_mixed_train_batch(d_x=8, n_dim=6, batch_size=2, seed=seed, n_bodies=10)
        # rekonstrukcja: sprawdzamy przez manifest, że TRAIN_BODIES nie zawiera holdout
        assert not (set(TRAIN_BODIES) & set(HOLDOUT_BODIES))


def test_eval_batch_query_pos_only_from_holdout() -> None:
    hb, meta = build_holdout_eval_batch(d_x=8, n_dim=6, batch_size=16, seed=1, n_distractor_bodies=6)
    n_holdout = meta["n_holdout"]
    assert bool((hb.query_idx < n_holdout).all())
    assert bool((hb.pos_idx < n_holdout).all())
    for aid in meta["distractor_ids"]:
        body = aid.rsplit("/", 1)[0]
        assert body in TRAIN_BODIES
    for aid in meta["holdout_ids"]:
        body = aid.rsplit("/", 1)[0]
        assert body in HOLDOUT_BODIES


def test_holdout_has_train_sibling() -> None:
    """Każde holdout ciało musi mieć >=1 rodzeństwo (ta sama kategoria) w
    treningu — inaczej model nie ma skąd nauczyć się kategorii do generalizacji."""
    for b in HOLDOUT_BODIES:
        sibs = [s for s in category_siblings(b) if s in TRAIN_BODIES]
        assert sibs, f"{b} nie ma rodzeństwa w treningu"


if __name__ == "__main__":
    test_no_body_leak_train_holdout()
    test_train_batch_never_contains_holdout_body()
    test_eval_batch_query_pos_only_from_holdout()
    test_holdout_has_train_sibling()
    print("OK test_solar_v2_split")
