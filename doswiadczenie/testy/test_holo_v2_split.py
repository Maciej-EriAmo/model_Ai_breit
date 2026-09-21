"""P5.1 — higiena splitu train/holdout (wzorem audytu DESI, opcja A). Gołe assert."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

from make_dump_holo_v2 import (
    HOLDOUT_RADICAL,
    TRAIN_RADICALS,
    build_holdout_eval_batch,
    build_train_batch,
    write_manifest,
)
from unihan_tracer import chars_for_radical


def test_no_char_leak_train_holdout() -> None:
    m = write_manifest()
    train_ids = set(m["train_ids"])
    holdout_ids = set(m["holdout_ids"])
    assert not (train_ids & holdout_ids)
    assert HOLDOUT_RADICAL not in TRAIN_RADICALS


def test_holdout_radical_absent_from_any_train_sequence() -> None:
    holdout_chars = set(chars_for_radical(HOLDOUT_RADICAL))
    for r in TRAIN_RADICALS:
        b = build_train_batch(r, d_x=8, n_dim=6, batch_size=2, seed=0)
        # train batch dla radykału r zawiera WYŁĄCZNIE znaki radykału r
        assert set(chars_for_radical(r)).isdisjoint(holdout_chars)
        assert b.content.shape[1] == len(chars_for_radical(r))


def test_manifest_records_holdout_radical() -> None:
    m = write_manifest()
    assert m["holdout_radical"] == HOLDOUT_RADICAL
    assert set(m["train_radicals"]) == set(TRAIN_RADICALS)
    assert len(m["holdout_ids"]) >= 10


def test_eval_batch_query_pos_only_from_holdout() -> None:
    hb, meta = build_holdout_eval_batch(d_x=8, n_dim=6, batch_size=8, seed=1)
    n_holdout = meta["n_holdout"]
    assert bool((hb.query_idx < n_holdout).all())
    assert bool((hb.pos_idx < n_holdout).all())
    # dystraktory faktycznie z radykałów treningowych, nie holdout
    for aid in meta["distractor_ids"]:
        radical = int(aid.split("/", 1)[0])
        assert radical in TRAIN_RADICALS


if __name__ == "__main__":
    test_no_char_leak_train_holdout()
    test_holdout_radical_absent_from_any_train_sequence()
    test_manifest_records_holdout_radical()
    test_eval_batch_query_pos_only_from_holdout()
    print("OK test_holo_v2_split")
