"""Determinizm Solar v2 — obowiązkowy przed sweepem seedów (lekcja P5 §2.5)."""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

from bench_solar_v2 import _most_ctor, _sm_ctor, train_eval_solar


def test_same_seed_bitwise_reproducible() -> None:
    for ctor in (_most_ctor, _sm_ctor):
        m1 = train_eval_solar(ctor, seed=0)
        m2 = train_eval_solar(ctor, seed=0)
        assert m1["r@1"] == m2["r@1"], (ctor.__name__, m1, m2)
        assert m1["mrr"] == m2["mrr"], (ctor.__name__, m1, m2)


if __name__ == "__main__":
    test_same_seed_bitwise_reproducible()
    print("OK test_solar_v2_determinism")
