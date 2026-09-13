"""P5 §2.5 — determinizm OBOWIĄZKOWY przed jakimkolwiek sweepem/gate.

Replikacja P4a pokazała brak bit-reprodukowalności mimo ustawionego seeda
(wielowątkowy CPU matmul ma niedeterministyczną kolejność redukcji fp) —
patrz doswiadczenie/glify (HOLO_WNIOSKI.md P4a). `bench_holo_v2.py` importuje
`torch.set_num_threads(1)` + `use_deterministic_algorithms` na module-load,
więc samo zaimportowanie modułu już wymusza tryb deterministyczny w tym procesie.
"""

from __future__ import annotations

from bench_holo_v2 import _most_ctor, _sm_ctor, train_eval_holo


def test_same_seed_bitwise_reproducible() -> None:
    for ctor in (_most_ctor, _sm_ctor):
        m1 = train_eval_holo(ctor, seed=0)
        m2 = train_eval_holo(ctor, seed=0)
        assert m1["r@1"] == m2["r@1"], (ctor.__name__, m1, m2)
        assert m1["mrr"] == m2["mrr"], (ctor.__name__, m1, m2)


if __name__ == "__main__":
    test_same_seed_bitwise_reproducible()
    print("OK test_holo_v2_determinism")
