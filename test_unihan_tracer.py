"""P5.0 — audyt niezależności τ (unihan) od hasha etykiety. Styl repo: gołe assert."""

from __future__ import annotations

import hashlib

from unihan_tracer import (
    RADICALS,
    all_chars,
    chars_for_radical,
    load_unihan,
    radical_id,
    stroke_count,
)


def test_stroke_count_known_chars() -> None:
    # wartości zweryfikowane wprost z Unihan_IRGSources.txt (kTotalStrokes), 2026-09-13
    assert stroke_count("女") == 3
    assert stroke_count("木") == 4
    assert stroke_count("水") == 4
    assert stroke_count("火") == 4
    assert stroke_count("人") == 2
    assert stroke_count("好") == 6
    assert stroke_count("燃") == 16


def test_radical_groups_match_known_family() -> None:
    for rad in RADICALS:
        chars = chars_for_radical(rad)
        assert len(chars) >= 10, f"radical {rad} ma za mało znaków: {len(chars)}"
        assert all(radical_id(ch) == rad for ch in chars)
    # znaki z różnych radykałów faktycznie się różnią
    assert radical_id("女") != radical_id("木")
    assert radical_id("水") != radical_id("火")


def test_stroke_count_not_hash_derived() -> None:
    # dwa znaki o identycznej długości etykiety/stringa (1 znak każdy) ale
    # różnej liczbie kresek — stroke_count faktycznie się różni (nie "tylko inny hash")
    pairs = [("女", "燃"), ("人", "海"), ("木", "婚")]
    for a, b in pairs:
        assert len(a) == len(b) == 1
        assert stroke_count(a) != stroke_count(b)


def test_no_trivial_correlation_with_label_hash() -> None:
    chars = all_chars()
    xs = [stroke_count(ch) for ch in chars]
    ys = [int(hashlib.sha256(ch.encode("utf-8")).hexdigest(), 16) % 1000 for ch in chars]

    n = len(chars)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    pearson = cov / ((vx * vy) ** 0.5)

    assert abs(pearson) < 0.3, f"stroke_count koreluje z hash(ch): r={pearson:.3f}"


def test_load_unihan_subset() -> None:
    subset = {"女", "木", "nieznany_znak"}
    out = load_unihan(subset)
    assert set(out.keys()) == {"女", "木"}
    assert out["女"] == {"strokes": 3, "radical": 38}
    assert out["木"] == {"strokes": 4, "radical": 75}


if __name__ == "__main__":
    test_stroke_count_known_chars()
    test_radical_groups_match_known_family()
    test_stroke_count_not_hash_derived()
    test_no_trivial_correlation_with_label_hash()
    test_load_unihan_subset()
    print("OK test_unihan_tracer")
