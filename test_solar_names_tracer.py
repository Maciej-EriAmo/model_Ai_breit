"""Audyt danych solar_names (real Wikidata) — analogiczne do test_unihan_tracer.py."""

from __future__ import annotations

import hashlib

from solar_names_tracer import (
    BODY_NAMES,
    LANGUAGES,
    all_bodies,
    body_log_distance,
    body_names,
    body_parent,
)


def test_corpus_shape() -> None:
    assert len(BODY_NAMES) == 33
    assert len(LANGUAGES) == 55
    for b in BODY_NAMES:
        names = body_names(b)
        assert len(names) == 55, f"{b}: {len(names)} języków, oczekiwano 55"


def test_known_distances() -> None:
    # zweryfikowane wprost z Wikidata P2233 (semi-major axis), 2026-09-13
    assert abs(body_log_distance("Earth") - 8.175) < 0.01  # ~149.6M km
    assert body_log_distance("Sun") == 0.0
    assert body_log_distance("Pluto") > body_log_distance("Neptune")
    assert body_log_distance("Moon") < body_log_distance("Mercury")


def test_parents() -> None:
    assert body_parent("Moon") == "Earth"
    assert body_parent("Phobos") == "Mars"
    assert body_parent("Sun") is None
    assert body_parent("Earth") == "Sun"


def test_no_trivial_correlation_with_label_hash() -> None:
    # log-dystans nie jest funkcją hasha angielskiej nazwy ciała
    bodies = all_bodies()
    xs = [body_log_distance(b) for b in bodies]
    ys = [int(hashlib.sha256(b.encode("utf-8")).hexdigest(), 16) % 1000 for b in bodies]
    n = len(bodies)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    pearson = cov / ((vx * vy) ** 0.5)
    assert abs(pearson) < 0.3, f"log_distance koreluje z hash(nazwa): r={pearson:.3f}"


if __name__ == "__main__":
    test_corpus_shape()
    test_known_distances()
    test_parents()
    test_no_trivial_correlation_with_label_hash()
    print("OK test_solar_names_tracer")
