"""
Solar Names tracer — τ z prawdziwej fizyki (odległość orbitalna), content = nazwa
ciała w danym języku. Analogiczne do `unihan_tracer.py`, ale ~18x większy korpus.

Źródło: Wikidata (unicode.org-style: pobrane 2026-09-13, nie zgadywane).
  - Nazwy: etykiety (labels) encji Wikidata dla 33 ciał Układu Słonecznego,
    ograniczone do 55 języków obecnych u WSZYSTKICH 33 ciał (przecięcie zbiorów).
  - τ: `P2233` (semi-major axis) — dystans od ciała centralnego (Słońce dla
    planet/Ceres/Plutona, planeta dla księżyców, Ziemia dla Księżyca).
    Jednostki znormalizowane do km (część encji miała AU — przeliczone).
  - Słońce nie ma własnego dystansu (jest centrum) — `distance_km=None`.

Dane: `solar_names_data.json` (bundlowane, nie live-fetch w runtime).

33 ciała = 33 "rodziny" (analog radykału w P5); 55 języków = 55 "znaków"/instancji
per rodzina (analog liczby znaków per radykał). content = nazwa w danym języku
(hash), τ = f(log(distance_km)) — fizyczna, niezależna od stringa nazwy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "solar_names_data.json"
_DATA = json.loads(_DATA_PATH.read_text(encoding="utf-8"))

LANGUAGES: tuple[str, ...] = tuple(_DATA["languages"])
BODIES: dict[str, dict] = _DATA["bodies"]
BODY_NAMES: tuple[str, ...] = tuple(BODIES.keys())

# Słońce nie ma dystansu (centrum) — dostaje log-dystans = 0 (mniejszy niż
# jakikolwiek prawdziwy dystans, Merkury i tak >> 0 po log), żeby energy_to_T
# dało mu sensowną, najniższą wartość na wspólnej skali, bez specjalnych wyjątków
# w reszcie pipeline'u.
_LOG_DIST = {
    name: (math.log10(info["distance_km"]) if info["distance_km"] else 0.0)
    for name, info in BODIES.items()
}
_LOG_DIST_REF = max(_LOG_DIST.values())


def body_names(body: str) -> dict[str, str]:
    """{język: nazwa} dla danego ciała, tylko języki z pełnym pokryciem (55)."""
    return dict(BODIES[body]["names"])


def body_parent(body: str) -> str | None:
    return BODIES[body]["parent"]


def body_log_distance(body: str) -> float:
    """log10(dystans_km) od ciała centralnego; Słońce = 0 (patrz moduł docstring)."""
    return _LOG_DIST[body]


def body_log_distance_ref() -> float:
    return _LOG_DIST_REF


def all_bodies() -> list[str]:
    return list(BODY_NAMES)


_PLANETS = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"}
_DWARFS = {"Pluto", "Ceres"}


def body_category(body: str) -> str:
    """Kategoria dla CONTENT (nie τ!): grupa rodzeństwa, żeby content sam w
    sobie nie identyfikował ciała — tylko jego typ/rodzica (jak `topic` w
    Fazie C: content ⇒ grupa niejednoznaczna, τ ⇒ rozstrzygnięcie w grupie)."""
    if body == "Sun":
        return "sun"
    if body in _PLANETS:
        return "planet"
    if body in _DWARFS:
        return "dwarf_planet"
    parent = body_parent(body)
    return f"moon_of_{parent}"


def category_siblings(body: str) -> list[str]:
    """Inne ciała tej samej kategorii (do sprawdzenia, że holdout ma rodzeństwo w train)."""
    cat = body_category(body)
    return [b for b in BODY_NAMES if b != body and body_category(b) == cat]
