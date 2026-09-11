"""Rozpoznawanie skryptu znaku (Unicode ranges)."""

from __future__ import annotations

import re

_HIRA_RE = re.compile(r"[\u3040-\u309F]")
_KATA_RE = re.compile(r"[\u30A0-\u30FF]")
_KANJI_RE = re.compile(r"[\u4E00-\u9FFF]")
_LAT_RE = re.compile(r"[A-Za-z0-9À-ÿ]")

# Rozszerzalny rejestr — kolejne alfabety (cyrylica, arabski, …) dokładać tu.
SCRIPT_IDS = ("lat", "hira", "kata", "kanji", "sym")


def detect_script_char(ch: str) -> str:
    if not ch:
        return "sym"
    if _HIRA_RE.match(ch):
        return "hira"
    if _KATA_RE.match(ch):
        return "kata"
    if _KANJI_RE.match(ch):
        return "kanji"
    if _LAT_RE.match(ch):
        return "lat"
    return "sym"


def script_prefix(script: str, ch: str | None, idx: int) -> tuple[str, str]:
    """Zwraca (id_script, S). Znak z tekstu dyktuje skrypt glifu gdy podany."""
    if ch is None or ch == "":
        sc = "sym" if script in ("auto", "") else script
        return sc, f"{sc}:#{idx}"
    sc = detect_script_char(ch)
    return sc, f"{sc}:{ch}"
