"""Polityka jądra semantycznego per skrypt."""

from __future__ import annotations

from typing import Literal

SemanticUnit = Literal["word", "char"]

# Alfabetyczne → semantyka na słowie; logosylabiczne/morfemiczne → na znaku.
_WORD_SCRIPTS = frozenset({"lat", "sym"})
_CHAR_SCRIPTS = frozenset({"hira", "kata", "kanji"})


def semantic_unit(script: str) -> SemanticUnit:
    """
    lat/sym → word (litera sama nie niesie leksemu).
    hira/kata/kanji → char (znak często = jednostka znaczenia).
    """
    sc = (script or "sym").lower()
    if sc in _CHAR_SCRIPTS:
        return "char"
    if sc in _WORD_SCRIPTS:
        return "word"
    # nieznany skrypt: ostrożnie jak alfabet (wymaga słowa), nie jak kanji
    return "word"


def is_word_nucleus_script(script: str) -> bool:
    return semantic_unit(script) == "word"


def is_char_nucleus_script(script: str) -> bool:
    return semantic_unit(script) == "char"
