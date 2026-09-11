"""Transkodowanie alfabetów → Φ (jeden Most, wiele pism)."""

from .phi_build import (
    CONTRACT_VERSION,
    BuildResult,
    TextSpan,
    build_from_spans,
    build_from_text,
    classical_token_sequence,
    parse_text_spans,
)
from .scripts import SCRIPT_IDS, detect_script_char, script_prefix
from .units import SemanticUnit, is_char_nucleus_script, is_word_nucleus_script, semantic_unit

__all__ = [
    "CONTRACT_VERSION",
    "BuildResult",
    "TextSpan",
    "SCRIPT_IDS",
    "SemanticUnit",
    "build_from_spans",
    "build_from_text",
    "classical_token_sequence",
    "detect_script_char",
    "is_char_nucleus_script",
    "is_word_nucleus_script",
    "parse_text_spans",
    "script_prefix",
    "semantic_unit",
]
