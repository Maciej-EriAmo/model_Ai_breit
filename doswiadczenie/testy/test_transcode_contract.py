"""
Kontrakt transkodowania alfabetów (glyph_unit_v2) + framing vs Softmax-twin.

Softmax --baselines = Softmax-twin on Φ (no ΔE), not train-on-classical_tokens.
"""

from __future__ import annotations

import _labroot  # noqa: F401  — root labu na sys.path

from transcode import (
    CONTRACT_VERSION,
    build_from_text,
    classical_token_sequence,
    detect_script_char,
    semantic_unit,
)
from transcode.phi_build import BuildResult, state_for_T


def test_detect_scripts() -> None:
    assert detect_script_char("M") == "lat"
    assert detect_script_char("こ") == "hira"
    assert detect_script_char("ア") == "kata"
    assert detect_script_char("東") == "kanji"


def test_semantic_policy() -> None:
    assert semantic_unit("lat") == "word"
    assert semantic_unit("hira") == "char"
    assert semantic_unit("kanji") == "char"


def test_latin_word_nuclei() -> None:
    r = build_from_text("MOST BRIDGE")
    words = r.word_nuclei
    letters = r.letters
    assert len(words) == 2, words
    assert {w["S"] for w in words} == {"lat:MOST", "lat:BRIDGE"}
    assert all(w["depth"] == 1 and w["unit"] == "word" for w in words)
    assert all(L["depth"] == 2 and L["unit"] == "letter" for L in letters)
    assert len(letters) == len("MOST") + len("BRIDGE")
    parents = [
        e for e in r.edges if e.get("name") == "letter_of" and e["kind"] == "parent"
    ]
    assert len(parents) == len(letters)
    seq = classical_token_sequence(r)
    assert seq[0] == "lat:M"
    assert "lat:B" in seq
    assert len(seq) == len(letters)


def test_hira_char_nuclei() -> None:
    r = build_from_text("こんにちは")
    assert r.word_nuclei == []
    chars = r.char_nuclei
    assert len(chars) == 5, [c["S"] for c in chars]
    assert all(c["depth"] == 2 and c["unit"] == "char" for c in chars)
    assert chars[0]["S"] == "hira:こ"
    seq = classical_token_sequence(r)
    assert seq == [c["S"] for c in chars]


def test_mixed_lat_hira_spaced() -> None:
    r = build_from_text("MOST こんにちは")
    assert len(r.word_nuclei) == 1
    assert r.word_nuclei[0]["S"] == "lat:MOST"
    assert len(r.char_nuclei) == 5


def test_glued_lat_hira() -> None:
    """Audyt bug1: MOSTこんにちは → jedno słowo MOST + 5 hira, nie lat:M…"""
    r = build_from_text("MOSTこんにちは")
    assert len(r.word_nuclei) == 1, [w["S"] for w in r.word_nuclei]
    assert r.word_nuclei[0]["S"] == "lat:MOST"
    assert len(r.letters) == 4
    assert len(r.char_nuclei) == 5
    assert [c["S"] for c in r.char_nuclei][0] == "hira:こ"


def test_glued_lat_kanji() -> None:
    r = build_from_text("MOST東")
    assert len(r.word_nuclei) == 1
    assert r.word_nuclei[0]["S"] == "lat:MOST"
    assert len(r.char_nuclei) == 1
    assert r.char_nuclei[0]["S"] == "kanji:東"


def test_classical_excludes_archive() -> None:
    r = build_from_text("AB")
    classical = classical_token_sequence(r)
    # mutate copy like make_dump — classical from BuildResult must ignore archive
    r.atoms.append(
        {
            "id": "lat/archive/X",
            "S": "lat:X",
            "E": "archive",
            "T": 0.08,
            "state": state_for_T(0.08),
            "depth": 2,
            "unit": "letter",
            "topic": "archive",
        }
    )
    classical2 = classical_token_sequence(r)
    assert "lat:X" not in classical2
    assert classical2 == classical
    assert len(classical2) == 2


def test_char_focus() -> None:
    r = build_from_text("あいう", char_focus=2)
    chars = r.char_nuclei
    assert chars[2]["T"] >= chars[0]["T"]
    assert chars[2]["state"] == "HOT"


def test_T_not_from_xy_fields() -> None:
    r = build_from_text("AB")
    for a in r.letters:
        assert isinstance(a["T"], float)
        assert 0.0 <= float(a["T"]) <= 1.0


def test_contract_version() -> None:
    r = build_from_text("Hi")
    assert r.contract == CONTRACT_VERSION == "glyph_unit_v2"


def test_word_id_uses_script_prefix() -> None:
    r = build_from_text("Hi")
    assert r.word_nuclei[0]["id"].startswith("lat/")


def test_photo_boxes_path_has_word_unit() -> None:
    """glyph_transcribe boxes_to_phi with labels → word nuclei."""
    from glyph_transcribe import GlyphBox, boxes_to_phi

    # synthetic boxes for M O S T
    boxes = [
        GlyphBox(1, 0, i * 10, 10, i * 10 + 8, 20, 0.5) for i in range(4)
    ]
    atoms, _edges = boxes_to_phi(
        [boxes],
        script="lat",
        text="MOST",
        word_id="demo",
        focus_index=0,
        source_name="t.png",
    )
    words = [a for a in atoms if a.get("unit") == "word"]
    letters = [a for a in atoms if a.get("unit") == "letter"]
    assert len(words) == 1
    assert words[0]["S"] == "lat:MOST"
    assert len(letters) == 4
    assert all("bbox=" in a["E"] for a in letters)


if __name__ == "__main__":
    test_detect_scripts()
    test_semantic_policy()
    test_latin_word_nuclei()
    test_hira_char_nuclei()
    test_mixed_lat_hira_spaced()
    test_glued_lat_hira()
    test_glued_lat_kanji()
    test_classical_excludes_archive()
    test_char_focus()
    test_T_not_from_xy_fields()
    test_contract_version()
    test_word_id_uses_script_prefix()
    test_photo_boxes_path_has_word_unit()
    print("OK test_transcode_contract")
