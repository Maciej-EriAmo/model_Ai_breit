"""
Kontrakt transkodowania alfabetów (glyph_unit_v2) + framing vs klasyka.

Klasyczna metoda = tokenizacja liter/znaków + następstwo (Softmax-twin).
Nowa metoda nie może być kaleką: musi mieć porównywalną sekwencję klasyczną
oraz poprawne jądra (lat=słowo, CJK=znak).
"""

from __future__ import annotations

from transcode import (
    CONTRACT_VERSION,
    build_from_text,
    classical_token_sequence,
    detect_script_char,
    semantic_unit,
)


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
    # litery parent → słowo
    parents = [
        e for e in r.edges if e.get("name") == "letter_of" and e["kind"] == "parent"
    ]
    assert len(parents) == len(letters)
    # klasyczna sekwencja = litery w kolejności
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


def test_mixed_lat_hira() -> None:
    r = build_from_text("MOST こんにちは")
    assert len(r.word_nuclei) == 1
    assert r.word_nuclei[0]["S"] == "lat:MOST"
    assert len(r.char_nuclei) == 5
    # brak fałszywego wiązania HOT lat↔hira jako jednego word bag
    assert all(w["unit"] == "word" for w in r.word_nuclei)


def test_T_not_from_xy_fields() -> None:
    r = build_from_text("AB")
    for a in r.letters:
        assert "bbox" not in str(a.get("T"))
        assert 0.0 <= float(a["T"]) <= 1.0
        # E może mieć meta, T jest skalarem
        assert isinstance(a["T"], float)


def test_contract_version() -> None:
    r = build_from_text("Hi")
    assert r.contract == CONTRACT_VERSION == "glyph_unit_v2"


def test_classical_baseline_nonempty() -> None:
    """Stara metoda musi mieć co tokenizować — inaczej porównanie jest puste."""
    r = build_from_text("MOST BRIDGE あいう")
    seq = classical_token_sequence(r)
    assert len(seq) >= len("MOSTBRIDGE") + 3
    # Softmax path = ta sekwencja; Most path = pełne Φ (słowa+litery+T)
    assert any(s.startswith("lat:") for s in seq)
    assert any(s.startswith("hira:") for s in seq)


if __name__ == "__main__":
    test_detect_scripts()
    test_semantic_policy()
    test_latin_word_nuclei()
    test_hira_char_nuclei()
    test_mixed_lat_hira()
    test_T_not_from_xy_fields()
    test_contract_version()
    test_classical_baseline_nonempty()
    print("OK test_transcode_contract")
