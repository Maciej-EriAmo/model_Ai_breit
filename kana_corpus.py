"""
Korpus kana (hiragana/katakana/romaji Hepburn), BEZ KANJI — pierwszy test
CELU PIERWOTNEGO raportu hologram-znaki (sekcja 1): czy ta reprezentacja
faktycznie pomaga Mostowi (nie tylko gołemu retrievalowi) uczyć się
prędzej. Generalizuje Lingua P1 (run_lingua_p1.py: 3 słowa, 11 atomów —
"za mało sygnału", patrz LINGUA_P1.md) na realny, większy korpus: 71
tożsamości dźwiękowych x 3 skrypty = 213 morfemów + 71 złożeń.

71 = 46 gojūon (w tym ん) + 20 dakuten + 5 handakuten — dokładnie ten sam,
zweryfikowany zestaw co w sekcji 10.2 raportu hologram-znaki. Kanji
świadomie wykluczone (jōyō kanji to >1200/2136 znaków — inny rząd
wielkości, inny eksperyment).

WAŻNE (znalezione przez pierwszą wersję tego modułu, PRZED napisaniem
tego komentarza — sanity-check retrievalu przed drogim treningiem
wykrył problem zanim zdążył zepsuć wynik): pierwsza wersja kodowała
tożsamość dźwiękową jako JEDNĄ oś ze WSZYSTKIMI 71 wartościami na
wspólnym slocie. compose()/unbind_position idzie przez bind/bundle/unbind
(HRR), który ma udokumentowany twardy próg pojemności ok. N=10-11
kandydatów na wspólnym slocie (sekcja 9A raportu: N=10 -> r@1=1.0, N=12
-> 0.667, N=32 -> 0.219) — 71 na jednym slocie dawało hira r@1=0.63,
kata=0.13, romaji=0.06 (sprawdzone empirycznie). Naprawa: tożsamość
dźwiękowa ROZBITA na TRZY małe osie — rząd spółgłoskowy (11 wartości),
samogłoska (6), diakrytyk (3) — dokładnie metoda z sekcji 10.2 (płaski
overlay dla tych trzech cech), każda daleko w bezpiecznej strefie
pojemności; oś 'script' (hiragana/katakana/romaji, 3 wartości) różni
3 członków JEDNEGO compose()'owanego słowa. Ta wersja: potwierdzone
empirycznie r@1=1.0 na wszystkich osiach dla obu enkodowań przed
uruchomieniem treningu (patrz kana_p1_sanity_check w tym pliku).
"""

from __future__ import annotations

from karmazyn_lingua import AtomEncoding, FeatureOverlay, LinguaLayer, PhaseRegistry

# (hiragana, katakana, hepburn romaji, rząd spółgłoskowy, samogłoska, diakrytyk)
KANA_TABLE: list[tuple[str, str, str, str, str, str]] = [
    ("あ", "ア", "a", "a", "a", "none"), ("い", "イ", "i", "a", "i", "none"),
    ("う", "ウ", "u", "a", "u", "none"), ("え", "エ", "e", "a", "e", "none"),
    ("お", "オ", "o", "a", "o", "none"),
    ("か", "カ", "ka", "k", "a", "none"), ("き", "キ", "ki", "k", "i", "none"),
    ("く", "ク", "ku", "k", "u", "none"), ("け", "ケ", "ke", "k", "e", "none"),
    ("こ", "コ", "ko", "k", "o", "none"),
    ("さ", "サ", "sa", "s", "a", "none"), ("し", "シ", "shi", "s", "i", "none"),
    ("す", "ス", "su", "s", "u", "none"), ("せ", "セ", "se", "s", "e", "none"),
    ("そ", "ソ", "so", "s", "o", "none"),
    ("た", "タ", "ta", "t", "a", "none"), ("ち", "チ", "chi", "t", "i", "none"),
    ("つ", "ツ", "tsu", "t", "u", "none"), ("て", "テ", "te", "t", "e", "none"),
    ("と", "ト", "to", "t", "o", "none"),
    ("な", "ナ", "na", "n", "a", "none"), ("に", "ニ", "ni", "n", "i", "none"),
    ("ぬ", "ヌ", "nu", "n", "u", "none"), ("ね", "ネ", "ne", "n", "e", "none"),
    ("の", "ノ", "no", "n", "o", "none"),
    ("は", "ハ", "ha", "h", "a", "none"), ("ひ", "ヒ", "hi", "h", "i", "none"),
    ("ふ", "フ", "fu", "h", "u", "none"), ("へ", "ヘ", "he", "h", "e", "none"),
    ("ほ", "ホ", "ho", "h", "o", "none"),
    ("ま", "マ", "ma", "m", "a", "none"), ("み", "ミ", "mi", "m", "i", "none"),
    ("む", "ム", "mu", "m", "u", "none"), ("め", "メ", "me", "m", "e", "none"),
    ("も", "モ", "mo", "m", "o", "none"),
    ("や", "ヤ", "ya", "y", "a", "none"), ("ゆ", "ユ", "yu", "y", "u", "none"),
    ("よ", "ヨ", "yo", "y", "o", "none"),
    ("ら", "ラ", "ra", "r", "a", "none"), ("り", "リ", "ri", "r", "i", "none"),
    ("る", "ル", "ru", "r", "u", "none"), ("れ", "レ", "re", "r", "e", "none"),
    ("ろ", "ロ", "ro", "r", "o", "none"),
    ("わ", "ワ", "wa", "w", "a", "none"), ("を", "ヲ", "wo", "w", "o", "none"),
    ("ん", "ン", "n", "N", "∅", "none"),
    # dakuten (ten sam rząd co niedźwięczny odpowiednik, diakrytyk=dakuten)
    ("が", "ガ", "ga", "k", "a", "dakuten"), ("ぎ", "ギ", "gi", "k", "i", "dakuten"),
    ("ぐ", "グ", "gu", "k", "u", "dakuten"), ("げ", "ゲ", "ge", "k", "e", "dakuten"),
    ("ご", "ゴ", "go", "k", "o", "dakuten"),
    ("ざ", "ザ", "za", "s", "a", "dakuten"), ("じ", "ジ", "ji", "s", "i", "dakuten"),
    ("ず", "ズ", "zu", "s", "u", "dakuten"), ("ぜ", "ゼ", "ze", "s", "e", "dakuten"),
    ("ぞ", "ゾ", "zo", "s", "o", "dakuten"),
    ("だ", "ダ", "da", "t", "a", "dakuten"), ("ぢ", "ヂ", "ji", "t", "i", "dakuten"),
    ("づ", "ヅ", "zu", "t", "u", "dakuten"), ("で", "デ", "de", "t", "e", "dakuten"),
    ("ど", "ド", "do", "t", "o", "dakuten"),
    ("ば", "バ", "ba", "h", "a", "dakuten"), ("び", "ビ", "bi", "h", "i", "dakuten"),
    ("ぶ", "ブ", "bu", "h", "u", "dakuten"), ("べ", "ベ", "be", "h", "e", "dakuten"),
    ("ぼ", "ボ", "bo", "h", "o", "dakuten"),
    # handakuten (rząd h, diakrytyk=handakuten)
    ("ぱ", "パ", "pa", "h", "a", "handakuten"), ("ぴ", "ピ", "pi", "h", "i", "handakuten"),
    ("ぷ", "プ", "pu", "h", "u", "handakuten"), ("ぺ", "ペ", "pe", "h", "e", "handakuten"),
    ("ぽ", "ポ", "po", "h", "o", "handakuten"),
]

assert len(KANA_TABLE) == 71, f"oczekiwano 71 (46 gojūon + 20 dakuten + 5 handakuten), jest {len(KANA_TABLE)}"
assert len({row for *_, row, _v, _d in KANA_TABLE}) == 11, "rząd spółgłoskowy powinien mieć 11 wartości"
assert len({v for *_, v, _d in KANA_TABLE}) == 6, "samogłoska powinna mieć 6 wartości"
assert len({d for *_, d in KANA_TABLE}) == 3, "diakrytyk powinien mieć 3 wartości"

ROW_SLOT, VOWEL_SLOT, DIACRITIC_SLOT, SCRIPT_SLOT = 0, 1, 2, 3


def przyklad_kana_korpus(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM,
) -> tuple["LinguaLayer", list]:
    """71 tożsamości dźwiękowych x 3 skrypty jako compose([hira, kata,
    romaji]) na realnym Store — generalizacja przyklad_pisac_rodzina:
    'rdzeń współdzielony' -> tu RZĄD+SAMOGŁOSKA+DIAKRYTYK (3 małe osie,
    współdzielone między 3 członkami tego samego słowa); 'prefiks na
    wspólnym slocie z fazą' -> tu SCRIPT (hiragana/katakana/romaji) różni
    3 członków od siebie.

    SYMBOLIC_ONEHOT jest tu NIEDOSTĘPNY: onehot adresuje wymiar
    BEZPOŚREDNIO (każda odrębna wartość = własny dedykowany slot spośród
    FEATURE_DIM=64); 11+6+3+3=23 dedykowanych wartości by się zmieściło,
    ale onehot i tak nie ma tu nic do udowodnienia (nie używa fazy) —
    porównanie w tym eksperymencie to HOLOGRAM vs SYMBOLIC, centralne
    pytanie sekcji 5 raportu (czy faza pomaga), teraz na 71 prawdziwych
    tożsamościach zamiast 3."""
    import karmazyn_substrate as sub

    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        raise NotImplementedError(
            "SYMBOLIC_ONEHOT nieobsługiwany w przyklad_kana_korpus — onehot nie "
            "używa fazy, więc nie ma tu nic do zmierzenia (pytanie to HOLOGRAM "
            "vs SYMBOLIC); użyj jednego z tych dwóch."
        )

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)

    row_reg = PhaseRegistry(capacity=11)
    vowel_reg = PhaseRegistry(capacity=6)
    diac_reg = PhaseRegistry(capacity=3)
    script_reg = PhaseRegistry(capacity=3)

    def _sound_features(row: str, vowel: str, diac: str) -> list[FeatureOverlay]:
        return [
            FeatureOverlay(f"row:{row}", 1.0, row_reg.phase_for(row), ROW_SLOT),
            FeatureOverlay(f"vowel:{vowel}", 0.9, vowel_reg.phase_for(vowel), VOWEL_SLOT),
            FeatureOverlay(f"diac:{diac}", 0.8, diac_reg.phase_for(diac), DIACRITIC_SLOT),
        ]

    slowa: list = []
    for hira, kata, romaji, row, vowel, diac in KANA_TABLE:
        sound_feats = _sound_features(row, vowel, diac)

        hira_m = lingua.make_morpheme(
            f"hira:{hira}",
            sound_feats + [FeatureOverlay("script_hira", 0.7, script_reg.phase_for("hira"), SCRIPT_SLOT)],
            encoding,
        )
        kata_m = lingua.make_morpheme(
            f"kata:{kata}",
            sound_feats + [FeatureOverlay("script_kata", 0.7, script_reg.phase_for("kata"), SCRIPT_SLOT)],
            encoding,
        )
        # ident zawiera hira (nie tylko romaji): Hepburn ji/zu kolidują
        # (じ=ぢ='ji', ず=づ='zu') — bez tego dwie różne tożsamości
        # dostałyby ten sam atom przez get_or_create.
        romaji_m = lingua.make_morpheme(
            f"romaji:{romaji}:{hira}",
            sound_feats + [FeatureOverlay("script_romaji", 0.7, script_reg.phase_for("romaji"), SCRIPT_SLOT)],
            encoding,
        )
        slowa.append(lingua.compose(f"{hira}/{kata}/{romaji}", [hira_m, kata_m, romaji_m], kind="word"))
    return lingua, slowa


GOJUON = [row for row in KANA_TABLE if row[5] == "none"]
assert len(GOJUON) == 46, f"gojūon = 46 (bez dakuten/handakuten), jest {len(GOJUON)}"


def przyklad_hiragana_gojuon(
    encoding: AtomEncoding = AtomEncoding.HOLOGRAM,
) -> tuple["LinguaLayer", list]:
    """P1b: sama hiragana, 46 gojūon (bez dakuten/handakuten, bez kata/romaji).

    Mniejszy graf niż pełne Kana P1 (286 atomów) — cel: wyjść ponad
    podłogę InfoNCE zanim mierzyć hologram vs symbolic. Każdy rząd
    spółgłoskowy = jedno compose([hira…]) → krawędzie next + parent/bind.
    """
    import karmazyn_substrate as sub
    from collections import defaultdict

    if encoding is AtomEncoding.SYMBOLIC_ONEHOT:
        raise NotImplementedError(
            "SYMBOLIC_ONEHOT nieobsługiwany w przyklad_hiragana_gojuon — "
            "porównanie to HOLOGRAM vs SYMBOLIC."
        )

    store = sub.Store()
    lingua = LinguaLayer(store, dim=sub.VEC_DIM)
    row_reg = PhaseRegistry(capacity=11)
    vowel_reg = PhaseRegistry(capacity=6)

    by_row: dict[str, list] = defaultdict(list)
    for hira, _kata, _romaji, row, vowel, _diac in GOJUON:
        feats = [
            FeatureOverlay(f"row:{row}", 1.0, row_reg.phase_for(row), ROW_SLOT),
            FeatureOverlay(f"vowel:{vowel}", 0.9, vowel_reg.phase_for(vowel), VOWEL_SLOT),
        ]
        m = lingua.make_morpheme(f"hira:{hira}", feats, encoding)
        by_row[row].append(m)

    slowa = [
        lingua.compose(f"row:{row}", morphs, kind="word")
        for row, morphs in by_row.items()
    ]
    return lingua, slowa
