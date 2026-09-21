"""
Unihan tracer — τ z zewnętrznego, offline źródła (nie hash(etykieta)).

Źródło: oficjalna baza Unihan, Unicode 17.0 (unicode.org/Public/UCD/latest/ucd/Unihan.zip),
pola `kTotalStrokes` (Unihan_IRGSources.txt) i `kRSUnicode` (radical.residual-strokes,
bierzemy pierwszy podany radykał Kangxi). Pobrane i zweryfikowane 2026-09-13.

Bundlowane tu (nie live-download w runtime) — offline, deterministyczne, zgodnie
z `docs/HOLO_P5_PLAN_IMPLEMENTACJI.md` P5.0.

5 radykałów × 20 znaków (większy korpus niż 6 znaków / 73 atomy z P0-P4a):
  9  人 (człowiek)
  38 女 (kobieta)  — ta sama rodzina co P0-P4a, teraz z prawdziwym τ zamiast hasha
  75 木 (drzewo)
  85 水 (woda)
  86 火 (ogień)

WAŻNE: to jest realny pomiar strukturalny znaku (jak jest zbudowany graficznie/
słownikowo), niezależny od `S`/`E`/etykiety używanej jako content w PhiGraph —
czyli spełnia zakaz [[tracer-musi-byc-obserwatorem]]: τ ≠ f(hash(content)).
"""

from __future__ import annotations

# cp -> (kTotalStrokes, kRSUnicode radical Kangxi)
_TABLE: dict[str, tuple[int, int]] = {
    # radical 9 — 人
    "人": (2, 9), "什": (4, 9), "仁": (4, 9), "仅": (4, 9), "仆": (4, 9),
    "今": (4, 9), "介": (4, 9), "仍": (4, 9), "从": (4, 9), "仙": (5, 9),
    "代": (5, 9), "付": (5, 9), "仰": (6, 9), "仲": (6, 9), "件": (6, 9),
    "任": (6, 9), "伊": (6, 9), "伏": (6, 9), "伐": (6, 9), "休": (6, 9),
    # radical 38 — 女
    "女": (3, 38), "好": (6, 38), "如": (6, 38), "妈": (6, 38), "妃": (6, 38),
    "妄": (6, 38), "妆": (6, 38), "妇": (6, 38), "妹": (8, 38), "姐": (8, 38),
    "始": (8, 38), "姿": (9, 38), "姓": (8, 38), "婚": (11, 38), "娘": (10, 38),
    "嫁": (13, 38), "妙": (7, 38), "妒": (7, 38), "妨": (7, 38), "姆": (8, 38),
    # radical 75 — 木
    "木": (4, 75), "未": (5, 75), "末": (5, 75), "本": (5, 75), "朱": (6, 75),
    "朴": (6, 75), "朵": (6, 75), "杀": (6, 75), "杂": (6, 75), "李": (7, 75),
    "杏": (7, 75), "材": (7, 75), "村": (7, 75), "杯": (8, 75), "枝": (8, 75),
    "松": (8, 75), "板": (8, 75), "枪": (8, 75), "果": (8, 75), "柏": (9, 75),
    # radical 85 — 水
    "水": (4, 85), "永": (5, 85), "汁": (5, 85), "汉": (5, 85), "池": (6, 85),
    "汤": (6, 85), "汽": (7, 85), "沙": (7, 85), "沟": (7, 85), "没": (7, 85),
    "河": (8, 85), "油": (8, 85), "泳": (8, 85), "泥": (8, 85), "泡": (8, 85),
    "法": (8, 85), "波": (8, 85), "泽": (8, 85), "洗": (9, 85), "海": (10, 85),
    # radical 86 — 火
    "火": (4, 86), "灰": (6, 86), "灯": (6, 86), "灵": (7, 86), "灾": (7, 86),
    "灿": (7, 86), "炉": (8, 86), "炊": (8, 86), "炎": (8, 86), "炒": (8, 86),
    "炮": (9, 86), "烂": (9, 86), "烈": (10, 86), "烧": (10, 86), "热": (10, 86),
    "煮": (12, 86), "熄": (14, 86), "燃": (16, 86), "烟": (10, 86), "烤": (10, 86),
}

RADICALS: tuple[int, ...] = (9, 38, 75, 85, 86)


def stroke_count(ch: str) -> int:
    """kTotalStrokes rzeczywistego znaku (Unihan). KeyError jeśli poza bundlem."""
    return _TABLE[ch][0]


def radical_id(ch: str) -> int:
    """Numer radykału Kangxi (z kRSUnicode, pierwszy podany). KeyError jeśli poza bundlem."""
    return _TABLE[ch][1]


def chars_for_radical(radical: int) -> list[str]:
    """Wszystkie znaki bundla należące do danego radykału (kolejność wg tabeli)."""
    return [ch for ch, (_, r) in _TABLE.items() if r == radical]


def load_unihan(chars: set[str]) -> dict[str, dict[str, int]]:
    """Zwraca {znak: {"strokes":.., "radical":..}} dla znaków obecnych w bundlu.

    Znaki spoza bundla są pomijane (nie KeyError) — dump-buildery mają same
    decydować, czy brakujący znak jest błędem konfiguracji czy nie.
    """
    return {
        ch: {"strokes": s, "radical": r}
        for ch, (s, r) in _TABLE.items()
        if ch in chars
    }


def all_chars() -> list[str]:
    return list(_TABLE.keys())
