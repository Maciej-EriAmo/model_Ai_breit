"""
Duży korpus glifów (glyph_unit_v2) → dumps/from_glyphs_large/

Cel audytu: ~500–1000+ atomów zanim claim multi-seed.
  python make_dump_glyphs_large.py
  python phimost_learn.py --src dumps/from_glyphs_large --no-warm --no-promote --baselines --epochs 20 --seeds 0,1,42 --n-train 400 --n-val 80
"""

from __future__ import annotations

import json
from pathlib import Path

from transcode import CONTRACT_VERSION, build_from_text, classical_token_sequence
from transcode.phi_build import state_for_T

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_glyphs_large"

# Słowa łacińskie (jądra) — mieszanka techniczna + codzienna
LAT_WORDS = """
MOST BRIDGE ENERGY FOCUS MEMORY ATTENTION TRANSFORMER SOFTMAX LORENTZ TRACE
VECTOR MATRIX TENSOR GRADIENT LAYER TOKEN SEQUENCE ALPHABET SCRIPT LATIN
HIRAGANA KATAKANA KANJI GLYPH ATOM BUBBLE STATE HEAT COLD WARM ARCHIVE
PAYLOAD CONTENT QUERY RECALL RANKING WINDOW FILTER SEAL MASK PENALTY
NUCLEUS LETTER CHARACTER MORPHEME ROOT DERIVATIVE PHRASE SENTENCE
DOCUMENT PAGE LINE COLUMN INDEX PREFIX SUFFIX CLUSTER GRAPH EDGE BIND
PARENT CHILD DEPTH TRACER COUPLING SOFTPLUS RESIDUAL ENCODER DECODER
CHECKPOINT DUMP MANIFEST CONTRACT VERSION BASELINE CONTROL ZERO ABLATION
""".split()

# Hiragana (znaki = jądra)
HIRA_PHRASES = [
    "こんにちは",
    "ありがとう",
    "あいうえお",
    "かきくけこ",
    "さしすせそ",
    "たちつてと",
    "なにぬねの",
    "はひふへほ",
    "まみむめも",
    "やゆよ",
    "らりるれろ",
    "わをん",
    "はじめまして",
    "よろしくお願いします",
    "おはよう",
    "こんばんは",
]

# Kanji (znaki = jądra) — krótkie
KANJI_PHRASES = [
    "東",
    "西",
    "南",
    "北",
    "女",
    "男",
    "子",
    "学",
    "校",
    "日",
    "月",
    "火",
    "水",
    "木",
    "金",
    "土",
    "人",
    "口",
    "手",
    "目",
]


def build_corpus_text(*, max_lat: int | None = 45) -> str:
    # max_lat ogranicza N (uwaga O(N²)); None = wszystkie słowa
    words = LAT_WORDS if max_lat is None else LAT_WORDS[:max_lat]
    parts: list[str] = [" ".join(words)]
    parts.extend(HIRA_PHRASES)
    parts.extend(KANJI_PHRASES)
    return " ".join(parts)


def _thin_edges(edges: list[dict]) -> list[dict]:
    """Wyrzuć gęste kliki script:* (HOT peers) — zostaw hierarchię + next."""
    keep = []
    for e in edges:
        name = str(e.get("name", ""))
        if name.startswith("script:"):
            continue
        keep.append(e)
    return keep


def build(*, max_lat: int | None = 45) -> tuple[list[dict], list[dict], dict]:
    text = build_corpus_text(max_lat=max_lat)
    result = build_from_text(
        text,
        corpus_note=f"LARGE glyph_unit_v2: lat≤{max_lat} + hira + kanji; seal-ready",
        char_focus=0,
    )
    classical = classical_token_sequence(result)
    n_words = len(result.word_nuclei)
    n_chars = len(result.char_nuclei)
    n_letters = len(result.letters)

    atoms = list(result.atoms)
    edges = _thin_edges(list(result.edges))

    # Fokus: pierwsza litera MOST + こ jeśli jest
    for a in atoms:
        if a.get("unit") == "letter" and a.get("S") == "lat:M" and "MOST" in a.get("id", ""):
            a["T"] = 0.95
            a["state"] = "HOT"
            a["payload"] = "M"
            break
    for a in atoms:
        if a.get("unit") == "char" and a.get("S") == "hira:こ":
            a["T"] = 0.95
            a["state"] = "HOT"
            a["payload"] = "こ"
            break

    # Kilka archive distractorów (różne T w TOMB)
    for i, (S, unit) in enumerate(
        (("lat:X", "letter"), ("lat:Q", "letter"), ("hira:ん", "char"), ("kanji:丶", "char"))
    ):
        sc = S.split(":")[0]
        atoms.append(
            {
                "id": f"{sc}/archive/{i}_{S.split(':')[1]}",
                "S": S,
                "E": "archive distractor large dump",
                "T": 0.06 + 0.02 * i,
                "state": state_for_T(0.06 + 0.02 * i),
                "bubble": "glyphs/archive",
                "depth": 2,
                "unit": unit,
                "payload": None,
                "mazur_tracer": 0.01,
                "topic": "archive",
            }
        )

    meta = {
        "contract": CONTRACT_VERSION,
        "classical_tokens": classical,
        "n_word_nuclei": n_words,
        "n_char_nuclei": n_chars,
        "n_letters": n_letters,
        "n_archive": 4,
        "parity_note": "Softmax-twin on Φ; most_seal on Most keys",
        "text_chars": len(text),
    }
    return atoms, edges, meta


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lat", type=int, default=45, help="ile słów łacińskich (0=all)")
    args = ap.parse_args()
    max_lat = None if args.max_lat <= 0 else args.max_lat

    DUMP.mkdir(parents=True, exist_ok=True)
    atoms, edges, meta = build(max_lat=max_lat)
    with (DUMP / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (DUMP / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    man = {
        "name": "from_glyphs_large",
        "version": "2.0",
        "contract": CONTRACT_VERSION,
        "description": "scaled glyph_unit_v2 corpus for multi-seed gate",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "source": "make_dump_glyphs_large.py",
        **{k: v for k, v in meta.items() if k != "classical_tokens"},
        "classical_token_count": len(meta["classical_tokens"]),
    }
    (DUMP / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (DUMP / "classical_tokens.json").write_text(
        json.dumps(meta["classical_tokens"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"OK {DUMP} atoms={len(atoms)} edges={len(edges)} "
        f"words={n_words if (n_words:=meta['n_word_nuclei']) else 0} "
        f"chars={meta['n_char_nuclei']} letters={meta['n_letters']} "
        f"classical={len(meta['classical_tokens'])}"
    )
    if len(atoms) < 500:
        print(f"WARN atoms={len(atoms)} < 500 — rozważ więcej LAT_WORDS / pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
