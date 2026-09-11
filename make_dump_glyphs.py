"""
PR-G0 / glyph_unit_v2: syntetyczny dump glifów → dumps/from_glyphs/

Łacina: słowa = jądra (depth 1), litery depth 2.
Hiragana: znaki = jądra (depth 2).
Nauka ze stanu atomu — nie bitmapa OCR.

  python make_dump_glyphs.py
  python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines
"""

from __future__ import annotations

import json
from pathlib import Path

from transcode import CONTRACT_VERSION, build_from_text, classical_token_sequence
from transcode.phi_build import state_for_T

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_glyphs"


def build() -> tuple[list[dict], list[dict], dict]:
    text = "MOST BRIDGE あいうえお こんにちは"
    # char_focus: indeks w runie CJK — こ jest 5. znakiem (0-based: あいうえお=0..4, こ=5)
    result = build_from_text(
        text,
        corpus_note="PR-G0 glyph_unit_v2: lat words + hira chars; learn from atom state",
        char_focus=5,
    )
    # Snapshot classical BEFORE archive mutation
    classical = classical_token_sequence(result)
    n_words = len(result.word_nuclei)
    n_chars = len(result.char_nuclei)
    n_letters = len(result.letters)

    atoms = list(result.atoms)  # copy — archive nie psuje result
    edges = list(result.edges)

    for a in atoms:
        if a.get("unit") == "letter" and a.get("S") == "lat:M":
            a["T"] = 0.95
            a["state"] = "HOT"
            a["payload"] = "M"
        if a.get("unit") == "char" and a.get("S") == "hira:こ":
            a["T"] = 0.95
            a["state"] = "HOT"
            a["payload"] = "こ"

    # Archive COLD distractors — poza classical_tokens
    for cid, S, E, unit, T in (
        ("lat/archive/X", "lat:X", "false positive latin", "letter", 0.08),
        ("hira/archive/ん", "hira:ん", "stale hiragana crumb", "char", 0.08),
    ):
        atoms.append(
            {
                "id": cid,
                "S": S,
                "E": E,
                "T": T,
                "state": state_for_T(T),
                "bubble": "glyphs/archive",
                "depth": 2,
                "unit": unit,
                "payload": None,
                "mazur_tracer": 0.02,
                "topic": "archive",
            }
        )

    meta = {
        "contract": CONTRACT_VERSION,
        "classical_tokens": classical,
        "n_word_nuclei": n_words,
        "n_char_nuclei": n_chars,
        "n_letters": n_letters,
        "parity_note": (
            "Softmax --baselines = Softmax-twin on same Φ dump (no ΔE). "
            "classical_tokens.json = reading-order inventory, not Softmax train input."
        ),
    }
    return atoms, edges, meta


def main() -> int:
    DUMP.mkdir(parents=True, exist_ok=True)
    atoms, edges, meta = build()

    with (DUMP / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (DUMP / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    man = {
        "name": "from_glyphs",
        "version": "2.0",
        "contract": CONTRACT_VERSION,
        "description": "glyph_unit_v2: lat word@1+letter@2; hira char@2",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "n_hot": n_hot,
        "source": "make_dump_glyphs.py",
        "canon": "PhiJSON; T≠XY; Bridge (x,T); Softmax-twin=Φ witness (not classical_tokens train)",
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
    case = f"""# CASE: from_glyphs ({CONTRACT_VERSION})

atoms={len(atoms)} edges={len(edges)} HOT={n_hot}
word_nuclei={meta['n_word_nuclei']} char_nuclei={meta['n_char_nuclei']} letters={meta['n_letters']}

classical_tokens (inventory): {meta['classical_tokens'][:12]}…
Softmax --baselines = Softmax-twin on Φ (no ΔE), not train-on-classical_tokens.

```bat
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines
```

See docs/TRANSCODE.md §2.
"""
    (DUMP / "CASE_GLYPHS.md").write_text(case, encoding="utf-8")
    print(
        f"OK {DUMP} atoms={len(atoms)} edges={len(edges)} "
        f"words={meta['n_word_nuclei']} chars={meta['n_char_nuclei']} "
        f"classical={len(meta['classical_tokens'])} contract={CONTRACT_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
