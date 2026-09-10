"""
PR-G0: syntetyczny dump glifów → dumps/from_glyphs/

Łacina + hiragana jako atomy Φ ze świadomym T (hot/cold).
Nie bitmapa OCR — nauka ze stanu atomu (S, E, T, state).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DUMP = ROOT / "dumps" / "from_glyphs"

# słowa: (script, chars, word_id, reading_focus_index)
# focus = glif „czytany teraz” → HOT; brzegi chłodniejsze
WORDS = [
    ("lat", list("MOST"), "word_most", 0),  # fokus na M
    ("lat", list("BRIDGE"), "word_bridge", 2),  # fokus na I
    ("hira", list("あいうえお"), "word_aiueo", 2),  # fokus う
    ("hira", list("こんにちは"), "word_konnichiwa", 0),  # fokus こ
]


def _state_for_T(T: float) -> str:
    if T >= 0.75:
        return "HOT"
    if T >= 0.45:
        return "WARM"
    if T >= 0.2:
        return "COLD"
    return "TOMB"


def _glyph_T(i: int, n: int, focus: int) -> float:
    """T spada z odległością od fokusa czytania — oś +1, nie XY."""
    dist = abs(i - focus)
    if dist == 0:
        return 0.95
    if dist == 1:
        return 0.70
    if dist == 2:
        return 0.48
    return 0.28


def _bind(edges: list, a: str, b: str, name: str) -> None:
    edges.append({"kind": "bind", "src": a, "dst": b, "name": name})
    edges.append({"kind": "bind", "src": b, "dst": a, "name": name})


def build() -> tuple[list[dict], list[dict]]:
    atoms: list[dict] = []
    edges: list[dict] = []

    # korzeń korpusu
    atoms.append(
        {
            "id": "corpus",
            "S": "glyphs/corpus",
            "E": "multi-script glyph corpus (lat+hira); learn from atom state",
            "T": 1.0,
            "state": "HOT",
            "bubble": "glyphs",
            "depth": 0,
            "payload": "PR-G0: Latin + hiragana atoms with explicit T",
            "mazur_tracer": 1.0,
            "topic": "corpus",
        }
    )

    for script, chars, word_id, focus in WORDS:
        page_id = f"page_{word_id}"
        line_id = f"line_{word_id}"
        bubble = f"glyphs/{script}/{word_id}"

        atoms.append(
            {
                "id": page_id,
                "S": f"{script}/page/{word_id}",
                "E": f"page/{script}: {''.join(chars)}",
                "T": 0.9,
                "state": "HOT",
                "bubble": bubble,
                "depth": 0,
                "payload": "".join(chars),
                "mazur_tracer": 0.9 if script == "lat" else 0.85,
                "topic": word_id,
            }
        )
        atoms.append(
            {
                "id": line_id,
                "S": f"{script}/line/{word_id}",
                "E": f"line/{script} reading order",
                "T": 0.8,
                "state": "WARM",
                "bubble": bubble,
                "depth": 1,
                "payload": None,
                "mazur_tracer": 0.7,
                "topic": word_id,
            }
        )
        _bind(edges, "corpus", page_id, f"has_page:{word_id}")
        _bind(edges, page_id, line_id, "has_line")
        edges.append(
            {"kind": "parent", "src": line_id, "dst": page_id, "name": "line_of"}
        )

        glyph_ids: list[str] = []
        n = len(chars)
        for i, ch in enumerate(chars):
            # slash-id → same_prefix w MemoryBus trzyma skrypt razem (lat/… vs hira/…)
            gid = f"{script}/{word_id}/{i}/{ch}"
            T = _glyph_T(i, n, focus)
            st = _state_for_T(T)
            # mazur: pozycja w słowie (osobna oś od T)
            mazur = (i + 0.5) / n
            atoms.append(
                {
                    "id": gid,
                    "S": f"{script}:{ch}",
                    "E": f"glyph/{script} pos={i}/{n} word={word_id}",
                    "T": T,
                    "state": st,
                    "bubble": bubble,
                    "depth": 2,
                    "payload": (ch if st == "HOT" else None),
                    "mazur_tracer": mazur,
                    "topic": word_id,
                }
            )
            glyph_ids.append(gid)
            edges.append(
                {"kind": "parent", "src": gid, "dst": line_id, "name": "glyph_of"}
            )
            _bind(edges, line_id, gid, f"contains:{ch}")

        # kolejność czytania (sąsiedzi)
        for i in range(len(glyph_ids) - 1):
            _bind(edges, glyph_ids[i], glyph_ids[i + 1], "next")

        # same-depth: pełny klik w obrębie słowa (Most widzi warstwę glifów)
        for i, a in enumerate(glyph_ids):
            for b in glyph_ids[i + 1 :]:
                _bind(edges, a, b, f"word:{word_id}")

    # peer same-script (depth 2): litery łacińskie między słowami MOST↔BRIDGE
    lat_glyphs = [a for a in atoms if a["id"].startswith("lat/") and a["depth"] == 2]
    hira_glyphs = [a for a in atoms if a["id"].startswith("hira/") and a["depth"] == 2]
    for group, tag in ((lat_glyphs, "script:lat"), (hira_glyphs, "script:hira")):
        # wiąż fokusy (HOT) między słowami tego skryptu
        hots = [a for a in group if a["state"] == "HOT"]
        for i, a in enumerate(hots):
            for b in hots[i + 1 :]:
                _bind(edges, a["id"], b["id"], tag)

    # archiwum: zimne „pomyłki OCR” — nie powinny konkurować o okno
    cold = [
        ("lat/archive/X", "lat:X", "false positive latin", "word_most", 0.12),
        ("hira/archive/ん", "hira:ん", "stale hiragana crumb", "word_konnichiwa", 0.15),
    ]
    for cid, S, E, topic, T in cold:
        atoms.append(
            {
                "id": cid,
                "S": S,
                "E": E,
                "T": T,
                "state": _state_for_T(T),
                "bubble": "glyphs/archive",
                "depth": 2,
                "payload": None,
                "mazur_tracer": 0.05,
                "topic": topic,
            }
        )
        # słabe bind do słowa (same depth) — recall musi odróżnić cold
        word_g = [
            a["id"]
            for a in atoms
            if a.get("topic") == topic
            and a["depth"] == 2
            and a["id"].startswith(("lat/", "hira/"))
            and "/archive/" not in a["id"]
        ]
        if word_g:
            _bind(edges, cid, word_g[0], "archive_near")

    return atoms, edges


def main() -> int:
    DUMP.mkdir(parents=True, exist_ok=True)
    atoms, edges = build()

    with (DUMP / "atoms.jsonl").open("w", encoding="utf-8") as f:
        for a in atoms:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (DUMP / "edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    scripts = sorted({a["S"].split(":")[0] for a in atoms if ":" in a["S"]})
    n_hot = sum(1 for a in atoms if a["state"] == "HOT")
    n_cold = sum(1 for a in atoms if a["state"] in ("COLD", "TOMB"))
    man = {
        "name": "from_glyphs",
        "version": "1.0",
        "description": "PR-G0: Latin+hiragana glyph atoms; learn from T/state not bitmap",
        "n_atoms": len(atoms),
        "n_edges": len(edges),
        "scripts": scripts,
        "n_hot": n_hot,
        "n_cold": n_cold,
        "source": "make_dump_glyphs.py (synthetic; photo transcribe = PR-G1)",
        "canon": "PhiJSON; T≠XY; Bridge input (x,T)",
    }
    (DUMP / "manifest.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    case = f"""# CASE: from_glyphs (PR-G0)

Atoms={len(atoms)} edges={len(edges)} scripts={scripts}
HOT={n_hot} COLD/TOMB={n_cold}

Słowa: MOST, BRIDGE, あいうえお, こんにちは — fokus czytania ma wysokie T.

```bat
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines
python phimost_learn.py --src dumps/from_glyphs --skip-load --skip-train --recall-id lat/word_most/0/M
```

Następny: PR-G1 `glyph_transcribe.py` (zdjęcie → ten sam format).
"""
    (DUMP / "CASE_GLYPHS.md").write_text(case, encoding="utf-8")
    print(f"OK dump → {DUMP} atoms={len(atoms)} edges={len(edges)} scripts={scripts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
