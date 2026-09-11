"""
Budowa atomów Φ z asymetrią jednostek (glyph_unit_v2).

Depth bins (τ n_dim=6, MAX_DEPTH_BINS=3):
  Latin:  page@0 → word@1 (jądro) → letter@2
  CJK:    page@0 → line@1 → char@2 (jądro)

Klasyczna sekwencja = łańcuch `next` na literach/znakach (depth 2) —
Softmax-twin na niej = baseline „starej metody”.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .scripts import detect_script_char, script_prefix
from .units import semantic_unit


CONTRACT_VERSION = "glyph_unit_v2"


def state_for_T(T: float) -> str:
    if T >= 0.75:
        return "HOT"
    if T >= 0.45:
        return "WARM"
    if T >= 0.2:
        return "COLD"
    return "TOMB"


def glyph_T(i: int, n: int, focus: int) -> float:
    dist = abs(i - focus)
    if dist == 0:
        return 0.95
    if dist == 1:
        return 0.70
    if dist == 2:
        return 0.48
    return 0.28


def bind(edges: list, a: str, b: str, name: str) -> None:
    edges.append({"kind": "bind", "src": a, "dst": b, "name": name})
    edges.append({"kind": "bind", "src": b, "dst": a, "name": name})


@dataclass
class TextSpan:
    """Jedna jednostka wejściowa: słowo łacińskie albo ciąg CJK."""

    script: str
    chars: list[str]
    span_id: str
    focus: int = 0  # indeks w chars


@dataclass
class BuildResult:
    atoms: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    contract: str = CONTRACT_VERSION

    @property
    def word_nuclei(self) -> list[dict]:
        return [a for a in self.atoms if a.get("unit") == "word"]

    @property
    def char_nuclei(self) -> list[dict]:
        return [a for a in self.atoms if a.get("unit") == "char"]

    @property
    def letters(self) -> list[dict]:
        return [a for a in self.atoms if a.get("unit") == "letter"]


def parse_text_spans(text: str, *, default_script: str = "auto") -> list[TextSpan]:
    """
    Łacina: split po whitespace → osobne słowa.
    CJK/kana: każdy znak = span (jądro char); bieżące runy tego samego skryptu
    dostają wspólny line_id później w builderze.
    """
    spans: list[TextSpan] = []
    if not text:
        return spans

    # Najpierw dzielimy na tokeny whitespace (dla łaciny); CJK bez spacji = jeden token run
    parts = text.split()
    if len(parts) <= 1 and text.strip() and any(
        semantic_unit(detect_script_char(c)) == "char" for c in text if not c.isspace()
    ):
        # Ciąg bez spacji z CJK → per-char spans
        chars = [c for c in text if not c.isspace()]
        for i, ch in enumerate(chars):
            sc = detect_script_char(ch)
            spans.append(
                TextSpan(script=sc, chars=[ch], span_id=f"{sc}_{i}_{ch}", focus=0)
            )
        return spans

    for pi, part in enumerate(parts):
        chars = list(part)
        if not chars:
            continue
        # skrypt z pierwszego znaku „treściowego”
        sc0 = detect_script_char(chars[0])
        if default_script not in ("auto", "") and all(
            detect_script_char(c) == "sym" or detect_script_char(c) == default_script
            for c in chars
        ):
            sc0 = default_script if default_script != "auto" else sc0

        if semantic_unit(sc0) == "word":
            spans.append(
                TextSpan(
                    script=sc0 if sc0 != "sym" else "lat",
                    chars=chars,
                    span_id=f"word_{pi}_{''.join(chars)}",
                    focus=0,
                )
            )
        else:
            for i, ch in enumerate(chars):
                sc = detect_script_char(ch)
                spans.append(
                    TextSpan(
                        script=sc,
                        chars=[ch],
                        span_id=f"{sc}_{pi}_{i}_{ch}",
                        focus=0,
                    )
                )
    return spans


def build_from_spans(
    spans: list[TextSpan],
    *,
    corpus_note: str = "transcode glyph_unit_v2",
    page_id: str = "page_main",
) -> BuildResult:
    atoms: list[dict] = []
    edges: list[dict] = []

    atoms.append(
        {
            "id": "corpus",
            "S": "glyphs/corpus",
            "E": corpus_note,
            "T": 1.0,
            "state": "HOT",
            "bubble": "glyphs",
            "depth": 0,
            "unit": "page",
            "payload": corpus_note,
            "mazur_tracer": 1.0,
            "topic": "corpus",
        }
    )
    atoms.append(
        {
            "id": page_id,
            "S": "page/main",
            "E": f"page n_spans={len(spans)}",
            "T": 0.9,
            "state": "HOT",
            "bubble": "glyphs",
            "depth": 0,
            "unit": "page",
            "payload": None,
            "mazur_tracer": 0.9,
            "topic": "page",
        }
    )
    bind(edges, "corpus", page_id, "has_page")

    word_ids: list[str] = []
    char_ids: list[str] = []
    letter_ids: list[str] = []
    global_letter_i = 0
    # dla CJK: jedna linia na run
    line_id = f"line_{page_id}"
    line_created = False

    for span in spans:
        unit = semantic_unit(span.script)
        n = len(span.chars)
        focus = int(max(0, min(span.focus, max(n - 1, 0))))

        if unit == "word":
            wid = f"lat/{span.span_id}"
            word_ids.append(wid)
            lemma = "".join(span.chars)
            atoms.append(
                {
                    "id": wid,
                    "S": f"{span.script}:{lemma}",
                    "E": f"word nucleus script={span.script} n_letters={n}",
                    "T": 0.92,
                    "state": "HOT",
                    "bubble": f"glyphs/{span.script}/{span.span_id}",
                    "depth": 1,
                    "unit": "word",
                    "payload": lemma,
                    "mazur_tracer": 0.85,
                    "topic": span.span_id,
                }
            )
            bind(edges, page_id, wid, f"has_word:{span.span_id}")
            edges.append(
                {"kind": "parent", "src": wid, "dst": page_id, "name": "word_of"}
            )

            gids: list[str] = []
            for i, ch in enumerate(span.chars):
                sc, S = script_prefix(span.script, ch, i)
                gid = f"{sc}/{span.span_id}/{i}/{ch}"
                T = glyph_T(i, n, focus)
                st = state_for_T(T)
                atoms.append(
                    {
                        "id": gid,
                        "S": S,
                        "E": f"letter/{sc} pos={i}/{n} word={span.span_id}",
                        "T": T,
                        "state": st,
                        "bubble": f"glyphs/{sc}/{span.span_id}",
                        "depth": 2,
                        "unit": "letter",
                        "payload": (ch if st == "HOT" else None),
                        "mazur_tracer": (i + 0.5) / max(n, 1),
                        "topic": span.span_id,
                    }
                )
                gids.append(gid)
                letter_ids.append(gid)
                edges.append(
                    {"kind": "parent", "src": gid, "dst": wid, "name": "letter_of"}
                )
                bind(edges, wid, gid, f"contains:{ch}")
                global_letter_i += 1
            for a, b in zip(gids, gids[1:]):
                bind(edges, a, b, "next")
        else:
            # char nucleus (CJK / kana)
            if not line_created:
                atoms.append(
                    {
                        "id": line_id,
                        "S": f"line/{page_id}",
                        "E": "line for char-nucleus scripts",
                        "T": 0.8,
                        "state": "WARM",
                        "bubble": "glyphs",
                        "depth": 1,
                        "unit": "line",
                        "payload": None,
                        "mazur_tracer": 0.7,
                        "topic": "line",
                    }
                )
                bind(edges, page_id, line_id, "has_line")
                edges.append(
                    {"kind": "parent", "src": line_id, "dst": page_id, "name": "line_of"}
                )
                line_created = True

            ch = span.chars[0]
            sc, S = script_prefix(span.script, ch, global_letter_i)
            cid = f"{sc}/{span.span_id}"
            T = glyph_T(len(char_ids), max(len(spans), 1), 0)
            # fokus na pierwszym char span — builder wywołań może nadpisać
            st = state_for_T(T)
            atoms.append(
                {
                    "id": cid,
                    "S": S,
                    "E": f"char nucleus script={sc}",
                    "T": T,
                    "state": st,
                    "bubble": f"glyphs/{sc}",
                    "depth": 2,
                    "unit": "char",
                    "payload": (ch if st == "HOT" else None),
                    "mazur_tracer": (len(char_ids) + 0.5) / max(len(spans), 1),
                    "topic": span.span_id,
                }
            )
            char_ids.append(cid)
            edges.append(
                {"kind": "parent", "src": cid, "dst": line_id, "name": "char_of"}
            )
            bind(edges, line_id, cid, f"contains:{S}")
            global_letter_i += 1

    # next między słowami (łacina)
    for a, b in zip(word_ids, word_ids[1:]):
        bind(edges, a, b, "next_word")
    # next między znakami CJK
    for a, b in zip(char_ids, char_ids[1:]):
        bind(edges, a, b, "next")

    # HOT peers same-script (word nuclei / char nuclei)
    for group, tag in (
        (word_ids, "script:lat"),
        ([c for c in char_ids if c.startswith("hira/")], "script:hira"),
        ([c for c in char_ids if c.startswith("kanji/")], "script:kanji"),
        ([c for c in char_ids if c.startswith("kata/")], "script:kata"),
    ):
        hots = [
            a
            for a in atoms
            if a["id"] in group and a["state"] == "HOT"
        ]
        for i, a in enumerate(hots):
            for b in hots[i + 1 :]:
                bind(edges, a["id"], b["id"], tag)

    return BuildResult(atoms=atoms, edges=edges, contract=CONTRACT_VERSION)


def build_from_text(
    text: str,
    *,
    default_script: str = "auto",
    corpus_note: str | None = None,
) -> BuildResult:
    spans = parse_text_spans(text, default_script=default_script)
    note = corpus_note or f"transcode v2 text={text!r}"
    return build_from_spans(spans, corpus_note=note)


def classical_token_sequence(result: BuildResult) -> list[str]:
    """
    Klasyczna tokenizacja: litery (lat) + znaki (CJK) w kolejności `next`.
    Softmax na tej liście = stara metoda.
    """
    seq: list[str] = []
    # letters first in word order, then chars
    for a in result.atoms:
        if a.get("unit") in ("letter", "char") and a.get("S"):
            seq.append(a["S"])
    return seq
