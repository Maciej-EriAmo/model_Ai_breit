"""
Budowa atomów Φ z asymetrią jednostek (glyph_unit_v2).

Depth bins (τ n_dim=6, MAX_DEPTH_BINS=3):
  Latin:  page@0 → word@1 (jądro) → letter@2
  CJK:    page@0 → line@1 → char@2 (jądro)

classical_token_sequence = litery/znaki w kolejności czytania (bez archive).
Softmax w --baselines = Softmax-twin na tym samym dumpie Φ (bez ΔE) — nie
osobny trening na classical_tokens; patrz docs/TRANSCODE.md §2.
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


def _unit_kind(ch: str) -> str:
    """word-run vs char-run (sym idzie z alfabetem / word)."""
    return semantic_unit(detect_script_char(ch))


def _script_runs(chars: list[str]) -> list[tuple[str, list[str]]]:
    """Dziel ciąg na runy: contiguous word-nucleus vs char-nucleus."""
    if not chars:
        return []
    runs: list[tuple[str, list[str]]] = []
    cur_kind = _unit_kind(chars[0])
    buf = [chars[0]]
    for ch in chars[1:]:
        k = _unit_kind(ch)
        # sym w środku word-run zostaje w word; char-run tylko char scripts
        if k == cur_kind or (cur_kind == "word" and detect_script_char(ch) == "sym"):
            buf.append(ch)
        else:
            runs.append((cur_kind, buf))
            cur_kind = k
            buf = [ch]
    runs.append((cur_kind, buf))
    return runs


@dataclass
class TextSpan:
    """Jedna jednostka wejściowa: słowo łacińskie albo pojedynczy znak CJK."""

    script: str
    chars: list[str]
    span_id: str
    focus: int = 0  # indeks w chars (word) albo 0 (char)


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

    @property
    def reading_units(self) -> list[dict]:
        """Litery + znaki w kolejności atomów (bez archive)."""
        return [
            a
            for a in self.atoms
            if a.get("unit") in ("letter", "char") and "/archive/" not in a.get("id", "")
        ]


def _spans_from_char_list(
    chars: list[str], *, part_i: int, default_script: str = "auto"
) -> list[TextSpan]:
    spans: list[TextSpan] = []
    si = 0
    for kind, run in _script_runs(chars):
        if kind == "word":
            sc0 = detect_script_char(run[0])
            if sc0 == "sym":
                sc0 = "lat"
            if default_script not in ("auto", "") and all(
                detect_script_char(c) in ("sym", default_script) for c in run
            ):
                sc0 = default_script
            lemma = "".join(run)
            spans.append(
                TextSpan(
                    script=sc0,
                    chars=list(run),
                    span_id=f"word_{part_i}_{si}_{lemma}",
                    focus=0,
                )
            )
            si += 1
        else:
            for j, ch in enumerate(run):
                sc = detect_script_char(ch)
                spans.append(
                    TextSpan(
                        script=sc,
                        chars=[ch],
                        span_id=f"{sc}_{part_i}_{si}_{j}_{ch}",
                        focus=0,
                    )
                )
            si += 1
    return spans


def parse_text_spans(text: str, *, default_script: str = "auto") -> list[TextSpan]:
    """
    Whitespace → granice słów łacińskich.
    Wewnątrz tokena: runy skryptów (MOSTこんにちは → word MOST + 5× hira).
    CJK/kana bez spacji: każdy znak = span char.
    """
    spans: list[TextSpan] = []
    if not text:
        return spans

    # Zachowaj kolejność; split na whitespace (spacje / newline)
    parts = text.split()
    if not parts:
        return spans

    for pi, part in enumerate(parts):
        chars = list(part)
        if not chars:
            continue
        spans.extend(_spans_from_char_list(chars, part_i=pi, default_script=default_script))
    return spans


def build_from_spans(
    spans: list[TextSpan],
    *,
    corpus_note: str = "transcode glyph_unit_v2",
    page_id: str = "page_main",
    char_focus: int | None = None,
) -> BuildResult:
    """
    char_focus: indeks w globalnej liście char-spans (CJK), domyślnie 0.
    Word spans używają własnego span.focus.
    """
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
    char_span_indices = [i for i, s in enumerate(spans) if semantic_unit(s.script) == "char"]
    n_chars = len(char_span_indices)
    cf = 0 if char_focus is None else int(max(0, min(char_focus, max(n_chars - 1, 0))))
    line_id = f"line_{page_id}"
    line_created = False
    char_ordinal = 0

    for span in spans:
        unit = semantic_unit(span.script)
        n = len(span.chars)
        focus = int(max(0, min(span.focus, max(n - 1, 0))))

        if unit == "word":
            sc_id = span.script if span.script != "sym" else "lat"
            wid = f"{sc_id}/{span.span_id}"
            word_ids.append(wid)
            lemma = "".join(span.chars)
            atoms.append(
                {
                    "id": wid,
                    "S": f"{sc_id}:{lemma}",
                    "E": f"word nucleus script={sc_id} n_letters={n}",
                    "T": 0.92,
                    "state": "HOT",
                    "bubble": f"glyphs/{sc_id}/{span.span_id}",
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
                sc, S = script_prefix(sc_id, ch, i)
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
                edges.append(
                    {"kind": "parent", "src": gid, "dst": wid, "name": "letter_of"}
                )
                bind(edges, wid, gid, f"contains:{ch}")
            for a, b in zip(gids, gids[1:]):
                bind(edges, a, b, "next")
        else:
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
            sc, S = script_prefix(span.script, ch, char_ordinal)
            cid = f"{sc}/{span.span_id}"
            # fokus w obrębie runu znaków (nie całej mieszanki ze słowami)
            T = glyph_T(char_ordinal, max(n_chars, 1), cf)
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
                    "mazur_tracer": (char_ordinal + 0.5) / max(n_chars, 1),
                    "topic": span.span_id,
                }
            )
            char_ids.append(cid)
            edges.append(
                {"kind": "parent", "src": cid, "dst": line_id, "name": "char_of"}
            )
            bind(edges, line_id, cid, f"contains:{S}")
            char_ordinal += 1

    for a, b in zip(word_ids, word_ids[1:]):
        bind(edges, a, b, "next_word")
    for a, b in zip(char_ids, char_ids[1:]):
        bind(edges, a, b, "next")

    for group, tag in (
        ([w for w in word_ids if w.startswith("lat/")], "script:lat"),
        ([c for c in char_ids if c.startswith("hira/")], "script:hira"),
        ([c for c in char_ids if c.startswith("kanji/")], "script:kanji"),
        ([c for c in char_ids if c.startswith("kata/")], "script:kata"),
    ):
        hots = [a for a in atoms if a["id"] in group and a["state"] == "HOT"]
        for i, a in enumerate(hots):
            for b in hots[i + 1 :]:
                bind(edges, a["id"], b["id"], tag)

    return BuildResult(atoms=atoms, edges=edges, contract=CONTRACT_VERSION)


def build_from_text(
    text: str,
    *,
    default_script: str = "auto",
    corpus_note: str | None = None,
    char_focus: int | None = None,
) -> BuildResult:
    spans = parse_text_spans(text, default_script=default_script)
    note = corpus_note or f"transcode v2 text={text!r}"
    return build_from_spans(spans, corpus_note=note, char_focus=char_focus)


def classical_token_sequence(result: BuildResult) -> list[str]:
    """
    Klasyczna tokenizacja: litery + znaki w kolejności czytania (kolejność atomów).
    Pomija archive. (Bind `next` jest dwukierunkowy — nie nadaje się do walki.)
    """
    return [a["S"] for a in result.reading_units if a.get("S")]
