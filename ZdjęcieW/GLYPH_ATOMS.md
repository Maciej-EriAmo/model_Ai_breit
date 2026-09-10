# Glify → atomy Φ (PR-G0)

Kontrakt: **zdjęcie → atomy → nauka ze stanu**, nie z bitmapy.
Kanon osi: `1d_do_2d.txt` — litera ma **T** (hot/cold).

## Atom glifu

| Pole | Sens |
|------|------|
| `id` | `lat/word_most/0/M`, `hira/…` — slash → `same_prefix` trzyma skrypt |
| `S` | `lat:H`, `hira:あ`, … — tożsamość znaku / skrypt |
| `E` | rola w słowie / sąsiedztwo (opis, nie T) |
| `T` | fokus / pewność / „czy wpuścić do okna” ∈ [0,1] |
| `state` | HOT / WARM / COLD / TOMB |
| `depth` | 0 strona → 1 wiersz → 2 glif |
| `payload` | opc. tekst znaku tylko gdy HOT |
| `mazur_tracer` | skrypt / pozycja w słowie (osobno od T) |
| `topic` | słowo / skrypt (do dump recall) |

**Zakaz:** T ≠ XY bbox; XY może dojść później jako osobna oś, nie zamiast T.

## Edges

- `bind` kolejność czytania (sąsiedzi w wierszu)
- `bind` same-depth w obrębie słowa / skryptu
- `parent` glif → wiersz → strona

## Dump

```text
dumps/from_glyphs/
  atoms.jsonl
  edges.jsonl
  manifest.json
  CASE_GLYPHS.md
```

Generator G0: `python make_dump_glyphs.py`  
Transkrypcja G1: `python glyph_transcribe.py --demo|--src foto.png --text "…" --out dumps/from_photo`  
Nauka: `python phimost_learn.py --src dumps/from_glyphs|dumps/from_photo --no-warm --no-promote --baselines`
