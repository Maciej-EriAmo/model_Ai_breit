# Glify → atomy Φ (kontrakt)

Kontrakt: **tekst/zdjęcie → atomy → nauka ze stanu**, nie z bitmapy.  
Kanon osi: `1d_do_2d.txt` — litera ma **T** (hot/cold).  
Produkt / asymetria skryptów: [`../docs/TRANSCODE.md`](../../docs/TRANSCODE.md) (`glyph_unit_v2`).

## Atom

| Pole | Sens |
|------|------|
| `id` | `lat/word_0_MOST/0/M`, `hira/…` — slash → `same_prefix` trzyma skrypt |
| `S` | `lat:MOST` (słowo), `lat:M` (litera), `hira:こ` (znak) |
| `E` | rola / sąsiedztwo / bbox (nie T) |
| `T` | fokus / pewność / „czy wpuścić” ∈ [0,1] |
| `state` | HOT / WARM / COLD / TOMB |
| `unit` | `page` \| `line` \| `word` \| `letter` \| `char` |
| `depth` | patrz tabela poniżej (max bin 0..2 w tracerze) |
| `payload` | treść jądra / HOT glyph |
| `mazur_tracer` | pozycja w strumieniu (osobno od T) |
| `topic` | id spanu |

**Zakaz:** T ≠ XY bbox.

## Depth (`glyph_unit_v2`)

| Skrypt | depth 0 | depth 1 | depth 2 |
|--------|---------|---------|---------|
| **Łacina** | strona | **słowo = jądro** | litera |
| **CJK / kana** | strona | linia | **znak = jądro** |

Semantyka łaciny jest na **słowie**; CJK/kana na **znaku**. Litera `M` ≠ leksem „MOST”.

## Edges

- `next` / `next_word` — następstwo (klasyka = ten łańcuch na literach/znakach)
- `contains` / `letter_of` / `char_of` / `word_of` — hierarchia
- `script:*` — HOT peers **tego samego** skryptu (nie lat↔hira)

## Klasyczna metoda vs nowa

| | Klasyczna | Nowa (Most) |
|--|-----------|-------------|
| Tokeny | litery/znaki w sekwencji | + jądra word/char + `T` |
| Uwaga | Softmax | Most (`ΔE` z τ) |
| Gate | Softmax nie może systematycznie wygrywać „za darmo” — Most musi dorównać albo wygrać uczciwie | |

## Dump / API

```text
dumps/from_glyphs/   ← make_dump_glyphs.py (v2)
transcode/           ← build_from_text, semantic_unit, classical_token_sequence
```

```bat
python make_dump_glyphs.py
python doswiadczenie/testy/test_transcode_contract.py
python glyph_transcribe.py --demo --out dumps/from_photo
python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines
```
