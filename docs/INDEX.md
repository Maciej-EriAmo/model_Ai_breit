# PhiMost — indeks dokumentacji

**Root:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz` (Holon / Karmin_Ae)  
**Ostatnia aktualizacja docs:** 2026-09-11 (`TRANSCODE` / `glyph_unit_v2`)

## Czytaj w tej kolejności

| # | Dokument | Po co |
|---|----------|--------|
| 1 | [`../README.md`](../README.md) | start, teza, komendy |
| 2 | [`INSTRUKCJA.md`](INSTRUKCJA.md) | **obsługa** — co klikać na co dzień |
| 3 | [`LEARN.md`](LEARN.md) | ładowanie danych + nauka (`phimost_learn`) |
| 4 | [`TRANSCODE.md`](TRANSCODE.md) | **produkt** — transkodowanie alfabetów; lat=słowo vs CJK=znak; gate vs Softmax |
| 4b | [`HOLO_ZALOZENIA.md`](HOLO_ZALOZENIA.md) | **Holo-Φ** — założenia |
| 4c | [`../doswiadczenie/glify/HOLO_WNIOSKI.md`](../doswiadczenie/glify/HOLO_WNIOSKI.md) | wnioski P0–P2 (blend **0.6**) |
| 5 | [`GLYPHS.md`](GLYPHS.md) | notatki badawcze G0/G1 (lab) |
| 6 | [`POROWNANIE_MODELE.md`](POROWNANIE_MODELE.md) | Most vs Transformer/LLM · czemu uczyć się na własnym |
| 7 | [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md) | **kanon** pamięci M1/M2/M3 + karmienie `x`/`τ` |
| 8 | [`../config.py`](../config.py) | hiperparametry empiryczne (kod) |
| 9 | [`AUDYT.md`](AUDYT.md) | luki, zgodność z wynikami |
| 10 | [`WYNIKI.md`](WYNIKI.md) | skrót tabel PASS/FAIL |
| 11 | [`MODULY.md`](MODULY.md) | mapa plików `.py` |
| 12 | [`STATUS.md`](STATUS.md) / [`PLAN_NEXT.md`](PLAN_NEXT.md) | stan + plan |
| 13 | [`../plan.txt`](../plan.txt) | plan faz A–E + decyzje |

## Fazy (szczegóły)

| Faza | Doc | Wyniki |
|------|------|--------|
| A–B | (README + testy) | [`../historia/WYNIKI_FAZA_B.md`](../historia/WYNIKI_FAZA_B.md) |
| C | — | [`../historia/WYNIKI_RECALL.md`](../historia/WYNIKI_RECALL.md) |
| D | [`FAZA_D.md`](FAZA_D.md) | [`../historia/WYNIKI_FAZA_D.md`](../historia/WYNIKI_FAZA_D.md) |
| E | [`FAZA_E.md`](FAZA_E.md) | [`../historia/WYNIKI_FAZA_E.md`](../historia/WYNIKI_FAZA_E.md) |
| Fizyka | — | [`../historia/WYNIKI_NBODY.md`](../historia/WYNIKI_NBODY.md) |
| **Glify G0–G1** | [`GLYPHS.md`](GLYPHS.md) | [`../doswiadczenie/glify/WYNIKI_GLYPHS.md`](../doswiadczenie/glify/WYNIKI_GLYPHS.md) · [`../doswiadczenie/glify/WYNIKI_GLYPHS_G1.md`](../doswiadczenie/glify/WYNIKI_GLYPHS_G1.md) · labnote [`../doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md`](../doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md) |

## Doświadczenie (osobny katalog)

Hub: [`../doswiadczenie/README.md`](../doswiadczenie/README.md) · checklist: [`../doswiadczenie/CHECKLIST.md`](../doswiadczenie/CHECKLIST.md)

| Tor | Wejście | Wyniki / notatki |
|-----|---------|------------------|
| **JoannaM** (kandydat optyczny) | [`../doswiadczenie/joannam/JOANNAM.md`](../doswiadczenie/joannam/JOANNAM.md) | weryfikacja [`WERYFIKACJA.md`](../doswiadczenie/joannam/WERYFIKACJA.md) · forum EN [`FORUM_EN.md`](../doswiadczenie/joannam/FORUM_EN.md) |
| **DESI / niebo** | [`../doswiadczenie/desi/`](../doswiadczenie/desi/) | `WYNIKI_DESI*.md`, holdout, BH/AGN, [`AUDYT_2026-09-10.md`](../doswiadczenie/desi/AUDYT_2026-09-10.md) |
| **Glify / transkodowanie** | [`TRANSCODE.md`](TRANSCODE.md) · [`GLYPHS.md`](GLYPHS.md) | [`../doswiadczenie/glify/`](../doswiadczenie/glify/) · wnioski [`AUDYT_WNIOSKI.md`](../doswiadczenie/glify/AUDYT_WNIOSKI.md) |

> `historia/` = artefakty faz A–E + n-ciał + learn. Tor nieba i glify → `doswiadczenie/`.

## Case study „plik ≠ płaski”

- [`../dumps/textfile_phi/CASE_TEXT.md`](../dumps/textfile_phi/CASE_TEXT.md)
- [`../dumps/textfile_phi/CASE_RUNTIME.md`](../dumps/textfile_phi/CASE_RUNTIME.md)

## Case study glify / alfabety

- Kontrakt: [`../ZdjęcieW/GLYPH_ATOMS.md`](../ZdjęcieW/GLYPH_ATOMS.md) · oś T: [`../ZdjęcieW/1d_do_2d.txt`](../ZdjęcieW/1d_do_2d.txt)
- Dump: [`../dumps/from_glyphs/CASE_GLYPHS.md`](../dumps/from_glyphs/CASE_GLYPHS.md)
- Metoda (dla badaczy): [`GLYPHS.md`](GLYPHS.md)

## Powiązane poza repo

| Tor | Ścieżka | Rola |
|-----|---------|------|
| Lab Most | `C:\Users\drwis\mcps\moje badania\` | empiria zamka, RAPORT |
| KarmazynOs | `Documents/MAZUR_CRYSTAL.md`, `software/mazur_crystal/` | skalarne R, λ_neural OFF |
| Holon SE | `C:\Users\drwis\Karmin_Ae` + `AGENTS.md` | boot / handoff agenta |

## Status / backlog / plan

[`STATUS.md`](STATUS.md) · [`PLAN_NEXT.md`](PLAN_NEXT.md) · artefakty faz: [`../historia/`](../historia/)

## External code review

[`REVIEW_EXTERNAL.md`](REVIEW_EXTERNAL.md) · [`REVIEW_SUMMARY.md`](REVIEW_SUMMARY.md)  
(od `/review` na katalogu — 4 bugs / 6 suggestions / 2 nits)
