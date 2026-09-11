# Lab note — 2026-09-10 — tor Glify / alfabety

**Autor sesji:** agent + Mazur (Karmin_Ae / Most_Lorentz)  
**Katalog:** `C:\Users\drwis\model Ai`

## Kontekst rozmowy

1. Pojawił się pomysł „zdjęcia wszechświata” (DESI Legacy) + drugi plik `1d_do_2d.txt`.
2. Doprecyzowanie użytkownika: **dwa cele** — nauka na mapie nieba *oraz* czytanie zdjęć (anomalie / info z innych projektów).
3. Review: „zerknij do lunety jak wyświetlane jest płótno — czy zdjęciem nauczymy?”  
   → **Nie.** Luneta maluje RGB; prawda = atomy Store z `T`.
4. Korekta kluczowa użytkownika: chodzi o **transkrypcję zdjęcia na atomy** i **naukę ze stanu atomu** — uniwersalnie dla kanji, hiragany, łaciny, alfabetów narodowych.
5. Decyzja: **przestawić plan** — alfabet/glify pierwsze; DESI później.
6. **PR-G0** zlecony i domknięty: PASS seed 42, ΔSM ≈ +0.028.

## Decyzje przyjęte dziś

| Decyzja | Uzasadnienie |
|---------|----------------|
| Nie trenować ze screena Lunety | Brak τ(T); degeneracja do Softmaxu na pikselach |
| Atom MVP = cały glif | Prostszy dump; stroke graph później |
| `id` ze slashami (`lat/word/i/ch`) | `MemoryBus.same_prefix` filtruje skrypt |
| T z fokusa czytania (nie z XY) | Kanon `1d_do_2d` |
| DESI kod (`desi_cutouts` itd.) zostaje szkicem | Nie kasować; nie jest kanonem alfabetu |
| Notatki w `docs/GLYPHS.md` + ten labnote | Pod opis dla potomności / badaczy |

## Artefakty powstałe

- `docs/GLYPHS.md` — metoda + teza + empiria (kanwa paper)
- `ZdjęcieW/GLYPH_ATOMS.md` — kontrakt
- `make_dump_glyphs.py`, `dumps/from_glyphs/`, `checkpoints/most_glyphs.pt`
- `doswiadczenie/glify/WYNIKI_GLYPHS.md`
- Szkice DESI: `desi_cutouts.py`, `image_batch.py`, `bench_desi.py` (FAIL gate — za mało / za łatwe contentem)

## PR-G1 (ten sam dzień)

- [x] `glyph_transcribe.py` — CC + linie + T(fokus) + `--text` → PhiJSON  
- [x] Demo `ZdjęcieW/demo_glyphs.png` → `dumps/from_photo` (9 glifów)  
- [x] Learn PASS seed0 ΔSM=+0.044 · `most_glyphs_photo.pt`  
- [x] Notatki: `doswiadczenie/glify/WYNIKI_GLYPHS_G1.md`, update `docs/GLYPHS.md`

## Otwarte na G2+

- [ ] Zmniejszyć wpływ `archive` COLD w recall (G0)  
- [ ] Drugi skrypt w osobnym korpusie (np. cyrylica lub kanji)  
- [ ] Ablacja `drop_T` udokumentowana w tabeli  
- [ ] Opcjonalny OCR tylko jako propozycja `S` (nie zamiast atomów)  
- [ ] Gdy będzie paper: abstrakt ze §8 `docs/GLYPHS.md`

## Cytowanie wewnętrzne

Komora Holon: fact PR-G0 PASS + work = PR-G1.  
Nie mylić z Karmin_DB / skarbcem — to tor `model Ai` + SE `Most_Lorentz`.
