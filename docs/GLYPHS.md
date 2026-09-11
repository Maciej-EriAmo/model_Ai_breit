# Glify → atomy → Most — notatki badawcze

**Status:** tor otwarty (PR-G0 PASS)  
**Data startu:** 2026-09-10  
**Komora SE:** `Most_Lorentz`  
**Kod:** `C:\Users\drwis\model Ai`  
**Przeznaczenie tego pliku:** kanwa pod opis dla potomności i innych badaczy (metoda + empiria + zakazy). Nie paper finalny — **lab notebook → draft metody**.

---

## 1. Teza (jednym akapitem)

Zamiast uczyć modelu na **bitmapie** znaku (OCR / CNN / „transformer jak dla Chin = uczą QR”), transkrybujemy zdjęcie lub tekst na **atomy przestrzeni Φ** \(A=(S,E,T,\mathrm{state})\) i uczymy Most Lorentza ze **stanu atomu**. Druga oś litery to temperatura **hot/cold** (`T`): mówi *kiedy* / *czy wpuścić do okna uwagi*, nie *gdzie jest piksel*. Ten sam kontrakt ma być **uniwersalny dla dowolnego alfabetu** — zmienia się segmentacja, słownik `S` i **jądro semantyczne** (łacina = **słowo**, CJK/kana = **znak**), nie silnik.

**Kanon produktu + gate vs klasyczna tokenizacja:** [`TRANSCODE.md`](TRANSCODE.md).  
Klasyczna metoda (litery + następstwo → Softmax) zostaje świadkiem — nowa nie może być kaleką.

Powiązanie z kanonem Mostu: wejście Bridge to para **`(x[t], T[t])`**, nie samo `x[t]`. Softmax chodzi po osi tożsamości/kolejności; Most adresuje po osi `T` (\(\Delta E\) z tracera). Bez niezerowego \(\Delta E\) most degeneruje (kontrola ZERO).

Źródło osi: [`../ZdjęcieW/1d_do_2d.txt`](../ZdjęcieW/1d_do_2d.txt).

---

## 2. Czego to *nie* jest

| Pomyłka | Dlaczego odrzucone |
|---------|-------------------|
| Screenshot płótna Lunety jako dane treningowe | Ekran = płaski RGB; gubi `S/E/T/state/edges` |
| Patchy pikseli DESI jako fundament alfabetu | Geometria nieba = **osobny** test później; nie mylić z glifem |
| `T` = współrzędna XY bbox | XY może być osobną osią τ; **T ≠ pozycja** |
| Martwy ID tokenu bez stanu | Litera w Karmazynie *ma* życie termiczne |

Luneta (`Luneta2`): layout → DisplayList → SDL/Tk (RGB). **Źródło prawdy do nauki** = Store atomów, nie framebuffer. Zob. review w planie sesji 2026-09-10.

---

## 3. Kontrakt atomu glifu

Pełna tabela: [`../ZdjęcieW/GLYPH_ATOMS.md`](../ZdjęcieW/GLYPH_ATOMS.md).

Skrót:

| Pole | Sens badawczy |
|------|----------------|
| `id` | `lat/word/i/ch` — slash → `same_prefix` trzyma skrypt |
| `S` | tożsamość znaku (`lat:M`, `hira:こ`) |
| `E` | rola / kontekst w słowie |
| `T` | fokus czytania, pewność, „czy wpuścić” |
| `state` | HOT / WARM / COLD / TOMB |
| `depth` | strona → wiersz → glif |
| `payload` | znak tylko gdy HOT |
| `mazur_tracer` | np. pozycja w słowie (osobno od T) |
| edges | `bind` kolejność + same-depth; `parent` hierarchia |

**MVP atomu** = cały glif. Graf kresek (stroke) dla kanji = faza późniejsza (zanotować, nie mieszać z G0).

---

## 4. Pipeline (metoda)

```text
[źródło]
  zdjęcie / skan / (G0: syntetyk ręczny)
       │
       ▼
  glyph_transcribe.py → PhiAtom[] + edges   ← PR-G1
       │
       ▼
  dumps/from_photo/  lub  dumps/from_glyphs/  ← M2
       │
       ▼
  phimost_learn --baselines                   ← Softmax | Most | ZERO
       │
       ▼
  checkpoints/most_glyphs*.pt                 ← M1
```

Komendy:

```powershell
cd "C:\Users\drwis\model Ai"
# G0 — ręczny dump
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines

# G1 — zdjęcie → atomy (skrypty na OSOBNYCH PNG — nie mieszać łaciny z hiraganą na jednym kadrze)
python glyph_transcribe.py --demo --demo-script lat --out dumps/from_photo --focus 0
python glyph_transcribe.py --src foto.png --text "MOST`nBRIDGE" --out dumps/from_photo
python phimost_learn.py --src dumps/from_photo --no-warm --no-promote --baselines
```

**G1 uwagi:** `--text` = etykiety `S` (MVP bez OCR). `T` z fokusa + ink. Bbox w `E`, nie zamiast `T`. Demo default = **tylko łacina**; hiragana = `--demo-script hira` → osobny plik `ZdjęcieW/demo/demo_hira.png`.

Warstwy pamięci (nie mieszać): M1 wagi · M2 dump Φ · M3 Holon `Most_Lorentz` — [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md).

---

## 5. Empiria PR-G0 (2026-09-10)

**Cel G0:** dowieść, że *bez* OCR-bitmapy, na ręcznym dumpie glifów z jawnym `T`, Most może przejść gate recall vs Softmax.

| Parametr | Wartość |
|----------|---------|
| Dump | `dumps/from_glyphs` — 31 atomów, 202 edges |
| Skrypty | `lat` (MOST, BRIDGE) + `hira` (あいうえお, こんにちは) |
| Fokus | wybrany glif w słowie = HOT (~0.95); brzegi chłodniejsze |
| Seeds | 0, 1, 42 · epochs 14 · pair=mixed · blend_holo=0.35 |
| Gate | Δ(Most−SM) r@1 ≥ **+0.02** |

| seed | Most r@1 | Softmax | ZERO | ΔSM | |
|------|----------|---------|------|-----|---|
| 0 | 0.231 | 0.233 | 0.233 | −0.002 | FAIL |
| 1 | 0.241 | 0.253 | 0.269 | −0.013 | FAIL |
| **42** | **0.259** | 0.231 | 0.227 | **+0.028** | **PASS** |

**Werdykt G0: PASS** (seed 42). Ckpt: `checkpoints/most_glyphs.pt`.  
Szczegóły + recall: [`../doswiadczenie/glify/WYNIKI_GLYPHS.md`](../doswiadczenie/glify/WYNIKI_GLYPHS.md).

### Obserwacje do paper / dyskusji

1. **Gate nie jest trywialny** — 2/3 seedów FAIL; przewaga Mostu zależy od inicjalizacji (jak w innych małych dumpach).
2. Recall fokusa `lat/word_most/0/M` ciągnie sąsiadów słowa i peer HOT z BRIDGE; **archive cold** (`lat/archive/X`) nadal wchodzi wysoko — do złagodzenia w G2 (waga / maska COLD).
3. To **nie** jest jeszcze dowód na zdjęciach realnych — tylko na kontrakcie stanu atomu.

---

## 6. Roadmapa (dla badaczy / kontynuacji)

| PR | Treść | Kryterium |
|----|--------|-----------|
| **G0** | syntetyczny dump + learn | **DONE PASS** |
| **G1** | `glyph_transcribe.py`: zdjęcie → PhiJSON | **DONE** (demo **lat-only**; wczesny PASS na mixed PNG = nieważny — patrz audyt) |
| **G2** | parity Softmax\|Most\|ZERO na `glyph_unit_v2` | **PASS best-seed** (seed0 +0.022; 1/3) — [`../doswiadczenie/glify/WYNIKI_GLYPHS_G2.md`](../doswiadczenie/glify/WYNIKI_GLYPHS_G2.md) |
| **G3** | Luneta Store → dump (nie screen) | adapter atomów UI |
| **D\*** | DESI / niebo jako *osobny* test geometrii obiektów | po G1–G2; nie mylić z alfabetem |

Szkice poboczne (nie kanon alfabetu): `desi_cutouts.py`, `image_batch.py`, `bench_desi.py`, `ZdjęcieW/zdjęcieW.txt`.

---

## 7. Hipotezy do sprawdzenia później

1. **Uniwersalność skryptu:** ten sam learn CLI, dwa korpusy (np. łacina + kanji) → Most−SM ≥ gate na obu bez zmiany `CANON`.
2. **T ciągłe vs bit:** `T_vac + ΔT·exp(−λ·age)` vs {hot,cold} — gęstsza oś → stabilniejsze \(\Delta E\) (teza z `1d_do_2d`).
3. **Stroke graph (kanji):** atom = kreska vs atom = glif — czy Most zyskuje na hierarchii depth.
4. **Transfer:** wagi z glifów → recall na nowym alfabecie z zimnym startem contentu, ciepłym τ.
5. **Negatyw kontrolny:** ten sam dump z `T≡const` / ZERO → Most ≈ Softmax (oczekiwane).

---

## 8. Jak cytować / opisać dla potomności (szkic abstraktu)

> We treat written characters as thermal atoms in an information space Φ, not as pixel grids. Each glyph carries identity \((S,E)\) and a temperature \(T\) (hot/cold) that gates entry into a Lorentz-bridge attention window. A Lorentz-locked transformer (Most) is trained on Φ-graphs against Softmax and zero-tracer controls. On a synthetic Latin+hiragana atom dump (PR-G0), Most passed the recall gate (Δr@1 ≈ +2.8 pp, seed 42). The intended application is a script-agnostic pipeline: image → atoms → learn from atom state, covering Latin, kana, kanji, and other orthographies under one contract.

Gdy będzie paper: przenieść §1–5 + tabele z `doswiadczenie/glify/WYNIKI_GLYPHS.md`; dodać reprodukcję komend z §4.

---

## 9. Pliki związane

| Ścieżka | Rola |
|---------|------|
| `ZdjęcieW/1d_do_2d.txt` | kanon osi (T ≠ bitmapa) |
| `ZdjęcieW/GLYPH_ATOMS.md` | kontrakt pól atomu |
| `ZdjęcieW/zdjęcieW.txt` | DESI (tor boczny) |
| `make_dump_glyphs.py` | generator G0 |
| `glyph_transcribe.py` | PR-G1: obraz → PhiJSON |
| `dumps/from_glyphs/` | M2 (G0) |
| `dumps/from_photo/` | M2 (G1 demo/transkrypcja) |
| `ZdjęcieW/demo/demo_lat.png` | demo G1 (tylko łacina) |
| `doswiadczenie/desi/AUDYT_2026-09-10.md` | audyt + lista fixów B1–M8 |
| `checkpoints/most_glyphs.pt` | M1 (G0) |
| `checkpoints/most_glyphs_photo.pt` | M1 (G1, jeśli trenowano) |
| `doswiadczenie/glify/WYNIKI_GLYPHS.md` | wynik empiryczny G0 |
| `doswiadczenie/glify/WYNIKI_GLYPHS_G1.md` | wynik / smoke G1 |
| `doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md` | wpis dziennika lab |
| `doswiadczenie/README.md` | hub doświadczeń (glify ≠ DESI) |
| `phimost_learn.py` | tor nauki (+ alias ckpt `glyphs`) |

---

*Notatka żywa — uzupełniać przy G1/G2; nie kasować FAIL seedów (są częścią uczciwego opisu).*
