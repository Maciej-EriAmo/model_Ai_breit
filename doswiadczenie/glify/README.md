# Glify / empiria — mapa katalogu

**Komora SE:** `Most_Lorentz`  
**Kanon produktu:** [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md)  
**Hub doświadczenia:** [`../README.md`](../README.md)  
**Aktualizacja mapy:** 2026-09-16

> Nie mylić torów: **transcode/G\*** ≠ **Holo-Φ** ≠ **lingua/znaki** ≠ **Mendelejew**.

## Szybki stan

| Tor | Werdykt | Wejście |
|-----|---------|---------|
| G0 / G1 | PASS | `WYNIKI_GLYPHS*.md` |
| G2 parity v2 | PASS **best-seed only** | `WYNIKI_GLYPHS_G2.md` |
| Holo P0–P4a | mieszane / FAIL @ P4a | `HOLO_P*.md`, `HOLO_WNIOSKI.md` |
| Holo P5.1 | **FAIL 0/8** holdout | `HOLO_P5_1.md` |
| Lingua warstwa | **6/6 TODO DONE**, 35/35 testów | `AUDYT_HOLOGRAM_ZNAKI.md` |
| Kana P1 / P1b | P1 negatyw; P1b Most≈SM | `KANA_P1.md`, `KANA_P1B.md` |
| PL P1b | Most≈SM | `POLISH_P1B.md` |
| Mendelejew P0 | **parity Most≈SM**, nonflat 3/3 | `MENDELEJEW_P0.md` · CV: `CV_BLURB_MOST_PARITY.md` |
| Hold-out Na | **negatyw** | `MENDELEJEW_HOLDOUT_Na.md` |
| Mendelejew FULL 118 | **DONE** 2026-10-06 Breit · r@1=0,575 · floor 3/3 | `MENDELEJEW_FULL.md` |
| Średni LM WikiText-2 | **DONE** Most lekko przed SM na valid i teście | `LM_MEDIUM_WIKITEXT_HOLDOUT.md` |
| Hold-out Na/K FULL (square) | **FAIL 2/2** | `MENDELEJEW_HOLDOUT_FULL_NaK.md` |
| Hold-out Na/K **onesided** | Na **PARTIAL** / K **FAIL** | `MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md` |
| Baselines FULL118 | Most≈SM · gate 0/5 · Most 1.7× wolniejszy | `MENDELEJEW_BASELINES_COMPARE.md` |

## Indeks plików (po roli)

### Transcode / G\*
- `WYNIKI_GLYPHS.md` · `WYNIKI_GLYPHS_G1.md` · `WYNIKI_GLYPHS_G2.md` · `WYNIKI_GLYPHS_LARGE.md`
- `LABNOTE_GLYPHS_2026-09-10.md` · `TRANSCODE.md` (stub → docs) · `AUDYT_WNIOSKI.md`

### Holo-Φ
- `HOLOGRAMY.md` · `HOLO_WNIOSKI.md` · `HOLO_WNIOSKIab.md`
- `HOLO_P1.md` … `HOLO_P5_1.md` (+ `*_results.json`)

### Lingua / znaki (Store+HRR)
- `AUDYT_HOLOGRAM_ZNAKI.md` — kanon audytu
- `LINGUA_P1.md` · `LINGUA_PROBE_MLP.md` · `KANA_P1.md` · `KANA_P1B.md` · `POLISH_P1B.md`
- Raport koncepcyjny: [`../../docs/hologram-znaki-raport.md`](../../docs/hologram-znaki-raport.md)

### Mendelejew / struktura
- `MENDELEJEW_P0.md` · `MENDELEJEW_P0_SMOKE.md` · `MENDELEJEW_HOLDOUT_Na.md`
- `JUTRO_MENDELEJEW_FULL.md` · `CV_BLURB_MOST_PARITY.md`
- Skrypty root: `run_mendelejew_p0.py`, `run_mendelejew_holdout.py`, `run_mendelejew_full.py`, `export_mendelejew_full.py`

### Inne
- `SOLAR_NAMES_V2.md` · `założenia.txt` · `KANA_LEARNING_DIAG.md`

### Jądro (2026-10-05)
- `JADRO_OGOLNE.md` — wnioski i program testów T0–T3
- `../../docs/PODSUMOWANIE_JADRA.md` — jedno podsumowanie Most i Softmax
- `NBODY_COMP.md` — relacja na katalogu słonecznym i trójce hierarchicznej

## Co pominięte / otwarte (higiena)

Zob. [`../CHECKLIST.md`](../CHECKLIST.md) § „Luki 2026-09-16” oraz [`../../docs/STATUS.md`](../../docs/STATUS.md).
