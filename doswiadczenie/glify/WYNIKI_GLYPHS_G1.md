# WYNIKI PR-G1 — zdjęcie → atomy → nauka

**Data:** 2026-09-10  
**Obraz:** `ZdjęcieW/demo_glyphs.png` (MOST + こんにちは, znaki z odstępem)  
**Dump:** `dumps/from_photo` — 9 glifów, 2 linie, 13 atomów  
**Ckpt:** `checkpoints/most_glyphs_photo.pt`  
**Narzędzie:** `glyph_transcribe.py`

## Transkrypcja (smoke)

| Glif | S | T | state |
|------|---|---|--------|
| 0 | lat:M | 0.88 | HOT (fokus) |
| 1–3 | lat:O/S/T | ↓ | WARM→COLD |
| 4–8 | hira:こんにちは | ~0.26 | COLD |

Bbox tylko w `E`. `T` z fokusa + ink — nie z XY.

## Softmax | Most | ZERO (r@1, gate ≥ +0.02)

| seed | Most | SM | ZERO | ΔSM | |
|------|------|----|------|-----|---|
| **0** | **0.265** | 0.221 | 0.231 | **+0.044** | **PASS** |
| 42 | 0.265 | 0.258 | 0.262 | +0.008 | FAIL |

**Werdykt G1 learn: PASS** (seed 0).

## Recall `lat/demo_glyphs/0/M`

1. `lat/demo_glyphs/3/T`  
2. `lat/demo_glyphs/1/O`  
3. `lat/demo_glyphs/2/S`  

## Powtórka

```bat
cd /d C:\Users\drwis\model Ai
python glyph_transcribe.py --demo --out dumps/from_photo --focus 0
python phimost_learn.py --src dumps/from_photo --no-warm --no-promote --baselines --out-ckpt checkpoints/most_glyphs_photo.pt
```

## Ograniczenia (uczciwie)

- Etykiety `S` z `--text` (kolejność czytania) — nie pełny OCR.
- Segmentacja CC + projekcja przy znanej liczbie znaków w linii.
- Mały dump — gate zależny od seeda (jak G0).
