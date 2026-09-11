# WYNIKI — duża partia glifów + photo smoke

**Data:** 2026-09-11  
**Seal:** `most_seal` ON · kontrakt `glyph_unit_v2`

## A) Syntetyka duża — `dumps/from_glyphs_large`

| | |
|--|--|
| Generator | `make_dump_glyphs_large.py --max-lat 45` |
| atoms / edges | **431** / 2031 (words 45 · chars 101 · letters 278) |
| Uwaga | Pełne 675+ atomów (wszystkie LAT_WORDS) = trening niepraktyczny O(N²); thinned bez klik `script:*` |
| Train | epochs 12 · seeds 0,1,42 · n_train 100 · n_val 28 · bs=4 · ckpt `most_glyphs_large.pt` |

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.018 | 0.000 | 0.036 | +0.0179 | FAIL |
| 1 | 0.000 | 0.000 | 0.000 | +0.0000 | FAIL |
| 42 | 0.027 | 0.018 | 0.018 | +0.0089 | FAIL |

**Werdykt A:** **FAIL** gate. Absolute r@1 **zapaść** (~0.02 vs ~0.33 na małym dumpie). Most ≈ Softmax ≈ ZERO na dnie — skala bez dobrego sampling/par nie pomaga; lekki +ΔSM na seed0 poniżej +0.02.

## B) Photo smoke — `dumps/from_photo_smoke`

| | |
|--|--|
| | `glyph_transcribe.py --demo --text "MOST BRIDGE ENERGY"` |
| atoms | 21 · contract v2 · word nuclei OK |
| Train | epochs 12 · seeds 0,1,42 · `most_glyphs_photo_smoke.pt` |

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.264 | 0.284 | 0.286 | −0.020 | FAIL |
| 1 | 0.292 | 0.317 | 0.294 | −0.025 | FAIL |
| 42 | 0.312 | 0.347 | 0.320 | −0.034 | FAIL |

**Werdykt B:** **FAIL** — Softmax-twin wyżej na wszystkich seedach.

## Wnioski z próby

1. **Więcej atomów ≠ lepszy gate** przy pełnym grafie w każdej próbce (O(N²) + rzadkie pary).
2. Na large: recall się **rozmywa** (r@1≈0) — trzeba subgraph / restricted candidates / więcej epoch albo inny loader.
3. Photo v2 działa kontraktowo; parity Most vs SM **nie**.
4. Seal nie wystarczy bez skali *i* właściwego protokołu treningu na dużym N.

## Komendy

```bat
python make_dump_glyphs_large.py --max-lat 45
python phimost_learn.py --src dumps/from_glyphs_large --skip-load --no-warm --no-promote --baselines --epochs 12 --seeds 0,1,42 --n-train 100 --n-val 28 --out-ckpt checkpoints/most_glyphs_large.pt

python glyph_transcribe.py --demo --text "MOST BRIDGE ENERGY" --out dumps/from_photo_smoke
python phimost_learn.py --src dumps/from_photo_smoke --skip-load --no-warm --no-promote --baselines --epochs 12 --seeds 0,1,42 --out-ckpt checkpoints/most_glyphs_photo_smoke.pt
```
