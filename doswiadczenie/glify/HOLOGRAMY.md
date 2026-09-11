# Holo-Φ — metoda 2 (próba labowa)

**Status:** pierwsza próba empiryczna **FAIL gate** (2026-09-11)  
**Założenia:** [`../../docs/HOLO_ZALOZENIA.md`](../../docs/HOLO_ZALOZENIA.md) · **wnioski P0–P2:** [`HOLO_WNIOSKI.md`](HOLO_WNIOSKI.md)  
**Skrót w produkcie:** [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md) §7  
**Kod:** `holo_phi.py` · `make_dump_holo.py` · dump `dumps/from_holo` · ckpt `checkpoints/most_holo.pt`

## Idea

| Poziom | Reprezentacja |
|--------|----------------|
| Znak (litera / kana / kanji) | fala \(A\,e^{i\varphi}\) |
| Słowo | interferencja Σ fal liter |
| Rdzeń (**女**) + pochodne | wspólna faza bazowa + offset; bind `holo:root:女` |

Mapowanie do Mostu: Φ + `mazur_tracer=φ/2π` + `blend_holo=0.4` + `most_seal`.

## Dump

| | |
|--|--|
| atoms / edges / HOT | **73** / 354 / 39 |
| lat words | MOST BRIDGE FOCUS TRACE SEAL WAVE ROOT PHASE |
| hira | こんにちは / ありがとう / あいうえお |
| root family | 女 安 姿 数 好 如 |

## Learn (epochs 16, seeds 0,1,42, blend_holo 0.4)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.145 | 0.155 | 0.128 | −0.009 | FAIL |
| 1 | 0.130 | 0.152 | 0.125 | −0.022 | FAIL |
| 42 | 0.155 | 0.152 | 0.127 | **+0.003** | FAIL |

**Werdykt:** **FAIL** (best ΔSM=+0.003 ≪ +0.02).

## Interpretacja

1. Holo-Φ **działa jako nośnik** (Most > ZERO wszędzie).
2. **Nie bije Softmax-twina** — FAIL gate.
3. Audyt założeń: [`../../docs/ocena_holo.txt`](../../docs/ocena_holo.txt) → wchłonięty do [`HOLO_ZALOZENIA.md`](../../docs/HOLO_ZALOZENIA.md) (A=placeholder, 3≠5 seedów, ablacja 女 tautologiczna, blend bez sweepu, ziarnistość r@1).

**Kolejność:** P0–**P4a DONE** ([`HOLO_WNIOSKI.md`](HOLO_WNIOSKI.md) · [`HOLO_P4a.md`](HOLO_P4a.md): **FAIL 1/8** @0.6) → opcjonalnie P4b skalowanie.

## Komendy

```bat
python make_dump_holo.py
python phimost_learn.py --src dumps/from_holo --skip-load --no-warm --no-promote --baselines --epochs 16 --seeds 0,1,42 --blend-holo 0.4 --out-ckpt checkpoints/most_holo.pt
```
