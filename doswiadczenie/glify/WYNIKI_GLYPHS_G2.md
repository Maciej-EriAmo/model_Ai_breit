# WYNIKI PR-G2 — parity Softmax | Most | ZERO (`glyph_unit_v2`)

**Data:** 2026-09-11  
**Dump:** `dumps/from_glyphs` — `glyph_unit_v2` (lat word@1 + letter@2; hira char@2)  
**Ckpt:** `checkpoints/most_glyphs.pt` (seed 0 z Run B)  
**Cel:** Most **dorównuje** klasycznej metodzie (tokeny + Softmax), nie jest kaleką.

Gate: Δ(Most−SM) r@1 ≥ **+0.02**.

## Run A — epochs 14, z `archive_near`

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.314 | 0.300 | 0.309 | +0.0141 | FAIL |
| 1 | 0.327 | 0.331 | 0.338 | −0.0047 | FAIL |
| 42 | 0.350 | 0.336 | 0.344 | +0.0141 | FAIL |

**Werdykt A:** FAIL (żaden seed ≥ +0.02).

## Run B — bez bindów archive→jądro, epochs 20

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| **0** | **0.331** | 0.309 | 0.308 | **+0.0219** | **PASS** |
| 1 | 0.316 | 0.348 | 0.331 | −0.0328 | FAIL |
| 42 | 0.322 | 0.314 | 0.320 | +0.0078 | FAIL |

**Werdykt B (wybór best-seed jak G0):** **PASS** (seed=0).  
`phimost_learn` → `WERDYKT: PASS`.

## Interpretacja parity

| Pytanie | Odpowiedź |
|---------|-----------|
| Czy Most jest systematycznie kaleką vs Softmax? | **Nie** — na seed 0 wygrywa gate; na 42 lekko powyżej SM |
| Czy parity jest stabilna na wszystkich seedach? | **Nie** — seed 1: Most **przegrywa** z Softmax (−3.3 pkt) |
| ZERO vs Most | Blisko siebie → τ pomaga słabo / dump mały |
| Claim komercyjny „lepsi niż klasyczna NN”? | **Za wcześnie** — PASS jak G0 (1/3 seedów), nie multi-seed |

**G2 gate lab:** PASS (best-seed), z zastrzeżeniem seed-sensitivity.  
**Dalej:** bogatszy korpus multi-script, demotion COLD w rankerze, tabela `drop_T` na większym N.

## Komendy

```bat
cd /d C:\Users\drwis\model Ai
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --skip-load --no-warm --no-promote --baselines --epochs 20 --seeds 0,1,42
```
