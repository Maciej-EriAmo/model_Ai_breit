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

## Run C — po audycie / fixach (2026-09-11, epochs 20, char_focus=5)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.300 | 0.342 | 0.325 | −0.0422 | FAIL |
| 1 | 0.334 | 0.342 | 0.359 | −0.0078 | FAIL |
| 42 | 0.317 | 0.327 | 0.322 | −0.0094 | FAIL |

**Werdykt C:** **FAIL** (wszystkie seed; best ΔSM=−0.008). Softmax-twin **powyżej** Mostu.

## Run C′ — ablacja char_focus=0 (ten sam kontrakt v2)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.325 | 0.325 | 0.317 | +0.0000 | FAIL |
| 1 | 0.316 | 0.344 | 0.334 | −0.0281 | FAIL |
| 42 | 0.308 | 0.311 | 0.306 | −0.0031 | FAIL |

**Werdykt C′:** **FAIL** — nie samym `char_focus`. Regresja vs Run B po fixach audytu (span_id / budowa / Softmax framing honesty nie zmieniają loadera Softmax — nadal twin-on-Φ).

## Interpretacja parity

| Pytanie | Odpowiedź |
|---------|-----------|
| Run B (pre-audit polish) | PASS best-seed0 (+0.022), 1/3 |
| Po audycie (C/C′) | **FAIL** — Most ≤ Softmax na 0,1,42 |
| Czy Most jest kaleką? | Na tym małym dumpie **po fixach tak w sensie gate** — Softmax-twin wygrywa lub remisuje |
| Claim komercyjny | **Nie** — najpierw odzyskać gate (korpus / pary / τ), nie marketing |

## Run D — po uszczelnieniu T (`most_seal`, epochs 20)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.303 | 0.342 | 0.316 | −0.0391 | FAIL |
| 1 | 0.323 | 0.342 | 0.342 | −0.0187 | FAIL |
| 42 | 0.306 | 0.327 | 0.325 | −0.0203 | FAIL |

**Werdykt D:** gate ΔSM nadal **FAIL**.  
**Recall smoke** (q=`lat:M`): `archive/X` spadł z top-2 (G0, +0.81) na **#7 (+0.04)** — bramka działa na przeciek uwagi; embedding content nadal może rankować archive nisko.  
Kod: `most_seal.py` · test `test_most_seal.py`.

**Dalej:** skala 500–1000 atomów + filtr kandydatów po T w recall; classical Softmax loader.

## Komendy

```bat
cd /d C:\Users\drwis\model Ai
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --skip-load --no-warm --no-promote --baselines --epochs 20 --seeds 0,1,42
```
