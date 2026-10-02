# LOO-CV bh-detekcja — po filtrze jakości

**Filtr:** `ON` · `min_snr=20.0` · epochs=40 · n_bh=9 · 430s

## Odrzucone

| obiekt | powód | snr_core_max | img_max |
|--------|-------|--------------|---------|
| `cena` | empty | 0.00 | 0 |
| `ton618` | snr_core_max<20 | 2.68 | 1 |

**Kept:** `3c273`, `3c279`, `3c345`, `3c48`, `cta102`, `m87`, `mrk421`, `ngc4151`, `oj287`

## Wyniki vs baseline (n=11 bez filtra)

| model | recall | precision | F1 | recall n=11 |
|-------|--------|-----------|----|-------------|
| Softmax | 0.556 (5/9) | 0.333 | 0.417 | 0.545 |
| Most | 0.556 (5/9) | 0.417 | 0.476 | 0.636 |
| ZERO | 0.667 (6/9) | 0.333 | 0.444 | 0.545 |

## Per obiekt (1=caught)

| obiekt | Softmax | Most | ZERO |
|--------|---------|------|------|
| `3c273` | 1 | 1 | 1 |
| `3c279` | 1 | 0 | 0 |
| `3c345` | 0 | 0 | 0 |
| `3c48` | 1 | 0 | 0 |
| `cta102` | 1 | 1 | 1 |
| `m87` | 0 | 0 | 1 |
| `mrk421` | 0 | 1 | 1 |
| `ngc4151` | 0 | 1 | 1 |
| `oj287` | 1 | 1 | 1 |

JSON: `loo_bh_filtered.json`
