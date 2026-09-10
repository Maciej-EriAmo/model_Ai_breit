# WYNIKI DESI — barwa→T (ciepłota) + jasność→content

**Data:** 2026-09-10  
**Zmiana prep:**  
- **content** = luminancja + morfologia (bez per-band color)  
- **τ.T** = ciepłota z barwy **g−r** (+ lekko r−z)  
- XY osobno; SNR osobno (nie jako T)  
- uczciwy split, size=128, maska  

**Set:** 40 train / 14 val (unikalne ra,dec)

## Softmax | Most | ZERO

| | Softmax | Most | ZERO |
|--|---------|------|------|
| acc | 0.857 | **0.929** | **0.929** |
| \|dE\| | 0 | 0.557 | 0 |

| Metryka | Δ | |
|---------|---|---|
| **Most − ZERO** | **0.00** | **FAIL** gate mostu (Most≈ZERO) |
| Most − Softmax | **+0.071** | **PASS** vs twin bez obserwatora |

## Porównanie z poprzednim (T≈SNR, color w content)

| | Softmax | Most | ZERO |
|--|---------|------|------|
| przed (SNR-as-T) | 0.93 | **0.50** | 0.86 |
| **po (barwa→T)** | 0.86 | **0.93** | 0.93 |

**Wniosek:** rozdzielenie jasność/barwa **podniosło Most z 0.50 → 0.93** i dało **+7 pkt vs Softmax**.  
Most≈ZERO: na tym val Bridge bez τ też trafia; barwa w T jeszcze nie daje przewagi Most−ZERO (ziarno 1/14≈7%). Kierunek prep jest właściwy.

```bat
python bench_desi.py --demo-only --force-import --size 128 --epochs 40 --augment 4 --val-frac 0.25
```
