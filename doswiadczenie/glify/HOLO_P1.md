# Holo-Φ P1 — kontrola eksperymentalna

**Data:** dump `C:\Users\drwis\model Ai\dumps\from_holo` · atoms=73 · epochs=16 · blend_holo=0.4

## Ziarnistość r@1

| | |
|--|--|
| n_queries (val) | **400** (= 25×16) |
| minimalny krok r@1 | **0.0025** |
| gate ΔSM | 0.02 ≈ **8.0** kroków |

ΔSM gate +0.02 jest rozróżnialny tylko jeśli |Δ| ≳ step; tu step=0.0025 → gate ≈ 8.0 kroków

## Softmax | Most | ZERO (wszystkie seedy)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.163 | 0.160 | 0.160 | +0.0025 | FAIL |
| 1 | 0.152 | 0.140 | 0.138 | +0.0125 | FAIL |
| 2 | 0.133 | 0.138 | 0.138 | -0.0050 | FAIL |
| 7 | 0.122 | 0.110 | 0.120 | +0.0125 | FAIL |
| 11 | 0.177 | 0.170 | 0.135 | +0.0075 | FAIL |
| 17 | 0.133 | 0.155 | 0.142 | -0.0225 | FAIL |
| 23 | 0.180 | 0.185 | 0.165 | -0.0050 | FAIL |
| 42 | 0.113 | 0.115 | 0.133 | -0.0025 | FAIL |

**PASS:** 0/8 · best ΔSM=+0.0125 (seed 7)

**Werdykt P1:** gate PASS 0/8 (kryterium docelowe ≥4/5 wymaga P4; tu kontrola P1)

## Krzywe (skrót)

Pełne JSON: `holo_p1_results.json` → `curves`.

### Most seed=0

| epoch | loss | r@1 |
|------:|-----:|----:|
| 1 | 4.2440 | 0.037 |
| 2 | 4.0794 | 0.055 |
| 3 | 3.8035 | 0.043 |
| 4 | 3.5633 | 0.077 |
| 5 | 3.3453 | 0.130 |
| 6 | 3.1480 | 0.133 |
| 7 | 2.9724 | 0.083 |
| 8 | 2.8802 | 0.113 |
| 9 | 2.8051 | 0.135 |
| 10 | 2.7233 | 0.135 |
| 11 | 2.6797 | 0.138 |
| 12 | 2.6387 | 0.138 |
| 13 | 2.6017 | 0.152 |
| 14 | 2.5682 | 0.160 |
| 15 | 2.5441 | 0.168 |
| 16 | 2.5335 | 0.163 |

Ostatnie 3 epoki r@1: [0.16, 0.1675, 0.1625] · rozrzut=0.0075 (jeszcze się rusza)

### Softmax seed=0 (końcówka)

ep16 loss=2.5712 r@1=0.160

