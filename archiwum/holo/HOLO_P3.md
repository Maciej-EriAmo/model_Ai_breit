# Holo-Φ P3 — ablacja 女 (bez tautologii)

blend_holo=**0.6** · seedy [0, 1, 23, 42] · epochs=16

## Design

| Warunek | phase shared | root binds | Sens |
|---------|:------------:|:----------:|------|
| `pp` (++ ) | tak | tak | pełna konstrukcja |
| `mp` (−+) | **nie** | tak | **klucz:** faza off, binds te same |
| `pm` (+−) | tak | nie | faza bez krawędzi rodziny |
| `mm` (−−) | nie | nie | podłoga |

Anty-tautologia: ++ vs −+ nie wyjmuje odpowiedzi z grafu — tylko z τ/fazy.

## Agregat

| cond | phase | binds | PASS | mean ΔSM | fam@3 Most | fam@3 SM | Δfam |
|------|:-----:|:-----:|-----:|---------:|-----------:|---------:|-----:|
| `pp` | True | True | 1/4 | +0.0044 | 1.000 | 1.000 | +0.000 |
| `mp` | False | True | 2/4 | +0.0138 | 1.000 | 1.000 | +0.000 |
| `pm` | True | False | 0/4 | +0.0019 | 1.000 | 1.000 | +0.000 |
| `mm` | False | False | 1/4 | +0.0075 | 1.000 | 1.000 | +0.000 |

**Efekt fazy (pp−mp, binds stałe):** Most fam@3 +0.000 · SM +0.000

**Werdykt:** ++→−+ (binds stałe): Δfam_Most=+0.000 Δfam_SM=+0.000. Faza wspólna mały efekt przy stałych binds — klaster idzie z krawędzi/content, nie z φ_root.

## Pełne wiersze

| cond | seed | ΔSM | gate | fam@3 Most | fam@3 SM |
|------|-----:|----:|:----:|-----------:|---------:|
| `pp` | 0 | +0.0275 | PASS | 1.00 | 1.00 |
| `pp` | 1 | +0.0100 | FAIL | 1.00 | 1.00 |
| `pp` | 23 | -0.0125 | FAIL | 1.00 | 1.00 |
| `pp` | 42 | -0.0075 | FAIL | 1.00 | 1.00 |
| `mp` | 0 | -0.0125 | FAIL | 1.00 | 1.00 |
| `mp` | 1 | +0.0150 | FAIL | 1.00 | 1.00 |
| `mp` | 23 | +0.0300 | PASS | 1.00 | 1.00 |
| `mp` | 42 | +0.0225 | PASS | 1.00 | 1.00 |
| `pm` | 0 | -0.0025 | FAIL | 1.00 | 1.00 |
| `pm` | 1 | +0.0025 | FAIL | 1.00 | 1.00 |
| `pm` | 23 | +0.0075 | FAIL | 1.00 | 1.00 |
| `pm` | 42 | +0.0000 | FAIL | 1.00 | 1.00 |
| `mm` | 0 | +0.0025 | FAIL | 1.00 | 1.00 |
| `mm` | 1 | -0.0075 | FAIL | 1.00 | 1.00 |
| `mm` | 23 | +0.0025 | FAIL | 1.00 | 1.00 |
| `mm` | 42 | +0.0325 | PASS | 1.00 | 1.00 |

JSON: `holo_p3_results.json`

