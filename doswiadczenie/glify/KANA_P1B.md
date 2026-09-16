# Kana P1b — hiragana gojūon (46), więcej epok

Seedy: [0, 1, 17] · epochs=64 · n_train=160 n_val=40 · encodings=['hologram', 'symbolic']

Odpowiedź na FAIL Kana P1 (r@1≈chance na N=286): mniejszy graf (sama hiragana gojūon, bez dakuten/kata/romaji) + 64 epoki + 160 batchy train. Próg „powyżej podłogi”: Most r@1 ≥ 5×chance.

## Agregat

| encoding | atoms | chance | above_floor | PASS | mean Most | mean SM | mean ZERO | mean DSM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hologram | 59 | 0.0172 | 3/3 | 0/3 | 0.293 | 0.295 | 0.290 | -0.0021 |
| symbolic | 59 | 0.0172 | 3/3 | 0/3 | 0.297 | 0.296 | 0.294 | +0.0005 |

**Werdykt:** best=symbolic above_chance=3/3 PASS=0/3 meanMost=0.297 meanDSM=+0.0005 (chance≈0.0172; above_chance = Most r@1 >= 5×chance)

## Per seed

| encoding | seed | Most | SM | ZERO | DSM | floor | gate |
|---|---:|---:|---:|---:|---:|:----:|:----:|
| hologram | 0 | 0.314 | 0.311 | 0.306 | +0.0031 | ABOVE | FAIL |
| hologram | 1 | 0.295 | 0.295 | 0.255 | +0.0000 | ABOVE | FAIL |
| hologram | 17 | 0.269 | 0.278 | 0.309 | -0.0094 | ABOVE | FAIL |
| symbolic | 0 | 0.320 | 0.309 | 0.297 | +0.0109 | ABOVE | FAIL |
| symbolic | 1 | 0.284 | 0.283 | 0.289 | +0.0016 | ABOVE | FAIL |
| symbolic | 17 | 0.286 | 0.297 | 0.295 | -0.0109 | ABOVE | FAIL |

## Krzywa (hologram seed=0)

### Most
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.978 | 0.0406 | 0.1187 | 0.1338 |
| 8 | 1.705 | 0.2938 | 0.7172 | 0.5264 |
| 16 | 1.593 | 0.2875 | 0.7234 | 0.5287 |
| 32 | 1.470 | 0.2953 | 0.7281 | 0.5368 |
| 48 | 1.380 | 0.3109 | 0.7578 | 0.5475 |
| 64 | 1.265 | 0.3141 | 0.7250 | 0.5408 |

### Softmax
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.959 | 0.0344 | 0.1297 | 0.1321 |
| 8 | 1.725 | 0.2859 | 0.7250 | 0.5240 |
| 16 | 1.620 | 0.2828 | 0.7219 | 0.5234 |
| 32 | 1.551 | 0.3000 | 0.7172 | 0.5326 |
| 48 | 1.480 | 0.3078 | 0.7500 | 0.5417 |
| 64 | 1.452 | 0.3109 | 0.7609 | 0.5473 |

### ZERO
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.981 | 0.0219 | 0.1000 | 0.1179 |
| 8 | 1.708 | 0.2922 | 0.7141 | 0.5276 |
| 16 | 1.607 | 0.2687 | 0.7250 | 0.5163 |
| 32 | 1.517 | 0.3078 | 0.7406 | 0.5436 |
| 48 | 1.478 | 0.3109 | 0.7500 | 0.5462 |
| 64 | 1.451 | 0.3063 | 0.7688 | 0.5439 |

JSON: `kana_p1b_results.json`
