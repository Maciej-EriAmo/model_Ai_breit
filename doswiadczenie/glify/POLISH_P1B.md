# Polish P1b — alfabet (32), metoda jak Kana P1b, 128 epok

Litery: `aąbcćdeęfghijklłmnńoóprsśtuwyzźż`
Seedy: [0, 1, 17] · epochs=128 (=2× Kana P1b) · n_train=160 n_val=40

Ta sama metoda: FeatureOverlay(grupa, pozycja) + compose grupy → dump Φ → Most/Softmax/ZERO. Próg floor: Most r@1 ≥ 5×chance.

## Agregat

| encoding | atoms | chance | above_floor | PASS | mean Most | mean SM | mean ZERO | mean DSM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hologram | 41 | 0.0250 | 3/3 | 0/3 | 0.279 | 0.312 | 0.307 | -0.0333 |
| symbolic | 41 | 0.0250 | 3/3 | 0/3 | 0.286 | 0.303 | 0.295 | -0.0167 |

**Werdykt:** best=symbolic above_chance=3/3 PASS=0/3 meanMost=0.286 meanDSM=-0.0167 (chance≈0.0250; epochs=128=2×kana_p1b)

Baseline Kana P1b (64 ep): mean Most holo=0.293 sym=0.297 (N=59).

## Per seed

| encoding | seed | Most | SM | ZERO | DSM | floor | gate |
|---|---:|---:|---:|---:|---:|:----:|:----:|
| hologram | 0 | 0.306 | 0.316 | 0.330 | -0.0094 | ABOVE | FAIL |
| hologram | 1 | 0.284 | 0.305 | 0.314 | -0.0203 | ABOVE | FAIL |
| hologram | 17 | 0.247 | 0.317 | 0.278 | -0.0703 | ABOVE | FAIL |
| symbolic | 0 | 0.297 | 0.308 | 0.302 | -0.0109 | ABOVE | FAIL |
| symbolic | 1 | 0.273 | 0.317 | 0.312 | -0.0438 | ABOVE | FAIL |
| symbolic | 17 | 0.287 | 0.283 | 0.272 | +0.0047 | ABOVE | FAIL |

## Krzywa (hologram seed=0)

### Most
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.580 | 0.0531 | 0.1187 | 0.1631 |
| 16 | 1.551 | 0.3234 | 0.7250 | 0.5473 |
| 32 | 1.470 | 0.3172 | 0.7219 | 0.5448 |
| 64 | 1.310 | 0.2922 | 0.7031 | 0.5265 |
| 96 | 1.125 | 0.3063 | 0.6953 | 0.5292 |
| 128 | 0.995 | 0.3063 | 0.7031 | 0.5293 |

### Softmax
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.559 | 0.0094 | 0.1203 | 0.1493 |
| 16 | 1.609 | 0.3000 | 0.7359 | 0.5399 |
| 32 | 1.491 | 0.3141 | 0.7422 | 0.5497 |
| 64 | 1.444 | 0.3328 | 0.7328 | 0.5539 |
| 96 | 1.374 | 0.3094 | 0.7391 | 0.5437 |
| 128 | 1.257 | 0.3156 | 0.7359 | 0.5485 |

### ZERO
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.572 | 0.0500 | 0.1453 | 0.1682 |
| 16 | 1.564 | 0.3234 | 0.7391 | 0.5545 |
| 32 | 1.504 | 0.3344 | 0.7531 | 0.5655 |
| 64 | 1.450 | 0.3266 | 0.7438 | 0.5557 |
| 96 | 1.413 | 0.3109 | 0.7312 | 0.5424 |
| 128 | 1.359 | 0.3297 | 0.7297 | 0.5501 |

JSON: `polish_p1b_results.json`
