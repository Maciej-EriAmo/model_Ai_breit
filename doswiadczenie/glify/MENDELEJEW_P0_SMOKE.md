# Mendelejew P0 — wieloosiowa Φ vs płaski tekst

Dump: `from_struktura` · symbols=['H', 'He', 'Li', 'C', 'N', 'O', 'Na', 'Fe'] · N=72 · chance=0.0141
Epoki=16 · seedy=[0] · n_train=160 · smoke=True

Hipoteza: pojęcia wielowymiarowe (period×group×shells) są dla Mostu łatwiejsze niż ta sama wiedza jako płaski payload.

## Agregat

| arm | mean r@1 |
|-----|---------:|
| A Most Φ | 0.456 |
| B Most flat | 0.227 |
| C Softmax | 0.448 |
| D Most ZERO | 0.466 |

| Δ(A−B) flat | +0.230 | nonflat_win 1/1 |
| Δ(A−C) SM | +0.008 | most_win 0/1 |
| Δ(A−D) τ | -0.009 | tau_win 0/1 |
| above floor | 1/1 |

**Werdykt (kind=2):** Most_A ≫ flat, ale Most≈SM≈ZERO — geometria grafu pomaga wszystkim; Most nie jest specjalny (jak Kana/PL P1b).

## Per seed

| seed | A Most | B flat | C SM | D ZERO | Δflat | ΔSM | Δτ | flags |
|-----:|-------:|-------:|-----:|-------:|------:|-----:|----:|-------|
| 0 | 0.456 | 0.227 | 0.448 | 0.466 | +0.230 | +0.008 | -0.009 | floor,nonflat |

## Krzywe (seed=0)

### A Most Φ
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.752 | 0.2469 | 0.4672 | 0.4122 |
| 4 | 1.523 | 0.4484 | 0.7172 | 0.6176 |
| 8 | 1.397 | 0.4641 | 0.7422 | 0.6363 |
| 12 | 1.373 | 0.4641 | 0.7391 | 0.6369 |
| 16 | 1.361 | 0.4562 | 0.7438 | 0.6335 |

### B Most flat
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 4.268 | 0.0172 | 0.0234 | 0.0674 |
| 4 | 4.257 | 0.0156 | 0.0453 | 0.0737 |
| 8 | 4.187 | 0.0250 | 0.0688 | 0.0957 |
| 12 | 3.796 | 0.0563 | 0.1797 | 0.1728 |
| 16 | 2.738 | 0.2266 | 0.4375 | 0.3729 |

### C Softmax
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.717 | 0.2078 | 0.4578 | 0.3726 |
| 4 | 1.549 | 0.4313 | 0.7406 | 0.6180 |
| 8 | 1.404 | 0.4500 | 0.7531 | 0.6305 |
| 12 | 1.378 | 0.4484 | 0.7562 | 0.6286 |
| 16 | 1.360 | 0.4484 | 0.7453 | 0.6263 |

### D Most ZERO
| ep | loss | r@1 | r@3 | mrr |
|---:|-----:|----:|----:|----:|
| 1 | 3.771 | 0.2109 | 0.4313 | 0.3761 |
| 4 | 1.523 | 0.4500 | 0.7500 | 0.6257 |
| 8 | 1.395 | 0.4672 | 0.7453 | 0.6383 |
| 12 | 1.376 | 0.4594 | 0.7547 | 0.6318 |
| 16 | 1.363 | 0.4656 | 0.7484 | 0.6358 |


Qumran / hologram-E — jutro (poza P0).
