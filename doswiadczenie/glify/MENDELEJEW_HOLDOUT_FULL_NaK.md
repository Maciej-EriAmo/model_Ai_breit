# Mendelejew FULL118 hold-out — MOST-only (Na / K)

Epoki=64 · seeds=[0, 1] · mode=nucleus+topology · arms=Most-only

Protokół: train bez ukrytego pierwiastka; query `{sym}/nucleus` wstrzyknięty po treningu; kandydaci = 117 jąder train. Kuzyn = ta sama grupa, najbliższy okres (H na końcu remisu).

**Gate:** cousin_symbol_rank ≤ 3 na wszystkich seedach (albo PARTIAL: dowolny group-mate ≤ 3).

## Hold-out `Na`

Train N=117 · cousin=**Li** · group=1
Group mates w train: ['H', 'Li', 'K', 'Rb', 'Cs', 'Fr']

**Werdykt:** FAIL: kuzyn Li NIE w top-3 symboli (ok 1/2); best group-mate top-3: 1/2. FULL118 Most uczy lokalną topologię, nie generalizuje „grupa → alkaliowiec” na ukryty `Na`.

| seed | val r@1 | cousin rank | best group rank | top symbole |
|-----:|--------:|------------:|----------------:|-------------|
| 0 | 0.583 | 29 | 14 | Zr, Ti, V, Bk, Nb |
| 1 | 0.562 | 2 | 2 | Be, Li, Cm, Bk, Eu |

### seed 0 — top atomy / group ranks
group_mate_ranks: `{'H': 27, 'Li': 29, 'K': 31, 'Rb': 25, 'Cs': 15, 'Fr': 14}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Zr/nucleus` | Zr | 0.466 |
| 2 | `Ti/nucleus` | Ti | 0.451 |
| 3 | `V/nucleus` | V | 0.447 |
| 4 | `Bk/nucleus` | Bk | 0.427 |
| 5 | `Nb/nucleus` | Nb | 0.354 |
| 6 | `Cf/nucleus` | Cf | 0.352 |
| 7 | `Sc/nucleus` | Sc | 0.350 |
| 8 | `Hf/nucleus` | Hf | 0.310 |

### seed 1 — top atomy / group ranks
group_mate_ranks: `{'H': 14, 'Li': 2, 'K': 21, 'Rb': 39, 'Cs': 37, 'Fr': 6}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Be/nucleus` | Be | 0.474 |
| 2 | `Li/nucleus` | Li | 0.468 |
| 3 | `Cm/nucleus` | Cm | 0.454 |
| 4 | `Bk/nucleus` | Bk | 0.431 |
| 5 | `Eu/nucleus` | Eu | 0.420 |
| 6 | `Fr/nucleus` | Fr | 0.417 |
| 7 | `Rn/nucleus` | Rn | 0.411 |
| 8 | `Db/nucleus` | Db | 0.391 |

## Hold-out `K`

Train N=117 · cousin=**Na** · group=1
Group mates w train: ['H', 'Li', 'Na', 'Rb', 'Cs', 'Fr']

**Werdykt:** FAIL: kuzyn Na NIE w top-3 symboli (ok 0/2); best group-mate top-3: 0/2. FULL118 Most uczy lokalną topologię, nie generalizuje „grupa → alkaliowiec” na ukryty `K`.

| seed | val r@1 | cousin rank | best group rank | top symbole |
|-----:|--------:|------------:|----------------:|-------------|
| 0 | 0.564 | 19 | 13 | Sr, Mg, Es, Ca, Ho |
| 1 | 0.562 | 27 | 27 | Ca, Mg, Sc, Sr, Th |

### seed 0 — top atomy / group ranks
group_mate_ranks: `{'H': 56, 'Li': 35, 'Na': 19, 'Rb': 27, 'Cs': 24, 'Fr': 13}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Sr/nucleus` | Sr | 0.469 |
| 2 | `Mg/nucleus` | Mg | 0.443 |
| 3 | `Es/nucleus` | Es | 0.435 |
| 4 | `Ca/nucleus` | Ca | 0.424 |
| 5 | `Ho/nucleus` | Ho | 0.411 |
| 6 | `Fm/nucleus` | Fm | 0.402 |
| 7 | `Bk/nucleus` | Bk | 0.377 |
| 8 | `Cf/nucleus` | Cf | 0.372 |

### seed 1 — top atomy / group ranks
group_mate_ranks: `{'H': 63, 'Li': 44, 'Na': 27, 'Rb': 52, 'Cs': 103, 'Fr': 95}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Ca/nucleus` | Ca | 0.681 |
| 2 | `Mg/nucleus` | Mg | 0.579 |
| 3 | `Sc/nucleus` | Sc | 0.482 |
| 4 | `Sr/nucleus` | Sr | 0.468 |
| 5 | `Th/nucleus` | Th | 0.401 |
| 6 | `Pm/nucleus` | Pm | 0.362 |
| 7 | `Al/nucleus` | Al | 0.340 |
| 8 | `Y/nucleus` | Y | 0.327 |

## Podsumowanie

FULL118 hold-out Most-only: PASS=0 PARTIAL=0 FAIL=2 na ['Na', 'K'] (seeds=[0, 1], epochs=64).

JSON: `mendelejew_holdout_full_NaK_results.json`
