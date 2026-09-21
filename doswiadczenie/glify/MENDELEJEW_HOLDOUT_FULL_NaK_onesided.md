# Mendelejew FULL118 hold-out — MOST (Na / K)

Epoki=64 · seeds=[0, 1] · mode=nucleus+topology
most_kw=`{'delta_mode': 'onesided', 'delta_alpha': 1.0, 'directed_bias': 0.25, 'gamma_init': 0.0, 'e_scale': 1.0, 'couple_temp': 1.0}`

Protokół: train bez ukrytego pierwiastka; query `{sym}/nucleus` wstrzyknięty po treningu; kandydaci = 117 jąder train. Kuzyn = ta sama grupa, najbliższy okres (H na końcu remisu).

**Gate:** cousin_symbol_rank ≤ 3 na wszystkich seedach (albo PARTIAL: dowolny group-mate ≤ 3).

## Hold-out `Na`

Train N=117 · cousin=**Li** · group=1
Group mates w train: ['H', 'Li', 'K', 'Rb', 'Cs', 'Fr']

**Werdykt:** PARTIAL: dokładny kuzyn Li nie zawsze top-3, ale jakiś mate grupy 1 jest w top-3 (2/2) — sygnał grupy częściowy.

| seed | val r@1 | cousin rank | best group rank | top symbole |
|-----:|--------:|------------:|----------------:|-------------|
| 0 | 0.570 | 10 | 1 | K, Cf, Bk, Ds, Ca |
| 1 | 0.576 | 6 | 3 | Bk, Ca, K, Be, Sc |

### seed 0 — top atomy / group ranks
group_mate_ranks: `{'H': 23, 'Li': 10, 'K': 1, 'Rb': 12, 'Cs': 43, 'Fr': 62}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `K/nucleus` | K | 0.493 |
| 2 | `Cf/nucleus` | Cf | 0.480 |
| 3 | `Bk/nucleus` | Bk | 0.473 |
| 4 | `Ds/nucleus` | Ds | 0.440 |
| 5 | `Ca/nucleus` | Ca | 0.431 |
| 6 | `Pd/nucleus` | Pd | 0.403 |
| 7 | `Pt/nucleus` | Pt | 0.401 |
| 8 | `Ar/nucleus` | Ar | 0.393 |

### seed 1 — top atomy / group ranks
group_mate_ranks: `{'H': 48, 'Li': 6, 'K': 3, 'Rb': 53, 'Cs': 95, 'Fr': 88}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Bk/nucleus` | Bk | 0.614 |
| 2 | `Ca/nucleus` | Ca | 0.478 |
| 3 | `K/nucleus` | K | 0.433 |
| 4 | `Be/nucleus` | Be | 0.430 |
| 5 | `Sc/nucleus` | Sc | 0.408 |
| 6 | `Li/nucleus` | Li | 0.342 |
| 7 | `Mg/nucleus` | Mg | 0.332 |
| 8 | `Ds/nucleus` | Ds | 0.326 |

## Hold-out `K`

Train N=117 · cousin=**Na** · group=1
Group mates w train: ['H', 'Li', 'Na', 'Rb', 'Cs', 'Fr']

**Werdykt:** FAIL: kuzyn Na NIE w top-3 symboli (ok 0/2); best group-mate top-3: 1/2. FULL118 Most uczy lokalną topologię, nie generalizuje „grupa → alkaliowiec” na ukryty `K`.

| seed | val r@1 | cousin rank | best group rank | top symbole |
|-----:|--------:|------------:|----------------:|-------------|
| 0 | 0.573 | 57 | 30 | Ho, Sm, Og, Ts, Ca |
| 1 | 0.565 | 59 | 3 | Og, Rn, Fr, Es, Ts |

### seed 0 — top atomy / group ranks
group_mate_ranks: `{'H': 41, 'Li': 30, 'Na': 57, 'Rb': 65, 'Cs': 75, 'Fr': 46}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Ho/nucleus` | Ho | 0.504 |
| 2 | `Sm/nucleus` | Sm | 0.465 |
| 3 | `Og/nucleus` | Og | 0.414 |
| 4 | `Ts/nucleus` | Ts | 0.409 |
| 5 | `Ca/nucleus` | Ca | 0.397 |
| 6 | `Dy/nucleus` | Dy | 0.366 |
| 7 | `Pm/nucleus` | Pm | 0.365 |
| 8 | `At/nucleus` | At | 0.329 |

### seed 1 — top atomy / group ranks
group_mate_ranks: `{'H': 26, 'Li': 20, 'Na': 59, 'Rb': 87, 'Cs': 15, 'Fr': 3}`
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Og/nucleus` | Og | 0.729 |
| 2 | `Rn/nucleus` | Rn | 0.673 |
| 3 | `Fr/nucleus` | Fr | 0.533 |
| 4 | `Es/nucleus` | Es | 0.396 |
| 5 | `Ts/nucleus` | Ts | 0.354 |
| 6 | `Be/nucleus` | Be | 0.331 |
| 7 | `N/nucleus` | N | 0.328 |
| 8 | `Sc/nucleus` | Sc | 0.312 |

## Podsumowanie

FULL118 hold-out Most (onesided): PASS=0 PARTIAL=1 FAIL=1 na ['Na', 'K'] (seeds=[0, 1], epochs=64).

JSON: `mendelejew_holdout_full_NaK_onesided_results.json`
