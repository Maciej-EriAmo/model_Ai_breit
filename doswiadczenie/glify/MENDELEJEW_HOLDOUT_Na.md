# Mendelejew hold-out — ukryty `Na`

Train bez `Na`: ['H', 'He', 'Li', 'C', 'N', 'O', 'Fe']
Oczekiwany kuzyn chemiczny: **Li** (grupa 1, najbliższy okres; H też g1, ale specjalny).
*(W biegu heurystyka błędnie wskazała H — poniżej ranking liczony też dla Li.)*
Epoki=48 · seed=0

Query: `Na/nucleus` wstrzyknięty z cechami Φ po treningu; kandydaci = tylko atomy z traina.

## Wyniki

| arm | val r@1 | top symbole | rank Li | rank H |
|-----|--------:|-------------|--------:|-------:|
| most | 0.416 | O, Fe, N, C, Li | **5** | 6 |
| softmax | 0.412 | N, Fe, He, O, C | ≥6 | ≥6 |
| flat | 0.375 | Fe, H, He, C, Li | 5 | **2** |

**Werdykt:** hold-out **nie** odzyskał miejsca grupy dla Na. Most/Softmax ciągną do O/Fe/N (bogate powłoki), nie do Li. Flat ma H wysoko — raczej artefakt (mało atomów H / Fe dominuje), nie chemia. Wzorzec tablicy uczy się **lokalnych bindów orbitali**, nie generalizuje „grupa 1 → alkaliowiec” na ukryty pierwiastek przy N=7.
### most — top atomy
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `O/1s/m0/up` | O | 0.538 |
| 2 | `O/1s/m0/down` | O | 0.513 |
| 3 | `Fe/3s/m0/down` | Fe | 0.460 |
| 4 | `O/2s/m0/up` | O | 0.410 |
| 5 | `Fe/nucleus` | Fe | 0.343 |

### softmax — top atomy
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `N/2p/m0/up` | N | 0.519 |
| 2 | `Fe/1s/m0/down` | Fe | 0.400 |
| 3 | `He/nucleus` | He | 0.399 |
| 4 | `O/2s/m0/up` | O | 0.383 |
| 5 | `O/1s/m0/down` | O | 0.373 |

### flat — top atomy
| rank | atom | sym | sim |
|-----:|------|-----|----:|
| 1 | `Fe/2s/m0/up` | Fe | 0.495 |
| 2 | `Fe/3d/m-2/up` | Fe | 0.471 |
| 3 | `Fe/2p/m-1/down` | Fe | 0.468 |
| 4 | `Fe/1s/m0/up` | Fe | 0.454 |
| 5 | `H/nucleus` | H | 0.442 |
