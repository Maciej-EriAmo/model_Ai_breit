# Asia — kandydat #2 (`desi_new_field_01`, po JoannaM)

**Karta:** [`ASIA.md`](ASIA.md) · **Weryfikacja:** [`CROSSMATCH_C2.md`](CROSSMATCH_C2.md) · JSON `CROSSMATCH_C2.json`  
**Data nazwania:** 2026-09-16 · dedykacja **Asi**  
**FITS:** `ZdjęcieW/demo/desi_new_field_01_ra142.5_dec6.5_s128.fits`  
**Fresh alias:** `desi_c2_fresh_ra142.5_dec6.5_s128.fits`  
**ckpt:** `most_desi_color.pt`

| | |
|--|--|
| **RA** | **142.5°** |
| **Dec** | **+6.5°** |
| **p_cluster** | **0.974** |
| **po masce Gaia @9.7″** | **0.800** (Δ=−0.17) |
| **gal** | l≈226.8° · b≈+38.0° |

## vs JoannaM

| | JoannaM | **#2** |
|--|--------:|-------:|
| Baseline p | 0.797 | **0.974** |
| Po masce jasnej gwiazdy | 0.24 | **0.80** |
| Mechanizm | gwiazda-only | **część gwiazda + silny residual** |
| SIMBAD 60″ | 0 | **0** |
| eROSITA ≤5′ | 0 | **0** |
| NVSS ≤5′ | 0 | **4** (nearest @ **2.49′**, poza cutoutem) |

## Katalog (skrót)

- LS: 1 bright — Gaia `GE:586076295021789056` r≈13.6 @ 9.7″ (π≈1.18 mas)
- DESI: nearest ~63″ (STAR + GALAXY z≈0.32) — **nie** w centrum
- eROSITA nearest @ 7.9′
- Fresh cutout: **sha identyczny**, score/ablacja replikują się

## Werdykt

**Nazwana: Asia.** Nie jest klonem JoannyM.  
**Nie BH claim.** Flaga Most = residual struktura/barwa + lekki boost od gwiazdy.

## Następne (opcjonalnie)

1. Ablacja multi-peak / mapa uwagi Most  
2. FULL118 Mendelejew gdy budżet czasu
