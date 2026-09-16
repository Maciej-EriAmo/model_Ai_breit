# JoannaM — ablacja maski gwiazdy Gaia (2026-09-16)

**FITS:** `desi_grid_004_ra120.0_dec5.0_s128.fits` · **ckpt:** `most_desi_color.pt`

## Baseline

| p_cluster | T_mean | \|dE\| | pred |
|----------:|-------:|------:|------|
| **0.7971** | 0.558 | 0.484 | cluster |

## Gwiazdy (pozycja)

| name | dist″ | mag_r | pix (x,y) | peak_lum | snap_shift |
|------|------:|------:|-----------|----------:|-----------:|
| gaia_8p8 | 8.80 | 17.68 | (33,50) | 2.94 | 0.54 |
| gaia_17 | 16.99 | 19.21 | (124,41) | 0.798 | 0.54 |

## Warunki

| warunek | p_cluster | Δp | T_mean | n_pix |
|---------|----------:|---:|-------:|------:|
| `mask_gaia_8p8_r2px` | 0.6387 | -0.1583 | 0.558 | 13 |
| `mask_gaia_8p8_r3px` | 0.4008 | -0.3962 | 0.557 | 29 |
| `mask_gaia_8p8_r4px` | 0.2407 | -0.5563 | 0.557 | 49 |
| `mask_gaia_8p8_r6px` | 0.1348 | -0.6623 | 0.554 | 113 |
| `mask_both_gaia_r3px` | 0.1918 | -0.6053 | 0.556 | 58 |
| `mask_both_gaia_r4px` | 0.0803 | -0.7167 | 0.555 | 97 |
| `ctrl_angle0_r3px` | 0.7971 | -0.0000 | 0.558 | 29 |
| `ctrl_angle90_r3px` | 0.7972 | +0.0001 | 0.558 | 29 |
| `ctrl_angle180_r3px` | 0.8163 | +0.0192 | 0.558 | 29 |
| `ctrl_angle270_r3px` | 0.7974 | +0.0004 | 0.558 | 29 |
| `mask_gaia_8p8_r4px_ZERO` | 0.2498 | -0.5472 | 0.557 | 49 |

## Werdykt

- max Δ (maska gwiazdy): **-0.7167**
- max Δ (kontrole kątowe): **-0.0000**
- gwiazda odpowiedzialna za flagę? **TAK**

If star_responsible: flaga Most silnie zależy od jasnej gwiazdy Gaia. If not: p_cluster trzyma się na strukturze/barwie pola (nie na jednej gwieździe).

JSON: `ABLATION_STAR_MASK.json`
