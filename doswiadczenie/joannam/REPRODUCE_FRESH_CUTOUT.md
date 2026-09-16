# JoannaM — reprodukcja na świeżym cutoucie (2026-09-16)

**UTC:** `20260916T060519Z` · **ckpt:** `most_desi_color.pt` · RA 120.0 Dec 5.0

## Pliki

| | plik | sha256[:16] | bajty |
|--|------|-------------|------:|
| stary | `desi_grid_004_ra120.0_dec5.0_s128.fits` | `c572001e20e8cbb3` | 403200 |
| świeży | `desi_joannam_fresh_20260916T060519Z_ra120.0_dec5.0_s128.fits` | `c572001e20e8cbb3` | 403200 |
| alias | `desi_joannam_fresh_ra120.0_dec5.0_s128.fits` | `c572001e20e8cbb3` | 403200 |

Bajty identyczne ze starym? **TAK**

## Score

| cutout | p_cluster | T_mean | \|dE\| | abl r=4px Δp | ctrl Δp |
|--------|----------:|-------:|------:|-------------:|--------:|
| stary | 0.7971 | 0.558 | 0.484 | -0.5563 | -0.0001 |
| **świeży** | **0.7971** | 0.558 | 0.484 | **-0.5563** | -0.0001 |

Δ(fresh−old) p_cluster = **+0.0000**

## Werdykt

- Score reprodukowalny (±5 pp)? **TAK**
- Ablacja gwiazdy nadal działa na świeżym? **TAK**

Fresh Legacy download scored with same most_desi_color.pt. Ablation of Gaia@8.8" should still drop p_cluster if star-driven.

JSON: `REPRODUCE_FRESH_CUTOUT.json`
