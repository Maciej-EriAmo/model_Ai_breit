# Kandydat #2 — cross-match pełny (`new_field_01`)

**UTC:** `20260916T061457Z` · **RA 142.5° · Dec 6.5°** · gal l=226.8 b=38.0
**FITS:** `desi_new_field_01_ra142.5_dec6.5_s128.fits` · **ckpt:** `most_desi_color.pt`

> **Nazwa:** **Asia** (2026-09-16, dedykacja Asi). Nie mylić z JoannaM.

## Score + ablacja

| cutout | p_cluster | abl Gaia r=4 Δp | p po masce | ctrl Δp |
|--------|----------:|----------------:|-----------:|--------:|
| stary | **0.9743** | -0.1738 | **0.8005** | -0.0000 |
| świeży | 0.9743 | -0.1738 | 0.8005 | -0.0000 |

Fresh sha=IDENTYCZNY (`adcb49b24024de5e` / `adcb49b24024de5e`).

Gaia: `GE:586076295021789056` r≈13.60 @ 9.692″.

**vs JoannaM:** tu po masce zostaje **~0.80**; JoannaM spadała do **~0.24**.

## Optyka / spektroskopia

| Katalog | Wynik |
|---------|--------|
| SIMBAD ≤60″ | **n=0** |
| SIMBAD ≤5′ | n=16 · nearest: SDSS J092958.22+063121.5 @ 85.77″ |
| LS DR10 ≤36″ | n=18, bright r<20: **1** |
| DESI DR1 ≤10′ | n=50 · nearest 63.22″ STAR z=5.718113574024049e-05 |

## X-ray / radio

| Katalog | ≤5′ | ≤15′ | nearest |
|---------|----:|-----:|---------|
| eROSITA eRASS1 | **0** | 5 | 1eRASS J092928.8+063137 @ 7.92' |
| NVSS | **4** | 9 | 093001+062733 @ 2.4912′ |
| 2RXS | — | 0 | — |

## Werdykt

- Spektrum w centrum cutoutu (SIMBAD 60″): **brak**
- DESI w cutoucie (<20″): **NIE**
- eROSITA ≤5′: **0** · NVSS ≤5′: **4**
- Flaga Most: **częściowo gwiazda + silny residual (~0.80)** — wart follow-upu, nie BH claim
- Fresh cutout: **reprodukuje**

JSON: `CROSSMATCH_C2.json` · alias FITS: `desi_c2_fresh_ra142.5_dec6.5_s128.fits`
