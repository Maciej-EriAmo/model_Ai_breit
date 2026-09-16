# JoannaM

**Nazwa nadana przez Mazura** — dedykacja żonie Joannie.  
**Data:** 2026-09-10  
**Status:** kandydat optyczny (Most + DESI Legacy), **nie** potwierdzona czarna dziura.

## Współrzędne

| | |
|--|--|
| **RA** | 120.000° |
| **Dec** | +5.000° |
| Survey | DESI Legacy Surveys `ls-dr10` |
| Cutout | 128×128 px, pasma `grz`, pixscale 0.262″ |
| Plik | `ZdjęcieW/demo/desi_grid_004_ra120.0_dec5.0_s128.fits` |
| Podgląd | `ZdjęcieW/demo/JoannaM.png` |

## Dlaczego flaga

- Skan `hunt_joannam.py`: 117 pól/grid (bez znanych AGN/klastrów z nazwy).
- Model: `checkpoints/most_desi_color.pt` (content = luminancja, **T = barwa g−r**).
- Score: **p_cluster ≈ 0.980**, T_mean ≈ 0.558, \|dE\| ≈ 0.484.
- W treningu pozycja była w siatce „field”; Most traktuje cutout jak silnie „klastrowy / gorący”.

## Uwaga naukowa

To **anomalia optyczna według Mostu**, nie odkrycie horyzontu zdarzeń.  
DESI Legacy pokazuje światło (galaktyki / jądro / struktura), nie samą czarną dziurę.

## Cross-match (2026-09-16)

Pełny raport: [`CROSSMATCH.md`](CROSSMATCH.md).

| Warstwa | Wynik |
|---------|--------|
| SDSS DR17 Spec / DESI DR1 zpix | **brak spektrum** w okolicy (5′ / ~20′) |
| SIMBAD ≤ 60″ | pusto |
| LS DR10 tractor ≤ 36″ | 37 źródeł; jasna gwiazda Gaia @ **8.8″** (r≈17.7, π≈0.34 mas) |
| eROSITA eRASS1 / NVSS ≤ 5′ | **0** (najbliższe X @ 7.4′, radio @ ~5.8′) |
| Ablacja maski Gaia @8.8″ | **gwiazda odpowiedzialna** — p_cluster 0.80→0.24 (r=4 px); kontrole Δ≈0 |
| Świeży cutout Legacy | **reprodukcja OK** — sha identyczny, p=0.797, ablacja Δ=−0.556 |

**Werdykt:** spektroskopia + X-ray + radio = **NO MATCH**; flaga Most w dużej mierze od **jasnej gwiazdy Gaia**, nie od ukrytego AGN/BH. Weryfikacja katalogowa + ablacja + fresh cutout — **domknięte**.

## Cytowanie wewnętrzne

Komora SE `Most_Lorentz`: fact JoannaM.  
Kandydaci: `doswiadczenie/joannam/JOANNAM_CANDIDATES.json`.  
Weryfikacja: [`WERYFIKACJA.md`](WERYFIKACJA.md) · cross-match: [`CROSSMATCH.md`](CROSSMATCH.md) · post forum: [`FORUM_EN.md`](FORUM_EN.md).  
Hub doświadczenia: [`../README.md`](../README.md).
