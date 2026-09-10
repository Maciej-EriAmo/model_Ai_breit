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
DESI pokazuje światło (galaktyki / jądro / struktura), nie samą czarną dziurę.  
Dalsza weryfikacja: katalogi SDSS/DESI, spektroskopia, cross-match X-ray/radio.

## Cytowanie wewnętrzne

Komora SE `Most_Lorentz`: fact JoannaM.  
Kandydaci: `historia/JOANNAM_CANDIDATES.json`.
