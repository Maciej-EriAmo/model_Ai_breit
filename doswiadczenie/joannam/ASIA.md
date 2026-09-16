# Asia

**Nazwa nadana przez Mazura** — dedykacja Asi.  
**Data:** 2026-09-16  
**Status:** kandydat optyczny (Most + DESI Legacy), **nie** potwierdzona czarna dziura.  
**Poprzednik w torze:** JoannaM (inny mechanizm flagi).

## Współrzędne

| | |
|--|--|
| **RA** | 142.5° |
| **Dec** | +6.5° |
| Survey | DESI Legacy Surveys `ls-dr10` |
| Cutout | 128×128 px, pasma `grz`, pixscale 0.262″ |
| Plik | `ZdjęcieW/demo/desi_new_field_01_ra142.5_dec6.5_s128.fits` |
| Fresh alias | `ZdjęcieW/demo/desi_c2_fresh_ra142.5_dec6.5_s128.fits` |
| Podgląd | `ZdjęcieW/demo/Asia.png` |

## Dlaczego flaga

- Model: `checkpoints/most_desi_color.pt` (content = luminancja, **T = barwa g−r**).
- Score: **p_cluster ≈ 0.974**, T_mean ≈ 0.591, \|dE\| ≈ 0.555.
- Po masce Gaia @9.7″ (r≈13.6): nadal **p ≈ 0.800** (Δ≈−0.17) — **nie** gwiazda-only jak JoannaM.
- Fresh cutout: sha identyczny, score/ablacja reprodukowalne.

## Cross-match (2026-09-16)

Pełny raport: [`CROSSMATCH_C2.md`](CROSSMATCH_C2.md).

| Warstwa | Wynik |
|---------|--------|
| SIMBAD ≤60″ | **0** |
| DESI w cutoucie (<20″) | **NIE** (nearest ~63″) |
| eROSITA ≤5′ | **0** (nearest @ 7.9′) |
| NVSS ≤5′ | **4** (nearest @ **2.49′**, poza cutoutem) |
| Ablacja / fresh | residual wysoki · reprodukcja OK |

**Werdykt:** anomalia optyczna Mostu z silnym residualem po masce gwiazdy; **BH/AGN niepotwierdzone**.

## Cytowanie wewnętrzne

Komora SE `Most_Lorentz`: fact Asia.  
Weryfikacja: [`CROSSMATCH_C2.md`](CROSSMATCH_C2.md) · screening: [`NEXT_CANDIDATE.md`](NEXT_CANDIDATE.md).  
Hub: [`../README.md`](../README.md).
