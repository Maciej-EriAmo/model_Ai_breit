# PhiMost — skrót wyników empirycznych

Źródła szczegółowe: pliki `historia/WYNIKI_*.md` (artefakty faz).  
Kanon silnika: **softplus × 2**, Softmax twin + **ZERO**, bez EMA-S.

## Tabela zbiorcza

| Eksperyment | Gate | Softmax | Most | ZERO | Δ Most−SM | Werdykt |
|-------------|------|---------|------|------|-----------|---------|
| **B** Φ-drzewo (n=1…8) | +3 pkt ≥2 n | ~0.36 | **0.39–0.65** | ~Softmax | **+3.5…+17** | **PASS 4/4** |
| **C** recall r@1 (N=24) | +2 pkt r@1 | 0.284 | **0.995** | 0.286 | **+71 pkt** | **PASS** |
| **D** dump textfile_phi | +2 pkt r@1 | 0.388 | **0.410** | ≈Most* | **+2.3 pkt** | **PASS**† |
| **E** flagi / gen / live | guardy | — | — | — | tracer pomaga MSE | **PASS** |
| **Solar** star/rocky/giant | +3 pkt | 0.356 | **0.621** | 0.367 | **+26.5 pkt** | **PASS** (po fix Moon) |
| **3 ciała** Verlet | +3 pkt | 0.523 | **0.742** | 0.506 | **+21.9 pkt** | **PASS** (po fix dim5) |
| **G0** glify lat+hira (recall) | +2 pkt r@1 | 0.231 | **0.259** | 0.227 | **+2.8 pkt** | **PASS** (seed 42) |
| **G1** photo→atoms (demo) | +2 pkt r@1 | 0.221 | **0.265** | 0.231 | **+4.4 pkt** | **PASS** (seed 0) |
| **G2** glyph_unit_v2 parity | +2 pkt r@1 | 0.309 | **0.331** | 0.308 | **+2.2 pkt** | **PASS** best-seed 0 (1/3; seed1 FAIL) |
| **Mendelejew P0** (8 el., N=72) | nonflat / parity | 0.445 | **0.427** | 0.444 | **−0.018** (≈SM) | **parity + nonflat 3/3** · flat 0.329 |
| **Kana P1b** (46 gojūon) | above_floor | ≈Most | ≈0.29 | ≈Most | ≈0 | above_floor 3/3; gate Most≫SM **0/3** |
| **Holo P5.1** holdout radical | Most−ZERO ≥+0.02 | ~0.02 | ~0.01 | ~0.01 | **~0** | **FAIL 0/8** (train 100%, holdout≈chance) |

### Doświadczenie nieba (osobny katalog)

| Eksperyment | Gate / metryka | Werdykt | Doc |
|-------------|----------------|---------|-----|
| DESI smoke (geometry) | Most−ZERO | **FAIL** (zachowane) | [`../doswiadczenie/desi/WYNIKI_DESI.md`](../doswiadczenie/desi/WYNIKI_DESI.md) |
| DESI demo / holdout | Most−ZERO | zob. pliki | [`../doswiadczenie/desi/`](../doswiadczenie/desi/) |
| **JoannaM** RA120 Dec+5 | p_cluster≈0.980 | kandydat optyczny, **nie** BH | [`../doswiadczenie/joannam/JOANNAM.md`](../doswiadczenie/joannam/JOANNAM.md) |

\* Na dumpie D Most ZERO ≈ Most (słabsza przewaga tracera niż w C) — odnotowane w audycie.  
† D: Most bije też skalarne R (~+12 pkt r@1).

## Negatywy (świadomie zachowane)

| Hipoteza | Wynik | Doc |
|----------|-------|------|
| Soft EMA-S / Most+S | regresja vs czysty Most | lab `RAPORT.md` / `WYNIKI_FINAL.md` |
| Obs w ewolucji S | FAIL 0/4 (+2 pkt) | lab `RAPORT.md` §10 |

## Controles obowiązkowe

1. **Softmax twin** — ten sam szkielet, uwagi bez zamka Lorentza.  
2. **ZERO tracer** — Most bez τ ≈ Softmax: **dowód, że tracer jest warunkiem mostu**, nie handicapem.  
3. **Flat payload** — Unix fiction gorsza niż Φ (C/D).

> Kanon: bez niezerowego \(\Delta E\) most nie powstaje / jest niestabilny. τ ≠ bonus do score.

## Pliki

| Plik | Treść |
|------|--------|
| `historia/WYNIKI_FAZA_B.md` | classify Φ-drzewo |
| `historia/WYNIKI_RECALL.md` | recall InfoNCE |
| `historia/WYNIKI_FAZA_D.md` | dump offline |
| `historia/WYNIKI_FAZA_E.md` | live/gen/λ_neural |
| `historia/WYNIKI_NBODY.md` | solar + three-body |
| `doswiadczenie/glify/WYNIKI_GLYPHS.md` | glify → atomy (G0) |
| `doswiadczenie/glify/WYNIKI_GLYPHS_G1.md` | zdjęcie → atomy (G1) |
| `doswiadczenie/glify/WYNIKI_GLYPHS_G2.md` | parity Softmax\|Most\|ZERO na v2 |
| `doswiadczenie/glify/MENDELEJEW_P0.md` | parity Most≈SM + nonflat win |
| `doswiadczenie/glify/MENDELEJEW_HOLDOUT_Na.md` | hold-out Na — negatyw |
| `doswiadczenie/glify/HOLO_P5_1.md` | P5.1 FAIL holdout + fixy seal |
| `doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md` | lingua 6/6 + mapa torów |
| `doswiadczenie/glify/KANA_P1B.md` | kana P1b Most≈SM |
| `doswiadczenie/glify/CV_BLURB_MOST_PARITY.md` | one-liner CV |
| `doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md` | dziennik lab / decyzje |
| `doswiadczenie/desi/` | DESI / holdout / BH-AGN / audyt |
| `doswiadczenie/joannam/` | JoannaM + weryfikacja + forum EN |
| `historia/phi_most_*.png` | wykresy faz A–E / n-ciał |
| `docs/GLYPHS.md` | metoda + teza (kanwa pod opis badawczy) |

Hub doświadczenia: [`../doswiadczenie/README.md`](../doswiadczenie/README.md) · mapa glify: [`../doswiadczenie/glify/README.md`](../doswiadczenie/glify/README.md).  
Lab (poza tym katalogiem): `C:\Users\drwis\mcps\moje badania\WYNIKI_FINAL.md`, `RAPORT.md`.
