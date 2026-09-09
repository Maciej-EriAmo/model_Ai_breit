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
| `historia/phi_most_*.png` | wykresy |

Lab (poza tym katalogiem): `C:\Users\drwis\mcps\moje badania\WYNIKI_FINAL.md`, `RAPORT.md`.
