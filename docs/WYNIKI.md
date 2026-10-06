# PhiMost — skrót wyników empirycznych

Źródła szczegółowe: pliki `archiwum/historia/WYNIKI_*.md` (artefakty faz).  
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
| **Mendelejew FULL** (118, Most-only, Breit 2026-10-06) | above_floor (≥5×chance) | — | **0.575** | — | (bez SM; parity na P0) | **DONE** floor **3/3** · r@3=0.892 · 768 s |
| **Hold-out Na/K FULL** | cousin ≤3 / all seeds | — | — | — | — | **FAIL 2/2** · Na Li 29\|2 · K Na 19\|27 |
| **BF16 gate #2** | zero NaN w autocast bf16 | — | **0.578** | — | — | **PASS** · 16 ep FULL dump · finite-check |
| **Baselines FULL118** (5 seedów) | ΔSM r@1 ≥ 0.02 | **0.563** | **0.568** | **0.568** | **+0.005** | gate **0/5** · Most≈SM · wall 1.7× · RSS≈ · VRAM N/A (CPU) |
| **Tiny LM #1b** (~116k) | obie krzywe↓ | loss **0.664** | loss **0.637** | — | ΔCE SM−Most **+0.027** | **DONE** smoke · Most ≤ SM loss · 1.7× wolniejszy |
| **Medium LM #1c** (~3.3M, notatki, 2026-10-06) | obie krzywe↓ | loss **2.353** | loss **2.416** | — | **+0.063** | **DONE** · SM nieco niżej · Most 1.32× · korpus dłuższy niż 21.09 |
| **Medium LM WikiText-2** (~3.3M) | valid i test, cały plik | valid **1.604** · test **1.627** | valid **1.557** · test **1.580** | — | **−0.047** | **DONE** · Most lekko niżej, 2/2 seedy · nie duży model |
| **Lost-in-mid @2k** (~4.3M) | mid_dip mass / ent_mid | dip≈**0** · ent_mid **6.93** | dip **+0.001** · ent_mid **4.74** | — | hipoteza Most-lepszy-mid | **FAIL** · secret_acc=0 · [`LOST_IN_MIDDLE_PROBE.md`](../doswiadczenie/glify/LOST_IN_MIDDLE_PROBE.md) |
| **Knobs sweep** @2k | ent_mid ≥ SM ∧ dip ≤ SM | ent **6.927** | **ultra_wide 6.928** | — | strojenie γ/e/τ | **PASS** (ε) · baseline Most 3.15→6.93 · [`MOST_KNOBS_SWEEP.md`](../doswiadczenie/glify/MOST_KNOBS_SWEEP.md) |
| **Signed ΔE** @2k | directed/onesided vs square | SM **6.928** | ultra_wide_dir **6.928** · onesided 6.66 · directed 6.38 | — | znak ΔE | **PASS ε** (szerokie g); sam directed ≤ square · [`SIGNED_DELTA_ABLATION.md`](../doswiadczenie/glify/SIGNED_DELTA_ABLATION.md) |
| **Atoms ΔE** FULL118 | r@1 vs SM/square | **0.579** | **onesided 0.591** · square 0.567 · dir 0.577 | — | onesided | **PARTIAL** · ΔSM +0.012 · gate 1/2 · [`ATOMS_DELTA_ABLATION.md`](../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md) |
| **Hold-out onesided** | cousin≤3 / group top-3 | — | Na Li **10/6** (K@1/3) · K Na **57/59** | — | vs square 29/2 · 19/27 | **Na PARTIAL / K FAIL** · [`MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md) |
| **Kana P1b** (46 gojūon) | above_floor | ≈Most | ≈0.29 | ≈Most | ≈0 | above_floor 3/3; gate Most≫SM **0/3** |
| **Holo P5.1** holdout radical | Most−ZERO ≥+0.02 | ~0.02 | ~0.01 | ~0.01 | **~0** | **FAIL 0/8** (train 100%, holdout≈chance) |

### Doświadczenie nieba (osobny katalog)

| Eksperyment | Gate / metryka | Werdykt | Doc |
|-------------|----------------|---------|-----|
| DESI smoke (geometry) | Most−ZERO | **FAIL** (zachowane) | [`../archiwum/desi/WYNIKI_DESI.md`](../archiwum/desi/WYNIKI_DESI.md) |
| DESI demo / holdout | Most−ZERO | zob. pliki | [`../archiwum/desi/`](../archiwum/desi/) |
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
| `archiwum/historia/WYNIKI_FAZA_B.md` | classify Φ-drzewo |
| `archiwum/historia/WYNIKI_RECALL.md` | recall InfoNCE |
| `archiwum/historia/WYNIKI_FAZA_D.md` | dump offline |
| `archiwum/historia/WYNIKI_FAZA_E.md` | live/gen/λ_neural |
| `archiwum/historia/WYNIKI_NBODY.md` | solar + three-body |
| `doswiadczenie/glify/WYNIKI_GLYPHS.md` | glify → atomy (G0) |
| `doswiadczenie/glify/WYNIKI_GLYPHS_G1.md` | zdjęcie → atomy (G1) |
| `doswiadczenie/glify/WYNIKI_GLYPHS_G2.md` | parity Softmax\|Most\|ZERO na v2 |
| `doswiadczenie/glify/MENDELEJEW_P0.md` | parity Most≈SM + nonflat win |
| `doswiadczenie/glify/MENDELEJEW_FULL.md` | FULL118 Most-only · r@1≈0.57 · floor 3/3 |
| `doswiadczenie/glify/MENDELEJEW_BASELINES_COMPARE.md` | Most vs SM vs ZERO · 5 seedów · timing/RSS |
| `doswiadczenie/glify/ATOMS_DELTA_ABLATION.md` | FULL118 ΔE variants vs Softmax · onesided PARTIAL |
| `doswiadczenie/glify/SIGNED_DELTA_ABLATION.md` | signed ΔE @2048 mid · knobs |
| `doswiadczenie/glify/LM_TINY_1B.md` | tiny causal LM Most vs SM · #1b smoke |
| `doswiadczenie/glify/LM_MEDIUM_1C.md` | medium ~3.3M na notatkach z repo · #1c |
| `doswiadczenie/glify/LM_MEDIUM_WIKITEXT_HOLDOUT.md` | ten sam model, WikiText-2, valid i test |
| `doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK.md` | FULL hold-out Na/K — **FAIL** gen. grupy |
| `doswiadczenie/glify/MENDELEJEW_HOLDOUT_Na.md` | hold-out Na (mini) — negatyw |
| `doswiadczenie/glify/HOLO_P5_1.md` | P5.1 FAIL holdout + fixy seal |
| `doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md` | lingua 6/6 + mapa torów |
| `doswiadczenie/glify/KANA_P1B.md` | kana P1b Most≈SM |
| `doswiadczenie/glify/CV_BLURB_MOST_PARITY.md` | one-liner CV |
| `doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md` | dziennik lab / decyzje |
| `archiwum/desi/` | DESI / holdout / BH-AGN / audyt |
| `doswiadczenie/joannam/` | JoannaM + weryfikacja + forum EN |
| `archiwum/historia/phi_most_*.png` | wykresy faz A–E / n-ciał |
| `docs/GLYPHS.md` | metoda + teza (kanwa pod opis badawczy) |

Hub doświadczenia: [`../doswiadczenie/README.md`](../doswiadczenie/README.md) · mapa glify: [`../doswiadczenie/glify/README.md`](../doswiadczenie/glify/README.md).  
Lab (poza tym katalogiem): `C:\Users\drwis\mcps\moje badania\WYNIKI_FINAL.md`, `RAPORT.md`.
