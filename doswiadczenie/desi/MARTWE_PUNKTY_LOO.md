# Martwe punkty LOO-CV (bh-detekcja) — audyt cutoutów

**Data:** 2026-09-15  
**Pytanie:** dlaczego `3c345` / `mrk421` / `ton618` są systematycznie NIE znajdowane przez Softmax, Most i ZERO w LOO-CV (11 foldów)?  
**Hipoteza wejściowa:** SNR / rozdzielczość cutoutu, nie model.

## Skrót wniosku

**Tak — to głównie dane/reprezentacja cutoutu, nie „ślepy model”.**  
Trzy martwe punkty padają z **różnych** powodów jakościowych; wspólny mianownik: nie wyglądają jak kompaktowe, jasne jądra, na których LOO się uczy (wzór „zawsze złapanych”).

Dodatkowo: w zestawie „zawsze złapanych” jest **`cena` z pustym FITS** (poza footprintem Legacy) — fałszywy sukces.

## LOO (przypomnienie)

| model | recall | precision | F1 |
|-------|--------|-----------|-----|
| Softmax | 0.545 (6/11) | 0.462 | 0.500 |
| Most | 0.636 (7/11) | 0.304 | 0.412 |
| ZERO | 0.545 (6/11) | 0.333 | 0.414 |

- Martwe (0/3 modele): `3c345`, `mrk421`, `ton618`
- Zawsze złapane (wg faktu): `3c48`, `cena`, `cta102`, `oj287`

Skrypt: scratchpad `check_bh_loo.py` (nie w repo).

## Surowe FITS (HDU0, Legacy grz)

| obiekt | grupa | max flux | mean | komentarz |
|--------|-------|----------|------|-----------|
| **ton618** | DEAD | **0.054** | 9e-5 | prawie szum |
| **3c345** | DEAD | 4.4 | 0.0056 | słabsze jądro |
| **mrk421** | DEAD | 11.9 | **0.58** | jasne, ale rozlane |
| 3c48 | CAUGHT | 13.7 | 0.024 | kompaktowe |
| cta102 | CAUGHT | 14.2 | 0.016 | kompaktowe |
| oj287 | CAUGHT | 20.1 | 0.060 | kompaktowe |
| **cena** | „CAUGHT” | **0** | **0** | pusty cutout |

## Morfologia po `normalize_log` (NPZ)

| obiekt | r50 (px) | core/ring | sat core | snr_core_max |
|--------|----------|-----------|----------|--------------|
| ton618 | **52** | **1.2** | 0.00 | **2.7** |
| 3c345 | **29** | 7.0 | 0.90 | 232 |
| mrk421 | **22** | **2.3** | 1.00 | 170 |
| 3c48 | 12 | 8.1 | 1.00 | 421 |
| cta102 | 14 | 10.4 | 1.00 | 301 |
| oj287 | 9 | 7.8 | 0.71 | 626 |
| cena | — | — | — | 0 (zerowy obraz) |

`r50` = promień obejmujący ~50% dodatniego fluxu (proxy half-light).  
`core/ring` = średni flux w r≤4 vs pierścień 8–24 px.

## Interpretacja per obiekt

### 1. `ton618` — za słaby cutout (hipoteza SNR: TAK)

- Raw max ~40×–300× niższy niż u złapanych kwazarów.
- Brak koncentracji (`core/ring≈1`, `r50≈52`).
- Klasyczny przypadek: obiekt „jest w katalogu”, ale w tym cutoucie Legacy praktycznie nie widać jądra.
- Zgodne z wcześniejszym unsupervised score: `p_cluster=0.073` → field.

### 2. `3c345` — słabsze / bardziej rozmyte jądro

- Realny sygnał, ale peak raw ~3× niższy niż u `3c48`/`cta102`.
- `r50≈29` vs ~9–14 u zawsze złapanych → mniej „point-source AGN”.
- Unsupervised już dawał `p_cluster=0.19` (field).
- Model uczy się wzorca kompaktowego jądra; ten cutout jest na brzegu dystrybucji.

### 3. `mrk421` — jasny, ale inna morfologia (nie „martwy SNR”)

- Jasny blazar/BL Lac z wypełnionym tłem (`nonzero≈99%` pikseli, niski `core/ring=2.3`).
- Nie jest za ciemny — jest **inny** niż kompaktowe kwazary, na których LOO wygrywa.
- Unsupervised: `p_cluster=0.80` (cluster) — inna oś niż supervised bh-vs-reszta; stąd rozjazd „score gorący / LOO martwy”.

### 4. `cena` — zanieczyszczenie zbioru

- Komentarz w `pobierz_bh_agn.py`: *„Cen A — może poza footprint”*.
- FITS i NPZ = same zera. „Zawsze złapany” nie liczy się jako sukces detekcji AGN.
- Po wyrzuceniu `cena` zestaw „pewnych hitów” to: `3c48`, `cta102`, `oj287` (kompaktowe, wysoki peak).

## Odpowiedź na pytanie A

| Czy to model? | Częściowo tylko jako *inductive bias* (uczy się kompaktowych jąder). |
| Czy to cutout/SNR? | **Tak dla ton618; częściowo dla 3c345; dla mrk421 to morfologia/host, nie SNR.** |
| Co robić dalej? | (1) wyrzucić / oznaczyć `cena` jako invalid; (2) dla ton618: inny layer/głębszy cutout albo wykluczyć z bh-setu; (3) LOO raportować z filtrem `max(raw)>τ` / `snr_core_max>τ`; (4) osobna ablacja: tylko kompaktowe vs extended hosts. |

## Artefakty

- Analiza: `model Ai/_analyze_dead_bh.py`, `model Ai/_check_fits_vs_npz.py`
- LOO: `model Ai/check_bh_loo.py` (filtr empty + `min_snr`, default 20)
- Re-LOO po filtrze → `WYNIKI_LOO_BH_FILTERED.md` / `loo_bh_filtered.json` (n=9: bez `cena`/`ton618`)

## Re-LOO po filtrze (2026-09-15)

Odrzucone automatycznie: `cena` (empty), `ton618` (`snr_core_max=2.68 < 20`).

| model | recall n=9 | F1 n=9 | recall n=11 | F1 n=11 |
|-------|------------|--------|-------------|---------|
| Softmax | 0.556 (5/9) | 0.417 | 0.545 | 0.500 |
| Most | 0.556 (5/9) | **0.476** | 0.636 | 0.412 |
| ZERO | **0.667** (6/9) | 0.444 | 0.545 | 0.414 |

- **`3c345` nadal martwy (0/3)** — twardy przypadek morfologii, nie artefakt filtra.
- **`mrk421` już nie systematycznie martwy**: Softmax miss, ale Most+ZERO łapią → po usunięciu śmieci z traina Most/ZERO generalizują lepiej na extended host.
- Consensus always-caught (3/3): `3c273`, `cta102`, `oj287`.
