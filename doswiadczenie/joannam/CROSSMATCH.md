# JoannaM — cross-match katalogowy (2026-09-16)

**Cel:** RA **120.000°** · Dec **+5.000°** (cutout `desi_grid_004_ra120.0_dec5.0_s128`, ~128×128 @ 0.262″/px ≈ **±17″** od środka; półprzekątna ~24″).  
**Pytanie:** czy jest znane spektrum / obiekt katalogowy tłumaczący flagę Most `p_cluster≈0.980`?

## Werdykt (skrót)

| Warstwa | Wynik |
|---------|--------|
| **Spektroskopia SDSS DR17** (`SpecObjAll`, r ≤ 5′) | **0 obiektów** |
| **Spektroskopia DESI DR1** (`desi_dr1.zpix`, box ~20′) | **0 obiektów** |
| **SIMBAD** r ≤ 60″ | **0 obiektów** |
| **Legacy Surveys DR10 tractor** r ≤ 36″ | **37 źródeł fotometrycznych** (głównie słabe) |
| **X-ray eROSITA eRASS1** (r ≤ 5′) | **0** · najbliższe @ **7.4′** |
| **X-ray 2RXS / Chandra CSC** (r ≤ 15′) | **0** |
| **Radio NVSS** (r ≤ 5′) | **0** · najbliższe @ **~5.8′** |
| **Radio FIRST / VLASS** (r ≤ 15′) | **0** (FIRST/VLASS) |
| **BH / AGN** | **niepotwierdzone** — brak z, brak X-ray/radio w cutoucie |

**PASS katalogowy (optyka + X/radio):** tak.  
**Identyfikacja BH/AGN:** **FAIL / NO MATCH** na wszystkich warstwach publicznych.

Most nadal widzi **anomalię optyczną / barwę**; to **nie** jest potwierdzenie czarnej dziury.

## 1. Spektroskopia

### SDSS DR17
SkyServer SQL: `SpecObjAll` w promieniu 5′ od (120, +5) → **No entries**.

### DESI DR1 (`zpix` via Astro Data Lab)
Box `mean_fiber_ra/dec` ±0.1° (~6′) oraz ±0.35° (~20′) → **0 wierszy**.  
Wniosek: w publicznym DR1 **nie ma włókna DESI** w bezpośredniej okolicy JoannyM (poza footprintem spektroskopowym DR1 albo nieobserwowane).

### SIMBAD (nazwane)
| Promień | N | Uwagi |
|---------|---|--------|
| ≤ 60″ | 0 | pusty środek cutoutu |
| ≤ 10′ | 43 | najbliższe **poza** cutoutem |

Najbliższe nazwane (poza FOV cutoutu):

| dist | ID | typ |
|-----:|----|-----|
| 174″ | Zel 0757+051 | Rad |
| 192″ | SDSS J075947.23+045938.4 | G |
| 240″ | CRTS J075959.7+050400 | EB\* |
| 249″ | Gaia DR3 3095367640144517632 | AGN |

Żaden z nich nie leży w samym cutoucie s128 (~33″ bok).

NED Near Position: timeout / 500 — **nie użyte** jako dowód.

## 2. Fotometria w cutoucie (LS DR10 tractor, r ≤ 36″)

**N = 37** · typy: PSF 18 · REX 15 · DEV 2 · EXP 2.

Źródła jasne (`r < 20`):

| dist | typ | g | r | z | Gaia / ref |
|-----:|-----|--:|--:|--:|------------|
| **8.80″** | PSF | 18.15 | **17.68** | 17.52 | `GE:3095364135451155712`, π≈0.34 mas → **gwiazda** |
| 16.99″ | PSF | 20.10 | 19.21 | 18.79 | `GE:3095364101091415936` |

Najbliżej środka (słabe):

| dist | typ | r |
|-----:|-----|--:|
| 5.12″ | PSF | 22.79 |
| 5.87″ | REX | 23.47 |
| 7.14″ | REX | 23.61 |

**Interpretacja:** pole jest **fotometrycznie zatłoczone** (słabe galaktyki/REX + ≥1 jasna gwiazda Gaia). Flaga Most (barwa g−r → T) może reagować na **strukturę/barwę pola**, nie na pojedynczy obiekt ze spektrum.

## 3. X-ray / radio (2026-09-16)

Galaktyczne: **l≈216.3° · b≈+17.5°** → w publicznym półkulowym footprintcie **eROSITA-DE**.

### X-ray

| Katalog | ≤5′ | ≤15′ | Najbliższe |
|---------|----:|-----:|------------|
| **eROSITA eRASS1** (`J/A+A/682/A34`) | **0** | 7 | `1eRASS J080025.8+045621` @ **7.40′** |
| ROSAT 2RXS | 0 | 0 | — |
| Chandra CSC 2.1 | 0 | 0 | — |

Najbliższe eRASS1 (poza cutoutem):

| dist | IAUName | RA | Dec |
|-----:|---------|---:|----:|
| 7.40′ | 1eRASS J080025.8+045621 | 120.108 | +4.939 |
| 8.63′ | 1eRASS J080016.2+050737 | 120.068 | +5.127 |
| 8.95′ | 1eRASS J075949.6+045125 | 119.957 | +4.857 |

### Radio

| Katalog | ≤5′ | ≤15′ | Najbliższe |
|---------|----:|-----:|------------|
| **NVSS** | **0** | 6 | `NVSS 080022+045826` @ **348″ (~5.8′)**, S1.4≈3.1 mJy |
| FIRST | 0 | 0 | — |
| SIMBAD Rad | — | — | `Zel 0757+051` @ 174″ (coords **przybliżone** `~`; **nie** w NVSS ≤5′) |

**Werdykt X/radio:** **NO MATCH** w skali cutoutu / 5′. Brak twardego counterpartu AGN/BH.

Artefakty: `CROSSMATCH_XRAY_RADIO.json` · `_tmp_erosita.json` · `_query_xray_radio2.py` · `_query_erosita.py`.

## 4. Artefakty optyczne / skrypt

| Plik | Rola |
|------|------|
| `CROSSMATCH_RA120_DEC5.json` | surowy log zapytań optycznych |
| `_tmp_ls_box.json` | 37 źródeł LS w 36″ |
| `_tmp_desi_box.json` / `_tmp_desi_near.json` | puste listy DESI |
| `crossmatch_joannam.py` | driver (SIMBAD/NED/SDSS/TAP) |
| `_query_box.py` · `_query_desi_near.py` | zapytania box Data Lab |

## 5. Ablacja maski gwiazdy (2026-09-16) — DONE

Raport: [`ABLATION_STAR_MASK.md`](ABLATION_STAR_MASK.md) · JSON: `ABLATION_STAR_MASK.json`.

| Warunek | p_cluster | Δp |
|---------|----------:|---:|
| Baseline (ten sam FITS+ckpt) | **0.797** | — |
| Maska Gaia @8.8″ · r=3 px | 0.401 | **−0.396** |
| Maska Gaia @8.8″ · r=4 px | 0.241 | **−0.556** |
| Obie gwiazdy Gaia · r=4 px | 0.080 | **−0.717** |
| Kontrole kątowe (r=3 px) | ≈0.797 | **~0** |

**Werdykt:** gwiazda Gaia @ **8.8″** (r≈17.7) **jest odpowiedzialna** za wysoką flagę Most — maska zrzuca `p_cluster` o dziesiątki punktów, kontrole w tej samej odległości od środka nie.

> Uwaga: baseline dziś **0.797**, nie 0.980 z pierwotnego skanu `hunt_joannam` — ten sam FITS/ckpt, możliwy dryf preprocessingu/seal; **Δ ablacji** jest miarodajna.

## 6. Świeży cutout (2026-09-16) — DONE

Raport: [`REPRODUCE_FRESH_CUTOUT.md`](REPRODUCE_FRESH_CUTOUT.md).

Niezależny download Legacy (`ls-dr10`, 128², pixscale 0.262″) na ten sam RA/Dec:

| | sha256[:16] | p_cluster | abl r=4px Δp |
|--|-------------|----------:|-------------:|
| stary `desi_grid_004_…` | `c572001e20e8cbb3` | 0.797 | −0.556 |
| świeży `desi_joannam_fresh_…` | `c572001e20e8cbb3` | 0.797 | −0.556 |

**Bajty identyczne** (serwer Legacy zwraca ten sam produkt) · Δp = 0 · ablacja gwiazdy **replikuje się**.  
Alias: `ZdjęcieW/demo/desi_joannam_fresh_ra120.0_dec5.0_s128.fits`.

## 7. Co dalej (nieblokujące)

1. ~~eROSITA / radio~~ — **DONE** (NO MATCH ≤5′).  
2. ~~Ablacja gwiazdy @8.8″~~ — **DONE** (gwiazda odpowiedzialna).  
3. ~~Świeży cutout~~ — **DONE** (identyczny produkt + score).  
4. Gdy DESI/SDSS dołoży spektrum w tym polu — powtórzyć cone.

## Cytowanie

Komora SE `Most_Lorentz` · kandydat [`JOANNAM.md`](JOANNAM.md) · checklist [`WERYFIKACJA.md`](WERYFIKACJA.md).
