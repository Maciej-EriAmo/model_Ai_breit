# Checklist dokumentacji doświadczeń

**Data przeglądu:** 2026-09-16  
**Cel:** wszystko ważne opisane; braki jawne, nie ukryte.

## Hub

| Pozycja | Stan |
|---------|------|
| `doswiadczenie/README.md` | OK (odświeżony 2026-09-16) |
| `doswiadczenie/glify/README.md` | OK (mapa torów) |
| Link z `docs/INDEX.md` | OK |
| `docs/STATUS.md` / `PLAN_NEXT.md` / `WYNIKI.md` | OK (dogonione do Mendelejew + lingua + P5.1) |
| Rozdział w `docs/WYNIKI.md` (DESI / JoannaM / Mendelejew) | OK |
| `archiwum/historia/` = tylko A–E / n-ciał / learn (bez nieba/glifów) | OK |

## JoannaM (`joannam/`)

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Opis kandydata + współrzędne | OK | `JOANNAM.md` |
| Score Most + model ckpt | OK | p_cluster≈0.980; `most_desi_color.pt` |
| Lista kandydatów JSON | OK | `JOANNAM_CANDIDATES.json` |
| Ostrzeżenie: nie potwierdzona BH | OK | |
| Ścieżki cutout / preview | OK | FITS/PNG lokalnie, gitignore |
| Checklist weryfikacji katalogowej | OK | `WERYFIKACJA.md` |
| Post EN na forum | OK | `FORUM_EN.md` |
| Cross-match SDSS/DESI spektroskopia | **DONE 2026-09-16** | **NO MATCH** — 0 Spec SDSS (5′) · 0 DESI DR1 (~20′) · LS 37 fot. · [`joannam/CROSSMATCH.md`](joannam/CROSSMATCH.md) |
| X-ray / radio cross-match | **DONE 2026-09-16** | eROSITA/NVSS ≤5′ = **0**; najbliższe X@7.4′ · NVSS@~5.8′ |
| Ablacja maski gwiazdy @8.8″ | **DONE 2026-09-16** | TAK odpowiedzialna — Δp≈−0.40…−0.56; kontrole ~0 · [`ABLATION_STAR_MASK.md`](joannam/ABLATION_STAR_MASK.md) |
| Świeży cutout (reprodukcja) | **DONE 2026-09-16** | sha identyczny · p=0.797 · ablacja replikuje · [`REPRODUCE_FRESH_CUTOUT.md`](joannam/REPRODUCE_FRESH_CUTOUT.md) |
| **Asia** (`new_field_01`) | **DONE + nazwana** | RA142.5 Dec+6.5 · p=0.974→0.80 · [`ASIA.md`](joannam/ASIA.md) · [`CROSSMATCH_C2.md`](joannam/CROSSMATCH_C2.md) |

## DESI / niebo (`desi/`)

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Smoke / wynik bazowy | OK | `WYNIKI_DESI.md` (FAIL gate — zachowane) |
| Demo-only | OK | `WYNIKI_DESI_DEMO.md` |
| Holdout | OK | `WYNIKI_DESI_HOLDOUT.md` |
| BH/AGN score | OK | `WYNIKI_BH_AGN_SCORE.md` + JSON |
| Audyt B1/B2 + fixy | OK | `AUDYT_2026-09-10.md` |
| Tor aktywny | **zawieszony** | seal DONE; skala / gate nadal FAIL |

## Glify / transkodowanie (`glify/`)

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Wyniki G0 | OK | `WYNIKI_GLYPHS.md` (stary kontrakt — przed v2) |
| Wyniki G1 | OK | `WYNIKI_GLYPHS_G1.md` |
| Labnote | OK | `LABNOTE_GLYPHS_2026-09-10.md` |
| Kanon produktu TRANSCODE | OK | `docs/TRANSCODE.md` + stub `glify/TRANSCODE.md` |
| Asymetria lat=słowo / CJK=znak | OK | `transcode/` · `glyph_unit_v2` |
| Gate: nie kaleka vs Softmax | **PASS lab (best-seed)** | `WYNIKI_GLYPHS_G2.md` — seed0 +0.022; seed1 FAIL |
| Classical Softmax loader | **TODO** | dziś Softmax-twin-on-Φ (TRANSCODE §2) |
| Multi-seed parity | **otwarte** | bez claimu zewnętrznego |
| Testy kontraktu | OK | `doswiadczenie/testy/test_transcode_contract.py` |

## Holo-Φ

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| P0–P4a notatki | OK | `HOLO_P*.md` |
| Plan P5 | OK | `docs/HOLO_P5_*` — status dogoniony |
| P5.1 wynik | OK | **FAIL 0/8** · `HOLO_P5_1.md` |
| P5.2+ | wstrzymane | brak sygnału holdout |

## Lingua / hologram-znaki

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Warstwa 6/6 TODO | **DONE** | `AUDYT_HOLOGRAM_ZNAKI.md` |
| Testy | OK | `python -m pytest doswiadczenie/testy/test_karmazyn_lingua.py` 35/35 |
| Kana P1 / P1b | OK (wyniki) | P1 negatyw; P1b Most≈SM |
| Raport koncepcyjny w git | OK | `docs/hologram-znaki-raport.md` |
| `observe_with_confidence` długie sekwencje | otwarte | sim≈0.1, nieosobno |

## Mendelejew / struktura

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| P0 + smoke | OK | parity Most≈SM · nonflat 3/3 |
| Hold-out Na (mini) | OK | **negatyw** — lokalne bindy, nie grupa |
| CV blurb | OK | `CV_BLURB_MOST_PARITY.md` |
| Dump FULL 118 | OK | `dumps/from_mendelejew_full` |
| Run FULL `--most-only` | **DONE** | mean r@1=0.567 · floor 3/3 · `MENDELEJEW_FULL.md` |
| Hold-out Na/K na FULL (square) | **FAIL 2/2** | `MENDELEJEW_HOLDOUT_FULL_NaK.md` |
| Hold-out Na/K **onesided** | **Na PARTIAL / K FAIL** | Li 10/6 (K top-3); Na 57/59 · `MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md` |
| hologram-E overlay | otwarte | period×group×block |

---

## Luki 2026-09-16 (co mogliśmy pominąć)

1. **STATUS / INDEX / PLAN / WYNIKI były zatrzymane na 2026-09-11** — Mendelejew P0, lingua DONE, Holo P5.1 FAIL nie były w kanonie nawigacji *(naprawione w tej sesji)*.
2. **`HOLO_P5_PLAN_IMPLEMENTACJI.md` mówił „kod jeszcze nie napisany”** mimo istniejącego P5.1 *(naprawione)*.
3. **`PLAN_NEXT.md` trzymał G2 jako NEXT** mimo PASS best-seed *(naprawione)*.
4. **Brak mapy `glify/README.md`** przy ~40 plikach w katalogu *(dodane)*.
5. ~~**`hologram-znaki-raport.md` poza gitem**~~ — przeniesiony do `docs/`.
6. **Classical Softmax loader** — nadal TODO; łatwo przeoczyć i mówić „Softmax klasyczny”, gdy to twin-on-Φ.
7. ~~**JoannaM weryfikacja pełna**~~ — **DONE 2026-09-16:** NO MATCH katalogów; flaga = gwiazda Gaia; fresh cutout reprodukuje score.
8. **Hold-out Na negatyw** był w SE/notatce, ale nie w `docs/WYNIKI.md` *(dodane)*.
9. **DESI „zawieszone”** vs seal DONE — status aktywności nie był w hubie doświadczenia.
10. **Qumran / hologram-E** — wzmianki w SE/JUTRO bez miejsca w INDEX *(dodane do kolejki STATUS)*.

## Świadomie poza tym katalogiem

- Fazy A–E, n-ciał → `archiwum/historia/` + `docs/WYNIKI.md`
- Kanon pamięci M1/M2/M3 → `docs/PAMIEC_I_KARMIENIE.md`
- Ciężkie FITS/PNG → `archiwum/ZdjęcieW/demo/` (nie w git)
