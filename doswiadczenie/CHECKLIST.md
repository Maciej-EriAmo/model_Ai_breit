# Checklist dokumentacji doświadczeń

**Data przeglądu:** 2026-09-11  
**Cel:** wszystko ważne opisane; braki jawne, nie ukryte.

## Hub

| Pozycja | Stan |
|---------|------|
| `doswiadczenie/README.md` | OK |
| Link z `docs/INDEX.md` | OK (po porządkach) |
| Rozdział w `docs/WYNIKI.md` (DESI / JoannaM) | OK (po porządkach) |
| `historia/` = tylko A–E / n-ciał / learn (bez nieba/glifów) | OK (po przenosinach) |

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
| Cross-match SDSS/DESI spektroskopia | **TODO** | jeszcze nie wykonane |
| X-ray / radio cross-match | **TODO** | opcjonalnie |
| Notatka „co Most *nie* dowodzi” w INDEX | OK | hub + JOANNAM |

## DESI / niebo (`desi/`)

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Smoke / wynik bazowy | OK | `WYNIKI_DESI.md` (FAIL gate — zachowane) |
| Demo-only | OK | `WYNIKI_DESI_DEMO.md` |
| Holdout | OK | `WYNIKI_DESI_HOLDOUT.md` |
| BH/AGN score | OK | `WYNIKI_BH_AGN_SCORE.md` + JSON |
| Audyt B1/B2 + fixy | OK | `AUDYT_2026-09-10.md` |
| Skrypty wskazują `doswiadczenie/desi/` | OK | po porządkach |
| Oddzielenie od toru glifów w docs | OK | GLYPHS / PLAN_NEXT |

## Glify / transkodowanie (`glify/`)

| Pozycja | Stan | Uwagi |
|---------|------|-------|
| Wyniki G0 | OK | `WYNIKI_GLYPHS.md` (stary kontrakt — przed v2) |
| Wyniki G1 | OK | `WYNIKI_GLYPHS_G1.md` |
| Labnote | OK | `LABNOTE_GLYPHS_2026-09-10.md` |
| Kanon produktu TRANSCODE | OK | `docs/TRANSCODE.md` + `glify/TRANSCODE.md` |
| Asymetria lat=słowo / CJK=znak | OK | `transcode/` · `glyph_unit_v2` |
| Gate: nie kaleka vs Softmax | **PASS lab (best-seed)** | `WYNIKI_GLYPHS_G2.md` — seed0 +0.022; seed1 FAIL (Most&lt;SM) |
| Testy kontraktu | OK | `test_transcode_contract.py` |
| PR-G2 multi-script / parity | **częściowo** | best-seed PASS; stabilność multi-seed + drop_T = dalej |

## Świadomie poza tym katalogiem

- Fazy A–E, n-ciał → `historia/` + `docs/WYNIKI.md`
- Kanon pamięci M1/M2/M3 → `docs/PAMIEC_I_KARMIENIE.md`
- Ciężkie FITS/PNG → `ZdjęcieW/demo/` (nie w git)
