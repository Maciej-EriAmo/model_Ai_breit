# PhiMost — indeks dokumentacji

**Root:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz` (Holon / Karmin_Ae)  
**Ostatnia aktualizacja docs:** 2026-09-16 (STATUS / Mendelejew / lingua / luki)

**Pakiet konstrukcji jądra (2026-10-07):** [`konstrukcja/00_START.md`](konstrukcja/00_START.md) — wzór, kontrakt, szkielet, trzy drzewa, pomiary, otwarte miejsca, sprawdzenie. Od tego zaczyna inżynier. Reszta indeksu zostaje mapą laboratorium.

**Licencje:** kod `*.py` — MIT ([`../LICENSE-MIT`](../LICENSE-MIT)); teksty `*.md`, `*.txt` i specyfikacje pomiarów `*.json` — CC BY 4.0 ([`../LICENSE-CC-BY-4.0`](../LICENSE-CC-BY-4.0)). Podział: [`../LICENSE`](../LICENSE).

## Czytaj w tej kolejności

| # | Dokument | Po co |
|---|----------|--------|
| 1 | [`../README.md`](../README.md) | start, teza, komendy |
| 2 | [`STATUS.md`](STATUS.md) | **stan na dziś** + kolejka |
| 3 | [`INSTRUKCJA.md`](INSTRUKCJA.md) | **obsługa** — co klikać na co dzień |
| 4 | [`LEARN.md`](LEARN.md) | ładowanie danych + nauka (`phimost_learn`) |
| 5 | [`TRANSCODE.md`](TRANSCODE.md) | **produkt** — transkodowanie alfabetów; lat=słowo vs CJK=znak; gate vs Softmax |
| 5b | [`HOLO_ZALOZENIA.md`](../archiwum/holo/HOLO_ZALOZENIA.md) | **Holo-Φ** — założenia |
| 5c | [`../doswiadczenie/glify/HOLO_WNIOSKI.md`](../archiwum/holo/HOLO_WNIOSKI.md) | wnioski P0–P2 (blend **0.6**) |
| 5d | [`HOLO_P5_TOR_DESI.md`](../archiwum/holo/HOLO_P5_TOR_DESI.md) | **projekt P5** — τ z Unihan (stroke/radical) wzorem DESI |
| 5e | [`HOLO_P5_PLAN_IMPLEMENTACJI.md`](../archiwum/holo/HOLO_P5_PLAN_IMPLEMENTACJI.md) | plan P5; **P5.0–P5.1 zrobione** → FAIL holdout |
| 5f | [`../doswiadczenie/glify/MENDELEJEW_P0.md`](../doswiadczenie/glify/MENDELEJEW_P0.md) | **parity Most≈SM** (CV) · FULL: [`JUTRO_MENDELEJEW_FULL.md`](../archiwum/historia/JUTRO_MENDELEJEW_FULL.md) |
| 5f2 | [`../doswiadczenie/glify/MENDELEJEW_ML3.md`](../doswiadczenie/glify/MENDELEJEW_ML3.md) | SL / UL / RL na 8 pierwiastkach — parity; Most+ tylko w UL(n) 5/5 |
| 5f3 | [`../doswiadczenie/glify/MENDELEJEW_ML3_FULL.md`](../doswiadczenie/glify/MENDELEJEW_ML3_FULL.md) | FULL118 SL/UL/RL + szukanie luk (Sc/Ga/Ge/Tc) — Most+ w UL (blok +0.21, 5/5); odkrycie FAIL (AUC≈0.5, reguła siatki 1.0) |
| 5f4 | [`../doswiadczenie/glify/MENDELEJEW_ML3_ZC.md`](../doswiadczenie/glify/MENDELEJEW_ML3_ZC.md) | FULL118 tylko Z+konfiguracja — okres wyłania się, grupa nie; SL „ta sama grupa” po korekcie przecieku Most 0.45 vs SM 0.31 (4/5); odkrycie konfiguracji: tylko Sc |
| 5f5 | [`../doswiadczenie/glify/MENDELEJEW_SEMI.md`](../doswiadczenie/glify/MENDELEJEW_SEMI.md) | semi-supervised Z+konfiguracja — K=54 zrównoważone pary: Most 0.32 vs SM 0.10 / ZERO 0.05 / MLP 0.15 (5/5); dane bez etykiet szkodzą (poziome) |
| 5f6 | [`../doswiadczenie/glify/MENDELEJEW_ACTIVE.md`](../doswiadczenie/glify/MENDELEJEW_ACTIVE.md) | aktywny dobór 20 par — strategie oparte na podobieństwie < losowe (zbierają blok f); wyrocznia po grupach Most 0.16 |
| 5f7 | [`../doswiadczenie/glify/MENDELEJEW_REL.md`](../doswiadczenie/glify/MENDELEJEW_REL.md) | nauka relacyjna — osobne głowice: Most rel_Z hit@1¬f 0.41 / NMI 0.78 (najlepsze w serii, 5/5 vs SM, MLP); zlanie relacji (mix) niszczy grupę |
| 5f8 | [`../doswiadczenie/glify/MENDELEJEW_COMP.md`](../doswiadczenie/glify/MENDELEJEW_COMP.md) | relacje kompozycyjne dół/prawo — złożenia nieuczone na znanych: Most 0.74–0.93 (5/5 vs SM/ZERO/MLP); na lukach ~0.3, Tc→Mo/Ru/Nb |
| 5f9 | [`../doswiadczenie/glify/MENDELEJEW_COMPREL.md`](../doswiadczenie/glify/MENDELEJEW_COMPREL.md) | kompozycja + relacja grupy (5 seedów; skorygowane przez 5f10) |
| 5f10 | [`../doswiadczenie/glify/MENDELEJEW_COMPREL_FROZEN.md`](../doswiadczenie/glify/MENDELEJEW_COMPREL_FROZEN.md) | zamrożony protokół, 20 przebiegów, Holm — połączenie +0.06…+0.14 (utrzymuje się); Most > SM/ZERO, ≈ MLP na nowych; kompozycja na znanych Most > wszyscy 17–20/20 |
| 5f11 | [`../doswiadczenie/glify/MENDELEJEW_KERNEL_ABLATION.md`](../doswiadczenie/glify/MENDELEJEW_KERNEL_ABLATION.md) | **czysta ablacja jądra** (te same parametry/dane/procedura, 20 przebiegów) — przewaga Mostu w serii to głównie informacja ΔE, nie jądro; jądro BW +0.07…+0.11 tylko w comp+rel·PoE |
| 5f12 | [`../doswiadczenie/glify/MENDELEJEW_RELATION_SHARE.md`](../doswiadczenie/glify/MENDELEJEW_RELATION_SHARE.md) | udział struktury relacyjnej przy stałym jądrze (Shapley) — relacja grupy (PoE) 32–68% poprawy na nowych pierwiastkach, w każdym jądrze i w MLP; na znanych 0 |
| 5f13 | [`../doswiadczenie/glify/MENDELEJEW_KERNEL_TOK.md`](../doswiadczenie/glify/MENDELEJEW_KERNEL_TOK.md) | τ w tokenach — Softmax ≥ Most (13/15), jądro bez znaczenia (30/30 n.s.); najlepsze: comp+mix z τ w tokenach, dowolne jądro |
| 5f14 | [`../doswiadczenie/glify/CROSSDOMAIN_COMP.md`](../doswiadczenie/glify/CROSSDOMAIN_COMP.md) | kompozycja na 3 dziedzinach (pierwiastki / kana / muzyka), ścisłe rozdzielenie — relacja kolumny (PoE) przenosi się (muzyka +0.07…+0.17, 5/5 ramion), nie w kanie (cechy bez struktury); jądro BW 0 wszędzie |
| 5f15 | [`../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md`](../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md) | **TOST ±5 pkt** — Most równoważny Softmaxowi przy τ w tokenach (26/30, 0 gorszych); przy τ tylko przez ΔE zależnie od dziedziny |
| 5f16 | [`ARTYKUL_POPULARNY.md`](ARTYKUL_POPULARNY.md) | **artykuł popularnonaukowy** — podsumowanie całej serii Mendelejewa (eksport z dokumentu Claude) |
| 5f17 | [`../doswiadczenie/glify/NBODY_COMP.md`](../doswiadczenie/glify/NBODY_COMP.md) | relacja na katalogu słonecznym i trójce hierarchicznej — kolumna pomaga, gdy jest cicha; jądro BW bez przewagi |
| 5f18 | [`../doswiadczenie/glify/JADRO_OGOLNE.md`](../doswiadczenie/glify/JADRO_OGOLNE.md) | **wnioski 2026-10-05** i program brakujących testów jądra ogólnego (T0–T3) |
| 5f19 | [`PODSUMOWANIE_JADRA.md`](PODSUMOWANIE_JADRA.md) | **jedno podsumowanie** Most i Softmax: wniosek, plugin, relacja, czas |
| 5f20 | [`ZLEW_UWAGI_PILOT.md`](ZLEW_UWAGI_PILOT.md) | **pilot 2026-10-26** — zlew uwagi i normalizacja L1: Most bez L1 wyraźnie gorszy (L1 zostaje); zlewu brak nawet w softmaksie — nierozstrzygnięte; seria C (skala log N, LSSA): most_logn najlepszy na 1 ziarnie, brak kontroli softmax_logn — plan dalszych badań |
| 5f21 | [`SIGMOID_W_UWADZE.md`](SIGMOID_W_UWADZE.md) | przegląd literatury 2023–2025: uwaga sigmoidalna, zlew uwagi, normalizacja L1, bramki — tło dla 5f20 |
| 5g | [`../doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md`](../doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md) | lingua 6/6 DONE · mapa torów (nie mylić z Holo-Φ) |
| 5h | [`../doswiadczenie/glify/CV_BLURB_MOST_PARITY.md`](../doswiadczenie/glify/CV_BLURB_MOST_PARITY.md) | one-liner pod CV / portfolio |
| 6 | [`GLYPHS.md`](GLYPHS.md) | notatki badawcze G0/G1 (lab) |
| 7 | [`POROWNANIE_MODELE.md`](POROWNANIE_MODELE.md) | Most vs Transformer/LLM · czemu uczyć się na własnym |
| 8 | [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md) | **kanon** pamięci M1/M2/M3 + karmienie `x`/`τ` |
| 9 | [`../config.py`](../config.py) | hiperparametry empiryczne (kod) |
| 10 | [`AUDYT.md`](AUDYT.md) | luki, zgodność z wynikami |
| 11 | [`WYNIKI.md`](WYNIKI.md) | skrót tabel PASS/FAIL |
| 12 | [`MODULY.md`](MODULY.md) | mapa plików `.py` |
| 13 | [`PLAN_NEXT.md`](PLAN_NEXT.md) | plan / backlog |
| 14 | [`plan.txt`](plan.txt) | plan faz A–E + decyzje |
| 15 | [`kryteria_sukcesu.txt`](kryteria_sukcesu.txt) · [`KRYTERIA_SUKCESU_MAPA.md`](KRYTERIA_SUKCESU_MAPA.md) | #5–#1c DONE · mid-probe FAIL · GPU nie z trendu |
| 20 | recenzja / knobs / [`SIGNED_DELTA_NOTE.md`](SIGNED_DELTA_NOTE.md) · [`../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md`](../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md) | ΔE: LM vs atomy FULL118 |
| 19 | [`../doswiadczenie/glify/LM_TINY_1B.md`](../doswiadczenie/glify/LM_TINY_1B.md) · [`LM_MEDIUM_1C.md`](../doswiadczenie/glify/LM_MEDIUM_1C.md) | tiny + medium LM Most vs SM |
| 18 | [`../doswiadczenie/glify/MENDELEJEW_BASELINES_COMPARE.md`](../doswiadczenie/glify/MENDELEJEW_BASELINES_COMPARE.md) | Most vs SM/ZERO · timing/RSS |
| 16 | [`../runs/README.md`](../runs/README.md) · `runlog.py` · `precision.py` | run-dir + BF16 autocast/finite-check |
| 17 | [`../doswiadczenie/glify/BF16_GATE.md`](../doswiadczenie/glify/BF16_GATE.md) | gate #2 PASS (Most bf16, FULL dump) |

## Fazy (szczegóły)

| Faza | Doc | Wyniki |
|------|------|--------|
| A–B | (README + testy) | [`../archiwum/historia/WYNIKI_FAZA_B.md`](../archiwum/historia/WYNIKI_FAZA_B.md) |
| C | — | [`../archiwum/historia/WYNIKI_RECALL.md`](../archiwum/historia/WYNIKI_RECALL.md) |
| D | [`FAZA_D.md`](FAZA_D.md) | [`../archiwum/historia/WYNIKI_FAZA_D.md`](../archiwum/historia/WYNIKI_FAZA_D.md) |
| E | [`FAZA_E.md`](FAZA_E.md) | [`../archiwum/historia/WYNIKI_FAZA_E.md`](../archiwum/historia/WYNIKI_FAZA_E.md) |
| Fizyka | — | [`../archiwum/historia/WYNIKI_NBODY.md`](../archiwum/historia/WYNIKI_NBODY.md) |
| **Glify G0–G2** | [`GLYPHS.md`](GLYPHS.md) | G0/G1/G2 w [`../doswiadczenie/glify/`](../doswiadczenie/glify/) · labnote [`LABNOTE_GLYPHS_2026-09-10.md`](../doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md) |
| **Holo-Φ P5.1** | [`HOLO_P5_TOR_DESI.md`](../archiwum/holo/HOLO_P5_TOR_DESI.md) | [`../doswiadczenie/glify/HOLO_P5_1.md`](../archiwum/holo/HOLO_P5_1.md) — **FAIL 0/8** holdout |
| **Mendelejew** | [`MENDELEJEW_FULL.md`](../doswiadczenie/glify/MENDELEJEW_FULL.md) | P0 · FULL **DONE** · hold-out Na/K **FAIL** ([`MENDELEJEW_HOLDOUT_FULL_NaK.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK.md)) |
| **Lingua / znaki** | [`AUDYT_HOLOGRAM_ZNAKI.md`](../doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md) | warstwa DONE; Kana P1 negatyw; P1b Most≈SM |

## Doświadczenie (osobny katalog)

Hub: [`../doswiadczenie/README.md`](../doswiadczenie/README.md) · checklist: [`../doswiadczenie/CHECKLIST.md`](../doswiadczenie/CHECKLIST.md) · mapa glify: [`../doswiadczenie/glify/README.md`](../doswiadczenie/glify/README.md) · testy: [`../doswiadczenie/testy/`](../doswiadczenie/testy/)

| Tor | Wejście | Wyniki / notatki |
|-----|---------|------------------|
| **JoannaM** (kandydat optyczny) | [`../doswiadczenie/joannam/JOANNAM.md`](../doswiadczenie/joannam/JOANNAM.md) | weryfikacja [`WERYFIKACJA.md`](../doswiadczenie/joannam/WERYFIKACJA.md) · forum EN [`FORUM_EN.md`](../doswiadczenie/joannam/FORUM_EN.md) |
| **DESI / niebo** | [`../archiwum/desi/`](../archiwum/desi/) | `WYNIKI_DESI*.md`, holdout, BH/AGN, [`AUDYT_2026-09-10.md`](../archiwum/desi/AUDYT_2026-09-10.md) — **zawieszone** |
| **Glify / transkodowanie** | [`TRANSCODE.md`](TRANSCODE.md) · [`GLYPHS.md`](GLYPHS.md) | [`../doswiadczenie/glify/`](../doswiadczenie/glify/) · wnioski [`AUDYT_WNIOSKI.md`](../doswiadczenie/glify/AUDYT_WNIOSKI.md) |
| **Mendelejew / struktura** | P0 + FULL | parity CV · FULL118 kolejka |
| **Lingua hologram-znaki** | `archiwum/skrypty/karmazyn_lingua.py` | audyt 2026-09-15; raport [`hologram-znaki-raport.md`](hologram-znaki-raport.md) |

> `archiwum/historia/` = artefakty faz A–E + n-ciał + learn. Tor nieba i glify → `doswiadczenie/`.

## Case study „plik ≠ płaski”

- `dumps/textfile_phi/CASE_TEXT.md` *(generowany, poza repo)*
- `dumps/textfile_phi/CASE_RUNTIME.md` *(generowany, poza repo)*

## Case study glify / alfabety

- Kontrakt: [`../archiwum/ZdjęcieW/GLYPH_ATOMS.md`](../archiwum/ZdjęcieW/GLYPH_ATOMS.md) · oś T: [`../archiwum/ZdjęcieW/1d_do_2d.txt`](../archiwum/ZdjęcieW/1d_do_2d.txt)
- Dump: `dumps/from_glyphs/CASE_GLYPHS.md` *(generowany, poza repo)*
- Metoda (dla badaczy): [`GLYPHS.md`](GLYPHS.md)

## Powiązane poza repo

| Tor | Ścieżka | Rola |
|-----|---------|------|
| Lab Most | `C:\Users\drwis\mcps\moje badania\` | empiria zamka, RAPORT |
| KarmazynOs | `Documents/MAZUR_CRYSTAL.md`, `software/mazur_crystal/` | skalarne R, λ_neural OFF |
| Holon SE | `C:\Users\drwis\Karmin_Ae` + `AGENTS.md` | boot / handoff agenta |

## Status / backlog / plan

[`STATUS.md`](STATUS.md) · [`PLAN_NEXT.md`](PLAN_NEXT.md) · artefakty faz: [`../archiwum/historia/`](../archiwum/historia/)

## External code review

[`REVIEW_EXTERNAL.md`](../archiwum/historia/review/REVIEW_EXTERNAL.md) · [`REVIEW_SUMMARY.md`](../archiwum/historia/review/REVIEW_SUMMARY.md)  
(od `/review` na katalogu — 4 bugs / 6 suggestions / 2 nits)
