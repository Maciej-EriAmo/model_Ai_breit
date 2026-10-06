# Status projektu PhiMost

**Data:** 2026-09-16 (porządek docs + luki)  
**Katalog:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz`

## Stan (skrót)

| Obszar | Stan |
|--------|------|
| Fazy A–E + n-ciał | **PASS** |
| P1 / P2 / U1–U3 | **zamknięte** |
| CLI nauki | **`phimost_learn.py`** (Most-only domyślnie) |
| Struktura atomów CSV/JSON | import + ckpt `most_struktura_atomow.pt` (r@1≈0.45) |
| Literatura `.md`/`.txt` | `import_literature.py` → `dumps/from_literature` |
| **Glify G0** | **PASS** seed42 ΔSM≈+0.028 · `dumps/from_glyphs` · `most_glyphs.pt` |
| **Glify G1** | **PASS** seed0 ΔSM≈+0.044 · `glyph_transcribe` → `dumps/from_photo` · `most_glyphs_photo.pt` |
| **G2 parity** | **PASS best-seed** (seed0 ΔSM=+0.022); seed1 Most&lt;SM — **nie** claim multi-seed |
| **Transcode v2** | `transcode/` + [`TRANSCODE.md`](TRANSCODE.md): lat=słowo@1, CJK=znak@2; Softmax=twin-on-Φ |
| **Holo-Φ P0–P4a** | P3 FAIL (faza); P4a FAIL @ blend 0.6; seal uszczelniony |
| **Holo-Φ P5.1** | **FAIL 0/8** holdout (τ=Unihan stroke); train→100%, holdout≈chance — metoda nie generalizuje |
| **karmazyn_lingua** | **6/6 TODO DONE** + reach-GC fail-loud; `python -m pytest doswiadczenie/testy/test_karmazyn_lingua.py` **35/35** |
| **Kana P1 / P1b** | P1 negatyw (0/5); P1b above_floor 3/3, Most≈SM (jak PL P1b) |
| **Mendelejew P0** | **DONE** — nonflat 3/3; Most≈0.43 ≈ Softmax≈0.45 (= **parity CV**) |
| **Hold-out Na** | **negatyw** — brak generalizacji grupy na ukryty Na (N=7) |
| **Mendelejew FULL 118** | **DONE MOST-only** — mean r@1=0.567 · r@3=0.897 · above_floor **3/3** · [`MENDELEJEW_FULL.md`](../doswiadczenie/glify/MENDELEJEW_FULL.md) |
| **Hold-out Na/K FULL** | **FAIL 2/2** — kuzyn nie top-3 (Na: Li 29/2; K: Na 19/27) · lokalna topologia ≠ grupa · [`MENDELEJEW_HOLDOUT_FULL_NaK.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK.md) |
| Softmax/ZERO w nauce | `--baselines` **wymagane** przy claimach parity |
| Recall orbitali | `exclude_nucleus=True` (bez jądra w top-k) |
| **Doświadczenie/** | hub `doswiadczenie/` — JoannaM · DESI · glify · Mendelejew |

## Następne (kolejka, nie wszystko naraz)

| Priorytet | Item | Czas / uwagi |
|-----------|------|--------------|
| **A** | ~~Hold-out Na/K na FULL118~~ | **FAIL 2/2** — brak generalizacji grupy ([`MENDELEJEW_HOLDOUT_FULL_NaK.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK.md)) |
| **A′** | ~~`run_mendelejew_full.py --most-only`~~ | **DONE** 2026-10-06 jądro Breita · r@1=0,575 · floor 3/3 · 768 s · [`MENDELEJEW_FULL.md`](../doswiadczenie/glify/MENDELEJEW_FULL.md) |
| **B′** | hologram-E (period×group×block) | opcjonalnie; hold-out FAIL zostaje jako negatyw |
| **S0** | ~~Kryterium #5 run-dir~~ | **DONE** — `runlog.py` · `runs/` · learn + FULL |
| **S1** | ~~Kryterium #2 BF16~~ | **DONE** — `precision.py` · `--bf16` · gate PASS ([`BF16_GATE.md`](../doswiadczenie/glify/BF16_GATE.md)) |
| **S2** | Kryterium **#1** — baselines na FULL118 | **DONE (zrzut)** — Most≈SM r@1 (+0.005), gate **0/5**, Most ~1.7× wolniejszy, RSS≈ · [`MENDELEJEW_BASELINES_COMPARE.md`](../doswiadczenie/glify/MENDELEJEW_BASELINES_COMPARE.md) |
| **S3** | ~~#1b tiny LM smoke~~ | **DONE** — ~116k params · Most loss 0.637 vs SM 0.664 · descend OK · [`LM_TINY_1B.md`](../doswiadczenie/glify/LM_TINY_1B.md) |
| **S4** | ~~#1c medium LM ~3.3M~~ | **DONE** 2026-10-06 notatki z repo · Most 2,416 vs SM 2,353 · 1,32× · [`LM_MEDIUM_1C.md`](../doswiadczenie/glify/LM_MEDIUM_1C.md) |
| **S4b** | średni model na WikiText-2 | **DONE** valid 1,557/1,604 · test 1,580/1,627 · Most lekko niżej, oba seedy · duży model nadal nie · [`LM_MEDIUM_WIKITEXT_HOLDOUT.md`](../doswiadczenie/glify/LM_MEDIUM_WIKITEXT_HOLDOUT.md) |
| **S5** | Lost-in-middle probe (recenzja) | **FAIL hipotezy** @3M/2k — SM płaski mid, Most nie wyższy ent_mid · [`LOST_IN_MIDDLE_PROBE.md`](../doswiadczenie/glify/LOST_IN_MIDDLE_PROBE.md) · [`ODPOWIEDZ_RECENZJA_LOST_IN_MIDDLE.md`](ODPOWIEDZ_RECENZJA_LOST_IN_MIDDLE.md) |
| **S6** | Strojenie gałek zamka (γ/e/τ) | **PASS cienki** — `ultra_wide` (γ=2.5, e=0.1, τ=2) ≈ SM ent_mid · [`MOST_KNOBS_SWEEP.md`](../doswiadczenie/glify/MOST_KNOBS_SWEEP.md) |
| **S7** | Skierowane ΔE (LM mid) | **PASS cienki** tylko z szerokim g; sam directed ≤ square · [`SIGNED_DELTA_ABLATION.md`](../doswiadczenie/glify/SIGNED_DELTA_ABLATION.md) |
| **S7b** | ΔE na atomach FULL118 | **PARTIAL** — **onesided** r@1=0.591 > SM 0.579 > square 0.567 · gate 1/2 · [`ATOMS_DELTA_ABLATION.md`](../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md) |
| **S7c** | Hold-out Na/K **onesided** | **Na PARTIAL / K FAIL** — Li 10/6 (K@1/3); Na 57/59 · [`MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md) |
| **S8** | #1d / GPU | nadal: retrieval + wyraźna przewaga, nie ε z szerokiego g |
| **B** | Classical Softmax loader | TODO w TRANSCODE §2 — dziś Softmax=twin-on-Φ |
| **B** | Multi-seed stabilność G2 / bogatszy korpus | bez claimu „lepsi niż NN” |
| **C** | hologram-E (period×group×block overlay) | po FULL lub równolegle short |
| **C** | Track `hologram-znaki-raport.md` w git | przeniesiony do `docs/` |
| **D** | JoannaM X-ray/radio + ablacja gwiazdy @8.8″ | spektroskopia NO MATCH już DONE |
| **D** | Qumran | później |
| — | DESI | zawieszone (gate FAIL; bloker T/COLD był, seal DONE — skala korpusu) |

## Artefakty ckpt (lokalne, gitignore)

| Plik | Rola |
|------|------|
| `most_dump_recall.pt` | default / DAY |
| `most_karmazyn_live.pt` | fine-tune Store |
| `most_struktura_atomow.pt` | CSV/JSON Aufbau |
| `most_qm_orbitals.pt` | QM syntetyczny |
| `most_literature.pt` | tekst |
| `most_glyphs.pt` | glify lat+hira (G0) |
| `most_glyphs_photo.pt` | glify z transkrypcji zdjęcia (G1) |
| `most_gen_head.pt` | gen E |

## Komendy dnia

```powershell
cd "C:\Users\drwis\model_Ai_breit"
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote
python phimost_learn.py --src dumps/from_struktura --skip-load --skip-train --recall-id C/2p/m-1/up
python phimost_day.py --query main -k 5 --skip-export
```

Mendelejew (gdy budżet czasu):

```powershell
python export_mendelejew_full.py
python run_mendelejew_full.py --most-only
# smoke: python run_mendelejew_full.py --most-only --smoke
```

Zdrowie lab:

```powershell
python doswiadczenie/testy/test_not_flat.py
python doswiadczenie/testy/test_mask.py
python doswiadczenie/testy/test_memory_bus.py
python doswiadczenie/testy/test_decorrelate.py
python doswiadczenie/testy/test_bubblefs_import.py
python -m pytest doswiadczenie/testy/test_karmazyn_lingua.py
```

## Zakazy

- EMA-S / obs-in-S  
- Mieszanie M1↔M2↔M3  
- JSONL jako „pamięć”  
- Softmax-only bez Most  
- Promote FAIL przy `--baselines`  
- Claim multi-seed / „lepsi niż Softmax” bez stabilnego gate  
- Mylenie torów: lingua-znaki ≠ Holo-Φ ≠ DESI ≠ Mendelejew  

## Docs

[`INDEX.md`](INDEX.md) · [`LEARN.md`](LEARN.md) · [`PLAN_NEXT.md`](PLAN_NEXT.md) · [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md) · [`AUDYT.md`](AUDYT.md) · [`TRANSCODE.md`](TRANSCODE.md) · hub [`../doswiadczenie/README.md`](../doswiadczenie/README.md)
