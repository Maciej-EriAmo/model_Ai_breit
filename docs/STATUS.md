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
| **karmazyn_lingua** | **6/6 TODO DONE** + reach-GC fail-loud; `pytest test_karmazyn_lingua.py` **35/35** |
| **Kana P1 / P1b** | P1 negatyw (0/5); P1b above_floor 3/3, Most≈SM (jak PL P1b) |
| **Mendelejew P0** | **DONE** — nonflat 3/3; Most≈0.43 ≈ Softmax≈0.45 (= **parity CV**) |
| **Hold-out Na** | **negatyw** — brak generalizacji grupy na ukryty Na (N=7) |
| **Mendelejew FULL 118** | dump + skrypt gotowe; **run MOST-only jeszcze nie** (~1.5–3 h) |
| Softmax/ZERO w nauce | `--baselines` **wymagane** przy claimach parity |
| Recall orbitali | `exclude_nucleus=True` (bez jądra w top-k) |
| **Doświadczenie/** | hub `doswiadczenie/` — JoannaM · DESI · glify · Mendelejew |

## Następne (kolejka, nie wszystko naraz)

| Priorytet | Item | Czas / uwagi |
|-----------|------|--------------|
| **A** | `run_mendelejew_full.py --most-only` | ~1.5–3 h; potem hold-out Na/K na FULL |
| **B** | Classical Softmax loader | TODO w TRANSCODE §2 — dziś Softmax=twin-on-Φ |
| **B** | Multi-seed stabilność G2 / bogatszy korpus | bez claimu „lepsi niż NN” |
| **C** | hologram-E (period×group×block overlay) | po FULL lub równolegle short |
| **C** | Track `hologram-znaki-raport.md` w git | nadal `??` untracked |
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
cd "C:\Users\drwis\model Ai"
python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote
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
python test_not_flat.py
python test_mask.py
python test_memory_bus.py
python test_decorrelate.py
python test_bubblefs_import.py
pytest test_karmazyn_lingua.py
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
