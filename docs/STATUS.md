# Status projektu PhiMost

**Data:** 2026-09-11 (transcode `glyph_unit_v2`)  
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
| **Doświadczenie/** | hub `doswiadczenie/` — JoannaM · DESI · glify |
| **Transcode v2** | `transcode/` + [`TRANSCODE.md`](TRANSCODE.md): lat=słowo@1, CJK=znak@2; gate vs Softmax |
| Recall orbitali | `exclude_nucleus=True` (bez jądra w top-k) |
| Softmax/ZERO w nauce | `--baselines` **wymagane** przy claimach parity |
| **G2 parity** | **PASS best-seed** (seed0 ΔSM=+0.022); seed1 Most&lt;SM — nie claim multi-seed |
| Następne | stabilność multi-seed / bogatszy korpus · hologramy = FUTURE §7 · JoannaM weryfikacja |

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

Zdrowie lab:

```powershell
python test_not_flat.py
python test_mask.py
python test_memory_bus.py
python test_decorrelate.py
python test_bubblefs_import.py
```

## Zakazy

- EMA-S / obs-in-S  
- Mieszanie M1↔M2↔M3  
- JSONL jako „pamięć”  
- Softmax-only bez Most  
- Promote FAIL przy `--baselines`

## Docs

[`LEARN.md`](LEARN.md) · [`PLAN_NEXT.md`](PLAN_NEXT.md) · [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md) · [`AUDYT.md`](AUDYT.md)
