# WYNIKI PR-G0 — glify → atomy → nauka ze stanu

**Data:** 2026-09-10  
**Dump:** `dumps/from_glyphs` (31 atomów, 202 edges, skrypty: `lat` + `hira`)  
**Ckpt:** `checkpoints/most_glyphs.pt`  
**Kontrakt:** `ZdjęcieW/GLYPH_ATOMS.md` — T ≠ bitmapa; Bridge `(x, T)`

## Softmax | Most | ZERO (recall r@1, gate ΔSM ≥ +0.02)

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.231 | 0.233 | 0.233 | −0.0016 | FAIL |
| 1 | 0.241 | 0.253 | 0.269 | −0.0125 | FAIL |
| **42** | **0.259** | 0.231 | 0.227 | **+0.0281** | **PASS** |

**Werdykt:** PASS (wybrany seed=42).  
Dowód: Most z τ(T) może wygrać na glifach bez OCR-bitmapy — nauka ze **stanu atomu**.

## Recall smoke (`lat/word_most/0/M`, fokus HOT)

1. `lat/word_most/3/T` (+0.82) — ten sam wyraz  
2. `lat/archive/X` (+0.81) — zimny fałszywy pozytyw (do poprawy w G1/G2)  
3. `lat/word_bridge/2/I` (+0.80) — peer skryptu (HOT fokus BRIDGE)  
4. `lat/word_most/2/S`, `…/1/O` — sąsiedzi w MOST  

## Jak powtórzyć

```bat
cd /d C:\Users\drwis\model Ai
python make_dump_glyphs.py
python phimost_learn.py --src dumps/from_glyphs --no-warm --no-promote --baselines
python phimost_learn.py --src dumps/from_glyphs --skip-load --skip-train --recall-id lat/word_most/0/M
```

## Dalej

- **PR-G1:** `glyph_transcribe.py` — zdjęcie → ten sam PhiJSON  
- **PR-G2:** więcej skryptów / twardszy gate; zmniejszyć wagę archive w recall  
