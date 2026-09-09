# Faza D — Offline dump Karmazyn → PhiMost

**Status:** PASS (2026-09-09)  
**Dump:** `dumps/textfile_phi`  
**Werdykt:** Most−SM r@1 ≈ **+2.3 pkt**; Most−R r@1 ≈ **+12 pkt**

## Cel

Model je **dump przestrzeni Φ** (nie tylko syntezator B/C).  
Case study: „plik tekstowy” = `__main__` + peer atomy ≠ flat bytes.

## Format dump (PhiJSON — kanon tego repo)

```text
dumps/<name>/
  manifest.json
  atoms.jsonl      # id, S, E, T, state, bubble, depth, payload?, mazur_tracer?, topic?
  edges.jsonl      # bind | parent | …
  CASE_TEXT.md
  CASE_RUNTIME.md  # generowane przy bench_dump
```

Atomstore / BubbleFS `*.bbl` — **nie kanoniczne** (adapter later).

## Artefakty kodu

| Plik | Rola |
|------|------|
| `make_dump_textfile.py` | budowa dumpa notatki Most/Φ |
| `phi_dump.py` | loader |
| `dump_recall.py` | recall z krawędzi bind (prefer same-depth) |
| `bench_dump.py` | Softmax \| Most \| ZERO \| R \| flat |

## Acceptance

- [x] dump → PhiGraph  
- [x] Most ≥ Softmax +2 pkt r@1  
- [x] HOT/WARM (WARM bez payload)  
- [x] case study nie-płaski  
- [x] docs  

## Uwagi (uczciwe)

- Brak live BubbleFS w sandboxie → dump **syntetyczno-realistyczny** w stylu Karmazyn.  
- Content dumpa: tom_content_vec (S/E/payload); strip payload ma efekt (po fix review).
- Most ZERO ≈ Most na N=15 — na małym dumpie warunek τ słabiej różnicuje; kanon: τ = warunek mostu, nie handicap.
## Wyniki

Zob. [`../historia/WYNIKI_FAZA_D.md`](../historia/WYNIKI_FAZA_D.md) · skrót: [`WYNIKI.md`](WYNIKI.md).

## Go

GO na Fazę E (zrobione) → dalej MemoryBus / checkpoints ([`STATUS.md`](STATUS.md)).
