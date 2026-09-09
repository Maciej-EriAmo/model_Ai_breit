# Audyt PhiMost — 2026-09-09

Indeks: [`INDEX.md`](INDEX.md) · Wyniki: [`WYNIKI.md`](WYNIKI.md) · Status: [`STATUS.md`](STATUS.md) · Nauka: [`LEARN.md`](LEARN.md)

## Werdykt skrót

Architektura i **domyślne kanony** (softplus×2, Softmax+ZERO w *benchach*, r@1, flagi E OFF, live RO) są zgodne z empirią A–E.  
P0–P2 + U1–U3 zamknięte.  
**Wieczór:** uniwersalny `phimost_learn` (Most-only default); CSV/JSON struktura atomów; literatura; `exclude_nucleus` w recall.

## Audyt wieczorny (cleanup)

| Punkt | Werdykt |
|-------|---------|
| Softmax/ZERO w rutynowej nauce | **wyłączone** (opc. `--baselines`) — 3× czas |
| JSONL jako pamięć | **nie** — tylko epizod |
| `phimost_learn` pokrywa CSV/JSON/md/PhiJSON/BubbleFS/presety | OK |
| Ckpt / dumps w git | **gitignore** |
| Recall orbitali bez nucleus | OK (`exclude_nucleus`) |
| Dokumentacja LEARN + STATUS + MODULY | uzupełnione |

## Canon (wyniki → config)

| Canon | Źródło | Stan |
|-------|--------|------|
| Most softplus × 2 + tracer | lab + Faza B | OK w `bridge.py` / encoders |
| Zakaz EMA-S / Most+S | RAPORT §10 | OK — brak w kodzie |
| Softmax twin + ZERO | B/C/D | OK |
| Gate recall = r@1 ≥ +2 pkt | Faza C | OK |
| Flagii E OFF | Faza E | OK |
| Live bez tick/heat | Faza E | OK |

## P0 naprawione w tej sesji

1. Stabilny `hash_embed` (`hashlib`, nie `hash()`)
2. Tracer `n_dim=6`: T + state + **mazur** (bez gubienia mazur@8)
3. `config.py` — jedna kanoniczna konfiguracja
4. Walidacja fizyczna: **3 ciała / Układ Słoneczny** (`bench_nbody.py`)

## External review — 4 bugs (naprawione)

Zob. `REVIEW_EXTERNAL.md`. Fixy: Moon vs giant, dump `atom_content_vec` + strip, mazur→encoder, live fail-closed. Test: `test_review_fixes.py`.

## P1 / P2 (zamknięte)

- ~~mask · checkpoints · MemoryBus · λ_neural+ckpt~~
- ~~BubbleFS importer~~ · ~~dekorelacja osi C~~ · ~~gen produkt~~
- Ablacje −depth/−T/−mazur informatywne po `decorrelate_axes=True`

## Regresja Mazur

`python -m mazur_crystal.test_bridge` — 6/6 OK (po Faza E).

## Walidacja n-ciał / Układ Słoneczny

Zob. `historia/WYNIKI_NBODY.md` — **PASS 2/2 (mocny)**:

| zadanie | Softmax | Most | ZERO | Most−SM |
|---------|---------|------|------|---------|
| solar (star/rocky/giant) | 0.400 | **0.606** | 0.408 | **+20.6 pkt** |
| three-body (Verlet, rank masy) | 0.523 | **0.923** | 0.506 | **+40.0 pkt** |

Content bez masy/hierarchii (te osie tylko w tracerze) — inaczej Softmax „oszukuje”.
