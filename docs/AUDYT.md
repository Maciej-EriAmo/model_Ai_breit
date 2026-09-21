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

Zob. `REVIEW_EXTERNAL.md`. Fixy: Moon vs giant, dump `atom_content_vec` + strip, mazur→encoder, live fail-closed. Test: `doswiadczenie/testy/test_review_fixes.py`.

## P1 / P2 (zamknięte)

- ~~mask · checkpoints · MemoryBus · λ_neural+ckpt~~
- ~~BubbleFS importer~~ · ~~dekorelacja osi C~~ · ~~gen produkt~~
- Ablacje −depth/−T/−mazur informatywne po `decorrelate_axes=True`

## Regresja Mazur

`python -m mazur_crystal.test_bridge` — 6/6 OK (po Faza E).

## Walidacja n-ciał / Układ Słoneczny

Zob. `archiwum/historia/WYNIKI_NBODY.md` — **PASS 2/2 (mocny)**:

| zadanie | Softmax | Most | ZERO | Most−SM |
|---------|---------|------|------|---------|
| solar (star/rocky/giant) | 0.3563 | **0.6208** | 0.3667 | **+26.5 pkt** |
| three-body (Verlet, rank masy) | 0.5229 | **0.7417** | 0.5062 | **+21.9 pkt** |

Content bez masy/hierarchii (te osie tylko w tracerze) — inaczej Softmax „oszukuje”.

*Liczby zweryfikowane 2026-09-13 (poprzednia wersja tej tabeli, +20.6/+40.0 pkt,
nie zgadzała się z `archiwum/historia/WYNIKI_NBODY.md` — rozjazd sprzed audytu seala,
patrz sekcja „Audyt seal (2026-09-13)” niżej). Zgodność z `archiwum/historia/WYNIKI_NBODY.md`
potwierdzona bit-do-bitu po naprawie.*

## Audyt seal (2026-09-13)

`most_seal.apply_t_key_seal` (bramka T hard-mask/COLD, dodana commitem `bad5bda`
pod eksperymenty Holo-glify) była wpięta bezwarunkowo w `bridge.py`/`recall_model.py`
przez globalny default `CANON.t_attn_seal=True` — bez świadomości, że:

- `nbody_data.py` pakuje w slot T (index 3) **energię Kepler**, nie świeżość
  pamięci (HOT/WARM/COLD/TOMB) — seal ją błędnie maskował jako "archiwum/TOMB",
  psując `three-body` (Most spadał do 0.275 vs SM 0.523, z PASS +40pkt na **FAIL** −25pkt).
- `bench_phi.py` (Faza B) przy `n_dim<4` w ogóle nie ma kanału T — hardcoded
  `t_dim=3` w seal rzucał `IndexError`, blokując uruchomienie całego benchu.

**Fix:** `apply_t_key_seal` no-opuje gdy tracer jest węższy niż `t_dim` (crash fix);
`BridgeTransformer`/`PhiBridgeEncoder` dostały opt-out `t_seal: bool | None = None`
(domyślnie `None` = zachowanie z `CANON`, bez zmian dla Holo/kanonicznych benchy);
`bench_nbody.py` i `bench_phi.py` (zamknięte, audytowane przed powstaniem seala)
jawnie ustawiają `t_seal=False`. Zweryfikowano: 17/17 testów zielone,
`archiwum/historia/WYNIKI_NBODY.md` po naprawie bit-identyczny z wersją już w git.

### Drugi bug seala: ZERO-tracer martwy przez hard floor (2026-09-13, P5.1)

Podczas P5.1 (`doswiadczenie/glify/HOLO_P5_1.md`) audyt kodu transformera
wykrył kolejny efekt tej samej rodziny bugów: `RecallBatch.ablate(zero_tracer=True)`
/ `bridge.batch_to_xy(zero_tracer=True)` zerują CAŁY wektor tracera (nie tylko T)
jako kontrolę ablacyjną ("Most bez τ"). Ponieważ T=0 jest zawsze < `t_attn_floor`
(0.20), `apply_t_key_seal` maskował to jako "archive/TOMB" — **100% kluczy
hard-masked**, potwierdzone bezpośrednim testem. Zamiast matematycznej degeneracji
formuły zamka (`ΔE=0 → A=C/γ`, wciąż pełna, sensowna macierz uwagi), tor ZERO
dostawał martwą uwagę — inny, dużo gorszy eksperyment niż "Most bez informacji
z tracera". To dotyczyło **każdego benchu z ZERO-tracer w całym projekcie**
(Holo P0-P4a raportowanie ZERO, DESI, dump recall) korzystającego z domyślnego
`t_seal`, nie tylko P5.1 — choć w P4a gate liczony był Most vs Softmax (nie
vs ZERO), więc werdykty P4a/P0-P3 się nie zmieniają; ich raportowane liczby
ZERO są jednak zaniżone/nieinformacyjne sprzed tej naprawy.

**Fix:** `most_seal.apply_t_key_seal` wykrywa teraz per-wiersz, czy CAŁY
tracer atomu (poza CLS) jest zerowy (sygnatura ablacji — realny atom zawsze ma
niezerowy depth one-hot z `atom_tracer_vec`) i pomija seal dla takich wierszy,
zamiast traktować je jak archiwum. Zweryfikowano bezpośrednio (fraction
hard-masked: 1.0 → 0.0) i przez pełny test suite (20/20 zielone) oraz
inspekcję wag uwagi: silnik Most poprawnie reaguje na τ (|Δw| realne vs
zero_tracer ≈ 0.0245, przed i po treningu) — mechanizm działa zgodnie z
deklaracją, bugi były w warstwie kontroli ablacyjnej, nie w formule zamka.
