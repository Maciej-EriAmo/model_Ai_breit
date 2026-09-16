# PhiMost — Most Lorentza nad przestrzenią Φ (KarmazynOs)

**Katalog roboczy:** `C:\Users\drwis\model Ai`  
**Indeks docs:** [`docs/INDEX.md`](docs/INDEX.md) · **Status:** [`docs/STATUS.md`](docs/STATUS.md)

## Teza

Most (transformer z zamkiem Lorentza) **powstaje dopiero przy niezerowym** \(\Delta E\) z tracera — τ to **warunek mostu**, nie handicap względem Softmaxu.  
Bez tracera most się nie formuje / jest niestabilny (degeneracja). Softmax to uwagi bez tej geometrii; porównanie ZERO pokazuje brak warunku, nie „fair fight”.  
W KarmazynOs „plik tekstowy” **nie jest płaski** — atomy `(S, E, T)` + bańki + HOT/WARM/COLD. Silnik = **NN Most**.

\[
A_{ij} = \frac{\gamma\, C_{ij}}{\gamma^2 + \Delta E_{ij}^2}
\quad (C=\mathrm{softplus}(QK^\top/\sqrt{d})^2;\ \Delta E \text{ z tracera})
\]

## Kanon

| Tak | Nie |
|-----|-----|
| Czysty Most softplus×2 + tracer (`config.CANON`) | EMA-S / Most+S / obs-in-S |
| Softmax twin + ZERO-tracer | Trening bez ablacji |
| Treść ← `E`/payload; zamek ← tracer | Wszystko w jednym torze tokenów |
| Φ-schema / PhiJSON | Atomstore jako kanon |
| Trzy pamięci M1·M2·M3 | Mieszanie Holon ↔ wagi ↔ GC Store |

Szczegóły: [`docs/PAMIEC_I_KARMIENIE.md`](docs/PAMIEC_I_KARMIENIE.md) · słownik: [`docs/SLOWNIK.md`](docs/SLOWNIK.md)

## MVP

Encoder Most → **recall/ranking** nad atomami (nie od razu generative LLM).

## Wyniki (skrót)

| Tor | Most − Softmax | Status |
|-----|----------------|--------|
| Φ-drzewo (B) | +3.5…+17 pkt | PASS |
| Recall r@1 (C) | **+71 pkt** | PASS |
| Dump Φ (D) | +2.3 pkt | PASS |
| Solar / 3 ciała | **+20.6 / +40 pkt** | PASS |
| G2 glify v2 | +2.2 pkt (best-seed) | PASS lab; nie multi-seed |
| Mendelejew P0 | ≈ Softmax (−0.02) | **parity** + nonflat ≫ flat |

Pełna tabela: [`docs/WYNIKI.md`](docs/WYNIKI.md) · stan: [`docs/STATUS.md`](docs/STATUS.md)

## Fazy

| Faza | Status | Artefakt |
|------|--------|----------|
| **A** kontrakt Φ + nie-płaski | PASS | `test_not_flat.py` |
| **B** Softmax\|Most\|ZERO Φ-drzewo | PASS | `historia/WYNIKI_FAZA_B.md` |
| **C** recall/ranking | PASS | `historia/WYNIKI_RECALL.md` |
| **D** offline dump | PASS | `docs/FAZA_D.md`, `dumps/textfile_phi/` |
| **E** live + gen + λ_neural | PASS | `docs/FAZA_E.md` |
| **Audyt + n-ciał** | PASS | `docs/AUDYT.md`, `historia/WYNIKI_NBODY.md` |
| **Pamięć / karmienie** | kanon | `docs/PAMIEC_I_KARMIENIE.md` |

## Mapa (skrót)

| Obszar | Pliki |
|--------|--------|
| Silnik | `bridge.py`, `recall_model.py`, `config.py` |
| Dane Φ | `phi_schema.py`, `synthetic_phi.py`, `phi_dump.py` |
| Fizyka | `nbody_data.py`, `bench_nbody.py` |
| Faza E | `live_store.py`, `gen_head.py`, `mazur_neural.py` |

Pełna mapa: [`docs/MODULY.md`](docs/MODULY.md)

## Start (PowerShell)

```powershell
cd "C:\Users\drwis\model Ai"
python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote
python phimost_learn.py --src dumps/from_struktura --skip-load --skip-train --recall-id C/2p/m-1/up
python test_not_flat.py; python test_mask.py; python test_memory_bus.py
```

**Obsługa:** [`docs/INSTRUKCJA.md`](docs/INSTRUKCJA.md) · **Nauka:** [`docs/LEARN.md`](docs/LEARN.md) · **Glify/alfabety:** [`docs/GLYPHS.md`](docs/GLYPHS.md) · **vs LLM:** [`docs/POROWNANIE_MODELE.md`](docs/POROWNANIE_MODELE.md).  
**Doświadczenie** (JoannaM / DESI / glify / Mendelejew): [`doswiadczenie/README.md`](doswiadczenie/README.md) · mapa glify: [`doswiadczenie/glify/README.md`](doswiadczenie/glify/README.md).  
Benchi: `bench_phi.py`, `bench_recall.py`, `bench_dump.py`, `bench_nbody.py`.

## Konfiguracja i flagi

- Kanon kodu: `config.py` — `softplus×2`, `n_dim=6` (dim5 = **mazur_tracer**)  
- Product OFF: `PHIMOST_LIVE`, `PHIMOST_GEN`, `PHIMOST_NEURAL_LAMBDA`  

## Powiązania

| Tor | Ścieżka |
|-----|---------|
| Lab Most | `C:\Users\drwis\mcps\moje badania\` (`most.py`, `RAPORT.md`) |
| Mazur produkt | `KarmazynOs\software\mazur_crystal\` |
| Agent SE | `C:\Users\drwis\Karmin_Ae` · komora `Most_Lorentz` |
| Plan | [`plan.txt`](plan.txt) |

## Status planu

P1 + **P2 zamknięte** (λ_neural · BubbleFS · dekorelacja C · gen produkt).  
Plan: [`docs/PLAN_NEXT.md`](docs/PLAN_NEXT.md) · status: [`docs/STATUS.md`](docs/STATUS.md).  
Artefakty faz: [`historia/`](historia/).

```powershell
# Uniwersalne ładowanie + nauka (PowerShell)
# domyślnie TYLKO Most (szybko); Softmax/ZERO: dodaj --baselines
python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote
python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote --baselines
python phimost_learn.py --src .\ksiazka.md --no-warm --no-promote
python phimost_day.py --query main -k 5
```
