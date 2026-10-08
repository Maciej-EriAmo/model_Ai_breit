# PhiMost — szkielet nad przestrzenią Φ

**Katalog tego forka:** `C:\Users\drwis\model_Ai_breit`  
**Indeks docs:** [`docs/INDEX.md`](docs/INDEX.md) · **Status:** [`docs/STATUS.md`](docs/STATUS.md)

## Fork — jądro Breita–Wignera (Most-1)

To jest fork `C:\Users\drwis\model Ai` z commita `592bf00`. Tamten katalog nie jest ruszany. Zdalnego remote tu nie ma.

## Licencje

Kod programu (`*.py`) jest na licencji MIT: [`LICENSE-MIT`](LICENSE-MIT).
Teksty (`*.md`, `*.txt`) i specyfikacje pomiarów (`*.json` z notatkami) są na licencji CC BY 4.0: [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0).
Podział: [`LICENSE`](LICENSE). Autor: Maciej Mazur, 2026.

Waga uwagi:

\[
\gamma = \gamma_0\,(1 + \beta\,\kappa),\quad \kappa = \max_j S0_{ij},\quad S0 = C / \sum C
\]

\[
A_{ij} = \frac{\gamma_i\, C_{ij}}{\gamma_i^2 + \Delta E_{ij}^2}
\]

`C` dalej jest `softplus(QK)²`. `ΔE` dalej pochodzi z tracera. Do wagi wchodzi `ΔE²`. `beta_raw = 0` daje `beta = softplus(0) ≈ 0.69`, więc szerokość od razu zależy od szczytu wiersza. Tryby `directed` i `onesided` są odrzucone.

Wzór liczą `breit_wigner_rows` i `breit_wigner_weights` w [`breit_kernel.py`](breit_kernel.py). Dostają gotowe `C` i `ΔE`. Transformer, który te tensory buduje, jest w trzech plikach: `bridge.BridgeTransformer`, `recall_model.PhiBridgeEncoder`, `lm_tiny._MostCausalBlock`. W każdym pliku obok jest bliźniak softmax. Seal jest w `most_seal.py`.

## Teza

Szkielet zostaje: waga powstaje przy niezerowym \(\Delta E\) z tracera. Softmax jest bliźniakiem bez tej geometrii.  
W tym forku szerokość wiersza to \(\gamma_0(1+\beta\kappa)\), a \(\kappa\) jest szczytem \(C\) w tym wierszu. Wzór jest na górze pliku.  
`C` bierze się z `softplus(QK)²`, `ΔE` z tracera.

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
| **A** kontrakt Φ + nie-płaski | PASS | `doswiadczenie/testy/test_not_flat.py` |
| **B** Softmax\|Most\|ZERO Φ-drzewo | PASS | `archiwum/historia/WYNIKI_FAZA_B.md` |
| **C** recall/ranking | PASS | `archiwum/historia/WYNIKI_RECALL.md` |
| **D** offline dump | PASS | `docs/FAZA_D.md`, `dumps/textfile_phi/` |
| **E** live + gen + λ_neural | PASS | `docs/FAZA_E.md` |
| **Audyt + n-ciał** | PASS | `docs/AUDYT.md`, `archiwum/historia/WYNIKI_NBODY.md` |
| **Pamięć / karmienie** | kanon | `docs/PAMIEC_I_KARMIENIE.md` |

## Mapa (skrót)

| Obszar | Pliki |
|--------|--------|
| Silnik | wzór `breit_kernel.py`; szkielet `bridge.py`, `recall_model.py`, `lm_tiny.py`; seal `most_seal.py`; `config.py` |
| Dane Φ | `phi_schema.py`, `synthetic_phi.py`, `phi_dump.py` |
| Fizyka | `nbody_data.py`, `bench_nbody.py` |
| Faza E | `live_store.py`, `gen_head.py`, `mazur_neural.py` |

Pełna mapa: [`docs/MODULY.md`](docs/MODULY.md)

## Start (PowerShell)

```powershell
cd "C:\Users\drwis\model_Ai_breit"
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote
python phimost_learn.py --src dumps/from_struktura --skip-load --skip-train --recall-id C/2p/m-1/up
python doswiadczenie/testy/test_not_flat.py; python doswiadczenie/testy/test_mask.py; python doswiadczenie/testy/test_memory_bus.py
```

**Obsługa:** [`docs/INSTRUKCJA.md`](docs/INSTRUKCJA.md) · **Nauka:** [`docs/LEARN.md`](docs/LEARN.md) · **Glify/alfabety:** [`docs/GLYPHS.md`](docs/GLYPHS.md) · **vs LLM:** [`docs/POROWNANIE_MODELE.md`](docs/POROWNANIE_MODELE.md).  
**Doświadczenie:** [`doswiadczenie/README.md`](doswiadczenie/README.md) · testy: [`doswiadczenie/testy/`](doswiadczenie/testy/).  
**Archiwum** (historia faz, ZdjęcieW, tory boczne): [`archiwum/README.md`](archiwum/README.md).  
Benchi kanonu: `bench_phi.py`, `bench_recall.py`, `bench_dump.py`, `bench_nbody.py`, `bench_gen.py`.

## Konfiguracja i flagi

- Kanon kodu: `config.py` — `softplus×2`, `n_dim=6` (dim5 = **mazur_tracer**)  
- Product OFF: `PHIMOST_LIVE`, `PHIMOST_GEN`, `PHIMOST_NEURAL_LAMBDA`  

## Powiązania

| Tor | Ścieżka |
|-----|---------|
| Lab Most | `C:\Users\drwis\mcps\moje badania\` (`most.py`, `RAPORT.md`) |
| Mazur produkt | `KarmazynOs\software\mazur_crystal\` |
| Agent SE | `C:\Users\drwis\Karmin_Ae` · komora `Most_Lorentz` |
| Plan | [`docs/plan.txt`](docs/plan.txt) |

## Status planu

P1 + **P2 zamknięte** (λ_neural · BubbleFS · dekorelacja C · gen produkt).  
Plan: [`docs/PLAN_NEXT.md`](docs/PLAN_NEXT.md) · status: [`docs/STATUS.md`](docs/STATUS.md).  
Artefakty faz: [`archiwum/historia/`](archiwum/historia/).

```powershell
# Uniwersalne ładowanie + nauka (PowerShell)
# domyślnie TYLKO Most (szybko); Softmax/ZERO: dodaj --baselines
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote --baselines
python phimost_learn.py --src .\ksiazka.md --no-warm --no-promote
python phimost_day.py --query main -k 5
```
