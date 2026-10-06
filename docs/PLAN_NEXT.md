# Plan dalej — PhiMost (po A–E + n-ciał)

**Data:** 2026-09-09  
**Root:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz`

## Co już jest (DONE)

1. **Faza A** — kontrakt Φ, test „plik ≠ płaski”
2. **Faza B** — Most > Softmax na Φ-drzewie (+3.5…+17 pkt)
3. **Faza C** — recall r@1 Most ≫ Softmax (+71 pkt), Most ≈ R
4. **Faza D** — offline dump `textfile_phi`, Most−SM +2.3 pkt
5. **Faza E** — live RO + gen head + λ_neural (flagi OFF default)
6. **Audyt P0** + external review fixes
7. **n-ciał** — solar +26.5, three +21.9 (PASS 2/2)
8. **Kanon** pamięci M1/M2/M3 + zakaz EMA-S / obs-in-S

## Elementy niezbędne do prawidłowej pracy AI

| Warstwa | Element | Status | Po co |
|---------|---------|--------|-------|
| **Silnik** | Most NN (`bridge` / `recall_model`) softplus×2 + Softmax twin + ZERO | jest | myśli zamkiem Lorentza |
| **Kanon** | `config.CANON` (d, n_dim=6, bramki, zakazy) | jest | nie rozjechać empirii |
| **Dane Φ** | `phi_schema` → content `x` + tracer `τ` rozdzielone | jest (częściowo uczciwe) | Most bez τ degeneruje |
| **M2 dump** | `dumps/textfile_phi` (atomy, edges, manifest) | jest | przestrzeń do recall |
| **M1 wagi** | `checkpoints/` load/save | **jest** | trwała pamięć sieci |
| **Karmienie** | `atom_content_vec(S/E/payload)` w dump recall | **jest** | uczciwy sygnał treści |
| **Maski** | honorowanie `mask` w InfoNCE / metrics | **jest** | poprawne r@k na paddingu |
| **Fasada** | cienki MemoryBus: ingest → encode → recall | **jest** | jeden tor produktowy |
| **Epizody** | JSONL obok dumpa (mirror M2) | **jest** | ślad bez Holona |
| **M3 sesja** | Karmin_Ae / `Most_Lorentz` | osobny tor | kontekst agenta, nie wagi |
| **Kontrole** | Softmax twin + ZERO + flat ablacje | jest | regresja / degeneracja |

**Niezbędne minimum runtime (dziś):** silnik + schema + dump + kanon.  
**Niezbędne minimum „prawdziwego AI” (docelowo):** powyższe + **checkpoints** + **uczciwy content** + **mask** + **MemoryBus**.

## Plan w punktach (kolejność)

### P0 — porządek (ten sprint)
1. Katalog `archiwum/historia/` na wyniki faz / logi / wykresy — **done**
2. Benchy piszą artefakty do `archiwum/historia/` — **done**
3. Docs wskazują `archiwum/historia/` — **done**

### P1 — karmienie i trwałość — **DONE**
4. ~~Dump recall: `atom_content_vec`~~ (`dump_recall.py`, test strip payload)
5. ~~`mask` w InfoNCE / metrics / attn~~ (`recall_model.py`, `doswiadczenie/testy/test_mask.py`)
6. ~~`checkpoints/` save/load~~ (`checkpoint.py`, zapis z `bench_dump` / smoke)
7. ~~Smoke load → recall bez re-treningu~~ (`smoke_checkpoint.py` PASS, Δ Most−SM ≈ +4.6)

### P1 — fasada produktowa — **DONE**
8. ~~`MemoryBus`~~ (`memory_bus.py`)
9. ~~JSONL epizodów~~ (`dumps/episodes.jsonl`, `--log-episode`)
10. ~~CLI recall~~ (`phimost_recall.py`)

### P2 — **DONE**
11. ~~λ_neural + autoload ckpt~~ (`mazur_neural.ensure_encoder`)
12. ~~Importer BubbleFS `*.bbl`~~ (`bubblefs_import.py` → `dumps/from_bubblefs`)
13. ~~Dekorelacja osi Fazy C~~ (`decorrelate_axes=True`; |corr| depth↔T ~0.08 vs ~1.0)
14. ~~Gen head produkt~~ (`phimost_gen.py`, `checkpoints/most_gen_head.pt`, target=content)

## Zakazy (nie wracać)

- EMA-S / Most+S / obserwator w superpozycji
- Mieszanie M1 ↔ M2 ↔ M3
- Atomstore jako kanon danych
- Live mutujący Store przy flagach OFF
- Softmax-only „bo prościej”; τ jako handicap

## Kryteria done dla „AI działa”

1. Encoder Most ładuje się z `checkpoints/`  
2. Content dumpa pochodzi z atomów Φ (S/E/payload), nie z topic cheat  
3. Recall na dumpie bije Softmax (gate +2 pkt r@1) **po load**, bez treningu w sesji  
4. MemoryBus ma jeden happy-path: dump → encode → top-k  
5. Softmax twin + ZERO nadal w benchu regresji  

**P1 + P2 + U1–U3 zamknięte.**  
Nauka: `python phimost_learn.py --src … --no-warm --no-promote`.  
Dzień: `python phimost_day.py`.  

### Opcjonalny polish (nieblokujący)
- re-bench C → `archiwum/historia/`
- eksport z pełnego `KarmazynOS.bubbles`
- PDF/EPUB w `import_literature` (dziś `.md`/`.txt`)
- `--baselines` na większych dumpach struktura (gate ΔSM)

---

## Tor glify / alfabety (od 2026-09-10)

**Teza:** zdjęcie → atomy → nauka ze stanu `(S,E,T)`; uniwersalny alfabet.  
**Notatki (dla potomności):** [`GLYPHS.md`](GLYPHS.md) · labnote [`../doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md`](../doswiadczenie/glify/LABNOTE_GLYPHS_2026-09-10.md) · hub [`../doswiadczenie/README.md`](../doswiadczenie/README.md)

| PR | Status | Treść |
|----|--------|--------|
| G0 | **DONE PASS** | `make_dump_glyphs` + learn baselines · `most_glyphs.pt` |
| G1 | **DONE** | `glyph_transcribe.py` — zdjęcie → PhiJSON · demo `dumps/from_photo` |
| G2 | **DONE best-seed** | parity v2 PASS seed0; **nie** multi-seed · dalej stabilność / korpus |
| G3 | opc. | Luneta Store → dump (nie screen) |
| D\* | zawieszone | DESI jako test geometrii — nie fundament alfabetu |

---

## Kolejka 2026-09-16 (po porządkach docs)

Stan żywy: [`STATUS.md`](STATUS.md) · mapa empirii: [`../doswiadczenie/glify/README.md`](../doswiadczenie/glify/README.md)

| # | Item | Status |
|---|------|--------|
| 1 | Mendelejew P0 (parity Most≈SM, nonflat 3/3) | **DONE** |
| 2 | Hold-out Na (mini) | **DONE negatyw** |
| 3 | `run_mendelejew_full.py --most-only` (118) | **DONE** 2026-10-06 na jądrze Breita · r@1=0,575 · floor 3/3 · 768 s |
| 4 | Hold-out Na/K na FULL | po #3 |
| 5 | Classical Softmax loader (nie twin-on-Φ) | **TODO** TRANSCODE §2 |
| 6 | Multi-seed G2 / bogatszy korpus | otwarte |
| 7 | hologram-E (period×group×block) | otwarte |
| 8 | Track `hologram-znaki-raport.md` | w `docs/` |
| 9 | JoannaM spektroskopia | TODO checklist |
| 10 | Qumran | później |

**Świadomie nie mylić torów:** lingua-znaki (warstwa Store) ≠ Holo-Φ (τ Unihan) ≠ Mendelejew (struktura Φ) ≠ DESI.

---

## Kolejka 2026-10-05 (przegląd kodu Mostu, commit 24afb2a)

Zrobione: κ = max(C)/ΣC, `breit_wigner_weights` (γ w liczniku skraca się w normalizacji), maski i seal raz na forward (`key_keep` w `bridge.py`, używane też w `recall_model.py`), `train_eval` podaje `batch.mask`. Wagi zgodne ze starą ścieżką (test w `test_breit_kernel.py`), forward+backward CPU 590 → 423 ms.

| # | Item | Status |
|---|------|--------|
| 1 | CLS ma własną energię (tracer 0 → E_CLS = bias `e`), Softmax nie — opisać przy porównaniach albo zablokować bias dla CLS i sprawdzić TOST | otwarte |
| 2 | Ostatnia warstwa: tylko wiersz CLS (≈ połowa uwagi przy 2 warstwach) — za flagą, bo `w_last` jest używane w testach | otwarte |
| 3 | `torch.compile` na jądrze (elementwise N² ≈ ¾ czasu) — na Windows/CPU wymaga kompilatora C++ | do sprawdzenia |
| 4 | `lm_tiny.py` dalej woła `breit_wigner_rows` + ręczną normalizację — przejść na `breit_wigner_weights` | drobne |
| 5 | `w_last` nie jest `detach()` (`dE_last` jest); `mean_dE` uśrednia przekątną, CLS i pad | drobne |
