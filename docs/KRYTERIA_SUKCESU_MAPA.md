# Mapa kryteriów sukcesu → Most / Φ

**Źródło:** [`kryteria_sukcesu.txt`](kryteria_sukcesu.txt)  
**Werdykt:** akceptuję je jako **długoterminowy gate skalowania** (produkt Most na torze LLM), **nie** jako exit-gate dzisiejszego labu PhiMost (A–E + empiria Φ).  
**FULL Mendelejew:** odłożony do decyzji po tej mapie (dump gotowy: `dumps/from_mendelejew_full`).

---

## Twarde kryteria (kanon zewnętrzny)

1. **Skalowanie błędu (100M–300M):** krzywa loss ≥ stromość Llama-style przy tej samej ilości danych.  
2. **Stabilność BF16:** brak NaN; bez ucieczki do FP32.  
3. **Kompromis sprzętowy:** mniejsza przepustowość vs FlashAttention **wymiernie** rekompensowana niższym VRAM (np. KV).  
4. **Długi kontekst (16k+):** Needle-in-a-Haystack — brak amnezji w środku.  
5. **Minimalny dowód:** działający PyTorch + odtwarzalne logi treningu.

---

## Mapowanie 1:1 na Most / Φ

| # | Kryterium | Co to znaczy u Mosta | Stan dziś (lab) | Co trzeba zbudować / zmierzyć |
|---|-----------|----------------------|-----------------|-------------------------------|
| **1** | Skalowanie loss 100M–300M vs Llama | Most (softplus×2 + τ) jako **uwaga** w transformerze tej skali; ten sam korpus/tokeny co baseline Softmax/Llama-attn | **CZĘŚCIOWO** — tiny (~116k) + medium (~3.3M) **DONE**; 100–300M **NIE** (brak CUDA) | (a) ten sam `lm_tiny` stack, (b) większy korpus, (c) loss @ 100M+, (d) Llama-attn / NVIDIA |
| **2** | Stabilność BF16 | Forward/backward zamka Lorentza + softplus² w bf16 bez NaN/Inf | **DONE (gate)** — `precision.py` + autocast bf16 + finite-check; `run_bf16_gate.py` **PASS** na FULL118 (16 ep); testy `test_bf16.py` | Utrzymać gate przy skalowaniu (#1); lab default nadal FP32 |
| **3** | VRAM vs FlashAttention | Kernel/ścieżka Most nie musi bić FA w tok/s, ale **musi** pokazać niższy VRAM (KV / kompresja / struktura Φ) przy porównywalnym zadaniu | **NIE** — brak kernela FA, brak pomiaru VRAM vs FA | (a) baseline FA (PyTorch SDPA / FA2), (b) Most attn path, (c) tabela: tok/s · peak VRAM · długość kontekstu, (d) jawny „trade accepted” |
| **4** | Needle 16k+ | Długi kontekst z τ/Φ: igła w środku sekwencji odzyskiwalna | **NIE** — MVP = recall/ranking nad atomami, nie LM 16k | (a) zadanie needle (synthetic lub tekst), (b) sekwencje ≥16k, (c) metryka retrieval @ pozycjach head/mid/tail, (d) ablacje τ=0 vs Most |
| **5** | Kod + logi odtwarzalne | PyTorch Most działa; trening ma seed, komendę, artefakt | **TAK + wzmocnione** — `runlog.RunDir` → `runs/<UTC>_<name>/` (meta, command, hparams, loss.csv, metrics); podpięte: `phimost_learn`, `run_mendelejew_full` | Utrzymać run-dir na kolejnych torach skali (#2→#1) |

---

## Co akceptuję / czego nie udaję

| Akceptuję | Nie udaję |
|-----------|-----------|
| Punkty **1–5** jako **gate wejścia w skalę produktową** | Że lab PhiMost „już spełnia” 1–4 |
| Punkt **5** jako **już obowiązujący** w labie (reprodukcja) | Softmax-twin-on-Φ jako FlashAttention / Llama baseline |
| Empiria Φ (P0 Mendelejew parity, nonflat, A–E) jako **fundament pod** skalowanie | Że FULL118 albo glify = dowód 100M / 16k needle |

---

## Dwie warstwy sukcesu (żeby się nie myliły)

### A. Lab Φ (teraz) — kryteria lokalne
- Nonflat ≫ flat; Most formuje się przy τ≠0  
- Parity / gate vs Softmax **tylko** gdy `--baselines` i jasny protokół  
- FULL118 (gdy ruszy): Most-only, `above_floor` vs chance, krzywe r@1 — wg [`../doswiadczenie/glify/JUTRO_MENDELEJEW_FULL.md`](../doswiadczenie/glify/JUTRO_MENDELEJEW_FULL.md)

### B. Skala LLM (później) — `kryteria_sukcesu.txt`
Kolejność sensowna po zamknięciu fundamentu Φ:

1. **#5 wzmocnione** — run-dir + odtwarzalność na jednym torze skali  
2. **#2 BF16** — zamek bez NaN na dłuższym runie  
3. **#1 mały LM** (najpierw ≪100M smoke, potem 100M–300M vs Llama-attn)  
4. **#3 VRAM** — pomiar vs FA przy tym samym context  
5. **#4 Needle 16k+** — dopiero gdy jest LM / długi kontekst

---

## Stan warstw (aktualizacja)

| Warstwa | Stan |
|---------|------|
| **A** FULL118 Most-only | **DONE** (r@1≈0.57, floor 3/3) |
| **A** Hold-out Na/K FULL | **FAIL 2/2** (brak gen. grupy — wynik negatywny OK) |
| **B #5** run-dir | **DONE** — `runlog.py` + `runs/` |
| **B #2** BF16 | **DONE (gate)** — `precision.py` · `--bf16` · `BF16_GATE.md` PASS |
| **B #1a** baselines na skali labu (FULL118) | **DONE** — Most≈SM, gate 0/5, Most wolniejszy; VRAM N/A (CPU) |
| **B #1b** tiny LM smoke ≪1M | **DONE** — [`LM_TINY_1B.md`](../doswiadczenie/glify/LM_TINY_1B.md) |
| **B #1c** medium LM ~3.3M | **DONE** — Most 2.715 vs SM 2.657 · descend · 1.34× · [`LM_MEDIUM_1C.md`](../doswiadczenie/glify/LM_MEDIUM_1C.md) |
| **B #1d** 100–300M vs Llama | **BLOCKED** sprzętowo + **brak trendu mid** z probe 3M/2k |
| **B mech.** lost-in-middle @2k | **FAIL** na default knobs — [`LOST_IN_MIDDLE_PROBE.md`](../doswiadczenie/glify/LOST_IN_MIDDLE_PROBE.md) |
| **B knobs** γ/e_scale/τ | **PASS cienki** — `ultra_wide` dogania SM ent_mid — [`MOST_KNOBS_SWEEP.md`](../doswiadczenie/glify/MOST_KNOBS_SWEEP.md) |
| **B signed ΔE** (LM) | słaby sam; PASS ε ze szerokim g — [`SIGNED_DELTA_ABLATION.md`](../doswiadczenie/glify/SIGNED_DELTA_ABLATION.md) |
| **B signed ΔE** (atomy) | **PARTIAL** — `onesided` r@1 0.591 > square/SM — [`ATOMS_DELTA_ABLATION.md`](../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md) |
| **B #3→#4** VRAM / Needle | zablokowane lokalnie (AMD); #4 nie mylić z proxy 2k |

## Decyzja

Idziemy **planem sukcesu warstwy B**: mid-FAIL na default; **strojenie zamka** pokazuje, że ent_mid da się dogonić Softmax (`ultra_wide`).
**NEXT:** (1) default knobs → `ultra_wide` w eksperymentach mid, (2) twardszy retrieval (secret_acc>0), (3) dopiero potem GPU.
