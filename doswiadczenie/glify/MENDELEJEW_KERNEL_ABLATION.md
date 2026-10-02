# Mendelejew KERNEL — czysta ablacja jądra uwagi (Softmax ↔ Most)

Zmienia się **wyłącznie funkcja wag uwagi**. Te same dane (zamrożona specyfikacja COMPREL FROZEN, SHA-256 `3d0af9a03a27edfb…`), te same reprezentacje (treść w tokenach, tracer tylko przez ΔE), te same moduły i parametry (26084 parametrów w każdym jądrze), identyczna inicjalizacja przy tym samym ziarnie (sprawdzone: state_dict równy dla wszystkich 4 jąder → **tak**), ta sama procedura (800 kroków, 20 przebiegów). Skrypt `archiwum/skrypty/run_mendelejew_kernel_ablation.py`; każde uruchomienie w `kernel_ablation_runs.jsonl`.

| | ΔE wyłączone | ΔE włączone |
|---|---|---|
| **Softmax** | `sm·0`: softmax(S) | `sm·ΔE`: softmax(S − ΔE²/γ0²) |
| **Most (Breit–Wigner)** | `bw·0`: γC/γ², C = softplus(S)² | `bw·ΔE`: γC/(γ²+ΔE²) |

**Kontrola odtworzenia:** `bw·ΔE` i `bw·0` vs dawne `most`/`zero` z COMPREL FROZEN — identycznych list pozycji: 1440/1440.

## 1. Średnie hit@1 (± odch. std. po przebiegach)

| miara | wariant | bw·ΔE | bw·0 | sm·ΔE | sm·0 |
|---|---|---:|---:|---:|---:|
| A_dół | comp | 0.309 ± 0.070 | 0.263 ± 0.096 | 0.275 ± 0.089 | 0.253 ± 0.088 |
| A_dół | comp+mix | 0.379 ± 0.105 | 0.243 ± 0.084 | 0.346 ± 0.105 | 0.266 ± 0.100 |
| A_dół | comp+rel·PoE | 0.388 ± 0.092 | 0.313 ± 0.115 | 0.299 ± 0.085 | 0.303 ± 0.127 |
| B_te_hop | comp | 0.315 ± 0.057 | 0.257 ± 0.063 | 0.282 ± 0.075 | 0.243 ± 0.062 |
| B_te_hop | comp+mix | 0.373 ± 0.089 | 0.245 ± 0.069 | 0.332 ± 0.079 | 0.257 ± 0.076 |
| B_te_hop | comp+rel·PoE | 0.373 ± 0.080 | 0.284 ± 0.091 | 0.306 ± 0.076 | 0.283 ± 0.097 |
| B_te_dd | comp | 0.227 ± 0.092 | 0.236 ± 0.074 | 0.175 ± 0.108 | 0.194 ± 0.084 |
| B_te_dd | comp+mix | 0.290 ± 0.116 | 0.222 ± 0.062 | 0.239 ± 0.109 | 0.212 ± 0.088 |
| B_te_dd | comp+rel·PoE | 0.370 ± 0.106 | 0.306 ± 0.112 | 0.261 ± 0.099 | 0.331 ± 0.127 |
| B_tr_op | comp | 0.764 ± 0.065 | 0.559 ± 0.078 | 0.785 ± 0.061 | 0.575 ± 0.080 |
| B_tr_op | comp+mix | 0.759 ± 0.075 | 0.470 ± 0.093 | 0.810 ± 0.064 | 0.496 ± 0.096 |
| B_tr_op | comp+rel·PoE | 0.742 ± 0.051 | 0.544 ± 0.091 | 0.749 ± 0.088 | 0.543 ± 0.096 |
| B_tr_dd | comp | 0.870 ± 0.106 | 0.585 ± 0.115 | 0.930 ± 0.070 | 0.601 ± 0.146 |
| B_tr_dd | comp+mix | 0.838 ± 0.108 | 0.524 ± 0.173 | 0.852 ± 0.097 | 0.524 ± 0.158 |
| B_tr_dd | comp+rel·PoE | 0.871 ± 0.095 | 0.628 ± 0.161 | 0.895 ± 0.090 | 0.671 ± 0.122 |

## 2. KONTRAST GŁÓWNY — samo jądro przy tej samej informacji: bw·ΔE − sm·ΔE (15 testów, Holm)

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|---:|:-:|
| comp | A_dół | +0.034 | [+0.003, +0.065] | 11/6 | 0.0550 | 0.3302 | nie |
| comp | B_te_hop | +0.033 | [+0.008, +0.056] | 14/3 | 0.0178 | 0.1955 | nie |
| comp | B_te_dd | +0.052 | [+0.015, +0.089] | 13/5 | 0.0150 | 0.1803 | nie |
| comp | B_tr_op | -0.021 | [-0.054, +0.008] | 9/11 | 0.2155 | 1.0000 | nie |
| comp | B_tr_dd | -0.060 | [-0.111, -0.016] | 4/11 | 0.0189 | 0.1955 | nie |
| comp+mix | A_dół | +0.033 | [-0.013, +0.080] | 10/8 | 0.2034 | 1.0000 | nie |
| comp+mix | B_te_hop | +0.042 | [+0.007, +0.077] | 13/6 | 0.0372 | 0.2601 | nie |
| comp+mix | B_te_dd | +0.051 | [+0.011, +0.094] | 14/6 | 0.0270 | 0.2386 | nie |
| comp+mix | B_tr_op | -0.051 | [-0.088, -0.011] | 6/14 | 0.0265 | 0.2386 | nie |
| comp+mix | B_tr_dd | -0.014 | [-0.062, +0.035] | 7/8 | 0.5883 | 1.0000 | nie |
| comp+rel·PoE | A_dół | +0.088 | [+0.068, +0.109] | 19/0 | 0.0000 | 0.0001 | **tak** |
| comp+rel·PoE | B_te_hop | +0.067 | [+0.047, +0.088] | 18/2 | 0.0000 | 0.0002 | **tak** |
| comp+rel·PoE | B_te_dd | +0.109 | [+0.077, +0.140] | 18/1 | 0.0000 | 0.0002 | **tak** |
| comp+rel·PoE | B_tr_op | -0.007 | [-0.048, +0.033] | 6/9 | 0.7539 | 1.0000 | nie |
| comp+rel·PoE | B_tr_dd | -0.025 | [-0.066, +0.019] | 4/9 | 0.2776 | 1.0000 | nie |

## 3. Jądro bez informacji ΔE: bw·0 − sm·0 (15 testów, Holm)

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|---:|:-:|
| comp | A_dół | +0.010 | [-0.013, +0.034] | 8/6 | 0.4431 | 1.0000 | nie |
| comp | B_te_hop | +0.014 | [-0.001, +0.029] | 12/7 | 0.0888 | 1.0000 | nie |
| comp | B_te_dd | +0.042 | [+0.016, +0.068] | 13/4 | 0.0077 | 0.1151 | nie |
| comp | B_tr_op | -0.016 | [-0.052, +0.020] | 9/11 | 0.4082 | 1.0000 | nie |
| comp | B_tr_dd | -0.016 | [-0.058, +0.027] | 7/9 | 0.5085 | 1.0000 | nie |
| comp+mix | A_dół | -0.023 | [-0.047, -0.002] | 4/9 | 0.0615 | 0.8613 | nie |
| comp+mix | B_te_hop | -0.012 | [-0.032, +0.008] | 7/11 | 0.2615 | 1.0000 | nie |
| comp+mix | B_te_dd | +0.010 | [-0.018, +0.036] | 10/6 | 0.5018 | 1.0000 | nie |
| comp+mix | B_tr_op | -0.026 | [-0.061, +0.009] | 6/11 | 0.1739 | 1.0000 | nie |
| comp+mix | B_tr_dd | +0.000 | [-0.059, +0.063] | 6/8 | 0.9954 | 1.0000 | nie |
| comp+rel·PoE | A_dół | +0.010 | [-0.027, +0.048] | 9/8 | 0.6111 | 1.0000 | nie |
| comp+rel·PoE | B_te_hop | +0.001 | [-0.027, +0.029] | 10/9 | 0.9412 | 1.0000 | nie |
| comp+rel·PoE | B_te_dd | -0.024 | [-0.064, +0.013] | 8/11 | 0.2476 | 1.0000 | nie |
| comp+rel·PoE | B_tr_op | +0.001 | [-0.024, +0.025] | 10/8 | 0.9675 | 1.0000 | nie |
| comp+rel·PoE | B_tr_dd | -0.043 | [-0.094, +0.004] | 6/9 | 0.1107 | 1.0000 | nie |

## 4. Efekt informacji ΔE w jądrze Mostu: bw·ΔE − bw·0 (15 testów, Holm)

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|---:|:-:|
| comp | A_dół | +0.046 | [+0.003, +0.091] | 11/6 | 0.0555 | 0.1111 | nie |
| comp | B_te_hop | +0.059 | [+0.029, +0.089] | 16/4 | 0.0014 | 0.0086 | **tak** |
| comp | B_te_dd | -0.009 | [-0.050, +0.029] | 10/9 | 0.6752 | 0.6752 | nie |
| comp | B_tr_op | +0.205 | [+0.157, +0.248] | 19/1 | 0.0000 | 0.0001 | **tak** |
| comp | B_tr_dd | +0.285 | [+0.210, +0.360] | 17/0 | 0.0000 | 0.0002 | **tak** |
| comp+mix | A_dół | +0.136 | [+0.093, +0.180] | 17/2 | 0.0000 | 0.0003 | **tak** |
| comp+mix | B_te_hop | +0.128 | [+0.096, +0.161] | 18/0 | 0.0000 | 0.0001 | **tak** |
| comp+mix | B_te_dd | +0.069 | [+0.024, +0.117] | 13/4 | 0.0112 | 0.0447 | **tak** |
| comp+mix | B_tr_op | +0.289 | [+0.245, +0.333] | 20/0 | 0.0000 | 0.0000 | **tak** |
| comp+mix | B_tr_dd | +0.313 | [+0.244, +0.384] | 19/0 | 0.0000 | 0.0000 | **tak** |
| comp+rel·PoE | A_dół | +0.075 | [+0.032, +0.117] | 15/4 | 0.0034 | 0.0170 | **tak** |
| comp+rel·PoE | B_te_hop | +0.088 | [+0.056, +0.121] | 18/2 | 0.0001 | 0.0005 | **tak** |
| comp+rel·PoE | B_te_dd | +0.063 | [+0.015, +0.110] | 14/5 | 0.0213 | 0.0640 | nie |
| comp+rel·PoE | B_tr_op | +0.198 | [+0.154, +0.243] | 20/0 | 0.0000 | 0.0000 | **tak** |
| comp+rel·PoE | B_tr_dd | +0.242 | [+0.167, +0.313] | 19/1 | 0.0000 | 0.0002 | **tak** |

## 5. Efekt informacji ΔE w jądrze Softmax: sm·ΔE − sm·0 (15 testów, Holm)

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|---:|:-:|
| comp | A_dół | +0.022 | [-0.020, +0.069] | 7/11 | 0.3621 | 1.0000 | nie |
| comp | B_te_hop | +0.039 | [+0.008, +0.072] | 14/6 | 0.0298 | 0.1785 | nie |
| comp | B_te_dd | -0.019 | [-0.072, +0.032] | 7/11 | 0.4914 | 1.0000 | nie |
| comp | B_tr_op | +0.210 | [+0.166, +0.253] | 19/1 | 0.0000 | 0.0001 | **tak** |
| comp | B_tr_dd | +0.329 | [+0.251, +0.402] | 18/0 | 0.0000 | 0.0001 | **tak** |
| comp+mix | A_dół | +0.080 | [+0.030, +0.128] | 15/4 | 0.0062 | 0.0435 | **tak** |
| comp+mix | B_te_hop | +0.074 | [+0.038, +0.106] | 17/3 | 0.0010 | 0.0090 | **tak** |
| comp+mix | B_te_dd | +0.028 | [-0.017, +0.070] | 10/5 | 0.2392 | 0.9568 | nie |
| comp+mix | B_tr_op | +0.314 | [+0.269, +0.356] | 20/0 | 0.0000 | 0.0000 | **tak** |
| comp+mix | B_tr_dd | +0.327 | [+0.246, +0.407] | 18/0 | 0.0000 | 0.0001 | **tak** |
| comp+rel·PoE | A_dół | -0.003 | [-0.045, +0.038] | 9/9 | 0.8772 | 1.0000 | nie |
| comp+rel·PoE | B_te_hop | +0.022 | [-0.006, +0.052] | 11/8 | 0.1628 | 0.8141 | nie |
| comp+rel·PoE | B_te_dd | -0.070 | [-0.114, -0.028] | 4/12 | 0.0048 | 0.0381 | **tak** |
| comp+rel·PoE | B_tr_op | +0.205 | [+0.148, +0.261] | 18/2 | 0.0000 | 0.0001 | **tak** |
| comp+rel·PoE | B_tr_dd | +0.224 | [+0.157, +0.288] | 17/1 | 0.0000 | 0.0002 | **tak** |

## 6. Interakcja: (bw·ΔE − sm·ΔE) − (bw·0 − sm·0) — czy jądro BW korzysta z ΔE lepiej niż Softmax

| wariant | miara | Δ średnia | 95% CI | p |
|---|---|---:|---|---:|
| comp | A_dół | +0.024 | [-0.013, +0.060] | 0.2151 |
| comp | B_te_hop | +0.019 | [-0.006, +0.043] | 0.1616 |
| comp | B_te_dd | +0.011 | [-0.040, +0.059] | 0.6864 |
| comp | B_tr_op | -0.006 | [-0.060, +0.047] | 0.8470 |
| comp | B_tr_dd | -0.044 | [-0.126, +0.024] | 0.3074 |
| comp+mix | A_dół | +0.056 | [+0.001, +0.112] | 0.0720 |
| comp+mix | B_te_hop | +0.054 | [+0.010, +0.095] | 0.0289 |
| comp+mix | B_te_dd | +0.041 | [-0.013, +0.100] | 0.1796 |
| comp+mix | B_tr_op | -0.025 | [-0.070, +0.023] | 0.3224 |
| comp+mix | B_tr_dd | -0.014 | [-0.080, +0.052] | 0.6902 |
| comp+rel·PoE | A_dół | +0.078 | [+0.040, +0.113] | 0.0009 |
| comp+rel·PoE | B_te_hop | +0.066 | [+0.037, +0.092] | 0.0005 |
| comp+rel·PoE | B_te_dd | +0.133 | [+0.093, +0.176] | 0.0000 |
| comp+rel·PoE | B_tr_op | -0.007 | [-0.054, +0.044] | 0.7763 |
| comp+rel·PoE | B_tr_dd | +0.018 | [-0.050, +0.090] | 0.6179 |

## 7. Dawny bliźniak Softmax (bez tracera, inne parametry) vs dopasowany `sm·0` i `sm·ΔE` — średnie hit@1

| wariant | miara | dawny softmax | sm·0 | sm·ΔE | bw·ΔE (Most) |
|---|---|---:|---:|---:|---:|
| comp | A_dół | 0.276 | 0.253 | 0.275 | 0.309 |
| comp | B_te_hop | 0.254 | 0.243 | 0.282 | 0.315 |
| comp | B_te_dd | 0.209 | 0.194 | 0.175 | 0.227 |
| comp | B_tr_op | 0.573 | 0.575 | 0.785 | 0.764 |
| comp | B_tr_dd | 0.597 | 0.601 | 0.930 | 0.870 |
| comp+mix | A_dół | 0.262 | 0.266 | 0.346 | 0.379 |
| comp+mix | B_te_hop | 0.257 | 0.257 | 0.332 | 0.373 |
| comp+mix | B_te_dd | 0.235 | 0.212 | 0.239 | 0.290 |
| comp+mix | B_tr_op | 0.491 | 0.496 | 0.810 | 0.759 |
| comp+mix | B_tr_dd | 0.537 | 0.524 | 0.852 | 0.838 |
| comp+rel·PoE | A_dół | 0.305 | 0.303 | 0.299 | 0.388 |
| comp+rel·PoE | B_te_hop | 0.275 | 0.283 | 0.306 | 0.373 |
| comp+rel·PoE | B_te_dd | 0.294 | 0.331 | 0.261 | 0.370 |
| comp+rel·PoE | B_tr_op | 0.576 | 0.543 | 0.749 | 0.742 |
| comp+rel·PoE | B_tr_dd | 0.696 | 0.671 | 0.895 | 0.871 |

> **UZUPEŁNIENIE (2026-10-02, [`MENDELEJEW_KERNEL_TOK.md`](MENDELEJEW_KERNEL_TOK.md)):** gdy Softmax dostaje τ w tokenach (standardowo) zamiast kary −ΔE²/γ², jest lepszy lub równy kanonicznemu Mostowi w 13/15 porównań, a różnica jąder przy τ w tokenach wynosi 0 (30/30 nieistotnych). Zysk jądra BW w PoE opisany niżej dotyczy wyłącznie słabego sposobu podania ΔE Softmaxowi.

> **UZUPEŁNIENIE 2 ([`CROSSDOMAIN_COMP.md`](CROSSDOMAIN_COMP.md)):** w ścisłym protokole (cechy elementów testowych niewidoczne w treningu) zysk jądra BW w PoE znika także w układzie okresowym (+0.003); w muzyce Softmax jest lepszy.

## Wnioski

**Ważność porównania:**
- Inicjalizacja jest identyczna dla 4 jąder (state_dict równy, 26 084 parametry).
- `bw·ΔE` i `bw·0` odtwarzają dawne Most i ZERO co do każdej pozycji (1440/1440).
- Kontrast główny i rodzina 15 testów zostały zadeklarowane przed uruchomieniem.

1. **Dawne porównanie „Most vs Softmax” mieszało dwa czynniki: jądro i dostęp do informacji z tracera.** Dawny bliźniak Softmax nie widział ΔE wcale. Jego wyniki pokrywają się z dopasowanym `sm·0` (np. comp+mix B_tr_op: 0.491 vs 0.496), więc to porównanie mierzyło głównie **informację**, nie jądro.
2. **Kompozycja na znanych pierwiastkach (B_tr): przewaga pochodzi z ΔE, nie z jądra.**
   - ΔE podnosi wynik w **obu** jądrach o ~0.20–0.33 (17–20/20, p_Holm ≤ 0.0002).
   - Przy tej samej informacji Softmax jest **równy lub lepszy**: `sm·ΔE` 0.79 / 0.93 wobec `bw·ΔE` 0.76 / 0.87 (comp, operator / dół∘dół). Żadna różnica nie przeżywa korekty.
   - Najmocniejsze dotąd twierdzenie serii („Most > wszyscy w kompozycji na znanych, 17–20/20”) to **efekt informacji ΔE**, a nie jądra Breita–Wignera.
3. **Na nowych pierwiastkach jądro pomaga, ale tylko w jednym wariancie.**

   | bw·ΔE − sm·ΔE (nowe pierwiastki) | comp | comp+mix | comp+rel·PoE |
   |---|---:|---:|---:|
   | krok „dół” | +0.03 (nie) | +0.03 (nie) | **+0.09 (19/0, p_Holm = 0.0001)** |
   | 2 kroki | +0.03 (nie) | +0.04 (nie) | **+0.07 (18/2, p_Holm = 0.0002)** |
   | dół∘dół | +0.05 (nie) | +0.05 (nie) | **+0.11 (18/1, p_Holm = 0.0002)** |

   W `comp` i `comp+mix` znak jest dodatni (CI zwykle nad zerem), ale po korekcie Holma efekt nie przeżywa. W `comp+rel·PoE` przewaga jądra jest wyraźna.
4. **Interakcja mówi, skąd ta różnica.** W PoE jądro BW korzysta z ΔE lepiej niż Softmax: +0.07…+0.13, p ≤ 0.001. Częściowo dlatego, że **Softmax z ΔE w tym wariancie traci** (`sm·ΔE − sm·0` w dół∘dół: −0.07, p_Holm = 0.04). Kara ΔE w wykładniku softmaxu koliduje z głowicą grupy, a lorentzowski kształt BW nie.
5. **Bez informacji ΔE jądra są równoważne** (`bw·0 − sm·0`: żadna z 15 różnic nie przeżywa korekty, |Δ| ≤ 0.04). Sama postać C = softplus(S)² zamiast exp(S) niczego nie wnosi.

**Zastrzeżenia:**
- (a) `sm·ΔE` to **jeden** sposób podania ΔE Softmaxowi (kara Gaussa −ΔE²/γ0² z tą samą szerokością). Inne sposoby, np. ΔE w tokenach albo bias liniowy, mogą dać Softmaxowi więcej.
- (b) Hiperparametry kanonu (lr, wymiary, temperatura) stroiono historycznie pod Most, co może sprzyjać jądru BW.
- (c) Ocena dotyczy tylko zadań tej serii (ZC, operatory „dół”/„prawo”).

## Werdykt

Czy samo jądro uwagi wnosi korzyść? **Przeważnie nie.** Większość dotychczasowej przewagi Mostu nad Softmaxem w tej serii wynikała z tego, że Most dostawał informację z tracera (ΔE), a bliźniak Softmax jej nie dostawał. Przy tej samej informacji:
- w kompozycji na znanych pierwiastkach Softmax dorównuje Mostowi;
- na nowych pierwiastkach jądro Breita–Wignera daje zysk +0.07…+0.11 **tylko** w połączeniu z głowicą grupy (PoE), gdzie lepiej niż Softmax łączy ΔE z drugą relacją;
- w pozostałych wariantach zysk jest mały (+0.03…+0.05) i nie przeżywa korekty.

**Konsekwencja dla wcześniejszych raportów:** każde porównanie Most vs Softmax z tej serii należy czytać jako „jądro + informacja ΔE vs brak ΔE”. Czysty efekt jądra mierzy tylko ten raport.

