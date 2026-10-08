# Mendelejew KERNEL·TOK — Softmax z informacją tracera w tokenach; powtórka ablacji jądra

Ta sama klasa enkodera dla wszystkich 8 ramion (26308 parametrów; identyczna inicjalizacja: **tak**; nowy rzut W_τ nie zmienia losowania reszty: **tak**). Dane: zamrożona specyfikacja (SHA-256 `3d0af9a03a27edfb…`), 800 kroków, 20 przebiegów. Skrypt `archiwum/skrypty/run_mendelejew_kernel_tok.py`; nowe uruchomienia w `kernel_tok_runs.jsonl`, dawne ramiona z `kernel_ablation_runs.jsonl`.

**Kontrola odtworzenia dawnych ramion nową klasą (przebieg 0, bw·ΔE i sm·ΔE, 3 tryby):** 72/72 list pozycji identycznych.

`tok` = τ w embeddingu tokenu (x = embed(treść) + W_τ·τ); `ΔE` = ΔE w wagach uwagi.

## 1. Średnie hit@1 (20 przebiegów)

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | sm·tok | bw·tok | sm·tok+ΔE | bw·tok+ΔE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_dół | comp | 0.309 | 0.275 | 0.263 | 0.253 | 0.394 | 0.401 | 0.379 | 0.385 |
| A_dół | comp+mix | 0.379 | 0.346 | 0.243 | 0.266 | 0.452 | 0.457 | 0.457 | 0.454 |
| A_dół | comp+rel·PoE | 0.388 | 0.299 | 0.313 | 0.303 | 0.407 | 0.414 | 0.364 | 0.381 |
| B_te_hop | comp | 0.315 | 0.282 | 0.257 | 0.243 | 0.384 | 0.384 | 0.357 | 0.384 |
| B_te_hop | comp+mix | 0.373 | 0.332 | 0.245 | 0.257 | 0.432 | 0.429 | 0.434 | 0.438 |
| B_te_hop | comp+rel·PoE | 0.373 | 0.306 | 0.284 | 0.283 | 0.400 | 0.406 | 0.354 | 0.365 |
| B_te_dd | comp | 0.227 | 0.175 | 0.236 | 0.194 | 0.232 | 0.237 | 0.241 | 0.260 |
| B_te_dd | comp+mix | 0.290 | 0.239 | 0.222 | 0.212 | 0.337 | 0.331 | 0.314 | 0.333 |
| B_te_dd | comp+rel·PoE | 0.370 | 0.261 | 0.306 | 0.331 | 0.330 | 0.328 | 0.355 | 0.331 |
| B_tr_op | comp | 0.764 | 0.785 | 0.559 | 0.575 | 0.865 | 0.862 | 0.871 | 0.867 |
| B_tr_op | comp+mix | 0.759 | 0.810 | 0.470 | 0.496 | 0.853 | 0.852 | 0.858 | 0.850 |
| B_tr_op | comp+rel·PoE | 0.742 | 0.749 | 0.544 | 0.543 | 0.804 | 0.831 | 0.836 | 0.833 |
| B_tr_dd | comp | 0.870 | 0.930 | 0.585 | 0.601 | 0.888 | 0.885 | 0.884 | 0.904 |
| B_tr_dd | comp+mix | 0.838 | 0.852 | 0.524 | 0.524 | 0.752 | 0.736 | 0.787 | 0.797 |
| B_tr_dd | comp+rel·PoE | 0.871 | 0.895 | 0.628 | 0.671 | 0.768 | 0.793 | 0.840 | 0.848 |

## 2. P1 (główny) — Most vs Softmax z τ w tokenach: bw·ΔE − sm·tok

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | -0.085 | [-0.131, -0.040] | 5/15 | 0.0205 | **tak** |
| comp | B_te_hop | -0.068 | [-0.106, -0.029] | 3/16 | 0.0324 | **tak** |
| comp | B_te_dd | -0.006 | [-0.038, +0.030] | 8/9 | 1.0000 | nie |
| comp | B_tr_op | -0.101 | [-0.137, -0.064] | 2/17 | 0.0014 | **tak** |
| comp | B_tr_dd | -0.018 | [-0.077, +0.039] | 7/8 | 1.0000 | nie |
| comp+mix | A_dół | -0.073 | [-0.112, -0.035] | 3/16 | 0.0196 | **tak** |
| comp+mix | B_te_hop | -0.059 | [-0.084, -0.034] | 2/18 | 0.0024 | **tak** |
| comp+mix | B_te_dd | -0.046 | [-0.099, +0.011] | 4/13 | 0.4772 | nie |
| comp+mix | B_tr_op | -0.094 | [-0.123, -0.064] | 1/19 | 0.0004 | **tak** |
| comp+mix | B_tr_dd | +0.086 | [+0.036, +0.137] | 12/4 | 0.0347 | **tak** |
| comp+rel·PoE | A_dół | -0.020 | [-0.055, +0.016] | 6/10 | 0.9281 | nie |
| comp+rel·PoE | B_te_hop | -0.027 | [-0.052, -0.002] | 5/14 | 0.3247 | nie |
| comp+rel·PoE | B_te_dd | +0.040 | [+0.003, +0.076] | 12/5 | 0.3247 | nie |
| comp+rel·PoE | B_tr_op | -0.062 | [-0.093, -0.029] | 4/15 | 0.0206 | **tak** |
| comp+rel·PoE | B_tr_dd | +0.102 | [+0.043, +0.161] | 13/4 | 0.0347 | **tak** |

## 3. P2 (główny) — samo jądro przy τ w tokenach: bw·tok − sm·tok

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | +0.007 | [-0.016, +0.028] | 11/7 | 1.0000 | nie |
| comp | B_te_hop | +0.001 | [-0.014, +0.014] | 10/7 | 1.0000 | nie |
| comp | B_te_dd | +0.005 | [-0.012, +0.020] | 6/4 | 1.0000 | nie |
| comp | B_tr_op | -0.003 | [-0.018, +0.009] | 8/7 | 1.0000 | nie |
| comp | B_tr_dd | -0.003 | [-0.025, +0.021] | 4/7 | 1.0000 | nie |
| comp+mix | A_dół | +0.004 | [-0.017, +0.027] | 7/6 | 1.0000 | nie |
| comp+mix | B_te_hop | -0.003 | [-0.020, +0.015] | 10/8 | 1.0000 | nie |
| comp+mix | B_te_dd | -0.006 | [-0.032, +0.018] | 8/9 | 1.0000 | nie |
| comp+mix | B_tr_op | -0.002 | [-0.018, +0.015] | 8/8 | 1.0000 | nie |
| comp+mix | B_tr_dd | -0.016 | [-0.051, +0.023] | 4/9 | 1.0000 | nie |
| comp+rel·PoE | A_dół | +0.007 | [-0.010, +0.023] | 10/7 | 1.0000 | nie |
| comp+rel·PoE | B_te_hop | +0.006 | [-0.007, +0.018] | 13/6 | 1.0000 | nie |
| comp+rel·PoE | B_te_dd | -0.002 | [-0.023, +0.019] | 5/7 | 1.0000 | nie |
| comp+rel·PoE | B_tr_op | +0.028 | [+0.009, +0.046] | 13/3 | 0.1456 | nie |
| comp+rel·PoE | B_tr_dd | +0.025 | [-0.006, +0.058] | 9/4 | 1.0000 | nie |

## 4. Jak dać informację Softmaxowi: sm·tok − sm·ΔE

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | +0.119 | [+0.076, +0.160] | 17/2 | 0.0013 | **tak** |
| comp | B_te_hop | +0.102 | [+0.065, +0.139] | 18/2 | 0.0008 | **tak** |
| comp | B_te_dd | +0.058 | [+0.007, +0.108] | 13/7 | 0.1010 | nie |
| comp | B_tr_op | +0.080 | [+0.046, +0.113] | 17/3 | 0.0026 | **tak** |
| comp | B_tr_dd | -0.042 | [-0.093, +0.008] | 4/11 | 0.1251 | nie |
| comp+mix | A_dół | +0.106 | [+0.062, +0.150] | 17/3 | 0.0032 | **tak** |
| comp+mix | B_te_hop | +0.101 | [+0.071, +0.132] | 18/1 | 0.0001 | **tak** |
| comp+mix | B_te_dd | +0.097 | [+0.048, +0.149] | 14/4 | 0.0134 | **tak** |
| comp+mix | B_tr_op | +0.044 | [+0.013, +0.076] | 13/6 | 0.0745 | nie |
| comp+mix | B_tr_dd | -0.100 | [-0.169, -0.031] | 4/13 | 0.0681 | nie |
| comp+rel·PoE | A_dół | +0.108 | [+0.064, +0.151] | 15/3 | 0.0032 | **tak** |
| comp+rel·PoE | B_te_hop | +0.094 | [+0.061, +0.128] | 16/2 | 0.0013 | **tak** |
| comp+rel·PoE | B_te_dd | +0.069 | [+0.031, +0.106] | 14/2 | 0.0163 | **tak** |
| comp+rel·PoE | B_tr_op | +0.055 | [+0.010, +0.099] | 12/6 | 0.1010 | nie |
| comp+rel·PoE | B_tr_dd | -0.127 | [-0.195, -0.058] | 3/15 | 0.0163 | **tak** |

## 5. Jądro przy τ w tokenach + ΔE w uwadze: bw·tok+ΔE − sm·tok+ΔE

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | +0.006 | [-0.028, +0.039] | 9/7 | 1.0000 | nie |
| comp | B_te_hop | +0.027 | [+0.005, +0.049] | 11/7 | 0.4832 | nie |
| comp | B_te_dd | +0.019 | [-0.010, +0.047] | 12/6 | 1.0000 | nie |
| comp | B_tr_op | -0.004 | [-0.024, +0.016] | 9/9 | 1.0000 | nie |
| comp | B_tr_dd | +0.020 | [-0.020, +0.060] | 8/4 | 1.0000 | nie |
| comp+mix | A_dół | -0.003 | [-0.037, +0.032] | 9/9 | 1.0000 | nie |
| comp+mix | B_te_hop | +0.004 | [-0.027, +0.035] | 12/8 | 1.0000 | nie |
| comp+mix | B_te_dd | +0.019 | [-0.014, +0.049] | 12/6 | 1.0000 | nie |
| comp+mix | B_tr_op | -0.009 | [-0.029, +0.013] | 3/10 | 1.0000 | nie |
| comp+mix | B_tr_dd | +0.010 | [-0.043, +0.065] | 9/8 | 1.0000 | nie |
| comp+rel·PoE | A_dół | +0.017 | [-0.010, +0.043] | 11/7 | 1.0000 | nie |
| comp+rel·PoE | B_te_hop | +0.011 | [-0.010, +0.031] | 11/7 | 1.0000 | nie |
| comp+rel·PoE | B_te_dd | -0.023 | [-0.057, +0.010] | 6/10 | 1.0000 | nie |
| comp+rel·PoE | B_tr_op | -0.004 | [-0.033, +0.025] | 8/10 | 1.0000 | nie |
| comp+rel·PoE | B_tr_dd | +0.008 | [-0.037, +0.053] | 10/6 | 1.0000 | nie |

## 6. Czy Mostowi pomaga τ w tokenach: bw·tok+ΔE − bw·ΔE

| wariant | miara | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | +0.076 | [+0.032, +0.119] | 12/3 | 0.0497 | **tak** |
| comp | B_te_hop | +0.069 | [+0.033, +0.103] | 15/4 | 0.0229 | **tak** |
| comp | B_te_dd | +0.033 | [+0.002, +0.065] | 11/4 | 0.4771 | nie |
| comp | B_tr_op | +0.104 | [+0.067, +0.138] | 17/3 | 0.0010 | **tak** |
| comp | B_tr_dd | +0.034 | [-0.021, +0.093] | 10/7 | 1.0000 | nie |
| comp+mix | A_dół | +0.075 | [+0.027, +0.124] | 14/5 | 0.0713 | nie |
| comp+mix | B_te_hop | +0.065 | [+0.027, +0.103] | 14/6 | 0.0501 | nie |
| comp+mix | B_te_dd | +0.043 | [-0.009, +0.089] | 13/4 | 0.7850 | nie |
| comp+mix | B_tr_op | +0.091 | [+0.062, +0.121] | 17/2 | 0.0004 | **tak** |
| comp+mix | B_tr_dd | -0.041 | [-0.092, +0.008] | 4/10 | 0.7850 | nie |
| comp+rel·PoE | A_dół | -0.007 | [-0.040, +0.027] | 10/10 | 1.0000 | nie |
| comp+rel·PoE | B_te_hop | -0.007 | [-0.033, +0.019] | 9/8 | 1.0000 | nie |
| comp+rel·PoE | B_te_dd | -0.038 | [-0.082, +0.006] | 6/13 | 0.7850 | nie |
| comp+rel·PoE | B_tr_op | +0.091 | [+0.061, +0.122] | 17/2 | 0.0005 | **tak** |
| comp+rel·PoE | B_tr_dd | -0.023 | [-0.075, +0.026] | 8/8 | 1.0000 | nie |

## Wnioski

**Ważność porównania:**
- 8 ramion ma identyczne parametry i inicjalizację.
- Dawne ramiona odtwarzają się nową klasą co do cyfry (72/72).
- Kontrasty P1 i P2 zostały zadeklarowane przed uruchomieniem.

1. **Najlepszy sposób podania informacji Softmaxowi to tokeny, nie kara ΔE w uwadze.** `sm·tok − sm·ΔE` na nowych pierwiastkach: +0.06…+0.12 (13–18/20, w 7 z 9 testów p_Holm < 0.02). Poprzednia ablacja (KERNEL) porównywała więc Mosta z **słabo** zasilonym Softmaxem.
2. **P1: Softmax z τ w tokenach jest lepszy od Mosta (bw·ΔE) albo mu równy.**

   | sm·tok vs bw·ΔE (Most) | comp | comp+mix | comp+rel·PoE |
   |---|---:|---:|---:|
   | krok „dół” (nowe) | **0.39 vs 0.31** (Softmax, p_Holm = 0.02) | **0.45 vs 0.38** (Softmax, 0.02) | 0.41 vs 0.39 (nie) |
   | 2 kroki (nowe) | **0.38 vs 0.32** (Softmax, 0.03) | **0.43 vs 0.37** (Softmax, 0.002) | 0.40 vs 0.37 (nie) |
   | dół∘dół (nowe) | 0.23 vs 0.23 | 0.34 vs 0.29 (nie) | 0.33 vs 0.37 (nie) |
   | złożenie, znane | **0.87 vs 0.76** (Softmax, 0.001) | **0.85 vs 0.76** (Softmax, 0.0004) | **0.80 vs 0.74** (Softmax, 0.02) |
   | dół∘dół, znane | 0.89 vs 0.87 (nie) | **0.75 vs 0.84** (Most, 0.03) | **0.77 vs 0.87** (Most, 0.03) |

   Most wygrywa wyłącznie w łańcuchach dół∘dół na znanych pierwiastkach, w wariantach z relacją grupy. W pozostałych 13 z 15 porównań jest gorszy albo równy.
3. **P2: przy informacji w tokenach jądro nie ma znaczenia.** `bw·tok − sm·tok`: wszystkie 15 różnic |Δ| ≤ 0.03, żadna nie przeżywa korekty. Tak samo z ΔE dodatkowo w uwadze (`bw·tok+ΔE − sm·tok+ΔE`: |Δ| ≤ 0.03, 0/15). **Zysk jądra BW z poprzedniej ablacji (+0.07…+0.11 w PoE) znika, gdy Softmax dostaje informację w standardowy sposób.**
4. **τ w tokenach pomaga też Mostowi** (`bw·tok+ΔE − bw·ΔE`: +0.07…+0.10 w comp, złożenie na znanych +0.09…+0.10 we wszystkich wariantach). Kanoniczny Most (tracer tylko w ΔE) nie korzysta więc z informacji w pełni.
5. **Najlepsze konfiguracje serii na nowych pierwiastkach** to `comp+mix` z τ w tokenach, **w dowolnym jądrze**: krok „dół” 0.45–0.46, hop 0.43–0.44. Najlepszy kanoniczny Most (PoE) daje 0.39 / 0.37.

## Werdykt

Czy samo jądro uwagi wnosi korzyść, gdy oba modele dostają te same dane w ten sam, standardowy sposób (w tokenach)? **Nie.** Różnice między jądrem Breita–Wignera a softmaksem są wtedy zerowe we wszystkich 30 porównaniach. Kanoniczny Most (tracer wyłącznie przez ΔE) przegrywa ze zwykłym Softmaxem z tracerem w tokenach w 7 z 15 testów, wygrywa w 2 (dół∘dół na znanych, z relacją grupy) i remisuje w 6.

Wcześniejsze przewagi Mostu w tej serii wynikały z tego, że **tylko on dostawał informację z tracera**, a potem (KERNEL) z tego, że Softmax dostawał ją w niekorzystnej postaci.

