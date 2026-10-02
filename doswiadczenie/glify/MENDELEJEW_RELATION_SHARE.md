# Mendelejew — jaka część poprawy wynika ze struktury relacyjnej

Bez nowego treningu: dane z ablacji jądra (`kernel_ablation_runs.jsonl`, 4 ramiona, ta sama klasa enkodera) i z COMPREL FROZEN (ramię mlp). Specyfikacja SHA-256 `3d0af9a03a27edfb…`, 20 przebiegów, identyczne zapytania. Skrypt `archiwum/skrypty/analyze_mendelejew_relation_share.py`.

Relacja grupy = wariant `comp+rel` (osobne głowice, odpowiedź z operatora), `comp+rel·PoE` (operator + głowica grupy), `comp+mix` (grupa w przestrzeni operatorów). Bez relacji = `comp`.

## 1. Efekt relacji przy stałym jądrze: wariant − comp (36 testów, Holm)

| ramię | wariant | miara | comp | z relacją | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---|---:|---:|---:|---|---:|---:|:-:|
| bw·ΔE | comp+rel | A_dół | 0.309 | 0.342 | +0.032 | [-0.004, +0.073] | 11/6 | 1.0000 | nie |
| bw·ΔE | comp+rel | B_te_hop | 0.315 | 0.332 | +0.017 | [-0.011, +0.047] | 9/10 | 1.0000 | nie |
| bw·ΔE | comp+rel | B_te_dd | 0.227 | 0.296 | +0.069 | [+0.032, +0.107] | 14/4 | 0.0867 | nie |
| bw·ΔE | comp+rel·PoE | A_dół | 0.309 | 0.388 | +0.078 | [+0.038, +0.117] | 16/3 | 0.0563 | nie |
| bw·ΔE | comp+rel·PoE | B_te_hop | 0.315 | 0.373 | +0.057 | [+0.029, +0.087] | 15/4 | 0.0341 | **tak** |
| bw·ΔE | comp+rel·PoE | B_te_dd | 0.227 | 0.370 | +0.143 | [+0.102, +0.186] | 19/1 | 0.0005 | **tak** |
| bw·ΔE | comp+mix | A_dół | 0.309 | 0.379 | +0.070 | [+0.027, +0.113] | 13/4 | 0.1792 | nie |
| bw·ΔE | comp+mix | B_te_hop | 0.315 | 0.373 | +0.058 | [+0.027, +0.090] | 13/6 | 0.0812 | nie |
| bw·ΔE | comp+mix | B_te_dd | 0.227 | 0.290 | +0.063 | [+0.027, +0.101] | 15/3 | 0.1104 | nie |
| bw·0 | comp+rel | A_dół | 0.263 | 0.315 | +0.052 | [+0.007, +0.100] | 15/4 | 0.9825 | nie |
| bw·0 | comp+rel | B_te_hop | 0.257 | 0.286 | +0.029 | [-0.003, +0.063] | 11/8 | 1.0000 | nie |
| bw·0 | comp+rel | B_te_dd | 0.236 | 0.239 | +0.003 | [-0.037, +0.043] | 9/10 | 1.0000 | nie |
| bw·0 | comp+rel·PoE | A_dół | 0.263 | 0.313 | +0.050 | [+0.003, +0.101] | 12/6 | 1.0000 | nie |
| bw·0 | comp+rel·PoE | B_te_hop | 0.257 | 0.284 | +0.028 | [-0.007, +0.065] | 12/8 | 1.0000 | nie |
| bw·0 | comp+rel·PoE | B_te_dd | 0.236 | 0.306 | +0.071 | [+0.025, +0.119] | 13/4 | 0.1978 | nie |
| bw·0 | comp+mix | A_dół | 0.263 | 0.243 | -0.020 | [-0.060, +0.022] | 5/11 | 1.0000 | nie |
| bw·0 | comp+mix | B_te_hop | 0.257 | 0.245 | -0.012 | [-0.038, +0.017] | 7/12 | 1.0000 | nie |
| bw·0 | comp+mix | B_te_dd | 0.236 | 0.222 | -0.014 | [-0.058, +0.025] | 8/8 | 1.0000 | nie |
| sm·ΔE | comp+rel | A_dół | 0.275 | 0.308 | +0.032 | [-0.005, +0.068] | 12/5 | 1.0000 | nie |
| sm·ΔE | comp+rel | B_te_hop | 0.282 | 0.308 | +0.026 | [-0.003, +0.055] | 12/7 | 1.0000 | nie |
| sm·ΔE | comp+rel | B_te_dd | 0.175 | 0.233 | +0.058 | [+0.002, +0.111] | 14/4 | 1.0000 | nie |
| sm·ΔE | comp+rel·PoE | A_dół | 0.275 | 0.299 | +0.024 | [-0.017, +0.065] | 12/7 | 1.0000 | nie |
| sm·ΔE | comp+rel·PoE | B_te_hop | 0.282 | 0.306 | +0.024 | [-0.007, +0.054] | 13/7 | 1.0000 | nie |
| sm·ΔE | comp+rel·PoE | B_te_dd | 0.175 | 0.261 | +0.086 | [+0.032, +0.144] | 15/4 | 0.1845 | nie |
| sm·ΔE | comp+mix | A_dół | 0.275 | 0.346 | +0.071 | [+0.025, +0.114] | 14/3 | 0.1926 | nie |
| sm·ΔE | comp+mix | B_te_hop | 0.282 | 0.332 | +0.049 | [+0.020, +0.077] | 15/5 | 0.1369 | nie |
| sm·ΔE | comp+mix | B_te_dd | 0.175 | 0.239 | +0.064 | [+0.013, +0.115] | 16/4 | 0.5841 | nie |
| sm·0 | comp+rel | A_dół | 0.253 | 0.278 | +0.025 | [-0.015, +0.064] | 10/3 | 1.0000 | nie |
| sm·0 | comp+rel | B_te_hop | 0.243 | 0.263 | +0.021 | [-0.006, +0.049] | 14/6 | 1.0000 | nie |
| sm·0 | comp+rel | B_te_dd | 0.194 | 0.280 | +0.086 | [+0.045, +0.134] | 16/2 | 0.0034 | **tak** |
| sm·0 | comp+rel·PoE | A_dół | 0.253 | 0.303 | +0.049 | [-0.001, +0.099] | 12/4 | 1.0000 | nie |
| sm·0 | comp+rel·PoE | B_te_hop | 0.243 | 0.283 | +0.041 | [+0.004, +0.077] | 14/5 | 0.9825 | nie |
| sm·0 | comp+rel·PoE | B_te_dd | 0.194 | 0.331 | +0.137 | [+0.089, +0.191] | 18/2 | 0.0004 | **tak** |
| sm·0 | comp+mix | A_dół | 0.253 | 0.266 | +0.013 | [-0.033, +0.059] | 10/8 | 1.0000 | nie |
| sm·0 | comp+mix | B_te_hop | 0.243 | 0.257 | +0.015 | [-0.020, +0.049] | 11/8 | 1.0000 | nie |
| sm·0 | comp+mix | B_te_dd | 0.194 | 0.212 | +0.018 | [-0.022, +0.059] | 10/9 | 1.0000 | nie |

## 1b. Dodatkowo (bez korekty): ramię mlp (bez uwagi) i miary na znanych pierwiastkach

| ramię | wariant | miara | comp | z relacją | Δ | 95% CI | p |
|---|---|---|---:|---:|---:|---|---:|
| mlp | comp+rel | A_dół | 0.381 | 0.410 | +0.029 | [+0.003, +0.054] | 0.0437 |
| mlp | comp+rel | B_te_hop | 0.302 | 0.316 | +0.014 | [-0.012, +0.041] | 0.3421 |
| mlp | comp+rel | B_te_dd | 0.288 | 0.293 | +0.005 | [-0.027, +0.040] | 0.7820 |
| mlp | comp+rel | B_tr_op | 0.591 | 0.602 | +0.010 | [-0.011, +0.031] | 0.3748 |
| mlp | comp+rel | B_tr_dd | 0.645 | 0.674 | +0.029 | [-0.016, +0.076] | 0.2527 |
| mlp | comp+rel·PoE | A_dół | 0.381 | 0.484 | +0.103 | [+0.060, +0.144] | 0.0003 |
| mlp | comp+rel·PoE | B_te_hop | 0.302 | 0.362 | +0.059 | [+0.030, +0.090] | 0.0012 |
| mlp | comp+rel·PoE | B_te_dd | 0.288 | 0.414 | +0.126 | [+0.081, +0.167] | 0.0001 |
| mlp | comp+rel·PoE | B_tr_op | 0.591 | 0.604 | +0.012 | [-0.009, +0.034] | 0.2823 |
| mlp | comp+rel·PoE | B_tr_dd | 0.645 | 0.683 | +0.038 | [-0.012, +0.090] | 0.1725 |
| mlp | comp+mix | A_dół | 0.381 | 0.384 | +0.002 | [-0.032, +0.035] | 0.8948 |
| mlp | comp+mix | B_te_hop | 0.302 | 0.299 | -0.003 | [-0.035, +0.028] | 0.8582 |
| mlp | comp+mix | B_te_dd | 0.288 | 0.183 | -0.105 | [-0.148, -0.058] | 0.0004 |
| mlp | comp+mix | B_tr_op | 0.591 | 0.485 | -0.107 | [-0.141, -0.070] | 0.0001 |
| mlp | comp+mix | B_tr_dd | 0.645 | 0.554 | -0.091 | [-0.155, -0.021] | 0.0189 |
| bw·ΔE | comp+rel | B_tr_op | 0.764 | 0.752 | -0.012 | [-0.043, +0.019] | 0.4718 |
| bw·ΔE | comp+rel | B_tr_dd | 0.870 | 0.910 | +0.040 | [-0.018, +0.094] | 0.1965 |
| bw·ΔE | comp+rel·PoE | B_tr_op | 0.764 | 0.742 | -0.022 | [-0.054, +0.010] | 0.2020 |
| bw·ΔE | comp+rel·PoE | B_tr_dd | 0.870 | 0.871 | +0.000 | [-0.058, +0.057] | 0.9866 |
| bw·ΔE | comp+mix | B_tr_op | 0.764 | 0.759 | -0.005 | [-0.042, +0.028] | 0.8068 |
| bw·ΔE | comp+mix | B_tr_dd | 0.870 | 0.838 | -0.032 | [-0.085, +0.023] | 0.2773 |
| bw·0 | comp+rel | B_tr_op | 0.559 | 0.548 | -0.011 | [-0.065, +0.046] | 0.7042 |
| bw·0 | comp+rel | B_tr_dd | 0.585 | 0.646 | +0.061 | [-0.007, +0.130] | 0.1044 |
| bw·0 | comp+rel·PoE | B_tr_op | 0.559 | 0.544 | -0.015 | [-0.072, +0.042] | 0.6027 |
| bw·0 | comp+rel·PoE | B_tr_dd | 0.585 | 0.628 | +0.043 | [-0.033, +0.114] | 0.2745 |
| bw·0 | comp+mix | B_tr_op | 0.559 | 0.470 | -0.089 | [-0.144, -0.041] | 0.0020 |
| bw·0 | comp+mix | B_tr_dd | 0.585 | 0.524 | -0.061 | [-0.166, +0.035] | 0.2771 |
| sm·ΔE | comp+rel | B_tr_op | 0.785 | 0.758 | -0.028 | [-0.072, +0.017] | 0.2487 |
| sm·ΔE | comp+rel | B_tr_dd | 0.930 | 0.930 | +0.000 | [-0.037, +0.039] | 0.9823 |
| sm·ΔE | comp+rel·PoE | B_tr_op | 0.785 | 0.749 | -0.037 | [-0.081, +0.007] | 0.1278 |
| sm·ΔE | comp+rel·PoE | B_tr_dd | 0.930 | 0.895 | -0.035 | [-0.085, +0.016] | 0.1971 |
| sm·ΔE | comp+mix | B_tr_op | 0.785 | 0.810 | +0.024 | [-0.007, +0.054] | 0.1472 |
| sm·ΔE | comp+mix | B_tr_dd | 0.930 | 0.852 | -0.078 | [-0.134, -0.024] | 0.0129 |
| sm·0 | comp+rel | B_tr_op | 0.575 | 0.531 | -0.044 | [-0.097, +0.009] | 0.1292 |
| sm·0 | comp+rel | B_tr_dd | 0.601 | 0.622 | +0.021 | [-0.051, +0.090] | 0.5688 |
| sm·0 | comp+rel·PoE | B_tr_op | 0.575 | 0.543 | -0.032 | [-0.083, +0.018] | 0.2479 |
| sm·0 | comp+rel·PoE | B_tr_dd | 0.601 | 0.671 | +0.071 | [+0.001, +0.135] | 0.0594 |
| sm·0 | comp+mix | B_tr_op | 0.575 | 0.496 | -0.079 | [-0.127, -0.029] | 0.0076 |
| sm·0 | comp+mix | B_tr_dd | 0.601 | 0.524 | -0.076 | [-0.169, +0.009] | 0.1199 |

## 2. Rozkład poprawy (sm·0, comp) → (bw·ΔE, comp+rel·PoE) — wartości Shapleya

Udział = φ / całkowita poprawa (średnie po przebiegach). φ: 95% CI i p po przebiegach.

| miara | start | koniec | całość | czynnik | φ | 95% CI | p | udział |
|---|---:|---:|---:|---|---:|---|---:|---:|
| A_dół | 0.253 | 0.388 | +0.134 | informacja ΔE | +0.039 | [+0.013, +0.066] | 0.0113 | 29% |
| | | | | jądro BW | +0.040 | [+0.026, +0.054] | 0.0001 | 30% |
| | | | | relacja grupy | +0.055 | [+0.025, +0.086] | 0.0016 | 41% |
| B_te_hop | 0.243 | 0.373 | +0.130 | informacja ΔE | +0.056 | [+0.038, +0.075] | 0.0000 | 43% |
| | | | | jądro BW | +0.033 | [+0.021, +0.044] | 0.0001 | 25% |
| | | | | relacja grupy | +0.041 | [+0.020, +0.064] | 0.0013 | 32% |
| B_te_dd | 0.194 | 0.370 | +0.176 | informacja ΔE | +0.002 | [-0.030, +0.033] | 0.9216 | 1% |
| | | | | jądro BW | +0.055 | [+0.038, +0.071] | 0.0000 | 31% |
| | | | | relacja grupy | +0.119 | [+0.086, +0.154] | 0.0000 | 68% |
| B_tr_op | 0.575 | 0.742 | +0.167 | informacja ΔE | +0.204 | [+0.177, +0.231] | 0.0000 | 123% |
| | | | | jądro BW | -0.011 | [-0.031, +0.007] | 0.3029 | -7% |
| | | | | relacja grupy | -0.027 | [-0.052, -0.001] | 0.0596 | -16% |
| B_tr_dd | 0.601 | 0.871 | +0.270 | informacja ΔE | +0.275 | [+0.220, +0.325] | 0.0000 | 102% |
| | | | | jądro BW | -0.031 | [-0.051, -0.010] | 0.0119 | -11% |
| | | | | relacja grupy | +0.025 | [-0.018, +0.063] | 0.2518 | 9% |

## 2. Rozkład poprawy (sm·0, comp) → (bw·ΔE, comp+mix) — wartości Shapleya

Udział = φ / całkowita poprawa (średnie po przebiegach). φ: 95% CI i p po przebiegach.

| miara | start | koniec | całość | czynnik | φ | 95% CI | p | udział |
|---|---:|---:|---:|---|---:|---|---:|---:|
| A_dół | 0.253 | 0.379 | +0.126 | informacja ΔE | +0.074 | [+0.045, +0.107] | 0.0000 | 59% |
| | | | | jądro BW | +0.016 | [-0.000, +0.033] | 0.0820 | 13% |
| | | | | relacja grupy | +0.036 | [+0.011, +0.064] | 0.0128 | 29% |
| B_te_hop | 0.243 | 0.373 | +0.130 | informacja ΔE | +0.078 | [+0.059, +0.100] | 0.0000 | 60% |
| | | | | jądro BW | +0.022 | [+0.011, +0.033] | 0.0014 | 17% |
| | | | | relacja grupy | +0.030 | [+0.014, +0.047] | 0.0017 | 23% |
| B_te_dd | 0.194 | 0.290 | +0.096 | informacja ΔE | +0.020 | [-0.009, +0.049] | 0.2208 | 20% |
| | | | | jądro BW | +0.041 | [+0.026, +0.055] | 0.0002 | 43% |
| | | | | relacja grupy | +0.035 | [+0.007, +0.063] | 0.0287 | 37% |
| B_tr_op | 0.575 | 0.759 | +0.184 | informacja ΔE | +0.253 | [+0.222, +0.281] | 0.0000 | 138% |
| | | | | jądro BW | -0.030 | [-0.049, -0.009] | 0.0120 | -16% |
| | | | | relacja grupy | -0.039 | [-0.067, -0.011] | 0.0159 | -21% |
| B_tr_dd | 0.601 | 0.838 | +0.237 | informacja ΔE | +0.316 | [+0.266, +0.365] | 0.0000 | 133% |
| | | | | jądro BW | -0.020 | [-0.046, +0.005] | 0.1580 | -8% |
| | | | | relacja grupy | -0.059 | [-0.117, -0.009] | 0.0389 | -25% |

## Wnioski

**Zastrzeżenia:**
- Rodzina 36 testów z korektą Holma jest surowa. Wiele efektów ma 13–19 wygranych z 20 i przedział ufności nad zerem, a mimo to nie przeżywa korekty. Opisuję je jako „spójny znak, niepotwierdzony”.
- Rozkład Shapleya dzieli interakcje czynników po równo i zależy od wybranych punktów startu i końca.

1. **Przy stałym jądrze relacja grupy pomaga na nowych pierwiastkach, ale silnie tylko w łańcuchach „dół”.**
   - Najmocniejszy i najpewniejszy efekt: `comp+rel·PoE` w dół∘dół. +0.14 w bw·ΔE (19/1, p_Holm = 0.0005), +0.14 w sm·0 (18/2, p_Holm = 0.0004), +0.07 do +0.09 w pozostałych ramionach, +0.13 w MLP. Działa **w każdym jądrze** i bez uwagi, więc jest to wkład samej struktury relacyjnej.
   - W pojedynczym kroku i w hop relacja daje +0.02 do +0.08, ze spójnym znakiem. Po korekcie przeżywa tylko bw·ΔE PoE hop (+0.06, p_Holm = 0.034).
2. **Wariant `mix` (grupa we wspólnej przestrzeni z operatorami) pomaga tylko wtedy, gdy jest ΔE.** Z ΔE daje +0.05 do +0.07 (bw·ΔE, sm·ΔE). Bez ΔE wychodzi 0 albo minus (bw·0: −0.01 do −0.02). W MLP **szkodzi**: dół∘dół −0.11, kompozycja na znanych −0.11.
3. **Na znanych pierwiastkach relacja grupy nie pomaga, a `mix` szkodzi** (B_tr_op od −0.08 do −0.09 w ramionach bez ΔE i w MLP). Kompozycja na znanych to w całości efekt informacji ΔE (sekcja 2: ΔE wyjaśnia ponad 100% poprawy, jądro i relacja wychodzą lekko na minus).
4. **Jaka część poprawy pochodzi ze struktury relacyjnej.** Droga od `sm·0 comp` do pełnego Mosta z PoE (`bw·ΔE comp+rel·PoE`), nowe pierwiastki:

   | miara | całość | ΔE | jądro BW | **relacja grupy** |
   |---|---:|---:|---:|---:|
   | krok „dół” | +0.13 | 29% | 30% | **41%** |
   | 2 kroki | +0.13 | 43% | 25% | **32%** |
   | dół∘dół | +0.18 | 1% | 31% | **68%** |

   Każdy z trzech czynników ma przedział ufności powyżej zera (wyjątek: ΔE w dół∘dół). Na drodze do `mix` relacja daje 23–37%, a dominuje informacja ΔE (59–60% w pojedynczym kroku i hop).

## Werdykt

Struktura relacyjna (relacja grupy z PoE) odpowiada za **około jedną trzecią do dwóch trzecich** poprawy na nowych pierwiastkach: 32–41% w pojedynczym kroku i dwóch krokach, 68% w łańcuchach dół∘dół. Działa niezależnie od jądra uwagi, także w MLP, więc jest to wkład samej struktury, a nie zasługa Mostu. Na znanych pierwiastkach relacja niczego nie dodaje, a cała poprawa to informacja ΔE. Wariant `mix` jest wrażliwy: pomaga tylko przy ΔE, a w MLP szkodzi.

