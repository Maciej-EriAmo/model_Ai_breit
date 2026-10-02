# Mendelejew COMPREL FROZEN — zamrożony protokół, 20 przebiegów, statystyka parami

Specyfikacja: `comprel_frozen_spec.json` · SHA-256 `3d0af9a03a27edfb…` · przebiegi=20 (podział + ziarno zamrożone per przebieg; 0–4 = dotychczasowe) · kroki=800 · każde uruchomienie w `comprel_frozen_runs.jsonl` (pozycja każdego zapytania) · skrypt `archiwum/skrypty/run_mendelejew_comprel_frozen.py`

Wszystkie warianty i armie oceniane na **identycznych zapytaniach** w danym przebiegu; porównania parami po przebiegach. Miary główne (zadeklarowane z góry): **A_dół**, **B_te_hop**, **B_te_dd** (hit@1). CI = 95% bootstrap średniej różnicy; p = dokładny test permutacyjny znaków; p_Holm = korekta w obrębie rodziny.

Liczba zapytań (średnio na przebieg): A_dół=36.1, B_te=126.0, B_te_dd=35.0.

## 1. Średnie hit@1 (± odch. std. po przebiegach)

| miara | arm | comp | comp+rel | comp+rel·PoE | comp+mix |
|---|---|---:|---:|---:|---:|
| A_dół | most | 0.309 ± 0.070 | 0.342 ± 0.089 | 0.388 ± 0.092 | 0.379 ± 0.105 |
| A_dół | softmax | 0.276 ± 0.084 | 0.275 ± 0.114 | 0.305 ± 0.122 | 0.262 ± 0.097 |
| A_dół | zero | 0.263 ± 0.096 | 0.315 ± 0.130 | 0.313 ± 0.115 | 0.243 ± 0.084 |
| A_dół | mlp | 0.381 ± 0.102 | 0.410 ± 0.093 | 0.484 ± 0.093 | 0.384 ± 0.104 |
| B_te_hop | most | 0.315 ± 0.057 | 0.332 ± 0.075 | 0.373 ± 0.080 | 0.373 ± 0.089 |
| B_te_hop | softmax | 0.254 ± 0.071 | 0.256 ± 0.083 | 0.275 ± 0.084 | 0.257 ± 0.072 |
| B_te_hop | zero | 0.257 ± 0.063 | 0.286 ± 0.094 | 0.284 ± 0.091 | 0.245 ± 0.069 |
| B_te_hop | mlp | 0.302 ± 0.090 | 0.316 ± 0.095 | 0.362 ± 0.092 | 0.299 ± 0.072 |
| B_te_dd | most | 0.227 ± 0.092 | 0.296 ± 0.091 | 0.370 ± 0.106 | 0.290 ± 0.116 |
| B_te_dd | softmax | 0.209 ± 0.087 | 0.259 ± 0.088 | 0.294 ± 0.106 | 0.235 ± 0.100 |
| B_te_dd | zero | 0.236 ± 0.074 | 0.239 ± 0.095 | 0.306 ± 0.112 | 0.222 ± 0.062 |
| B_te_dd | mlp | 0.288 ± 0.076 | 0.293 ± 0.106 | 0.414 ± 0.111 | 0.183 ± 0.078 |
| A_prawo | most | 0.546 ± 0.062 | 0.573 ± 0.065 | 0.573 ± 0.065 | 0.525 ± 0.069 |
| A_prawo | softmax | 0.558 ± 0.068 | 0.564 ± 0.077 | 0.564 ± 0.077 | 0.499 ± 0.076 |
| A_prawo | zero | 0.558 ± 0.080 | 0.578 ± 0.088 | 0.578 ± 0.088 | 0.506 ± 0.077 |
| A_prawo | mlp | 0.384 ± 0.086 | 0.397 ± 0.083 | 0.397 ± 0.083 | 0.348 ± 0.047 |
| B_tr_op | most | 0.764 ± 0.065 | 0.752 ± 0.050 | 0.742 ± 0.051 | 0.759 ± 0.075 |
| B_tr_op | softmax | 0.573 ± 0.069 | 0.565 ± 0.107 | 0.576 ± 0.099 | 0.491 ± 0.086 |
| B_tr_op | zero | 0.559 ± 0.078 | 0.548 ± 0.089 | 0.544 ± 0.091 | 0.470 ± 0.093 |
| B_tr_op | mlp | 0.591 ± 0.086 | 0.602 ± 0.081 | 0.604 ± 0.082 | 0.485 ± 0.054 |
| B_te_op | most | 0.270 ± 0.061 | 0.284 ± 0.065 | 0.304 ± 0.074 | 0.293 ± 0.075 |
| B_te_op | softmax | 0.250 ± 0.060 | 0.259 ± 0.069 | 0.269 ± 0.069 | 0.230 ± 0.056 |
| B_te_op | zero | 0.257 ± 0.044 | 0.271 ± 0.066 | 0.290 ± 0.074 | 0.229 ± 0.052 |
| B_te_op | mlp | 0.244 ± 0.061 | 0.249 ± 0.063 | 0.283 ± 0.057 | 0.193 ± 0.041 |

## 2. Rodzina A — warianty dla Mostu (miary główne, 9 testów, Holm)

| miara | porównanie | Δ średnia | 95% CI | wygr./przegr. | p | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|---:|:-:|
| A_dół | comp+mix − comp | +0.070 | [+0.027, +0.113] | 13/4 | 0.0066 | 0.0199 | **tak** |
| A_dół | comp+rel·PoE − comp | +0.078 | [+0.038, +0.117] | 16/3 | 0.0018 | 0.0106 | **tak** |
| A_dół | comp+mix − comp+rel·PoE | -0.008 | [-0.052, +0.035] | 10/8 | 0.7172 | 1.0000 | nie |
| B_te_hop | comp+mix − comp | +0.058 | [+0.027, +0.090] | 13/6 | 0.0026 | 0.0131 | **tak** |
| B_te_hop | comp+rel·PoE − comp | +0.057 | [+0.029, +0.087] | 15/4 | 0.0010 | 0.0083 | **tak** |
| B_te_hop | comp+mix − comp+rel·PoE | +0.001 | [-0.028, +0.029] | 11/8 | 0.9671 | 1.0000 | nie |
| B_te_dd | comp+mix − comp | +0.063 | [+0.027, +0.101] | 15/3 | 0.0038 | 0.0152 | **tak** |
| B_te_dd | comp+rel·PoE − comp | +0.143 | [+0.102, +0.186] | 19/1 | 0.0000 | 0.0001 | **tak** |
| B_te_dd | comp+mix − comp+rel·PoE | -0.080 | [-0.122, -0.039] | 3/16 | 0.0011 | 0.0083 | **tak** |

## 3. Rodzina B — Most vs reszta (27 testów, Holm)

| wariant | miara | vs | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---|---:|---|---:|---:|:-:|
| comp | A_dół | softmax | +0.033 | [-0.009, +0.076] | 11/7 | 0.9572 | nie |
| comp | A_dół | zero | +0.046 | [+0.003, +0.091] | 11/6 | 0.4845 | nie |
| comp | A_dół | mlp | -0.072 | [-0.118, -0.026] | 4/14 | 0.1030 | nie |
| comp | B_te_hop | softmax | +0.061 | [+0.028, +0.095] | 14/6 | 0.0457 | **tak** |
| comp | B_te_hop | zero | +0.059 | [+0.029, +0.089] | 16/4 | 0.0271 | **tak** |
| comp | B_te_hop | mlp | +0.013 | [-0.021, +0.042] | 14/5 | 1.0000 | nie |
| comp | B_te_dd | softmax | +0.018 | [-0.019, +0.053] | 12/7 | 1.0000 | nie |
| comp | B_te_dd | zero | -0.009 | [-0.050, +0.029] | 10/9 | 1.0000 | nie |
| comp | B_te_dd | mlp | -0.061 | [-0.102, -0.019] | 6/14 | 0.1530 | nie |
| comp+mix | A_dół | softmax | +0.117 | [+0.076, +0.161] | 17/1 | 0.0009 | **tak** |
| comp+mix | A_dół | zero | +0.136 | [+0.093, +0.180] | 17/2 | 0.0008 | **tak** |
| comp+mix | A_dół | mlp | -0.005 | [-0.059, +0.048] | 9/8 | 1.0000 | nie |
| comp+mix | B_te_hop | softmax | +0.116 | [+0.084, +0.149] | 17/2 | 0.0004 | **tak** |
| comp+mix | B_te_hop | zero | +0.128 | [+0.096, +0.161] | 18/0 | 0.0002 | **tak** |
| comp+mix | B_te_hop | mlp | +0.074 | [+0.030, +0.117] | 16/4 | 0.0820 | nie |
| comp+mix | B_te_dd | softmax | +0.055 | [+0.004, +0.109] | 13/7 | 0.4845 | nie |
| comp+mix | B_te_dd | zero | +0.069 | [+0.024, +0.117] | 13/4 | 0.1340 | nie |
| comp+mix | B_te_dd | mlp | +0.107 | [+0.053, +0.157] | 15/2 | 0.0244 | **tak** |
| comp+rel·PoE | A_dół | softmax | +0.082 | [+0.033, +0.133] | 12/5 | 0.0820 | nie |
| comp+rel·PoE | A_dół | zero | +0.075 | [+0.032, +0.117] | 15/4 | 0.0578 | nie |
| comp+rel·PoE | A_dół | mlp | -0.097 | [-0.141, -0.052] | 2/16 | 0.0141 | **tak** |
| comp+rel·PoE | B_te_hop | softmax | +0.097 | [+0.063, +0.134] | 18/1 | 0.0006 | **tak** |
| comp+rel·PoE | B_te_hop | zero | +0.088 | [+0.056, +0.121] | 18/2 | 0.0016 | **tak** |
| comp+rel·PoE | B_te_hop | mlp | +0.011 | [-0.029, +0.050] | 12/8 | 1.0000 | nie |
| comp+rel·PoE | B_te_dd | softmax | +0.076 | [+0.031, +0.124] | 14/4 | 0.0820 | nie |
| comp+rel·PoE | B_te_dd | zero | +0.063 | [+0.015, +0.110] | 14/5 | 0.2132 | nie |
| comp+rel·PoE | B_te_dd | mlp | -0.044 | [-0.088, +0.001] | 4/13 | 0.5325 | nie |

## 4. Odporność — te same porównania na MRR (bez korekty)

| miara | porównanie | Δ MRR | 95% CI | p |
|---|---|---:|---|---:|
| A_dół | comp+mix − comp | +0.039 | [+0.008, +0.070] | 0.0290 |
| A_dół | comp+rel·PoE − comp | +0.038 | [+0.007, +0.065] | 0.0257 |
| A_dół | comp+mix − comp+rel·PoE | +0.001 | [-0.029, +0.031] | 0.9390 |
| B_te_hop | comp+mix − comp | +0.033 | [+0.008, +0.060] | 0.0241 |
| B_te_hop | comp+rel·PoE − comp | +0.032 | [+0.008, +0.056] | 0.0222 |
| B_te_hop | comp+mix − comp+rel·PoE | +0.001 | [-0.023, +0.026] | 0.9165 |
| B_te_dd | comp+mix − comp | +0.067 | [+0.035, +0.100] | 0.0009 |
| B_te_dd | comp+rel·PoE − comp | +0.115 | [+0.080, +0.149] | 0.0000 |
| B_te_dd | comp+mix − comp+rel·PoE | -0.048 | [-0.083, -0.015] | 0.0139 |

## 5. Replikacja: przebiegi 0–4 (dotychczasowe) vs 5–19 (nowe), Most, Δ względem comp

| miara | wariant | Δ 0–4 | Δ 5–19 |
|---|---|---:|---:|
| A_dół | comp+mix | +0.154 | +0.042 |
| A_dół | comp+rel·PoE | +0.091 | +0.074 |
| B_te_hop | comp+mix | +0.099 | +0.044 |
| B_te_hop | comp+rel·PoE | +0.048 | +0.060 |
| B_te_dd | comp+mix | +0.074 | +0.060 |
| B_te_dd | comp+rel·PoE | +0.156 | +0.139 |

## 5b. Kompozycja na znanych pierwiastkach — Most vs reszta (`comp`, 9 testów, Holm)

| miara | Most | vs | Δ średnia | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---:|---|---:|---|---:|---:|:-:|
| B_tr_op | 0.764 | softmax | +0.191 | [+0.155, +0.227] | 20/0 | 0.0000 | **tak** |
| B_tr_op | 0.764 | zero | +0.205 | [+0.157, +0.248] | 19/1 | 0.0000 | **tak** |
| B_tr_op | 0.764 | mlp | +0.172 | [+0.131, +0.212] | 19/1 | 0.0000 | **tak** |
| B_tr_hop | 0.956 | softmax | +0.313 | [+0.265, +0.356] | 19/0 | 0.0000 | **tak** |
| B_tr_hop | 0.956 | zero | +0.304 | [+0.258, +0.352] | 20/0 | 0.0000 | **tak** |
| B_tr_hop | 0.956 | mlp | +0.194 | [+0.162, +0.225] | 20/0 | 0.0000 | **tak** |
| B_tr_dd | 0.870 | softmax | +0.273 | [+0.196, +0.342] | 18/2 | 0.0000 | **tak** |
| B_tr_dd | 0.870 | zero | +0.285 | [+0.210, +0.360] | 17/0 | 0.0000 | **tak** |
| B_tr_dd | 0.870 | mlp | +0.225 | [+0.164, +0.283] | 17/1 | 0.0000 | **tak** |

## 6. Pytania Mendelejewa (Most) — mediana pozycji [IQR], trafień na 1. miejscu

| pytanie | comp | comp+rel | comp+rel·PoE | comp+mix |
|---|---|---|---|---|
| dół(Al)=Ga | 25 [15–43] · 1/20 | 16 [7–41] · 1/20 | 16 [9–25] · 1/20 | 16 [9–34] · 1/20 |
| dół(Si)=Ge | 5 [2–17] · 2/20 | 3 [2–40] · 4/20 | 5 [2–21] · 3/20 | 5 [3–21] · 1/20 |
| dół(Mn)=Tc | 5 [4–8] · 0/20 | 5 [4–10] · 0/20 | 10 [3–19] · 0/20 | 5 [3–7] · 1/20 |
| dół∘dół(B)=Ga | 28 [19–47] · 0/20 | 16 [9–36] · 0/20 | 15 [12–28] · 0/20 | 26 [13–34] · 0/20 |
| dół∘dół(C)=Ge | 9 [3–27] · 2/20 | 7 [4–39] · 0/20 | 10 [5–22] · 0/20 | 8 [4–15] · 1/20 |
| prawo(Zn)=Ga | 81 [41–91] · 0/20 | 55 [29–93] · 0/20 | 55 [29–93] · 0/20 | 23 [12–36] · 0/20 |
| prawo(Ga)=Ge | 1 [1–1] · 19/20 | 1 [1–1] · 19/20 | 1 [1–1] · 19/20 | 1 [1–1] · 18/20 |
| prawo(Ca)=Sc | 1 [1–1] · 16/20 | 1 [1–1] · 19/20 | 1 [1–1] · 19/20 | 1 [1–2] · 15/20 |
| prawo∘dół(Mg)=Sc | 3 [2–14] · 0/20 | 3 [2–17] · 0/20 | 3 [2–17] · 0/20 | 3 [2–6] · 5/20 |
| dół∘prawo(Al)=Ge | 7 [3–31] · 2/20 | 4 [2–58] · 2/20 | 4 [2–58] · 2/20 | 11 [3–38] · 4/20 |

## Wnioski

**Protokół:**
- Podziały, fakty, pary i wszystkie zapytania zostały zamrożone **przed** treningiem (SHA-256 w każdym rekordzie). 20 przebiegów, 240 uruchomień, każde z pozycją każdego zapytania w JSONL.
- Przebieg 0 odtwarza COMPREL seed 0 co do cyfry, więc zamrożenie nie zmieniło protokołu.
- Miary główne i rodziny testów zadeklarowano z góry. Sekcję 5b (kompozycja na znanych) dodano do raportu **po** przebiegach, jako sprawdzenie twierdzenia z COMP, a nie nową hipotezę.

1. **Efekt połączenia z relacją grupy utrzymuje się po uwzględnieniu zmienności.** Oba warianty są lepsze od samego `comp` we wszystkich trzech miarach głównych (p_Holm ≤ 0.02, 95% CI powyżej zera). Ten sam znak daje MRR.

   | Most, hit@1 | comp | comp+mix | comp+rel·PoE |
   |---|---:|---:|---:|
   | dół z pierwiastkiem testowym | 0.31 | 0.38 (+0.07) | 0.39 (+0.08) |
   | dwa kroki przez test | 0.32 | 0.37 (+0.06) | 0.37 (+0.06) |
   | dół∘dół przez test | 0.23 | 0.29 (+0.06) | **0.37 (+0.14, 19/20)** |

2. **`comp+mix` i `PoE` nie różnią się, z wyjątkiem łańcuchów „dół”.** W pojedynczym kroku i w hop różnica wynosi ±0.01 (p = 0.72 i 0.97). W dół∘dół PoE wygrywa: +0.08, 16/3, p_Holm = 0.008. Wcześniejsza sugestia, że `mix` jest najlepszy w pojedynczym kroku, **nie utrzymała się**.
3. **Pierwsze 5 seedów przeszacowało efekt.** `comp+mix` w kroku „dół”: przebiegi 0–4 dają +0.15, nowe 5–19 tylko +0.04. Wynik COMPREL „0.32 → 0.48” to na 20 przebiegach **0.31 → 0.38**. PoE replikuje się stabilnie (+0.09 i +0.07).
4. **Przewaga Mostu nad innymi modelami na nowych pierwiastkach jest węższa, niż twierdził COMPREL.**
   - Nad Softmax i ZERO **utrzymuje się** w `comp+mix` (krok „dół” +0.12 / +0.14, hop +0.12 / +0.13, wszystkie p_Holm < 0.001) oraz w PoE w hop (+0.10 / +0.09).
   - Nad **MLP nie**: w `comp+mix` krok „dół” −0.005 (9/8), hop +0.07 (p_Holm = 0.08). W PoE MLP jest **lepszy** od Mostu w kroku „dół” (−0.10, 2/16, p_Holm = 0.014).
   - Twierdzenie COMPREL „Most 5/5 wobec wszystkich modeli” było artefaktem 5 seedów.
5. **Kompozycja na znanych pierwiastkach to najmocniejszy i najpewniejszy wynik Mostu.** Most osiąga 0.76 (operator), 0.96 (dwa kroki) i 0.87 (dół∘dół), wobec 0.56–0.76 u pozostałych modeli. To 17–20 wygranych z 20, CI daleko od zera, p_Holm < 0.0001 wobec Softmax, ZERO **i** MLP. Twierdzenie z COMP („88%”) potwierdza się na 20 przebiegach: 87%.
6. **Pytania Mendelejewa (mediana z 20 przebiegów):** Tc na 5. miejscu, Sc (prawo∘dół z Mg) na 3. Ge pod Si: mediana 3–5, ale w szerokim zakresie (IQR do 40). Ga pod Al: 16. Na 1. miejscu luki trafiają rzadko: Sc 5/20 w `mix`, Ge 4/20 w `rel`. prawo(Zn) → Ga pozostaje najtrudniejsze (mediana 23–81).

## Werdykt

Po zamrożeniu protokołu i 20 przebiegach:
- **Połączenie operatorów z relacją grupy realnie pomaga** na nowych pierwiastkach (+0.06 do +0.14), ale mniej, niż sugerowały pierwsze wyniki.
- **PoE jest najlepsze dla łańcuchów „dół”.**
- **Wyjątkowość Mostu** dotyczy przede wszystkim **składania relacji na znanych pierwiastkach**: wygrywa z każdym modelem, także z MLP. Na nowych pierwiastkach wygrywa z Softmax i ZERO, ale MLP mu dorównuje.

Korekty wcześniejszych twierdzeń są w `MENDELEJEW_COMPREL.md` i w artykule popularnonaukowym.

