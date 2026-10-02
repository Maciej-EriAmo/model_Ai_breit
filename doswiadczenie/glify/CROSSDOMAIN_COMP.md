# Zadanie kompozycyjne na innych zbiorach relacji — układ okresowy vs kana vs muzyka

Te same relacje („dół” = ta sama kolumna, następny wiersz; „prawo” = następny element; „grupa” = ta sama kolumna), te same ramiona (jedna klasa enkodera z ablacji jądra + MLP), warianty i 800 kroków; 20 przebiegów na dziedzinę; specyfikacje zamrożone (`crossdomain_spec_<dziedzina>.json`). **Ścisłe rozdzielenie:** w treningu enkoder przetwarza tylko elementy treningowe — cechy testowych pojawiają się dopiero przy ocenie. Skrypt `archiwum/skrypty/run_crossdomain_comp.py`; każde uruchomienie w `crossdomain_runs.jsonl`.

| dziedzina | elementy | kolumny | cechy | podobieństwo w surowych cechach |
|---|---:|---:|---|---|
| periodic | 118 | 18 | konfiguracja elektronowa (24) + τ6 | 1-NN: ta sama kolumna 24%, ten sam wiersz 100% |
| kana | 45 | 5 | bitmapy glifów 2×12×12 + τ6 | 1-NN: ta sama kolumna 9%, ten sam wiersz 7% |
| music | 84 | 12 | widmo 8 harmonicznych (120 półtonów) + τ6 | 1-NN: ta sama kolumna 100%, ten sam wiersz 0% |

## periodic

Zapytań na przebieg (średnio): A_dół=36.1, B_te=126.0, B_te_dd=35.0, B_tr=64.0, B_tr_dd=16.9.

Średnie hit@1:

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp |
|---|---|---:|---:|---:|---:|---:|
| A_dół | comp | 0.316 | 0.309 | 0.269 | 0.247 | 0.393 |
| A_dół | comp+rel·PoE | 0.356 | 0.353 | 0.322 | 0.315 | 0.511 |
| A_dół | comp+mix | 0.375 | 0.368 | 0.293 | 0.281 | 0.394 |
| B_te_hop | comp | 0.297 | 0.287 | 0.253 | 0.241 | 0.330 |
| B_te_hop | comp+rel·PoE | 0.322 | 0.330 | 0.290 | 0.291 | 0.415 |
| B_te_hop | comp+mix | 0.340 | 0.326 | 0.275 | 0.266 | 0.304 |
| B_te_dd | comp | 0.180 | 0.179 | 0.235 | 0.246 | 0.303 |
| B_te_dd | comp+rel·PoE | 0.258 | 0.252 | 0.311 | 0.322 | 0.437 |
| B_te_dd | comp+mix | 0.238 | 0.225 | 0.198 | 0.196 | 0.225 |
| B_tr_op | comp | 0.552 | 0.621 | 0.486 | 0.517 | 0.576 |
| B_tr_op | comp+rel·PoE | 0.536 | 0.638 | 0.494 | 0.499 | 0.589 |
| B_tr_op | comp+mix | 0.593 | 0.603 | 0.446 | 0.416 | 0.448 |
| B_tr_dd | comp | 0.617 | 0.712 | 0.518 | 0.542 | 0.638 |
| B_tr_dd | comp+rel·PoE | 0.580 | 0.687 | 0.521 | 0.554 | 0.638 |
| B_tr_dd | comp+mix | 0.635 | 0.607 | 0.478 | 0.461 | 0.538 |

Testy zadeklarowane z góry (27, Holm):

| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| T1 Most − dawny Softmax | B_tr_op | +0.035 | [-0.030, +0.101] | 14/6 | 1.0000 | nie |
| T1 jądro | B_tr_op | -0.069 | [-0.113, -0.027] | 5/14 | 0.1210 | nie |
| T1 informacja ΔE | B_tr_op | +0.104 | [+0.059, +0.149] | 17/2 | 0.0114 | **tak** |
| T1 Most − dawny Softmax | B_tr_dd | +0.075 | [-0.025, +0.173] | 13/7 | 0.9586 | nie |
| T1 jądro | B_tr_dd | -0.095 | [-0.193, -0.001] | 5/12 | 0.6442 | nie |
| T1 informacja ΔE | B_tr_dd | +0.171 | [+0.090, +0.252] | 16/2 | 0.0239 | **tak** |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.040 | [-0.004, +0.085] | 10/9 | 0.7338 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.025 | [-0.008, +0.058] | 12/8 | 0.9586 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.079 | [+0.037, +0.123] | 14/4 | 0.0331 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.044 | [+0.003, +0.089] | 11/6 | 0.6442 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | +0.043 | [+0.006, +0.079] | 11/8 | 0.5286 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.073 | [+0.029, +0.117] | 13/4 | 0.1102 | nie |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.053 | [+0.004, +0.107] | 11/7 | 0.6442 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.037 | [+0.002, +0.073] | 13/6 | 0.6442 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.076 | [+0.022, +0.131] | 11/4 | 0.2170 | nie |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.068 | [+0.031, +0.106] | 15/3 | 0.0450 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.049 | [+0.021, +0.079] | 15/2 | 0.0558 | nie |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.076 | [+0.029, +0.123] | 14/5 | 0.1210 | nie |
| T2 relacja (PoE−comp) · mlp | A_dół | +0.117 | [+0.083, +0.151] | 18/0 | 0.0002 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.085 | [+0.061, +0.110] | 19/1 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.134 | [+0.088, +0.181] | 18/2 | 0.0013 | **tak** |
| T3 jądro w PoE | A_dół | +0.003 | [-0.034, +0.036] | 11/7 | 1.0000 | nie |
| T3 jądro w PoE | B_te_hop | -0.007 | [-0.036, +0.021] | 9/11 | 1.0000 | nie |
| T3 jądro w PoE | B_te_dd | +0.006 | [-0.049, +0.064] | 11/8 | 1.0000 | nie |
| T4 mix−comp · bw·ΔE | A_dół | +0.059 | [+0.021, +0.097] | 12/6 | 0.1517 | nie |
| T4 mix−comp · bw·ΔE | B_te_hop | +0.043 | [+0.006, +0.080] | 13/6 | 0.5286 | nie |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.058 | [+0.021, +0.094] | 14/5 | 0.1279 | nie |

## kana

Zapytań na przebieg (średnio): A_dół=21.4, B_te=77.0, B_te_dd=23.9, B_tr=36.0, B_tr_dd=11.2.

Średnie hit@1:

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp |
|---|---|---:|---:|---:|---:|---:|
| A_dół | comp | 0.052 | 0.057 | 0.038 | 0.038 | 0.026 |
| A_dół | comp+rel·PoE | 0.039 | 0.021 | 0.021 | 0.019 | 0.019 |
| A_dół | comp+mix | 0.042 | 0.029 | 0.033 | 0.042 | 0.035 |
| B_te_hop | comp | 0.041 | 0.042 | 0.028 | 0.033 | 0.022 |
| B_te_hop | comp+rel·PoE | 0.050 | 0.036 | 0.032 | 0.034 | 0.027 |
| B_te_hop | comp+mix | 0.039 | 0.034 | 0.032 | 0.033 | 0.026 |
| B_te_dd | comp | 0.038 | 0.040 | 0.035 | 0.028 | 0.025 |
| B_te_dd | comp+rel·PoE | 0.069 | 0.087 | 0.082 | 0.081 | 0.066 |
| B_te_dd | comp+mix | 0.049 | 0.053 | 0.030 | 0.036 | 0.042 |
| B_tr_op | comp | 0.589 | 0.583 | 0.655 | 0.677 | 0.686 |
| B_tr_op | comp+rel·PoE | 0.687 | 0.667 | 0.670 | 0.688 | 0.749 |
| B_tr_op | comp+mix | 0.627 | 0.633 | 0.626 | 0.607 | 0.825 |
| B_tr_dd | comp | 0.351 | 0.354 | 0.361 | 0.391 | 0.320 |
| B_tr_dd | comp+rel·PoE | 0.494 | 0.460 | 0.426 | 0.414 | 0.428 |
| B_tr_dd | comp+mix | 0.552 | 0.549 | 0.492 | 0.460 | 0.684 |

Testy zadeklarowane z góry (27, Holm):

| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| T1 Most − dawny Softmax | B_tr_op | -0.088 | [-0.117, -0.059] | 1/19 | 0.0003 | **tak** |
| T1 jądro | B_tr_op | +0.006 | [-0.021, +0.032] | 12/8 | 1.0000 | nie |
| T1 informacja ΔE | B_tr_op | -0.094 | [-0.128, -0.063] | 1/17 | 0.0006 | **tak** |
| T1 Most − dawny Softmax | B_tr_dd | -0.040 | [-0.110, +0.031] | 4/8 | 1.0000 | nie |
| T1 jądro | B_tr_dd | -0.003 | [-0.076, +0.069] | 8/9 | 1.0000 | nie |
| T1 informacja ΔE | B_tr_dd | -0.037 | [-0.111, +0.032] | 7/10 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | -0.012 | [-0.043, +0.018] | 4/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.008 | [-0.005, +0.023] | 8/7 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.031 | [+0.009, +0.053] | 12/3 | 0.2588 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | -0.036 | [-0.061, -0.012] | 3/11 | 0.2205 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | -0.006 | [-0.019, +0.007] | 6/10 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.048 | [+0.018, +0.078] | 12/3 | 0.1517 | nie |
| T2 relacja (PoE−comp) · bw·0 | A_dół | -0.016 | [-0.033, +0.000] | 3/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.004 | [-0.007, +0.015] | 10/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.047 | [+0.021, +0.074] | 13/3 | 0.0710 | nie |
| T2 relacja (PoE−comp) · sm·0 | A_dół | -0.019 | [-0.040, +0.002] | 5/10 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.001 | [-0.010, +0.013] | 8/9 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.053 | [+0.029, +0.078] | 15/3 | 0.0313 | **tak** |
| T2 relacja (PoE−comp) · mlp | A_dół | -0.007 | [-0.026, +0.012] | 5/6 | 1.0000 | nie |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.005 | [-0.004, +0.014] | 8/5 | 1.0000 | nie |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.041 | [+0.016, +0.063] | 16/3 | 0.1234 | nie |
| T3 jądro w PoE | A_dół | +0.018 | [+0.002, +0.037] | 7/2 | 1.0000 | nie |
| T3 jądro w PoE | B_te_hop | +0.013 | [-0.002, +0.030] | 10/8 | 1.0000 | nie |
| T3 jądro w PoE | B_te_dd | -0.019 | [-0.043, +0.005] | 5/9 | 1.0000 | nie |
| T4 mix−comp · bw·ΔE | A_dół | -0.009 | [-0.040, +0.017] | 9/7 | 1.0000 | nie |
| T4 mix−comp · bw·ΔE | B_te_hop | -0.002 | [-0.020, +0.014] | 9/7 | 1.0000 | nie |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.012 | [-0.009, +0.034] | 7/6 | 1.0000 | nie |

## music

Zapytań na przebieg (średnio): A_dół=36.0, B_te=132.4, B_te_dd=38.9, B_tr=69.6, B_tr_dd=21.1.

Średnie hit@1:

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp |
|---|---|---:|---:|---:|---:|---:|
| A_dół | comp | 0.539 | 0.572 | 0.477 | 0.462 | 0.683 |
| A_dół | comp+rel·PoE | 0.679 | 0.734 | 0.643 | 0.634 | 0.749 |
| A_dół | comp+mix | 0.608 | 0.636 | 0.546 | 0.568 | 0.682 |
| B_te_hop | comp | 0.386 | 0.431 | 0.317 | 0.312 | 0.521 |
| B_te_hop | comp+rel·PoE | 0.503 | 0.552 | 0.455 | 0.452 | 0.596 |
| B_te_hop | comp+mix | 0.443 | 0.460 | 0.403 | 0.413 | 0.554 |
| B_te_dd | comp | 0.128 | 0.131 | 0.116 | 0.132 | 0.185 |
| B_te_dd | comp+rel·PoE | 0.236 | 0.267 | 0.224 | 0.226 | 0.243 |
| B_te_dd | comp+mix | 0.291 | 0.319 | 0.292 | 0.305 | 0.283 |
| B_tr_op | comp | 0.710 | 0.729 | 0.709 | 0.714 | 0.778 |
| B_tr_op | comp+rel·PoE | 0.760 | 0.762 | 0.771 | 0.777 | 0.803 |
| B_tr_op | comp+mix | 0.811 | 0.834 | 0.850 | 0.862 | 0.888 |
| B_tr_dd | comp | 0.368 | 0.427 | 0.372 | 0.371 | 0.451 |
| B_tr_dd | comp+rel·PoE | 0.401 | 0.390 | 0.431 | 0.435 | 0.467 |
| B_tr_dd | comp+mix | 0.639 | 0.672 | 0.682 | 0.725 | 0.750 |

Testy zadeklarowane z góry (27, Holm):

| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| T1 Most − dawny Softmax | B_tr_op | -0.004 | [-0.033, +0.025] | 7/9 | 1.0000 | nie |
| T1 jądro | B_tr_op | -0.019 | [-0.042, +0.002] | 7/10 | 0.6046 | nie |
| T1 informacja ΔE | B_tr_op | +0.015 | [-0.022, +0.054] | 8/9 | 1.0000 | nie |
| T1 Most − dawny Softmax | B_tr_dd | -0.003 | [-0.070, +0.068] | 8/9 | 1.0000 | nie |
| T1 jądro | B_tr_dd | -0.059 | [-0.132, +0.002] | 6/9 | 0.6046 | nie |
| T1 informacja ΔE | B_tr_dd | +0.055 | [-0.034, +0.151] | 9/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.140 | [+0.102, +0.175] | 18/1 | 0.0002 | **tak** |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.116 | [+0.089, +0.146] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.107 | [+0.075, +0.138] | 16/0 | 0.0005 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.161 | [+0.127, +0.199] | 19/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | +0.120 | [+0.094, +0.150] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.137 | [+0.102, +0.175] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.167 | [+0.106, +0.229] | 17/2 | 0.0011 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.138 | [+0.103, +0.179] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.108 | [+0.081, +0.134] | 18/1 | 0.0002 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.171 | [+0.129, +0.215] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.140 | [+0.109, +0.173] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.093 | [+0.064, +0.120] | 17/2 | 0.0006 | **tak** |
| T2 relacja (PoE−comp) · mlp | A_dół | +0.066 | [+0.029, +0.102] | 13/3 | 0.0386 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.075 | [+0.045, +0.107] | 17/3 | 0.0006 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.057 | [+0.025, +0.089] | 15/4 | 0.0386 | **tak** |
| T3 jądro w PoE | A_dół | -0.055 | [-0.085, -0.026] | 4/15 | 0.0257 | **tak** |
| T3 jądro w PoE | B_te_hop | -0.049 | [-0.073, -0.025] | 3/15 | 0.0090 | **tak** |
| T3 jądro w PoE | B_te_dd | -0.032 | [-0.056, -0.010] | 3/11 | 0.1256 | nie |
| T4 mix−comp · bw·ΔE | A_dół | +0.069 | [+0.020, +0.116] | 14/6 | 0.1235 | nie |
| T4 mix−comp · bw·ΔE | B_te_hop | +0.057 | [+0.018, +0.094] | 14/6 | 0.1003 | nie |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.163 | [+0.116, +0.211] | 18/1 | 0.0002 | **tak** |

## Zestawienie między dziedzinami (Δ, w nawiasie: utrzymuje się po Holmie?)

| test | miara | periodic | kana | music |
|---|---|---:|---:|---:|
| T1 Most − dawny Softmax | B_tr_op | +0.035 (nie) | -0.088 (tak) | -0.004 (nie) |
| T1 jądro | B_tr_op | -0.069 (nie) | +0.006 (nie) | -0.019 (nie) |
| T1 informacja ΔE | B_tr_op | +0.104 (tak) | -0.094 (tak) | +0.015 (nie) |
| T1 Most − dawny Softmax | B_tr_dd | +0.075 (nie) | -0.040 (nie) | -0.003 (nie) |
| T1 jądro | B_tr_dd | -0.095 (nie) | -0.003 (nie) | -0.059 (nie) |
| T1 informacja ΔE | B_tr_dd | +0.171 (tak) | -0.037 (nie) | +0.055 (nie) |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.040 (nie) | -0.012 (nie) | +0.140 (tak) |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.025 (nie) | +0.008 (nie) | +0.116 (tak) |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.079 (tak) | +0.031 (nie) | +0.107 (tak) |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.044 (nie) | -0.036 (nie) | +0.161 (tak) |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | +0.043 (nie) | -0.006 (nie) | +0.120 (tak) |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.073 (nie) | +0.048 (nie) | +0.137 (tak) |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.053 (nie) | -0.016 (nie) | +0.167 (tak) |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.037 (nie) | +0.004 (nie) | +0.138 (tak) |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.076 (nie) | +0.047 (nie) | +0.108 (tak) |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.068 (tak) | -0.019 (nie) | +0.171 (tak) |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.049 (nie) | +0.001 (nie) | +0.140 (tak) |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.076 (nie) | +0.053 (tak) | +0.093 (tak) |
| T2 relacja (PoE−comp) · mlp | A_dół | +0.117 (tak) | -0.007 (nie) | +0.066 (tak) |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.085 (tak) | +0.005 (nie) | +0.075 (tak) |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.134 (tak) | +0.041 (nie) | +0.057 (tak) |
| T3 jądro w PoE | A_dół | +0.003 (nie) | +0.018 (nie) | -0.055 (tak) |
| T3 jądro w PoE | B_te_hop | -0.007 (nie) | +0.013 (nie) | -0.049 (tak) |
| T3 jądro w PoE | B_te_dd | +0.006 (nie) | -0.019 (nie) | -0.032 (nie) |
| T4 mix−comp · bw·ΔE | A_dół | +0.059 (nie) | -0.009 (nie) | +0.069 (nie) |
| T4 mix−comp · bw·ΔE | B_te_hop | +0.043 (nie) | -0.002 (nie) | +0.057 (nie) |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.058 (nie) | +0.012 (nie) | +0.163 (tak) |

## Wnioski

**Zastrzeżenia:**
- Kana ma tylko 45 elementów (test ~14, ~21 zapytań „dół” na przebieg). Losowe hit@1 ≈ 0.02.
- Cechy muzyki są syntetyczne (idealne widmo harmoniczne), kana to prawdziwe glify, układ okresowy to prawdziwe konfiguracje.
- Ścisły protokół (bez cech testowych w treningu) różni się od poprzednich raportów, w których enkoder widział cechy wszystkich elementów (bez etykiet). Dlatego liczby dla układu okresowego są tu niższe.
- Kolejność ramion w tabelach średnich: bw·ΔE, sm·ΔE, bw·0, sm·0, mlp.

1. **Struktura relacyjna (relacja grupy + PoE) przenosi się poza układ okresowy, o ile cechy niosą jakikolwiek sygnał o kolumnie.**

   | PoE − comp, nowe elementy | układ okresowy | muzyka | kana |
   |---|---|---|---|
   | krok „dół” | +0.04…+0.12 (2/5 ramion przeżywa korektę) | **+0.07…+0.17 (5/5)** | −0.04…−0.01 (0/5) |
   | 2 kroki | +0.03…+0.09 (1/5) | **+0.08…+0.14 (5/5)** | ≈ 0 (0/5) |
   | dół∘dół | +0.07…+0.13 (2/5) | **+0.06…+0.14 (5/5)** | +0.03…+0.05 (1/5) |

   - W muzyce efekt jest największy i najpewniejszy, we wszystkich 5 ramionach (każde jądro i MLP). Cechy (wspólne harmoniczne) są tam „pionowe”, więc relacja grupy tylko wzmacnia to, co cechy już niosą.
   - W układzie okresowym znak jest zawsze dodatni, a efekt najsilniejszy w MLP.
   - W kanie nie ma nic do przeniesienia: kształt glifu nie mówi o samogłosce, więc na nowych znakach **wszystkie** modele są na poziomie losowym (0.02–0.06).
2. **Jądro Breita–Wignera nie daje przewagi w żadnej dziedzinie.**
   - Kompozycja na znanych, `bw·ΔE − sm·ΔE`: n.s. w 6/6 (Δ od −0.09 do +0.01).
   - Jądro w PoE: 0 w układzie okresowym i kanie. **W muzyce Softmax jest lepszy** (−0.05, p_Holm < 0.05).
   - Uwaga: zysk jądra w PoE z ablacji KERNEL (+0.09, protokół transduktywny) **znika w ścisłym protokole** także w samym układzie okresowym (+0.003). Był więc zależny od tego, że enkoder widział w treningu cechy elementów testowych.
3. **Wpływ informacji ΔE zależy od dziedziny.** Pomaga w układzie okresowym (kompozycja na znanych +0.10 / +0.17, przeżywa korektę), szkodzi w kanie (−0.09, przeżywa), w muzyce jest obojętna. ΔE pomaga tylko wtedy, gdy tracer niesie informację istotną dla relacji (zewnętrzne podpowłoki). Statystyki glifu takiej informacji nie niosą.
4. **Kompozycja na znanych elementach działa wszędzie** (0.55–0.80 we wszystkich dziedzinach i ramionach), także w kanie, gdzie cechy nie mają struktury. Operatory „dół”/„prawo” uczą się składać na elementach, które model zna, niezależnie od cech. Indukcja na nowe elementy wymaga już cech niosących strukturę.
5. **W ścisłym protokole najlepiej generalizuje MLP bez uwagi.** Krok „dół” w PoE: układ okresowy 0.51 (uwaga 0.32–0.36), muzyka 0.75 (0.63–0.73). Uwaga między elementami nie pomaga, gdy podczas treningu model nie widzi kontekstu elementów testowych.

## Werdykt

Odkryty efekt **nie jest specyficzny dla układu okresowego, ale jest warunkowy**. Dodatkowa struktura relacyjna (relacja kolumny z PoE) poprawia składanie relacji na nowych elementach także dla dźwięków muzycznych, i to silniej niż dla pierwiastków, w każdym jądrze i bez uwagi. Nie pomaga tam, gdzie cechy elementów nie niosą żadnej informacji o kolumnie (glify kana).

Jądro Breita–Wignera nie przenosi przewagi do żadnej dziedziny. W ścisłym protokole nie ma jej nawet w układzie okresowym.

**Uogólnia się więc idea „osobnej relacji dla osobnej osi struktury”, a nie konkretny mechanizm uwagi.**

