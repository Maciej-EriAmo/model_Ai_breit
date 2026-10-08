# Zadanie kompozycyjne — Układ Słoneczny i trójka hierarchiczna

Ten sam protokół co przy pierwiastkach, kanie i muzyce (`run_crossdomain_comp.py`): relacje „dół” i „prawo”, ramiona bw·ΔE / sm·ΔE / bw·0 / sm·0 / mlp, warianty comp / PoE / mix, 800 kroków, 20 przebiegów. **Ścisłe rozdzielenie:** w treningu enkoder widzi tylko elementy treningowe. Testy T1–T4 zadeklarowane przed uruchomieniem, Holm w obrębie dziedziny. Skrypt `archiwum/skrypty/run_nbody_comp.py`; przebiegi w `nbody_comp_runs.jsonl`; specyfikacje `nbody_comp_spec_<dziedzina>.json`.

Stary bench n-ciał (etykieta warstwy, one-hot hierarchii w tracerze) nie jest tym pomiarem.

| dziedzina | elementy | kolumny | osie | cechy | podobieństwo w surowych cechach |
|---|---:|---:|---|---|---|
| solar | 80 | 8 | kolumna=klasa masy, wiersz=powłoka a, dół=ta sama masa dalej, prawo=kolejność (a, masa) | ε, v, ℓ, log m, log a, log r_Hilla (z-score, L2); τ = te same 6, min-max | 1-NN: ta sama kolumna 4%, ten sam wiersz 95% |
| three | 84 | 3 | kolumna=rola w trójce, wiersz=a_out, dół=ta sama rola szersza trójka, prawo=następna rola wewnątrz trójki | log m, r_CM, v, ε, Lz, potencjał, r_min (z-score, L2); τ = pierwsze 6, min-max | 1-NN: ta sama kolumna 100%, ten sam wiersz 0% |

## solar

kolumna=klasa masy, wiersz=powłoka a, dół=ta sama masa dalej, prawo=kolejność (a, masa)

Zapytań na przebieg (średnio): A_dół=37.4, B_te=138.3, B_te_dd=42.9, B_tr=67.7, B_tr_dd=21.1.

Średnie hit@1:

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp |
|---|---|---:|---:|---:|---:|---:|
| A_dół | comp | 0.904 | 0.813 | 0.925 | 0.921 | 0.908 |
| A_dół | comp+rel·PoE | 0.963 | 0.927 | 0.972 | 0.975 | 0.984 |
| A_dół | comp+mix | 0.972 | 0.934 | 0.972 | 0.962 | 0.943 |
| B_te_hop | comp | 0.861 | 0.787 | 0.895 | 0.889 | 0.832 |
| B_te_hop | comp+rel·PoE | 0.929 | 0.898 | 0.938 | 0.937 | 0.902 |
| B_te_hop | comp+mix | 0.934 | 0.896 | 0.945 | 0.942 | 0.861 |
| B_te_dd | comp | 0.431 | 0.362 | 0.485 | 0.490 | 0.519 |
| B_te_dd | comp+rel·PoE | 0.484 | 0.442 | 0.539 | 0.543 | 0.574 |
| B_te_dd | comp+mix | 0.578 | 0.583 | 0.620 | 0.605 | 0.386 |
| B_tr_op | comp | 0.835 | 0.786 | 0.866 | 0.868 | 0.719 |
| B_tr_op | comp+rel·PoE | 0.841 | 0.800 | 0.857 | 0.855 | 0.695 |
| B_tr_op | comp+mix | 0.763 | 0.733 | 0.821 | 0.822 | 0.507 |
| B_tr_dd | comp | 0.699 | 0.664 | 0.722 | 0.720 | 0.625 |
| B_tr_dd | comp+rel·PoE | 0.678 | 0.667 | 0.678 | 0.683 | 0.629 |
| B_tr_dd | comp+mix | 0.659 | 0.672 | 0.717 | 0.731 | 0.481 |

Testy zadeklarowane z góry (27, Holm):

| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| T1 Most − dawny Softmax | B_tr_op | -0.033 | [-0.067, +0.001] | 6/13 | 0.5421 | nie |
| T1 jądro | B_tr_op | +0.049 | [+0.018, +0.077] | 16/4 | 0.0669 | nie |
| T1 informacja ΔE | B_tr_op | -0.082 | [-0.111, -0.053] | 2/18 | 0.0011 | **tak** |
| T1 Most − dawny Softmax | B_tr_dd | -0.021 | [-0.076, +0.041] | 5/12 | 0.5421 | nie |
| T1 jądro | B_tr_dd | +0.035 | [-0.016, +0.086] | 9/8 | 0.5421 | nie |
| T1 informacja ΔE | B_tr_dd | -0.056 | [-0.115, +0.003] | 8/11 | 0.5421 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.059 | [+0.032, +0.092] | 15/1 | 0.0029 | **tak** |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.068 | [+0.039, +0.102] | 19/1 | 0.0004 | **tak** |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.053 | [-0.008, +0.115] | 14/6 | 0.5421 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.114 | [+0.079, +0.150] | 18/1 | 0.0002 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | +0.111 | [+0.078, +0.147] | 18/1 | 0.0002 | **tak** |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.080 | [+0.045, +0.116] | 16/3 | 0.0071 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.047 | [+0.029, +0.066] | 16/1 | 0.0006 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.043 | [+0.021, +0.065] | 15/4 | 0.0137 | **tak** |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.055 | [+0.000, +0.106] | 13/6 | 0.5421 | nie |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.055 | [+0.037, +0.073] | 17/0 | 0.0004 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.048 | [+0.029, +0.066] | 16/4 | 0.0026 | **tak** |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.053 | [-0.004, +0.110] | 13/6 | 0.5421 | nie |
| T2 relacja (PoE−comp) · mlp | A_dół | +0.076 | [+0.054, +0.100] | 20/0 | 0.0001 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.070 | [+0.050, +0.090] | 18/1 | 0.0003 | **tak** |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.055 | [+0.016, +0.094] | 11/7 | 0.1334 | nie |
| T3 jądro w PoE | A_dół | +0.037 | [+0.018, +0.055] | 14/3 | 0.0200 | **tak** |
| T3 jądro w PoE | B_te_hop | +0.031 | [+0.014, +0.048] | 15/4 | 0.0340 | **tak** |
| T3 jądro w PoE | B_te_dd | +0.042 | [-0.001, +0.086] | 12/7 | 0.5421 | nie |
| T4 mix−comp · bw·ΔE | A_dół | +0.068 | [+0.037, +0.104] | 15/1 | 0.0032 | **tak** |
| T4 mix−comp · bw·ΔE | B_te_hop | +0.072 | [+0.042, +0.108] | 18/1 | 0.0005 | **tak** |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.147 | [+0.106, +0.190] | 17/0 | 0.0004 | **tak** |

## three

kolumna=rola w trójce, wiersz=a_out, dół=ta sama rola szersza trójka, prawo=następna rola wewnątrz trójki

Zapytań na przebieg (średnio): A_dół=41.5, B_te=124.3, B_te_dd=51.6, B_tr=61.6, B_tr_dd=26.4.

Średnie hit@1:

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp |
|---|---|---:|---:|---:|---:|---:|
| A_dół | comp | 0.874 | 0.914 | 0.877 | 0.884 | 0.580 |
| A_dół | comp+rel·PoE | 0.917 | 0.931 | 0.918 | 0.922 | 0.554 |
| A_dół | comp+mix | 0.619 | 0.686 | 0.578 | 0.579 | 0.201 |
| B_te_hop | comp | 0.449 | 0.545 | 0.477 | 0.501 | 0.367 |
| B_te_hop | comp+rel·PoE | 0.487 | 0.537 | 0.531 | 0.534 | 0.355 |
| B_te_hop | comp+mix | 0.355 | 0.402 | 0.343 | 0.335 | 0.125 |
| B_te_dd | comp | 0.440 | 0.432 | 0.469 | 0.463 | 0.231 |
| B_te_dd | comp+rel·PoE | 0.483 | 0.433 | 0.497 | 0.501 | 0.183 |
| B_te_dd | comp+mix | 0.313 | 0.300 | 0.283 | 0.323 | 0.084 |
| B_tr_op | comp | 0.348 | 0.357 | 0.341 | 0.401 | 0.303 |
| B_tr_op | comp+rel·PoE | 0.354 | 0.347 | 0.405 | 0.399 | 0.301 |
| B_tr_op | comp+mix | 0.256 | 0.267 | 0.313 | 0.303 | 0.065 |
| B_tr_dd | comp | 0.502 | 0.434 | 0.509 | 0.541 | 0.370 |
| B_tr_dd | comp+rel·PoE | 0.489 | 0.421 | 0.525 | 0.505 | 0.301 |
| B_tr_dd | comp+mix | 0.354 | 0.336 | 0.348 | 0.385 | 0.063 |

Testy zadeklarowane z góry (27, Holm):

| test | miara | Δ | 95% CI | wygr./przegr. | p_Holm | utrzymuje się? |
|---|---|---:|---|---:|---:|:-:|
| T1 Most − dawny Softmax | B_tr_op | -0.053 | [-0.140, +0.030] | 9/11 | 1.0000 | nie |
| T1 jądro | B_tr_op | -0.009 | [-0.091, +0.075] | 7/13 | 1.0000 | nie |
| T1 informacja ΔE | B_tr_op | -0.044 | [-0.108, +0.020] | 8/12 | 1.0000 | nie |
| T1 Most − dawny Softmax | B_tr_dd | -0.039 | [-0.127, +0.046] | 10/9 | 1.0000 | nie |
| T1 jądro | B_tr_dd | +0.068 | [-0.035, +0.165] | 13/7 | 1.0000 | nie |
| T1 informacja ΔE | B_tr_dd | -0.107 | [-0.173, -0.042] | 5/13 | 0.1377 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.043 | [-0.005, +0.090] | 14/5 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.038 | [-0.008, +0.082] | 11/7 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.043 | [-0.036, +0.120] | 12/7 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.017 | [-0.009, +0.045] | 9/5 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | -0.008 | [-0.055, +0.039] | 9/10 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.001 | [-0.058, +0.056] | 12/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.041 | [-0.006, +0.087] | 14/5 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.054 | [-0.015, +0.120] | 12/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.028 | [-0.049, +0.106] | 10/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.037 | [+0.004, +0.070] | 13/4 | 0.9821 | nie |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.033 | [-0.013, +0.080] | 13/7 | 1.0000 | nie |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.038 | [-0.041, +0.115] | 12/8 | 1.0000 | nie |
| T2 relacja (PoE−comp) · mlp | A_dół | -0.026 | [-0.072, +0.020] | 9/11 | 1.0000 | nie |
| T2 relacja (PoE−comp) · mlp | B_te_hop | -0.012 | [-0.050, +0.028] | 7/12 | 1.0000 | nie |
| T2 relacja (PoE−comp) · mlp | B_te_dd | -0.048 | [-0.088, -0.009] | 5/13 | 0.6928 | nie |
| T3 jądro w PoE | A_dół | -0.014 | [-0.043, +0.015] | 7/9 | 1.0000 | nie |
| T3 jądro w PoE | B_te_hop | -0.050 | [-0.094, -0.007] | 8/12 | 0.8836 | nie |
| T3 jądro w PoE | B_te_dd | +0.050 | [-0.002, +0.107] | 11/7 | 1.0000 | nie |
| T4 mix−comp · bw·ΔE | A_dół | -0.255 | [-0.325, -0.188] | 0/19 | 0.0001 | **tak** |
| T4 mix−comp · bw·ΔE | B_te_hop | -0.094 | [-0.142, -0.047] | 5/15 | 0.0356 | **tak** |
| T4 mix−comp · bw·ΔE | B_te_dd | -0.127 | [-0.204, -0.054] | 4/14 | 0.1022 | nie |

## Zestawienie (Δ, w nawiasie: utrzymuje się po Holmie?)

| test | miara | solar | three |
|---|---|---:|---:|
| T1 Most − dawny Softmax | B_tr_op | -0.033 (nie) | -0.053 (nie) |
| T1 jądro | B_tr_op | +0.049 (nie) | -0.009 (nie) |
| T1 informacja ΔE | B_tr_op | -0.082 (tak) | -0.044 (nie) |
| T1 Most − dawny Softmax | B_tr_dd | -0.021 (nie) | -0.039 (nie) |
| T1 jądro | B_tr_dd | +0.035 (nie) | +0.068 (nie) |
| T1 informacja ΔE | B_tr_dd | -0.056 (nie) | -0.107 (nie) |
| T2 relacja (PoE−comp) · bw·ΔE | A_dół | +0.059 (tak) | +0.043 (nie) |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_hop | +0.068 (tak) | +0.038 (nie) |
| T2 relacja (PoE−comp) · bw·ΔE | B_te_dd | +0.053 (nie) | +0.043 (nie) |
| T2 relacja (PoE−comp) · sm·ΔE | A_dół | +0.114 (tak) | +0.017 (nie) |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_hop | +0.111 (tak) | -0.008 (nie) |
| T2 relacja (PoE−comp) · sm·ΔE | B_te_dd | +0.080 (tak) | +0.001 (nie) |
| T2 relacja (PoE−comp) · bw·0 | A_dół | +0.047 (tak) | +0.041 (nie) |
| T2 relacja (PoE−comp) · bw·0 | B_te_hop | +0.043 (tak) | +0.054 (nie) |
| T2 relacja (PoE−comp) · bw·0 | B_te_dd | +0.055 (nie) | +0.028 (nie) |
| T2 relacja (PoE−comp) · sm·0 | A_dół | +0.055 (tak) | +0.037 (nie) |
| T2 relacja (PoE−comp) · sm·0 | B_te_hop | +0.048 (tak) | +0.033 (nie) |
| T2 relacja (PoE−comp) · sm·0 | B_te_dd | +0.053 (nie) | +0.038 (nie) |
| T2 relacja (PoE−comp) · mlp | A_dół | +0.076 (tak) | -0.026 (nie) |
| T2 relacja (PoE−comp) · mlp | B_te_hop | +0.070 (tak) | -0.012 (nie) |
| T2 relacja (PoE−comp) · mlp | B_te_dd | +0.055 (nie) | -0.048 (nie) |
| T3 jądro w PoE | A_dół | +0.037 (tak) | -0.014 (nie) |
| T3 jądro w PoE | B_te_hop | +0.031 (tak) | -0.050 (nie) |
| T3 jądro w PoE | B_te_dd | +0.042 (nie) | +0.050 (nie) |
| T4 mix−comp · bw·ΔE | A_dół | +0.068 (tak) | -0.255 (tak) |
| T4 mix−comp · bw·ΔE | B_te_hop | +0.072 (tak) | -0.094 (tak) |
| T4 mix−comp · bw·ΔE | B_te_dd | +0.147 (tak) | -0.127 (nie) |

## Wnioski

**Zastrzeżenia:**
- To katalogi syntetyczne, nie 9 planet i nie chaotyczne trzy ciała o równych masach. Słońce: 80 orbit kołowych, masa centralna 1. Trójka: 28 współliniowych trójek hierarchicznych (para wewnętrzna + ciało na szerszej orbicie).
- W Słońcu cztery z sześciu składowych są funkcją samej odległości, więc masa jest w wektorze, ale nie w sąsiedztwie (1-NN: ta sama powłoka 95%, ta sama masa 4%).
- W trójce rola jest całą sąsiedztwem (1-NN: ta sama rola 100%). Krok „dół” idzie po gładkim torze tej roli.
- Kolejność ramion: bw·ΔE, sm·ΔE, bw·0, sm·0, mlp. Holm osobno w każdej dziedzinie, 27 testów.

1. **Osobna relacja kolumny pomaga tam, gdzie kolumna jest składową, ale nie jest najbliższym sąsiadem.** W układzie słonecznym PoE − comp na nowych ciałach: jeden krok „dół” +0.05…+0.11 i hop +0.04…+0.11, wszystkie 5 ramion, p_Holm ≤ 0.014. Dwa razy „dół” przeżywa korektę tylko u Softmaxu z ΔE (+0.08). Sam operator bez relacji kolumny już trafia w ~0.81–0.93, bo powłoki są gładkie. Relacja masy dokłada kilka punktów, nie tworzy umiejętności od zera.

2. **Gdy kolumna jest już całym sąsiedztwem, dodatkowa głowica nic nie daje, a wmieszanie jej w tę samą przestrzeń szkodzi.** W trójce żaden z 15 testów PoE − comp nie przeżywa korekty (Δ od −0.05 do +0.05). Krok „dół” bez relacji to już 0.87–0.91 dla jąder uwagi. Mix − comp dla Mostu: −0.26 na kroku „dół” (0/19) i −0.09 na hopie. Oba przeżywają korektę. Rola jest w cechach w całości, więc druga kopia tej samej relacji rozmywa tor odległości.

3. **Jądro Breita–Wignera nie wygrywa z najlepszym Softmaxem.** W trójce różnica jąder nie przeżywa korekty. W Słońcu bw·ΔE jest lepszy od sm·ΔE o +0.04 i +0.03 (krok „dół” i hop, p_Holm 0.02 i 0.03), ale ta para to Softmax z karą ΔE. Ta kara sama obniża składanie na znanych ciałach: sm·ΔE − sm·0 = −0.08, p_Holm 0.001. Wobec Softmaxa bez kary (sm·0) Most nie jest lepszy (T1, obie miary, korekta nie przechodzi). Na kroku „dół” z PoE najwyżej jest MLP (0.984), potem sm·0 (0.975) i bw·0 (0.972), a Most z ΔE ma 0.963.

4. **Składanie na znanych ciałach zależy od tego, czy druga oś jest blisko w cechach.** W Słońcu B_tr_op to 0.72–0.87. W trójce tylko 0.30–0.40: rola pierwotny → wtórny → zewnętrzny jest skokiem, nie przesunięciem po gładkim torze, więc złożenie operatorów na znanych ciałach zostaje słabe mimo że pojedynczy krok wzdłuż roli jest łatwy.

## Werdykt

Relacja jako osobna oś przenosi się na te dwie dziedziny tylko w jedną stronę. W układzie słonecznym masa jest cichą składową (w wektorze, nie w sąsiedztwie) i osobna relacja kolumny podnosi krok wzdłuż masy o kilka punktów, we wszystkich ramionach. W trójce hierarchicznej rola jest głośną składową i ten sam dodatek nic nie daje, a wmieszanie go w jedną przestrzeń obniża wynik o ćwierć.

Jądro Breita–Wignera nie jest tu potrzebne. Tam, gdzie wychodzi liczbowo nad Softmaxem, Softmax jest obciążony karą ΔE, a bez tej kary różnicy nie ma.

**Powtarza się więc warunek z pierwiastków i muzyki, z dopiskiem:** osobna relacja pomaga, gdy oś jest w stanie, a model sam jej nie widzi jako najbliższego sąsiada. Nie pomaga, gdy oś już jest całym sąsiedztwem. Wzór uwagi nie jest tą składową.

