# T1 — notacja muzyczna jako dodatkowe zmienne

Katalog syntetyczny, 80 zdarzeń. Widmo ośmiu harmonicznych jak dotychczasowa muzyka. Pięć par enharmonicznych (Cs/Db, Ds/Eb, Fs/Gb, Gs/Ab, As/Bb) w ośmiu oktawach: bliźniak ma to samo widmo. Kolumna = pisownia, wiersz = oktawa, dół = ta sama pisownia i następna oktawa. W tracerze (plugin, 6 składowych): pisownia, wartość rytmiczna, miejsce w takcie, dynamika, artykulacja, głos. Nie wynikają z widma.

Ścisła indukcja, 800 kroków, 20 przebiegów. Specyfikacja `jadro_t1_spec.json`, przebiegi `jadro_t1_runs.jsonl`. Skrypt `archiwum/skrypty/run_jadro_t1.py`. Na samym widmie: ta sama pisownia 0%, ta sama oktawa 100%.

Rodzina Holma (11 testów) ustalona przed uruchomieniem. N1: PoE − comp na A_dół, pięć ramion protokołu. N2: bw·tok − sm·tok, wariant comp, miary nowych zdarzeń. N3: sm·tok − sm·ΔE, te same miary. Hop i dwa kroki w dół są w N2 i N3, bo program mówi o miarach liczby mnogiej; N1 zostaje przy jednym kroku.

## Średnie hit@1

| miara | wariant | bw·ΔE | sm·ΔE | bw·0 | sm·0 | mlp | bw·tok | sm·tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_dół | comp | 0.134 | 0.121 | 0.439 | 0.488 | 0.601 | 0.430 | 0.432 |
| A_dół | comp+rel·PoE | 0.236 | 0.207 | 0.479 | 0.471 | 0.726 | 0.489 | 0.481 |
| A_dół | comp+mix | 0.233 | 0.205 | 0.494 | 0.536 | 0.704 | 0.445 | 0.445 |
| B_te_hop | comp | 0.147 | 0.127 | 0.292 | 0.318 | 0.359 | 0.248 | 0.250 |
| B_te_hop | comp+rel·PoE | 0.199 | 0.173 | 0.319 | 0.318 | 0.452 | 0.312 | 0.308 |
| B_te_hop | comp+mix | 0.188 | 0.152 | 0.296 | 0.320 | 0.451 | 0.287 | 0.288 |
| B_te_dd | comp | 0.067 | 0.045 | 0.337 | 0.352 | 0.233 | 0.154 | 0.155 |
| B_te_dd | comp+rel·PoE | 0.171 | 0.175 | 0.396 | 0.394 | 0.304 | 0.262 | 0.272 |
| B_te_dd | comp+mix | 0.158 | 0.159 | 0.384 | 0.424 | 0.366 | 0.250 | 0.246 |
| B_tr_op | comp | 0.608 | 0.628 | 0.556 | 0.604 | 0.804 | 0.715 | 0.718 |
| B_tr_op | comp+rel·PoE | 0.639 | 0.661 | 0.590 | 0.620 | 0.819 | 0.789 | 0.793 |
| B_tr_op | comp+mix | 0.681 | 0.690 | 0.553 | 0.598 | 0.913 | 0.913 | 0.912 |

## Testy zadeklarowane

| test | kontrast | miara | Δ | 95% CI | 90% CI | wygr./przegr. | p_Holm | utrzymuje się? | ±5? |
|---|---|---|---:|---|---|---:|---:|:-:|:-:|
| N1 | bw·ΔE | A_dół | +0.103 | [+0.063, +0.143] | [+0.070, +0.138] | 16/1 | 0.0003 | tak | nie |
| N1 | sm·ΔE | A_dół | +0.086 | [+0.054, +0.121] | [+0.059, +0.116] | 17/1 | 0.0006 | tak | nie |
| N1 | bw·0 | A_dół | +0.040 | [+0.012, +0.068] | [+0.017, +0.064] | 12/3 | 0.0723 | nie | nie |
| N1 | sm·0 | A_dół | -0.016 | [-0.043, +0.011] | [-0.039, +0.007] | 6/9 | 1.0000 | nie | tak |
| N1 | mlp | A_dół | +0.125 | [+0.081, +0.172] | [+0.088, +0.164] | 16/0 | 0.0002 | tak | nie |
| N2 | bw·tok − sm·tok | A_dół | -0.002 | [-0.019, +0.014] | [-0.016, +0.012] | 6/9 | 1.0000 | nie | tak |
| N3 | sm·tok − sm·ΔE | A_dół | +0.311 | [+0.268, +0.363] | [+0.274, +0.354] | 20/0 | 0.0000 | tak | nie |
| N2 | bw·tok − sm·tok | B_te_hop | -0.002 | [-0.013, +0.009] | [-0.011, +0.007] | 8/10 | 1.0000 | nie | tak |
| N3 | sm·tok − sm·ΔE | B_te_hop | +0.123 | [+0.096, +0.157] | [+0.100, +0.151] | 20/0 | 0.0000 | tak | nie |
| N2 | bw·tok − sm·tok | B_te_dd | -0.000 | [-0.011, +0.010] | [-0.009, +0.009] | 6/6 | 1.0000 | nie | tak |
| N3 | sm·tok − sm·ΔE | B_te_dd | +0.110 | [+0.078, +0.144] | [+0.083, +0.138] | 18/0 | 0.0001 | tak | nie |

## T3 — czas i stabilność

| ramię | suma czasu | mediana zadania |
|---|---:|---:|
| bw·ΔE | 243 s | 4.10 s |
| sm·ΔE | 224 s | 3.82 s |
| bw·0 | 236 s | 3.99 s |
| sm·0 | 208 s | 3.58 s |
| mlp | 88 s | 1.30 s |
| bw·tok | 238 s | 4.07 s |
| sm·tok | 210 s | 3.63 s |

Suma czasu Most / Softmax (ramiona bw* wobec sm*, bez mlp) = 717 s / 642 s = 1.12×.

Każdy przebieg ma skończoną stratę i ostatni krok niższy niż pierwszy.

## Wniosek

N2 i N3 wychodzą tak, jak program zapisał. N1 nie powtarza Słońca na wszystkich ramionach. T3 jest spełnione.

1. **Przy notacji w tokenie jądra są zamienne.** `bw·tok − sm·tok`: Δ = −0,002, −0,002 i 0,000. Żadna różnica nie przeżywa korekty. Wszystkie trzy 90% CI mieszczą się w ±5 punktach, i w ±2 też.

2. **Token bije karę w wykładniku.** `sm·tok − sm·ΔE`: +0,31 na kroku pisowni, +0,12 na hopie, +0,11 na dwóch krokach. Wszystkie trzy przeżywają korektę, 18–20 wygranych na 20. Kara −ΔE²/γ² zrzuca krok z 0,49 (Softmax bez tracera) do 0,12. Token trzyma go przy 0,43.

3. **Osobna relacja pisowni pomaga tylko tam, gdzie model tę pisownię już dostaje.** PoE − comp na kroku, po Holmie: Most z ΔE +0,10, Softmax z ΔE +0,09, MLP +0,13. Most i Softmax bez tracera nie przeżywają korekty (Δ = +0,04 i −0,02). To nie jest wynik Słońca, gdzie relacja dokładała punkty we wszystkich pięciu ramionach. Widmo pisowni nie niesie (1-NN 0%). Krok „ta sama pisownia, oktawa wyżej” jest zarazem krokiem o dwanaście półtonów, a bliźniak enharmoniczny ma ten sam wektor, więc remis w rankingu nie karze złej pisowni. Ramiona bez tracera mogą iść po oktawie i nie rozstrzygać pisowni.

4. **T3.** 420 przebiegów, każdy ze skończoną stratą, ostatni krok niższy niż pierwszy. Most trwa 1,12 raza dłużej niż Softmax (717 s wobec 642 s). Wolniejsze jądro nie jest porażką.

Wmieszanie relacji w tę samą przestrzeń tu nie obniża kroku. Najwyższy krok daje MLP z PoE (0,73). Notacja działa jako plugin w tokenie albo w MLP, nie jako ułamek Breita–Wignera.

