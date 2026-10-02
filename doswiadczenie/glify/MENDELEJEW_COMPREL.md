# Mendelejew COMPREL — relacje kompozycyjne + nauka relacyjna

Cechy jak ZC (tylko Z + konfiguracja); operatory liniowe „dół”/„prawo” jak COMP; głowice „grupa” (K=all par) i „Z” jak REL. Wszystkie fakty i pary tylko spośród pierwiastków treningowych. Seedy=[0, 1, 2, 3, 4] · kroki=800 · czas=213s · skrypt `archiwum/skrypty/run_mendelejew_comprel.py`

`comp+rel·PoE` = ten sam model co `comp+rel`, ale dla łańcuchów samych „dół” wynik = cos(operator) + cos(głowica grupy). Dla łańcuchów z „prawo” PoE = sam operator (cel nie jest w tej samej grupie).

## A. Jeden krok z pierwiastkiem testowym — hit@1

Kolumny: wariant × arm.

| test | comp | comp+rel | comp+rel·PoE | comp+mix |
|---|---:|---:|---:|---:|
| most · dół | 0.325 ± 0.112 | 0.414 ± 0.108 | 0.416 ± 0.114 | 0.478 ± 0.063 |
| most · prawo | 0.553 ± 0.045 | 0.607 ± 0.044 | 0.607 ± 0.044 | 0.532 ± 0.077 |
| softmax · dół | 0.253 ± 0.107 | 0.300 ± 0.195 | 0.343 ± 0.181 | 0.287 ± 0.145 |
| softmax · prawo | 0.607 ± 0.055 | 0.614 ± 0.052 | 0.614 ± 0.052 | 0.539 ± 0.065 |
| zero · dół | 0.248 ± 0.127 | 0.373 ± 0.156 | 0.366 ± 0.159 | 0.263 ± 0.140 |
| zero · prawo | 0.605 ± 0.081 | 0.627 ± 0.064 | 0.627 ± 0.064 | 0.533 ± 0.075 |
| mlp · dół | 0.338 ± 0.138 | 0.397 ± 0.159 | 0.476 ± 0.110 | 0.390 ± 0.090 |
| mlp · prawo | 0.378 ± 0.097 | 0.389 ± 0.057 | 0.389 ± 0.057 | 0.332 ± 0.038 |

## B. Złożenia nieuczone — hit@1

Kolumny: wariant × arm.

| test | comp | comp+rel | comp+rel·PoE | comp+mix |
|---|---:|---:|---:|---:|
| most · treningowa · op | 0.742 ± 0.059 | 0.755 ± 0.058 | 0.739 ± 0.061 | 0.745 ± 0.045 |
| most · treningowa · hop | 0.931 ± 0.060 | 0.969 ± 0.042 | 0.959 ± 0.039 | 0.980 ± 0.014 |
| most · treningowa · dół∘dół | 0.882 ± 0.041 | 0.935 ± 0.070 | 0.874 ± 0.107 | 0.825 ± 0.055 |
| most · dotyka testu · op | 0.269 ± 0.051 | 0.291 ± 0.063 | 0.310 ± 0.081 | 0.308 ± 0.053 |
| most · dotyka testu · hop | 0.341 ± 0.087 | 0.384 ± 0.094 | 0.390 ± 0.083 | 0.440 ± 0.052 |
| most · dotyka testu · dół∘dół | 0.233 ± 0.083 | 0.318 ± 0.112 | 0.389 ± 0.117 | 0.306 ± 0.109 |
| softmax · treningowa · op | 0.597 ± 0.084 | 0.528 ± 0.146 | 0.550 ± 0.139 | 0.453 ± 0.101 |
| softmax · treningowa · hop | 0.592 ± 0.141 | 0.675 ± 0.113 | 0.637 ± 0.091 | 0.690 ± 0.042 |
| softmax · treningowa · dół∘dół | 0.585 ± 0.224 | 0.610 ± 0.191 | 0.695 ± 0.171 | 0.499 ± 0.116 |
| softmax · dotyka testu · op | 0.251 ± 0.083 | 0.282 ± 0.093 | 0.297 ± 0.085 | 0.233 ± 0.073 |
| softmax · dotyka testu · hop | 0.253 ± 0.076 | 0.292 ± 0.132 | 0.322 ± 0.118 | 0.287 ± 0.102 |
| softmax · dotyka testu · dół∘dół | 0.241 ± 0.126 | 0.296 ± 0.114 | 0.352 ± 0.086 | 0.258 ± 0.106 |
| zero · treningowa · op | 0.523 ± 0.073 | 0.551 ± 0.102 | 0.559 ± 0.095 | 0.429 ± 0.047 |
| zero · treningowa · hop | 0.638 ± 0.110 | 0.664 ± 0.070 | 0.682 ± 0.075 | 0.584 ± 0.080 |
| zero · treningowa · dół∘dół | 0.541 ± 0.077 | 0.645 ± 0.052 | 0.674 ± 0.066 | 0.544 ± 0.128 |
| zero · dotyka testu · op | 0.257 ± 0.043 | 0.298 ± 0.087 | 0.327 ± 0.106 | 0.246 ± 0.061 |
| zero · dotyka testu · hop | 0.254 ± 0.080 | 0.325 ± 0.119 | 0.339 ± 0.113 | 0.261 ± 0.119 |
| zero · dotyka testu · dół∘dół | 0.241 ± 0.089 | 0.257 ± 0.081 | 0.362 ± 0.113 | 0.226 ± 0.059 |
| mlp · treningowa · op | 0.538 ± 0.096 | 0.543 ± 0.114 | 0.554 ± 0.124 | 0.476 ± 0.041 |
| mlp · treningowa · hop | 0.751 ± 0.066 | 0.746 ± 0.018 | 0.762 ± 0.021 | 0.683 ± 0.090 |
| mlp · treningowa · dół∘dół | 0.561 ± 0.092 | 0.570 ± 0.139 | 0.617 ± 0.131 | 0.586 ± 0.093 |
| mlp · dotyka testu · op | 0.227 ± 0.071 | 0.241 ± 0.086 | 0.270 ± 0.074 | 0.194 ± 0.069 |
| mlp · dotyka testu · hop | 0.289 ± 0.093 | 0.301 ± 0.148 | 0.338 ± 0.110 | 0.282 ± 0.063 |
| mlp · dotyka testu · dół∘dół | 0.288 ± 0.100 | 0.299 ± 0.145 | 0.403 ± 0.117 | 0.174 ± 0.117 |

## Efekt połączenia dla Mostu (Δ względem `comp`, wygrane seedy)

| test | comp+rel | comp+rel·PoE | comp+mix |
|---|---:|---:|---:|
| A_dół | +0.089 (4/5) | +0.091 (4/5) | +0.154 (5/5) |
| A_prawo | +0.054 (5/5) | +0.054 (5/5) | -0.021 (1/5) |
| B_tr_op | +0.013 (3/5) | -0.003 (3/5) | +0.003 (2/5) |
| B_te_op | +0.022 (2/5) | +0.041 (3/5) | +0.040 (5/5) |
| B_te_hop | +0.042 (3/5) | +0.048 (3/5) | +0.099 (4/5) |
| B_te_dd | +0.085 (4/5) | +0.156 (5/5) | +0.074 (5/5) |

## Most vs reszta w `comp+rel·PoE` (wygrane seedy)

| test | vs softmax | vs zero | vs mlp |
|---|---:|---:|---:|
| A_dół | 2/5 | 4/5 | 1/5 |
| B_tr_op | 4/5 | 5/5 | 5/5 |
| B_te_op | 2/5 | 2/5 | 3/5 |
| B_te_hop | 4/5 | 4/5 | 3/5 |

## Pytania Mendelejewa (Most) — średnia pozycja (trafień 1.) · top-3 seed 0

| pytanie | comp | comp+rel | comp+rel·PoE | comp+mix |
|---|---|---|---|---|
| dół(Al)=Ga | 33.2 (0) · Si, B, He | 25.0 (0) · Si, B, Be | 20.4 (0) · Si, B, Nh | 31.0 (0) · Si, B, Be |
| dół(Si)=Ge | 23.0 (1) · P, Al, N | 14.2 (1) · Al, P, B | 10.2 (0) · Al, B, P | 24.6 (0) · Al, B, C |
| dół(Mn)=Tc | 9.4 (0) · Mo, Ru, Nb | 7.2 (0) · Mo, Ru, Nb | 14.8 (0) · Mo, Ru, Re | 5.2 (0) · Ru, Mo, Tc |
| dół∘dół(B)=Ga | 35.8 (0) · Al, Re, Si | 26.4 (0) · Al, Si, Fl | 22.0 (0) · Al, Si, Nh | 34.6 (0) · Al, Ba, Si |
| dół∘dół(C)=Ge | 25.6 (1) · N, Al, Si | 17.4 (0) · Si, Al, P | 11.6 (0) · Si, Fl, Pb | 25.4 (0) · Ba, Cs, Al |
| prawo(Zn)=Ga | 80.6 (0) · Cu, Cd, Ni | 39.6 (0) · Cu, Cd, Nh | 39.6 (0) · Cu, Cd, Nh | 24.4 (0) · Cd, Hg, Cu |
| prawo(Ga)=Ge | 1.0 (5) · Ge, As, Sn | 1.0 (5) · Ge, As, In | 1.0 (5) · Ge, As, In | 1.4 (4) · Ge, Mn, As |
| prawo(Ca)=Sc | 1.0 (5) · Sc, Ti, Y | 1.0 (5) · Sc, Ti, Y | 1.0 (5) · Sc, Ti, Y | 1.6 (4) · Sr, Ra, No |
| prawo∘dół(Mg)=Sc | 7.6 (0) · K, Ar, Ca | 4.2 (0) · K, Ca, Ar | 4.2 (0) · K, Ca, Ar | 3.0 (2) · Ca, K, Sc |
| dół∘prawo(Al)=Ge | 31.0 (0) · Pb, C, Os | 19.8 (0) · Si, B, C | 19.8 (0) · Si, B, C | 33.8 (1) · Si, B, Ba |

## Wnioski

**Zastrzeżenia:** 5 seedów. Zapytań „dół” z pierwiastkiem testowym jest ~20 na seed (1 trafienie ≈ 5 pkt). Liczę się tylko ze stałym znakiem efektu. `comp` odtwarza wynik z COMP (A_dół 0.32 vs 0.33), co jest kontrolą spójności.

**Uwaga o nazwie `comp+mix`:** tutaj w przestrzeni operatorów jest **wyłącznie** relacja grupy. W REL „mix” niszczyło grupę, bo zlewało ją z relacjami **poziomymi** (Z, okres). Sama grupa we wspólnej przestrzeni odpowiada REL `single`, które działało. Oba wyniki są spójne.

1. **Połączenie poprawia wąskie gardło, czyli pierwszy krok na nowym pierwiastku.** Most, hit@1:

   | test | comp | comp+rel | comp+rel·PoE | comp+mix |
   |---|---:|---:|---:|---:|
   | dół z pierwiastkiem testowym | 0.32 | 0.41 (4/5) | 0.42 (4/5) | **0.48** (5/5) |
   | złożenie dotyka testu · 2 kroki | 0.34 | 0.38 | 0.39 | **0.44** (4/5) |
   | dół∘dół dotyka testu | 0.23 | 0.32 (4/5) | **0.39** (5/5) | 0.31 (5/5) |
   | złożenia na znanych | 0.74 | 0.76 | 0.74 | 0.75 |

   Kompozycja na znanych pierwiastkach pozostaje bez zmian (~0.75). Zysk jest dokładnie tam, gdzie COMP był słaby: indukcja „w dół” rośnie o +0.09 do +0.15.
2. **Dwa sposoby łączenia działają w różnych miejscach.**
   - **Grupa we wspólnej przestrzeni (`comp+mix`):** najlepszy pojedynczy krok i hop. Most wygrywa tu **5/5 z Softmax, ZERO i MLP** zarówno w „dół z testem” (0.48 vs 0.29 / 0.26 / 0.39), jak i w hop (0.44 vs 0.29 / 0.26 / 0.28). To najmocniejszy wynik Mostu dla indukcji w całej serii.
   - **Iloczyn ekspertów (`PoE`):** najlepszy dla czystych łańcuchów „dół” (dół∘dół z testem: 0.39, +0.16, 5/5). Operator wskazuje okres, głowica grupy kolumnę, a ich suma zawęża wybór. Przewaga Mostu nad innymi jest tu jednak mała (0.39 vs 0.35–0.40). PoE wyrównuje modele, bo głowica grupy jest „mocna” w każdym.
3. **Osobne głowice (`comp+rel`) pomagają też „w prawo”** (0.55 → 0.61, 5/5). Głowica Z porządkuje sąsiedztwo w Z. `comp+mix` tego nie daje (0.53).
4. **Pytania Mendelejewa: pozycje wyraźnie lepsze, ale wciąż rzadko pierwsze miejsce.** Średnia pozycja z 5 seedów, losowo ≈ 59:

   | pytanie | comp | najlepszy wariant |
   |---|---:|---|
   | pod Al → Ga | 33 | 20 (PoE; top: Si, B, **Nh** — Nh to grupa 13) |
   | pod Si → Ge | 23 | 10 (PoE) |
   | dół∘dół(C) → Ge | 26 | 12 (PoE; top: Si, Fl, **Pb** — grupa 14) |
   | pod Mn → Tc | 9.4 | **5.2** (mix; top-3 seed 0: Ru, Mo, **Tc**) |
   | prawo(Zn) → Ga | 81 | 24 (mix) |
   | prawo∘dół(Mg) → Sc | 7.6 | **3.0, 2/5 trafień** (mix; top: Ca, K, **Sc**) |

   Kandydaci przesunęli się z „ten sam okres” na „ta sama kolumna”: pod Al pojawia się Nh, pod C Pb. Sam brakujący pierwiastek trafia jednak na 1. miejsce tylko w pojedynczych seedach.

## Werdykt

Połączenie relacji kompozycyjnych z relacją grupy **poprawia indukcję na nowe pierwiastki** o +0.09 do +0.16 (4–5/5 seedów), nie psując kompozycji na znanych (~0.75). W wariancie z grupą we wspólnej przestrzeni Most jest najlepszy **5/5 wobec wszystkich modeli** zarówno w pojedynczym kroku „w dół”, jak i w dwóch krokach przez nieznane pierwiastki. Luki Mendelejewa są coraz bliżej (Tc 5., Sc 3., Ge 10. miejsce z 117), ale model wciąż wskazuje raczej „sąsiedztwo” niż sam pierwiastek.

**Kierunki:**
1. Połączyć oba warianty: grupę we wspólnej przestrzeni i PoE z osobną głowicą, bo pomagają w różnych testach.
2. Spójność cyklu (operator „góra” z warunkiem M_g M_d ≈ I) jako regularyzacja indukcji.
3. 10–20 seedów, bo zapytań „dół” z testem jest ~20 na seed.

