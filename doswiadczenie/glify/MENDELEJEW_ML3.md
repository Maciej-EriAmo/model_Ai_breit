# Mendelejew ML3 — nadzorowane · nienadzorowane · ze wzmocnieniem

Set: `archiwum/dane/struktura_atomow.csv` · pierwiastki=['H', 'He', 'Li', 'C', 'N', 'O', 'Na', 'Fe'] · węzły Φ=72 (8 jąder + 64 elektronów) · krawędzie bind=137
Seedy=[0, 1, 2, 3, 4] · kroki SL=400 UL=400 RL=600 · czas=184s · skrypt `archiwum/skrypty/run_mendelejew_ml3.py`

Armie: **most** (Most-1 Breita–Wignera) · **softmax** (twin) · **zero** (Most, tracer=0) · **mlp** (per-węzeł, bez uwagi). Wartości: średnia ± odch. std. po seedach.

## 1. Nadzorowane — klasyfikacja węzłów (test = 30% wstrzymanych węzłów)

| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| element | 8 | 0.364 | 0.227 ± 0.032 | 0.255 ± 0.083 | 0.245 ± 0.052 | 0.382 ± 0.123 | 1/5 |
| n | 4 | 0.526 | 0.495 ± 0.080 | 0.379 ± 0.086 | 0.432 ± 0.069 | 0.916 ± 0.047 | 3/5 |
| l | 3 | 0.474 | 0.379 ± 0.151 | 0.389 ± 0.109 | 0.347 ± 0.060 | 0.663 ± 0.088 | 3/5 |

Macro-F1 (test):

| zadanie | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| element | 0.121 ± 0.033 | 0.156 ± 0.061 | 0.143 ± 0.054 | 0.214 ± 0.067 |
| n | 0.294 ± 0.082 | 0.234 ± 0.058 | 0.259 ± 0.064 | 0.697 ± 0.040 |
| l | 0.367 ± 0.106 | 0.306 ± 0.088 | 0.305 ± 0.112 | 0.535 ± 0.137 |

Train acc (czy model w ogóle dopasował zbiór uczący):

| zadanie | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| element | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.964 ± 0.050 |
| n | 1.000 ± 0.000 | 0.996 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| l | 1.000 ± 0.000 | 0.996 ± 0.010 | 1.000 ± 0.000 | 0.996 ± 0.010 |

## 2. Nienadzorowane — k-means (k = liczba klas) → NMI

### 2a. Surowe cechy (bez uczenia)

| cechy | NMI element | NMI n | NMI ℓ | ARI element | ARI n | ARI ℓ |
|---|---:|---:|---:|---:|---:|---:|
| content | 0.195 ± 0.025 | 0.044 ± 0.006 | 0.049 ± 0.023 | -0.007 ± 0.022 | -0.009 ± 0.007 | -0.001 ± 0.017 |
| tracer | 0.218 ± 0.007 | 0.846 ± 0.036 | 0.328 ± 0.000 | -0.008 ± 0.025 | 0.734 ± 0.112 | 0.221 ± 0.000 |
| concat | 0.278 ± 0.016 | 0.825 ± 0.001 | 0.328 ± 0.000 | 0.085 ± 0.017 | 0.667 ± 0.005 | 0.221 ± 0.000 |

### 2b. Embeddingi enkodera: losowa inicjalizacja vs po samonadzorze (InfoNCE na krawędziach bind)

| arm | stan | NMI element | NMI n | NMI ℓ | ARI element | ARI n | ARI ℓ |
|---|---|---:|---:|---:|---:|---:|---:|
| most | untrained | 0.197 ± 0.033 | 0.077 ± 0.019 | 0.041 ± 0.013 | 0.000 ± 0.024 | 0.003 ± 0.022 | -0.006 ± 0.019 |
| most | ssl | 0.641 ± 0.033 | 0.444 ± 0.040 | 0.148 ± 0.025 | 0.398 ± 0.039 | 0.307 ± 0.041 | 0.056 ± 0.029 |
| softmax | untrained | 0.210 ± 0.027 | 0.063 ± 0.015 | 0.068 ± 0.032 | 0.013 ± 0.021 | 0.015 ± 0.009 | 0.002 ± 0.025 |
| softmax | ssl | 0.651 ± 0.049 | 0.374 ± 0.021 | 0.123 ± 0.004 | 0.414 ± 0.089 | 0.255 ± 0.022 | 0.029 ± 0.006 |
| zero | untrained | 0.198 ± 0.026 | 0.078 ± 0.027 | 0.046 ± 0.020 | -0.003 ± 0.017 | 0.005 ± 0.024 | -0.003 ± 0.020 |
| zero | ssl | 0.645 ± 0.029 | 0.361 ± 0.052 | 0.125 ± 0.015 | 0.406 ± 0.064 | 0.235 ± 0.033 | 0.033 ± 0.019 |
| mlp | untrained | 0.271 ± 0.016 | 0.635 ± 0.110 | 0.254 ± 0.089 | 0.068 ± 0.037 | 0.500 ± 0.140 | 0.161 ± 0.079 |
| mlp | ssl | 0.459 ± 0.060 | 0.663 ± 0.015 | 0.300 ± 0.024 | 0.208 ± 0.089 | 0.555 ± 0.018 | 0.187 ± 0.025 |

## 3. Ze wzmocnieniem — kontekstowy bandyta (REINFORCE)

Stan = węzeł-zapytanie; akcja = wybór jednego z pozostałych węzłów; nagroda 1, gdy to partner bind.
Polityka = softmax(cos(h_q, h_a)/0.1) z enkodera. Ocena: greedy hit@1 na 30% zapytań wstrzymanych.
`sl_ref` = ten sam enkoder uczony nadzorem (CE po zbiorze partnerów) — górna referencja.

> **Uwaga (2026-10-02):** RL i SL-ref są tu **transduktywne** — pozytywy/nagrody obejmują też węzły testowe (krawędzie grafu jako dane wejściowe), więc „test hit@1” to predykcja połączeń przy znanym grafie, nie uogólnienie na nowe węzły. W wariancie ZC, gdzie partnerzy = etykiety grupy, to przeciek — zob. korektę w `MENDELEJEW_ML3_ZC.md` pkt 5.

| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |
|---|---:|---:|---:|---:|
| most | 0.309 ± 0.099 | 0.700 ± 0.060 | 0.555 ± 0.075 | 0.049 |
| softmax | 0.255 ± 0.127 | 0.596 ± 0.241 | 0.573 ± 0.083 | 0.049 |
| zero | 0.255 ± 0.149 | 0.684 ± 0.093 | 0.527 ± 0.069 | 0.049 |
| mlp | 0.255 ± 0.076 | 0.612 ± 0.033 | 0.518 ± 0.105 | 0.049 |

Most > Softmax (RL test) na 3/5 seedach.

Krzywa nagrody treningowej (seed 0, 10 okien):

| arm | nagroda |
|---|---|
| most | 0.05 → 0.07 → 0.07 → 0.17 → 0.24 → 0.28 → 0.31 → 0.37 → 0.40 → 0.46 |
| softmax | 0.05 → 0.08 → 0.08 → 0.12 → 0.08 → 0.15 → 0.26 → 0.32 → 0.39 → 0.44 |
| zero | 0.05 → 0.07 → 0.07 → 0.11 → 0.13 → 0.16 → 0.24 → 0.33 → 0.44 → 0.45 |
| mlp | 0.13 → 0.20 → 0.27 → 0.33 → 0.39 → 0.40 → 0.44 → 0.44 → 0.45 → 0.47 |

## Wnioski

**Zastrzeżenie skali:** 8 pierwiastków → 72 węzły; testy SL mają 19–22 węzły, RL 22 zapytania. Jeden węzeł ≈ 5 pkt. Fe daje 27/72 węzłów, więc klasa większościowa jest mocna. Różnice < ~0.1 przy 5 seedach to szum, chyba że znak jest stały we wszystkich seedach.

1. **Nadzorowane — enkodery z uwagą zapamiętują, ale nie uogólniają.** Most, Softmax i ZERO mają train acc = 1.00, a test acc **poniżej klasy większościowej** we wszystkich trzech zadaniach. MLP bez uwagi wygrywa: n 0.92, ℓ 0.66. Przyczyna jest architektoniczna, nie wynika z treningu. W kanonie tracer (T, mazur) **nie wchodzi do treści tokenu**, tylko do wagi uwagi (ΔE). Content to hash napisu S/E, więc niesie tożsamość, a nie fizykę. Węzeł nie „widzi” własnego n/ℓ, a hash uczy się na pamięć. MLP dostaje tracer wprost jako cechę. Element nie daje się przewidzieć nikomu: MLP 0.38 ≈ większościowa 0.36, bo informacja o Z jest w cechach tylko szczątkowa (`0.3·(Z mod 17)/17` w mazur).

2. **Nienadzorowane — struktura grafu uczy elementów, a tracer uczy powłok.**
   - Surowy tracer sam grupuje powłoki: NMI n = 0.85. Content to szum (NMI ≈ 0.04).
   - Samonadzór InfoNCE na krawędziach bind podnosi NMI(element) z ~0.20 do **~0.64** we wszystkich trzech enkoderach z uwagą (Most≈SM≈ZERO). To efekt struktury grafu, nie jądra.
   - **Jedyny powtarzalny efekt Mostu:** NMI(n) po samonadzorze wynosi 0.444, wobec 0.374 dla Softmax (**5/5 seedów**, średnio +0.07) i 0.361 dla ZERO (**5/5**, +0.08). ΔE z tracera przenosi informację o powłoce do embeddingu, choć tylko pośrednio, przez to, na kogo węzeł patrzy. Efekt jest jednak dużo słabszy niż podanie tracera wprost (MLP 0.66, surowy tracer 0.85).

3. **Ze wzmocnieniem — REINFORCE działa, ale kosztuje ~połowę skuteczności.**
   - Wszystkie armie uczą się z samej nagrody: test hit@1 0.26–0.31, wobec losowej polityki 0.05, czyli 5–6×.
   - Ten sam enkoder uczony nadzorem (SL-ref) osiąga ~0.52–0.57. Sygnał skalarny daje więc mniej więcej połowę tego wyniku.
   - Duża luka train/test (0.6–0.7 vs 0.3): przy 50 zapytaniach treningowych polityka przeucza się.
   - Most ma najwyższą średnią w RL (0.31 vs 0.26), ale tylko 3/5 seedów i std ≈ 0.1, więc **nie jest to wynik**.
   - MLP najszybciej rośnie w nagrodzie (krzywa), bo od pierwszego kroku ma tracer w cechach.

4. **Werdykt dla Mostu na małym secie:** parity z Softmax w SL i RL (zgodnie z Mendelejew P0: Most≈SM≈ZERO). Mała, ale stabilna przewaga jest tylko w nienadzorowanym grupowaniu powłok. Najsilniejszym czynnikiem we wszystkich trzech paradygmatach jest **dostęp węzła do własnego tracera**, a nie typ jądra uwagi.

**Następny krok (do decyzji, zmienia kanon):** arm „Most + τ w treści” (`x ⊕ tracer` w embed). Rozstrzygnie, czy Most coś dodaje, gdy fizyka jest dostępna także w treści. Druga opcja: powtórzyć na `struktura_atomow_full.csv` (118 pierwiastków), żeby zejść z poziomu szumu.

