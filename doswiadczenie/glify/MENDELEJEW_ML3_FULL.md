# Mendelejew ML3 FULL — pełna tablica (118) + szukanie nowego pierwiastka

Graf FULL118 (jądra; bind: Z-sąsiedzi / grupa / okres) · krawędzie bind=189 · seedy=[0, 1, 2, 3, 4] · kroki SL=400 UL=400 RL=600 · czas=330s · skrypt `archiwum/skrypty/run_mendelejew_ml3_full.py`

Tracer węzła = (głębokość z okresu, T ~ okres, mazur = grupa/18); treść = hash napisu (tożsamość, nie fizyka).

## A1. Nadzorowane — klasyfikacja węzłów (test = 30% wstrzymanych pierwiastków)

| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| period | 7 | 0.286 | 0.246 ± 0.043 | 0.211 ± 0.063 | 0.211 ± 0.094 | 0.794 ± 0.062 | 4/5 |
| group | 18 | 0.286 | 0.200 ± 0.020 | 0.131 ± 0.059 | 0.149 ± 0.055 | 0.286 ± 0.035 | 5/5 |
| block | 4 | 0.314 | 0.360 ± 0.147 | 0.286 ± 0.078 | 0.291 ± 0.042 | 0.623 ± 0.068 | 4/5 |

Train acc:

| zadanie | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| period | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| group | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| block | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## A2. Nienadzorowane — k-means → NMI

| cechy / arm | stan | NMI okres | NMI grupa | NMI blok |
|---|---|---:|---:|---:|
| surowe content | — | 0.103 ± 0.018 | 0.400 ± 0.009 | 0.034 ± 0.019 |
| surowe tracer | — | 0.560 ± 0.001 | 0.575 ± 0.005 | 0.303 ± 0.000 |
| surowe concat | — | 0.552 ± 0.006 | 0.410 ± 0.009 | 0.150 ± 0.009 |
| most | untrained | 0.089 ± 0.018 | 0.393 ± 0.011 | 0.042 ± 0.016 |
| most | ssl | 0.333 ± 0.053 | 0.597 ± 0.039 | 0.480 ± 0.094 |
| softmax | untrained | 0.084 ± 0.013 | 0.390 ± 0.016 | 0.039 ± 0.019 |
| softmax | ssl | 0.240 ± 0.034 | 0.570 ± 0.028 | 0.273 ± 0.050 |
| zero | untrained | 0.091 ± 0.019 | 0.394 ± 0.012 | 0.042 ± 0.016 |
| zero | ssl | 0.256 ± 0.073 | 0.548 ± 0.026 | 0.274 ± 0.085 |
| mlp | untrained | 0.506 ± 0.038 | 0.398 ± 0.023 | 0.119 ± 0.038 |
| mlp | ssl | 0.648 ± 0.031 | 0.502 ± 0.010 | 0.388 ± 0.081 |

## A3. Ze wzmocnieniem — bandyta REINFORCE (partner bind)

> **Uwaga (2026-10-02):** RL i SL-ref są tu **transduktywne** — pozytywy/nagrody obejmują też węzły testowe (krawędzie grafu jako dane wejściowe), więc „test hit@1” to predykcja połączeń przy znanym grafie, nie uogólnienie na nowe węzły. W wariancie ZC, gdzie partnerzy = etykiety grupy, to przeciek — zob. korektę w `MENDELEJEW_ML3_ZC.md` pkt 5.

| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |
|---|---:|---:|---:|---:|
| most | 0.000 ± 0.000 | 0.027 ± 0.016 | 0.446 ± 0.104 | 0.028 |
| softmax | 0.029 ± 0.029 | 0.024 ± 0.019 | 0.440 ± 0.122 | 0.028 |
| zero | 0.011 ± 0.016 | 0.031 ± 0.039 | 0.429 ± 0.057 | 0.028 |
| mlp | 0.309 ± 0.024 | 0.535 ± 0.079 | 0.491 ± 0.142 | 0.028 |

Most > Softmax (RL test) na 0/5 seedach.

## B. Szukanie nowego pierwiastka (nienadzorowane)

Usunięte z danych: **Sc** (eka-bor, g3/p4), **Ga** (eka-glin, g13/p4), **Ge** (eka-krzem, g14/p4), **Tc** (eka-mangan, g7/p5).
Kandydaci: 40 wolnych komórek siatki 18×7 = 4 prawdziwe luki + 36 komórek nieistniejących.
Wynik komórki = średni cos sondy do 4 najbliższych znanych pierwiastków (gęstość). Idealnie: 4 luki na miejscach 1–4, AUC = 1. Losowo: średnia pozycja ≈ 20.5, AUC = 0.5.

| wynik | metoda | średnia pozycja luk | luki w top-4 | luki w top-8 | AUC |
|---|---|---:|---:|---:|---:|
| gęstość | reguła siatki (bez ML) | 2.5 | 4.0/4 | 4.0/4 | 1.000 |
| gęstość | surowe cechy kNN | 4.0 | 2.0/4 | 4.0/4 | 0.958 |
| gęstość | most | 19.1 | 0.4/4 | 1.0/4 | 0.540 ± 0.129 |
| gęstość | softmax | 17.3 | 0.8/4 | 1.4/4 | 0.589 ± 0.129 |
| gęstość | zero | 19.1 | 0.6/4 | 0.8/4 | 0.537 ± 0.162 |
| gęstość | mlp | 37.5 | 0.0/4 | 0.0/4 | 0.029 ± 0.029 |
| CSLS (post-hoc) | surowe cechy kNN | 8.0 | 1.0/4 | 3.0/4 | 0.847 |
| CSLS (post-hoc) | most | 18.9 | 0.4/4 | 0.8/4 | 0.546 ± 0.135 |
| CSLS (post-hoc) | softmax | 19.6 | 0.0/4 | 1.0/4 | 0.524 ± 0.135 |
| CSLS (post-hoc) | zero | 20.2 | 0.6/4 | 0.8/4 | 0.507 ± 0.155 |
| CSLS (post-hoc) | mlp | 28.8 | 0.0/4 | 0.4/4 | 0.269 ± 0.067 |

### Szczegóły luk (seed 0): pozycja, najbliżsi sąsiedzi, przewidziany blok

| luka | arm | pozycja (gęstość) | pozycja (CSLS) | 4-NN (znane) | blok przewid. | blok prawdz. |
|---|---|---:|---:|---|:-:|:-:|
| Sc | most | 7 | 8 | Sr, Y, Ba, Ca | s ✗ | d |
| Sc | softmax | 3 | 7 | Sr, Y, Cu, Pd | d ✓ | d |
| Sc | zero | 24 | 29 | Y, Ir, Ag, Pd | d ✓ | d |
| Sc | mlp | 39 | 23 | Y, Ca, V, Ti | d ✓ | d |
| Ga | most | 17 | 16 | Ag, P, As, Cu | d ✗ | p |
| Ga | softmax | 18 | 24 | Kr, Rb, Sr, Ce | s ✗ | p |
| Ga | zero | 16 | 11 | Pr, Ce, Nh, La | f ✗ | p |
| Ga | mlp | 38 | 39 | As, Se, Kr, Br | p ✓ | p |
| Ge | most | 19 | 18 | Sn, Pb, Sb, Bi | p ✓ | p |
| Ge | softmax | 2 | 5 | I, Te, Po, At | p ✓ | p |
| Ge | zero | 1 | 1 | Te, I, Br, Po | p ✓ | p |
| Ge | mlp | 37 | 30 | In, Sn, Br, Ag | p ✓ | p |
| Tc | most | 40 | 40 | N, Mn, Fe, Pr | d ✓ | d |
| Tc | softmax | 29 | 34 | Og, Ho, Rn, Lv | p ✗ | d |
| Tc | zero | 25 | 33 | Te, I, Sb, Xe | p ✗ | d |
| Tc | mlp | 40 | 40 | Rh, Cr, Fe, Ag | d ✓ | d |

### Top-8 komórek wg Mostu, wynik CSLS (seed 0)

| # | komórka | CSLS | 4-NN |
|---:|---|---:|---|
| 1 | g5/p2 | 0.917 | Li, Be, Na, Ca |
| 2 | g3/p1 | 0.908 | H, Li, Be, Na |
| 3 | g12/p1 | 0.902 | Co, Fe, Rh, Ru |
| 4 | g9/p1 | 0.894 | Be, Mg, Li, Na |
| 5 | g9/p2 | 0.875 | Mg, B, Be, Al |
| 6 | g12/p3 | 0.874 | C, N, Si, P |
| 7 | g2/p1 | 0.873 | Li, Na, K, Be |
| 8 | g3/p4 ← **Sc** | 0.864 | Sr, Y, Ba, Ca |

### Eksploracja: okres 8 (pierwiastki 119+), ranking CSLS wg Mostu i MLP (seed 0)

| # | most: komórka | 4-NN | mlp: komórka | 4-NN |
|---:|---|---|---|---|
| 1 | g14/p8 (0.907) | Mc, Nh, Fl, Lv | g16/p8 (0.976) | Ts, Lv, Cn, Og |
| 2 | g18/p8 (0.89) | Ts, Mc, Lv, Nh | g17/p8 (0.975) | Ts, Mc, Lv, Cn |
| 3 | g11/p8 (0.887) | Cn, Rg, Au, Nh | g12/p8 (0.965) | Cn, Nh, Fl, Rg |
| 4 | g3/p8 (0.876) | Pu, Pr, Ce, Np | g11/p8 (0.961) | Cn, Rg, Nh, Ds |
| 5 | g16/p8 (0.863) | Ts, Lv, At, Og | g14/p8 (0.956) | Mc, Lv, Nh, Cn |
| 6 | g17/p8 (0.804) | Fl, Nh, Bi, Pb | g3/p8 (0.951) | Cm, Pu, Db, Pa |

## Wnioski

**Zastrzeżenia:** test SL ma ~35 pierwiastków, RL ~35 zapytań, a odkrycie tylko 4 luki. Przy 5 seedach liczę się wyłącznie ze stałym znakiem efektu. Tracer w FULL118 koduje okres (T) i grupę (mazur = g/18) **wprost**, więc tablica jest „podana” w cechach. Pytanie nie brzmi, czy da się ją odtworzyć, tylko czy uczenie coś dodaje.

### A. Powtórka na 118 pierwiastkach

1. **Nadzorowane:**
   - Enkodery z uwagą znów uczą się na pamięć (train 1.00) i na teście nie przebijają klasy większościowej. Jedyny wyjątek to Most na bloku: 0.36 wobec 0.31.
   - Most jest konsekwentnie lepszy od Softmax: okres 4/5, grupa 5/5, blok 4/5. Działa to jednak w reżimie, w którym żaden enkoder z uwagą nie generalizuje.
   - MLP z tracerem w cechach: okres 0.79, blok 0.62.
   - Grupy (18 klas) nie przewiduje nikt. Grupa 3 obejmuje lantanowce i aktynowce, co daje klasę większościową 0.29.
2. **Nienadzorowane — najmocniejszy wynik Mostu we wszystkich testach.** Po samonadzorze Most bije Softmax i ZERO w grupowaniu wszystkich trzech osi:

   | oś | Most vs Softmax | Most vs ZERO | Most vs MLP |
   |---|---|---|---|
   | okres (NMI) | +0.09 (5/5) | +0.08 (4/5) | przegrywa (−0.32) |
   | grupa | +0.03 (4/5) | +0.05 (4/5) | **wygrywa** (+0.10, 5/5) |
   | blok | **+0.21 (5/5)** | **+0.21 (5/5)** | **wygrywa** (+0.09, 4/5) |

   Blok NMI 0.48 to najlepszy wynik spośród wszystkich metod, także surowego tracera (0.30). Bloku nie ma wprost w cechach: wynika z kombinacji grupy, okresu i krawędzi. Most składa go lepiej niż reszta. Na małym secie przewaga Mostu w nienadzorowanym była mała (+0.07 na powłokach). Na pełnej tablicy rośnie.
3. **Ze wzmocnieniem — enkodery z uwagą nie uczą się wcale.** Most, Softmax i ZERO mają hit@1 ≈ 0.00–0.03, czyli poziom losowy 0.03, nawet na zbiorze uczącym. Ten sam enkoder uczony nadzorem (SL-ref) osiąga ~0.44. Problemem jest więc rzadka nagroda przy 117 możliwych akcjach, a nie zdolność modelu. MLP uczy się (0.31), bo od startu ma strukturę tablicy w cechach. Na małym secie (72 węzły) RL działał we wszystkich modelach, więc skala akcji łamie REINFORCE dla uwagi.

### B. Szukanie nowego pierwiastka

4. **Nie odkrył żaden wyuczony model. Reguła Mendelejewa odkryła wszystkie.**
   - Reguła siatki: „ile sąsiednich komórek jest zajętych”, bez ML. Wszystkie 4 luki (Sc, Ga, Ge, Tc) na miejscach 1–4, AUC = 1.00.
   - Surowe cechy z kNN: AUC 0.96, wszystkie 4 luki w top-8.
   - Embeddingi po samonadzorze: Most 0.54, Softmax 0.59, ZERO 0.54, czyli poziom losowy. MLP jest wręcz odwrócony (0.03). Korekta CSLS (post-hoc) tego nie zmienia: Most 0.55, MLP 0.27.
   - Most ustawia najwyżej komórki, które nie istnieją (g5/p2, g3/p1), bo wpadają w ciasny klaster s-bloku z okresów 1–2.
5. **„Jak wyglądałby” — działa częściowo; „gdzie jest luka” — nie działa.** Gdy wskaże się komórkę luki, sąsiedzi sondy bywają chemicznie trafni. Przykład MLP: Sc → Y, Ca, V, Ti; Ga → As, Se, Kr, Br; Tc → Rh, Cr, Fe, Ag. Blok z głosowania sąsiadów: MLP 4/4, Most / Softmax / ZERO po 2/4. Wyuczona przestrzeń zachowuje więc **podobieństwo**, ale traci **informację o tym, czego brakuje**.
6. **Dlaczego uczenie przeszkadza:**
   - (a) InfoNCE uczy tylko „przyciągaj istniejących sąsiadów”. Nigdy nie widzi pozycji nieistniejących, więc nie ma powodu odróżniać „pustej, ale możliwej” od „niemożliwej”.
   - (b) Sonda ma treść = hash „?”, którego model nie widział. Enkodery z uwagą mocno opierają się na zapamiętanej treści, więc sonda jest dla nich poza rozkładem.
   - (c) Luka w regularnej strukturze to pytanie o **brak**. Reguła siatki koduje to wprost, a podobieństwo embeddingów nie.
7. **Okres 8 (119+): wynik negatywny.** Most stawia na g14/p8, MLP na g16/p8. Według fizyki następne są g1/p8 (119) i g2/p8 (120). Model nie ma informacji o kolejności zapełniania (Z, konfiguracja elektronowa), więc nie może tego wiedzieć. Ranking p8 nie niesie treści.

### Werdykt

Na pełnej tablicy Most wyraźnie wygrywa tylko w **nienadzorowanym grupowaniu**, najmocniej dla bloku: +0.21 NMI w 5/5 seedów. W uczeniu nadzorowanym i ze wzmocnieniem przewaga jest zerowa albo działa w reżimie, w którym nic nie działa. **Nowego pierwiastka uczenie nienadzorowane nie znalazło.** W tym zadaniu wyuczone embeddingi są gorsze od surowych cech i od prostej reguły Mendelejewa.

**Co mogłoby zmienić wynik B (następne kroki):**
1. Treść sondy = średnia treść znanych pierwiastków zamiast nieznanego hasha (mniej poza rozkładem).
2. Wynik komórki z własnego celu modelu: prawdopodobieństwo krawędzi bind do przewidzianych sąsiadów zamiast gęstości.
3. Trudniejszy, uczciwszy wariant: cechy **bez** jawnej grupy i okresu (tylko Z i konfiguracja elektronowa). Wtedy reguła siatki nie ma z czego działać, a model musiałby sam odkryć okresowość.

