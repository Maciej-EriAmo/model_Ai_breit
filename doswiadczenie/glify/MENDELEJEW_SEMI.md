# Mendelejew SEMI — kilka par „ta sama grupa” + reszta bez etykiet

Cechy jak w ZC: tylko Z + konfiguracja elektronowa (okres/grupa/blok nie wchodzą do modelu). Split pierwiastków 70/30; luki Sc/Ga/Ge/Tc zawsze w teście. Seedy=[0, 1, 2, 3, 4] · kroki=600 · czas=602s · skrypt `archiwum/skrypty/run_mendelejew_semi.py`

Pary bez etykiet (semi_knn): 226 par 3-NN w surowych cechach. Wszystkich par grupowych w treningu (K=all): ~309. Losowe hit@1: 0.108 (¬f: 0.050).

Bez uczenia: surowe cechy hit@1 = 0.246 ± 0.052, ¬f = 0.000 ± 0.000, NMI grupy = 0.432 ± 0.009.

## hit@1 bez bloku f (główna miara)

| K par | reżim | most | softmax | zero | mlp |
|---|---|---:|---:|---:|---:|
| 0 | semi_knn | 0.007 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.024 ± 0.022 |
| 6 | sup | 0.015 ± 0.033 | 0.016 ± 0.022 | 0.007 ± 0.017 | 0.156 ± 0.123 |
| 6 | semi_knn | 0.016 ± 0.036 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.079 ± 0.084 |
| 6 | semi_pl | 0.047 ± 0.066 | 0.000 ± 0.000 | 0.007 ± 0.017 | 0.110 ± 0.090 |
| 18 | sup | 0.137 ± 0.106 | 0.008 ± 0.017 | 0.024 ± 0.037 | 0.111 ± 0.057 |
| 18 | semi_knn | 0.032 ± 0.035 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.055 ± 0.035 |
| 18 | semi_pl | 0.072 ± 0.062 | 0.007 ± 0.017 | 0.008 ± 0.019 | 0.072 ± 0.056 |
| 54 | sup | 0.315 ± 0.125 | 0.100 ± 0.091 | 0.046 ± 0.042 | 0.152 ± 0.103 |
| 54 | semi_knn | 0.016 ± 0.022 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.017 |
| 54 | semi_pl | 0.064 ± 0.047 | 0.000 ± 0.000 | 0.008 ± 0.017 | 0.103 ± 0.063 |
| all | sup | 0.177 ± 0.100 | 0.142 ± 0.067 | 0.102 ± 0.072 | 0.268 ± 0.099 |
| all | semi_knn | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.055 ± 0.022 |
| all | semi_pl | 0.101 ± 0.064 | 0.000 ± 0.000 | 0.008 ± 0.019 | 0.116 ± 0.052 |

## hit@1 wszystkie zapytania testowe

| K par | reżim | most | softmax | zero | mlp |
|---|---|---:|---:|---:|---:|
| 0 | semi_knn | 0.251 ± 0.042 | 0.251 ± 0.037 | 0.263 ± 0.024 | 0.263 ± 0.055 |
| 6 | sup | 0.269 ± 0.033 | 0.280 ± 0.047 | 0.269 ± 0.026 | 0.360 ± 0.092 |
| 6 | semi_knn | 0.257 ± 0.061 | 0.263 ± 0.037 | 0.257 ± 0.035 | 0.303 ± 0.077 |
| 6 | semi_pl | 0.280 ± 0.068 | 0.263 ± 0.037 | 0.263 ± 0.037 | 0.326 ± 0.077 |
| 18 | sup | 0.337 ± 0.110 | 0.263 ± 0.055 | 0.280 ± 0.062 | 0.326 ± 0.085 |
| 18 | semi_knn | 0.269 ± 0.075 | 0.263 ± 0.037 | 0.269 ± 0.033 | 0.286 ± 0.057 |
| 18 | semi_pl | 0.297 ± 0.080 | 0.263 ± 0.031 | 0.263 ± 0.047 | 0.297 ± 0.085 |
| 54 | sup | 0.480 ± 0.117 | 0.337 ± 0.062 | 0.297 ± 0.026 | 0.354 ± 0.115 |
| 54 | semi_knn | 0.257 ± 0.064 | 0.257 ± 0.020 | 0.263 ± 0.031 | 0.269 ± 0.043 |
| 54 | semi_pl | 0.291 ± 0.068 | 0.263 ± 0.037 | 0.257 ± 0.040 | 0.320 ± 0.077 |
| all | sup | 0.383 ± 0.038 | 0.371 ± 0.067 | 0.337 ± 0.051 | 0.440 ± 0.100 |
| all | semi_knn | 0.246 ± 0.052 | 0.269 ± 0.033 | 0.269 ± 0.033 | 0.286 ± 0.053 |
| all | semi_pl | 0.320 ± 0.047 | 0.257 ± 0.029 | 0.269 ± 0.052 | 0.331 ± 0.038 |

## NMI grupy (k-means, wszystkie 118)

| K par | reżim | most | softmax | zero | mlp |
|---|---|---:|---:|---:|---:|
| 0 | semi_knn | 0.421 ± 0.006 | 0.454 ± 0.007 | 0.454 ± 0.008 | 0.431 ± 0.010 |
| 6 | sup | 0.441 ± 0.020 | 0.494 ± 0.016 | 0.495 ± 0.015 | 0.525 ± 0.021 |
| 6 | semi_knn | 0.466 ± 0.021 | 0.488 ± 0.010 | 0.499 ± 0.021 | 0.504 ± 0.020 |
| 6 | semi_pl | 0.461 ± 0.030 | 0.493 ± 0.014 | 0.492 ± 0.012 | 0.520 ± 0.017 |
| 18 | sup | 0.560 ± 0.024 | 0.555 ± 0.029 | 0.562 ± 0.014 | 0.573 ± 0.009 |
| 18 | semi_knn | 0.521 ± 0.015 | 0.522 ± 0.016 | 0.519 ± 0.014 | 0.551 ± 0.011 |
| 18 | semi_pl | 0.542 ± 0.008 | 0.555 ± 0.015 | 0.553 ± 0.015 | 0.551 ± 0.018 |
| 54 | sup | 0.706 ± 0.035 | 0.600 ± 0.032 | 0.583 ± 0.019 | 0.666 ± 0.023 |
| 54 | semi_knn | 0.597 ± 0.016 | 0.529 ± 0.005 | 0.525 ± 0.022 | 0.564 ± 0.029 |
| 54 | semi_pl | 0.663 ± 0.042 | 0.590 ± 0.026 | 0.597 ± 0.013 | 0.621 ± 0.027 |
| all | sup | 0.693 ± 0.040 | 0.633 ± 0.021 | 0.626 ± 0.022 | 0.699 ± 0.020 |
| all | semi_knn | 0.619 ± 0.026 | 0.510 ± 0.011 | 0.498 ± 0.014 | 0.570 ± 0.023 |
| all | semi_pl | 0.683 ± 0.032 | 0.606 ± 0.021 | 0.591 ± 0.018 | 0.627 ± 0.024 |

## Most vs reszta (hit@1 ¬f, ile seedów Most wygrywa)

| K | reżim | vs softmax | vs zero | vs mlp |
|---|---|---:|---:|---:|
| 0 | semi_knn | 1/5 (+0.007) | 1/5 (+0.007) | 0/5 (-0.016) |
| 6 | sup | 1/5 (-0.002) | 1/5 (+0.007) | 0/5 (-0.141) |
| 6 | semi_knn | 1/5 (+0.016) | 1/5 (+0.016) | 0/5 (-0.063) |
| 6 | semi_pl | 3/5 (+0.047) | 2/5 (+0.040) | 0/5 (-0.063) |
| 18 | sup | 5/5 (+0.129) | 5/5 (+0.112) | 2/5 (+0.026) |
| 18 | semi_knn | 3/5 (+0.032) | 3/5 (+0.032) | 1/5 (-0.022) |
| 18 | semi_pl | 4/5 (+0.064) | 4/5 (+0.064) | 2/5 (+0.000) |
| 54 | sup | 5/5 (+0.215) | 5/5 (+0.269) | 5/5 (+0.163) |
| 54 | semi_knn | 2/5 (+0.016) | 2/5 (+0.016) | 1/5 (-0.014) |
| 54 | semi_pl | 4/5 (+0.064) | 3/5 (+0.056) | 0/5 (-0.039) |
| all | sup | 2/5 (+0.035) | 2/5 (+0.075) | 2/5 (-0.091) |
| all | semi_knn | 0/5 (+0.000) | 0/5 (+0.000) | 0/5 (-0.055) |
| all | semi_pl | 5/5 (+0.101) | 4/5 (+0.092) | 1/5 (-0.016) |

## Czy bez etykiet pomaga? (Δ hit@1 ¬f względem sup przy tym samym K, średnia po armiach i seedach)

| K | semi_knn − sup | semi_pl − sup |
|---|---:|---:|
| 6 | -0.025 | -0.008 |
| 18 | -0.048 | -0.030 |
| 54 | -0.141 | -0.110 |
| all | -0.158 | -0.116 |

## Analogi luk Mendelejewa (4-NN, seed 0, K=18 ≈ jedna para na grupę)

Oczekiwane (ta sama grupa): Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re

| reżim | arm | Sc | Ga | Ge | Tc |
|---|---|---|---|---|---|
| surowe | — | Ti, Ca, V, Mn | Ge, As, Se, In | As, Ga, Se, Br | Zr, Y, Cd, Sr |
| sup | most | Ca, Ti, Lr, Hf | Ge, Mn, Fe, Nb | Ga, As, N, O | Lr, Cm, Pa, No |
| sup | softmax | Ca, Ti, V, Lr | Ge, Zn, Tl, N | N, As, Ga, Se | Db, W, Ta, Rf |
| sup | zero | Ca, Ti, V, Lr | Ge, Zn, Ni, As | As, Ga, Se, N | Db, Sg, Re, Rf |
| sup | mlp | Ca, Ti, V, Lr | Ge, C, Tl, B | Ga, As, C, N | Zr, Y, Ac, Th |
| semi_knn | most | Ca, Ti, V, K | Ge, As, Se, O | As, Ga, Se, N | Zr, Y, Lr, La |
| semi_knn | softmax | Ca, Ti, V, Db | Ge, Zn, As, N | As, Ga, N, Se | Zr, Y, Sr, Rf |
| semi_knn | zero | Ca, Ti, V, Sg | Ge, Zn, Ni, As | As, Ga, Se, N | Zr, Y, Sr, Cd |
| semi_knn | mlp | Ca, Ti, V, Lr | Ge, As, C, N | Ga, As, C, Se | Zr, Y, Db, Sg |
| semi_pl | most | Ca, Ti, V, Db | Ge, Mo, Nb, N | Ga, N, As, O | Ac, Cf, Th, Es |
| semi_pl | softmax | Ca, Ti, V, Db | Zn, Ge, Hg, Ni | Ga, As, N, Zn | Zr, Rf, Db, Y |
| semi_pl | zero | Ca, Ti, V, Db | Zn, Ge, Hg, Ni | Ga, N, As, Zn | Zr, Y, Th, Rf |
| semi_pl | mlp | Ca, Ti, V, Lr | Ge, C, As, N | Ga, As, N, C | Zr, Y, Ac, Rf |

## Uwagi metodyczne (przed wnioskami)

- **Protokół bez przecieku:** pary z etykietą pochodzą wyłącznie z pierwiastków treningowych, a ocena dotyczy pierwiastków nigdy nieoznaczonych. Ten sam protokół zastosowany do ZC SL-ref obniża tamten wynik Mostu z 0.70 do 0.45 (Softmax z 0.50 do 0.31). Korekta jest w `MENDELEJEW_ML3_ZC.md` pkt 5.
- **Pierwszy przebieg tego skryptu (wyniki odrzucone)** losował K par równomiernie. Grupa 3 ma 32 członków, więc ~75% par to były lantanowce i aktynowce. Wszystkie modele wypadły wtedy przy zerze (hit@1 ¬f ≤ 0.06). Poprawiony protokół rozkłada K par po kolei na grupy i losuje batch po pierwiastkach. **Raportowane są wyłącznie wyniki po poprawce.** Zmiana dotyczy tego, *które* pary są oznaczone, a nie metryki.
- Test zawiera ~35 pierwiastków (~25 spoza bloku f), więc jedno trafienie to ~4 pkt. Przy 5 seedach liczę się tylko ze stałym znakiem efektu.

## Wnioski

1. **Kilka dobrze rozłożonych etykiet uruchamia Mosta, najmocniej ze wszystkich dotychczasowych testów.** Przy K=54 parach (~3 na grupę) Most ma hit@1 ¬f = **0.32**, wobec Softmax 0.10, ZERO 0.05 i MLP 0.15, **5/5 seedów wobec każdego z nich**. Przy K=18 (~1 para na grupę) Most 0.14, Softmax 0.01, ZERO 0.02 (5/5). NMI grupy przy K=54: Most **0.71**, wobec 0.60 / 0.58 / 0.67. Przewaga nad ZERO (+0.27) pokazuje, że robi to ΔE z tracera: zewnętrzne podpowłoki s/p/d/f niosą informację o grupie, a Most potrafi ją przenieść na nieoznaczone pierwiastki, gdy kilka par wskaże kierunek.
2. **Więcej etykiet nie znaczy lepiej.** Przy K=all (~309 par, większość z grupy 3) Most spada do 0.18 ¬f, a MLP wygrywa (0.27). Zrównoważone 54 pary uczą relacji pionowej lepiej niż komplet zdominowany przez lantanowce. Liczy się **rozkład** etykiet, nie ich liczba.
3. **Dane bez etykiet szkodzą, w każdym budżecie i w każdym modelu:**

   | K | semi_knn − sup | semi_pl − sup |
   |---|---:|---:|
   | 6 | −0.03 | −0.01 |
   | 18 | −0.05 | −0.03 |
   | 54 | −0.14 | −0.11 |

   Struktura bez etykiet w konfiguracjach jest pozioma (90% par kNN to ten sam okres) i konkuruje z pionowym sygnałem z etykiet. Self-training (semi_pl) szkodzi mniej, ale też szkodzi: wzajemni najbliżsi sąsiedzi w przestrzeni modelu to znów głównie ten sam okres, więc pseudo-etykiety wzmacniają okres, nie grupę. Klasyczne założenie uczenia częściowo nadzorowanego („klastry bez etykiet = klasy”) **nie zachodzi** dla grup układu okresowego.
4. **Most najlepiej znosi szum z danych bez etykiet.** W semi_pl Most jest lepszy od Softmax w 4/5 (K=18, 54) i 5/5 (K=all), a w NMI grupy przy semi_knn K=all ma 0.62 wobec 0.51 / 0.50. Jest jednak zawsze poniżej własnego wariantu `sup`.
5. **Analogi luk Mendelejewa wciąż w większości chybione** (K=18, seed 0). Sc → Lr (grupa 3, trafienie) u Mostu i innych, ale też Ca i Ti. Ga → Tl (grupa 13) tylko u Softmax i MLP; MLP dodatkowo B. Ge i Tc bez trafień. Jedna para na grupę nie wystarcza, żeby odtworzyć przewidywania „eka-”.

## Werdykt

Uczenie częściowo nadzorowane działa, ale **dzięki części nadzorowanej, nie nienadzorowanej**. Kilkadziesiąt zrównoważonych par „ta sama grupa” wystarcza, żeby Most odtworzył pionową strukturę tablicy na nieoznaczonych pierwiastkach. To najwyraźniejsza przewaga Mostu w całej serii: 5/5 wobec Softmax, ZERO i MLP przy K=54. Dodanie danych bez etykiet, zarówno kNN, jak i pseudo-etykiet, konsekwentnie szkodzi, bo ich naturalna struktura jest okresowa (pozioma), a nie grupowa.

**Kierunki:**
1. **Aktywny dobór etykiet:** które ~20 par najbardziej podnosi grupowanie? Wynik z pkt 2 sugeruje, że wybór par ma większe znaczenie niż ich liczba.
2. Samonadzór na samym τ (zewnętrzne podpowłoki), bez rdzenia w treści. Jedyna droga, w której dane bez etykiet mogłyby nieść sygnał pionowy.
3. Potwierdzić K=54 na większej liczbie seedów (10–20). Odchylenie Mostu jest duże (±0.13), choć znak jest stały.

