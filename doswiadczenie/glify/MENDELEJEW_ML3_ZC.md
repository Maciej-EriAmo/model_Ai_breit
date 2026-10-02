# Mendelejew ML3 ZC — pełna tablica, cechy tylko z Z i konfiguracji elektronowej

Model **nie widzi** okresu, grupy ani bloku (ani w treści, ani w tracerze, ani w krawędziach). Etykiety służą tylko do oceny i jako nagroda środowiska w RL.

Treść x = obsadzenie 24 podpowłok (kolejność Madelunga) · tracer τ = [Z, zewn. s/p/d/f, liczba podpowłok] · T-seal wyłączony · krawędzie samonadzoru = 3-NN w surowych cechach (226 par: ta sama grupa 22%, ten sam okres 90%, sąsiedzi Z 38%)

Seedy=[0, 1, 2, 3, 4] · kroki SL=400 UL=400 RL=600 · czas=231s · skrypt `archiwum/skrypty/run_mendelejew_ml3_zc.py`

## A1. Nadzorowane — czy okres/grupę/blok da się przewidzieć z konfiguracji (test 30%)

| zadanie | klasy | większościowa | most | softmax | zero | mlp | most>SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| period | 7 | 0.286 | 0.983 ± 0.026 | 0.983 ± 0.026 | 0.989 ± 0.016 | 0.943 ± 0.020 | 0/5 |
| group | 18 | 0.286 | 0.309 ± 0.071 | 0.246 ± 0.016 | 0.251 ± 0.031 | 0.269 ± 0.059 | 4/5 |
| block | 4 | 0.314 | 0.817 ± 0.048 | 0.794 ± 0.074 | 0.771 ± 0.081 | 0.857 ± 0.029 | 4/5 |

## A2. Nienadzorowane — czy okresowość wyłania się bez etykiet (k-means → NMI)

| cechy / arm | stan | NMI okres | NMI grupa | NMI blok |
|---|---|---:|---:|---:|
| surowe content | — | 0.975 ± 0.017 | 0.422 ± 0.007 | 0.201 ± 0.026 |
| surowe tracer | — | 0.436 ± 0.023 | 0.530 ± 0.016 | 0.140 ± 0.004 |
| surowe concat | — | 0.581 ± 0.049 | 0.458 ± 0.022 | 0.190 ± 0.000 |
| most | untrained | 0.801 ± 0.053 | 0.398 ± 0.008 | 0.175 ± 0.022 |
| most | ssl | 0.602 ± 0.060 | 0.425 ± 0.013 | 0.128 ± 0.036 |
| softmax | untrained | 0.944 ± 0.042 | 0.431 ± 0.010 | 0.180 ± 0.028 |
| softmax | ssl | 0.736 ± 0.033 | 0.457 ± 0.009 | 0.132 ± 0.059 |
| zero | untrained | 0.888 ± 0.062 | 0.426 ± 0.011 | 0.172 ± 0.036 |
| zero | ssl | 0.738 ± 0.035 | 0.453 ± 0.006 | 0.169 ± 0.065 |
| mlp | untrained | 0.562 ± 0.063 | 0.436 ± 0.012 | 0.114 ± 0.028 |
| mlp | ssl | 0.537 ± 0.051 | 0.436 ± 0.007 | 0.021 ± 0.007 |

## A3. Ze wzmocnieniem — nagroda: wybrany pierwiastek z tej samej grupy

| arm | RL test hit@1 | RL train hit@1 | SL-ref test hit@1 | losowa |
|---|---:|---:|---:|---:|
| most | 0.217 ± 0.052 | 0.251 ± 0.022 | 0.703 ± 0.092 | 0.092 |
| softmax | 0.240 ± 0.069 | 0.258 ± 0.022 | 0.497 ± 0.198 | 0.092 |
| zero | 0.229 ± 0.061 | 0.243 ± 0.040 | 0.491 ± 0.127 | 0.092 |
| mlp | 0.434 ± 0.118 | 0.508 ± 0.093 | 0.634 ± 0.051 | 0.092 |

Most > Softmax (RL test) na 1/5 seedach.

## B. Odkrycie — jaka jest konfiguracja brakującego pierwiastka?

Usunięte: **Sc** (Z=21, eka-bor), **Ga** (Z=31, eka-glin), **Ge** (Z=32, eka-krzem), **Tc** (Z=43, eka-mangan). Z luki wynika z ciągu Z; model ma wybrać jej konfigurację spośród kandydatów (prawdziwa + warianty z 1–2 elektronami przesuniętymi w zewnętrznych podpowłokach). Wynik kandydata = CSLS w przestrzeni embeddingów (uczonej bez luk). Pozycja 1 = trafienie.

| luka | kandydatów | prawdziwa konfiguracja (koniec) | Aufbau | surowe kNN | most | softmax | zero | mlp |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Sc | 87 | 2p6 3s2 3p6 4s2 3d1 | 1 | 2 | 1.4 (4/5×1.) | 1.0 (5/5×1.) | 1.2 (4/5×1.) | 1.0 (5/5×1.) |
| Ga | 87 | 3s2 3p6 4s2 3d10 4p1 | 1 | 23 | 19.4 (0/5×1.) | 8.8 (0/5×1.) | 11.2 (0/5×1.) | 15.6 (0/5×1.) |
| Ge | 90 | 3s2 3p6 4s2 3d10 4p2 | 1 | 13 | 10.8 (0/5×1.) | 11.8 (0/5×1.) | 14.0 (0/5×1.) | 9.4 (0/5×1.) |
| Tc | 90 | 4s2 3d10 4p6 5s2 4d5 | 1 | 19 | 38.2 (0/5×1.) | 24.6 (0/5×1.) | 28.4 (0/5×1.) | 37.8 (0/5×1.) |

Średnia pozycja (w nawiasie: ile seedów trafiło na 1. miejsce). Losowo ≈ (kandydaci+1)/2.

### Analogi i przewidziana grupa/blok (sonda z prawdziwą konfiguracją, 4-NN wśród znanych, seed 0)

| luka | grupa prawdz. | metoda | 4-NN | grupa przewid. | blok przewid. |
|---|---:|---|---|---:|:-:|
| Sc | 3 | surowe cechy kNN | Ti, Ca, V, Mn | 4 ✗ | d |
| Sc | 3 | most | Ti, Ca, V, K | 4 ✗ | d |
| Sc | 3 | softmax | Ca, Ti, V, K | 2 ✗ | s |
| Sc | 3 | zero | Ca, Ti, V, K | 2 ✗ | s |
| Sc | 3 | mlp | Ca, Ti, V, K | 2 ✗ | s |
| Ga | 13 | surowe cechy kNN | As, Se, In, Sn | 15 ✗ | p |
| Ga | 13 | most | As, Se, Br, Kr | 15 ✗ | p |
| Ga | 13 | softmax | Zn, As, Se, Cu | 12 ✗ | d |
| Ga | 13 | zero | Zn, As, Se, Cu | 12 ✗ | d |
| Ga | 13 | mlp | As, Se, Br, In | 15 ✗ | p |
| Ge | 14 | surowe cechy kNN | As, Se, Br, Sn | 15 ✗ | p |
| Ge | 14 | most | As, Se, Br, Kr | 15 ✗ | p |
| Ge | 14 | softmax | As, Se, Zn, Br | 15 ✗ | p |
| Ge | 14 | zero | As, Se, Zn, Br | 15 ✗ | p |
| Ge | 14 | mlp | As, Se, Br, Kr | 15 ✗ | p |
| Tc | 7 | surowe cechy kNN | Zr, Y, Cd, Sr | 4 ✗ | d |
| Tc | 7 | most | Zr, Ce, Y, Gd | 3 ✗ | d |
| Tc | 7 | softmax | Zr, Y, Cd, Sr | 4 ✗ | d |
| Tc | 7 | zero | Zr, Y, Cd, Sr | 4 ✗ | d |
| Tc | 7 | mlp | Zr, Y, La, Ce | 3 ✗ | d |

Trafność grupy z 4-NN po wszystkich seedach:

| metoda | Sc | Ga | Ge | Tc | razem |
|---|---:|---:|---:|---:|---:|
| surowe cechy kNN | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| most | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| softmax | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| zero | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| mlp | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |

### Eksploracja: Z=119 i 120 (seed 0, top-3 wg CSLS)

Teoria (ekstrapolacja Aufbau / obliczenia relatywistyczne): 119 = [Og] 8s1, 120 = [Og] 8s2. Uwaga: podpowłoki 8s/5g/6f/7d/8p nie są obsadzone u żadnego znanego pierwiastka — ich wagi w treści są nieuczone; o wyborze może decydować tylko τ.

| Z | arm | 1. | 2. | 3. | 4-NN zwycięzcy |
|---|---|---|---|---|---|
| 119 | most | +8p1 (0.975) | +5g1 (0.973) | +8s1 (0.951) | Fl, Nh, Mc, Lv |
| 119 | softmax | +6f1 (0.993) | +5g1 (0.992) | +8p1 (0.992) | Og, Ts, Lv, Mc |
| 119 | zero | +8p1 (0.993) | +6f1 (0.992) | +5g1 (0.992) | Og, Ts, Lv, Mc |
| 119 | mlp | +8p1 (0.979) | +5g1 (0.951) | +8s1 (0.562) | Nh, Fl, Mc, Lv |
| 120 | most | +8p2 (0.989) | +5g1 +8p1 (0.976) | +5g2 (0.973) | Mc, Fl, Nh, Lv |
| 120 | softmax | +6f2 (0.993) | +5g1 +6f1 (0.992) | +6f1 +8p1 (0.992) | Og, Ts, Lv, Mc |
| 120 | zero | +6f1 +8p1 (0.993) | +6f2 (0.992) | +5g1 +8p1 (0.992) | Og, Ts, Lv, Mc |
| 120 | mlp | +8p2 (0.999) | +5g1 +8p1 (0.979) | +5g2 (0.949) | Fl, Mc, Nh, Lv |

## Wnioski

**Zastrzeżenia:** test SL ma ~35 pierwiastków, odkrycie tylko 4 luki, 5 seedów. Liczę się wyłącznie ze stałym znakiem efektu. Krawędzie samonadzoru (3-NN w surowych cechach) wyszły w 90% wewnątrz okresu, a tylko w 22% wewnątrz grupy. To własność samych konfiguracji, nie wybór modelu.

### Co z konfiguracji wyłania się samo, a co nie

1. **Okres wyłania się trywialnie, grupa wcale.**
   - Surowe obsadzenia podpowłok grupują okresy niemal idealnie (k-means NMI 0.975). Klasyfikacja okresu daje ~0.98 we wszystkich modelach.
   - Grupy nie odtwarza nic. NMI grupy wynosi 0.42–0.53 w każdej metodzie, przy czym najlepszy jest surowy tracer z zewnętrznymi podpowłokami (0.53). Klasyfikacja grupy daje 0.25–0.31, przy klasie większościowej 0.29.
   - Podobieństwo konfiguracji jest **poziome**: najbliżsi sąsiedzi to ten sam okres. Odkrycie Mendelejewa było **pionowe**: podobieństwo właściwości w grupie przy różnej liczbie powłok. Tego model nie znajduje sam.
2. **Samonadzór pogarsza okres.** NMI okresu spada z 0.98 (surowe) do 0.60–0.74 po InfoNCE. Most traci najwięcej: 0.60, wobec Softmax/ZERO 0.74, przegrywa 0/5. Most nie wygrywa też w grupie ani w bloku. To odwrócenie wyniku z FULL118, gdzie przy jawnej grupie i okresie w tracerze Most był najlepszy w nienadzorowanym.
3. **Blok da się nauczyć z nadzorem (0.77–0.86), ale nie bez niego** (NMI ≤ 0.2).

### Most

4. **Nadzorowane:** w grupie Most jest lepszy od Softmax (0.31 vs 0.25, 4/5) i w bloku (0.82 vs 0.79, 4/5). Grupa pozostaje jednak na poziomie klasy większościowej.
5. ~~Nowy, mocny wynik: Most 0.70 vs Softmax 0.50 (5/5)~~ — **KOREKTA (2026-10-02): wynik zawyżony przez przeciek etykiet.** `run_rl(mode="sl_ref")` liczy jako pozytywy WSZYSTKICH członków grupy, także pierwiastki testowe, więc trening przyciągał testowe pierwiastki do ich grup. Po ograniczeniu celów do pierwiastków treningowych (ten sam split i seedy): **Most 0.45, Softmax 0.31** — Most nadal lepszy w 4/5 seedów (+0.13), ale wartości bezwzględne o ~0.25 niższe. Ten sam przeciek dotyczy kolumny RL tego raportu (nagroda za trafienie w testowy pierwiastek). Szczegóły i czysty protokół: [`MENDELEJEW_SEMI.md`](MENDELEJEW_SEMI.md).
6. **To samo zadanie uczone samą nagrodą:** wszystkie modele są powyżej losowego (0.22–0.24 przy losowym 0.09), ale Most nie wygrywa (1/5). MLP jest najlepszy (0.43). Sygnał 0/1 nie wystarcza, żeby uruchomić przewagę Mostu.

### Odkrycie: jaka konfiguracja ma brakujący pierwiastek

7. **Reguła Aufbau (Madelung, bez ML) trafia 4/4.** Modele trafiają tylko Sc: pozycja 1 w 4–5/5 seedów u wszystkich, także w surowych cechach (2.). Dla Ga i Ge prawdziwa konfiguracja ląduje na pozycji 9–19 z ~88: lepiej niż losowo (~44), ale bez trafienia. Dla Tc na pozycji 25–38, czyli prawie losowo. Najmniej źle wypada Softmax (średnio 8.8 / 11.8 / 24.6). Most jest gorszy, zwłaszcza dla Tc (38).
8. **Analogi z konfiguracji są złe chemicznie.** Najbliżsi znani dla Ga to As, Se, Br (ten sam okres), a nie Al, In (ta sama grupa). Grupa przewidziana z sąsiadów jest trafna w **0%** przypadków we wszystkich metodach. Bez jawnej grupy model nie odtworzy przewidywań „eka-” Mendelejewa.
9. **Z=119/120: wynik negatywny.** Żaden model nie stawia 8s1/8s2 na pierwszym miejscu. Most i MLP wybierają 8p, Softmax i ZERO 6f/5g. Most i MLP mają 8s na trzecim miejscu. Podpowłok 8s–8p nikt nie widział w treningu, więc to ekstrapolacja poza dane.

### Werdykt

Z samych Z i konfiguracji uczenie nienadzorowane **odtwarza okresy, ale nie grupy**. Nie odkrywa układu okresowego w sensie Mendelejewa ani nie znajduje konfiguracji nowych pierwiastków lepiej od reguły Madelunga. Most traci tu przewagę z FULL118 w nienadzorowanym.

W **nadzorowanym uczeniu relacji „ta sama grupa”** Most pozostaje lepszy od Softmax po usunięciu przecieku (0.45 vs 0.31, 4/5), ale pierwotne 0.70 vs 0.50 było zawyżone (korekta w pkt 5).

**Kierunki:**
1. Cel samonadzoru, który nie premiuje poziomego podobieństwa, np. kontrast na samych zewnętrznych podpowłokach (τ) albo maskowanie rdzenia.
2. Reprezentacja „walencyjna” (konfiguracja minus rdzeń gazu szlachetnego). Wprowadza to jednak pojęcie rdzenia, czyli wiedzę spoza czystej konfiguracji.
3. Połączyć przewagę Mostu z pkt 5 z małą liczbą etykiet (semi-supervised): kilka znanych par grupowych i reszta bez etykiet.

