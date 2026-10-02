# Mendelejew ACTIVE — aktywny dobór ~20 par „ta sama grupa?”

Cechy jak ZC (tylko Z + konfiguracja). Wyrocznia odpowiada tak/nie na pary pierwiastków treningowych; budżet 20 zapytań w rundach po 5; ocena na pierwiastkach nigdy nieodpytanych. Seedy=[0, 1, 2, 3, 4] · kroki=600 · czas=1074s · skrypt `archiwum/skrypty/run_mendelejew_active.py`

Udział par „tak” w puli: 9.1%. Losowe hit@1 ¬f: 0.050. „wyrocznia” zna etykiety przy doborze (20 par „tak” po grupach) — to odniesienie, nie strategia aktywna.

## Po 20 zapytaniach — hit@1 bez bloku f (główna miara)

| strategia | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| losowe | 0.032 ± 0.033 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.047 ± 0.049 |
| najbliżsi | 0.008 ± 0.018 | 0.007 ± 0.017 | 0.000 ± 0.000 | 0.031 ± 0.018 |
| niepewność | 0.016 ± 0.022 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.018 |
| pokrycie | 0.016 ± 0.022 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.031 ± 0.018 |
| kontrast | 0.024 ± 0.022 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.055 ± 0.061 |
| wyrocznia | 0.159 ± 0.130 | 0.063 ± 0.071 | 0.015 ± 0.021 | 0.103 ± 0.048 |

## Ile „tak” zebrano / ile grup pokryto (średnio)

| strategia | most: tak / grupy | softmax: tak / grupy | zero: tak / grupy | mlp: tak / grupy |
|---|---:|---:|---:|---:|
| losowe | 2.0 / 2.0 | 2.0 / 2.0 | 2.0 / 2.0 | 2.0 / 2.0 |
| najbliżsi | 5.4 / 1.0 | 5.0 / 1.0 | 5.0 / 1.0 | 7.0 / 1.0 |
| niepewność | 6.2 / 1.6 | 6.2 / 1.6 | 6.4 / 1.4 | 7.0 / 1.8 |
| pokrycie | 5.6 / 1.4 | 3.8 / 1.4 | 3.8 / 1.4 | 4.8 / 1.6 |
| kontrast | 0.4 / 0.4 | 0.4 / 0.4 | 0.4 / 0.4 | 0.4 / 0.4 |
| wyrocznia | 20.0 / 17.4 | 20.0 / 17.4 | 20.0 / 17.4 | 20.0 / 17.4 |

## NMI grupy po 20 zapytaniach

| strategia | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| losowe | 0.393 ± 0.029 | 0.383 ± 0.019 | 0.397 ± 0.019 | 0.439 ± 0.052 |
| najbliżsi | 0.341 ± 0.042 | 0.385 ± 0.014 | 0.387 ± 0.011 | 0.424 ± 0.021 |
| niepewność | 0.364 ± 0.033 | 0.385 ± 0.040 | 0.372 ± 0.024 | 0.458 ± 0.033 |
| pokrycie | 0.347 ± 0.024 | 0.385 ± 0.022 | 0.388 ± 0.037 | 0.410 ± 0.041 |
| kontrast | 0.356 ± 0.024 | 0.382 ± 0.029 | 0.405 ± 0.040 | 0.451 ± 0.045 |
| wyrocznia | 0.574 ± 0.019 | 0.578 ± 0.015 | 0.559 ± 0.021 | 0.586 ± 0.017 |

## Krzywa uczenia Mostu (hit@1 ¬f po 5/10/15/20 zapytaniach)

| strategia | 5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|
| losowe | 0.024 | 0.016 | 0.015 | 0.032 |
| najbliżsi | 0.000 | 0.000 | 0.000 | 0.008 |
| niepewność | 0.000 | 0.007 | 0.008 | 0.016 |
| pokrycie | 0.015 | 0.007 | 0.008 | 0.016 |
| kontrast | 0.015 | 0.015 | 0.015 | 0.024 |
| wyrocznia | 0.023 | 0.071 | 0.048 | 0.159 |

## Najlepsza strategia aktywna vs losowe (hit@1 ¬f, wygrane seedy)

| arm | najbliżsi | niepewność | pokrycie | kontrast | wyrocznia |
|---|---:|---:|---:|---:|---:|
| most | 0/5 (-0.024) | 0/5 (-0.016) | 0/5 (-0.016) | 0/5 (-0.008) | 4/5 (+0.127) |
| softmax | 1/5 (+0.007) | 0/5 (+0.000) | 0/5 (+0.000) | 0/5 (+0.000) | 3/5 (+0.063) |
| zero | 0/5 (+0.000) | 0/5 (+0.000) | 0/5 (+0.000) | 0/5 (+0.000) | 2/5 (+0.015) |
| mlp | 1/5 (-0.015) | 1/5 (-0.015) | 1/5 (-0.015) | 2/5 (+0.009) | 3/5 (+0.056) |

## Most vs reszta przy tej samej strategii (hit@1 ¬f, wygrane seedy)

| strategia | vs softmax | vs zero | vs mlp |
|---|---:|---:|---:|
| losowe | 3/5 | 3/5 | 0/5 |
| najbliżsi | 1/5 | 1/5 | 0/5 |
| niepewność | 2/5 | 2/5 | 0/5 |
| pokrycie | 2/5 | 2/5 | 0/5 |
| kontrast | 3/5 | 3/5 | 1/5 |
| wyrocznia | 4/5 | 5/5 | 3/5 |

## Odpytane pary „tak” — Most, seed 0

| strategia | pary „tak” (grupa) |
|---|---|
| losowe | C–Sn (14) |
| najbliżsi | Md–No (3), Fm–Md (3), Tm–Yb (3), Er–Tm (3), Ho–Er (3) |
| niepewność | Md–No (3), Fm–Md (3), Tm–Yb (3), Er–Tm (3), Ho–Er (3), Y–Gd (3) |
| pokrycie | Pu–Am (3), Na–K (1), Md–No (3), Np–Cm (3), Sm–Eu (3), Am–Cf (3), Pr–Nd (3), Ho–Er (3) |
| kontrast | — |
| wyrocznia | Ru–Os (8), N–As (15), Zn–Hg (12), Ar–Og (18), Rh–Mt (9), Re–Bh (7), Zr–Rf (4), Pd–Pt (10), Ac–Cf (3), V–Db (5), H–K (1), Au–Rg (11), C–Pb (14), Sr–Ra (2), S–Te (16), I–Ts (17), B–Nh (13), W–Sg (6), Os–Hs (8), N–Bi (15) |

## Analogi luk (Most, seed 0, 4-NN po 20 zapytaniach)

Oczekiwane: Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re

| strategia | Sc | Ga | Ge | Tc |
|---|---|---|---|---|
| losowe | Ca, K, Ti, Cr | V, Ti, Cr, Ge | Mn, V, Ga, P | Ag, Rb, Zr, Br |
| najbliżsi | Ti, Ca, V, Mn | Ge, Fe, Mn, Co | Ga, As, Co, Fe | Rh, Ru, In, Zr |
| niepewność | Ti, Cr, Ca, V | Ge, Mn, Cu, Fe | Cu, As, Ga, Fe | Sb, Rb, Sn, Cd |
| pokrycie | Ru, Rh, Ne, Ca | Ge, As, Mn, Fe | Ga, As, Mn, V | Rh, Ag, Rb, Sn |
| kontrast | Ti, Cr, Ca, V | Mn, Cu, Ge, Fe | Fe, Co, Cu, As | Sn, Ag, In, Kr |
| wyrocznia | Ca, Ti, He, Y | Mn, Fe, Ge, Mo | Fe, Ga, As, Mn | Sg, W, Bh, Re |

## Wnioski

**Zastrzeżenia:** ~25 pierwiastków testowych spoza bloku f (1 trafienie ≈ 4 pkt), 5 seedów. Strategia `kontrast` została **dodana po obejrzeniu** porażki pozostałych (post-hoc) i nie zastępuje ich wyników.

1. **Aktywny dobór par oparty na podobieństwie modelu przegrywa z losowym.** Po 20 zapytaniach Most osiąga hit@1 ¬f: losowe 0.03, najbliżsi 0.01, niepewność 0.02, pokrycie 0.02 (0/5 seedów lepiej niż losowo). Tak samo jest u Softmax, ZERO i MLP. NMI grupy także spada poniżej losowego (Most 0.34–0.36 vs 0.39).
2. **Przyczyna: strategie zbierają „tanie” trafienia z bloku f.** Najbliżsi, niepewność i pokrycie znajdują 4–7 odpowiedzi „tak” (losowe ~2), ale prawie wyłącznie w **jednej** grupie. Przykłady z seed 0: Md–No, Tm–Yb, Er–Tm, Ho–Er, Pu–Am (wszystkie z grupy 3). Podobieństwo konfiguracji jest poziome, a jedyne miejsce, gdzie „poziomy sąsiad = ta sama grupa”, to lantanowce i aktynowce: w tej tablicy umownie wszystkie mają grupę 3. Model pyta tam, gdzie już się zgadza, więc odpowiedzi niczego nie uczą o blokach s/p/d. To klasyczne **błędne koło próbkowania w aktywnym uczeniu**: wybór oparty na własnej reprezentacji wzmacnia jej skrzywienie.
3. **Informatywne pary są pionowe i model ich nie proponuje.** Wyrocznia (20 par „tak” po grupach, 17 grup) daje Most **0.16** (4/5 vs losowe; 5/5 vs ZERO, 4/5 vs Softmax), czyli tyle co SEMI przy K=18 (0.14). Jej pary to N–As, C–Pb, S–Te, Ar–Og, Zn–Hg: duża różnica rdzenia, podobna walencja. Wartość tych 20 par wynika z **rozproszenia po grupach**, a nie z liczby.
4. **Post-hoc „kontrast” (podobne τ, inna treść) też zawodzi** (0–1 „tak” na 20). Tracer τ nie jest deskryptorem walencji: obsadzenie zewnętrznych d/f liczy też pełne podpowłoki rdzenia (N ma d=0, As d=1.0, choć oba są w grupie 15). Heurystyka wybiera pary lekki + ciężki z bloku f (N–Np, C–Cm). Bez pojęcia rdzenia gazu szlachetnego nie da się bez etykiet wskazać pionowych par.
5. **Most przy dobrze rozłożonych etykietach dalej wygrywa:** przy wyroczni 5/5 vs ZERO i 4/5 vs Softmax (spójnie z SEMI). Przy strategiach aktywnych nie ma czego przenosić, więc wszystkie modele są przy zerze.

## Werdykt

Aktywny wybór „najbardziej informatywnych” par **nie działa**, gdy kryterium informatywności pochodzi z reprezentacji, która sama nie zna grup. Strategie zbierają trafienia w bloku f i są gorsze od losowania. Informatywne pary (pionowe, jedna lub dwie na grupę) są dla tej reprezentacji niewidoczne. Wybór 20 par daje wynik dopiero, gdy ktoś **wie**, że mają pokryć różne grupy, a tę wiedzę wnosi wyrocznia, nie model.

**Kierunki:**
1. Strategia „pokrycie” z wykluczeniem par o bliskim Z (|ΔZ| > 8). To prosta, bezetykietowa reguła przeciw poziomym parom, ale również post-hoc.
2. Deskryptor walencji (elektrony ponad rdzeń gazu szlachetnego) w τ. Zmienia założenie „tylko Z + konfiguracja”, bo wnosi pojęcie rdzenia.
3. Zapytania o pierwiastek („do której grupy należy X?”) zamiast o pary: jedno zapytanie daje etykietę, a z niej wiele par.

