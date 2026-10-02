# Mendelejew COMP — relacje kompozycyjne („w dół”, „w prawo”)

Cechy jak ZC (tylko Z + konfiguracja). Fakty uczone tylko między pierwiastkami treningowymi; „dół” = ta sama grupa, okres+1 (bez bloku f); „prawo” = Z+1. Seedy=[0, 1, 2, 3, 4] · kroki=800 · czas=123s · skrypt `archiwum/skrypty/run_mendelejew_comp.py`

Fakty (wszystkie): dół=70, prawo=117. Losowe hit@1 ≈ 0.009. `op` = złożony operator; `hop` = dwa dyskretne kroki; `nn` = bez operatora (najbliższy sąsiad).

## A. Jeden krok — fakty z pierwiastkiem testowym (indukcja), hit@1

| test | lin:most | lin:softmax | lin:zero | lin:mlp | trans:most | trans:softmax | trans:zero | trans:mlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| operator | 0.467 ± 0.061 | 0.475 ± 0.043 | 0.471 ± 0.056 | 0.363 ± 0.093 | 0.244 ± 0.033 | 0.280 ± 0.071 | 0.288 ± 0.077 | 0.424 ± 0.073 |
| bez operatora | 0.317 ± 0.030 | 0.305 ± 0.021 | 0.291 ± 0.013 | 0.227 ± 0.041 | 0.259 ± 0.034 | 0.290 ± 0.040 | 0.293 ± 0.037 | 0.210 ± 0.025 |
| (kontrola: fakty treningowe) | 0.970 ± 0.027 | 0.787 ± 0.038 | 0.801 ± 0.045 | 0.789 ± 0.033 | 0.829 ± 0.044 | 0.640 ± 0.079 | 0.638 ± 0.072 | 0.621 ± 0.057 |
| operator · tylko dół | 0.325 ± 0.112 | 0.253 ± 0.107 | 0.248 ± 0.127 | 0.338 ± 0.138 | 0.120 ± 0.032 | 0.189 ± 0.095 | 0.187 ± 0.119 | 0.245 ± 0.142 |
| bez operatora · tylko dół | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.024 ± 0.025 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| operator · tylko prawo | 0.553 ± 0.045 | 0.607 ± 0.055 | 0.605 ± 0.081 | 0.378 ± 0.097 | 0.318 ± 0.052 | 0.333 ± 0.091 | 0.347 ± 0.086 | 0.532 ± 0.063 |
| bez operatora · tylko prawo | 0.506 ± 0.042 | 0.488 ± 0.036 | 0.465 ± 0.029 | 0.362 ± 0.061 | 0.400 ± 0.063 | 0.464 ± 0.078 | 0.469 ± 0.064 | 0.335 ± 0.036 |

## B. Złożenia nigdy nieuczone, hit@1

| test | lin:most | lin:softmax | lin:zero | lin:mlp | trans:most | trans:softmax | trans:zero | trans:mlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ścieżka treningowa · op | 0.742 ± 0.059 | 0.597 ± 0.084 | 0.523 ± 0.073 | 0.538 ± 0.096 | 0.232 ± 0.056 | 0.261 ± 0.029 | 0.250 ± 0.054 | 0.307 ± 0.012 |
| ścieżka treningowa · hop | 0.931 ± 0.060 | 0.592 ± 0.141 | 0.638 ± 0.110 | 0.751 ± 0.066 | 0.774 ± 0.081 | 0.499 ± 0.098 | 0.472 ± 0.123 | 0.512 ± 0.081 |
| ścieżka treningowa · nn | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ścieżka dotyka testu · op | 0.269 ± 0.051 | 0.251 ± 0.083 | 0.257 ± 0.043 | 0.227 ± 0.071 | 0.094 ± 0.035 | 0.149 ± 0.025 | 0.114 ± 0.028 | 0.152 ± 0.063 |
| ścieżka dotyka testu · hop | 0.341 ± 0.087 | 0.253 ± 0.076 | 0.254 ± 0.080 | 0.289 ± 0.093 | 0.096 ± 0.046 | 0.124 ± 0.034 | 0.121 ± 0.058 | 0.177 ± 0.069 |
| ścieżka dotyka testu · nn | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

Złożenia osobno (ścieżka treningowa, operator, hit@1):

| test | lin:most | lin:softmax | lin:zero | lin:mlp | trans:most | trans:softmax | trans:zero | trans:mlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dół∘dół | 0.882 ± 0.041 | 0.585 ± 0.224 | 0.541 ± 0.077 | 0.561 ± 0.092 | 0.284 ± 0.177 | 0.179 ± 0.080 | 0.177 ± 0.067 | 0.195 ± 0.079 |
| prawo∘dół | 0.638 ± 0.103 | 0.540 ± 0.074 | 0.455 ± 0.122 | 0.461 ± 0.112 | 0.196 ± 0.073 | 0.281 ± 0.034 | 0.247 ± 0.085 | 0.309 ± 0.042 |
| dół∘prawo | 0.744 ± 0.072 | 0.659 ± 0.074 | 0.574 ± 0.142 | 0.599 ± 0.093 | 0.234 ± 0.084 | 0.294 ± 0.109 | 0.304 ± 0.101 | 0.389 ± 0.059 |

Liczba zapytań B (seed 0): treningowe=52, dotykające testu=138.

## B — MRR (złożony operator, ścieżka dotyka testu)

| test | lin:most | lin:softmax | lin:zero | lin:mlp | trans:most | trans:softmax | trans:zero | trans:mlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MRR | 0.476 ± 0.067 | 0.468 ± 0.098 | 0.475 ± 0.061 | 0.442 ± 0.079 | 0.226 ± 0.039 | 0.311 ± 0.053 | 0.290 ± 0.058 | 0.315 ± 0.066 |

## C. Przemienność operatorów liniowych ‖M_d M_p − M_p M_d‖ / ‖M_d M_p‖ (0 = siatka)

| most | softmax | zero | mlp |
|---:|---:|---:|---:|
| 0.281 ± 0.009 | 0.254 ± 0.016 | 0.253 ± 0.014 | 0.399 ± 0.041 |

## Most vs reszta (B dotyka testu, operator, hit@1 — wygrane seedy)

| operatory | vs softmax | vs zero | vs mlp |
|---|---:|---:|---:|
| lin | 3/5 | 3/5 | 4/5 |
| trans | 0/5 | 1/5 | 1/5 |

## D. Pytania Mendelejewa (Most, operator liniowy) — pozycja prawdziwej odpowiedzi

Średnia pozycja po seedach (1 = trafienie; losowo ≈ 59) · top-3 dla seed 0 · czy głowa była treningowa (seed 0)

| pytanie | średnia pozycja | trafień 1. | top-3 (seed 0) | głowa w treningu |
|---|---:|---:|---|:-:|
| dół(Al)=Ga | 33.2 | 0/5 | Si, B, He | tak |
| dół(Si)=Ge | 23.0 | 1/5 | P, Al, N | nie |
| dół(Mn)=Tc | 9.4 | 0/5 | Mo, Ru, Nb | nie |
| dół∘dół(B)=Ga | 35.8 | 0/5 | Al, Re, Si | tak |
| dół∘dół(C)=Ge | 25.6 | 1/5 | N, Al, Si | tak |
| prawo(Zn)=Ga | 80.6 | 0/5 | Cu, Cd, Ni | tak |
| prawo(Ga)=Ge | 1.0 | 5/5 | Ge, As, Sn | nie |
| prawo(Ca)=Sc | 1.0 | 5/5 | Sc, Ti, Y | tak |
| prawo∘dół(Mg)=Sc | 7.6 | 0/5 | K, Ar, Ca | nie |
| dół∘prawo(Al)=Ge | 31.0 | 0/5 | Pb, C, Os | tak |

> **KOREKTA (2026-10-02, czysta ablacja jądra — [`MENDELEJEW_KERNEL_ABLATION.md`](MENDELEJEW_KERNEL_ABLATION.md)):** bliźniak Softmax w tym raporcie nie dostawał informacji z tracera (ΔE), więc porównanie Most vs Softmax mierzy „jądro + ΔE vs brak ΔE”. Przy tej samej informacji Softmax dorównuje Mostowi w kompozycji na znanych pierwiastkach; czysty zysk jądra BW występuje tylko w wariancie comp+rel·PoE na nowych pierwiastkach (+0.07…+0.11).

## Wnioski

**Zastrzeżenia:** 5 seedów. Zapytań złożonych z samymi pierwiastkami treningowymi jest ~50 na seed, dotykających testu ~140. „Prawo” (Z+1) jest łatwe, bo Z jest wprost w τ. Właściwy test to „dół”.

1. **Złożenia działają: nigdy nieuczone dwa kroki trafiają, gdy pierwiastki są znane.** Ścieżka w całości treningowa, operator liniowy:

   | | Most | Softmax | ZERO | MLP |
   |---|---:|---:|---:|---:|
   | złożony operator (B-op) | **0.74** | 0.60 | 0.52 | 0.54 |
   | dwa kroki (B-hop) | **0.93** | 0.59 | 0.64 | 0.75 |
   | dół∘dół (operator) | **0.88** | 0.59 | 0.54 | 0.56 |

   Most wygrywa **5/5 seedów z każdym modelem** w B-op i B-hop (dół∘dół: 4/5 vs Softmax, 5/5 vs ZERO i MLP). Operator „dół” złożony sam ze sobą (M_d·M_d) trafia „dwa okresy niżej” w 88%, choć nigdy nie był tego uczony. To najwyraźniejsza przewaga Mostu w całej serii. Przy samym podobieństwie (`nn`) ruch „w dół” daje **0** w każdym wariancie: bez operatora relacji pionowej nie ma w ogóle.
2. **Wąskim gardłem jest indukcja na nowe pierwiastki, nie kompozycja.** Pojedynczy krok „dół” z pierwiastkiem testowym daje 0.25–0.34 (Most 0.33, 3/5 vs Softmax, czyli bez efektu). Złożenia dotykające testu: 0.23–0.34. Najlepszy jest Most z dwoma krokami (0.34, 4/5 vs Softmax i ZERO, 5/5 vs MLP), ale przy dużym rozrzucie. Na znanych pierwiastkach operatory składają się prawie bezbłędnie (fakty treningowe: Most 0.97). Na nowych błąd pierwszego kroku przenosi się dalej.
3. **Operator liniowy ≫ translacja (TransE).** Translacja przegrywa wszędzie (B-op 0.23–0.31 wobec 0.52–0.74). „Dół” nie jest stałym przesunięciem, bo jego wielkość zależy od okresu (8, 18, 32 elementy), a macierz to unosi. Most z TransE wypada najgorzej spośród modeli.
4. **To nie jest siatka.** Względna nieprzemienność ‖M_d M_p − M_p M_d‖/‖M_d M_p‖ wynosi 0.25–0.40. „Prawo” przechodzi przez granice okresów (Ne → Na) i bloków (Zn → Ga), więc dół∘prawo ≠ prawo∘dół z natury danych. MLP jest najdalej od siatki (0.40).
5. **Pytania Mendelejewa (Most, operator liniowy, średnia pozycja z 5 seedów, losowo ≈ 59):**
   - „prawo” trafia zawsze: Ca → Sc 5/5, Ga → Ge 5/5. Wyjątek: **Zn → Ga, pozycja 81**, czyli gorzej niż losowo. Model przewiduje Cu, Cd, Ni, więc przejście z bloku d do p na granicy podpowłoki jest dla niego „skokiem”.
   - „dół” prawie nigdy nie trafia: pod Al → Ga (poz. 33, top: Si, B, He), pod Si → Ge (23, 1/5), pod Mn → Tc (**9.4**, top: Mo, Ru, Nb, czyli sąsiedzi Tc w tablicy).
   - Złożenia z luką: dół∘dół(B) → Ga (36), prawo∘dół(Mg) → Sc (7.6, top: K, Ar, Ca), dół∘prawo(Al) → Ge (31).

   Model „wie”, w którą okolicę tablicy iść (Tc ≈ Mo/Ru/Nb), ale nie trafia w sam pierwiastek, gdy ten był nieznany.

## Werdykt

Relacje kompozycyjne działają **systematycznie na znanych pierwiastkach**. Wyuczone osobno operatory „dół” i „prawo” składają się w nigdy nieuczone ruchy po tablicy z trafnością 0.74–0.93, a **Most jest tu najlepszy 5/5 wobec Softmax, ZERO i MLP**. Na nowych pierwiastkach (luki Mendelejewa) kompozycja jest tylko tak dobra jak pierwszy krok indukcji (~0.3). Model trafia w okolicę (Tc → Mo/Ru/Nb), a nie w sam pierwiastek.

**Kierunki:**
1. Połączyć z REL: głowica „grupa” plus operatory, czyli lepsza reprezentacja pionowa dla indukcji pierwszego kroku.
2. Operator odwrotny „góra” (M_d⁻¹ albo osobno uczony) i spójność cyklu M_g M_d ≈ I jako regularyzacja.
3. Uczyć „prawo” tylko wewnątrz bloku (bez skoków d → p), żeby sprawdzić, czy przemienność, czyli siatka, wtedy się pojawia.

