# Czy Most może zastąpić Softmax? Test równoważności (TOST)

Bez nowego treningu. Pary ramion różniące się **wyłącznie jądrem** (ta sama informacja, parametry, procedura). Równoważne = 90% CI różnicy hit@1 (bootstrap po 20 przebiegach) w całości w ±0.05 (5 punktów procentowych; TOST, α = 0.05 na stronę; margines ustalony przed analizą). Skrypt `archiwum/skrypty/analyze_kernel_equivalence.py`.

Kategorie: **równoważne** · **nierozstrzygnięte** (CI obejmuje 0 i wychodzi poza margines — za mało danych) · **X lepszy, niepewne czy > margines** (CI po jednej stronie zera, ale sięga marginesu) · **X lepszy (> margines)**.

## Podsumowanie

| dane | porównanie | równoważne | nierozstrz. | Most lepszy* | Softmax lepszy* | razem |
|---|---|---:|---:|---:|---:|---:|
| periodic-transd | bw·0 − sm·0 | 9 | 4 | 1 | 1 | 15 |
| periodic-transd | bw·ΔE − sm·ΔE | 2 | 3 | 8 | 2 | 15 |
| periodic-transd | bw·tok − sm·tok | 14 | 1 | 0 | 0 | 15 |
| periodic-transd | bw·tok+ΔE − sm·tok+ΔE | 12 | 3 | 0 | 0 | 15 |
| periodic | bw·0 − sm·0 | 11 | 3 | 1 | 0 | 15 |
| periodic | bw·ΔE − sm·ΔE | 9 | 2 | 0 | 4 | 15 |
| kana | bw·0 − sm·0 | 11 | 4 | 0 | 0 | 15 |
| kana | bw·ΔE − sm·ΔE | 11 | 4 | 0 | 0 | 15 |
| music | bw·0 − sm·0 | 14 | 0 | 0 | 1 | 15 |
| music | bw·ΔE − sm·ΔE | 7 | 1 | 0 | 7 | 15 |
| **razem** | | **100** | 25 | 10 | 15 | 150 |

\* łącznie z „niepewne czy > margines”.

## Porównania, które NIE są równoważne

| dane | porównanie | wariant | miara | Δ | 90% CI | werdykt |
|---|---|---|---|---:|---|---|
| periodic-transd | bw·0 − sm·0 | comp | B_te_dd | +0.042 | [+0.020, +0.064] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·0 − sm·0 | comp | B_tr_dd | -0.016 | [-0.051, +0.021] | nierozstrzygnięte |
| periodic-transd | bw·0 − sm·0 | comp+mix | B_tr_op | -0.026 | [-0.055, +0.003] | nierozstrzygnięte |
| periodic-transd | bw·0 − sm·0 | comp+mix | B_tr_dd | +0.000 | [-0.050, +0.053] | nierozstrzygnięte |
| periodic-transd | bw·0 − sm·0 | comp+rel·PoE | B_te_dd | -0.024 | [-0.057, +0.008] | nierozstrzygnięte |
| periodic-transd | bw·0 − sm·0 | comp+rel·PoE | B_tr_dd | -0.043 | [-0.085, -0.003] | Softmax lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp | A_dół | +0.034 | [+0.008, +0.061] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp | B_te_hop | +0.033 | [+0.013, +0.053] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp | B_te_dd | +0.052 | [+0.020, +0.083] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp | B_tr_dd | -0.060 | [-0.102, -0.022] | Softmax lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp+mix | A_dół | +0.033 | [-0.007, +0.072] | nierozstrzygnięte |
| periodic-transd | bw·ΔE − sm·ΔE | comp+mix | B_te_hop | +0.042 | [+0.012, +0.071] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp+mix | B_te_dd | +0.051 | [+0.017, +0.087] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp+mix | B_tr_op | -0.051 | [-0.084, -0.016] | Softmax lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp+mix | B_tr_dd | -0.014 | [-0.055, +0.028] | nierozstrzygnięte |
| periodic-transd | bw·ΔE − sm·ΔE | comp+rel·PoE | A_dół | +0.088 | [+0.071, +0.105] | Most lepszy (> margines) |
| periodic-transd | bw·ΔE − sm·ΔE | comp+rel·PoE | B_te_hop | +0.067 | [+0.050, +0.084] | Most lepszy, niepewne czy > margines |
| periodic-transd | bw·ΔE − sm·ΔE | comp+rel·PoE | B_te_dd | +0.109 | [+0.082, +0.136] | Most lepszy (> margines) |
| periodic-transd | bw·ΔE − sm·ΔE | comp+rel·PoE | B_tr_dd | -0.025 | [-0.060, +0.011] | nierozstrzygnięte |
| periodic-transd | bw·tok − sm·tok | comp+rel·PoE | B_tr_dd | +0.025 | [-0.002, +0.052] | nierozstrzygnięte |
| periodic-transd | bw·tok+ΔE − sm·tok+ΔE | comp | B_tr_dd | +0.020 | [-0.015, +0.053] | nierozstrzygnięte |
| periodic-transd | bw·tok+ΔE − sm·tok+ΔE | comp+mix | B_tr_dd | +0.010 | [-0.038, +0.057] | nierozstrzygnięte |
| periodic-transd | bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | B_te_dd | -0.023 | [-0.050, +0.005] | nierozstrzygnięte |
| periodic | bw·0 − sm·0 | comp | B_tr_dd | -0.024 | [-0.067, +0.020] | nierozstrzygnięte |
| periodic | bw·0 − sm·0 | comp+mix | B_tr_op | +0.030 | [+0.011, +0.051] | Most lepszy, niepewne czy > margines |
| periodic | bw·0 − sm·0 | comp+mix | B_tr_dd | +0.018 | [-0.017, +0.057] | nierozstrzygnięte |
| periodic | bw·0 − sm·0 | comp+rel·PoE | B_tr_dd | -0.033 | [-0.069, +0.003] | nierozstrzygnięte |
| periodic | bw·ΔE − sm·ΔE | comp | B_tr_op | -0.069 | [-0.105, -0.034] | Softmax lepszy, niepewne czy > margines |
| periodic | bw·ΔE − sm·ΔE | comp | B_tr_dd | -0.095 | [-0.176, -0.016] | Softmax lepszy, niepewne czy > margines |
| periodic | bw·ΔE − sm·ΔE | comp+mix | B_tr_dd | +0.027 | [-0.028, +0.077] | nierozstrzygnięte |
| periodic | bw·ΔE − sm·ΔE | comp+rel·PoE | B_te_dd | +0.006 | [-0.041, +0.055] | nierozstrzygnięte |
| periodic | bw·ΔE − sm·ΔE | comp+rel·PoE | B_tr_op | -0.103 | [-0.141, -0.062] | Softmax lepszy (> margines) |
| periodic | bw·ΔE − sm·ΔE | comp+rel·PoE | B_tr_dd | -0.107 | [-0.181, -0.035] | Softmax lepszy, niepewne czy > margines |
| kana | bw·0 − sm·0 | comp | B_tr_dd | -0.031 | [-0.075, +0.013] | nierozstrzygnięte |
| kana | bw·0 − sm·0 | comp+mix | B_tr_op | +0.019 | [-0.018, +0.060] | nierozstrzygnięte |
| kana | bw·0 − sm·0 | comp+mix | B_tr_dd | +0.031 | [-0.020, +0.083] | nierozstrzygnięte |
| kana | bw·0 − sm·0 | comp+rel·PoE | B_tr_dd | +0.012 | [-0.040, +0.065] | nierozstrzygnięte |
| kana | bw·ΔE − sm·ΔE | comp | B_tr_dd | -0.003 | [-0.065, +0.059] | nierozstrzygnięte |
| kana | bw·ΔE − sm·ΔE | comp+mix | B_tr_dd | +0.003 | [-0.064, +0.069] | nierozstrzygnięte |
| kana | bw·ΔE − sm·ΔE | comp+rel·PoE | B_tr_op | +0.020 | [-0.014, +0.052] | nierozstrzygnięte |
| kana | bw·ΔE − sm·ΔE | comp+rel·PoE | B_tr_dd | +0.034 | [-0.035, +0.101] | nierozstrzygnięte |
| music | bw·0 − sm·0 | comp+mix | B_tr_dd | -0.043 | [-0.067, -0.019] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp | A_dół | -0.033 | [-0.067, -0.000] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp | B_te_hop | -0.045 | [-0.071, -0.021] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp | B_tr_dd | -0.059 | [-0.120, -0.007] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp+mix | B_te_dd | -0.028 | [-0.053, -0.002] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp+mix | B_tr_dd | -0.033 | [-0.069, +0.003] | nierozstrzygnięte |
| music | bw·ΔE − sm·ΔE | comp+rel·PoE | A_dół | -0.055 | [-0.081, -0.030] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp+rel·PoE | B_te_hop | -0.049 | [-0.070, -0.029] | Softmax lepszy, niepewne czy > margines |
| music | bw·ΔE − sm·ΔE | comp+rel·PoE | B_te_dd | -0.032 | [-0.051, -0.013] | Softmax lepszy, niepewne czy > margines |

## Wnioski

**Zastrzeżenia:**
- Margines ±5 pkt proc. jest arbitralny, choć ustalony przed analizą.
- 20 przebiegów to mało dla wąskich przedziałów. Część wyników „nierozstrzygniętych” to po prostu za mało danych, zwłaszcza dla dół∘dół na znanych, gdzie zapytań jest ~11–21 na przebieg.

1. **Gdy informacja jest podana w standardowy sposób (w tokenach), Most jest równoważny Softmaxowi.** W 26 z 30 porównań różnica mieści się w ±5 pkt proc., 4 są nierozstrzygnięte, a **żadne nie wskazuje, że Most jest gorszy**.
2. **Bez informacji ze znacznika** (`bw·0` vs `sm·0`) jądra są równoważne w 45 z 60 porównań, we wszystkich trzech dziedzinach. Nieliczne różnice są małe i idą w obie strony.
3. **Gdy informacja płynie wyłącznie przez uwagę (ΔE), jądra nie są równoważne, a kierunek zależy od warunków.**
   - Układ okresowy, protokół transduktywny: Most lepszy na nowych pierwiastkach (do +0.11).
   - Ten sam układ w ścisłym protokole oraz muzyka: Softmax lepszy (do −0.10 na znanych w układzie okresowym, −0.03…−0.06 w muzyce).

   Tu równoważności nie ma i nie da się wskazać zwycięzcy niezależnie od dziedziny.

## Werdykt

**Most może zastąpić Softmax**: przy informacji podanej w tokenach jest z nim równoważny w granicach ±5 pkt proc. (26/30, 0 gorszych). **Zastąpienie nie przynosi jednak przewagi**, a w kanonicznej postaci Mosta (informacja tylko przez ΔE) wynik zależy od dziedziny i protokołu.

Do sprawdzenia w dalszych badaniach:
- większe zadania i inne dziedziny, zwłaszcza modelowanie języka;
- węższy margines przy większej liczbie przebiegów;
- koszt obliczeń i stabilność uczenia.

