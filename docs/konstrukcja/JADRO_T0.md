# T0 — margines ±2 punkty na porównaniach z tracerem w tokenach

Bez nowego treningu. Te same 30 porównań co TOST z 2 października (`bw·tok − sm·tok` i `bw·tok+ΔE − sm·tok+ΔE`, układ okresowy, protokół transduktywny, 3 warianty × 5 miar, 20 przebiegów). 90% CI różnicy hit@1, bootstrap 10 000, ziarno 4242, jak w `archiwum/skrypty/analyze_kernel_equivalence.py`.

Próg **±2 punkty** (±0.02) jest progim T0 z `JADRO_OGOLNE.md`, ustalonym przed tym rachunkiem. Równoważne = cały przedział mieści się w progu. Kontrola: ten sam CI przy ±5 punktach daje 26/30, zgodnie z `MENDELEJEW_KERNEL_EQUIV.md`.

| próg | w marginesie | wychodzi poza próg | nierozstrzygnięte | razem |
|---|---:|---:|---:|---:|
| ±5 punktów (kontrola) | 26 | 0 | 4 | 30 |
| ±2 punkty (T0) | 7 | 0 | 23 | 30 |

„Wychodzi” = cały przedział leży poza progiem (różnica większa niż próg). „Nierozstrzygnięte” = przedział nie mieści się w progu i nie leży cały poza nim.

## Trzydzieści porównań

| porównanie | wariant | miara | Δ | 90% CI | ±5 | ±2 |
|---|---|---|---:|---|---|---|
| bw·tok − sm·tok | comp | A_dół | +0.007 | [-0.012, +0.024] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp | B_te_hop | +0.001 | [-0.011, +0.012] | równoważne | równoważne |
| bw·tok − sm·tok | comp | B_te_dd | +0.005 | [-0.010, +0.018] | równoważne | równoważne |
| bw·tok − sm·tok | comp | B_tr_op | -0.003 | [-0.015, +0.007] | równoważne | równoważne |
| bw·tok − sm·tok | comp | B_tr_dd | -0.003 | [-0.022, +0.017] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp+mix | A_dół | +0.004 | [-0.014, +0.023] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp+mix | B_te_hop | -0.003 | [-0.017, +0.012] | równoważne | równoważne |
| bw·tok − sm·tok | comp+mix | B_te_dd | -0.006 | [-0.027, +0.014] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp+mix | B_tr_op | -0.002 | [-0.015, +0.012] | równoważne | równoważne |
| bw·tok − sm·tok | comp+mix | B_tr_dd | -0.016 | [-0.045, +0.016] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp+rel·PoE | A_dół | +0.007 | [-0.007, +0.021] | równoważne | nierozstrzygnięte |
| bw·tok − sm·tok | comp+rel·PoE | B_te_hop | +0.006 | [-0.005, +0.016] | równoważne | równoważne |
| bw·tok − sm·tok | comp+rel·PoE | B_te_dd | -0.002 | [-0.019, +0.016] | równoważne | równoważne |
| bw·tok − sm·tok | comp+rel·PoE | B_tr_op | +0.028 | [+0.013, +0.043] | równoważne | Most lepszy, niepewne czy > margines |
| bw·tok − sm·tok | comp+rel·PoE | B_tr_dd | +0.025 | [-0.002, +0.052] | nierozstrzygnięte | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp | A_dół | +0.006 | [-0.022, +0.034] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp | B_te_hop | +0.027 | [+0.009, +0.045] | równoważne | Most lepszy, niepewne czy > margines |
| bw·tok+ΔE − sm·tok+ΔE | comp | B_te_dd | +0.019 | [-0.005, +0.042] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp | B_tr_op | -0.004 | [-0.021, +0.013] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp | B_tr_dd | +0.020 | [-0.015, +0.053] | nierozstrzygnięte | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+mix | A_dół | -0.003 | [-0.033, +0.027] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+mix | B_te_hop | +0.004 | [-0.023, +0.030] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+mix | B_te_dd | +0.019 | [-0.008, +0.044] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+mix | B_tr_op | -0.009 | [-0.026, +0.009] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+mix | B_tr_dd | +0.010 | [-0.038, +0.057] | nierozstrzygnięte | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | A_dół | +0.017 | [-0.005, +0.039] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | B_te_hop | +0.011 | [-0.007, +0.028] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | B_te_dd | -0.023 | [-0.050, +0.005] | nierozstrzygnięte | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | B_tr_op | -0.004 | [-0.028, +0.021] | równoważne | nierozstrzygnięte |
| bw·tok+ΔE − sm·tok+ΔE | comp+rel·PoE | B_tr_dd | +0.008 | [-0.031, +0.046] | równoważne | nierozstrzygnięte |

## Wniosek

Zdanie „wzory są zamienne” obowiązuje przy ±5 punktach, nie przy ±2.

W marginesie ±2 zostaje 7 z 30 porównań. To wyraźnie mniej niż 26/30 przy ±5, więc odczyt zapisany w programie jest spełniony. Żadne z 30 nie wychodzi całe poza ±2: nie ma tu dowodu, że jądra różnią się o więcej niż 2 punkty. 23 porównania są nierozstrzygnięte, bo przedział z 20 przebiegów jest szerszy niż ±2. Średnie różnice są małe (największa +0,028).

Dwa porównania mają przedział cały po stronie Mostu i sięgają poza +2 punkty, ale nie leżą całe poza progiem: złożenie na znanych przy PoE (`bw·tok − sm·tok`, Δ = +0,028, CI [+0,013, +0,043]) oraz hop na nowych przy samym składaniu z ΔE w tokenie (Δ = +0,027, CI [+0,009, +0,045]). Przy ±5 oba były równoważne.

Ciaśniejszej zgody te przebiegi nie udźwigną. Więcej seedów byłoby nowym treningiem, a T0 go nie przewiduje.

