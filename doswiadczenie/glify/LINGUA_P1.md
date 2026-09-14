# Lingua P1 — czy Most uczy się recall na karmazyn_lingua?

Seedy: [0, 1, 17, 23, 42] · epochs=16 · encodings=['hologram', 'symbolic', 'onehot']

Pierwszy P-numerowany eksperyment dla `karmazyn_lingua.py`: dotąd `to_phi_atoms`/`make_dump_lingua.py` był ślepym eksporterem — nikt nie sprawdził, czy silnik Most (phimost_learn/recall_model) faktycznie wyciąga coś z tej reprezentacji, ani czy hologram/symbolic/onehot różnią się z punktu widzenia Mostu (nie tylko gołego nearest-neighbour r@1 z demo w karmazyn_lingua.py).

## Agregat per encoding

| encoding | atoms | PASS | mean ΔSM | best ΔSM | worst ΔSM | mean Most | mean SM | mean ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hologram | 11 | 0/5 | -0.0029 | +0.0036 | -0.0109 | 0.245 | 0.248 | 0.247 |
| symbolic | 11 | 0/5 | -0.0051 | +0.0036 | -0.0109 | 0.243 | 0.248 | 0.247 |
| onehot | 11 | 0/5 | -0.0065 | +0.0073 | -0.0182 | 0.241 | 0.248 | 0.247 |

**Najlepszy encoding (PASS, potem meanΔ):** **hologram**

## Pełna tabela

| encoding | seed | Most | SM | ZERO | ΔSM | gate |
|---|---:|---:|---:|---:|---:|:----:|
| hologram | 0 | 0.215 | 0.218 | 0.222 | -0.0036 | FAIL |
| hologram | 1 | 0.298 | 0.298 | 0.298 | -0.0000 | FAIL |
| hologram | 17 | 0.240 | 0.244 | 0.244 | -0.0036 | FAIL |
| hologram | 23 | 0.229 | 0.240 | 0.236 | -0.0109 | FAIL |
| hologram | 42 | 0.244 | 0.240 | 0.236 | +0.0036 | FAIL |
| symbolic | 0 | 0.215 | 0.218 | 0.222 | -0.0036 | FAIL |
| symbolic | 1 | 0.295 | 0.298 | 0.298 | -0.0036 | FAIL |
| symbolic | 17 | 0.233 | 0.244 | 0.244 | -0.0109 | FAIL |
| symbolic | 23 | 0.229 | 0.240 | 0.236 | -0.0109 | FAIL |
| symbolic | 42 | 0.244 | 0.240 | 0.236 | +0.0036 | FAIL |
| onehot | 0 | 0.222 | 0.218 | 0.222 | +0.0036 | FAIL |
| onehot | 1 | 0.284 | 0.298 | 0.298 | -0.0145 | FAIL |
| onehot | 17 | 0.225 | 0.244 | 0.244 | -0.0182 | FAIL |
| onehot | 23 | 0.229 | 0.240 | 0.236 | -0.0109 | FAIL |
| onehot | 42 | 0.247 | 0.240 | 0.236 | +0.0073 | FAIL |

**Werdykt:** best encoding=hologram PASS 0/5 meanΔSM=-0.0029; gate = Most r@1 - Softmax r@1 >= CANON.gate_recall_r1 (0.02)

## Uwaga: prawdopodobny confound — graf za mały

We wszystkich trzech encodingach `Most ≈ SM ≈ ZERO` (różnice rzędu 0.003,
znacznie poniżej bramki 0.02) — nie tylko Most nie bije Softmax, ale Most
z wyzerowanym tracerem (`zero_tracer` ablation) daje PRAWIE TEN SAM wynik
co pełny Most. To sugeruje, że tracer (mechanizm, na którym w ogóle opiera
się przewaga Mostu nad Softmax w innych benchach tego repo) nie ma tu
czego uchwycić — graf ma tylko 11 atomów (3 słowa × ~4 węzły), więc
mechanizm uwagi ma za mało sygnału/przykładów, żeby się odróżnić od
gołego Softmax. To NIE jest odpowiedź na pytanie "czy hologram/symbolic/
onehot różnią się dla Mostu" (za mało mocy statystycznej, żeby to pytanie
w ogóle zadać na tej skali) — to jest odpowiedź na pytanie "czy ten
konkretny mały graf demo wystarcza do testowania Mostu" (nie wystarcza).
Następny krok, jeśli warto go zrobić: większy korpus morfemów (dziesiątki
słów, nie 3) zanim ΔSM cokolwiek powie o encoding.

JSON: `lingua_p1_results.json`

