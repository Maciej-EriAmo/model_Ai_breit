# Tiny LM smoke — kryterium #1b (≪1M params)

d=64 L=2 h=2 vocab=128 seq=64 steps=300 seeds=[0, 1]
params Most/SM = **116268** / **115968**

**Werdykt:** #1b tiny LM: params Most/SM=116268/115968 (≪1M) | mean loss Most=0.637 SM=0.664 Δ(SM−Most)=+0.027 | time Most/SM≈1.76× | descend=True

To NIE jest jeszcze gate 100M–300M vs Llama. Smoke = ten sam data/pipeline, mierzalna krzywa loss Most vs Softmax na CPU.

| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |
|-----:|--------------|------------|-----------:|-------:|-----:|------:|
| 0 | 40.066→0.585 | 40.141→0.629 | +0.044 | 4.8 | 2.4 | 2.02 |
| 1 | 42.259→0.690 | 42.671→0.700 | +0.010 | 3.8 | 2.6 | 1.50 |

## Aggregate

- mean final loss Most/SM = **0.637** / **0.664**
- mean Δ(SM−Most) = **+0.027** (dodatnie → Most niższy loss)
- mean time ratio Most/SM = **1.76×**
- both descend = **True**

elapsed=14s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_081554_lm_tiny_smoke_1b`

JSON: `LM_TINY_1B_results.json`
