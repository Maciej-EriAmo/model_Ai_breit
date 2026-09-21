# Medium LM — kryterium #1c (~2–5M params)

preset=medium · d=256 L=4 h=4 vocab=256
seq=128 batch=8 steps=600 seeds=[0, 1]
params Most/SM = **3287668** / **3286528**
data_bytes = 691696

**Werdykt:** #1c medium LM: params=3287668 | mean loss Most=2.715 SM=2.657 Δ(SM−Most)=-0.058 | time≈1.34× | descend=True

Skala medium (~3M), nie gate 100–300M vs Llama. Ten sam korpus/pipeline co #1b; #3 VRAM nadal wymaga NVIDIA.

| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |
|-----:|--------------|------------|-----------:|-------:|-----:|------:|
| 0 | 166.237→2.819 | 165.913→2.707 | -0.112 | 75.4 | 52.5 | 1.43 |
| 1 | 164.858→2.612 | 164.267→2.607 | -0.005 | 67.3 | 54.5 | 1.24 |

## Aggregate

- mean final loss Most/SM = **2.715** / **2.657**
- mean Δ(SM−Most) = **-0.058**
- mean time ratio = **1.34×**
- both descend = **True**

elapsed=250s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_082007_lm_medium_1c`

JSON: `LM_MEDIUM_1C_results.json`
