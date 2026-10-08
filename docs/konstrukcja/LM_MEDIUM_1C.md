# Medium LM — kryterium #1c (~2–5M params)

preset=medium · d=256 L=4 h=4 vocab=256
seq=128 batch=8 steps=600 seeds=[0, 1]
params Most/SM = **3287684** / **3286528**
data_bytes = 1267399

**Werdykt:** #1c medium LM: params=3287684 | mean loss Most=2.416 SM=2.353 Δ(SM−Most)=-0.064 | time≈1.32× | descend=True

Skala medium (~3M), nie gate 100–300M vs Llama. Ten sam korpus/pipeline co #1b; #3 VRAM nadal wymaga NVIDIA.

| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |
|-----:|--------------|------------|-----------:|-------:|-----:|------:|
| 0 | 167.447→2.092 | 167.120→2.108 | +0.016 | 62.8 | 46.9 | 1.34 |
| 1 | 167.939→2.741 | 167.616→2.597 | -0.144 | 60.2 | 46.4 | 1.30 |

## Aggregate

- mean final loss Most/SM = **2.416** / **2.353**
- mean Δ(SM−Most) = **-0.064**
- mean time ratio = **1.32×**
- both descend = **True**

elapsed=216s · run-dir=`C:\Users\drwis\model_Ai_breit\runs\20261006_061043_lm_medium_1c`

JSON: `LM_MEDIUM_1C_results.json`
