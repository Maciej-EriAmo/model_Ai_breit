# Medium LM — kryterium #1c (~2–5M params)

preset=medium · d=256 L=4 h=4 vocab=256
seq=128 batch=8 steps=10685 seeds=[0, 1]
corpus=C:\Users\drwis\model_Ai_breit\dane\wikitext-2-raw-v1\wikitext-2-raw\wiki.train.raw
params Most/SM = **3287684** / **3286528**
data_bytes = 10940747

**Werdykt:** #1c medium LM: params=3287684 | mean loss Most=1.554 SM=1.587 Δ(SM−Most)=+0.033 | time≈1.14× | descend=True

Skala medium (~3M), nie gate 100–300M vs Llama. Korpus to trening WikiText-2, jeden przejazd na seed. Strata jest treningowa, nie na pliku valid/test.

| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |
|-----:|--------------|------------|-----------:|-------:|-----:|------:|
| 0 | 170.377→1.555 | 170.533→1.573 | +0.018 | 1258.8 | 1179.7 | 1.07 |
| 1 | 170.323→1.554 | 169.846→1.601 | +0.048 | 1496.6 | 1231.4 | 1.22 |

## Aggregate

- mean final loss Most/SM = **1.554** / **1.587**
- mean Δ(SM−Most) = **+0.033**
- mean time ratio = **1.14×**
- both descend = **True**

elapsed=5167s · run-dir=`C:\Users\drwis\model_Ai_breit\runs\20261006_063037_lm_medium_wikitext`

JSON: `LM_MEDIUM_WIKITEXT_results.json`
