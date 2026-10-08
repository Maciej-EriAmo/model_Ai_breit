# Medium LM — kryterium #1c (~2–5M params)

preset=medium · d=256 L=4 h=4 vocab=256
seq=128 batch=8 steps=10685 seeds=[0, 1]
corpus=dane\wikitext-2-raw-v1\wikitext-2-raw\wiki.train.raw
params Most/SM = **3287684** / **3286528**
data_bytes = 10940747

**Werdykt:** #1c medium LM: params=3287684 | mean loss Most=1.554 SM=1.587 Δ(SM−Most)=+0.033 | time≈1.44× | descend=True

Skala medium (~3M), nie gate 100–300M vs Llama. Korpus z --corpus. Valid i test to średnia strata na całym pliku, okna bez zachodzenia.

| seed | Most loss₀→T | SM loss₀→T | Δ(SM−Most) | Most s | SM s | ratio |
|-----:|--------------|------------|-----------:|-------:|-----:|------:|
| 0 | 170.377→1.555 | 170.533→1.573 | +0.018 | 1422.0 | 1054.0 | 1.35 |
| 1 | 170.323→1.554 | 169.846→1.601 | +0.048 | 1302.2 | 853.1 | 1.53 |

## Aggregate

- mean final loss Most/SM = **1.554** / **1.587**
- mean Δ(SM−Most) = **+0.033**
- mean time ratio = **1.44×**
- both descend = **True**

## Valid

- mean loss Most/SM = **1.557** / **1.604**
- mean Δ(SM−Most) = **+0.047**

| seed | Most | SM | Δ(SM−Most) | ppl Most | ppl SM |
|-----:|-----:|---:|-----------:|---------:|-------:|
| 0 | 1.571 | 1.600 | +0.029 | 4.81 | 4.95 |
| 1 | 1.544 | 1.609 | +0.065 | 4.68 | 5.00 |

## Test

- mean loss Most/SM = **1.580** / **1.627**
- mean Δ(SM−Most) = **+0.047**

| seed | Most | SM | Δ(SM−Most) | ppl Most | ppl SM |
|-----:|-----:|---:|-----------:|---------:|-------:|
| 0 | 1.593 | 1.623 | +0.031 | 4.92 | 5.07 |
| 1 | 1.567 | 1.631 | +0.063 | 4.79 | 5.11 |

elapsed=4913s · run-dir=`C:\Users\drwis\model_Ai_breit\runs\20261006_080933_lm_medium_wikitext_holdout`

JSON: `LM_MEDIUM_WIKITEXT_HOLDOUT_results.json`
