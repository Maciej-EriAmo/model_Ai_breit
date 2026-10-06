# Mendelejew FULL (118) — MOST-ONLY

N=118 edges=328 · epochs=64 · seeds=[0, 1, 17] · mode=MOST-ONLY

Domowa konstrukcja Most dorównała Softmaxowi na P0 Mendelejew (Most≈0.43 vs SM≈0.45, nonflat win) — parity z kanonem; FULL118 = kompetencja Mosta, nie re-bench vs Softmax.

**Werdykt:** FULL118 MOST-ONLY mean r@1=0.575 r@3=0.892 mrr=0.739 above_floor=3/3 (parity vs Softmax już na P0: Most≈0.43 SM≈0.45 — bez re-porównania)

| seed | Most r@1 | r@3 | mrr | floor |
|-----:|---------:|----:|----:|:-----:|
| 0 | 0.581 | 0.895 | 0.745 | Y |
| 1 | 0.565 | 0.876 | 0.730 | Y |
| 17 | 0.578 | 0.904 | 0.741 | Y |

JSON: `mendelejew_full_results.json`
