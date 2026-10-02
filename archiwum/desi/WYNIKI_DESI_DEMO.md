# WYNIKI DESI (patchy cutouts) — po audycie

**Kanon:** tracer = obserwator gwarantujący prawidłowy most.
Główny warunek = **Most vs ZERO**. Softmax = twin bez obserwatora.

- Softmax acc=0.7027 loss=0.7071  (brak τ)
- Most     acc=0.7838 loss=0.6163 |dE|=0.5024
- ZERO     acc=0.6757 loss=0.8232  (Most z τ=0)
- **Δ(Most−ZERO)=+0.1081**  gate=0.03 → most prawidłowy? **PASS**
- Δ(Most−SM)=+0.0811  gate=0.03 → vs Softmax **PASS**

- size=128 N=64 train_files=112 val_files=37 (augment tylko train; maska w uwadze)
- geometry_only=False demo_only=True

Most<SM nie znaczy „tracer zbędny” — Softmax nie formuje mostu.
