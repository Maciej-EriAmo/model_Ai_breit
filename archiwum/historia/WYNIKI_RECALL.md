# Wyniki Faza C — recall / ranking Φ

Pozytyw: ten sam **topic** + ta sama **depth** (bind w warstwie bańki).
N=24; hard negs = ten sam topic, inna depth.
Loss neural: InfoNCE; skalarne R = g·V²/(g²+dE²).
Gate: **r@1** (r@3 raportowane — Softmax często saturuje top-3 same-topic).
P2.13: T/state/mazur **dekorelowane** od depth.

Korelacja osi (val[0]): |corr|(depth,T)=0.136, |corr|(depth,mazur)=0.076, |corr|(depth,state)=0.433

| wariant | r@1 | r@3 | r@5 | MRR |
|---------|-----|-----|-----|-----|
| Softmax | 0.312 | 0.737 | 0.998 | 0.552 |
| Most | 0.985 | 0.998 | 1.000 | 0.991 |
| Most ZERO | 0.304 | 0.725 | 0.997 | 0.542 |
| Most −depth | 0.453 | 0.810 | 0.991 | 0.648 |
| Most −T | 0.304 | 0.725 | 0.997 | 0.542 |
| Most −mazur | 0.995 | 1.000 | 1.000 | 0.997 |
| flat payload | 0.087 | 0.245 | 0.388 | 0.239 |
| R (depth dE) | 0.984 | 1.000 | 1.000 | 0.992 |
| R (tracer L2) | 0.869 | 0.999 | 1.000 | 0.928 |
| cosine only | 0.202 | 0.606 | 0.980 | 0.457 |

**Δ Most−Softmax r@1** = +0.6727 → PASS
**Δ Most−R r@1** = +0.0008 → PASS
(info) Δ Most−Softmax r@3 = +0.2617
**Werdykt:** **PASS**

## Ablacje
- Most ZERO / flat payload: kontrola obserwatora i „płaskiego pliku”.
- Most −depth / −T / −mazur: osie Φ niezależne (po dekorelacji).

## Go / no-go live Store
GO na Faza D (offline dump), potem live read-only — Most wnosi geometrię Φ.
