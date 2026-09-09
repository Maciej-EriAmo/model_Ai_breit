# Wyniki Faza C — recall / ranking Φ

Pozytyw: ten sam **topic** + ta sama **depth** (bind w warstwie bańki).
N=24; hard negs = ten sam topic, inna depth.
Loss neural: InfoNCE; skalarne R = g·V²/(g²+dE²).
Gate: **r@1** (r@3 raportowane — Softmax często saturuje top-3 same-topic).

| wariant | r@1 | r@3 | r@5 | MRR |
|---------|-----|-----|-----|-----|
| Softmax | 0.284 | 0.686 | 0.998 | 0.524 |
| Most | 0.995 | 0.999 | 1.000 | 0.997 |
| Most ZERO | 0.286 | 0.695 | 0.998 | 0.526 |
| Most −depth | 0.999 | 1.000 | 1.000 | 1.000 |
| Most −T | 1.000 | 1.000 | 1.000 | 1.000 |
| flat payload | 0.230 | 0.412 | 0.518 | 0.374 |
| R (depth dE) | 0.990 | 1.000 | 1.000 | 0.995 |
| R (tracer L2) | 0.990 | 1.000 | 1.000 | 0.995 |
| cosine only | 0.187 | 0.574 | 0.985 | 0.443 |

**Δ Most−Softmax r@1** = +0.7117 → PASS
**Δ Most−R r@1** = +0.0055 → PASS
(info) Δ Most−Softmax r@3 = +0.3133
**Werdykt:** **PASS**

## Ablacje
- Most ZERO ≈ Softmax → **obserwator konieczny** (jak w Fazie B).
- flat payload pada → „płaski plik” bez osi Φ nie wystarcza.
- Most −depth / −T nadal wysokie: T, state i mazur_tracer **korelują z depth** w syntezie — pełna izolacja osi wymagałaby niezależnego szumu; ZERO jest właściwą kontrolą geometrii.

## Go / no-go live Store
GO na Faza D (offline dump), potem live read-only — Most wnosi geometrię Φ.
