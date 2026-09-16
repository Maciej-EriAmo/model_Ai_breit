# Kana P1 — diagnostyka poziomu nauki

Seed=0, encoding=hologram, epochs=16, N=286, **chance r@1 = 1/(N-1) = 0.00351 (0.351%)**.

InfoNCE recall po całym grafie (286 węzłów) — Kana P1 raportowało końcowe r@1≈0.00–0.02 bez krzywej. Tu: cold start vs krzywa vs ×chance.

## Podsumowanie

| model | cold r@1 | final r@1 | ×chance | Δloss | uczy się? |
|-------|----------|-----------|---------|-------|-----------|
| Most | 0.0075 | 0.0075 | 2.1× | +0.399 | słabo/nie |
| Softmax | 0.0075 | 0.0025 | 0.7× | +0.415 | słabo/nie |
| ZERO | 0.0075 | 0.0075 | 2.1× | +0.409 | słabo/nie |

## Interpretacja

- Poziom przypadku ≈ **0.35%**. Kana P1 mean Most≈0.5% ≈ **~1–2×chance** — praktycznie nie nauczył się rankingu.
- Holo-Φ P2 (dla porównania) miało mean Most≈0.14–0.16 (~40×chance przy mniejszym N) — inna skala nauki.
- Jeśli loss spada, a r@1 zostaje przy podłodze: model minimalizuje CE, ale nie generalizuje rankingu na val.

JSON: `kana_learning_diag.json`
