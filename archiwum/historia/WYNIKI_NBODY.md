# Wyniki — walidacja n-ciał / Układ Słoneczny

Kanon: `softplus×2`, d=32, n_dim=6, gate Δ≥0.03.

| zadanie | Softmax | Most | ZERO | Most−SM | |
|---------|---------|------|------|---------|---|
| solar | 0.3563 | 0.6208 | 0.3667 | +0.2646 | PASS |
| three | 0.5229 | 0.7417 | 0.5062 | +0.2188 | PASS |

**Werdykt:** **PASS** (2/2 zadań ≥+3 pkt) — mocny

## Interpretacja
- **solar**: warstwy star / rocky / giant (+Księżyc w rocky); energia Kepler w tracerze; content bez masy/hierarchii.
- **three**: 3 ciała + Verlet; warstwy = rank masy; label = interakcje wewnątrz warstw (jak drzewo lab).
- ZERO ≈ Softmax ⇒ bez τ most nie powstaje (τ = warunek, nie handicap).

## Config po audycie
- `config.CANON` — jedyne źródło domyślnych hiperparametrów.
- `hash_embed` stabilny (SHA256).
- tracer n_dim=6 zawiera mazur (dim5).

