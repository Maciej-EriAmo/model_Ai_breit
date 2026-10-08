# Trzy drzewa

Jedno jądro ma trzy implementacje. Inżynier pracuje w jednym drzewie na raz i nie zrównywa ich plików.

## 1. Lorentz — `C:\Users\drwis\model Ai`

Zamrożony oryginał. Ostatni znany commit na `master`: `592bf00`. Tego katalogu się nie edytuje.

Szerokość jest jedna na głowę:

```text
gamma = softplus(log_gamma) + 1e-3
A     = gamma * C / (gamma^2 + ΔE^2)
```

Nie ma `beta` ani `kappa`. W kodzie Lorentza zostają tryby `directed` i `onesided`. W tym forku te tryby są odrzucone.

Stare biegi językowe z 21 września 2026 leżą w tym drzewie. Nie są porównaniem z biegiem Breita–Wignera z 6 października.

## 2. Ten fork — `C:\Users\drwis\model_Ai_breit`

Klon oryginału z dopisanym Most-1. Tu jest enkoder, recall, mały i średni LM, seal, checkpointy i notatki pomiarów.

```text
gamma = gamma0 * (1 + beta * kappa)
kappa = max(C) / sum(C)
```

Maski i seal wchodzą w `C` przed `kappa`. Wzór w `breit_kernel.py` zostaje. Przyspieszenie przez złożenie operacji w jedno wywołanie jest osobnym zadaniem i ma dać te same wagi.

Fork nie importuje `C:\Users\drwis\transformer2`.

## 3. TSRA — `C:\Users\drwis\transformer2`

Biblioteka numeryczna jednej głowy. Specyfikacja jest w `Plan.txt`. Numpy jest źródłem wzoru, torch jest opcjonalnym odpowiednikiem jednego kroku.

Publiczne wejście:

```text
gamma = resonance_width(kappa, gamma0, beta)
W     = breit_wigner_weights(C, ΔE, gamma, mask=None)
S     = tsra_relax(C, ΔE, S0, gamma0, beta, rho, K, ...)
```

Różnice względem forka, które zostają:

| | Fork | TSRA |
|--|------|------|
| Kto buduje `C` i `ΔE` | sieć, z `QK` i tracera | wołający, z zewnątrz |
| `kappa` | `max(C) / sum(C)` | domyślnie `max(S)`; `lse` jest flagą, `τ = 0.05` |
| Krok | jeden | Most-1 to `K = 1`, `ρ = 1`; Most-K iteruje |
| `ΔE` w pętli | nie ma pętli | kopia, tylko odczyt |
| Podłoga `gamma0` | `softplus + 1e-3` | `max(gamma0, 1e-8)` |
| Pusty nośnik | wiersz zer | `ValueError` |
| Kształt | `[B, h, N, N]` | `(N,)` albo `(B, N)`, jeden rozkład na wiersz |
| Uczenie | wsteczna propagacja przez sieć | `update_inputs` rusza `C` i `ΔE` ze skalaru wołającego |

`(B, N)` w TSRA to `B` osobnych rozkładów długości `N`. To nie jest macierz uwagi `N×N`.

Siatka faz `β ≤ 8` jest zamknięta. Oscylacje przy `ρ = 1` i `β ≥ 12` są wynikiem, nie usterką do strojenia. Po mapie wolno zmienić tylko głębokość studni albo `gamma0`, i powtórzyć `scripts/run_phase.py`.

## Czego nie przenosić

- Nazwy `lorentz_attention`. W TSRA jądro nazywa się `breit_wigner`.
- Pętli `K` do enkodera forka. Pomiary T0–T3 i WikiText liczyły jeden krok Most-1.
- `QK`, FFN i pętli treningu do TSRA.
- Podłogi `1e-3` z forka do TSRA albo podłogi `1e-8` z TSRA do forka.
