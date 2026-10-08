# Wzór Most-1 w tym forku

Źródło: `breit_kernel.py`. Komentarz nad funkcjami i ciała `_row_gamma`, `breit_wigner_rows`, `breit_wigner_weights` są specyfikacją. Ten plik je spisuje.

## Wejścia

| Tensor | Znaczenie | Kształt |
|--------|-----------|---------|
| `coupling` | sprzężenie `C` | dowolny, ostatnia oś to klucze |
| `delta_e` | rozstrojenie `ΔE` | identyczny jak `C` |
| `log_gamma0` | parametr głowy, przed softplus | broadcast do `C` |
| `beta_raw` | parametr głowy, przed softplus | broadcast do `C` |

W uwadze typowy kształt `C` i `ΔE` to `[B, h, N, N]`. Parametry głowy w enkoderze to `[h, 1, 1]`.

Niezgodny kształt `C` i `ΔE` podnosi `ValueError`.

## Szerokość

```text
mass   = clamp_min(sum_j C_ij, 1e-12)
kappa  = max_j(C_ij) / mass
gamma0 = softplus(log_gamma0) + 1e-3
beta   = softplus(beta_raw)
gamma  = gamma0 * (1 + beta * kappa)
```

`max(C) / sum(C)` jest tym samym co `max(C / sum(C))`, gdy suma jest dodatnia. Kod liczy pierwszą postać i nie dzieli całej macierzy.

Start wag: `log_gamma0 = 0`, `beta_raw = 0`.

```text
softplus(0) = ln 2 ≈ 0.693147
gamma0      ≈ 0.694147
beta        ≈ 0.693147
```

`gamma0` jest co najmniej `1e-3` powyżej zera softplusa. `beta` jest nieujemne, bo softplus jest nieujemny. `kappa` dla niepustego wiersza leży w `[1/n, 1]`, gdzie `n` jest liczbą kluczy o dodatnim `C`. Pusty wiersz daje `kappa = 0` przez `clamp_min`, więc `gamma = gamma0`.

Gradient `max` wraca tylko przez argmax. `kappa` nie jest odłączane od grafu.

## Amplituda

`breit_wigner_rows`:

```text
A = gamma * C / (gamma^2 + ΔE^2 + 1e-12)
```

`breit_wigner_weights` pomija `gamma` w liczniku, bo w jednym wierszu jest stałe i skraca się przy normalizacji:

```text
A = C / (gamma^2 + ΔE^2 + 1e-12)
w = A / clamp_min(sum_j A, 1e-12)
```

Wiersz samych zer zostaje wierszem zer. Suma wag żywego wiersza wynosi 1.

`lm_tiny.py` woła `breit_wigner_rows`, a potem dzieli przez sumę jeszcze raz. Przy `gamma` stałym w wierszu wynik jest tym samym rozkładem co `breit_wigner_weights`. Test `test_weights_equal_normalised_rows` to trzyma. Ręczne dzielenie zostaje, dopóki ten test jest zielony.

## Co waga widzi

Waga znormalizowana jest proporcjonalna do

```text
C / (gamma^2 + ΔE^2)
```

Większe `gamma` spłaszcza udział szczytu, gdy `ΔE` nie jest zerem. Test `test_a_larger_beta_lowers_the_peak_share` to sprawdza.

`ΔE` wchodzi jako kwadrat. Zamiana znaku `ΔE` nie zmienia `A`. Test `test_row_matches_the_closed_form_and_ignores_the_sign` porównuje wynik z tym wzorem w `float64`.

## Dwa ramiona obok tego wzoru

Domyślne wywołanie zostaje przy skali `kappa`. `beta_raw = 0` oznacza `β ≈ ln 2`. Płaski wiersz ma wtedy `κ = 1/n`, więc skala startowa zależy od liczby kluczy.

`beta_exact=0` (`attn="beta0"`) stawia `β = 0` i `γ = γ0` na każdym wierszu. `scale="excess"` (`attn="excess"`) daje 0 na płaskim wierszu, więc taki wiersz startuje od `γ0` także przy niezerowym `β`. Oba ramiona są w `lm_tiny.py`. Bez biegu, który je porównuje, zależność szerokości od wiersza zostaje opisem.

## Most-K

Ten fork liczy jeden krok: `gamma` z aktualnego `C`, potem jedno `A`. Nie ma pętli `K` ani mieszania ze stanem `S`.

Pętla Most-K, `S ← (1-ρ) S + ρ W`, jest w `C:\Users\drwis\transformer2`, funkcja `tsra_relax`. Tam `kappa` bierze się ze stanu `S`, a `C` i `ΔE` są zamrożone w pętli. To osobna biblioteka. Opis różnicy: [`05_TRZY_DRZEWA.md`](05_TRZY_DRZEWA.md).
