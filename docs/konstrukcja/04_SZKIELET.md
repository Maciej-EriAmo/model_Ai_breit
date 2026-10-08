# Szkielet wokół jądra

Trzy wywołania. Wzór w `breit_kernel.py` jest jeden. Różnią się tym, skąd biorą `C` i `ΔE` oraz co zerują przed `kappa`.

## Klasyfikator — `bridge.BridgeTransformer`

```text
x  → Linear → doklej CLS
τ  → doklej zerowy wiersz CLS
S  = Q K^T / sqrt(d_h)
E  = Linear_h(τ)          # wagi startują od jedynek, bias od zera
ΔE = E_i - E_j            # bez e_scale; w tym pliku skala jest 1
C  = softplus(S)^2 * keep
w  = breit_wigner_weights(C, ΔE, log_gamma[warstwa], beta_raw[warstwa])
```

`keep` jest liczone raz na forward, bo zależy od tracera i maski, nie od warstwy.

Zwracane `w_last` trzyma graf autogradu. Zwracane `ΔE` jest odłączone. Pętla treningu w `train_eval` obu nie używa do straty.

Bias warstwy `e` skraca się w różnicy dwóch atomów. Na CLS tracer jest zerem, więc energia CLS na starcie jest samym biasem. Różnica CLS z atomem ten bias widzi.

`mean_dE` w `train_eval` to średnia z `|ΔE|` po całym tensorze `[B, h, N, N]`: przekątna, CLS, padding. To nie jest średnia po dozwolonych kluczach.

## Enkoder recall — `recall_model.PhiBridgeEncoder`

Ta sama kolejność, z dwiema różnicami:

```text
ΔE = e_scale * (E_i - E_j)     # e_scale domyślnie 1
C  = softplus(S / couple_temp)^2 * keep
```

`couple_temp` domyślnie 1, obcięte od dołu do `1e-3`. Wynik to embeddingi tokenów bez CLS, plus odłączone `ΔE` ostatniej warstwy. Wag uwagi enkoder nie zwraca.

`PhiSoftmaxEncoder` ignoruje tracer i woła `masked_softmax`.

## Mały LM — `lm_tiny._MostCausalBlock`

```text
S  = Q K^T / sqrt(d_h)
E  = Linear_h(τ_pozycji)
ΔE = e_scale * (E_i - E_j)
C  = softplus(S / couple_temp)^2
C  = 0 na przyszłych kluczach
A  = breit_wigner_rows(C, ΔE, log_gamma, beta_raw)
w  = A / clamp_min(sum A, 1e-12)
```

Tracer pozycji, `position_tracer`: `t`, `1-t`, sinus, kosinus, z `t` w `[0, 1]` wzdłuż okna. To nie jest temperatura archiwum. Osi 3 nie wolno tu podawać do sealu.

Plugin źródła dokumentu jest osobno: `TinyLMConfig.side_sources` i argument `side` w `TinyLM.forward`. Domyślnie 0. `train_lm` woła model bez `side`, o ile skrypt treningu sam go nie poda.

`_SoftmaxCausalBlock` nie ma projekcji energii.

## Kto woła jądro

| Wołający | Funkcja jądra | Maska przed kappa |
|----------|----------------|-------------------|
| `BridgeTransformer` | `breit_wigner_weights` | seal, padding, przekątna |
| `PhiBridgeEncoder` | `breit_wigner_weights` | seal, padding, przekątna |
| `_MostCausalBlock` | `breit_wigner_rows` | tylko przyszłość |

Skrypty doświadczeń mają iść przez te trzy klasy. Nowe wywołanie `breit_wigner_*` z maską nałożoną po `kappa` łamie kontrakt z [`03_KONTRAKT.md`](03_KONTRAKT.md).

## Parametry, które forward Most-1 czyta

| Parametr | Czytany |
|----------|---------|
| `log_gamma` / `log_gammas` | tak |
| `beta_raw` | tak |
| `e_scale` | tak, w enkoderze i w LM |
| `couple_temp` | tak, w enkoderze i w LM |
| `coupling` `softplus` albo `relu` | tak, w enkoderze i klasyfikatorze; LM jest na stałe `softplus` |
| `t_seal` | tak, gdy nie jest `None`; `None` znaczy kanon |
| `delta_alpha`, `directed_bias` | zapisane, w trybie `square` nieczytane |
