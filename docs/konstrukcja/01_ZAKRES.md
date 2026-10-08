# Zakres konstrukcji

## Co jest jądrem

Jądro to dwie funkcje w `breit_kernel.py`:

- `breit_wigner_rows` — surowe amplitudy `A`, ten sam kształt co `C`
- `breit_wigner_weights` — te amplitudy po podzieleniu przez sumę kluczy

Obie dostają już gotowe `C` i `ΔE`. Nie liczą `Q`, `K` ani projekcji energii.

Szerokość jednego wiersza:

```text
kappa  = max(C) / sum(C)
gamma  = gamma0 * (1 + beta * kappa)
A      = gamma * C / (gamma^2 + ΔE^2)
```

`gamma0` i `beta` są wagami głowy. `kappa` jest liczbą z wiersza `C`.

## Co jest szkieletem, a nie jądrem

Szkielet dokłada do jądra rzeczy, które jądro zakłada jako wejście:

| Element | Gdzie | Rola |
|---------|--------|------|
| `C = softplus(S / τ)²` | `bridge.py`, `recall_model.py`, `lm_tiny.py` | sprzężenie z treści |
| `ΔE` z liniowej projekcji tracera | te same trzy pliki | rozstrojenie |
| maska, przekątna, seal | mnożnik na `C` przed jądrem | zabroniony klucz ma `C = 0` |
| normalizacja wiersza w LM | `lm_tiny.py` po `breit_wigner_rows` | wagi sumują się do 1 |
| bliźniak softmax | `masked_softmax` | ta sama maska, inny wzór wagi |
| FFN, LayerNorm, CLS, głowa klasy | enkoder i LM | reszta sieci |

Zmiana FFN albo rozmiaru `d_model` nie jest zmianą jądra. Zmiana kolejności maski względem `kappa` jest zmianą jądra.

## Kanon rozmiaru sieci

Z `config.py`, klasa `PhiMostCanon`:

- sprzężenie `softplus`
- 2 warstwy, `d_model = 32`, 2 głowy, więc 4 jądra w enkoderze kanonu
- treść `d_x = 8`, tracer `n_dim = 6`
- seal włączony: próg twardego zera `0.20`, pasmo zimne `0.45`, `exp_k = 12`

Średni model językowy jest osobnym presecie (`TinyLMConfig.preset("medium")`): `d = 256`, 4 warstwy, 4 głowy, słownik 256. To nie jest kanon enkodera Φ.

## Poza zakresem tego opracowania

- Przepisanie wzoru na inną szerokość albo na znak `ΔE`. Tryby `directed` i `onesided` ten fork odrzuca przy starcie.
- Nowe warstwy, FFN, pozycje, stos transformera w `C:\Users\drwis\transformer2`.
- Uczenie Holona wagami Mostu. Holon trzyma sesję. Wagi są w `checkpoints/`. Dumpy Φ są osobno.
- Karta graficzna i NPU. Pomiary równoważności są na CPU.
- Nazwa „jądro ogólne” jako werdykt. Zgoda ze softmaxem jest zmierzona przy ±5 punktach, gdy oba dostają ten sam opis. Duży model językowy nie jest zmierzony.
