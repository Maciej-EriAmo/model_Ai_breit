# Jak sprawdzić zmianę jądra

Katalog roboczy: `C:\Users\drwis\model_Ai_breit`.

## Testy kontraktu

Pytestem. Odpalenie `python doswiadczenie/testy/test_breit_kernel.py` staje na bloku `__main__` i nie woła testów zdefiniowanych niżej.

```text
python -m pytest doswiadczenie/testy/test_breit_kernel.py doswiadczenie/testy/test_most_seal.py -q
```

2026-10-08, z katalogu forka: 15 passed.

`test_breit_kernel.py` trzyma:

| Test | Kontrakt |
|------|----------|
| `test_row_matches_the_closed_form_and_ignores_the_sign` | wzór i brak znaku `ΔE` |
| `test_a_larger_beta_lowers_the_peak_share` | większe `beta` spłaszcza szczyt |
| `test_blocked_key_stays_zero_and_does_not_set_kappa` | maska przed `kappa` |
| `test_sign_modes_are_rejected_and_a_forward_stays_finite` | `square` tylko; forward skończony |
| `test_softmax_twin_survives_a_huge_self_score` | przekątna przed softmaxem |
| `test_beta_zero_ignores_the_row_and_the_raw_parameter` | `beta_exact=0` daje γ = γ0 i ignoruje `beta_raw` |
| `test_excess_scale_starts_flat_rows_at_gamma0` | płaski wiersz skali `excess` startuje od γ0 |
| `test_width_arms_build_and_stay_finite` | ramiona `beta0` i `excess` budują się i zostają skończone |
| `test_weights_equal_normalised_rows` | ręczna normalizacja LM równa `breit_wigner_weights` |
| `test_bridge_matches_per_layer_masking` | klasyfikator maskuje tak samo w każdej warstwie |

Seal: `doswiadczenie/testy/test_most_seal.py`. Kanon ma `t_attn_seal is True`.

## Co dopisać przy konkretnej zmianie

| Zmiana | Nowy test |
|--------|-----------|
| `t_seal` w checkpointcie | zapis, odczyt, flaga ta sama; brak pola w starym pliku zostawia kanon |
| `mean_dE` tylko po żywych kluczach | wiersz z paddingiem i zerem na przekątnej nie wchodzi do średniej |
| odłączenie `w_last` | `w_last.requires_grad` jest fałszem po forwardzie treningowym |
| złożenie jądra w jedno wywołanie | różnica wag wobec `breit_wigner_weights` poniżej `1e-6` na kształcie `[2, 2, 8, 8]` |

## Czego ten zestaw nie sprawdza

Pytest jądra nie powtarza T0–T2 ani WikiText-2. Zmiana, która przechodzi testy i rusza wzór, nadal wymaga biegu z [`06_POMIARY.md`](06_POMIARY.md), zanim wolno powiedzieć, że zgodność ze softmaxem została.

TSRA sprawdza się u siebie:

```text
cd C:\Users\drwis\transformer2
python -m pytest -q
```

Ostatni zapisany wynik tej biblioteki to 38 testów po `update_inputs`. Ten fork tego zestawu nie uruchamia.

## Regresja liczb

Jedna zmiana jądra, jeden raport. W raporcie: seed, próg zamrożony przed seedem, strata albo r@1, stosunek czasu. Stary bieg Lorentza z `C:\Users\drwis\model Ai` nie jest kontrolą tego forka.
