# Kontrakt numeryczny

Każda zmiana jądra ma zachować punkty poniżej. Testy są w [`08_SPRAWDZENIE.md`](08_SPRAWDZENIE.md).

## Maski wchodzą do C przed kappa

`kappa` czyta cały wiersz `C`. Klucz zabroniony ma mieć `C = 0` zanim liczone jest `max` i suma. Inaczej przyszły token, padding albo zimny klucz podnosi szerokość wiersza.

W enkoderze i klasyfikatorze mnożnik `keep` składa trzy rzeczy (`bridge.key_keep`):

1. seal temperatury na kluczu (`most_seal.apply_t_key_seal`)
2. padding (`mask < 0.5` jest kluczem zabronionym; pozycja CLS zostaje otwarta)
3. zero na przekątnej

Potem `C = softplus(S)² * keep`.

W małym LM maska przyczynowa zeruje przyszłe klucze na `C` przed `breit_wigner_rows`. Przekątna w LM zostaje. LM nie woła sealu.

Test `test_blocked_key_stays_zero_and_does_not_set_kappa` pilnuje, że zablokowany klucz zostaje zerem i nie ustawia `kappa`.

## Pusty wiersz zostaje skończony

Gdy suma `C` jest zerem, `clamp_min(1e-12)` daje `kappa = 0` i `gamma = gamma0`. Surowe `A` jest zerem. Po normalizacji wiersz jest zerem, nie `NaN`.

`assert_finite` w `bridge.py`, `recall_model.py` i `lm_tiny.py` przerywa forward, gdy `S`, `C`, `ΔE` albo waga nie są skończone.

## Znak ΔE

Do wagi wchodzi `ΔE²`. Gałki `delta_alpha` i `directed_bias` są zapisane w konstruktorze `PhiBridgeEncoder` i `MostKnobs`, a w trybie `square` forward ich nie czyta. `delta_mode` inny niż `square` podnosi `ValueError` w `lm_tiny.py` i w `recall_model.py`.

## Podłogi

| Miejsce | Wartość | Po co |
|---------|---------|--------|
| `gamma0` | `softplus(log_gamma0) + 1e-3` | szerokość nie schodzi do zera |
| suma `C` w `kappa` | `clamp_min(1e-12)` | pusty wiersz |
| mianownik `A` | `+ 1e-12` | przy `gamma ≥ 1e-3` jest zbędne; zostaje |
| suma wag | `clamp_min(1e-12)` | pusty wiersz daje zero, nie dzielenie przez zero |

W bibliotece TSRA podłoga jest inna: `gamma0 = max(gamma0, 1e-8)`, a ujemne `C` jest błędem. Tych dwóch kontraktów nie wolno zrównywać w jednym pliku.

## Seal

Domyślnie włączony, `config.CANON.t_attn_seal = True`. Oś tracera o indeksie 3.

| Warunek | Skutek na sprzężeniu klucza |
|---------|------------------------------|
| `T < 0.20` | twarde zero |
| `0.20 ≤ T < 0.45` | mnożnik `exp(-12 * (0.45 - T))` |
| `T = 0.20` | mnożnik `exp(-3) ≈ 0.0498`, bez twardego zera |
| `T ≥ 0.45` | mnożnik 1 |
| pozycja 0 (CLS) | zawsze otwarta |
| brak osi 3 (`n_dim ≤ 3`) | seal nic nie robi |

Gdy cały tracer atomów poza CLS jest zerem, seal zwraca sprzężenie bez zmian. Ablacja `zero_tracer` zostaje pełną macierzą `C`, a nie martwą uwagą.

Oś 3 jest temperaturą archiwum tylko w tracerze stylu Karmazyn. W strukturze, QM i zdjęciach glifów jest energią albo barwą. `phimost_learn.seal_for_source` zwraca `False` dla nazw `struktura`, `qm`, `glyphs_photo`. Inna nazwa zwraca `None`, a `None` w `apply_t_key_seal` znaczy kanon, czyli seal włączony.

`checkpoint.save_encoder` nie zapisuje `t_seal`. `load_encoder` buduje model bez tej flagi, więc po wczytaniu seal znowu idzie z kanonu.

Bliźniak softmax sealu nie używa.

## Bliźniak softmax

`bridge.masked_softmax` wycina przekątną i padding wartością `-1e9` przed `F.softmax`, potem jeszcze raz zeruje te pozycje i normalizuje żywy wiersz. Przy własnym logitcie około 80 wycięcie przekątnej dopiero po softmaxie zeruje wiersz w float32. Test `test_softmax_twin_survives_a_huge_self_score` trzyma kolejność sprzed softmaxu.

Softmax w LM wycina przyszłość tą samą maską przyczynową, wartością `-1e9` na logitach, i nie zeruje przekątnej.

## Pamięć

Trzy magazyny zostają osobno (`docs/LEARN.md`):

- M1 — wagi, `checkpoints/*.pt`
- M2 — dumpy Φ
- M3 — sesja Holon, komora `Most_Lorentz`

Jądro uczy się w M1, jak patrzeć. M2 mówi, na co patrzeć. M3 nie wchodzi do forwardu.
