# Miejsca do opracowania

Każdy punkt jest dziurą w konstrukcji albo w pomiarze. Żaden nie upoważnia do zmiany wzoru z [`02_WZOR.md`](02_WZOR.md).

## Konstrukcja, którą kod już pokazuje

1. **Seal nie wraca z checkpointu.** `checkpoint.save_encoder` zapisuje wymiary i `coupling`. `load_encoder` stawia `t_seal=None`, czyli kanon włączony. Model uczony ze sealem wyłączonym po wczytaniu liczy inaczej, dopóki wołający nie ustawi flagi jeszcze raz. Opracowanie: zapisać flagę w `arch` i odtworzyć ją w `build_encoder`, z testem round-trip.

2. **`mean_dE` miesza pozycje.** `bridge.train_eval` średnia `|ΔE|` po całym `[B, h, N, N]`, łącznie z przekątną, CLS i paddingiem. Opracowanie: średnia po kluczach, które `keep` zostawia dodatnie, albo osobna nazwa metryki, żeby raport nie udawał rozstrojenia żywych par.

3. **`w_last` trzyma graf.** W `BridgeTransformer` odłączone jest tylko `ΔE`. Zwrócone wagi nadal wiszą na grafie forwardu. Opracowanie: odłączyć je tak samo jak `ΔE`, jeśli żaden wołający nie liczy przez nie straty. Przed zmianą sprawdzić wołających.

4. **Bias CLS.** Energia CLS to bias warstwy `e`, bo tracer CLS jest zerem. Między dwoma atomami bias znika. Opracowanie: albo zostawić i wpisać to do metryki `mean_dE`, albo liczyć `ΔE` bez biasu i mieć na to test.

5. **Martwe gałki.** `delta_alpha` i `directed_bias` siedzą w konstruktorze i w `MostKnobs`. Forward Most-1 ich nie czyta. Tryb inny niż `square` pada przy starcie. Opracowanie: usunąć je z publicznego konstruktora forka albo zostawić z jawnym komentarzem, że należą do zamrożonego drzewa Lorentza. Skrypty `archiwum/skrypty/run_signed_delta_ablation.py` i `run_atoms_delta_ablation.py` dziś padają na starcie.

6. **Ręczna normalizacja w LM.** `lm_tiny` dzieli wynik `breit_wigner_rows`. Wynik jest równy `breit_wigner_weights`. Zostaje, dopóki test równości jest zielony. Złożenie w jedno wywołanie jest dozwolone tylko przy teście równości wag.

## Pomiary, które plan jeszcze trzyma

7. **Hold-out Na/K na FULL118.** `docs/PLAN_NEXT.md`, punkt 4. Pełna tablica Most-only jest zrobiona (`MENDELEJEW_FULL.md`). Hold-out na tym dumpie zostaje do zrobienia tym samym protokołem, bez ruszania wzoru.

8. **Klasyczny loader softmaxu.** Zostaje w planie jako loader spoza bliźniaka Φ (`TRANSCODE`). Bliźniak `PhiSoftmaxEncoder` zostaje kontrolą regresji.

9. **Złożone jądro.** Czas 1,1–2,2× jest czasem łańcucha osobnych operacji. Jedno wywołanie o tych samych wagach jest osobnym zadaniem. Bez testu równości nie wolno ogłosić, że Most jest szybszy.

10. **Duży język.** Średni model, około 3,3 mln wag, jest zmierzony na WikiText-2. Model 100–300 mln wag nie był liczony. Na tej maszynie budżet około trzech godzin zostaje przy średnim LM.

## Szerokość zależna od wiersza — kod jest, pomiaru nie ma

Historyczne Most-1 liczy `γ = γ0 · (1 + β · κ)` przy `κ = max(C) / sum(C)`. `beta_raw = 0` daje `β ≈ ln 2`, nie zero. Na płaskim wierszu `κ = 1/n`, więc krótki wiersz startuje szerszy niż długi. W LM pozycja 0 ma jeden klucz i `κ = 1`.

To jest opis mechanizmu. Żeby był pomiarem, następny bieg ma wziąć dwa dodatkowe ramiona z `lm_tiny.py`:

| ramię | co robi przy starcie |
|-------|----------------------|
| `beta0` | `β = 0`, więc `γ = γ0` na każdym wierszu |
| `excess` | płaski wiersz i wiersz z jednym kluczem też startują od `γ0`; `β` zostaje uczone |

Domyślna para `most,softmax` zostaje historycznym wzorem. Uruchomienie:

```text
python archiwum/skrypty/run_lm_medium_1c.py --arms most,softmax,beta0,excess
```

Dopóki ten bieg nie jest zapisany, nie wolno mówić, że szerokość zależna od wiersza coś zmienia w stracie.

## Świadomie zostawione

- Seal na FULL118 włączony, bo oś `T` tam jest proxy okresu, a skrypt tak woła model. Ablacje jądra mają `t_seal=False`.
- Siatka `β` w TSRA. Oscylacja przy `ρ = 1`, `β ≥ 12` jest wynikiem mapy faz.
- Flagi produktu `PHIMOST_LIVE`, `PHIMOST_GEN`, `PHIMOST_NEURAL_LAMBDA` zostają wyłączone.
