# Zlew uwagi i normalizacja jądra — pilot na CPU

**Data:** 2026-10-26  
**Skrypt:** [`../doswiadczenie/run_zlew_uwagi.py`](../doswiadczenie/run_zlew_uwagi.py)  
**Surowe wyniki:** [`../archiwum/historia/zlew_uwagi/`](../archiwum/historia/zlew_uwagi/)  
**Tło literaturowe:** [`SIGMOID_W_UWADZE.md`](SIGMOID_W_UWADZE.md)  
**Status:** pilot, 1 ziarno na ramię. Nie spełnia zasad porównań serii Mendelejewa (≥20 przebiegów, TOST).

## Pytania

1. Literatura: po zastąpieniu softmaxa uwagą sigmoidalną **bez** normalizacji zlew uwagi (masa wiersza zbierana na pierwszym tokenie) nie pojawia się w modelach do ~1 mld parametrów. Zlew wiąże się z wymogiem sumowania wag do 1, nie z exp. Czy Most ma zlew?
2. Literatura: uwaga sigmoidalna bez normalizacji traci jakość na dłuższych sekwencjach, a normalizacja L1 przywraca poziom softmaxa. Jak normalizować jądro Lorentza?

Most już normalizuje L1: jądro A = γ·softplus²(S)/(γ²+ΔE²) ≥ 0, a wagi to w = A / ΣA ([`../lm_tiny.py`](../lm_tiny.py), `_MostCausalBlock.forward`). Jest więc w wariancie „sigmoida + L1”. Jeśli zlew bierze się z normalizacji, Most powinien go mieć jak softmax. Dodatkowo szerokość γ zależy od skupienia wiersza κ = max C / ΣC, co może zlew wzmacniać albo tłumić.

## Ramiona i metryka

| ramię | wagi |
|---|---|
| `softmax` | softmax(QK) z maską przyczynową |
| `most` | Most-1, w = A / ΣA |
| `raw` | Most-1 bez normalizacji wiersza, w = A / N (stała skala) |

- **Zlew:** dla każdej głowy średni udział masy wiersza na tokenie 0, po zapytaniach i ≥ 16. Głowa jest zlewem, gdy udział > 0,3. Dla `raw` udział liczony z A_i0 / Σ_j A_ij (tylko do metryki).
- **Odniesienie:** udział przy równym rozkładzie, średnia 1/(i+1).
- **Jakość:** strata następnego znaku na zbiorze kontrolnym (ostatnie 10% korpusu), także w ćwiartkach pozycji.

Korpus: `load_text_corpus()` (docs/ + glify + teksty domyślne), losowe okna znaków, AdamW lr 1e-3, batch 16.

## Wyniki

**Seria A** — preset `medium` (~3,3 mln), N=256, 2000 kroków, bez BOS. Odniesienie równego rozkładu: 0,011.

| ramię | głowy-zlewy | śr. udział tokenu 0 | maks. udział | strata | ćwiartki pozycji |
|---|---|---|---|---|---|
| softmax | 0/16 | 0,006 | 0,016 | 1,734 | 1,76 / 1,70 / 1,71 / 1,75 |
| most | 0/16 | 0,001 | 0,002 | **1,667** | 1,72 / 1,62 / 1,65 / 1,66 |
| raw | 0/16 | 0,005 | 0,007 | 2,087 | 2,02 / 2,07 / 2,12 / 2,12 |

**Seria B** — `small` (d=128, h=4, 4 warstwy, ~0,8 mln), N=128, 6000 kroków, stały token BOS (kod 2) na pozycji 0. Odniesienie: 0,018.

| ramię | głowy-zlewy | śr. udział tokenu 0 | maks. udział | strata | ćwiartki pozycji |
|---|---|---|---|---|---|
| softmax | 0/16 | 0,013 | 0,052 | 1,282 | 1,49 / 1,23 / 1,21 / 1,21 |
| most | 0/16 | 0,003 | 0,006 | **1,237** | 1,44 / 1,15 / 1,17 / 1,19 |
| raw | 0/16 | 0,002 | 0,006 | 1,462 | 1,62 / 1,39 / 1,43 / 1,42 |

## Wnioski

1. **Normalizacja L1 zostaje.** Most bez niej jest wyraźnie gorszy (strata +0,22…+0,42 względem Mostu z L1). Przy N=256 strata `raw` rośnie z pozycją (2,02 → 2,12), u pozostałych ramion nie. Kierunkowo zgadza się to z literaturą o uwadze sigmoidalnej. Most z L1 jest na poziomie softmaxa. Niższa strata Mostu w obu seriach to jedno ziarno — bez wniosku.
2. **Zlew: nierozstrzygnięte.** Nie ma go w żadnym ramieniu, także w softmaksie, nawet z BOS. Bez dodatniej kontroli porównanie nic nie mówi o zlewie w Moście. Jedyna obserwacja: Most (z L1 i bez) daje tokenowi 0 3–10× mniej, niż wynikałoby z równego rozkładu. Softmax z BOS ma jedną głowę ~3× powyżej równego rozkładu. Czy to cecha jądra, czy skali — nie wiadomo.

## Co dalej

- Rozstrzygający test wymaga skali, w której softmax ma zlew: większy model i korpus, najpewniej GPU. Dopiero wtedy `most` vs `raw` vs `softmax`.
- Potem ≥20 ziaren, zamrożona specyfikacja, TOST, jak w serii Mendelejewa ([`../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md`](../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md)).
- Osobno: długość. Test jakości `raw` vs `most` przy N ≫ 256 (preset `medium_long`) sprawdziłby narastanie straty z pozycją, sygnalizowane w serii A.

## Odtworzenie

```
python doswiadczenie/run_zlew_uwagi.py --preset medium --steps 2000 --seq-len 256 --seeds 0 --out archiwum/historia/zlew_uwagi/seria_A_medium_bez_bos.jsonl
python doswiadczenie/run_zlew_uwagi.py --preset small --steps 6000 --seq-len 128 --bos 2 --seeds 0 --out archiwum/historia/zlew_uwagi/seria_B_small_bos.jsonl
```

Czas na CPU (8 wątków, sam proces): ~17 min na przebieg softmax w serii A. Most jest droższy. Przy kilku procesach naraz czasy rosną 2–3×.
