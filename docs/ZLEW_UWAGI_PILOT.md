# Zlew uwagi i normalizacja jądra — pilot na CPU

**Data:** 2026-10-26  
**Skrypt:** [`../doswiadczenie/run_zlew_uwagi.py`](../doswiadczenie/run_zlew_uwagi.py)  
**Surowe wyniki:** [`../archiwum/historia/zlew_uwagi/`](../archiwum/historia/zlew_uwagi/)  
**Tło literaturowe:** [`SIGMOID_W_UWADZE.md`](SIGMOID_W_UWADZE.md)  
**Status:** pilot, 1 ziarno na ramię. Seria C (ablacja log N / LSSA) niekompletna — plan w sekcji „Plan kolejnych badań”. Nie spełnia zasad porównań serii Mendelejewa (≥20 przebiegów, TOST).

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

## Najbliższy krewny w literaturze: LSSAR

Gao, Spratling, Gionfrida, *Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models* (2025, https://arxiv.org/abs/2501.13428). Rozkładają softmax na „dodatniość + normalizacja L1” i zamieniają exp na softplus — tę samą konstrukcję, którą ma Most.

| | LSSAR | Most-1 |
|---|---|---|
| dodatniość | softplus(s · QKᵀ) | softplus²(QKᵀ/√d) |
| normalizacja | L1 w wierszu | L1 w wierszu |
| skala | s = log d · log N_i (N_i = liczba widzianych kluczy), Q i K normalizowane L2 | stała 1/√d, temperatura `couple_temp` |
| dodatkowy czynnik | ponowne ważenie ReLUᵖ(A·N − O), znów L1 (wyostrzanie) | Lorentz 1/(γ² + ΔE²), γ zależne od skupienia wiersza κ |
| wynik | prawie stała strata przy 16× długości treningu | — (nie testowane na długości) |

Ich Tabela 1 to też źródło drugiego pytania tej noty. Sigmoida (b = −log N) bez L1 ma stratę 7,46 przy 2K i 14,50 przy 8K. Z L1 ma 4,35 i 6,19, a softmax 4,17 i 6,28.

Wnioski dla Mostu:

- Rdzeń „softplus + L1” nie jest nowością Mostu. Recenzent o to zapyta. Wkładem Mostu jest tylko czynnik Lorentza z ΔE i zależną od wiersza szerokością γ.
- Most nie ma skali zależnej od długości (log N_i). Według LSSAR to ona, razem z wyostrzaniem, daje ekstrapolację długości. Naturalna ablacja: Most z s = log d · log N_i zamiast 1/√d.
- Uczciwe porównanie na długości wymaga ramienia LSSA, czyli softplus + L1 + skala log N, bez Lorentza. Inaczej nie da się oddzielić wkładu jądra Lorentza od wkładu samego softplus + L1. To ten sam problem co w ablacji jądra serii Mendelejewa (5f11).

## Seria C — ablacja skali log N i LSSA (niekompletna)

**Data:** 2026-10-26. Przerwana o 12:40 decyzją autora i przełożona na kolejny termin. Policzone 6 z 16 zaplanowanych przebiegów.

Preset `small` (~0,8 mln), N=256, 2000 kroków, bez BOS. Odniesienie równego rozkładu: 0,011. Wynik LSSA: s = log d · log N_i · cos(q, k), N_i = i + 1.

| ramię | wynik | jądro | normalizacja |
|---|---|---|---|
| `softmax` | QKᵀ/√d | exp | L1 (softmax) |
| `most` | QKᵀ/√d | softplus² · Lorentz | L1 |
| `most_logn` | LSSA | softplus² · Lorentz | L1 |
| `lssa` | LSSA | softplus | L1 |
| `lssa2` | LSSA | softplus² | L1 |
| `softmax_logn` | LSSA | exp | L1 (softmax) |

Wyniki (ziarno 0, softmax także ziarno 1):

| ramię | strata | ćwiartki pozycji | śr. udział tokenu 0 | czas przebiegu |
|---|---|---|---|---|
| softmax | 2,131 · 2,167 | 2,10 / 2,10 / 2,11 / 2,21 | 0,005 | 54 min |
| most | 2,025 | 2,06 / 1,99 / 1,97 / 2,08 | 0,003 | 97 min |
| lssa | 1,996 | 2,01 / 1,96 / 1,96 / 2,04 | 0,005 | 72 min |
| lssa2 | 1,958 | 1,98 / 1,91 / 1,92 / 2,02 | 0,004 | 72 min |
| **most_logn** | **1,891** | 1,95 / 1,86 / 1,85 / 1,90 | 0,003 | 101 min |
| softmax_logn | — (przerwane) | | | |

Czasy przy 5 procesach naraz po 3 wątki. Zlewu nie ma w żadnym ramieniu (0/16 głów).

Wstępne różnice (jedno ziarno, rozrzut softmaxa między ziarnami 0,04):

| porównanie | co izoluje | Δ straty |
|---|---|---|
| `most_logn` − `most` | skala LSSA w Moście | **−0,13** |
| `most_logn` − `lssa2` | czynnik Lorentza (reszta wspólna) | −0,07 |
| `lssa2` − `lssa` | kwadrat softplusa | −0,04 |
| `lssa` − `softmax` | wynik LSSA + softplus zamiast exp (dwie zmiany naraz) | −0,14 |

Ponadto tylko `most_logn` nie ma skoku straty w ostatniej ćwiartce pozycji (+0,05 wobec +0,07…+0,11 w pozostałych ramionach).

**Czego te liczby nie mówią:**

- Bez `softmax_logn` nie wiadomo, czy sama skala LSSA nie daje softmaxowi tyle samo. Wtedy przewaga `most_logn` należałaby do skali, nie do jądra.
- Różnice −0,04 i −0,07 są rzędu rozrzutu między ziarnami. Tylko −0,13 jest wyraźnie ponad nim.
- 2000 kroków na małym modelu to trening niezbieżny. Różnice mogą mierzyć szybkość uczenia, nie jakość końcową.
- Wszystko w obrębie długości treningowej. Ekstrapolacji, z której znany jest LSSAR, tu nie ma (wyuczone embeddingi pozycji w `lm_tiny`).

## Plan kolejnych badań

Kolejność według tego, co najtaniej rozstrzyga najwięcej.

1. **`softmax_logn`, ziarno 0** (~1 h na CPU). Rozstrzyga, czy przewaga `most_logn` to skala, czy jądro.
   - `softmax_logn` ≈ 1,89 → przewagę daje skala. Wniosek dla Mostu: przyjąć skalę log N, jądro bez przewagi (zgodnie z 5f11 i 5f15).
   - `softmax_logn` ≥ ~2,0 → skala razem z jądrem Lorentza. Warto iść do kroku 2.
2. **Dokończyć serię C: ziarna 1 i 2** dla wszystkich sześciu ramion (11 przebiegów, ~3–4 h przy równoległym uruchomieniu). Średnie i rozrzut zamiast pojedynczych liczb.
3. **Dłuższy trening** dla ramion, które przeżyją kroki 1–2 (np. 10 000 kroków). Sprawdza, czy różnice zostają po zbieżności.
4. **Ekstrapolacja długości.** Wariant `lm_tiny` bez wyuczonych embeddingów pozycji (pozycja tylko przez tracer ΔE albo RoPE). Trening przy N=256, ocena przy 512–2048. Tu LSSAR ma swoją główną przewagę, a Most z ΔE może mieć naturalną przewagę pozycyjną. Do porównania: `softmax`, `softmax_logn`, `lssa`, `most`, `most_logn`; opcjonalnie pełny LSSAR z ponownym ważeniem ReLUᵖ.
5. **Test równoważności / przewagi** na ramionach, które zostaną: zamrożona specyfikacja, ≥20 ziaren, Holm/TOST, jak w serii Mendelejewa ([`../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md`](../doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md)). Na CPU to dziesiątki godzin, więc raczej GPU.
6. **Zlew uwagi** (pytanie 1 tej noty, otwarte). Potrzebna skala, w której softmax ma zlew: większy model i korpus, najpewniej GPU. Dopiero wtedy `most` vs `raw` vs `softmax`, a także `most_logn`.
7. **Długość dla `raw` vs `most`** przy N ≫ 256 (preset `medium_long`). Potwierdza narastanie straty z pozycją bez L1, sygnalizowane w serii A. Niski priorytet, bo L1 i tak zostaje.

Uwaga praktyczna: zadania w tle sesji Claude mają limit 2 h. Przebiegi trwają do ~100 min, a przy wielu procesach naraz dłużej. Uruchamiać je jako procesy odłączone (PowerShell `Start-Process`), po jednym ziarnie na proces. Każdy wynik dopisuje się do JSONL od razu po przebiegu, a przerwany przebieg przepada w całości.

## Odtworzenie

```
python doswiadczenie/run_zlew_uwagi.py --preset medium --steps 2000 --seq-len 256 --seeds 0 --out archiwum/historia/zlew_uwagi/seria_A_medium_bez_bos.jsonl
python doswiadczenie/run_zlew_uwagi.py --preset small --steps 6000 --seq-len 128 --bos 2 --seeds 0 --out archiwum/historia/zlew_uwagi/seria_B_small_bos.jsonl
# seria C: jedno ramię i jedno ziarno na proces
python doswiadczenie/run_zlew_uwagi.py --preset small --steps 2000 --seq-len 256 --seeds 0 --arms softmax_logn --threads 3 --out archiwum/historia/zlew_uwagi/seria_C_softmax_logn.jsonl
```

Czas na CPU (8 wątków, sam proces): ~17 min na przebieg softmax w serii A. Most jest droższy. Przy kilku procesach naraz czasy rosną 2–3×.
