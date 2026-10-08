# Pomiary, których konstrukcja nie może przekręcić

Źródło liczb: `docs/PODSUMOWANIE_JADRA.md` i notatki w `doswiadczenie/glify/`. Maszyna: CPU, Python 3.14, PyTorch 2.13. Data serii: 2026-10-05 i 2026-10-06.

Zdanie, które te biegi trzymają: przy tym samym opisie Most i softmax są wzorami wagi w granicy ±5 punktów. Ułamek Breita–Wignera sam punktów nie dokłada. Tracer w tokenie dokłada. Osobna relacja kolumny dokłada wtedy, gdy ta oś jest w stanie, a najbliższy sąsiad jej nie pokazuje.

## Zgoda wzorów

| Pomiar | Wynik | Notatka |
|--------|--------|---------|
| τ w tokenie, 30 porównań pierwiastków | 26 w ±5 pkt, 4 nierozstrzygnięte, Most w żadnym gorszy | `MENDELEJEW_KERNEL_EQUIV.md`, próg powtórzony w `JADRO_T0.md` |
| ten sam próg ±2 | 7 w progu, 0 całkiem poza, 23 nierozstrzygnięte | `JADRO_T0.md` |
| nuty, pisownia w tokenie | różnica trafień około 0 | `JADRO_T1.md` |
| tiny LM, numer źródła w tokenie | strata Most−softmax −0,0016, przedział wewnątrz ±0,05 | `JADRO_T2.md` |
| tiny LM, ΔE wyzerowane | +0,0009, przedział wewnątrz ±0,05 | `JADRO_T2.md` |

Margines straty ±0,05 był zamrożony przed pierwszym seedem T2. Margines ±5 punktów był zamrożony przed analizą. Ciaśniejszej zgody ta seria nie niesie.

Most, w którym tracer wchodzi tylko przez `ΔE`, na tiny LM ma stratę wyższą o 0,0045 od softmaxu bez pluginu. To porównanie miesza wzór z tym, co model dostaje.

## Co dokłada punkty

- Wpisanie tracera w token bije karę `−ΔE²/γ²` w wykładniku softmaxu: na nowych pierwiastkach o 6–12 punktów, na nutach o 11, 12 i 31 punktów kroku, na tiny LM obniża stratę o 0,0075.
- Relacja kolumny na katalogu słonecznym: +5…+11 punktów. W trójce hierarchicznej rola jest już całym sąsiedztwem i relacja nic nie dokłada. Wmieszana w jedną przestrzeń obniża krok o 0,26. Notatka: `NBODY_COMP.md`.

## Pełna tablica i język

| Bieg | Wynik |
|------|--------|
| FULL118, sam Most, 3 seedy, 64 epoki | r@1 = 0,575, r@3 = 0,892, MRR = 0,739, próg 3/3, 768 s |
| średni LM, notatki z repo, 600 kroków, 2 seedy | strata 2,416 wobec 2,353, czas 1,32× |
| WikiText-2, jeden przejazd, 2 seedy, ~3,3 mln wag | train 1,554 wobec 1,587; valid 1,557 wobec 1,604; test 1,580 wobec 1,627 |
| ten sam valid, 400 okien | trafienie bajta 54,6% wobec 53,8% |

Generowany tekst obu jąder na tym średnim modelu zapętla się. Duży model, 100–300 mln wag, nie był liczony.

Na FULL118 oś `T` jest proxy okresu, a skrypt zostawił seal kanonu. Ablacje jądra w `archiwum/skrypty/run_mendelejew_kernel_*.py` podają `t_seal=False`.

## Czas

Zmierzony koszt to niezłożony łańcuch: softplus, kwadrat, maska, suma, max, dzielenie, ścieżka tracera. Softmax jest jednym wywołaniem biblioteki.

Stosunek czasu Most/softmax: nuty 1,12×, tiny LM 2,15×, średni LM z 6 października 1,32×. Przepisanie `kappa` na `max(C)/sum(C)` zmienia wagi o rząd `1e-8` i prawie nie skraca kroku. Wzór został.

Wolniejszy krok nie jest warunkiem porażki. Warunkiem jest strata skończona i malejąca oraz zgodność wag z kontraktem.
