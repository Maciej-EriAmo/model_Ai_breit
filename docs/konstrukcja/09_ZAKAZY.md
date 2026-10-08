# Zakazy przy opracowaniu

## Drzewa

- Nie edytować `C:\Users\drwis\model Ai`.
- Nie importować `transformer2` do forka i nie dokładać do TSRA pętli treningu, FFN, pozycji ani softmaxu jako modelu.
- Nie nazywać jądra `lorentz_attention`.
- Nie przenosić podłogi `1e-3` ani `1e-8` między drzewami.

## Wzór

- Nie zmieniać `A = gamma * C / (gamma^2 + ΔE^2)` po to, żeby dogonić czas softmaxu.
- Nie wkładać znaku `ΔE` do wagi forka. Tryby `directed` i `onesided` zostają odrzucone.
- Nie nakładać maski, sealu ani przekątnej po obliczeniu `kappa`.
- Nie odłączać `kappa` od grafu przy okazji porządków. Gradient przez argmax jest częścią Most-1.
- Nie stroić siatki `β ≤ 8` w TSRA po zobaczeniu mapy.

## Porównania

- Nie zestawiać Mostu z tracerem w `ΔE` z softmaxem bez tego tracera i nie nazywać różnicy właściwością jądra.
- Nie ogłaszać jądra ogólnym. Zgoda jest przy ±5 punktach na zmierzonych siatkach i na małym oraz średnim LM.
- Nie brać września `STATUS.md` ani biegu z 21 września 2026 za kontrolę Breita–Wignera.
- Nie traktować wolniejszego kroku jako porażki. Porażką jest `NaN`, rosnąca strata albo waga inna niż kontrakt.

## Pamięć i produkt

- Nie mieszać wag, dumpów Φ i sesji Holon.
- Nie włączać domyślnie flag `live`, `gen` i `neural_lambda`.
- Nie wyłączać sealu w kanonie. Wyłączenie jest per źródło, gdy oś 3 nie jest temperaturą archiwum.
- Nie zapisywać wagi nauczonej ze sealem wyłączonym i nie wczytywać jej przez `load_encoder` bez ponownego ustawienia flagi. Dziś checkpoint tej flagi nie niesie. Szczegół: [`07_OTWARTE.md`](07_OTWARTE.md).
