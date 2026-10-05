# Podsumowanie — Most i Softmax

**Data:** 2026-10-05  
**Fork:** `C:\Users\drwis\model_Ai_breit`  
**Komora:** `Most_Lorentz`

Jeden dokument. Liczby i protokoły leżą w notatkach wymienionych na końcu.

## Wniosek

Most i Softmax są zamienne jako wzór uwagi, gdy tracer jest już podany obu w tokenie. Ułamek Breita–Wignera nic do tego nie dokłada. Softmax usprawnia się tracerem wpisanym w token.

Tracer zostaje częścią Mostu: wiąże go przez niezerowe ΔE. Dla Softmaxu tracer jest pluginem. Osobna relacja kolumny dokłada punkty wtedy, gdy oś jest w stanie, a model sam jej nie widzi jako najbliższego sąsiada.

Zgoda wzorów obowiązuje przy ±5 punktach, nie przy ±2. Język zmierzony tutaj jest mały i powtarzalny. Most liczy się 1,1–2,2 raza dłużej, bo jego wzór jest kilkunastoma osobnymi operacjami, a Softmax jednym wywołaniem biblioteki. Oba siedzą w tym samym Pythonie i tym samym PyTorchu.

## Słowa

**Token** to opis, który model i tak czyta: widmo nuty, składowe ciała, bajty tekstu.

**Tracer** to zmienna trzymana obok tego opisu. Dla nuty jest nią pisownia, rytm, takt, dynamika. Dla tiny LM numer dokumentu. Dla Mostu tracer jest częścią mechanizmu: bez niezerowego ΔE Most rozpada się do zwykłej uwagi. Dla Softmaxu ta sama zmienna jest dodatkiem. Można ją wpisać w token albo włożyć jako karę −ΔE²/γ² w wykładnik. Jądro zostaje Softmaxem.

**Wzór uwagi** zamienia opisy na wagi: kogo model słucha mocniej. Most liczy to ułamkiem Breita–Wignera,

γ = γ₀ (1 + β κ), A = γ C / (γ² + ΔE²),

z C = softplus(QK)². Softmax liczy wagę funkcją wykładniczą. Gdy opis podany obu jest ten sam, zamiana wzoru nie zmienia wyniku.

**Relacja kolumny** to osobna oś, nie wzór uwagi. „Dół” znaczy: ta sama kolumna, następny wiersz. Głowica tej relacji dokłada punkty, gdy kolumna jest w stanie, a najbliższy sąsiad jej nie pokazuje.

## Co wyszło z pomiarów

Wszystko poniżej jest na CPU, Python 3.14, PyTorch 2.13. Ścisłe rozdzielenie tam, gdzie zadanie jest siatką: enkoder w treningu nie widzi elementów testowych. 20 przebiegów, o ile nie napisano inaczej.

### Wzór, gdy tracer jest już w tokenie

Na pierwiastkach, przy τ w tokenach: 26 z 30 porównań mieści się w ±5 punktach, 4 są nierozstrzygnięte, w żadnym Most nie jest gorszy. Przy tym samym progu ±2 zostaje 7 z 30, żadne nie wychodzi całe poza próg, 23 są nierozstrzygnięte. Dwadzieścia przebiegów nie udźwiga ciaśniejszej zgody.

Na nutach z pisownią w tokenie różnica Most − Softmax to −0,002, −0,002 i 0,000. Na tiny LM, z numerem źródła w tokenie, różnica straty to −0,0016, przedział [−0,0040, +0,0006]. Bez dodatkowej zmiennej, Most z wyzerowanym ΔE wobec Softmaxu: +0,0009, przedział [−0,0005, +0,0022]. Margines straty ±0,05 był zamrożony przed pierwszym seedem.

### Plugin w tokenie, nie kara w wykładniku

Na nowych pierwiastkach Softmax z tracerem w opisie bije Softmax z karą −ΔE²/γ² o 6–12 punktów.

Na nutach ta sama kara zrzuca krok pisowni z 0,49 (Softmax bez tracera) do 0,12. Wpisanie pisowni w token trzyma krok przy 0,43. Wobec kary jest to +0,31 na jednym kroku, +0,12 na hopie, +0,11 na dwóch krokach. Wszystkie trzy przeżywają korektę Holma.

Na tiny LM Softmax z numerem dokumentu ma stratę 0,0421 wobec 0,0497 bez niego. Różnica −0,0075, przedział [−0,0109, −0,0041], cały pod zerem.

### Relacja kolumny

| dziedzina | co widać w sąsiedztwie | relacja kolumny |
|---|---|---|
| katalog słoneczny, 80 ciał | ta sama powłoka 95%, ta sama masa 4% | +5…+11 punktów na kroku masy, wszystkie 5 ramion |
| trójka hierarchiczna, 84 ciała | ta sama rola 100% | nic nie dokłada; wmieszana w jedną przestrzeń obniża krok o 0,26 |
| nuty, 80 zdarzeń | ta sama oktawa 100%, ta sama pisownia 0% | dokłada +0,09…+0,13 tylko tam, gdzie model pisownię już dostaje (ΔE albo MLP) |

Masa w katalogu słonecznym jest w wektorze stanu, ale odległość ją zagłusza. Rola w trójce jest całym sąsiedztwem. Pisownia nuty nie jest w widmie: Cis i Des mają to samo widmo, a krok „ta sama pisownia, oktawa wyżej” jest zarazem krokiem o dwanaście półtonów.

Najwyższe trafienia kroku: na Słońcu MLP z relacją 0,984, potem Softmax bez kary 0,975. Na nutach MLP z relacją 0,73.

### Czas

Oba jądra woła PyTorch. `F.softmax` to jedno wywołanie biblioteki, liczone w C++. Most to kilkanaście małych operacji (softplus, kwadrat, maska, suma, maksimum, dzielenie) plus ścieżka tracera. Każda też schodzi do C++, każda odpalana osobno.

Na nutach Most trwał 1,12 raza dłużej (717 s wobec 642 s). Na tiny LM 2,15 raza (253 s wobec 117 s). Na nutach prawie cały czas zjada pętla Pythona, więc wzór mało waży. Na LM uwaga jest większą częścią kroku. Trafienia i strata z tej różnicy nie wynikają.

Wszystkie przebiegi T1 (420) i T2 (100) miały skończoną stratę, a ostatni krok niższy niż pierwszy.

### Most kanoniczny

Most, w którym tracer pozycji wchodzi tylko przez ΔE, na tiny LM ma stratę wyższą od Softmaxu bez pluginu o 0,0045. Przedział [+0,0028, +0,0063] leży nad zerem i zarazem w ±0,05. To porównanie miesza wzór z tym, co model dostaje. Nie jest werdyktem o samym jądrze.

Stary bench n-ciał, w którym Most wygrywał etykietę warstwy, dawał hierarchię jako one-hot w tracerze, a Softmax jej nie widział. Tego pomiaru tutaj nie powtarzamy.

## Czego to nie obejmuje

- Zgody ciaśniejszej niż ±5 punktów na siatkach. T0 jej nie rozstrzygnął.
- Dużego języka. Tiny LM ma sześć krótkich, powtarzalnych dokumentów i stratę około 0,05. Średniego modelu nie było.
- Kosztu po złożeniu wzoru Mostu w jedno wywołanie. Zmierzony czas jest czasem niezłożonego łańcucha.
- Karty graficznej i NPU. Maszyna ma Radeona 780M i NPU XDNA 16 TOPS (Hawk Point). Laboratorium zostaje na CPU: tensory są małe, a NPU liczy tylko gotowy, skwantyzowany model.

## Gdzie leżą liczby

| dokument | co w nim jest |
|---|---|
| `doswiadczenie/glify/MENDELEJEW_KERNEL_EQUIV.md` | ±5 punktów, tracer w tokenach, 26/30 |
| `doswiadczenie/glify/MENDELEJEW_KERNEL_TOK.md` | token wobec kary ΔE na pierwiastkach |
| `doswiadczenie/glify/NBODY_COMP.md` | Słońce i trójka hierarchiczna |
| `doswiadczenie/glify/JADRO_T0.md` | próg ±2 na tych samych 30 porównaniach |
| `doswiadczenie/glify/JADRO_T1.md` | pisownia nuty jako dodatkowa zmienna |
| `doswiadczenie/glify/JADRO_T2.md` | tiny LM, numer źródła w tokenie |
| `doswiadczenie/glify/JADRO_OGOLNE.md` | program testów i bramka nazwy |
