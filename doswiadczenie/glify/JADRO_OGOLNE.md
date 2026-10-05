# Wnioski i program testów — jądro ogólnego zastosowania

**Data:** 2026-10-05  
**Fork:** `C:\Users\drwis\model_Ai_breit`  
**Komora:** `Most_Lorentz`  
**Stoi na:** [`MENDELEJEW_KERNEL_TOK.md`](MENDELEJEW_KERNEL_TOK.md) · [`MENDELEJEW_KERNEL_EQUIV.md`](MENDELEJEW_KERNEL_EQUIV.md) · [`CROSSDOMAIN_COMP.md`](CROSSDOMAIN_COMP.md) · [`NBODY_COMP.md`](NBODY_COMP.md) · [`../../docs/ARTYKUL_POPULARNY.md`](../../docs/ARTYKUL_POPULARNY.md)

## Wnioski

1. **Tracer jest częścią Mostu.** Wiąże most przez niezerowe ΔE z τ. Bez niego Most rozpada się do zwykłej uwagi. Treść i tracer zostają na osobnych kanałach.
2. **Dla Softmaxu tracer jest pluginem.** Softmax wiąże się z samych składowych treści. Tracer dopisany do tokenu albo jako kara −ΔE²/γ² w wykładniku nie zmienia tego, że jądro jest Softmaxem.
3. **Jako wzory są zamienne tylko wtedy, gdy tracer jest już podany obu w ten sam sposób.** Przy τ w tokenach: 26 z 30 porównań w ±5 punktach, 4 nierozstrzygnięte, w żadnym Most nie jest gorszy. Sam ułamek Breita–Wignera nic nie dokłada (30/30 bez różnicy). Zamiana wzoru nie daje przewagi.
4. **Softmax usprawnia się tracerem wpisanym w token, nie karą w wykładniku.** Na nowych pierwiastkach `sm·tok` bije `sm·ΔE` o +0,06…+0,12. Na katalogu słonecznym kara w wykładniku obniża składanie na znanych ciałach o 0,08. Najlepsze wyniki serii: tracer w tokenie plus osobna relacja kolumny, przy dowolnym jądrze.
5. **Osobna relacja kolumny pomaga, gdy oś jest w stanie, a nie jest najbliższym sąsiadem.** Katalog słoneczny (masa cicha: 1-NN ta sama masa 4%): +5…+11 punktów, wszystkie 5 ramion. Trójka hierarchiczna (rola już jest całym sąsiedztwem): relacja nic nie dodaje, a wmieszana w jedną przestrzeń obniża krok w dół o 0,26.
6. **Ułamek Breita–Wignera nie jest jądrem ogólnego zastosowania.** Tam, gdzie Most wychodzi nad Softmaxem, Softmax jest bez tracera albo dostaje tracer w słabszej postaci.

Porównanie Mostu z tracerem do Softmaxu bez tracera miesza część Mostu z pluginem. Tego porównania nie czytamy jako werdyktu o jądrze.

## Czego brakuje

| # | Luka | Co jest dziś |
|---|------|----------------|
| 1 | Węższy margines | Zgoda ±5 punktów. Na tych zbiorach jedno trafienie przesuwa wynik o kilka punktów. |
| 2 | Reguła dodatkowych zmiennych | Tracer w testach był wektorem ułożonym pod dziedzinę. |
| 3 | Sekwencja, zwłaszcza język, z tą samą informacją dla obu jąder | Równoważność jest na siatkach. Tiny LM: Most 0,637 / Softmax 0,664. Średni: 2,715 / 2,657. Te smoki nie podawały tracera obu stronom tak samo. |
| 4 | Koszt i stabilność | Most był 1,3–1,8 raza wolniejszy. Równoważność trafień tego nie mierzy. |

## Program

Kolejność: **T0, potem T1, potem T2**. T3 mierzy się przy T1 i T2, bez osobnego treningu. Margines, testy i specyfikacja zamrażane przed pierwszym przebiegiem. Holm w obrębie jednego zadania. Ścisłe rozdzielenie tam, gdzie zadanie jest siatką: enkoder w treningu nie widzi elementów testowych.

Poza programem: dostrajanie kanonu (lr, wymiary, β), nowy stos, FFN, promocja wag, powtórka starego bencha n-ciał z one-hotem hierarchii w tracerze. T1 samo nie uprawnia do nazwy „jądro ogólne”.

### T0 — węższy margines, bez nowego treningu

Na istniejących przebiegach z τ w tokenach (`kernel_tok_runs.jsonl`, analiza jak [`../../archiwum/skrypty/analyze_kernel_equivalence.py`](../../archiwum/skrypty/analyze_kernel_equivalence.py)).

- To samo 30 porównań co TOST z 2 października.
- Drugi próg, ustalony teraz: **±2 punkty** (90% CI w całości w ±0,02).
- Zapis: ile porównań zostaje w ±2, ile wychodzi, ile jest nierozstrzygniętych.

**Odczyt:** jeśli wyraźnie mniej niż 26/30 zostaje w ±2, zdanie „wzory są zamienne” obowiązuje przy ±5, nie ciaśniej. Nowego treningu tu nie ma.

### T1 — notacja muzyczna jako dodatkowe zmienne

Dotychczasowa muzyka to 84 nuty i widmo ośmiu harmonicznych. Klasa wysokości jest w widmie w całości (1-NN ta sama kolumna 100%), więc relacja kolumny tylko wzmacniała brzmienie.

Katalog syntetyczny, rozmiar rzędu dotychczasowej muzyki, zamrożony w specyfikacji. Każde zdarzenie ma widmo (jak dotąd) oraz krotkę notacji, która **nie jest funkcją widma**: pisownia enharmoniczna (Cis i Des mają to samo widmo), wartość rytmiczna, miejsce w takcie, dynamika, artykulacja, głos. Kolumna relacji to ta oś notacji, której najbliższy sąsiad liczony na samym widmie nie odtwarza. `raw_sim` raportowany przed treningiem.

Protokół jak [`../../archiwum/skrypty/run_nbody_comp.py`](../../archiwum/skrypty/run_nbody_comp.py): 20 przebiegów, 800 kroków, ramiona bw·ΔE, sm·ΔE, bw·0, sm·0, mlp, warianty comp / PoE / mix, oraz ramię z τ w tokenach dla obu jąder.

Testy zadeklarowane z góry:

| test | pytanie | odczyt pozytywny |
|------|---------|------------------|
| N1 | PoE − comp na kroku wzdłuż kolumny notacji, nowe zdarzenia | jak Słońce: CI nad zerem po Holmie |
| N2 | bw·tok − sm·tok na tych samych miarach | różnica nie przeżywa korekty i mieści się w ±5 punktach |
| N3 | sm·tok − sm·ΔE | token bije karę w wykładniku (CI nad zerem po Holmie) |

**Odczyt negatywny:** kolumna notacji nie niesie sygnału i PoE nic nie daje (jak kana), albo notacja jest już całym sąsiedztwem i mix obniża wynik (jak trójka). T1 zamyka lukę reguły zmiennych na jednej dziedzinie. Nie zamyka języka ani kosztu.

### T2 — język, ta sama informacja dla obu jąder

Punkt startu: istniejący tiny LM ([`../../archiwum/skrypty/run_lm_tiny_smoke.py`](../../archiwum/skrypty/run_lm_tiny_smoke.py), [`../../lm_tiny.py`](../../lm_tiny.py)), nie nowy stos. Średni LM tylko wtedy, gdy T2 na tiny jest skończone i skończone liczbowo.

Przed seedem 0 specyfikacja zamraża trzy warunki i margines straty:

| warunek | co jest podane |
|---------|----------------|
| A | same tokeny treści, bez dodatkowej zmiennej (bw·0 wobec sm·0) |
| B | ta sama dodatkowa zmienna w tokenie dla obu jąder |
| C | Most kanoniczny (tracer tylko przez ΔE) wobec Softmaxu bez pluginu — porównanie mieszane, nie werdykt o jądrze |

Dodatkowa zmienna w B nie może wynikać z sąsiedztwa tokenów. Kandydat do zamrożenia w specyfikacji: etykieta, której nie ma w ciągu (źródło albo znacznik składniowy trzymany z boku). Jeśli korpus tiny jej nie ma, T2 czeka na zadeklarowany zbiór. Nie dobieramy etykiety po wyniku.

Margines straty wpisany w specyfikację przed pierwszym seedem. Odczyt równoważności wzoru: 90% CI różnicy średniej straty (B oraz osobno A) w całości w tym marginesie. Odczyt pluginu: Softmax z B ma niższą stratę niż Softmax z A.

### T3 — koszt i stabilność, przy okazji T1 i T2

Przy każdym ramieniu: czas, oraz czy miara zostaje skończona na każdym przebiegu. Stosunek czasu Most/Softmax wchodzi do raportu. Wolniejsze jądro nie jest przez to porażką.

**Porażka:** którykolwiek przebieg niefinitywny albo strata, która nie schodzi.

## Kiedy wolno powiedzieć „jądro ogólne”

Wszystkie trzy naraz:

- T0 zapisany (wiadomo, czy zgoda jest przy ±5, czy ciaśniej).
- T2: w warunku B wzory równoważne w zamrożonym marginesie, a plugin obniża stratę Softmaxu wobec A.
- T3: oba jądra skończone, stosunek czasu zapisany.

T1 jest do tego potrzebne jako sprawdzenie reguły zmiennych. Samo T1 nie wystarcza.
