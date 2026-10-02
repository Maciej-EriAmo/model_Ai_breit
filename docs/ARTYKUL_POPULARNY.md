# Czy sztuczna inteligencja odkryje układ okresowy?

2 października 2026 · Maciej Mazur

> Eksport z dokumentu Claude (wersja rev 40). Wersja do edycji i komentowania: https://claude.ai/code/artifact/482f7b2c-5de9-477e-8a3e-d3ffcdda5e4d · raporty źródłowe: [`docs/INDEX.md`](INDEX.md) (pozycje 5f2–5f15).

W 1869 roku Dmitrij Mendelejew ułożył pierwiastki w tablicę i zostawił w niej puste miejsca. Przewidział, że brakujące pierwiastki istnieją i jakie będą mieć właściwości. Kilka lat później odkryto gal, skand i german.

Sprawdziliśmy, czy sztuczna inteligencja powtórzy ten krok z samych danych o atomach. **Krótka odpowiedź: jeszcze nie, ale wiemy już, co jej w tym pomaga, a co nie.**

- **Pomaga struktura relacji.** Model, który uczy się osobno „kolumny” i „wiersza” tablicy, nauczył się po niej chodzić. Dwa kroki w dół, których nikt mu nie pokazał, trafiał w 87% przypadków. Brakujący technet stawiał zwykle na 5., a skand na 3. miejscu ze 117 kandydatów. Ta sama zasada zadziałała też dla dźwięków muzycznych.
- **Sam wzór uwagi zaczerpnięty z fizyki nie daje przewagi.** Testowaliśmy model Most, który zamiast zwykłej uwagi używa krzywej rezonansowej. Gdy porównanie było w pełni uczciwe, jego przewaga znikała, ale też niczego nie tracił. Most może więc zastąpić zwykłą uwagę, choć wymaga to dalszych badań.

## Model Most i jego rywale

Współczesne modele językowe działają dzięki mechanizmowi **uwagi**: każdy element danych „patrzy” na inne i decyduje, które są dla niego ważne. Standardowo służy do tego funkcja *softmax*.

**Most** zastępuje ją wzorem z fizyki: krzywą rezonansową Breita–Wignera, tą samą, która opisuje rezonanse w zderzeniach cząstek. Każdy pierwiastek dostaje dodatkowy „znacznik” (z liczby atomowej i konfiguracji elektronowej). Uwaga jest najsilniejsza między pierwiastkami o zbliżonej „energii” tego znacznika, jak dwa kamertony nastrojone na podobny ton.

Żeby wynik coś znaczył, Most zawsze stawał do wyścigu z trzema rywalami:

- **Softmax**: ta sama architektura z klasyczną uwagą; w pierwszych eksperymentach bez znacznika (to okazało się kluczowe, zob. niżej);
- **ZERO**: Most z wyzerowanym znacznikiem, czyli test, czy liczy się sam wzór, czy informacja w znaczniku;
- **MLP**: prosta sieć bez uwagi, która widzi te same dane, ale każdy pierwiastek osobno.

Pierwsze testy powtarzaliśmy 5 razy, kluczowe już 20 razy. Wtedy zestaw zadań był zamrożony z góry, a wyniki przechodziły korektę statystyczną na wielokrotne porównania. Za efekt uznawaliśmy tylko to, co przetrwało tę korektę.

## Siedem kroków: od wkuwania do chodzenia po tablicy

Każdy eksperyment odpowiadał na pytanie, które zostawił poprzedni. Uwaga: w tych krokach Softmax nie dostawał znacznika, więc różnice Most–Softmax mówią tu głównie o wartości samej informacji, a nie o wzorze uwagi.

1. **Mały zestaw, 8 pierwiastków.** Modele z uwagą zapamiętały dane treningowe w 100%, ale na nowych danych wypadły gorzej niż zgadywanie najczęstszej odpowiedzi. Prosta sieć MLP radziła sobie lepiej. Powód: w Moście znacznik decyduje tylko o tym, *na kogo* pierwiastek patrzy, a nie o tym, co wie o sobie.
2. **Pełna tablica, 118 pierwiastków, z podanym okresem i grupą.** Bez żadnych etykiet Most najlepiej pogrupował pierwiastki w bloki s, p, d, f: zgodność z prawdziwymi blokami wyższa o 0,21 niż u Softmaxu bez znacznika, w 5 na 5 powtórzeń.
3. **Tylko liczba atomowa i konfiguracja elektronowa.** Okresy wyłoniły się same, niemal bezbłędnie. Grupy nie wyłoniły się wcale. Najbliższymi „krewnymi” galu okazały się arsen i selen z tego samego wiersza, a nie glin i ind z tej samej kolumny.
4. **Kilka podpowiedzi.** Wystarczyło pokazać 54 dobrze rozrzucone pary „ci dwaj są w tej samej grupie”. Most wskazywał wtedy właściwego krewnego nowego pierwiastka w 32% przypadków, Softmax bez znacznika w 10%. Dokładanie danych bez podpowiedzi *szkodziło*.
5. **Model sam wybiera, o co pytać.** Porażka: pytał prawie wyłącznie o lantanowce, gdzie i tak się nie mylił. Wypadł gorzej niż przy pytaniach losowych.
6. **Każda relacja w osobnej „przestrzeni”.** Wiersze i kolumny przestały się gryźć. Most trafiał krewnych nowych pierwiastków w 41% przypadków, Softmax bez znacznika w 20%. Te same relacje wrzucone do jednej przestrzeni zbijały wynik do zera.
7. **Ruchy po tablicy.** Model uczył się dwóch ruchów: „w dół” (ta sama grupa, następny okres) i „w prawo” (następny pierwiastek). Ruchu „dwa razy w dół” nikt go nie uczył, a Most wykonał go poprawnie w 87% przypadków (rywale bez znacznika 59–65%; 20 powtórzeń). Test opisany niżej pokazał, że to zasługa znacznika, a nie samego wzoru uwagi. Po połączeniu ruchów z relacją grupy krok w dół do *nieznanego* pierwiastka trafiał w 38–39% przypadków, wcześniej w 31%. Tu jednak prosta sieć MLP radziła sobie równie dobrze.

## Luki Mendelejewa w laboratorium

Ukryliśmy przed modelem cztery pierwiastki przewidziane przez Mendelejewa i sprawdziliśmy, jak blisko trafi. Pytanie brzmiało np. „co leży pod glinem?”. Kandydatów było 117, więc losowe zgadywanie dałoby średnio miejsce 59.

| Pierwiastek | Nazwa u Mendelejewa | Odkryty | Miejsce w modelu (na 117) | Co model wskazał obok |
| --- | --- | --- | --: | --- |
| Skand (Sc) | eka-bor | 1879 | **3.**, w 2 na 5 prób 1. | wapń, potas |
| Technet (Tc) | eka-mangan | 1937 | **5.** | ruten, molibden |
| German (Ge) | eka-krzem | 1886 | 10. | glin, bor, fosfor |
| Gal (Ga) | eka-glin | 1875 | 20. | krzem, bor, nihon |

Model trafia w **okolicę** brakującego pierwiastka, ale rzadko w niego samego. Przy technecie wskazuje jego dwóch sąsiadów z wiersza. Przy galu jest już nihon, który leży w tej samej kolumnie co gal.

W innym teście model miał wskazać, *gdzie* w tablicy brakuje pierwiastka. Wyuczone modele wypadły tam na poziomie losowym. Prosta reguła Mendelejewa „ile sąsiednich miejsc jest zajętych” znalazła wszystkie cztery luki.

## Trzy pomyłki, które złapaliśmy po drodze

**Pierwsza: przeciek odpowiedzi.** W jednym z wcześniejszych testów Most wskazywał właściwego krewnego w 70% przypadków, a Softmax w 50%. Wynik wyglądał świetnie i był zawyżony.

Przyczyna: podczas uczenia model widział jako „poprawne odpowiedzi” także pierwiastki, na których później go sprawdzaliśmy. To tak, jakby uczeń znał część odpowiedzi z egzaminu. Po uszczelnieniu testu wyniki spadły do **45% dla Mostu i 31% dla Softmax**. Przewaga Mostu przetrwała, ale była mniejsza, niż się wydawało.

Korektę opisaliśmy w raportach, a wszystkie późniejsze eksperymenty robiliśmy już na szczelnym teście.

**Druga: za mało powtórzeń.** Po 5 powtórzeniach połączenie ruchów z relacją grupy podnosiło trafność z 32% do 48%, a Most wygrywał ze wszystkimi rywalami. Po 20 powtórzeniach na zamrożonym zestawie zadań zysk wyniósł 31% → 38%. Przewaga Mostu nad prostą siecią MLP zniknęła.

**Trzecia: model widział za dużo.** W czasie nauki model widział cechy wszystkich pierwiastków, także testowych (bez odpowiedzi). Gdy elementy testowe całkowicie ukryliśmy przed nauką, zniknęła jedyna przewaga, którą wzór Breita–Wignera miał w uczciwym porównaniu.

Każda z tych pomyłek działała na korzyść Mostu. Bez kontroli wnioski byłyby znacznie bardziej optymistyczne, niż pozwalają dane.

## Czy to zasługa fizyki? Test samego wzoru uwagi

**Przewaga Mostu wynikała ze znacznika, którego Softmax nie dostawał, a nie z samego wzoru Breita–Wignera.**

We wcześniejszych krokach Softmax nie widział znacznika. Porównanie mieszało więc dwie rzeczy: wzór uwagi i dostęp do informacji. Zbudowaliśmy zatem jeden model, w którym zmienia się *wyłącznie* wzór. Te same dane, ta sama liczba parametrów, ten sam start uczenia i ten sam znacznik dla obu wersji, 20 powtórzeń.

- **Składanie ruchów na znanych pierwiastkach:** Softmax ze znacznikiem dorównał Mostowi, a nawet go lekko wyprzedził (93% wobec 87% przy „dwa razy w dół”).
- **Nowe pierwiastki:** wzór Breita–Wignera dał wyraźny zysk, 7–11 punktów procentowych w 18–19 na 20 powtórzeń, ale tylko w jednym ustawieniu: przy łączeniu ruchów z relacją grupy. W pozostałych zysk był mały i statystycznie niepewny. Także ten jeden zysk zniknął, gdy model nie widział w nauce pierwiastków testowych (trzecia pomyłka).
- **Bez znacznika** oba wzory działały tak samo.

Potem daliśmy znacznik obu wersjom w standardowy sposób: dopisany do opisu każdego pierwiastka, tak jak robią to zwykłe transformery. **Różnica między wzorami zniknęła wtedy całkowicie** we wszystkich 30 porównaniach. Co więcej, zwykły Softmax ze znacznikiem w opisie wypadał lepiej od Mostu w jego pierwotnej postaci (znacznik tylko przez wzór uwagi) w 7 z 15 testów, a gorzej tylko w 2. Najlepsze wyniki całej serii dał znacznik w opisie pierwiastka plus relacja grupy, niezależnie od wzoru uwagi.

Wszystkie wcześniejsze porównania Most–Softmax w tym tekście dotyczą więc Softmaxu bez znacznika i mówią głównie o wartości samej informacji.

**Czy Most może więc zastąpić Softmax? W testowanych warunkach tak.** Brak różnicy to jeszcze nie dowód, że dwa modele są równie dobre. Sprawdziliśmy to więc osobnym testem równoważności: czy różnica na pewno mieści się w 5 punktach procentowych.

- **Znacznik podany w standardowy sposób:** równoważne w 26 z 30 porównań, pozostałe 4 nierozstrzygnięte, w żadnym Most nie był gorszy.
- **Znacznik tylko przez wzór uwagi:** wynik zależał od tablicy. W jednych ustawieniach lepszy był Most, w innych Softmax.

Zamiana nie daje więc przewagi, ale też nie szkodzi. To, czy Most sprawdzi się w dużych zadaniach, na przykład w modelach języka, wymaga dalszych badań.

## Czy to działa poza chemią?

**Struktura relacyjna działa także poza chemią, wzór Breita–Wignera nie.**

To samo zadanie, „krok w dół, krok w prawo”, powtórzyliśmy na dwóch innych tablicach. Tym razem model w czasie nauki w ogóle nie widział elementów testowych.

| Tablica | Kolumny | Co model widzi | Czy relacja kolumny pomogła? |
| --- | --- | --- | --- |
| Pierwiastki | grupy | konfigurację elektronową | tak, umiarkowanie |
| Dźwięki muzyczne (C1–B7) | nuty C, C#, D… | widmo harmonicznych | **tak, najsilniej: +7 do +17 punktów** |
| Japońskie kana (gojūon) | samogłoski a, i, u, e, o | kształt znaku | nie |

Relacja kolumny pomaga, gdy cechy niosą choćby ślad tej struktury. W muzyce nuty oddalone o oktawę mają wspólne harmoniczne. Kształt japońskiego znaku nie zdradza samogłoski, więc tam nie pomogło nic i żaden model nie odgadł nowych znaków. We wszystkich trzech tablicach wzór Breita–Wignera nie dał przewagi.

## Czego się nauczyliśmy

- **Podobieństwo w danych jest „poziome”, a genialny krok Mendelejewa był „pionowy”.** Konfiguracje elektronowe najbardziej upodabniają sąsiadów z wiersza. Mendelejew dostrzegł podobieństwo w kolumnie, między pierwiastkami bardzo różnymi masą. Bez podpowiedzi model tego nie zobaczył.
- **Osobna relacja dla osobnej osi.** Gdy wiersze i kolumny trafiały do jednej przestrzeni, kolumny przegrywały. Rozdzielenie relacji poprawiało przewidywania dla nowych elementów: w pierwiastkach, a jeszcze silniej w muzyce, niezależnie od wzoru uwagi.
- **Informacja jest ważniejsza niż wzór.** Znacznik z konfiguracji elektronowej dawał duże zyski. Wzór Breita–Wignera nie dawał żadnych, gdy obie wersje dostały znacznik w ten sam sposób. Może jednak zastąpić zwykłą uwagę bez straty.
- **Kilka dobrze dobranych przykładów jest warte więcej niż wiele przypadkowych.** 54 pary rozrzucone po wszystkich grupach dały więcej niż ponad 300 par zdominowanych przez jedną grupę.
- **Model, który sam wybiera pytania, wpada w błędne koło.** Pyta tam, gdzie już się zgadza, więc niczego nowego się nie uczy.
- **Uczciwe porównanie zmienia wnioski.** Te same dane i ten sam sposób ich podania, więcej powtórzeń, pełne ukrycie testów: każdy z tych kroków zmniejszał przewagę nowego modelu.

## Ograniczenia i co dalej

To eksperyment laboratoryjny, a nie odkrycie nowego pierwiastka.

- **Mała skala.** Pierwiastków jest tylko 118, japońskich znaków 45. W każdym teście sprawdzaliśmy kilkadziesiąt elementów, więc jedno trafienie przesuwa wynik o kilka punktów procentowych.
- **Część testów miała tylko 5 powtórzeń.** Kluczowe wnioski (ruchy po tablicy, wzór uwagi, inne dziedziny) opierają się na 20 powtórzeniach z korektą statystyczną. Wcześniejsze kroki traktujemy jako wstępne.
- **Cechy wybrane przez ludzi.** Sposób zapisu konfiguracji elektronowej sami zaprojektowaliśmy, a on wpływa na to, co model może zobaczyć. Dźwięki muzyczne to idealne, wygenerowane widma, a nie nagrania.
- **Pierwiastki 119 i 120.** Teoria przewiduje dla nich nowy poziom elektronowy, którego model nigdy nie widział. Żaden wariant nie wskazał go jako najbardziej prawdopodobnego.

Następne kroki: sprawdzić zasadę „osobna relacja dla osobnej osi” na prawdziwych danych spoza fizyki, np. na odmianie słów (osoba × czas), oraz sprawdzić, czy model potrafi sam odkryć, które osie istnieją, zamiast dostawać je od nas. Osobno warto sprawdzić jądro Most w dużych zadaniach, np. w modelu języka: jego jakość, koszt obliczeń i stabilność uczenia.

## Słowniczek

| Pojęcie | Co znaczy |
| --- | --- |
| Uwaga (attention) | Mechanizm, w którym każdy element danych decyduje, na które inne patrzeć |
| Softmax | Standardowy sposób liczenia uwagi w modelach językowych |
| Rezonans Breita–Wignera | Wzór z fizyki cząstek; w Moście uwaga jest najsilniejsza między elementami o zbliżonej „energii” |
| Znacznik | Dodatkowa informacja o elemencie (u pierwiastków: z konfiguracji elektronowej), z której Most liczy „energię” |
| Uczenie nadzorowane | Model dostaje przykłady z poprawnymi odpowiedziami |
| Uczenie nienadzorowane | Model szuka struktury sam, bez odpowiedzi |
| Uczenie ze wzmocnieniem | Model dostaje tylko nagrodę: „dobrze” albo „źle” |
| Uczenie częściowo nadzorowane | Kilka przykładów z odpowiedziami plus dużo danych bez nich |
| Uczenie relacyjne | Model uczy się powiązań między obiektami, np. „ta sama grupa” |
| Relacje kompozycyjne | Proste ruchy, które model potrafi składać w nowe, np. dwa razy „w dół” |
| Przeciek danych | Błąd, w którym model widzi w nauce odpowiedzi z testu |
| Ścisłe rozdzielenie | Elementy testowe są całkowicie ukryte w czasie nauki, nawet bez odpowiedzi |
