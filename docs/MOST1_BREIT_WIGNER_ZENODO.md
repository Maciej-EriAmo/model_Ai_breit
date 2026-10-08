# Most-1: jądro uwagi Breita–Wignera z szerokością zależną od wiersza

**Autor:** Maciej Mazur (badacz niezależny, Warszawa)
**Wersja dokumentu:** 2026-10-07
**Seria pomiarów:** 2026-10-05 i 2026-10-06
**Kod:** fork `model_Ai_breit` — [uzupełnić adres repozytorium]
**Licencja tekstu:** Creative Commons Uznanie autorstwa 4.0 Międzynarodowe (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/deed.pl
**Licencja kodu:** MIT — plik `LICENSE-MIT` w forku `model_Ai_breit`

---

## Streszczenie

Opisuję Most-1: jądro uwagi, które zamiast funkcji wykładniczej liczy wagę ułamkiem Breita–Wignera. Treść tokenu buduje sprzężenie $C = \mathrm{softplus}(QK^\top)^2$. Osobna zmienna trzymana obok tokenu, tracer, buduje rozstrojenie $\Delta E$. Waga jest proporcjonalna do $C/(\gamma^2+\Delta E^2)$. Nowością wobec wcześniejszego jądra Lorentza jest szerokość $\gamma$, która nie jest już stałą głowy, tylko własnością wiersza: $\gamma_i = \gamma_0(1+\beta\kappa_i)$, gdzie $\kappa_i$ jest udziałem największego sprzężenia w wierszu.

Dokument podaje wzór tak, jak liczy go kod, kontrakt numeryczny (maski przed $\kappa$, pusty wiersz, podłogi, seal temperatury), trzy miejsca wywołania jądra oraz serię pomiarów na CPU przeciw bliźniaczemu softmaxowi.

Wynik pomiarów jest ostrożny. Gdy oba jądra dostają ten sam opis, Most i softmax są zamienne w granicy ±5 punktów na zadaniach siatkowych i ±0,05 straty na małym modelu językowym. Sam ułamek Breita–Wignera punktów nie dokłada. Wcześniejsze przewagi Mostu w tej serii okazały się efektem informacji z tracera, a nie wzoru: znikają, gdy softmax dostaje tracer wpisany w token. Na WikiText-2, przy modelu około 3,3 mln wag, Most ma stratę testową 1,580 wobec 1,627 softmaxu (2 seedy). Krok Mostu trwa około 1,1–1,5 raza dłużej. Duży model językowy nie był liczony i dokument nie nazywa jądra ogólnym.

**Słowa kluczowe:** mechanizm uwagi, Breit–Wigner, profil Lorentza, softmax, tracer, szerokość rezonansu, transformer, pomiar równoważności

---

## 1. Zakres

### 1.1. Co jest jądrem

Jądro to dwie funkcje w pliku `breit_kernel.py`:

- `breit_wigner_rows` zwraca surowe amplitudy $A$, w tym samym kształcie co $C$;
- `breit_wigner_weights` zwraca te amplitudy podzielone przez sumę po kluczach.

Obie dostają gotowe $C$ i $\Delta E$. Nie liczą $Q$, $K$ ani projekcji energii. Wszystko, co buduje te dwa tensory, nazywam dalej szkieletem: projekcje `QKV`, projekcję energii, maski, seal, FFN, LayerNorm, CLS i głowę klasy. Zmiana FFN albo `d_model` nie jest zmianą jądra. Zmiana kolejności maski względem $\kappa$ jest.

### 1.2. Co ten dokument twierdzi

1. Most-1 jest poprawnie określonym jądrem uwagi: wagi są skończone, nieujemne, w żywym wierszu sumują się do 1, a pusty wiersz daje zera.
2. Przy tym samym opisie podanym obu jądrom Most-1 i softmax dają wyniki w granicy ±5 punktów na zmierzonych siatkach oraz ±0,05 straty na małym modelu językowym.
3. Jądro uczy się na tekście: na WikiText-2, przy około 3,3 mln wag, strata jest skończona, malejąca i nie gorsza od bliźniaczego softmaxu.

### 1.3. Czego nie twierdzi

- Że zgoda jest ciaśniejsza niż ±5 punktów. Przy progu ±2 większość porównań jest nierozstrzygnięta.
- Że jądro jest ogólne. Model 100–300 mln wag nie był liczony.
- Że Most jest szybszy. Zmierzony czas jest czasem niezłożonego łańcucha operacji.
- Że znak $\Delta E$ niesie informację. W tym forku do wagi wchodzi tylko $\Delta E^2$.

---

## 2. Słowa i oznaczenia

**Token** to opis, który model i tak czyta: widmo nuty, składowe ciała, bajty tekstu.

**Tracer** to zmienna trzymana obok tego opisu. Dla nuty jest nią pisownia, rytm, takt i dynamika, dla małego modelu językowego numer dokumentu albo pozycja w oknie. W Moście tracer jest częścią mechanizmu, bo to on daje $\Delta E$. Dla softmaxu ta sama zmienna jest dodatkiem: można ją wpisać w token albo włożyć jako karę $-\Delta E^2/\gamma^2$ w wykładnik.

**Wzór uwagi** zamienia opisy na wagi, czyli rozstrzyga, kogo model słucha mocniej.

**Relacja kolumny** to osobna oś, a nie wzór uwagi. „Dół” znaczy: ta sama kolumna, następny wiersz.

| Symbol | Znaczenie |
|--------|-----------|
| $S_{ij}$ | iloczyn skalarny zapytania i klucza, podzielony przez $\sqrt{d_h}$ |
| $C_{ij}$ | sprzężenie, nieujemne |
| $\tau_i$ | tracer pozycji $i$ |
| $E_i$ | energia pozycji $i$, liniowa projekcja tracera |
| $\Delta E_{ij}$ | rozstrojenie pary |
| $\vartheta$ | temperatura sprzężenia (`couple_temp`); nie jest tracerem |
| $e_{\mathrm{scale}}$ | skala energii |
| $\kappa_i$ | udział największego sprzężenia w wierszu $i$ |
| $\gamma_0$, $\beta$ | uczone parametry głowy |
| $\gamma_i$ | szerokość wiersza $i$ |
| $n_i$ | liczba kluczy o dodatnim $C$ w wierszu $i$ |
| $T$ | oś 3 tracera w stylu Karmazyn: temperatura stanu, czytana przez seal |

---

## 3. Szkielet: skąd biorą się $C$ i $\Delta E$

Dla głowy o wymiarze $d_h$:

$$
S_{ij}=\frac{Q_i\cdot K_j}{\sqrt{d_h}},\qquad
C_{ij}=\mathrm{softplus}\!\left(\frac{S_{ij}}{\vartheta}\right)^{2}\cdot \mathrm{keep}_{ij}
$$

$$
E_i=W_e\,\tau_i+b_e,\qquad
\Delta E_{ij}=e_{\mathrm{scale}}\,(E_i-E_j)
$$

Treść buduje zapytanie, klucz, wartość i sprzężenie. Tracer buduje energię. Mnożnik $\mathrm{keep}$ niesie maski i jest opisany w rozdziale 5.

Ponieważ $\mathrm{softplus}(\cdot)^2>0$, wiersz $C$ może być zerowy tylko przez maski.

Bias $b_e$ skraca się w różnicy $E_i-E_j$, więc dla $\Delta E$ między dwoma atomami gradient do $b_e$ wynosi zero. Wyjątkiem jest para z CLS: tracer CLS jest zerem, więc energia CLS to sam bias i różnica CLS–atom ten bias widzi.

Wagi projekcji energii startują od jedynek, bias od zera. Energia głowy na starcie jest więc sumą osi tracera, identyczną dla wszystkich głów. Symetrię głów łamią dopiero projekcje `QKV`. Druga konsekwencja: w tracerze kanonu trzy osie są jedynką (one-hot) głębokości, ich suma wynosi zawsze 1 i na starcie skraca się w $\Delta E$. Na starcie rozstrojenie widzi tylko pozostałe osie. Kolumny głębokości nie są martwe, bo gradient do nich jest niezerowy.

---

## 4. Jądro

### 4.1. Poprzednik: Lorentz

Wcześniejsze jądro (drzewo `model Ai`, zamrożone) ma jedną szerokość na głowę:

$$
\gamma=\mathrm{softplus}(\log\gamma)+10^{-3},\qquad
A_{ij}=\frac{\gamma\,C_{ij}}{\gamma^{2}+\Delta E_{ij}^{2}}
$$

Most między pozycją $i$ a pamięcią $j$ otwiera się, gdy $\Delta E_{ij}$ jest małe, i gaśnie, gdy rozstrojenie rośnie. Nie ma tam $\beta$ ani $\kappa$. Kod Lorentza zawiera też dwa tryby czytające znak $\Delta E$ (`directed`, `onesided`), domyślnie wyłączone. W forku Breita–Wignera oba są odrzucane przy starcie.

### 4.2. Most-1: szerokość jest własnością wiersza

Wejścia jądra:

| Tensor | Znaczenie | Kształt |
|--------|-----------|---------|
| `coupling` | sprzężenie $C$ | dowolny, ostatnia oś to klucze |
| `delta_e` | rozstrojenie $\Delta E$ | identyczny jak $C$ |
| `log_gamma0` | parametr głowy przed softplus | broadcast do $C$ |
| `beta_raw` | parametr głowy przed softplus | broadcast do $C$ |

W uwadze typowy kształt $C$ i $\Delta E$ to `[B, h, N, N]`, a parametry głowy mają kształt `[h, 1, 1]`. Niezgodny kształt $C$ i $\Delta E$ podnosi `ValueError`.

Szerokość:

$$
\mathrm{mass}_i=\max\Bigl(\sum_k C_{ik},\;10^{-12}\Bigr),\qquad
\kappa_i=\frac{\max_j C_{ij}}{\mathrm{mass}_i}
$$

$$
\gamma_0=\mathrm{softplus}(\log\gamma_0)+10^{-3},\qquad
\beta=\mathrm{softplus}(\beta_{\mathrm{raw}}),\qquad
\gamma_i=\gamma_0\,(1+\beta\,\kappa_i)
$$

Amplituda (`breit_wigner_rows`):

$$
A_{ij}=\frac{\gamma_i\,C_{ij}}{\gamma_i^{2}+\Delta E_{ij}^{2}+10^{-12}}
$$

Waga (`breit_wigner_weights`):

$$
w_{ij}=\frac{A_{ij}}{\max\bigl(\sum_k A_{ik},\;10^{-12}\bigr)}
$$

W jednym wierszu $\gamma_i$ w liczniku jest stałe i skraca się przy normalizacji, więc `breit_wigner_weights` pomija je w liczniku. Efektywnie

$$
w_{ij}\;\propto\;\frac{C_{ij}}{\gamma_i^{2}+\Delta E_{ij}^{2}}.
$$

Licznik $\gamma_i$ ma znaczenie tylko tam, gdzie używa się nieznormalizowanego $A$.

$\kappa$ bliskie 1 znaczy, że jeden klucz wziął sprzężenie: wiersz dostaje szerszą szczelinę i przepuszcza większe $\Delta E$. $\kappa$ małe znaczy sprzężenie rozlane po wielu kluczach: szczelina się zwęża.

Jądro liczy jeden krok: $\gamma$ z aktualnego $C$, potem jedno $A$. Nie ma pętli ani stanu między krokami. Wariant iterowany (Most-K) jest w osobnej bibliotece, patrz rozdział 7.

### 4.3. Start

Parametry startują od $\log\gamma_0=0$ i $\beta_{\mathrm{raw}}=0$:

$$
\gamma_0=\ln 2+10^{-3}\approx 0{,}6941,\qquad \beta=\ln 2\approx 0{,}6931.
$$

Szerokość od pierwszego kroku zależy od szczytu wiersza i leży między $\gamma_0\approx 0{,}694$ (granica długiego, płaskiego wiersza) a $\gamma_0(1+\beta)\approx 1{,}175$ (jeden klucz bierze wszystko).

### 4.4. Własności

**Zakres $\kappa$.** Dla niepustego wiersza $\kappa_i\in[1/n_i,\,1]$. Zero osiąga tylko pusty wiersz, przez podłogę sumy.

**Szerokość zależy od długości wiersza.** Najwęższa szerokość wiersza o $n_i$ otwartych kluczach wynosi

$$
\gamma_i^{\min}=\gamma_0\,(1+\beta/n_i),
$$

więc $\gamma_0$ osiąga się tylko w granicy długich wierszy. W modelu językowym z maską przyczynową wiersz $i$ ma $i+1$ kluczy. Pozycja 0 ma zawsze $\kappa=1$ i największą możliwą szerokość, a wczesne pozycje mają z konstrukcji szerszą szczelinę. Jest to bias pozycyjny wbudowany w mechanizm, niezależny od tracera.

**Znak rozstrojenia nie wchodzi do wagi.** $\Delta E$ wchodzi jako kwadrat. Zamiana znaku nie zmienia $A$.

**Większe $\gamma$ spłaszcza szczyt**, gdy $\Delta E$ nie jest zerem.

**Gradient przez $\kappa$.** $\kappa$ nie jest odłączane od grafu. Ponieważ zależy od $Q$ i $K$, szerokość otwiera dodatkową ścieżkę gradientu do projekcji `QK`. Operacja $\max$ przepuszcza gradient tylko przez pozycję argmax. Ta ścieżka jest częścią Most-1.

**Tracer nie ustawia szerokości.** $\kappa$ bierze się ze sprzężenia, $\gamma_0$ i $\beta$ są wagami. Tracer wchodzi tylko przez $\Delta E$ i, w tracerze stylu Karmazyn, przez seal.

### 4.5. Przypadek ZERO

Wyzerowany tracer daje $\Delta E=0$. Tak samo działa każdy tracer stały dla wszystkich pozycji. Szerokość jest wtedy wspólna dla wiersza i skraca się przy normalizacji:

$$
w_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
$$

Most się nie formuje i zostaje samo sprzężenie treści. Jest to naturalna ablacja jądra: wynik z prawdziwym tracerem porównany z ZERO mierzy wkład geometrii ponad samo sprzężenie.

### 4.6. Uwaga analityczna: Most-1 zapisany jako softmax

Ta uwaga jest tożsamością algebraiczną, a nie wynikiem pomiaru. Dla kluczy o dodatnim $C$ waga Most-1 jest softmaxem po innych logitach:

$$
w_{ij}=\mathrm{softmax}_j\Bigl(\,2\ln\mathrm{softplus}(S_{ij}/\vartheta)\;+\;\ln\mathrm{keep}_{ij}\;-\;\ln\bigl(\gamma_i^{2}+\Delta E_{ij}^{2}\bigr)\Bigr).
$$

Wynikają z niej trzy rzeczy.

1. Dla silnie ujemnych $S$ jest $\mathrm{softplus}(S)\approx e^{S}$, więc człon treści dąży do $2S$: Most zachowuje się jak softmax o połowie temperatury. Dla dużych dodatnich $S$ człon treści rośnie jak $2\ln S$, czyli logarytmicznie. Most jest tam łagodniejszy od softmaxu i trudniej mu skupić wagę na jednym kluczu.
2. Dla małego $|\Delta E|/\gamma$ jest $-\ln(\gamma^2+\Delta E^2)\approx -2\ln\gamma-\Delta E^2/\gamma^2$. Człon energii zgadza się w pierwszym rzędzie z karą gaussowską $-\Delta E^2/\gamma^2$ użytą w softmaxie kontrolnym. Różnica jest w ogonach: profil Lorentza (Cauchy'ego) gaśnie potęgowo, kara gaussowska wykładniczo.
3. Jedyną częścią Most-1, której nie da się zapisać jako stałych logitów, jest zależność $\gamma_i$ od wiersza przez $\kappa_i$.

Ta postać jest też punktem wyjścia do złożenia jądra w jedno wywołanie biblioteki (rozdział 11), pod warunkiem testu równości wag.

---

## 5. Kontrakt numeryczny

Każda zmiana jądra ma zachować punkty poniżej.

### 5.1. Maski wchodzą do $C$ przed $\kappa$

$\kappa$ czyta cały wiersz $C$. Klucz zabroniony musi mieć $C=0$, zanim liczone są maksimum i suma. Inaczej przyszły token, padding albo zimny klucz zmienia szerokość wiersza, a w modelu językowym informacja z przyszłości przecieka do $\gamma_i$.

W enkoderze i klasyfikatorze mnożnik $\mathrm{keep}$ składa trzy rzeczy:

1. seal temperatury na kluczu,
2. padding (pozycja CLS zostaje otwarta),
3. zero na przekątnej.

$\mathrm{keep}$ jest liczone raz na forward, bo zależy od tracera i maski, a nie od warstwy.

W małym modelu językowym maska przyczynowa zeruje przyszłe klucze na $C$ przed wywołaniem jądra. Przekątna zostaje. Seal nie jest wołany.

To odróżnia fork od Lorentza, w którym przekątna i seal były nakładane na gotową wagę $A$, po jądrze. Tam było to obojętne, bo szerokość nie zależała od wiersza.

### 5.2. Pusty wiersz zostaje skończony

Gdy suma $C$ jest zerem, podłoga $10^{-12}$ daje $\kappa=0$ i $\gamma=\gamma_0$. Surowe $A$ jest zerem. Po normalizacji wiersz jest zerem, a nie `NaN`. Przykład z enkodera: wiersz CLS ma zero na przekątnej, a wszystkie pozostałe klucze są zasealowane.

Kontrola `assert_finite` przerywa forward, gdy $S$, $C$, $\Delta E$ albo waga nie są skończone.

### 5.3. Podłogi

| Miejsce | Wartość | Po co |
|---------|---------|-------|
| $\gamma_0$ | $\mathrm{softplus}(\log\gamma_0)+10^{-3}$ | szerokość nie schodzi do zera |
| suma $C$ w $\kappa$ | podłoga $10^{-12}$ | pusty wiersz |
| mianownik $A$ | $+10^{-12}$ | zbędne przy $\gamma\ge10^{-3}$ (wtedy $\gamma^2\ge10^{-6}$); zostaje |
| suma wag | podłoga $10^{-12}$ | pusty wiersz daje zero, nie dzielenie przez zero |

### 5.4. Gwarancje z konstrukcji

- $\gamma_i\ge\gamma_0\ge10^{-3}$. Równość $\gamma_i=\gamma_0$ zachodzi tylko dla pustego wiersza.
- Mianownik $\gamma^2+\Delta E^2$ jest dodatni, więc $A$ jest skończone przy każdym $\Delta E$.
- $C>0$ poza maskami, a po normalizacji $w_{ij}\in[0,1]$.
- Suma wag żywego wiersza wynosi 1. Wiersz samych zer zostaje wierszem zer.

### 5.5. Seal temperatury

Seal jest częścią szkieletu, nie jądra, ale wchodzi do $C$ przed $\kappa$, więc należy do kontraktu. Czyta oś 3 tracera ($T$) i domyślnie jest włączony.

| Warunek | Skutek na sprzężeniu klucza |
|---------|------------------------------|
| $T<0{,}20$ | twarde zero |
| $0{,}20\le T<0{,}45$ | mnożnik $\exp\bigl(-12\,(0{,}45-T)\bigr)$ |
| $T=0{,}20$ | mnożnik $e^{-3}\approx0{,}0498$, bez twardego zera |
| $T\ge0{,}45$ | mnożnik 1 |
| pozycja 0 (CLS) | zawsze otwarta |
| brak osi 3 | seal nic nie robi |

Przy $T=0{,}45$ przejście jest ciągłe. Przy $T=0{,}20$ mnożnik skacze z $0{,}0498$ do 0. Czy ten skok jest zamierzony, pozostaje pytaniem otwartym.

Ciągła kara w paśmie zimnym też wchodzi do $\kappa$: tłumi część kluczy, podnosi udział szczytu i przesuwa $\gamma$ całego wiersza.

Seal ma strażnika ablacji. Gdy cały tracer atomów poza CLS jest zerem, seal zwraca sprzężenie bez zmian. Prawdziwy atom, nawet zarchiwizowany, ma niezerową jedynkę głębokości, więc odróżnia się od sztucznego zera. Ablacja ZERO z rozdziału 4.5 daje więc pełną macierz $C$ także przy włączonym sealu i nie wymaga osobnego przebiegu.

Oś 3 jest temperaturą archiwum tylko w tracerze stylu Karmazyn. W innych źródłach (struktura, QM, zdjęcia glifów) jest energią albo barwą i seal trzeba tam wyłączyć. W modelu językowym tracer pozycji ma na tej osi kosinus pozycji i seal nie może go czytać.

Bliźniak softmax sealu nie używa.

### 5.6. Bliźniak softmax

Softmax kontrolny wycina przekątną i padding wartością $-10^{9}$ przed wywołaniem `softmax`, potem jeszcze raz zeruje te pozycje i normalizuje żywy wiersz. Kolejność jest istotna: przy własnym logicie około 80 wycięcie przekątnej dopiero po softmaxie zeruje cały wiersz w `float32`.

W modelu językowym softmax wycina przyszłość tą samą maską przyczynową i nie zeruje przekątnej.

---

## 6. Trzy wywołania jądra

Wzór jest jeden. Wywołania różnią się tym, skąd biorą $C$ i $\Delta E$ oraz co zerują przed $\kappa$.

| Wołający | Funkcja jądra | Maska przed $\kappa$ | $\vartheta$, $e_{\mathrm{scale}}$ |
|----------|----------------|----------------------|-----------------------------------|
| klasyfikator `BridgeTransformer` | `breit_wigner_weights` | seal, padding, przekątna | stałe, równe 1 |
| enkoder `PhiBridgeEncoder` | `breit_wigner_weights` | seal, padding, przekątna | gałki, domyślnie 1 |
| blok językowy `_MostCausalBlock` | `breit_wigner_rows` + ręczna normalizacja | tylko przyszłość | gałki, domyślnie 1 |

**Klasyfikator** dokleja token CLS do treści i zerowy wiersz CLS do tracera, liczy $\mathrm{keep}$ raz i woła jądro w każdej warstwie z jej własnymi $\log\gamma_0$ i $\beta_{\mathrm{raw}}$.

**Enkoder** zwraca embeddingi tokenów bez CLS. Temperatura sprzężenia jest obcinana od dołu do $10^{-3}$. Jego bliźniak softmax ignoruje tracer.

**Blok językowy** używa tracera pozycji: $t$, $1-t$, sinus, kosinus, z $t\in[0,1]$ wzdłuż okna. Pozycja jest też w strumieniu tokenu, tak samo u Mostu i u softmaxu. Kanał $\Delta E$ nie jest tu czystą pozycją: mieszanie tracera (`tracer_mix`) widzi także embedding bajta, więc w modelu językowym do energii wchodzi również treść. Woła `breit_wigner_rows` i dzieli wynik przez sumę. Przy $\gamma$ stałym w wierszu jest to ten sam rozkład co `breit_wigner_weights`, co trzyma osobny test. Numer dokumentu źródłowego może wejść osobnym wejściem bocznym, domyślnie wyłączonym. Bliźniak softmax nie ma projekcji energii.

### 6.1. Rozmiary

Kanon enkodera: sprzężenie `softplus`, 2 warstwy, `d_model = 32`, 2 głowy ($d_h=16$), czyli 4 jądra. Treść `d_x = 8`, tracer 6 osi: jedynka głębokości (3 osie), $T$, stan, mazur. Seal włączony z progami 0,20 i 0,45 oraz wykładnikiem 12.

Średni model językowy jest osobnym presetem: `d = 256`, 4 warstwy, 4 głowy, słownik 256 (bajty), około 3,3 mln wag.

---

## 7. Trzy implementacje

| | Lorentz (`model Ai`) | Fork Most-1 (`model_Ai_breit`) | Biblioteka TSRA (`transformer2`) |
|--|----------------------|--------------------------------|-----------------------------------|
| Status | zamrożony oryginał | przedmiot tego dokumentu | osobna biblioteka numeryczna jednej głowy |
| Szerokość | jedna na głowę | $\gamma_0(1+\beta\kappa)$ na wiersz | $\gamma_0(1+\beta\kappa)$ |
| Kto buduje $C$ i $\Delta E$ | sieć | sieć, z $QK$ i tracera | wołający, z zewnątrz |
| $\kappa$ | brak | $\max(C)/\sum C$ | domyślnie $\max(S)$ ze stanu; wariant `lse` z $\tau=0{,}05$ |
| Krok | jeden | jeden | Most-1 to $K=1$, $\rho=1$; Most-K iteruje |
| Znak $\Delta E$ | dostępny, domyślnie wyłączony | odrzucony | — |
| Maski, seal | na $A$, po jądrze | na $C$, przed $\kappa$ | opcjonalna maska |
| Podłoga $\gamma_0$ | $\mathrm{softplus}+10^{-3}$ | $\mathrm{softplus}+10^{-3}$ | $\max(\gamma_0,10^{-8})$ |
| Pusty nośnik | wiersz zer | wiersz zer | `ValueError` |
| Kształt | `[B, h, N, N]` | `[B, h, N, N]` | `(N,)` albo `(B, N)`: jeden rozkład na wiersz |
| Uczenie | propagacja wsteczna | propagacja wsteczna | `update_inputs` rusza $C$ i $\Delta E$ ze skalaru wołającego |

Fork nie importuje biblioteki TSRA. Podłogi i kontrakty obu drzew są różne i nie są zrównywane.

**Związek obu wzorów.** W TSRA Most-1 to jeden strzał $S_1=W\bigl(\gamma(\kappa(S_0))\bigr)$, gdzie stan początkowy $S_0$ jest osobnym wejściem. Fork liczy $\kappa$ z $C/\sum C$, czyli odpowiada temu samemu strzałowi ze stanem początkowym równym znormalizowanemu sprzężeniu.

### 7.1. Most-K i mapa faz

Pętla $S\leftarrow(1-\rho)\,S+\rho\,W$ istnieje tylko w TSRA. $\kappa$ bierze się tam ze stanu $S$, a $C$ i $\Delta E$ są zamrożone w pętli. Głowa nie ma własnych wag: $\beta$, $\rho$ i $\gamma_0$ są pokrętłami siatki.

Mapa faz (2026-09-30): $N=32$, 100 startów z rozkładu Dirichleta, $K=64$, $\gamma_0=0{,}3$, $\beta\in\{0;\,0{,}25;\,0{,}5;\,1;\,2;\,4;\,8\}$, $\rho\in\{0{,}2;\,0{,}4;\,0{,}6;\,0{,}8;\,1\}$, $\kappa=\max(S)$. Promień widma liczony jest z jakobianu jednego kroku $J=(1-\rho)\,I+\rho\,\partial W/\partial S$.

- Reżimy dwóch studni i niemal zdegenerowany zbiegają w każdej komórce siatki. Niemal zdegenerowany zostaje płaski, z entropią około $\ln 32\approx3{,}466$.
- Jedna studnia przy $\beta=8$, $\rho=1$ dzwoni, ale zbiega: przy $K=200$ mediana 150 kroków, promień 0,914. Przy $K=64$ ucięta trajektoria dostaje etykietę oscylacji.
- Powyżej tej siatki, na jednej studni przy $\rho=1$, czysty okres 2 idzie w parze z promieniem ponad 1:

| $\beta$ | mediana kroków | mediana promienia $J$ | etykieta |
|---------|----------------|-----------------------|----------|
| 8 | 150 | 0,914 | zbiega |
| 12 | 200 | 1,605 | oscyluje |
| 16 | 200 | 2,871 | oscyluje |
| 24 | 200 | 4,376 | oscyluje |

Przy $\rho=0{,}2$ i $\rho=0{,}5$ ta sama studnia zbiega jeszcze przy $\beta=24$, a promień zostaje przy $|1-\rho|$.

Wynik mapy to istnienie progu i lokalne kryterium jednego kroku. Nie jest dowodem globalnej zbieżności. Prognoza oscylacji dla $N=8$, $\beta=8$, $\rho=1$ nie zaszła: ten rozmiar zostaje ściągający. W reżimie dwóch równych studni jakobian przy remisie argmax jest niewiarygodny i nie wchodzi do mediany.

Wszystkie pomiary w rozdziale 8 liczyły jeden krok Most-1. Pętla $K$ nie była użyta w żadnej sieci.

---

## 8. Pomiary

### 8.1. Protokół

Wszystkie biegi: CPU, Python 3.14, PyTorch 2.13.

| Seria | Dane | Protokół | Ramiona | Biegi |
|-------|------|----------|---------|-------|
| ablacja jądra | układ okresowy, zamrożona specyfikacja | transduktywny | 4 ramiona, po 26 084 wag | 800 kroków × 20 |
| tracer w tokenie | ta sama specyfikacja | transduktywny | 8 ramion, po 26 308 wag | 800 kroków × 20 |
| międzydziedzinowa | układ okresowy, kana, muzyka | ścisły | ramiona ·0 i ·ΔE | 20 |
| n-ciał | katalog słoneczny (80), trójka hierarchiczna (84) | ścisły | 4 ramiona + MLP | 800 kroków × 20 |
| T1, nuty | 80 zdarzeń syntetycznych | ścisła indukcja | 7 ramion | 800 kroków × 20 (420 przebiegów) |
| T2, mały LM | 6 dokumentów ze wspólnym prefiksem | — | 5 ramion, $d=64$, 2 warstwy | 300 kroków × 20 seedów |
| średni LM | notatki z repozytorium, WikiText-2 | valid i test osobno | Most, softmax | 2 seedy |

W protokole transduktywnym cechy elementów testowych są widoczne w treningu. W ścisłym enkoder widzi tylko elementy treningowe.

**Ramiona.** `bw` to Most, `sm` to softmax. `·0`: bez tracera. `·ΔE`: tracer tylko w uwadze — w Moście przez mianownik, w softmaxie jako kara $-\Delta E^2/\gamma_0^2$ w wykładniku. `·tok`: tracer w embeddingu tokenu, $x=\mathrm{embed}(\text{treść})+W_\tau\tau$. `·tok+ΔE`: oba naraz. Ramiona jednej serii mają tę samą liczbę wag i identyczną inicjalizację przy tym samym ziarnie (sprawdzona równość `state_dict`).

**Warianty.** `comp`: sam operator kompozycji. `comp+rel·PoE`: osobna głowica relacji kolumny, łączona iloczynem ekspertów. `comp+mix`: relacja wmieszana w tę samą przestrzeń.

**Miary** (hit@1). `A_dół`: jeden krok „dół” na nowych elementach. `B_te_hop`: dwa kroki na nowych. `B_te_dd`: dół∘dół na nowych. `B_tr_op`: złożenie operatorów na znanych. `B_tr_dd`: dół∘dół na znanych.

**Statystyka.** Równoważność: cały 90% przedział ufności różnicy (bootstrap po 20 przebiegach, 10 000 losowań) mieści się w marginesie (TOST, $\alpha=0{,}05$ na stronę). Różnice: rodziny testów z korektą Holma, zadeklarowane przed uruchomieniem.

Margines ±5 punktów był zamrożony przed analizą. Margines straty ±0,05 był zamrożony przed pierwszym seedem serii T2.

Zasada porównania: oba jądra dostają ten sam opis. Zestawienie Mostu z tracerem w $\Delta E$ z softmaxem bez tego tracera miesza wzór z tym, co model dostaje, i nie jest werdyktem o jądrze.

### 8.2. Zgoda wzorów

Test równoważności przy progu ±5 punktów objął 150 porównań par ramion różniących się wyłącznie jądrem.

| Dane | Porównanie | równoważne | nierozstrzygnięte | Most lepszy | softmax lepszy |
|------|------------|-----------:|------------------:|------------:|---------------:|
| układ okresowy, transduktywny | bw·0 − sm·0 | 9 | 4 | 1 | 1 |
| układ okresowy, transduktywny | bw·ΔE − sm·ΔE | 2 | 3 | 8 | 2 |
| układ okresowy, transduktywny | bw·tok − sm·tok | 14 | 1 | 0 | 0 |
| układ okresowy, transduktywny | bw·tok+ΔE − sm·tok+ΔE | 12 | 3 | 0 | 0 |
| układ okresowy, ścisły | bw·0 − sm·0 | 11 | 3 | 1 | 0 |
| układ okresowy, ścisły | bw·ΔE − sm·ΔE | 9 | 2 | 0 | 4 |
| kana | bw·0 − sm·0 | 11 | 4 | 0 | 0 |
| kana | bw·ΔE − sm·ΔE | 11 | 4 | 0 | 0 |
| muzyka | bw·0 − sm·0 | 14 | 0 | 0 | 1 |
| muzyka | bw·ΔE − sm·ΔE | 7 | 1 | 0 | 7 |
| **razem** | | **100** | 25 | 10 | 15 |

Kolumny „lepszy” obejmują także przypadki, w których przedział leży po jednej stronie zera, ale sięga marginesu. Każdy wiersz to 15 porównań.

Trzy odczyty tej tabeli:

1. **Tracer w tokenie: wzory są zamienne.** 26 z 30 porównań w progu, 4 nierozstrzygnięte, w żadnym Most nie jest gorszy. Ten wynik pochodzi z protokołu transduktywnego na układzie okresowym.
2. **Bez tracera: wzory są zamienne.** 45 z 60 porównań w progu, we wszystkich dziedzinach. W ablacji żadna z 15 różnic nie przeżywa korekty Holma, $|\Delta|\le0{,}04$. Sama postać $\mathrm{softplus}(S)^2$ zamiast $\exp(S)$ niczego nie wnosi.
3. **Tracer tylko przez uwagę: wzory nie są zamienne, a kierunek zależy od dziedziny i protokołu.** W protokole transduktywnym Most jest lepszy na nowych pierwiastkach, do +0,11 w wariancie z głowicą relacji. W ścisłym protokole na tym samym układzie lepszy jest softmax, do −0,10 na znanych pierwiastkach. W muzyce softmax jest lepszy o 0,03–0,06. Na kanie różnic nie ma. Zysk Mostu z wariantu PoE znika, gdy softmax dostaje tracer w tokenie, i znika w ścisłym protokole (+0,003).

Punkt 3 porównuje Most z softmaxem obciążonym karą gaussowską. To jeden z wielu sposobów podania $\Delta E$ softmaxowi i, jak pokazuje 8.3, sposób słaby.

**Co te ramiona mówią o szerokości.** W ramionach `·0` i `·tok` rozstrojenie jest wyłączone, więc $\gamma_i$ skraca się przy normalizacji i waga Mostu to $C_{ij}/\sum_k C_{ik}$ (rozdział 4.5). Parametry $\gamma_0$, $\beta$ i liczba $\kappa$ nie mają tam wpływu na wynik. Odczyty 1 i 2 mierzą zatem zamianę $\exp(S)$ na $\mathrm{softplus}(S)^2$. Szerokość zależna od wiersza pracuje tylko w ramionach z włączonym $\Delta E$: `·ΔE`, `·tok+ΔE` (12 z 15 w progu, 3 nierozstrzygnięte), ramię C małego LM i średni LM.

**Ramiona małego LM (T2).** A: $\Delta E$ zgaszone, softmax bez dodatku. B: ten sam numer źródła wpisany w token obu stronom, $\Delta E$ Mostu zgaszone. C: Most kanoniczny z tracerem pozycji przez $\Delta E$ wobec softmaxu bez dodatku; jedyne ramię z włączonym $\Delta E$ i jawnie mieszane.

**Tracer w tokenie — zestawienie.**

| Pomiar | Wynik |
|--------|-------|
| pierwiastki, $\tau$ w tokenie, 30 porównań, próg ±5 | 26 w progu, 4 nierozstrzygnięte, Most w żadnym gorszy |
| te same 30 porównań, próg ±2 | 7 w progu, 0 całkiem poza, 23 nierozstrzygnięte |
| nuty, pisownia w tokenie, różnica trafień Most − softmax | −0,002, −0,002, 0,000 |
| mały LM, numer źródła w tokenie, różnica straty | −0,0016, przedział [−0,0040, +0,0006] |
| mały LM, Most z wyzerowanym $\Delta E$ wobec softmaxu | +0,0009, przedział [−0,0005, +0,0022] |

Dwadzieścia przebiegów nie udźwiga zgody ciaśniejszej niż ±5 punktów. Przy ±2 dwa porównania mają przedział cały po stronie Mostu, sięgający poza próg: złożenie na znanych w wariancie PoE (+0,028, przedział [+0,013, +0,043]) i dwa kroki na nowych przy tracerze w tokenie i w uwadze (+0,027, przedział [+0,009, +0,045]). Największa średnia różnica w całej trzydziestce to +0,028.

Wszystkie przebiegi serii nutowej (420) i serii małego LM (100) miały skończoną stratę, a ostatni krok niższy niż pierwszy.

### 8.3. Co dokłada punkty: tracer w tokenie, nie kara w wykładniku

Softmax może dostać tracer na dwa sposoby: wpisany w token albo jako karę $-\Delta E^2/\gamma^2$ w wykładniku.

| Zadanie | Tracer w tokenie wobec kary |
|---------|------------------------------|
| nowe pierwiastki | +6…+12 punktów |
| nuty, jeden krok | +0,31 (kara zrzuca krok pisowni z 0,49 do 0,12; token trzyma go przy 0,43) |
| nuty, hop | +0,12 |
| nuty, dwa kroki | +0,11 |

Wszystkie trzy różnice na nutach przeżywają korektę Holma.

Na układzie okresowym pełny obraz jest następujący:

- Sama informacja z tracera, podana przez uwagę, podnosi złożenie na znanych pierwiastkach w **obu** jądrach o 0,20–0,33 względem ramion bez tracera. Dawne porównania „Most wobec softmaxu”, w których softmax tracera nie dostawał, mierzyły więc informację, a nie jądro.
- Softmax z tracerem w tokenie wobec kanonicznego Mostu (tracer tylko przez $\Delta E$): softmax lepszy w 7 z 15 testów, Most w 2 (dół∘dół na znanych, w wariantach z relacją), remis w 6.
- Tracer w tokenie pomaga także Mostowi: +0,07…+0,10 względem Mostu kanonicznego. Kanoniczny Most nie wykorzystuje więc informacji z tracera w pełni.
- Najlepsze konfiguracje na nowych pierwiastkach to wariant `comp+mix` z tracerem w tokenie, w dowolnym jądrze: krok „dół” 0,45–0,46.

**Zastrzeżenie do nut.** Token bije karę, ale nie bije braku tracera na miarach nowych zdarzeń. Softmax bez tracera ma krok 0,488 wobec 0,432 z pisownią w tokenie, a na dwóch krokach w dół 0,352 wobec 0,155. Pisownia w tokenie wygrywa dopiero na złożeniu na znanych (0,718 wobec 0,604). Powód leży w zadaniu: bliźniak enharmoniczny ma ten sam wektor, więc remis w rankingu nie karze złej pisowni, a ramię bez tracera może iść po samej oktawie. Zdanie „tracer w tokenie dokłada punkty” dotyczy na nutach porównania z karą, nie z brakiem tracera.

Na małym LM softmax z numerem dokumentu w tokenie ma stratę 0,0421 wobec 0,0497 bez niego. Różnica −0,0075, przedział [−0,0109, −0,0041], cały pod zerem.

Most w wersji kanonicznej, w której tracer pozycji wchodzi tylko przez $\Delta E$, ma na małym LM stratę wyższą o 0,0045 od softmaxu bez dodatku. Przedział [+0,0028, +0,0063] leży nad zerem i zarazem wewnątrz ±0,05. Zgodnie z zasadą z 8.1 to porównanie nie rozstrzyga o samym wzorze.

### 8.4. Relacja kolumny

| Dziedzina | Co widać w sąsiedztwie | Relacja kolumny |
|-----------|-------------------------|-----------------|
| katalog słoneczny, 80 ciał | ta sama powłoka 95%, ta sama masa 4% | +5…+11 punktów na kroku masy, wszystkie 5 ramion |
| trójka hierarchiczna, 84 ciała | ta sama rola 100% | nic nie dokłada; wmieszana w jedną przestrzeń obniża krok o 0,26 |
| nuty, 80 zdarzeń | ta sama oktawa 100%, ta sama pisownia 0% | +0,09…+0,13 tylko tam, gdzie model pisownię już dostaje |

Osobna relacja dokłada punkty wtedy, gdy oś jest w stanie, a najbliższy sąsiad jej nie pokazuje. Masa w katalogu słonecznym jest w wektorze stanu, ale odległość ją zagłusza. Rola w trójce jest całym sąsiedztwem. Pisownia nuty nie jest w widmie: Cis i Des mają to samo widmo.

Najwyższe trafienia kroku: na katalogu słonecznym MLP z relacją 0,984, potem softmax bez kary 0,975 i Most bez tracera 0,972; Most z tracerem ma 0,963. Na nutach MLP z relacją 0,73.

Na katalogu słonecznym tracer podany przez uwagę nie pomaga, a softmaxowi szkodzi: kara obniża złożenie na znanych ciałach o 0,08 (przeżywa korektę). Tracer jest tam zbudowany z tych samych sześciu cech co treść. Most jest lepszy od softmaxu z karą o 0,03–0,04 w wariancie PoE, ale wobec softmaxu bez kary lepszy nie jest.

Oba katalogi są syntetyczne: 80 orbit kołowych wokół masy centralnej oraz 28 współliniowych trójek hierarchicznych.

### 8.5. Pełna tablica pierwiastków

118 pierwiastków, 328 krawędzi, sam Most, 3 seedy, 64 epoki:

| r@1 | r@3 | MRR | próg | czas |
|-----|-----|-----|------|------|
| 0,575 | 0,892 | 0,739 | 3/3 | 768 s |

Po seedach (0, 1, 17): r@1 = 0,581, 0,565, 0,578. Ten bieg mierzy kompetencję samego Mostu i nie jest ponownym porównaniem z softmaxem; parytet obu jąder na mniejszym zadaniu z tej samej dziedziny był zmierzony wcześniej (Most około 0,43, softmax około 0,45).

Wcześniejszy bieg na jądrze Lorentza miał r@1 = 0,567. Seal był w tym biegu włączony, a oś $T$ na tym zbiorze jest przybliżeniem okresu. Ablacje jądra z 8.2 liczono z wyłączonym sealem.

### 8.6. Model językowy

Średni model, około 3,3 mln wag, 2 seedy.

| Bieg | Most | Softmax |
|------|------|---------|
| notatki z repozytorium, 600 kroków, strata | 2,416 | 2,353 |
| WikiText-2 (10,9 MB), jeden przejazd, ostatni batch treningu | 1,554 | 1,587 |
| WikiText-2, cały valid | 1,557 | 1,604 |
| WikiText-2, cały test | 1,580 | 1,627 |
| WikiText-2, perplexity na teście (na bajt) | ≈ 4,9 | ≈ 5,1 |
| WikiText-2, trafienie następnego bajta, 400 okien valid | 54,6% | 53,8% |

Parametry: okno 128 bajtów, batch 8. Most ma 3 287 684 wagi, softmax 3 286 528; różnica 1 156 to ścieżka energii i szerokości. Notatki: 1,27 MB, 600 kroków, czyli mniej niż pół przejazdu. WikiText-2: 10 685 kroków, jeden przejazd.

Po seedach:

| Bieg | seed | Most | Softmax | softmax − Most |
|------|-----:|-----:|--------:|---------------:|
| notatki, ostatni batch | 0 | 2,092 | 2,108 | +0,016 |
| notatki, ostatni batch | 1 | 2,741 | 2,597 | −0,144 |
| WikiText-2, ostatni batch | 0 | 1,555 | 1,573 | +0,018 |
| WikiText-2, ostatni batch | 1 | 1,554 | 1,601 | +0,048 |
| WikiText-2, valid | 0 | 1,571 | 1,600 | +0,029 |
| WikiText-2, valid | 1 | 1,544 | 1,609 | +0,065 |
| WikiText-2, test | 0 | 1,593 | 1,623 | +0,031 |
| WikiText-2, test | 1 | 1,567 | 1,631 | +0,063 |

Wiersze „ostatni batch” to strata treningowa jednego batcha, nie pomiar na osobnym zbiorze. Valid i test to średnia po całym pliku, okna bez zachodzenia.

Na WikiText-2 oba seedy mają ten sam znak różnicy na valid i teście. Na notatkach seedy się nie zgadzają: średnią przewagę softmaxu daje jeden seed. Przy dwóch seedach i bez przedziału ufności ta seria pokazuje, że jądro uczy się na tekście i nie odstaje od softmaxu. Nie pokazuje przewagi żadnego z nich.

Dwa zastrzeżenia do odczytu tej tabeli:

1. Średni LM nie ma ramion A i B z serii T2. Jest tylko para „Most z $\Delta E$, softmax bez $\Delta E$”, czyli układ wejść ramienia C. Softmax nie jest ślepy na pozycję: pozycja siedzi w strumieniu tokenu obu sieci. Most ma jednak dodatkowy kanał $\Delta E$, którego softmax nie ma, i ten kanał nie jest czystym numerem pozycji, bo mieszanie tracera widzi też embedding bajta. Różnica na WikiText-2 (Most niżej o około 0,05 na valid i teście) ma więc kształt ramienia C i nie jest werdyktem o wzorze. Na T2 to samo ramię dało Mostowi stratę wyższą o 0,0045. Znak się odwrócił, układ wejść nie.
2. Strata w pierwszym kroku wynosi 167–170 w obu jądrach, przy $\ln 256\approx5{,}55$ dla rozkładu jednostajnego. Wskazuje to na bardzo dużą skalę logitów przy inicjalizacji. Dotyczy obu jąder jednakowo, ale wymaga sprawdzenia.

Tekst generowany przez oba jądra na tym modelu zapętla się w ten sam sposób.

### 8.7. Czas

| Zadanie | Most / softmax |
|---------|----------------|
| nuty, trzy ramiona każdego jądra | 1,12× (717 s wobec 642 s) |
| mały LM, na ramię | około 1,44× (83–86 s wobec 58–59 s) |
| średni LM, notatki | 1,32× (seedy: 1,34 i 1,30) |
| średni LM, WikiText-2, pierwszy bieg | 1,14× (seedy: 1,07 i 1,22) |
| średni LM, WikiText-2, powtórzenie tego samego treningu | 1,44× (seedy: 1,35 i 1,53) |

Stosunek dla małego LM jest liczony na ramię. Notatka źródłowa podaje 2,15× jako 253 s wobec 117 s, ale 253 s to suma trzech ramion Mostu, a 117 s suma dwóch ramion softmaxu.

Dwa biegi WikiText-2 liczyły ten sam trening i dały identyczne straty, a stosunek czasu wyszedł 1,14× i 1,44×. Szum pomiaru czasu na tej maszynie jest więc tego samego rzędu co mierzony efekt. Bezpieczny odczyt: Most jest wolniejszy o około 10–50%.

Oba jądra woła ten sam PyTorch. Softmax jest jednym wywołaniem biblioteki. Most jest łańcuchem kilkunastu osobnych operacji (softplus, kwadrat, maska, suma, maksimum, dzielenie) plus ścieżką tracera. Na nutach prawie cały czas zjada pętla Pythona, więc wzór mało waży. Na modelu językowym uwaga jest większą częścią kroku.

Wolniejszy krok nie jest tu kryterium porażki. Kryterium jest strata skończona i malejąca oraz zgodność wag z kontraktem.

---

## 9. Interpretacja

### 9.1. Dwa znaczenia słowa „Lorentz”

Wzór $\gamma/(\gamma^2+\Delta E^2)$ to profil Lorentza (rozkład Cauchy'ego): kształt linii rezonansowej tłumionego oscylatora. Breit–Wigner jest jego odpowiednikiem z fizyki rezonansów. Stąd nazwy obu jąder.

Nazwa „Most” jest analogią do mostu Einsteina–Rosena i przejezdnych tuneli Morrisa–Thorne'a. W tamtej literaturze „lorentzowski” odnosi się do sygnatury metryki, a nie do kształtu linii. Wzór pochodzi z pierwszego znaczenia, obraz mostu z drugiego. Jądro uwagi nie ma czasoprzestrzeni, dynamiki metryki ani warunków energetycznych. Porównanie dotyczy struktury ról, nie fizyki.

### 9.2. Podział ról

| Most ER / Morris–Thorne | Most uwagi |
|---|---|
| geometria łącząca dwa końce | $\Delta E_{ij}$ z tracera |
| gardziel | mianownik $\gamma^2+\Delta E_{ij}^2$ |
| materia trzymająca gardziel | Lorentz: $\gamma\ge10^{-3}$; Most-1: $\gamma_i\ge\gamma_0(1+\beta/n_i)$ |
| zapadnięcie | $\gamma$ przy podłodze: most zwęża się do sąsiadów energetycznych |
| brak mostu | ZERO: $\Delta E=0$, zostaje samo sprzężenie treści |

Tracer formuje most, bo daje geometrię rozróżniającą końce. Szerokość trzyma go otwartym. Most-1 dokłada do tego reakcję na obciążenie: gdy jeden klucz bierze sprzężenie ($\kappa_i\to1$), $\gamma_i$ rośnie i most się poszerza. W Lorentzu stabilizacją jest sama podłoga.

### 9.3. Degeneracja zamiast zapadnięcia

Zagrożeniem nie jest zapadnięcie numeryczne, tylko degeneracja w trakcie uczenia:

1. **$\gamma$ bardzo duże.** $\gamma^2\gg\Delta E^2$, most przestaje czuć energię i zachowuje się jak ZERO.
2. **$\gamma$ przy podłodze.** Otwierają się tylko pary o niemal identycznej energii.

Most pracuje, gdy $\gamma$ ustala się w pasie, w którym $\Delta E$ jest porównywalne z $\gamma$, a model z prawdziwym tracerem wyraźnie wygrywa z ZERO i z tracerem potasowanym między pozycjami.

### 9.4. Proponowane kryteria pomiarowe

| Pomiar | Co pokazuje |
|---|---|
| przebieg $\gamma_0$ i $\beta$ w trakcie uczenia | czy szerokość nie ucieka do skrajności |
| rozkład $\Delta E_{ij}/\gamma_i$ na wierszach | czy most pracuje w paśmie, gdzie energia ma znaczenie |
| histogram $\kappa_i$ | czy stabilizacja dynamiczna faktycznie działa |
| entropia wag na wiersz | czy most nie zwęża się do jednego klucza ani nie rozlewa |
| wynik wobec ZERO | wkład geometrii ponad samo sprzężenie treści |
| wynik wobec tracera potasowanego | czy liczy się treść tracera, a nie sama obecność $\Delta E$ |
| liczba `NaN` i pustych wierszy | obsługa brzegów |

Te pomiary nie wchodzą w skład serii z rozdziału 8 i są wskazane jako następny krok.

---

## 10. Ograniczenia

- **Próg zgody.** Zgoda obowiązuje przy ±5 punktach. Przy ±2 seria jej nie rozstrzyga.
- **Skala języka.** Zmierzony jest model około 3,3 mln wag, jeden zbiór publiczny, dwa seedy. Model 100–300 mln wag nie był liczony.
- **Czas.** Zmierzony koszt dotyczy niezłożonego łańcucha operacji. Kosztu po złożeniu w jedno wywołanie nie znam. Pomiar czasu ma duży szum: ten sam trening dał stosunek 1,14× i 1,44×.
- **Protokół zgody.** Wynik 26/30 dla tracera w tokenie pochodzi z protokołu transduktywnego na jednej dziedzinie. W ścisłym protokole ramiona z tracerem w tokenie były liczone tylko na nutach i na małym LM.
- **Kanon Mostu.** W postaci kanonicznej, z tracerem tylko przez $\Delta E$, Most nie jest równoważny softmaxowi: wynik zależy od dziedziny i protokołu, a softmax z tracerem w tokenie jest częściej lepszy.
- **Strojenie.** Hiperparametry kanonu (krok uczenia, wymiary, temperatura) były historycznie strojone pod Most, co może mu sprzyjać.
- **Dane.** Zadania siatkowe są małe, a katalogi n-ciał i nut syntetyczne. Margines ±5 punktów jest arbitralny, choć ustalony przed analizą.
- **Porównanie językowe.** Średni LM był liczony tylko w układzie ramienia C: Most z kanałem $\Delta E$ wobec softmaxu bez niego (rozdział 8.6). Ramion o wyrównanej informacji na tej skali nie ma.
- **Szerokość wiersza w testach zgody.** W ramionach z wyłączonym $\Delta E$ szerokość skraca się przy normalizacji (rozdział 4.5). Zgoda zmierzona w tych ramionach dotyczy sprzężenia $\mathrm{softplus}^2$ wobec $\exp$, a nie szerokości zależnej od wiersza, która jest nowością Most-1 (rozdział 8.2).
- **Sprzęt.** Wszystko liczono na CPU. Karta graficzna i NPU nie były użyte: tensory są małe, a NPU liczy tylko gotowy, skwantyzowany model.
- **Bias pozycyjny.** W modelu językowym szerokość zależy od liczby otwartych kluczy (rozdział 4.4). Jego wpływ na wynik nie był mierzony osobno.
- **Asymetria sealu.** Seal działa tylko w Moście. Porównania jąder z rozdziału 8.2 liczono z wyłączonym sealem. Pełna tablica z 8.5 miała seal włączony i nie jest porównaniem z softmaxem.
- **Znak rozstrojenia.** Fork go odrzuca. Wynik nie mówi nic o jądrach kierunkowych.
- **Starsze biegi.** Biegi na jądrze Lorentza, w tym językowe z 21 września 2026, miały inny korpus i inne jądro. Nie są kontrolą dla Most-1.

---

## 11. Znane usterki implementacji i prace otwarte

### 11.1. W konstrukcji

1. **Seal nie wraca z checkpointu.** Zapis enkodera nie niesie flagi sealu. Po wczytaniu model wraca do kanonu, czyli sealu włączonego. Model uczony z wyłączonym sealem liczy po wczytaniu inaczej, dopóki wołający nie ustawi flagi ponownie. Do zrobienia: zapis flagi w opisie architektury i test zapis–odczyt, przy czym brak pola w starym pliku ma zostawiać kanon.
2. **Metryka `mean_dE` miesza pozycje.** Średnia z $|\Delta E|$ jest liczona po całym tensorze `[B, h, N, N]`, łącznie z przekątną, CLS i paddingiem. Nie jest średnim rozstrojeniem żywych par. Do zrobienia: średnia po kluczach, które $\mathrm{keep}$ zostawia dodatnie, albo inna nazwa metryki.
3. **Zwracane wagi klasyfikatora trzymają graf.** Odłączone jest tylko $\Delta E$. Do zrobienia: odłączyć wagi, po sprawdzeniu, że żaden wołający nie liczy przez nie straty.
4. **Bias CLS.** Energia CLS to sam bias projekcji energii. Do rozstrzygnięcia: zostawić i opisać w metryce albo liczyć $\Delta E$ bez biasu, z testem.
5. **Martwe gałki.** Parametry `delta_alpha` i `directed_bias` zostały w konstruktorze po jądrze Lorentza. Forward Most-1 ich nie czyta. Dwa archiwalne skrypty ablacji znaku padają dziś na starcie.
6. **Ręczna normalizacja w bloku językowym.** Jest równa `breit_wigner_weights` i zostaje, dopóki test równości jest zielony.
7. **Skok sealu przy $T=0{,}20$.** Do potwierdzenia, czy nieciągłość jest zamierzona.

### 11.2. W pomiarach

8. **Hold-out Na/K na pełnej tablicy**, tym samym protokołem, bez ruszania wzoru.
9. **Klasyczny loader softmaxu** spoza bliźniaka, jako niezależna kontrola.
10. **Złożone jądro.** Jedno wywołanie o tych samych wagach, z testem różnicy poniżej $10^{-6}$ na kształcie `[2, 2, 8, 8]`. Bez tego testu nie wolno ogłaszać zmiany czasu.
11. **Duży model językowy.**
12. **Kryteria z rozdziału 9.4**, w tym osobny pomiar biasu pozycyjnego.
13. **Tracer w tokenie w ścisłym protokole** na układzie okresowym, kanie i muzyce, żeby zgoda 26/30 nie wisiała na jednym protokole.
14. **Średni LM z ramionami A i B**, jak w T2: A gasi $\Delta E$ obu stronom, B podaje tę samą dodatkową informację w tokenie obu stronom. Do tego więcej seedów i przedział ufności. Dopiero te ramiona rozdzielą wzór od kanału $\Delta E$ na skali WikiText-2.
15. **Strata początkowa 167–170 w średnim LM.** Sprawdzić skalę logitów przy inicjalizacji.
16. **Pomiar czasu** z powtórzeniami i na nieobciążonej maszynie.

---

## 12. Odtwarzalność

### 12.1. Środowisko

CPU, Python 3.14, PyTorch 2.13. Maszyna laboratoryjna ma zintegrowaną kartę Radeon 780M i NPU XDNA; żadne z nich nie brało udziału w pomiarach.

### 12.2. Testy kontraktu

Z katalogu forka:

```text
python -m pytest doswiadczenie/testy/test_breit_kernel.py doswiadczenie/testy/test_most_seal.py -q
```

Stan na 2026-10-07: 12 testów zielonych. Testy trzeba wołać przez `pytest`. Uruchomienie pliku testów bezpośrednio przez `python` zatrzymuje się na bloku `__main__` i nie woła testów zdefiniowanych niżej.

| Test | Kontrakt |
|------|----------|
| `test_row_matches_the_closed_form_and_ignores_the_sign` | wzór w `float64` i brak znaku $\Delta E$ |
| `test_a_larger_beta_lowers_the_peak_share` | większe $\beta$ spłaszcza szczyt |
| `test_blocked_key_stays_zero_and_does_not_set_kappa` | maska przed $\kappa$ |
| `test_sign_modes_are_rejected_and_a_forward_stays_finite` | tylko tryb `square`; forward skończony |
| `test_softmax_twin_survives_a_huge_self_score` | przekątna wycinana przed softmaxem |
| `test_weights_equal_normalised_rows` | ręczna normalizacja równa `breit_wigner_weights` |
| `test_bridge_matches_per_layer_masking` | klasyfikator maskuje tak samo w każdej warstwie |

Testy kontraktu nie powtarzają pomiarów z rozdziału 8. Zmiana, która przechodzi testy i rusza wzór, wymaga powtórzenia biegów, zanim wolno powiedzieć, że zgodność ze softmaxem została.

Biblioteka TSRA ma własny zestaw (38 testów), uruchamiany osobno.

### 12.3. Raport z biegu

Jedna zmiana jądra, jeden raport. W raporcie: seed, próg zamrożony przed seedem, strata albo r@1, stosunek czasu.

### 12.4. Trzy magazyny

Wagi (checkpointy), zrzuty przestrzeni Φ (to, na co model patrzy) i pamięć sesji są trzymane osobno. Jądro uczy się w wagach, jak patrzeć. Pamięć sesji nie wchodzi do forwardu.

---

## 13. Zasady porównań

Reguły, których trzymała się ta seria i których powinny trzymać się następne.

1. Oba jądra dostają ten sam opis. Różnica między Mostem z tracerem w $\Delta E$ a softmaxem bez tracera nie jest właściwością jądra.
2. Próg zgody zamraża się przed pierwszym seedem.
3. Maska, seal i przekątna wchodzą do $C$ przed $\kappa$.
4. $\kappa$ nie odłącza się od grafu. Gradient przez argmax jest częścią Most-1.
5. Wzoru nie zmienia się po to, żeby dogonić czas softmaxu.
6. Biegi na innym jądrze albo innym korpusie nie są kontrolą.
7. Wolniejszy krok nie jest porażką. Porażką jest `NaN`, rosnąca strata albo waga niezgodna z kontraktem.
8. Seal wyłącza się per źródło, gdy oś 3 nie jest temperaturą archiwum, a nie globalnie.

---

## Dodatek A. Notatki źródłowe pomiarów

Liczby z rozdziału 8 pochodzą z notatek w katalogu `doswiadczenie/glify/` forka.

| Notatka | Zawartość |
|---------|-----------|
| `MENDELEJEW_KERNEL_EQUIV.md` | ±5 punktów, tracer w tokenach, 26/30 |
| `MENDELEJEW_KERNEL_TOK.md` | token wobec kary $\Delta E$ na pierwiastkach |
| `MENDELEJEW_KERNEL_ABLATION.md` | cztery ramiona, czysta ablacja jądra, efekt informacji $\Delta E$ |
| `CROSSDOMAIN_COMP.md` | ścisły protokół na układzie okresowym, kanie i muzyce |
| `NBODY_COMP.md` | katalog słoneczny i trójka hierarchiczna |
| `JADRO_T0.md` | próg ±2 na tych samych 30 porównaniach |
| `JADRO_T1.md` | pisownia nuty jako dodatkowa zmienna |
| `JADRO_T2.md` | mały LM, numer źródła w tokenie |
| `JADRO_OGOLNE.md` | program testów i bramka nazwy |
| `MENDELEJEW_FULL.md` | pełna tablica na jądrze Breita–Wignera |
| `LM_MEDIUM_1C.md` | średni model na notatkach z repozytorium |
| `LM_MEDIUM_WIKITEXT.md` | średni model, trening WikiText-2 |
| `LM_MEDIUM_WIKITEXT_HOLDOUT.md` | valid i test WikiText-2 |

Mapa faz Most-K z rozdziału 7.1 pochodzi z biblioteki TSRA: specyfikacja `Plan.txt`, wyniki w `docs/NOTATKA_ROZWOJOWA.md` i katalogu `artifacts/`.

Tor nauki enkodera w forku to `phimost_learn.py`. Domyślnie uczy sam Most; flaga `--baselines` dolicza softmax i ablację ZERO.

## Dodatek B. Wzór w jednym miejscu

```text
S      = Q K^T / sqrt(d_h)
C      = softplus(S / temp)^2 * keep          # maski PRZED kappa
E      = Linear(tracer)
dE     = e_scale * (E_i - E_j)

mass   = clamp_min(sum_j C, 1e-12)
kappa  = max_j(C) / mass
gamma0 = softplus(log_gamma0) + 1e-3
beta   = softplus(beta_raw)
gamma  = gamma0 * (1 + beta * kappa)

A      = gamma * C / (gamma^2 + dE^2 + 1e-12)
w      = A / clamp_min(sum_j A, 1e-12)
```
