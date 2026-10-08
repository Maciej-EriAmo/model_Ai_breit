# Charakterystyka mostu

**Lorentz:** `C:\Users\drwis\model Ai`  
**Breit–Wigner (Most-1):** `C:\Users\drwis\model_Ai_breit`  
**Kanon rozmiaru:** 2 warstwy × 2 głowy = 4 jądra, `d_model=32` (\(d_h=16\)), tracer `n_dim=6`

Lorentz stawia most między daną a pamięcią. Breit–Wigner jest rezonansem na tym samym moście: szerokość przejścia oddycha z tym, jak mocno jedna pamięć bierze sprzężenie.

## Oznaczenia

| Symbol | Znaczenie |
|--------|-----------|
| \(\tau_i\) | tracer atomu \(i\) (wektor 6 osi) |
| \(T\) | czwarta oś tracera (temperatura stanu, używana przez seal) |
| \(\vartheta\) | temperatura sprzężenia (w kodzie parametr „temperatura”; **nie** tracer) |
| \(e_{\mathrm{scale}}\) | skala energii |
| \(n_i\) | liczba otwartych kluczy w wierszu \(i\) (po masce, sealu, paddingu i przekątnej) |

## Dana i pamięć

Dana to atom-zapytanie: treść `content` (`x`) i obserwator `tracer` (\(\tau\)).  
Pamięć czytana w uwadze to pozostałe atomy przestrzeni Φ (M2).  
Wagi, które uczą się *jak* patrzeć, siedzą w M1: projekcje `QKV`, projekcja energii `e`, szerokości.

Treść buduje zapytanie, klucz, wartość i sprzężenie. Tracer buduje energię.

## Wspólny szkielet

Dla głowy o wymiarze \(d_h\):

\[
S_{ij}=\frac{Q_i\cdot K_j}{\sqrt{d_h}},\qquad
C_{ij}=\mathrm{softplus}\!\left(\frac{S_{ij}}{\vartheta}\right)^2
\]

\[
E_i=W_e\tau_i+b_e,\qquad
\Delta E_{ij}=e_{\mathrm{scale}}\,(E_i-E_j)
\]

W `BridgeTransformer` jest \(\vartheta=1\) i \(e_{\mathrm{scale}}=1\). W enkoderze recallu oba są gałkami i domyślnie też są równe 1. Sprzężenie kanonu to `softplus`, nie `relu`.

\(\mathrm{softplus}(\cdot)^2>0\) zawsze, więc wiersz \(C\) może być zerowy tylko przez maski (seal, padding, przekątna, maska przyczynowa).

Bias \(b_e\) skraca się w różnicy \(E_i-E_j\). Dla \(\Delta E\) jest martwy: gradient do \(b_e\) przez \(\Delta E\) wynosi zero. Ma znaczenie tylko tam, gdzie \(E\) jest używane bez różnicy.

Po wadze wiersz jest normalizowany:

\[
w_{ij}=\frac{A_{ij}}{\sum_k A_{ik}+\varepsilon},\qquad \varepsilon=10^{-12}.
\]

Sprawdzone w kodzie (`breit_kernel.py`, `bridge.py`): \(10^{-12}\) jest dodawany w **trzech** miejscach — w mianowniku samego jądra \(A\) (`gamma.square()+delta_e.square()+1e-12`, zbędne, bo \(\gamma\ge10^{-3}\) już daje \(\gamma^2\ge10^{-6}\)), w mianowniku \(\kappa\) (jako `clamp_min(1e-12)` na sumie \(C\), patrz sekcja Breit–Wigner) i w normalizacji wiersza (powyżej, też `clamp_min`, nie dosłowne dodawanie). Tylko drugie dwa są potrzebne.

## Lorentz — most

Katalog `model Ai`. Jedna szerokość na głowę, uczona jako \(\log\gamma\):

\[
\gamma=\mathrm{softplus}(\log\gamma)+10^{-3}
\]

\[
A_{ij}=\frac{\gamma\,C_{ij}}{\gamma^2+\Delta E_{ij}^2}
\]

Most między daną \(i\) a pamięcią \(j\) otwiera się, gdy \(\Delta E_{ij}\) jest małe, i gaśnie, gdy rozstrojenie rośnie. Przy starcie \(\log\gamma=0\), więc \(\gamma=\ln 2+10^{-3}\approx 0{,}694\).

\(\gamma\) w liczniku jest stałe w wierszu i skraca się w normalizacji. Efektywnie

\[
w_{ij}\propto\frac{C_{ij}}{\gamma^2+\Delta E_{ij}^2}.
\]

Licznik ma znaczenie tylko tam, gdzie używa się nieznormalizowanego \(A\).

Dwa dalsze tryby są w kodzie, domyślnie wyłączone. Ustawiony zamek to `square`.

`onesided` — szerokość zależy od znaku energii:

\[
\gamma_{ij}=\gamma\,\bigl(1+\mathrm{softplus}(\alpha\,\Delta E_{ij})\bigr).
\]

`directed` — gotowa waga dostaje bramkę, z podłogą \(b=0{,}25\):

\[
A_{ij}=A^{\mathrm{square}}_{ij}\,\bigl(b+(1-b)\,\sigma(\alpha\,\Delta E_{ij})\bigr).
\]

Sprawdzone w kodzie (`lm_tiny.py`, `recall_model.py` w `model Ai`): \(\alpha\) (`delta_alpha`) to zwykły hiperparametr konstruktora, nieuczony, wspólny dla wszystkich głów i warstw, domyślnie \(\alpha=1\). Podłoga `directed_bias` \(b=0{,}25\) — też stała, nie parametr.

W forku Breit–Wigner oba tryby są odrzucone.

## Breit–Wigner — rezonans na moście

Katalog `model_Ai_breit`. Przejście zostaje lorentzowskie, a szerokość jest własnością wiersza:

\[
\kappa_i=\max_j\frac{C_{ij}}{\sum_k C_{ik}+\varepsilon}
\]

\[
\gamma_0=\mathrm{softplus}(\log\gamma_0)+10^{-3},\qquad
\beta=\mathrm{softplus}(\beta_{\mathrm{raw}})
\]

\[
\gamma_i=\gamma_0\,(1+\beta\,\kappa_i)
\]

\[
A_{ij}=\frac{\gamma_i\,C_{ij}}{\gamma_i^2+\Delta E_{ij}^2}
\]

\(\kappa\) bliskie 1 znaczy, że jedna pamięć wzięła sprzężenie: most puszcza szersze \(\Delta E\). \(\kappa\) małe znaczy sprzężenie rozlane po wielu atomach: szczelina się zwęża.

Zakres \(\kappa\) to \([1/n_i,\,1]\), nie \([0,1]\). Najwęższa szerokość wiersza wynosi więc

\[
\gamma_i^{\min}=\gamma_0\,(1+\beta/n_i),
\]

czyli \(\gamma_0\) osiąga się tylko w granicy długich wierszy. Skutki:

- \(\gamma_i\) zależy od długości sekwencji, nie tylko od kształtu sprzężenia;
- w Tiny LM wiersz \(i\) ma \(i+1\) kluczy, więc pozycja 0 ma zawsze \(\kappa=1\) i maksymalne \(\gamma\). Wczesne pozycje mają z konstrukcji szerszy most. To bias pozycyjny wbudowany w mechanizm.

Tak jak w Lorentzcie, \(\gamma_i\) w liczniku jest stałe w wierszu i skraca się w normalizacji. Efektywnie \(w_{ij}\propto C_{ij}/(\gamma_i^2+\Delta E_{ij}^2)\).

Start: \(\log\gamma_0=0\) i \(\beta_{\mathrm{raw}}=0\), więc \(\gamma_0\approx 0{,}694\) i \(\beta=\ln2\approx 0{,}693\). Szerokość od pierwszego kroku zależy od szczytu wiersza.

\(\kappa\) bierze się ze sprzężenia \(C\). \(\gamma_0\) i \(\beta\) to wagi M1. Tracer nie ustawia żadnego z nich.

\(\kappa\) zależy od \(Q\) i \(K\), więc \(\gamma_i\) otwiera dodatkową ścieżkę gradientu do projekcji `QK`. \(\max_j\) przepuszcza gradient tylko przez pozycję argmax.

Sprawdzone w kodzie (`breit_kernel.py`): \(\kappa\) **nie** jest odłączane. `state0 = coupling/mass` i `kappa = state0.amax(...)` nie mają `detach`, więc gradient faktycznie wraca przez \(\max_j\) do \(C\) i dalej do projekcji `QK`.

### Pusty wiersz

Jeśli wszystkie \(C_{ik}=0\), bez zabezpieczenia byłoby \(\kappa_i=0/0\) i NaN przechodziłby przez \(\gamma_i\) do całego wiersza \(A\). Przykład z enkodera: wiersz CLS ma wyzerowaną przekątną na \(C\), a wszystkie pozostałe klucze są zasealowane. Sprawdzone w kodzie (`breit_kernel.py`): mianownik \(\kappa\) jest już zabezpieczony — `mass = coupling.sum(-1).clamp_min(1e-12)` — więc pusty wiersz daje \(\kappa_i=0\) i \(\gamma_i=\gamma_0\), nie NaN. W Lorentzcie ten przypadek jest też bezpieczny: wiersz zer daje \(w=0\).

### Maska przyczynowa (Tiny LM)

Maska przyczynowa musi wejść na \(C\) **przed** \(\kappa\), tak jak seal, padding i przekątna. Jeśli kod nakłada ją dopiero na \(A\), to \(\max_j\) w \(\kappa_i\) liczy się także po przyszłych tokenach i informacja z przyszłości przecieka do \(\gamma_i\).

Sprawdzone w kodzie (`lm_tiny.py` w tym forku): kolejność jest poprawna. `causal_mask` zeruje \(C\) (`C.masked_fill(blocked, 0.0)`) **przed** wywołaniem `breit_wigner_rows`, więc \(\kappa_i\) nie widzi przyszłych tokenów — przecieku nie ma.

## Rola tracera w obu

Tracer jest obserwatorem. Sześć osi kanonu: jedynka (one-hot) głębokości (3 osie), \(T\), stan, mazur. Wszystkie wchodzą do energii \(E\) przez liniową projekcję \(W_e\). Przy starcie wagi tej projekcji są jedynkami, więc energia głowy jest sumą osi \(\tau\).

Konsekwencja startu: suma jedynki głębokości wynosi zawsze 1, więc przy \(W_e\) z jedynek skraca się w \(\Delta E\). Na starcie \(\Delta E\) widzi tylko \(T\), stan i mazur. Kolumny głębokości nie są martwe, bo gradient do nich jest niezerowy, ale startują od zerowego wpływu.

Sprawdzone w kodzie (`bridge.py`): `nn.init.ones_(layer["e"].weight)` wypełnia jedynkami całą macierz wag `e` (kształt `[h, dim_tracer]`), więc na starcie \(W_e\) jest identyczne dla wszystkich głów. Głowy startują z identyczną energią; symetrię łamie dopiero `QKV`.

\[
\Delta E_{ij}=e_{\mathrm{scale}}\,(E_i-E_j)
\]

wchodzi tylko do mianownika. To warunek mostu: bez niezerowego \(\Delta E\) geometria znika.

Osobną robotę ma \(T\), czwarta współrzędna. Seal, w obu katalogach domyślnie włączony:

| Próg | Skutek |
|------|--------|
| \(T<0{,}20\) | klucz wyzerowany (TOMB / archiwum) |
| \(0{,}20\le T<0{,}45\) | kara \(\exp\bigl(-12\,(0{,}45-T)\bigr)\) |
| CLS, pozycja 0 | zawsze otwarty |

Przy \(T=0{,}45\) kara wynosi 1, więc przejście jest ciągłe. Przy \(T=0{,}20\) kara wynosi \(e^{-3}\approx0{,}050\), a poniżej skacze do 0. To nieciągłość progu TOMB.

> Do potwierdzenia: czy skok przy \(T=0{,}20\) jest zamierzony.

W Lorentzcie seal mnoży gotową wagę \(A\). W forku Breit–Wigner seal dostaje sprzężenie \(C\) jeszcze przed jądrem, razem z zerem na przekątnej i paddingiem. Zasealowane klucze nie wnoszą więc nic do \(\kappa\). Ciągła kara COLD też wchodzi do \(\kappa\): tłumi część kluczy, podnosi udział szczytu i przesuwa \(\gamma\) całego wiersza.

## Co jest ustawione

| | Lorentz | Breit–Wigner |
|--|---------|----------------|
| Jądra | 4 (2×2) | 4 (2×2) |
| Człony wagi | \(C\), \(\Delta E\), jedno \(\gamma\) | \(C\), \(\Delta E\), \(\kappa\), \(\gamma_0\), \(\beta\) |
| Znak \(\Delta E\) | dostępny (`directed`, `onesided`), domyślnie `square` | odrzucony |
| Przekątna w enkoderze i klasyfikatorze | zero na \(A\), po jądrze | zero na \(C\), przed \(\kappa\) |
| Tiny LM | przekątna zostaje, maska przyczynowa | przekątna zostaje, maska przyczynowa (na \(C\) przed \(\kappa\), sprawdzone w kodzie) |
| Seal \(T\) | na \(A\) | na \(C\) |
| Pusty wiersz | bezpieczny (\(w=0\)) | bezpieczny — `clamp_min(1e-12)` w \(\kappa\), sprawdzone w kodzie |
| Zależność \(\gamma\) od długości wiersza | brak | \(\gamma_i^{\min}=\gamma_0(1+\beta/n_i)\) |

## ZERO

Wyzerowany tracer daje \(\Delta E=0\). Tak samo działa każdy tracer stały dla wszystkich atomów. Szerokość jest wtedy wspólna dla całego wiersza i skraca się przy normalizacji. W obu zostaje

\[
w_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
\]

Most się nie formuje. Zostaje samo sprzężenie treści.

**Seal: nie jest to już problem.** Sprawdzone w kodzie (`most_seal.py`, `apply_t_key_seal`): funkcja ma jawny guard na ablację. Wiersz liczy się jako „ablowany”, gdy *cały* tracer atomu (nie tylko oś \(T\)) jest zerowy (`real_atom = tracer.abs().sum(-1) > 0`) — a prawdziwy atom, nawet TOMB, ma niezerowy one-hot głębokości, więc odróżnia się od sztucznego zera. Dla wierszy w pełni ablowanych kara COLD i hard-mask TOMB są obie pomijane (`torch.where(ablated, ones, ...)`, `cold_hard & ~ablated`) i funkcja zwraca \(A\) bez zmian. Czyli: przy ablacji ZERO (cały tracer, włącznie z \(T\), wyzerowany) seal **nie** włącza się, mimo że jest domyślnie `enabled=True`. Wynik \(w_{ij}=C_{ij}/\sum_k C_{ik}\) z początku tej sekcji zachodzi więc także z domyślnie włączonym sealem — wcześniejsza wersja tego dokumentu była tu błędna.

## Analogia i stabilność

### Dwa znaczenia „Lorentz”

Wzór jądra

\[
\frac{\gamma}{\gamma^2+\Delta E^2}
\]

to profil Lorentza (rozkład Cauchy'ego): kształt linii rezonansowej tłumionego oscylatora w modelu Lorentza. Breit–Wigner jest jego odpowiednikiem z fizyki rezonansów, stąd nazwy obu katalogów.

Most jest analogią *per analogiam* do tunelu czasoprzestrzennego: mostu Einsteina–Rosena i przejezdnych tuneli Morrisa–Thorne'a. W tamtej literaturze „lorentzowski” odnosi się do sygnatury metryki, nie do kształtu linii. Wzór pochodzi z pierwszego znaczenia, obraz mostu z drugiego. Jądro uwagi nie ma czasoprzestrzeni, dynamiki metryki ani warunków energetycznych. Porównanie dotyczy struktury ról, nie fizyki.

### Podział ról: tracer formuje, szerokość trzyma

Most Einsteina–Rosena zapada się, zanim cokolwiek zdąży przejść. Utrzymanie gardzieli wymaga egzotycznej materii. W tym moście role są rozdzielone:

- **Tracer (obserwator) formuje most.** Daje \(\Delta E_{ij}\), czyli geometrię rozróżniającą końce. Bez niezerowego \(\Delta E\) most się nie formuje (sekcja ZERO).
- **Szerokość \(\gamma\) trzyma most otwarty.** Podłoga szerokości nie pozwala gardzieli się zamknąć.

| Most ER / Morris–Thorne | Most uwagi |
|---|---|
| geometria łącząca dwa końce | \(\Delta E_{ij}\) z tracera — obserwator nadaje kształt |
| gardziel | mianownik \(\gamma^2+\Delta E_{ij}^2\) |
| egzotyczna materia trzymająca gardziel | Lorentz: \(\gamma\ge10^{-3}\); Breit–Wigner: \(\gamma_i\ge\gamma_0(1+\beta/n_i)\) |
| zapadnięcie | \(\gamma\to\) podłoga: most zawęża się do sąsiadów energetycznych |
| brak mostu | ZERO: \(\Delta E=0\), zostaje samo sprzężenie treści |

Breit–Wigner dokłada stabilizację dynamiczną: gardziel reaguje na obciążenie. Gdy jedna pamięć bierze sprzężenie (\(\kappa_i\to1\)), \(\gamma_i\) rośnie i most się poszerza. W Lorentzcie stabilizacją jest sama podłoga.

### Gwarancje z konstrukcji

- Szerokość nie zamyka się do zera: \(\gamma\ge10^{-3}\) (Lorentz), \(\gamma_i\ge\gamma_0\) (Breit–Wigner; równość tylko dla pustego wiersza, gdzie \(\kappa_i=0\)).
- Mianownik \(\gamma^2+\Delta E^2\) jest zawsze dodatni, więc \(A\) jest skończone przy każdym \(\Delta E\).
- \(C>0\) poza maskami, a po normalizacji \(w_{ij}\in[0,1]\).
- Pusty wiersz w Breit–Wignerze jest zabezpieczony: `clamp_min` w mianowniku \(\kappa\) daje \(\kappa_i=0\), nie NaN (sekcja Breit–Wigner).

### Degeneracja zamiast zapadnięcia

Realnym zagrożeniem nie jest zapadnięcie numeryczne, tylko degeneracja w trakcie uczenia. Możliwe są dwie skrajności:

1. **\(\gamma\) bardzo duże.** \(\gamma^2\gg\Delta E^2\), most staje się nieczuły na energię i zachowuje się jak ZERO.
2. **\(\gamma\) przy podłodze.** Otwierają się tylko pary o niemal identycznej energii, most zwęża się do sąsiedztwa energetycznego.

Most jest stabilny w sensie mierzalnym, gdy \(\gamma\) ustala się w pasie, w którym \(\Delta E\) jest porównywalne z \(\gamma\), a model z prawdziwym tracerem wyraźnie wygrywa z ZERO i z tracerem potasowanym.

### Kryteria pomiarowe

| Pomiar | Co pokazuje |
|---|---|
| przebieg \(\gamma\) (Lorentz) oraz \(\gamma_0\), \(\beta\) (Breit–Wigner) w trakcie uczenia | czy szerokość nie ucieka do skrajności |
| rozkład \(\Delta E_{ij}/\gamma_i\) na wierszach | czy most pracuje w paśmie, gdzie energia ma znaczenie |
| histogram \(\kappa_i\) (Breit–Wigner) | czy stabilizacja dynamiczna faktycznie działa |
| entropia wag uwagi na wiersz | czy most nie zwęża się do jednego klucza ani nie rozlewa |
| wynik vs ZERO (seal może zostać włączony — guard na ablację go wyłącza automatycznie) | wkład geometrii mostu ponad samo sprzężenie treści |
| wynik vs tracer potasowany między atomami | czy liczy się treść tracera, a nie sama obecność \(\Delta E\) |
| liczba NaN / pustych wierszy | poprawność obsługi brzegów |

> Uwaga: seal ma guard na ablację ZERO (sekcja ZERO, `most_seal.py`), więc wyniki ZERO przy domyślnie włączonym sealu mierzą sprzężenie treści \(C\), nie skupienie na CLS — osobny przebieg z wyłączonym sealem nie jest potrzebny do tego porównania.
