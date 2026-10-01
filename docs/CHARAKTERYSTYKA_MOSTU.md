# Charakterystyka mostu

**Lorentz:** `C:\Users\drwis\model Ai`  
**Breit–Wigner (Most-1):** `C:\Users\drwis\model_Ai_breit`  
**Kanon rozmiaru:** 2 warstwy × 2 głowy = 4 jądra, `d_model=32`, tracer `n_dim=6`

Lorentz stawia most między daną a pamięcią. Breit–Wigner jest rezonansem na tym samym moście: szerokość przejścia oddycha z tym, jak mocno jedna pamięć bierze sprzężenie.

## Dana i pamięć

Dana to atom-zapytanie: treść `content` (`x`) i obserwator `tracer` (`τ`).  
Pamięć czytana w uwadze to pozostałe atomy przestrzeni Φ (M2).  
Wagi, które uczą się *jak* patrzeć, siedzą w M1: projekcje `QKV`, projekcja energii `e`, szerokości.

Treść buduje zapytanie, klucz, wartość i sprzężenie. Tracer buduje energię.

## Wspólny szkielet

Dla głowy o wymiarze \(d_h\):

\[
S_{ij}=\frac{Q_i\cdot K_j}{\sqrt{d_h}},\qquad
C_{ij}=\mathrm{softplus}\!\left(\frac{S_{ij}}{\tau}\right)^2
\]

\[
E_i=W_e\tau_i+b_e,\qquad
\Delta E_{ij}=e_{\mathrm{scale}}\,(E_i-E_j)
\]

W `BridgeTransformer` jest \(\tau=1\) i \(e_{\mathrm{scale}}=1\). W enkoderze recallu oba są gałkami i domyślnie też są równe 1. Sprzężenie kanonu to `softplus`, nie `relu`.

Po wadze wiersz jest normalizowany:

\[
w_{ij}=\frac{A_{ij}}{\sum_k A_{ik}}.
\]

W mianowniku jądra kod dodaje \(10^{-12}\).

## Lorentz — most

Katalog `model Ai`. Jedna szerokość na głowę, uczona jako \(\log\gamma\):

\[
\gamma=\mathrm{softplus}(\log\gamma)+10^{-3}
\]

\[
A_{ij}=\frac{\gamma\,C_{ij}}{\gamma^2+\Delta E_{ij}^2}
\]

Most między daną \(i\) a pamięcią \(j\) otwiera się, gdy \(\Delta E_{ij}\) jest małe, i gaśnie, gdy rozstrojenie rośnie. Przy starcie \(\log\gamma=0\), więc \(\gamma\approx 0{,}693\).

Dwa dalsze tryby są w kodzie, domyślnie wyłączone. Ustawiony zamek to `square`.

`onesided` — szerokość zależy od znaku energii:

\[
\gamma_{ij}=\gamma\,\bigl(1+\mathrm{softplus}(\alpha\,\Delta E_{ij})\bigr).
\]

`directed` — gotowa waga dostaje bramkę, z podłogą \(b=0{,}25\):

\[
A_{ij}=A^{\mathrm{square}}_{ij}\,\bigl(b+(1-b)\,\sigma(\alpha\,\Delta E_{ij})\bigr).
\]

W tym forku oba tryby są odrzucone.

## Breit–Wigner — rezonans na moście

Ten katalog. Przejście zostaje lorentzowskie, a szerokość jest własnością wiersza:

\[
\kappa_i=\max_j\frac{C_{ij}}{\sum_k C_{ik}}
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

\(\kappa\) bliskie 1 znaczy, że jedna pamięć wzięła sprzężenie: most puszcza szersze \(\Delta E\). \(\kappa\) małe znaczy sprzężenie rozlane po wielu atomach: zostaje wąska szczelina przy \(\gamma_0\).

Start: \(\log\gamma_0=0\) i \(\beta_{\mathrm{raw}}=0\), więc \(\gamma_0\approx\beta\approx 0{,}693\). Szerokość od pierwszego kroku zależy od szczytu wiersza.

\(\kappa\), \(\beta\) i \(\gamma_0\) biorą się ze sprzężenia \(C\) i z wag M1. Tracer ich nie ustawia.

## Rola tracera w obu

Tracer jest obserwatorem. Sześć osi kanonu: jedynka głębokości (3), \(T\), stan, mazur. Wszystkie wchodzą do energii \(E\) przez liniową projekcję \(W_e\). Przy starcie wagi tej projekcji są jedynkami, więc energia głowy jest sumą osi \(\tau\).

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

W Lorentzcie seal mnoży gotową wagę \(A\). W tym forku seal dostaje sprzężenie \(C\) jeszcze przed jądrem, razem z zerem na przekątnej i paddingiem. Twarde zero przez to nie wchodzi do \(\kappa\). Ciągła kara COLD też wchodzi do \(\kappa\), więc przesuwa \(\gamma\) całego wiersza.

## Co jest ustawione

| | Lorentz | Breit–Wigner |
|--|---------|----------------|
| Jądra | 4 (2×2) | 4 (2×2) |
| Człony wagi | \(C\), \(\Delta E\), jedno \(\gamma\) | \(C\), \(\Delta E\), \(\kappa\), \(\gamma_0\), \(\beta\) |
| Znak \(\Delta E\) | dostępny (`directed`, `onesided`), domyślnie `square` | odrzucony |
| Przekątna w enkoderze i klasyfikatorze | zero na \(A\), po jądrze | zero na \(C\), przed \(\kappa\) |
| Tiny LM | przekątna zostaje, maska przyczynowa | tak samo |
| Seal \(T\) | na \(A\) | na \(C\) |

## ZERO

Wyzerowany tracer daje \(\Delta E=0\). Szerokość jest wtedy wspólna dla całego wiersza i skraca się przy normalizacji. W obu zostaje

\[
w_{ij}=\frac{C_{ij}}{\sum_k C_{ik}}.
\]

Most się nie formuje. Zostaje samo sprzężenie treści.
