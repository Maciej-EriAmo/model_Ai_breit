# Notatka rozwojowa — wnioski po mapie faz

Data: 2026-09-30.  
Stan kodu: lokalny `main`. Notatka weszła jako `7c07d62`. W drzewie, jeszcze bez commita: korekta jakobianu, dym bez pinezki `beta=16`, batch `(B, N)` i `update_inputs`. Zdalnego repozytorium nie ma.  
`pytest -q`: 38 testów zielonych. Mapy: `artifacts/two_well`, `artifacts/single_well`, `artifacts/near_degenerate`, plus dodatkowa `artifacts/single_well_boundary`.

## Werdykt

Głowa działa jako jedna mapa stanu `S`. Pomiary to potwierdzają: wagi są Breita–Wignera, suma zostaje na simpleksie, `delta_E` stoi w całej pętli `K`, a lokalny promień widma jednego kroku zgadza się z obserwowanym okresem 2, gdy pętla jest dość długa.

Plan w punkcie 8 chciał czytelnej granicy `beta(rho)`. Ta granica jest czytelna. Architektury przy tym wyniku się nie zmienia.

Jedna liczba z planu nie zachodzi. Dym `N=8`, `beta=8`, `rho=1` miał dać oscylację albo mieszankę. Przy wzorze `gamma = gamma0 * (1 + beta * kappa)`, przy `gamma0=0.3` i tle studni `BACKGROUND_DELTA=3.5`, rozmiar `N=8` zostaje ściągający. To jest zła prognoza dla tego rozmiaru, nie zepsute jądro.

Głowa nie ma własnych wag. `C` i `delta_E` wchodzą z zewnątrz, kształt `(N,)`. `beta`, `rho` i `gamma0` są pokrętłami siatki. Mapa potwierdza jądro. Nie ma jeszcze czego uczyć.

## Co widać na mapach

Siatka planu: `N=32`, 100 startów, `K=64`, `gamma0=0.3`, `beta` do 8, `kappa=max`.

- `two_well` i `near_degenerate` zbiegają w każdej komórce.
- `near_degenerate` zostaje płaskie: entropia około `log(32) ≈ 3.466`, przy `beta=8`, `rho=1` promień około `1.9e-4`, mediana kroków 3.
- `two_well` przy `beta=8`, `rho=1` zbiega (mediana 30 kroków), ale mediana promienia jest pusta: 100/100 jakobianów to remis `argmax` dwóch równych studni. Taki promień nie wchodzi do linii na `boundary.png`.
- `single_well`, `beta=8`, `rho=1`, `K=64`: etykieta oscylacja, 100/100, promień `0.916`. Punkt stały jest stabilny. Dzwonienie nie schodzi pod `tol=1e-6` w 64 krokach. Etykieta `converged` wygrywa dopiero wtedy, gdy końcowe `dS` jest pod tolerancją, więc ucięta trajektoria dostaje etykietę oscylacji.
- Ta sama komórka przy `K=200` (`artifacts/single_well_boundary`, 40 startów): zbiega, mediana kroków 150, promień `0.914`, zero niewiarygodnych jakobianów.

Czysty okres 2 i promień powyżej 1 idą razem, na jednej studni, przy `rho=1`:

| beta | mediana kroków | mediana rho(J) | etykieta |
|------|----------------|----------------|----------|
| 8    | 150            | 0.914          | converged |
| 12   | 200            | 1.605          | oscillate |
| 16   | 200            | 2.871          | oscillate |
| 24   | 200            | 4.376          | oscillate |

Przy `rho=0.2` i `rho=0.5` ta sama studnia zbiega jeszcze przy `beta=24`, a promień zostaje przy `|1-rho|`.

SSRT w tym jądrze to istnienie progu i lokalne kryterium jednego kroku. To nie jest dowód globalnej zbieżności.

## Odstępstwa od Plan.txt

Żadne z trzech nie zmienia wzoru wag ani kroku relaksacji. Żadne nie otwiera pętli uczącej.

1. **Jakobian.** Plan (PR5) chciał kolumn z różnic skończonych. Mapa dalej bierze promień ze wzoru analitycznego `J = (1-rho) I + rho * dW/dS`, rzędu jeden przez `dW/dgamma`. Obok stoi `finite_difference_jacobian`: kolumna `k` to `S' = normalize(S ± h e_k)` i centralna różnica `W`, ten sam wzór `J`. Przy jednoznacznym liderze `max` oba promienie się zgadzają. Luka dwóch największych składników `≤ 1e-8`, albo pchnięcie, które zmienia `argmax`, zostaje `unreliable` i nie wchodzi do mediany. Mediana mapy zostaje przy wzorze analitycznym. Wzoru wag to nie rusza.

2. **Test dymny.** `uniform_flat` i `single_well` idą siatką z planu: `N=8`, 10 startów, `beta` w `{0, 1, 8}`, `rho` w `{0.2, 0.5, 1}`. `uniform_flat` zawsze zbiega, `W` jest płaskie. `single_well` przy małym `beta` i `rho≤0.5` zbiega w co najmniej 90%. Komórka `beta=8`, `rho=1` na `N=8` też zbiega: prognoza oscylacji z planu przy tym rozmiarze nie zachodzi, więc dym tego nie udaje. Pinezki `beta=16` nie ma. Osobny, krótki bieg `N=32` sprawdza tylko, że gdzieś na siatce jest komórka z `frac_oscillate ≥ 0.5`, promieniem `> 1` i zerem niewiarygodnych jakobianów, bez nazwy bety. Studni nie ruszano. Granica przy `N=32` zostaje.

3. **`most1` w torchu.** Plan każe po trzech reżimach stanąć, a `tsra/tsra_torch.py` dopisać dopiero jako osobny ticket po czytelnej granicy. Plik jest w drzewie, bo ten ticket został zamówiony po mapie. Liczy jeden strzał `S = W(gamma(kappa(S0)))`, ten sam co numpy przy `K=1`, `rho=1`. Nie ma `torch.nn` ani pętli uczącej. Zostaje. Usunięcie cofnęłoby zamówiony ticket i nie naprawiłoby jądra.

Drobniejsze różnice litery, bez wpływu na krok: plik specyfikacji nazywa się `Plan.txt`, checkboxy planu są puste, CSV ma dodatkową kolumnę `seed` (plan osobno każe zapisać ziarno), `frac_mixed` to 0 albo 1 (wskaźnik, że komórka dostała więcej niż jedną etykietę), a `artifacts/single_well_boundary` jest siatką ponad trzy wymagane biegi.

## Czego następny krok nie robi

Nie stroi dalej siatki `beta≤8`. Nie dokłada stosu, FFN, softmaxu ani pozycji.

Głowa przyjmuje `(B, N)`: każdy wiersz to własny simpleks, a `gamma0`, `beta`, `rho` i `K` są wspólne. `(N,)` zostaje.

`update_inputs` woła `most1`, a potem schodzi krok po gradiencie skalara podanego z zewnątrz. `C` i `delta_E` zmieniają się dopiero po powrocie. `S0` się nie uczy. Skalar nie jest wpisany w bibliotekę: wołający go przynosi.

Jeśli próg ma zejść w stronę mniejszego `beta` albo w stronę `N=8`, zmienia się wyłącznie reżim `delta_E` albo `gamma0`, potem ten sam `scripts/run_phase.py`.
