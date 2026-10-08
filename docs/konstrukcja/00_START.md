# Konstrukcja jądra Most — start dla inżyniera

**Data:** 2026-10-07  
**Gdzie pracować:** `C:\Users\drwis\model_Ai_breit`  
**Komora sesji:** `Most_Lorentz` (pamięć agenta, nie wagi)

Ten katalog jest kompletem do opracowania. Specyfikacja to pliki `00`–`09`. Obok leżą kopie notatek pomiarów, charakterystyki i planu TSRA. Oryginały zostają na swoich ścieżkach; przy rozjechaniu się tekstu wygrywa oryginał. Spis kopii: [`SPIS.md`](SPIS.md).

Wrześniowe `docs/STATUS.md` i `docs/PLAN_NEXT.md` opisują starszy stan i do tego katalogu nie weszły. Żywy wynik jądra jest w `PODSUMOWANIE_JADRA.md` w tym katalogu.

## Kolejność czytania

| # | Plik | Po co |
|---|------|--------|
| 1 | [`01_ZAKRES.md`](01_ZAKRES.md) | co jest jądrem, co jest szkieletem, czego nie budować |
| 2 | [`02_WZOR.md`](02_WZOR.md) | wzór Most-1 tak, jak liczy go kod |
| 3 | [`03_KONTRAKT.md`](03_KONTRAKT.md) | kształty, maski, pusty wiersz, seal, znak ΔE |
| 4 | [`04_SZKIELET.md`](04_SZKIELET.md) | skąd bierze się C i ΔE w trzech wywołaniach |
| 5 | [`05_TRZY_DRZEWA.md`](05_TRZY_DRZEWA.md) | Lorentz, ten fork, biblioteka TSRA |
| 6 | [`06_POMIARY.md`](06_POMIARY.md) | fakty, których zmiana kodu nie może przekręcić |
| 7 | [`07_OTWARTE.md`](07_OTWARTE.md) | miejsca do opracowania |
| 8 | [`08_SPRAWDZENIE.md`](08_SPRAWDZENIE.md) | testy, które muszą zostać zielone |
| 9 | [`09_ZAKAZY.md`](09_ZAKAZY.md) | czego nie ruszać przy okazji |

## Trzy zdania, zanim ruszysz plik

Jądro zamienia sprzężenie `C` i rozstrojenie `ΔE` na wagę wiersza. Treść buduje `C` przez `softplus(QK)²`. Tracer buduje `ΔE`. Szerokość wiersza rośnie ze szczytem sprzężenia.

Wagi po normalizacji zależą od `ΔE²`. Znak rozstrojenia do wagi nie wchodzi.

Biblioteka w `C:\Users\drwis\transformer2` liczy ten sam ułamek na gotowych `C` i `ΔE`. Nie jest uwagą nad sekwencją i ten fork jej nie importuje.
