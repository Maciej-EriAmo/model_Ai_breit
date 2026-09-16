# Holo-Φ P5.1 — kontrola na τ=Unihan (stroke_count), holdout=radykał

**Data:** 2026-09-13
**Plan:** [`../../docs/HOLO_P5_PLAN_IMPLEMENTACJI.md`](../../docs/HOLO_P5_PLAN_IMPLEMENTACJI.md) ·
[`../../docs/HOLO_P5_TOR_DESI.md`](../../docs/HOLO_P5_TOR_DESI.md)
**Kod:** `unihan_tracer.py`, `make_dump_holo_v2.py`, `bench_holo_v2.py`

---

## Setup

- τ (T) = `energy_to_T(stroke_count(ch), ref)` — **prawdziwy pomiar Unihan**
  (Unicode 17.0, `kTotalStrokes`/`kRSUnicode`), nie hash(etykieta). Niezależność
  od hasha zweryfikowana w P5.0 (`test_unihan_tracer.py`: |Pearson| z hashem < 0.3).
- Zadanie/etykieta = **radykał Kangxi** (fakt słownikowy, nie wstrzyknięty ręcznie).
- Split (opcja A, §0 planu): 5 radykałów × 20 znaków. Train = 4 radykały
  (9 人, 38 女, 75 木, 86 火). **Holdout = radykał 85 (水), CAŁKOWICIE nieobecny
  w żadnym forward passie treningowym** (zweryfikowane: `test_holo_v2_split.py`).
- Determinizm: `torch.set_num_threads(1)` + `use_deterministic_algorithms` +
  reset RNG przed budową danych — zweryfikowany bit-do-bitu
  (`test_holo_v2_determinism.py`) **przed** sweepem seedów, zgodnie z §2.5.
- Gate: Most vs ZERO na holdout r@1 (≥+0.02), 8 seedów, ≥7/8 = PASS metody.

## Dwa bugi znalezione i naprawione W TRAKCIE tej sesji (audyt kodu, nie tylko wyniku)

Pierwszy przebieg dał FAIL (1/8), ale zanim uznano to za wynik, użytkownik
zażądał audytu samego silnika ("sprawdź poprawność kodu transformera i czy
faktycznie działa") — słusznie, bo znaleziono dwa realne bugi:

1. **Brak negatywów w treningu.** `build_train_batch` (v1) budował JEDEN graf
   = JEDEN radykał — każdy kandydat w puli BYŁ prawdziwą rodziną. Model nigdy
   nie musiał niczego dyskryminować, więc τ nie miało żadnej pracy (dowolne
   samo-spójne osadzenie 20 elementów spełnia InfoNCE). Fix: `build_mixed_train_batch`
   — jeden graf = wszystkie 4 radykały treningowe razem, pozytyw = ten sam
   radykał co query, kandydaci = cały graf (prawdziwe negatywy cross-radical).
2. **ZERO-tracer martwy przez seal.** `RecallBatch.ablate(zero_tracer=True)`
   zeruje CAŁY tracer → T=0 wszędzie → `most_seal.apply_t_key_seal`'s hard floor
   (T<0.20) maskował **100% kluczy** (potwierdzone bezpośrednim testem: `fraction
   hard-masked = 1.0`). "ZERO" miało matematycznie degenerować Most do
   `A=C/γ` (pełna, nie-zerowa macierz uwagi z formuły zamka przy ΔE=0) — zamiast
   tego dostawał martwą uwagę, INNY (i dużo gorszy) eksperyment niż zamierzony.
   Fix w `most_seal.py`: wykrywanie "cały tracer=0" (ablacja) odróżnione od
   "realnie zimny atom" (zawsze ma niezerowy depth one-hot) — ablowane wiersze
   pomijają seal całkowicie. **To dotyczyło każdego benchu z ZERO-tracer w
   całym projekcie** (Holo P0-P4a, DESI, dump recall), nie tylko P5.1 — zob.
   `docs/AUDYT.md` §"Audyt seal".

Po obu fixach: `test_most_seal.py`, `test_holo_v2_split.py`,
`test_holo_v2_determinism.py` i cały istniejący test suite (20/20) — zielone.
Bezpośrednia inspekcja wag uwagi potwierdziła mechanizm **działa poprawnie**:
attention realnie różni się między prawdziwym τ a zero_tracer (|Δw|≈0.0245),
zarówno przed jak i po treningu — silnik Most robi to, co deklaruje.

## Wynik ostateczny (8 seedów, epochs=60, po obu fixach)

| seed | Most | SM | ZERO | ΔZERO | ΔSM | train_sanity | gate |
|-----:|-----:|---:|-----:|------:|----:|--------------:|:----:|
| 0 | 0.010 | 0.010 | 0.005 | +0.0050 | +0.0000 | 1.000 | FAIL |
| 1 | 0.005 | 0.005 | 0.005 | +0.0000 | +0.0000 | 1.000 | FAIL |
| 2 | 0.020 | 0.015 | 0.015 | +0.0050 | +0.0050 | 1.000 | FAIL |
| 7 | 0.035 | 0.010 | 0.035 | +0.0000 | +0.0250 | 1.000 | FAIL |
| 11 | 0.005 | 0.025 | 0.005 | +0.0000 | −0.0200 | 1.000 | FAIL |
| 17 | 0.005 | 0.030 | 0.020 | −0.0150 | −0.0250 | 0.938 | FAIL |
| 23 | 0.005 | 0.010 | 0.005 | +0.0000 | −0.0050 | 1.000 | FAIL |
| 42 | 0.015 | 0.025 | 0.010 | +0.0050 | −0.0100 | 0.812 | FAIL |

**P5.1 FAIL: 0/8** · mean ΔZERO=**0.0000** · mean ΔSM=**−0.0038** — dokładne
zero, w żadną stronę. Wynik NIE zmienił się jakościowo po naprawie obu bugów
(przed fixami: FAIL 1/8, mean ΔZERO≈−0.003; po: FAIL 0/8, mean ΔZERO=0.000) —
bugi były realne i warte naprawy, ale nie one stały za FAIL. JSON: `holo_p5_1_results.json`.

## Diagnostyka krytyczna — dlaczego, nie tylko czy

Krzywa uczenia (wymóg §2 planu: "krzywa ucząca do plateau"), żeby odróżnić
"τ nie pomaga" od "model niczego się tu nie uczy" (floor effect) — identyczna
dla Most, Softmax i (po naprawie) ZERO:

| epoka | holdout r@1 (wszystkie 3 tory) | train r@1 (in-distribution, mieszany graf) |
|------:|------------:|--------------------------------------------------:|
| 0 | ~0.02 | 0.000 |
| 30 | ~0.02 | 0.25–0.88 |
| 45–60 | ~0.01 (lekko SPADA) | **1.000** |
| 60–119 | ~0.01, płasko | 1.000 (plateau) |

**Model uczy się doskonale** (train r@1 → 100%, dla WSZYSTKICH trzech torów,
także po dodaniu prawdziwych negatywów cross-radical) — to nie jest
niedouczony/zepsuty setup, i nie jest to już artefakt brakujących negatywów.
Ale **holdout r@1 nigdy nie odrywa się od poziomu szansy (~1/51≈0.02) i lekko
SPADA** w miarę jak train-accuracy rośnie do 100%, identycznie dla Most/SM/ZERO
— klasyczny podpis **przeuczenia na tożsamość konkretnych 80 atomów
treningowych** (dzięki residual connections + unikalne content-hashe każdego
znaku, sieć może "przeciekać" tożsamość wprost do wyjścia, z pominięciem
uwagi), nie brak/obecność mostu Lorentza.

## Wniosek

1. **Silnik (Most/Bridge attention) działa poprawnie** — zweryfikowano kodowo
   i eksperymentalnie (wagi uwagi realnie reagują na τ; oba znalezione bugi
   były w warstwie eksperymentu/kontroli ablacji, nie w formule
   `A=γC/(γ²+ΔE²)` ani w propagacji gradientu).
2. **P5.1 FAIL, z innej przyczyny niż P4a.** P4a: τ=hash(etykieta) było
   tautologią. P5.1: τ jest realne i niezależne (Unihan), a mimo to **żaden**
   z trzech torów nie generalizuje na holdout — wszystkie płasko na poziomie
   szansy, wszystkie osiągają 100% na treningu. Wąskim gardłem nie jest wybór
   między zamkiem Lorentza a Softmaxem — jest nim **skala/reprezentacja**:
   `content` = `hash_embed(znak)` nie niesie cechy współdzielonej między
   radykałami, a tylko 4 rodziny × 20 elementów to zbyt mało, by odróżnić
   "nauczył się reguły τ→rodzina" od "zapamiętał 80 tożsamości" — obie
   hipotezy dają identyczny trening-r@1=100%.
3. Próba zmniejszenia pojemności modelu (d=8, 1 warstwa) spowalnia
   zapamiętywanie (train_r1 tylko do ~31% po 150 epokach), ale NIE ujawnia
   różnicy Most vs Softmax na holdout — sugeruje, że problem jest w skali
   danych, nie w architekturze.
4. Zgodnie z regułą stop/go planu (§2.6) — **nie uruchamiam P5.2** (sweep
   tylko przy jakimkolwiek sygnale; tu mean Δ = dokładne zero).

## Rekomendacja na przyszłość (nie zaimplementowane w tej sesji)

Żeby test miał szansę wykazać przewagę Mostu, potrzeba **skali** (dziesiątki
radykałów, nie 4 — analogicznie do tego, co realnie zadziałało w DESI: duża
liczba obiektów pokonuje przeuczenie na tożsamość) i/lub `content`, który
niesie strukturę współdzieloną między znakami (dekompozycja graficzna/IDS
zamiast `hash(cały znak)`). Baza Unihan już pobrana lokalnie — rozszerzenie
korpusu jest tanie, jeśli ta ścieżka ma być kontynuowana.

---

Jednym zdaniem: **kod transformera działa poprawnie (zweryfikowane wagami
uwagi i naprawą dwóch realnych bugów kontrolnych); P5.1 FAIL trzyma się po
obu naprawach — to przeuczenie na tożsamość przy zbyt małej skali danych, nie
defekt mechanizmu Most ani brak/obecność mostu Lorentza.**
