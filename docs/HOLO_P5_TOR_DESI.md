# Holo-Φ P5 — tor nauki wzorowany na DESI (τ = obserwator)

**Status:** projekt (nie uruchomione) — odpowiedź na P4a FAIL ([`HOLO_ZALOZENIA.md`](HOLO_ZALOZENIA.md) §8b, [`../doswiadczenie/glify/HOLO_WNIOSKI.md`](../doswiadczenie/glify/HOLO_WNIOSKI.md))
**Wzór:** [`../doswiadczenie/desi/`](../doswiadczenie/desi/) — jedyny tor, na którym Most realnie pokonał Softmax-twin (holdout +5,6 pkt)
**Zasada:** [[tracer-musi-byc-obserwatorem]] — τ musi być niezależnym pomiarem, nie f(hash(S/E))

---

## 1. Diagnoza — czym DESI różni się strukturalnie od P0–P4a

| Element | Holo-Φ P0–P4a (FAIL) | DESI (PASS na holdout) |
|---|---|---|
| **τ pochodzi z** | `hashlib.sha256(etykieta)` — ta sama etykieta co content | barwa fotometryczna g−r, zmierzona z pikseli, niezależna od `content` (luminancja/morfologia) |
| **Sygnał rodziny/klasy** | ręcznie wstrzyknięty `φ_root` (ta sama konstrukcja, którą test miał "odkryć") | etykieta cluster/field z zewnętrznego `hint`, nieskorelowana wprost z τ |
| **Gate testowany na** | r@1 w obrębie **tego samego, znanego** grafu (73 atomy) | **hold-out: obiekty nigdy niewidziane w train** |
| **Higiena splitu** | brak audytu przecieku augmentacji | plik-level dedupe (ra,dec), augment tylko train, assert brak przecieku |
| **Wynik** | P3: usunięcie fazy nic nie zmienia (Δ=0.000) → τ nie niosło informacji | Most +5,6 pkt nad Softmax **i** nad ZERO, wyłącznie na nowych obiektach |

Wniosek: FAIL P4a nie jest tylko kwestią hiperparametru (`blend_holo`) — jest **strukturalny**: τ=hash nie może dać przewagi, bo koduje dokładnie to, co już widzi content. DESI pokazuje też **drugą** lekcję, niezależną od pierwszej: prawdziwy dowód pojawia się na **generalizacji do nowych obiektów**, nie na retrieval w znanym grafie (na val in-distribution DESI też miał Most≈Softmax≈ZERO).

---

## 2. Nowy tor — P5

### 2.1 τ: zastąpić hash niezależnym pomiarem strukturalnym

Kandydaci (źródło: **Unihan** — zewnętrzna baza Unicode, nie liczone z `S`/etykiety w tym repo):

| Pomiar | Pole Unihan | Dlaczego to obserwator, nie hash |
|---|---|---|
| **liczba kresek** | `kTotalStrokes` | fizyczna własność zapisu znaku, niezależna od tego jak zbudowano `content` |
| **indeks radykału** | `kRSUnicode` / `kRSKangXi` | strukturalne przypisanie z zewnętrznego słownika, nie z hasha stringa |

`A` (amplituda) zostaje jak w P0 (częstość w korpusie). **`T` liczyć z `energy_to_T(stroke_count, ref=max_stroke_w_zbiorze)`** zamiast/obok obecnej energii fali — kanon P0 (`energy_to_T`) się nie zmienia, zmienia się **wejście**.

`φ` (faza) zostaje hash — to OK, faza buduje interferencję/klaster, nie steruje bezpośrednio seal (jak w P0). Czego **nie** robić: nie wstrzykiwać już ręcznie `φ_root` jako "sygnału rodziny" — to była tautologia z audytu pkt 3.

### 2.2 Etykieta zadania: zewnętrzna, nie wstrzyknięta

Zamiast ręcznej listy `["女","安","姿","数","好","如"]` z ręcznie przypisaną wspólną fazą — etykieta klasy = **prawdziwa przynależność radykałowa z Unihan** (`kRSKangXi` radical id). Test: czy Most grupuje znaki po radykale **korzystając wyłącznie z τ=stroke/radical**, nie z etykiety wstrzykniętej do `φ`.

### 2.3 Hold-out jako gate główny (nie r@1 w znanym grafie)

- Wybrać 2–3 radykały jako **korpus** (trenowalne), i **odłożyć całe, nigdy niewidziane znaki** tych samych radykałów (analogicznie do DESI: nowe obiekty tego samego typu `cluster`/`field`, nigdy w train) jako **holdout**.
- Gate główny: Most vs ZERO **na holdout**. Gate drugorzędny: Most vs Softmax na holdout.
- r@1 in-distribution (jak w P0–P4a) zostaje jako **kontrola sanity**, nie jako dowód.

### 2.4 Higiena danych (kopiowana wprost z audytu DESI, `AUDYT_2026-09-10.md`)

- Dedupe przed splitem: jeśli dump ma warianty tego samego znaku (augmentacja/warianty renderu), split **po znaku**, nie po wariancie — augmentacja tylko po stronie train.
- Jeśli glify mają różne rozmiary/padding: honorować maskę w `BridgeTransformer`/`SoftmaxTransformer` (to był bug B2 w DESI — sprawdzić czy w torze glify jest ten sam problem).
- Assert w kodzie: `set(train_ids) & set(holdout_ids) == set()` (jak `assert not (set(meta["train_files"]) & set(meta["val_files"]))` w `bench_desi.py`).

### 2.5 Determinizm — obowiązkowe przed jakimkolwiek sweepem

Replikacja P4a pokazała brak bit-reprodukowalności mimo `torch.manual_seed`. Przed P5:
```python
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
```
Potwierdzić dwa przebiegi tego samego seeda dają identyczny wynik **zanim** jakikolwiek pojedynczy seed wejdzie do raportu jako PASS.

### 2.6 Kolejność (ta sama dyscyplina P0→P4, teraz na nowym τ)

| Faza | Co | Kryterium przejścia dalej |
|---|---|---|
| **P5.0** | Policzyć `stroke_count`/`radical` dla korpusu z Unihan; zweryfikować że to NIE koreluje trywialnie z długością hasha/etykiety (audyt niezależności) | wykres stroke_count vs len(etykieta) — brak silnej korelacji |
| **P5.1** | Kontrola: ≥8 seedów, determinizm przypięty, krzywa ucząca do plateau, ziarnistość metryki na nowym (większym niż 73 atomy) zbiorze | jak P1, ale z τ=stroke |
| **P5.2** | Mini-sweep `blend_holo` **tylko jeśli P5.1 pokaże jakikolwiek sygnał** | nie sweepować w ciemno jak P2 |
| **P5.3** | Ablacja: τ=const (usuń stroke/radical, zostaw resztę) → sprawdzić czy klaster na **holdout** ginie | to teraz realny test (etykieta nie jest już wstrzyknięta w τ) |
| **P5.4** | Pełny gate: Most vs ZERO **na holdout**, ≥7/8 seedów | dopiero tu wolno mówić PASS/FAIL o metodzie |

---

## 3. Zakazy (rozszerzenie §7 `HOLO_ZALOZENIA.md`)

7. τ nie może być funkcją tego samego stringa co `content`/etykieta — musi dać się policzyć z zewnętrznego źródła (Unihan) bez znajomości hasha.
8. Gate PASS/FAIL raportować **na holdout**, nie na r@1 w znanym grafie — in-distribution to tylko sanity check.
9. Pojedynczy seed nie wchodzi do komunikacji jako dowód bez potwierdzonej bit-reprodukowalności (§2.5).

---

## 4. Co zostaje bez zmian

- Packing `d_x=8`, `n_dim=6`, `d_model=32` (kanon).
- Trzy tory: Softmax-twin | Most+seal | ZERO (`most_seal` logika P0 się nie zmienia — zmienia się tylko co wpuszcza jako `T`).
- `energy_to_T` — kanon normalizacji się nie zmienia, zmienia się wejście `E`.

Powiązane: [[tracer-musi-byc-obserwatorem]], [[holo-phi-p1-p4a-wnioski]], `desi_cutouts.py::pack_photometry` (wzorzec separacji content/τ), `bench_desi.py::train_eval_image` (wzorzec 3-torowego benchu), `image_batch.py::load_desi_train_val` (wzorzec splitu bez przecieku).
