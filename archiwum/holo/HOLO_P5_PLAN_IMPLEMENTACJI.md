# Holo-Φ P5 — plan implementacji i testów

**Status:** P5.0 + P5.1 **zrobione** → **FAIL holdout 0/8** (2026-09-13). P5.2+ wstrzymane (brak sygnału).  
**Plan bazowy:** [`HOLO_P5_TOR_DESI.md`](HOLO_P5_TOR_DESI.md) · wynik: [`HOLO_P5_1.md`](HOLO_P5_1.md)  
**Sesja planu:** 2026-09-11 · **Sesja wyniku:** 2026-09-13 (opcja A graf-per-radykał; 2 bugi seal/negatywy naprawione w trakcie)

---

## 0. Otwarta decyzja architektoniczna (do rozstrzygnięcia PRZED P5.1)

DESI ma holdout naturalnie: każdy obiekt to **osobny cutout/batch** — obiekt nigdy w train fizycznie nie wchodzi w żaden forward pass treningowy.

Holo-Φ ma **jeden duży graf transduktywny** (`phimost_learn.py::run_seed` → `recall_model.py::train_encoder`), retrieval InfoNCE po `query_idx`/`pos_idx` w obrębie tej samej sekwencji tokenów. Atom może być "held-out jako query/positive", ale jeśli siedzi w tej samej sekwencji co atomy treningowe, i tak dostaje kontekstualizację przez self-attention w trakcie treningu innych par — to **nie jest** ta sama gwarancja co "obiekt nigdy widziany" w DESI.

**Dwie opcje, wybrać jedną przed P5.1:**

| Opcja | Jak | Koszt |
|---|---|---|
| **A. Graf-per-radykał (rekomendowana)** | Każdy radykał = osobny mały graf/sekwencja (jak cutout w DESI). Train = sekwencje z radykałów treningowych. Holdout = sekwencje z radykału nigdy widzianego w żadnym forward passie treningowym. | Wymaga zmiany batchowania w `phimost_learn.py` (dziś: jeden graf → jedna/kilka sekwencji z paddingiem); zbliża Holo do `image_batch.py`-stylu (lista batchy, nie jeden graf). |
| **B. Maskowanie w jednym grafie** | Zostać przy jednym grafie, ale w treningu maskować atomy holdout tak, by nigdy nie wystąpiły w żadnej sekwencji treningowej (osobny `attention_mask`/filtrowanie sekwencji), doklejać je dopiero w ewaluacji. | Mniejsza zmiana kodu, ale trudniej **udowodnić** brak przecieku (self-attention w tej samej sekwencji, subtelne błędy jak B1/B2 z audytu DESI). |

Rekomendacja: **A** — bo to dokładnie ten sam wzorzec co `image_batch.py::load_desi_train_val` (plik = jednostka izolacji), więc higiena splitu (dedupe, assert braku przecieku) przenosi się 1:1 z DESI zamiast wymyślać nową.

---

## 1. Nowe / zmienione pliki

| Plik | Zmiana | Rola |
|---|---|---|
| `unihan_tracer.py` **(nowy)** | tabela `kTotalStrokes`/`kRSKangXi` dla korpusu znaków (bundlowana, nie live-download — offline, deterministyczne); `stroke_count(ch) -> int`, `radical_id(ch) -> int`, `load_unihan(chars: set[str]) -> dict[str, dict]` | źródło τ — niezależne od hasha |
| `holo_phi.py` **(rozszerzenie)** | nowy param `tracer_source: Literal["hash","unihan"] = "hash"` w `build_holo_graph`; gdy `"unihan"`: `T = energy_to_T(stroke_count(ch), ref=max_stroke_w_korpusie)` zamiast/obok obecnej energii fali; **usunąć** ręczne `φ_root` gdy `tracer_source="unihan"` (etykieta rodziny nie może być wstrzykiwana, gdy test ma być realny) | domyślny `"hash"` zachowuje istniejące testy `test_holo_AT.py` bez zmian — nowa ścieżka jest opt-in |
| `make_dump_holo_v2.py` **(nowy)**, wzorem `make_dump_holo.py` | generuje **osobne grafy per radykał** (opcja A) zamiast jednego grafu; zapisuje `holdout_radical` w manifest.json; korpus: 3-4 radykały treningowe + 1 radykał holdout (znaki nigdy w żadnym pliku treningowym) | dump wejściowy dla P5 |
| `bench_holo_v2.py` **(nowy)**, wzorem `bench_desi.py` | 3 tory Softmax\|Most+seal\|ZERO; `train_eval_holo(model, train_graphs, holdout_graphs, ...)`; gate główny = Most vs ZERO **na holdout**; `assert not (train_char_ids & holdout_char_ids)` | bench P5.1–P5.4 |
| `phimost_learn.py` / `recall_model.py` | **bez zmian w silniku** — `bench_holo_v2.py` woła istniejące `PhiBridgeEncoder`/`train_encoder` na nowych batchach z opcji A, tak jak `bench_desi.py` woła `BridgeTransformer` bez zmian w `bridge.py` | kanon zamku Lorentza nietknięty |
| dowolny skrypt uruchamiający bench (`bench_holo_v2.py`, `run_holo_p*.py`) | dodać na starcie `torch.set_num_threads(1)` (i rozważyć `torch.use_deterministic_algorithms(True)`) | wymóg z §2.5 `HOLO_P5_TOR_DESI.md` |

---

## 2. Kolejność wykonania (P5.0 → P5.4), z kryteriami przejścia

| Krok | Co | Plik(i) | Kryterium PASS → dalej |
|---|---|---|---|
| **P5.0** | Zbudować `unihan_tracer.py`; policzyć stroke/radical dla korpusu (≥4 radykały, po ~10-20 znaków każdy — większe niż 73 atomy/6 znaków z P0-P4a, żeby ziarnistość r@1 była drobniejsza) | `unihan_tracer.py` + `test_unihan_tracer.py` | testy P5.0 (§3) zielone; brak trywialnej korelacji stroke_count↔len(hash(etykieta)) |
| **P5.0b** | Rozstrzygnąć opcję A vs B (§0); jeśli A: zaprojektować kształt batcha per-radykał | notatka w `HOLO_P5_TOR_DESI.md` §0 lub commit message | decyzja zapisana, nie odkładana na "zobaczymy w trakcie" |
| **P5.1** | Kontrola eksperymentalna na nowym τ: ≥8 seedów, determinizm przypięty i **zweryfikowany bit-for-bit** (2 przebiegi tego samego seeda identyczne), krzywa ucząca do plateau, ziarnistość r@1 na nowym (większym) korpusie | `bench_holo_v2.py`, `test_holo_v2_determinism.py` | determinizm potwierdzony ORAZ krzywa faktycznie płaszczeje przed końcem epok |
| **P5.2** | Mini-sweep `blend_holo` **tylko jeśli P5.1 pokazał jakikolwiek niezerowy sygnał** (choćby 1 seed PASS) | `bench_holo_v2.py --blend-sweep` | — (krok warunkowy, może być pominięty) |
| **P5.3** | Ablacja: `tracer_source="unihan"` z `stroke_count := const` (usuń informację, zostaw resztę pipeline'u) → sprawdzić czy Most traci przewagę **na holdout** | `bench_holo_v2.py --ablate-tracer-const` | spadek Δ(Most−ZERO) na holdout gdy τ stały — to dowód, że to τ niosło sygnał, nie artefakt |
| **P5.4** | Pełny gate: Most vs ZERO na holdout, ≥7/8 seedów, zbiorczy raport (nie pojedynczy seed) | `bench_holo_v2.py --full-gate`, wynik → `doswiadczenie/glify/WYNIKI_HOLO_P5.md` | to jest punkt, w którym wolno napisać PASS/FAIL o metodzie |

---

## 3. Plan testów (styl repo: gołe `test_*` + `assert`, bez frameworka — jak `test_holo_AT.py`/`test_karmazyn_lingua.py`)

### `test_unihan_tracer.py`
- `test_stroke_count_known_chars()` — 女=3, kilka innych znanych znaków ze zweryfikowanymi wartościami z realnej tabeli Unihan (nie zgadywać liczb — wpisać dopiero po sprawdzeniu źródła przy implementacji).
- `test_stroke_count_not_hash_derived()` — dla dwóch znaków o identycznej długości etykiety/stringa ale różnej liczbie kresek: `stroke_count` się różni (odrzuca "to tylko inny hash").
- `test_radical_groups_match_known_family()` — znaki z tego samego radykału Unihan mają ten sam `radical_id`; znaki spoza — inny.
- `test_no_trivial_correlation_with_label_hash()` — koreluje `stroke_count(ch)` z `int(hashlib.sha256(ch).hexdigest(), 16) % N`; assert brak silnej korelacji (np. |Pearson| < 0.3) — audyt niezależności z P5.0.

### `test_holo_v2_split.py` (opcja A — graf-per-radykał)
- `test_no_char_leak_train_holdout()` — `assert not (train_char_ids & holdout_char_ids)`.
- `test_holdout_radical_absent_from_any_train_sequence()` — żaden atom holdout nie występuje w żadnej sekwencji/batchu treningowym (nie tylko "nie jest query") — to jest właściwy odpowiednik `assert not (set(train_files) & set(val_files))` z `bench_desi.py`.
- `test_manifest_records_holdout_radical()` — `manifest.json` z `make_dump_holo_v2.py` zawiera jawnie `holdout_radical` i listę holdout char ids (audytowalność, jak `family_ids.json` dziś).

### `test_holo_v2_determinism.py`
- `test_same_seed_bitwise_reproducible()` — dwa przebiegi `bench_holo_v2` z tym samym seedem i `torch.set_num_threads(1)` dają identyczne (do epsilon maszynowego) ΔSM/r@1 — musi przejść **zanim** P5.1 uzna jakikolwiek wynik za wiarygodny (replikacja P4a pokazała, że bez tego pojedynczy seed to szum).

### `test_bench_holo_v2.py` (smoke na małym syntetycznym korpusie, szybki w CI)
- `test_three_tracks_shapes_sane()` — Softmax/Most/ZERO zwracają metryki o oczekiwanym kształcie na miniaturowym (2 radykały × 4 znaki) dumpie.
- `test_zero_degenerates_toward_softmax()` — własność sanity z formuły zamka (τ=0 → zamek Lorentza degeneruje się): `abs(zero_r1 - softmax_r1)` mniejsze niż jakiś luźny próg, na tym samym syntetycznym dumpie — regresja pilnująca, że ktoś przypadkiem nie złamie degeneracji ZERO przy zmianach w `bridge.py`.
- `test_gate_computation_on_holdout_not_indistribution()` — assert że `bench_holo_v2.py` liczy gate PASS/FAIL z metryk na `holdout_graphs`, nie na `train`/val in-distribution (regresja pilnująca lekcję #2 z DESI — łatwo przez pomyłkę wrócić do starego gate'u).

### Regresja na istniejącym kodzie (nie psuć P0–P4a)
- `test_holo_AT.py` musi dalej przechodzić bez zmian (domyślny `tracer_source="hash"` w `build_holo_graph`).
- `test_karmazyn_lingua.py` bez zmian (inny moduł, nie dotknięty).

---

## 4. Definicja "zrobione" na tę sesję

Ta sesja: **plan zapisany** (ten plik + `HOLO_P5_TOR_DESI.md`), **kod nie napisany**. Następna sesja zaczyna od P5.0 (`unihan_tracer.py` + jego testy) po rozstrzygnięciu §0.

Powiązane: [[tracer-musi-byc-obserwatorem]], [[holo-phi-p1-p4a-wnioski]], `HOLO_P5_TOR_DESI.md`, `archiwum/desi/AUDYT_2026-09-10.md` (wzorzec audytu przecieku/maski, do powtórzenia dla opcji A).
