# Audyt: holograficzne przedstawienie znaków

**Data:** 2026-09-15  
**Root:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz`  
**Werdykt testów (dziś):** `pytest test_karmazyn_lingua.py` → **35 passed in 1.69s**

---

## 1. Mapa torów (nie mylić)

| Tor | Co to jest | Status względem „znaków” |
|-----|------------|---------------------------|
| **karmazyn_lingua** (`karmazyn_lingua.py`) | Hologram jako nośnik cech znaku/morfemu na realnym Store+HRR | **Główny tor audytu** — retrieval + kompozycja n+1 |
| **Lingua P1** (`run_lingua_p1.py`) | Most vs Softmax na małym dumpie (11 atomów) | FAIL / confound „za mały graf” |
| **Lingua MLP probe** | Trenowany dekoder vs analityczny NN | DONE — hologram_close niesie sygnał (0.72), symbolic nie |
| **Kana P1** (`run_kana_p1.py`) | Cel pierwotny raportu: czy hologram przyspiesza trening Mostu | **DONE: 0/5 PASS; r@1≈podłoga (N=286). `KANA_P1.md`** |
| **Kana P1b** (`run_kana_p1b.py`) | 46 gojūon hiragana, 64 epoki, N=59 | **DONE: above_floor 3/3; mean Most≈0.29; gate 0/3 (Most≈SM). `KANA_P1B.md`** |
| **Holo-Φ P0–P5** (glify Unihan / DESI) | Inny eksperyment (rodzina glifów / τ-obserwator) | Osobny tor; nie mylić z lingua znaków |

---

## 2. Raport vs kod (dryf dokumentacji)

Źródło koncepcji: `hologram-znaki-raport.md` (**untracked**, kopia `(1).md` też).  
Sekcja **11.3** raportu nadal mówi „TODO / niezaimplementowane” — **to już nieprawda**. Kod i commity domknęły 6 punktów + bugfix reach-GC.

| # | Item z raportu §11.3 | Stan kodu | Dowód |
|---|----------------------|-----------|--------|
| 1 | `SYMBOLIC_ONEHOT` | **DONE** | `AtomEncoding.SYMBOLIC_ONEHOT`, testy `test_onehot_*`; commit `f221e4b` |
| 2 | Projektor skanujący | **DONE** | `Projector.scan` / `best`; testy `test_projector_*`; `58c5a91` |
| 3 | Ekstraktor cech nowego znaku | **DONE** | `glyph_features_from_unicode` + `PhaseRegistry`; `248f186` |
| 4 | `observe_with_confidence` | **DONE** (było w raporcie częściowo; domknięte + skalibrowane) | testy confidence; `8dc1c1e` |
| 5 | Yōon sekwencyjne `compose()` | **DONE** | `test_yoon_sequential_compose_*`; `1806ee1` |
| 6 | Trwałość atomów (reach-GC / tick) | **DONE** | `test_*_reach_gc_*`; `1806ee1` |
| + | Cicha degradacja `unbind_position` po GC | **DONE** (bug znaleziony po drodze) | `RuntimeError` zamiast cichego blind; `1ccba2e` |

**Commity wątku (skrót):**  
`f221e4b` → `58c5a91` → `248f186` → `8dc1c1e` → `1806ee1` → `1ccba2e` → `8d7f2e0` (Kana P1 harness).

---

## 3. Co działa empirycznie (retrieval / dekoder)

| Eksperyment | Wynik | Znaczenie |
|-------------|-------|-----------|
| Overlay hiragana 71/71 (raport §10) | r@1=1.0 na 3 osiach | Płaski overlay skaluje się na prawdziwym piśmie |
| Sekwencja compose — yōon 33 | r@1=1.0 (wiersz + mały kana) | Pierwszy sekwencyjny test na realnym piśmie; N=11 OK |
| Sweep pojemności compose | próg ~N=10–12 przy `FEATURE_DIM=64` | Limit sekwencji ≠ limit overlay |
| SYMBOLIC vs HOLOGRAM (fazy) | symbolic≈przypadek; hologram daleki=1.0 | Faza niesie info tylko przy separacji + 2D Re/Im |
| MLP probe | close: 0.72 vs analityczne 0.33 | Analityczny NN zaniża nośność hologramu |
| Lingua P1 (Most) | 0/5 PASS, Most≈SM≈ZERO | Za mały graf (11 atomów) — nie rozstrzyga Mostu |
| test_karmazyn_lingua (dziś) | **35/35 PASS** | Regresja warstwy zielona |

---

## 4. Co nadal otwarte

| Priorytet | Lukа | Stan |
|-----------|------|------|
| ~~P0~~ | ~~Cel pierwotny (Kana P1)~~ | **Rozstrzygnięty negatywnie pod tym protokołem:** hologram 0/5, symbolic 0/5; mean Most≪bramki; Most≈ZERO. Caveat: absolutne r@1 przy podłodze (~0.01) — możliwy underfit (epochs/n_train/task), nie tylko „hologram nie pomaga” |
| ~~P1~~ | ~~Odświeżenie `hologram-znaki-raport.md` §11.3~~ | **DONE** 2026-09-15 (6/6 w raporcie) |
| P1 | Raport nadal **untracked** (`??`) | Ryzyko utraty — `git add hologram-znaki-raport.md` |
| P2 | `observe_with_confidence` na długich sekwencjach (sim≈0.1) | Nadal nieosobno przetestowane |
| P2 | Skala Mostu na Kana (po wynikach P1) | Zależy od P0 |
| P3 | Kanji / >FEATURE_DIM wartości na jednej osi | Świadomie poza Kana P1 |

---

## 5. Jakość kodu (krótko)

**Mocne:**
- Warstwa siedzi na prawdziwym Store+Atom+HRR (nie mock).
- Jawne baseline’y: SYMBOLIC (słaby), SYMBOLIC_ONEHOT (twardy), HOLOGRAM.
- Ablacje: `blind` vs oracle, Projector dwuosiowy, reach-GC fail-loud.
- Testy nazywają hipotezy (nie tylko happy-path).

**Słabe / higiena:**
- Raport koncepcyjny poza gitem i ze starą listą TODO.
- Komentarz w `glyph_features_from_unicode` nadal mówi „brakujący komponent” — historyczny, mylący.
- Duplikat `hologram-znaki-raport (1).md`.
- Tor Holo-Φ vs lingua łatwo pomylić w `doswiadczenie/glify/`.

---

## 6. Werdykt postępu

```
Warstwa reprezentacji znaków (lingua):     ████████████░░░░  ~85%  (API+testy DONE)
Eksperymenty retrieval / dekoder:          ██████████████░░  ~90%
Most / cel pierwotny (mniej epok):         ████████████░░░░  ~75%  (Kana P1 DONE; wynik negatywny / podłoga)
Dokumentacja kanoniczna:                   ██████░░░░░░░░░░  ~55%  (raport + audyt odświeżone)
```

**Jednym zdaniem:** warstwa holograficznego kodowania znaków jest **domknięta i zielona w testach**; cel pierwotny (zysk treningowy Mostu) na Kana P1 **nie potwierdzony** — 0/5 PASS, r@1 przy podłodze dla hologram i symbolic.

---

## 7. Rekomendowane następne kroki

1. Zinterpretować Kana P1: diagnostyka **dlaczego r@1≈0** (za mało epok / zły pair_mode / graf nie niesie recall edges?) zanim wnioskować „hologram nie działa dla Mostu”.
2. Ewentualne **Kana P1b**: więcej epok / n_train, albo inna metryka niż gate ΔSM=0.02 przy podłodze.
3. Scommitować raport + `KANA_P1.md` (bez duplikatu `(1).md`).
