# JUTRO — nauka pełnej tablicy Mendelejewa (~2–3 h)

**Data zapisu:** 2026-09-15 · komora `Most_Lorentz`  
**Aktualizacja:** jutro **tylko Most** — bez Softmax, bez ZERO, bez porównań armii.

## Cel

Nauczyć wzorzec **całej tablicy (Z=1..118)** na **samym Moście**.
Nie robimy benchy vs Softmax — to już domknięte na mini-tablicy / P1b.

### Rama sukcesu (do CV / notatki)

Domowa konstrukcja **Most** na nie-płaskim nośniku Φ **dorównała** klasycznemu Softmaxowi
(dzieło ustalonej linii badawczej) na tym samym zadaniu recall — P0 Mendelejew:
Most≈0.43 · Softmax≈0.45 · nonflat win vs płaski tekst. To wystarczy jako dowód
„parity z kanonem”; jutro budujemy **kompetencję Mosta na pełnej tablicy**, nie re-proof.

## Wykluczenia jutro

| NIE | TAK |
|-----|-----|
| Softmax | **Most Φ** |
| ZERO / ablacje porównawcze | krzywe loss/r@1 Mosta |
| gate Δ(Most−SM) | above_floor vs chance; jakość rankingu |
| 4 armie × czas | 1 armia → więcej epok/seedów w tym samym budżecie ~2–3 h |

## Dlaczego nucleus+topology (nie pełne orbitale)

Pełne orbitale 1–118 ≈ **~7k atomów** → attention O(N²) nie wejdzie w 2–3 h.
Dump `dumps/from_mendelejew_full`: **118 atomów** + bindy grupa/okres/Z —
wieloosiowy nośnik (group×period×block×Z).

## Komendy (odpal rano)

```bat
cd /d C:\Users\drwis\model Ai
python export_mendelejew_full.py
python run_mendelejew_full.py --most-only
```

Smoke (opcjonalnie):

```bat
python run_mendelejew_full.py --most-only --smoke
```

Wyniki → `doswiadczenie/glify/MENDELEJEW_FULL.md` + `mendelejew_full_results.json`

## Szacunek czasu

`--most-only`: 64 ep × 3 seedy × **1 armia** × N=118 ≈ **~1–1.5 h**  
(albo ten sam budżet 2–3 h → więcej epok / dodatkowy seed — do decyzji przy starcie).

## Kontekst z dziś (nie powtarzać)

| Done dziś | Werdykt |
|-----------|---------|
| Mendelejew P0 (8 el.) | nonflat 3/3; **Most ≈ Softmax** ← sukces parity |
| Hold-out Na | brak generalizacji grupy na ukryty Na |
| Kana/PL P1b | nauka OK; hologram≈symbolic |

## Po FULL (kolejka)

1. Hold-out Na/K na pełnej tablicy (czy wtedy kuzyn działa) — też **Most-only**
2. hologram-E (period×group×block overlay)
3. Qumran — później
