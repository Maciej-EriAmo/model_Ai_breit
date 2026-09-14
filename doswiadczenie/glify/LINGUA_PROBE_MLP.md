# Lingua MLP probe — trenowany klasyfikator vs analityczny HRR unbind

Seedy: [0, 1, 17, 23, 42] · epochs=60 · noise=0.08 · n/klasę train=80 val=20

Kontrast z `run_lingua_p1.py` (Most vs Softmax na CAŁYM grafie Φ) i z analitycznym `unbind_position`+`nearest_morpheme` w `_run_r1`/`_run_onehot_baseline` (karmazyn_lingua.py): tu zwykły MLP (cross-entropy, bez żadnej wiedzy o HRR/pozycji) uczy się na surowym złożonym wektorze słowa, czy odróżni prefiks — pytanie: czy TRENOWANY dekoder robi lepiej niż analityczny najbliższy-sąsiad tam, gdzie ten drugi FAILuje (hologram, fazy stłoczone).

## Wynik vs referencja analityczna

| warunek | mean acc (MLP) | min | max | powyżej przypadku (1/3) | r@1 analityczny |
|---|---:|---:|---:|---:|---:|
| symbolic | 0.353 | 0.317 | 0.383 | +0.020 | 0.333 |
| hologram_close | 0.720 | 0.650 | 0.783 | +0.387 | 0.333 |
| hologram_far | 1.000 | 1.000 | 1.000 | +0.667 | 1.000 |
| onehot | 1.000 | 1.000 | 1.000 | +0.667 | 1.000 |

## Pełna tabela

| warunek | seed | acc |
|---|---:|---:|
| symbolic | 0 | 0.383 |
| symbolic | 1 | 0.317 |
| symbolic | 17 | 0.383 |
| symbolic | 23 | 0.333 |
| symbolic | 42 | 0.350 |
| hologram_close | 0 | 0.733 |
| hologram_close | 1 | 0.750 |
| hologram_close | 17 | 0.783 |
| hologram_close | 23 | 0.683 |
| hologram_close | 42 | 0.650 |
| hologram_far | 0 | 1.000 |
| hologram_far | 1 | 1.000 |
| hologram_far | 17 | 1.000 |
| hologram_far | 23 | 1.000 |
| hologram_far | 42 | 1.000 |
| onehot | 0 | 1.000 |
| onehot | 1 | 1.000 |
| onehot | 17 | 1.000 |
| onehot | 23 | 1.000 |
| onehot | 42 | 1.000 |

## Wniosek

**Analityczny dekoder (unbind_position + nearest-neighbour po cosine
similarity) nie jest optymalny — zaniża, ile informacji faktycznie niesie
hologram.** Dla `hologram_close` (fazy 30°/45°/60°, ten sam warunek, w
którym `_run_r1` w `karmazyn_lingua.py` dostaje r@1=0.33, poziom
przypadku) trenowany MLP osiąga **0.72** — ponad dwa razy więcej niż
przypadek i wielokrotnie więcej niż metoda analityczna. To NIE jest ta
sama sytuacja co `symbolic` (mean=0.353 ≈ 1/3, brak sygnału ponad
przypadek dla ŻADNEGO dekodera, bo tam trzy prefiksy to dosłownie ten sam
wektor — sufit informacyjny, nie problem dekodowania).

Rozróżnienie, które to ujawnia:
- `symbolic`: informacja **zniszczona** na etapie kodowania (Im=0) — żaden
  dekoder, trenowany czy nie, nie może jej odzyskać.
- `hologram_close`: informacja **obecna**, ale nearest-neighbour do
  pojedynczego prototypu morfemu jest złym dekoderem w obecności szumu
  bind/bundle z rdzenia — MLP znajduje granicę decyzyjną odporną na ten
  konkretny wzorzec szumu, prosty NN jej nie znajduje.

Poprawka do wniosku z nagłówka `karmazyn_lingua.py` ("separacja faz jest
za mała względem szumu z bind/bundle" dla `hologram_close`): to zdanie
myli "trudne dla tego konkretnego dekodera" z "niedostępne w zasadzie".
Sygnał JEST dostępny (72% >> 33%) — po prostu `nearest_morpheme()` go nie
wydobywa. Otwarte pytanie na przyszłość: czy Most (graf, nie ten
pojedynczy probe) też by to wydobył lepiej niż wynikało z Lingua P1 — tam
confound małego grafu (N=11) uniemożliwił odpowiedź.

JSON: `lingua_probe_results.json`

