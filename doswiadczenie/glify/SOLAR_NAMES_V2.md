# Solar Names v2 — τ z fizyki orbitalnej, korpus 18x większy niż P5.1

**Data:** 2026-09-13
**Pomysł użytkownika:** nazwy ciał Układu Słonecznego (+ Słońce) i ich księżyców,
we wszystkich językach, z ich powiązaniami.
**Kod:** `solar_names_tracer.py`, `solar_names_data.json`, `make_dump_solar_v2.py`,
`bench_solar_v2.py`

---

## Dane (prawdziwe, zweryfikowane — Wikidata, nie zgadywane)

- **33 ciała**: Słońce + 8 planet + Księżyc + 20 innych księżyców (Phobos/Deimos,
  4 galileańskie, 7 księżyców Saturna, 5 księżyców Urana, Tryton/Nereida) +
  Pluton + Charon + Ceres. QID-y zweryfikowane opisem encji przed użyciem
  (np. `Io (Q3123): "innermost of the 4 Galilean moons of Jupiter"`).
- **55 języków** — przecięcie zbiorów etykiet obecnych u WSZYSTKICH 33 ciał
  (od dużych: en/fr/de/es/ru/zh/ja/ar/hi, po rzadsze: an/azb/kv/nan/wuu/war).
- **τ = log10(semi-major axis w km)** — `P2233` z Wikidata, jednostki
  ujednolicone (część encji miała AU, przeliczone × 149597870.7). Słońce
  (brak własnej orbity) dostaje log-dystans=0 jako umowne centrum.
- **1815 atomów** (33×55) — 18× więcej niż P5.1 (100 atomów, 5 radykałów).

## Setup (zastosowane od razu lekcje z P5.1)

- Family/task = **ciało** (np. "Mars"); instancja = nazwa w jednym z 55 języków.
- **Mieszane grafy treningowe od pierwszego batcha** (8 losowych ciał
  treningowych naraz, prawdziwe negatywy cross-body) — nie osobno-per-ciało
  jak błędny `build_train_batch` w P5.1.
- **`t_seal=False`** od razu — log-dystans orbitalny nie ma semantyki
  HOT/WARM/COLD/TOMB, więc seal nie powinien go interpretować jako "archiwum"
  (lekcja z audytu seala w n-body/Faza B).
- Holdout: **7 ciał całkowicie nieobecnych w treningu** — Neptun+Tryton+Nereida
  (cały zewnętrzny układ), Ceres, Deimos, Mimas, Charon (rozstrzelone po
  typach: planeta/księżyc/planeta karłowata, blisko/daleko).
- Determinizm zweryfikowany bit-do-bitu przed sweepem (`test_solar_v2_determinism.py`).
- Split bez przecieku zweryfikowany (`test_solar_v2_split.py`).

## Wynik (8 seedów, epochs=60)

| seed | Most | SM | ZERO | ΔZERO | ΔSM | train_sanity | gate |
|-----:|-----:|---:|-----:|------:|----:|--------------:|:----:|
| 0 | 0.020 | 0.000 | 0.010 | +0.0100 | +0.0200 | 0.500 | FAIL |
| 1 | 0.020 | 0.020 | 0.020 | +0.0000 | +0.0000 | 0.500 | FAIL |
| 2 | 0.020 | 0.000 | 0.020 | +0.0000 | +0.0200 | 0.750 | FAIL |
| 7 | 0.010 | 0.000 | 0.000 | +0.0100 | +0.0100 | 0.625 | FAIL |
| 11 | 0.010 | 0.020 | 0.020 | −0.0100 | −0.0100 | 0.375 | FAIL |
| 17 | 0.000 | 0.010 | 0.010 | −0.0100 | −0.0100 | 0.375 | FAIL |
| 23 | 0.020 | 0.000 | 0.020 | +0.0000 | +0.0200 | 0.750 | FAIL |
| 42 | 0.030 | 0.010 | 0.020 | +0.0100 | +0.0200 | 0.500 | FAIL |

**FAIL: 0/8** · mean ΔZERO=**+0.0012** · mean ΔSM=**+0.0088** — w granicach
szumu (krok ewaluacji = 1/100 = 0.01). JSON: `solar_v2_results.json`.

## Różnica względem P5.1 — co skala naprawiła, a czego nie

**Naprawiła:** `train_sanity` (recall in-distribution) zatrzymuje się na
**37.5–75%** po 60 epokach, NIE dochodzi do 100% jak w P5.1 (gdzie osiągał
pełne zapamiętywanie już w epoce ~45). 18x większy korpus (1815 vs 100
atomów, przy tej samej pojemności modelu) realnie utrudnia czyste
zapamiętywanie tożsamości — dokładnie zgodnie z hipotezą "skala pokonuje
przeuczenie" postawioną po P5.1.

**Nie naprawiła:** mimo to **holdout nadal płaski** i **identyczny dla
Most/Softmax/ZERO** — żaden z trzech torów nie wykazuje przewagi. Ciekawostka
z diagnostyki jednego seeda: holdout r@1 był ~0.02 już przy epoce 0 (przed
treningiem), czyli ~16× powyżej czystej szansy (1/824≈0.0012) — sugeruje to,
że SAMA architektura (LayerNorm, struktura tracera) ma jakąś wbudowaną,
przed-treningową skłonność do grupowania po τ, ale trening jej nie pogłębia
ani nie różnicuje między Most a Softmax.

## Wniosek

1. Skalowanie korpusu (18×, prawdziwe dane) **poprawiło jakość eksperymentu**
   (mniej czystego zapamiętywania identity), ale **nie ujawniło przewagi
   Most nad Softmax** na holdout — wynik jakościowo ten sam co P5.1.
2. To zawęża hipotezy: problem nie jest (wyłącznie) skalą danych. Zostają:
   (a) sama architektura/mechanizm nie daje mierzalnej przewagi na tym
   RODZAJU zadania (recall nazwy przez inny język tej samej rzeczy) niezależnie
   od skali, (b) `content = hash(nazwa)` nadal nie niesie żadnej struktury
   współdzielonej MIĘDZY ciałami (w przeciwieństwie do np. n-body, gdzie
   content i tracer razem opisywały tę samą fizyczną scenę) — tu jedyny
   związek ciała z jego τ jest "z zewnątrz", nigdy nie widoczny w treści,
   (c) 60 epok / bieżąca pojemność modelu może wciąż być za mało dla w pełni
   nasyconego porównania (train_sanity nie osiągnął plateau, tylko 37-75%).
3. Faza C (`bench_recall.py`) nadal jest jedynym miejscem, gdzie mechanizm
   Most demonstrowalnie i przyczynowo wygrywa (+67pkt, ablacja −T konkretnie
   niszczy przewagę) — tam content i tracer pochodzą z tej samej, spójnej,
   syntetycznej struktury Φ-drzewa. To może być kluczowa różnica: recall
   działa, gdy content i τ opisują TĘ SAMĄ rzecz w tym samym schemacie, a nie
   gdy τ jest zewnętrznym faktem "doczepionym" do niezależnie zahaszowanej
   nazwy.

---

## Runda 2 — content = kategoria rodzeństwa (wzór Fazy C), nie hash nazwy

Hipoteza z rundy 1: Faza C działa, bo content=kategoria (topic) robi kilku
kandydatów NIEROZRÓŻNIALNYMI treścią, a τ (depth) jest jedynym sygnałem
rozstrzygającym który jest prawdziwym pozytywem. W P4a/P5.1/Solar-v1
content = unikalny hash KAŻDEJ instancji — zero współdzielonej struktury.

**Redesign:** `content = 0.75·hash(kategoria) + 0.15·hash(język) + 0.10·szum`
(dokładny wzór `recall_data._topic_content`). Kategoria = grupa rodzeństwa
(`moon_of_Jupiter`, `planet`, `dwarf_planet`, ...). Zweryfikowano: cos(Europa,
Io) [ta sama kategoria] = **0.97**, cos(Europa, Mercury) [inna] = **0.14** —
dokładnie odtworzona ambiguacja z Fazy C. Holdout przeprojektowany: 7 ciał
(Neptun, Ceres, Deimos, Europa, Mimas, Umbriel, Nereida), każde z **≥1
rodzeństwem zostającym w treningu** (np. Europa holdout, ale Io/Ganymede/
Callisto w train) — model mógł nauczyć się kategorii z rodzeństwa, musi użyć
τ, żeby rozpoznać KONKRETNIE nowego członka.

**Wynik #1 (batche stałe przez cały trening, jak w rundzie 1):** FAIL 0/8,
ale `train_sanity` wróciło do 0.75–1.0. Sprawdzenie: czy to prawdziwe użycie
τ, czy znowu zapamiętywanie (tym razem zaszumionego "odcisku palca" zamiast
tożsamości wprost)? Test: przebudowa batchy (nowy losowy podzbiór ciał +
nowy szum + nowe query/pos) **co epokę** zamiast raz na cały trening.

**Wynik #2 (świeże batche co epokę):** `train_sanity=0.000 dla WSZYSTKICH 8
seedów` — model w ogóle się nie uczy nawet in-distribution w 60 epokach.
Potwierdza podejrzenie: wysoki `train_sanity` z Wyniku #1 był zapamiętywaniem
zaszumionego wzorca konkretnego (query,pos), nie nauczoną regułą "dopasuj po
τ w kategorii" — bo gdy nie ma czego zapamiętać (świeże batche), model nie
radzi sobie wcale, nie tylko "gorzej".

**Diagnoza napięcia (nierozwiązana w tej sesji):** stałe batche → zbyt łatwo
zapamiętać (fałszywie wysoki train_sanity, zero transferu). Świeże batche co
epokę → zbyt mało powtórzeń tej samej kombinacji, żeby spadek gradientu
złapał regułę (train_sanity=0, wynik nieinformacyjny). Potrzebny środek: N
(np. 20-30) stałych, ale różnorodnych batchy, cyklicznie powtarzanych przez
znacznie więcej epok — niezrealizowane w tej sesji (decyzja: zamknąć i
podsumować zamiast kolejnego ~90-min przebiegu).

## Wniosek końcowy sesji

1. **Silnik Most działa poprawnie i przyczynowo zależy od τ** — potwierdzone
   niezależnie od tego eksperymentu: Faza C (`bench_recall.py`) reprodukuje
   się w pełni na aktualnym kodzie (+67pkt, ZERO≈Softmax, ablacja −T
   konkretnie niszczy przewagę, −mazur nie). To nie jest kwestionowane.
2. **P4a, P5.1, Solar v2 (obie rundy) — wszystkie FAIL**, ale z coraz
   głębiej zrozumianych przyczyn: P4a = τ tautologiczne (hash etykiety);
   P5.1 = brak skali + zbugowany ZERO (oba naprawione, FAIL się utrzymał);
   Solar v2 runda 1 = 18× skali, ale content bez żadnej struktury
   współdzielonej; Solar v2 runda 2 = content ze strukturą (jak Faza C), ale
   ujawniło to inny, nierozwiązany problem metodologiczny (stałe vs świeże
   batche) zamiast dać odpowiedź.
3. **Kluczowa nierozstrzygnięta hipoteza:** czy recall Most>Softmax na
   HOLDOUT (nowe, nigdy niewidziane instancje) jest w ogóle osiągalny dla
   tego typu zadania (symboliczne etykiety + zewnętrzne τ), czy Faza C
   wygrywa tylko dlatego, że jest **in-distribution** (ten sam graf, znane
   topiki) i teza nie przenosi się na prawdziwą generalizację. To pytanie
   pozostaje otwarte.

---

Jednym zdaniem: **przeprojektowanie contentu na wzór Fazy C (kategoria +
szum) było dobrym, uzasadnionym krokiem i dało czystszy diagnostyk (00%
train_sanity ujawniło, że poprzedni wysoki wynik był artefaktem stałych
batchy) — ale nie rozstrzygnęło pytania o holdout, bo napotkało nowy,
osobny problem (stałe vs świeże batche), który wymaga kolejnej iteracji
poza zakresem tej sesji.**
