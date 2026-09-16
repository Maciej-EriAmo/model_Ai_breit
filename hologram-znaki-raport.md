# Hologram jako nośnik kodowania znaków pisma — raport z eksperymentu

Substrat: [[karmazynos|KarmazynOS]] (`karmazyn_atom.py`, `karmazyn_substrate.py`,
`karmazyn_hrr.py`). Kod: `karmazyn_lingua.py`.

---

## 1. Koncepcja wyjściowa

Metoda kodowania znaków wszystkich pism świata przy użyciu hologramu o
określonych parametrach jako nośnika. Nakładanie znaków ma pozwalać
tworzyć litery narodowe alfabetu łacińskiego, arabskiego, cyrylicy, lub
formy ewolucyjne pisma japońskiego, chińskiego, koreańskiego.

Rozszerzenie: tworzenie chunków językowych na zasadzie sylabariuszy,
umożliwiające konstrukcje gramatyczne — partykuły w językach słowiańskich,
przedrostki, zaimki.

**Cel pierwotny:** nauka modeli AI przy mniejszej liczbie przebiegów
treningowych — oszczędność czasu i zużycia energii.

**Status celu pierwotnego (stan 2026-09-15):** **Kana P1 uruchomione i
zapisane** (`doswiadczenie/glify/KANA_P1.md`). Werdykt: hologram **0/5
PASS**, symbolic **0/5 PASS**; mean Most r@1 ≈ 0.005–0.008 (podłoga),
Most ≉ przewaga nad Softmax (bramka ΔSM≥0.02). Cel „mniej epok dzięki
hologramowi” — **niepotwierdzony** pod tym protokołem; caveat: absolutne
r@1 przy podłodze sugeruje też underfit zadania. Audyt:
`doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md`.

---

## 2. Decyzja architektoniczna: n+1

W koncepcji krzyżują się dwa różne zjawiska językowe, wymagające dwóch
różnych struktur:

- **cechy paradygmatyczne** (diakrytyk, przypadek, aspekt) — kompozycja
  płaska, na jednym poziomie → **overlay na atomie**
- **kompozycja syntagmatyczna** (przedrostki, partykuły, fraza) —
  kompozycja rekurencyjna, wielopoziomowa → **hierarchia n+1**:
  atom (morfem) → Bubble (słowo) → Bubble jako atom wyższego rzędu →
  Bubble (fraza/klauzula)

Hangul jest fizycznym dowodem słuszności wariantu hierarchicznego: blok
sylabowy to dosłownie Bubble zawierający atomy-jamo, a słowo to sekwencja
takich bloków.

---

## 3. Weryfikacja na realnym kodzie substratu

Po otrzymaniu prawdziwych plików (`karmazyn_atom.py`,
`karmazyn_substrate.py`, `karmazyn_hrr.py`) okazało się, że wcześniejsze
założenia wymagały korekty:

- **`Store` NIE MA `bind()`.** HRR w `karmazyn_substrate.py` to wyłącznie
  `atom_vector()` (generuje wektor Z NAZWY przez `name_to_vector`,
  cache'owany, nienadpisywany jeśli ustawi się `atom.vector` ręcznie) i
  `resonance()` (top-k po podobieństwie w całym magazynie).
- **`Bubble` to czysty scope/GC.** Rodzic + nazwane wiązania,
  `create_bubble(label, atom_ids, root=True/False)` do grupowania i
  ochrony przed reach-GC — nie liczy żadnego wektora.
- **`karmazyn_hrr.py`** dostarczył prawdziwą matematykę: `bind`/`unbind`
  (splot kołowy przez FFT), `bundle` (suma), `similarity` (cosine),
  `name_to_vector`, oraz `HRROperations` z `P_power(n)` do kodowania
  pozycyjnego (`P^n` — wiązanie pozycji `n` w sekwencji).

**Ostateczna architektura:**
- **ATOM** = morfem/glif LUB już złożone słowo/fraza (nierozróżnialne na
  poziomie substratu — obie to `Atom`, stąd naturalna rekurencja n+1).
  Wektor `D=2048` = `[overlay cech, FEATURE_DIM=64]` +
  `[content-wektor z nazwy, generowany przez Store.atom_vector, reszta
  wymiarów]`.
- **BUBBLE** = grupowanie/ochrona GC (`Store.create_bubble`) — bez
  wektorów.
- **`compose()`** = krok n+1: `bind(P^i, cechy_i)` dla `i=1,2,...`,
  `bundle` (suma), rejestracja wyniku jako nowy atom.

---

## 4. Dwie poprawki znalezione przez testy (nie przez czytanie kodu)

### 4.1 Pozycja 0 jest zdegenerowana

`P^0` (z `HRROperations.P_power`) to wektor delta — element
IDENTYCZNOŚCIOWY splotu: `bind(P^0, x) = x`. Sprawdzone empirycznie:

```
unbind(bundle(bind(P^0,pref), bind(P^1,root)), P^0) == cała suma BEZ ZMIAN
(np.allclose = True)
```

`unbind()` na pozycji 0 nic nie separuje — oddaje bundle z powrotem.
**Naprawa:** `compose()` numeruje pozycje od 1, nigdy od 0.

### 4.2 Overlay 1D jest ślepy na fazę pod cosine similarity

Pierwsza wersja kodowała cechę jako pojedynczą liczbę rzeczywistą
`A·cos(φ)` w jednym slocie. Cosine similarity jest niezmiennicza względem
skali — dwa warianty różniące się TYLKO długością wzdłuż tej samej osi
(fazy w tej samej ćwiartce 0°-90°) mają similarity=1.0 między sobą.
Sprawdzone empirycznie: r@1=0.33 (poziom przypadku) NAWET dla trybu
hologram, dopóki slot miał 1 wymiar.

**Naprawa:** każdy slot cechy = 2 wymiary rzeczywiste (Re, Im) zamiast 1
— faza staje się kierunkiem w płaszczyźnie, nie tylko długością na jednej
osi. `FEATURE_DIM=64` → `N_SLOTS=32` kanałów cech.

---

## 5. Wyniki r@1: symbolic vs hologram (na-/prze-/wy-pisać)

| tryb | separacja faz | r@1 |
|---|---|---|
| symbolic (Im=0 zawsze) | dowolna | 0.333 (zawsze przypadek) |
| hologram | 30°-60° (stłoczone) | 0.333 (przypadek) |
| hologram | 0°/120°/240° (rozstawione) | 1.000 |

**Wniosek metodologiczny:** "reprezentacja MOŻE nieść informację o
fazie" i "reprezentacja FAKTYCZNIE poprawia retrieval" to dwa różne
twierdzenia. Drugie zależy od separacji wartości względem szumu
bind/bundle i wymiarowości — nie jest automatyczne. Ten sam typ pytania,
na które projekt [[holo-phi|Holo-Φ]] odpowiedział "nie" w gate P3/P4a; tu
odpowiedź brzmi "tak, ale warunkowo".

---

## 6. Metafora optyczna: płótno, obserwator, światło, projektor

Metafora hologramu optycznego okazała się generatywna — wymusiła pytania
kontrolne, które same znalazły błędy z sekcji 4, zanim jeszcze
uruchomiliśmy kod:

- **Płótno** = `atom.vector` (fragment `FEATURE_DIM`), wynik `bundle()`
- **Światło** = `P^i` użyte w `bind()` przy zapisie (`compose()`)
- **Projektor** = to samo `P^i`, użyte w `unbind()` przy odczycie —
  świeci na płótno tym samym wzorcem
- **Obserwator** = `unbind()` + `nearest_morpheme()` — porównanie
  odzyskanego sygnału do listy kandydatów

Uczciwie: światło i projektor to w naszym kodzie dosłownie ten sam
obiekt matematyczny w dwóch momentach — to nie naciągnięcie, większość
prawdziwych hologramów też rekonstruuje tą samą wiązką co przy zapisie.

**Miejsce, w którym metafora się psuje, wskazało lukę:** prawdziwy
obserwator optyczny jest bierny i nie musi niczego wiedzieć z góry. Nasz
musi znać klucz (`position`) i listę kandydatów — nie "patrzy i widzi",
tylko "dopasowuje do już zarejestrowanego". Brakujący komponent: **projektor
skanujący** (próbujący wielu kluczy/ról zamiast biernie przyjmować
pozycję z góry) — w raporcie pierwotnym zaproponowany; **zaimplementowany
później** jako `Projector.scan` / `best` (commit `58c5a91`, testy
`test_projector_*`).

---

## 7. Poruszający się obserwator: `observe_with_confidence`

Rozszerzenie metafory: obserwator porusza się (paralaksa), światło stoi.
To test STABILNOŚCI dopasowania, nie tożsamości — prawdziwe dopasowanie
trzyma kształt pod drobnym poruszeniem punktu widzenia, artefakt się
rozjeżdża.

**Mechanizm:** dokłada się `n_jitter` razy mały szum do odzyskanego
wektora, za każdym razem robi się `nearest_morpheme`, i głosuje.

**Błąd nr 3, znaleziony przez uruchomienie testu na samym teście:**
pierwsza wersja używała stałego, bezwzględnego `jitter_sigma=0.05`.
Wynik: confidence=0.9–1.0 NAWET dla błędnych dopasowań (przypadek "fazy
blisko", gdzie r@1=0.33) — szum był za mały wobec marginesu decyzji, więc
"ruch" nigdy nie miał szans niczego przewrócić. Metryka kłamała
pewnością.

**Naprawa:** szum skalowany WZGLĘDEM normy wektora (`jitter_fraction`
jako ułamek, domyślnie 0.5) zamiast liczby bezwzględnej. Wynik po
naprawie:

| przypadek | confidence (3 słowa) |
|---|---|
| fazy blisko (r@1=0.33) | 0.72 / 0.40 / 0.98 — niepewny, zmienny |
| fazy rozstawione (r@1=1.00) | 1.00 / 1.00 / 1.00 — stabilny |

Obserwator sam wykrywa niepewność, BEZ znajomości prawdy (ground truth),
z samej stabilności pod jitterem.

---

## 8. Test regresyjny (28/28)

Pełny test na PRAWDZIWYM stosie (Store+Atom+HRR, zero dublerów):
kontrakt `AtomStore` (`karmazyn_atomstore.conforms`), typy (prawdziwy
`karmazyn_atom.Atom`), get-or-create, guardy (slot poza `N_SLOTS`,
`dim<FEATURE_DIM`, pusta lista w `compose()`, `position=0`, puste
`candidates`), rekurencja n+1 z poprawnym grupowaniem w Bubble, r@1
dokładnie zgodne z sekcją 5, kalibracja obserwatora (średnia
"blisko" < średnia "daleko"), `store.stats()`.

**Wynik: 28/28 PASS.**

Dodatkowo sprawdzona ścieżka fallback bez `karmazyn_hrr.py` (plik
tymczasowo ukryty, uruchomione, przywrócony): `compose()` łagodnie
degraduje do zwykłej sumy; `unbind_position`/`nearest_morpheme`/
`observe_with_confidence` jawnie odmawiają (`NotImplementedError`)
zamiast po cichu zwracać coś bezsensownego.

---

## 9. Testy na danych syntetycznych

### A) Sweep pojemności (fazy równomiernie rozstawione, przez `compose()`)

| N kandydatów na wspólnym slocie | r@1 |
|---|---|
| 3–10 | 1.000 |
| 12 | 0.667 |
| 16 | 0.500 |
| 20 | 0.350 |
| 24 | 0.292 |
| 32 | 0.219 |

Twardy próg między N=10 a N=12 przy `FEATURE_DIM=64`. **Nie** ten sam
limit co "D/8" z docstringu `karmazyn_hrr.py` (to dotyczy bundlowania
niezależnych wektorów pełnowymiarowych) — to nowa, osobna liczba dla
naszego konkretnego mechanizmu (N punktów na jednym okręgu 2D,
degradowanych szumem ze splotu na całej 64-wymiarowej przestrzeni).

### B) Losowe (nie wyidealizowane) fazy — N=5, 30 prób

Średnia r@1 = 0.787, std = 0.163, zakres 0.400–1.000.

**Wniosek praktyczny:** losowy przydział fazy jest niestabilny — czasem
idealny, czasem gorszy niż baza. Fazy wymagają ŚWIADOMEJ alokacji (np.
`2π/N` rozstawienie, jak konstelacja PSK w telekomunikacji), nie
przypadkowych wartości.

### C) Wpływ długości sekwencji na retrieval pozycji 1

| długość sekwencji | sim(top1) | top1 poprawny |
|---|---|---|
| 1 | 0.739 | tak |
| 2 | 0.373 | tak |
| 4 | 0.115 | tak |
| 9 | 0.107 | tak |
| 13 | 0.101 | tak |

Sygnał maleje zgodnie z teorią (szum bundlowania rośnie z liczbą
składników), ale top-1 zostaje poprawny na całym przetestowanym zakresie
— z zastrzeżeniem, że przy długości 9-13 wartość zbliża się do poziomu
szumu (~0.1); `observe_with_confidence` przy takich długościach
NIEPRZETESTOWANE.

---

## 10. Test na piśmie rzeczywistym: hiragana

### 10.1 Wiersz は (5 znaków bazowych × 3 warianty = 15 znaków)

Pierwszy test na prawdziwym piśmie, nie syntetycznych polskich
morfemach. Każdy znak = JEDEN atom z dwiema nałożonymi cechami na
rozłącznych slotach (baza — slot 0, 5 wartości; diakrytyk — slot 1, 3
wartości) — płaski overlay, bez `compose()`/bind/bundle.

**Wynik: 15/15 na obu osiach.** は→ば→ぱ, ひ→び→ぴ, ふ→ぶ→ぷ, へ→べ→ぺ,
ほ→ぼ→ぽ — wszystkie poprawnie rozpoznane po bazie i po diakrytyku.

### 10.2 Pełny standardowy zestaw — 71 znaków

46 gojūon + 20 dakuten + 5 handakuten. Trzy cechy nałożone na jednym
atomie: rząd spółgłoskowy (slot 0, **11 wartości**: a,k,s,t,n,h,m,y,r,w,N),
samogłoska (slot 1, 6 wartości), diakrytyk (slot 2, 3 wartości).

**Wynik: 71/71 na WSZYSTKICH TRZECH osiach naraz.**

| oś | liczba wartości | r@1 |
|---|---|---|
| rząd spółgłoskowy | 11 | 71/71 = 1.000 |
| samogłoska | 6 | 71/71 = 1.000 |
| diakrytyk | 3 | 71/71 = 1.000 |

**Kluczowe odkrycie, rozstrzygające pytanie otwarte z sekcji 9A:** rząd
spółgłoskowy ma N=11 — WIĘCEJ niż N=12, gdzie sweep pojemności (sekcja
9A, przez `compose()`) pokazał już wyraźne załamanie do r@1=0.667. Tu
N=11 działa idealnie.

**Wyjaśnienie:** to dwa różne mechanizmy. Sweep 9A testował
sekwencyjne `bind`/`bundle`/`unbind` (szum z konwolucji z innymi
związanymi składnikami). Test hiragany to płaski overlay — cechy na
rozłącznych slotach w jednym atomie, bez żadnego bind/unbind. Przy
cosine similarity wobec czystych wzorców jednoosiowych sygnał z innych
slotów w ogóle nie wchodzi do iloczynu skalarnego (dowód: template ma
zera wszędzie poza swoim slotem, więc iloczyn skalarny zależy tylko od
tego slotu; wspólny mianownik `|character|` skaluje wszystkie
porównania jednakowo, nie zmienia rankingu). **Limit pojemności z 9A
jest właściwością SEKWENCJI (`compose`), nie właściwością samego
NAKŁADANIA cech (overlay).** Dwa różne mechanizmy, dwie różne granice.

To najsilniejsze dotychczasowe potwierdzenie koncepcji z sekcji 1: sam
mechanizm nakładania znaków skaluje się czysto na pełnym, prawdziwym
systemie pisma.

### 10.3 Przygotowanie danych — cztery różne źródła

Warte rozdzielenia, bo mieszają się w jednym teście:

1. **Przypisanie znak↔wiersz↔samogłoska↔diakrytyk** (np. ぎ = rząd k +
   samogłoska i + dakuten) — z wiedzy ugruntowanej, nie z wyszukiwania
   (jak alfabet łaciński — nie jest to fakt zmienny czy sporny).
2. **Liczba znaków w zestawie** — zweryfikowana wyszukiwaniem, bo to
   konkretna teza liczbowa. Wynik: **71** (bez yōon, same monografy) i
   **104** (46+25+33, ze standardowym liczeniem yōon) to OBA poprawne,
   różny zakres definicji, nie sprzeczność. Źródła podają też warianty
   z 36 yōon (107 łącznie) — drobna wariancja konwencji w zależności od
   tego, czy liczy się rzadkie ぢゃ/ぢゅ/ぢょ.
3. **Wartości fazy** (`2π·i/N` na każdej osi) — całkowicie sztuczne,
   zero związku z fonetyką czy historią pisma. Czysta decyzja
   inżynieryjna: rozstaw N wartości równomiernie po okręgu (lekcja z
   testu 9B). Kolejność w liście determinuje fazę — inna kolejność, inna
   faza, identyczny wynik testu.
4. **Amplitude=0.8** — arbitralna stała dla spójności z wcześniejszymi
   testami, też bez znaczenia fonetycznego.

---

## 11. Wnioski ogólne

### 11.1 Metodologiczne

- **Metafora jest generatywna, nie dowodowa.** Pytania "co widzi
  obserwator", "czym świeci projektor" wygenerowały pytania kontrolne,
  które same znalazły błędy z sekcji 4 i 7 — żaden z nich nie wyszedłby
  z samej lektury kodu. Ale metafora niczego nie dowodzi sama z siebie;
  dopiero test na prawdziwym kodzie rozstrzyga, czy przeczucie się
  utrzymuje. Rytm: metafora → hipoteza → test na prawdziwym
  Store+HRR → wynik. Ten sam rytm co gate'y P3/P4a w [[holo-phi|Holo-Φ]].
- **"Może nieść informację" ≠ "faktycznie poprawia wynik".** Powtórzone
  w sekcji 5 (faza) i sekcji 10.2 (skala) — druga teza wymaga
  spełnionych warunków (separacja, wymiarowość, właściwy mechanizm
  kompozycji), nigdy nie jest darmowa.
- **Samo narzędzie testujące trzeba przetestować.** `observe_with_confidence`
  w pierwszej wersji dawało fałszywą pewność — błąd znaleziony dopiero
  przez uruchomienie testu NA teście.
- **Nawyk weryfikacji ponad zaufanie.** Nie przyjęto założeń o `Store`
  bez prawdziwego kodu; nie przyjęto liczby "104" ani wcześniejszego
  "71" bez sprawdzenia. Każdy z trzech głównych błędów w tym raporcie
  (sekcja 4.1, 4.2, 7) wymagał odpalenia kodu — żaden nie był widoczny
  z samej inspekcji.

### 11.2 Techniczne

- Dwa niezależne mechanizmy kompozycji, dwie różne granice pojemności:
  **overlay płaski** (rozłączne sloty w jednym atomie) skaluje się
  czysto — potwierdzone na 71 prawdziwych znakach; **sekwencja przez
  bind/bundle/unbind** ma twardy próg ok. N=10-11 kandydatów na
  wspólnym slocie przy `FEATURE_DIM=64`.
- Faza jako nośnik informacji wymaga: (a) reprezentacji 2D (Re,Im), nie
  1D; (b) pozycji ≥1 w kompozycji sekwencyjnej (nigdy P^0); (c)
  świadomej, rozstawionej alokacji wartości, nie przypadkowej.

### 11.3 Status otwartych punktów (aktualizacja 2026-09-15)

Lista z pierwotnego raportu — **6/6 TODO warstwy zamknięte w kodzie**.
Regresja: `pytest test_karmazyn_lingua.py` → **35 passed** (audyt
2026-09-15). Szczegóły: `doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md`.

| Item | Stan | Gdzie |
|------|------|-------|
| `SYMBOLIC_ONEHOT` | **DONE** | `AtomEncoding.SYMBOLIC_ONEHOT`; `f221e4b` |
| Projektor skanujący | **DONE** | `Projector.scan`/`best`; `58c5a91` |
| Ekstraktor cech (nowy znak) | **DONE** | `glyph_features_from_unicode` + `PhaseRegistry`; `248f186` |
| `observe_with_confidence` (kalibracja) | **DONE** | jitter względny; `8dc1c1e` |
| Yōon sekwencyjne `compose()` | **DONE** | 33 komb.; `1806ee1` |
| Trwałość atomów (reach-GC / tick) | **DONE** | + fail-loud po GC; `1806ee1`, `1ccba2e` |

**Nadal otwarte (nauka / skala, nie braki API):**

- **Cel pierwotny (sekcja 1)** — **Kana P1 DONE, wynik negatywny**
  (0/5 PASS hologram i symbolic; r@1≈podłoga). Lingua P1 był za mały;
  Kana P1 (286 atomów) też nie dał Mostowi ΔSM≥0.02. Otwarte: czy to
  limit reprezentacji, czy underfit protokołu (epochs=16 / n_train=80).
- `observe_with_confidence` przy sekwencjach o niskim marginesie
  (długość 9–13, sim≈0.1) — nadal nieosobno przetestowane.
- Skala powyżej kana (kanji, >`FEATURE_DIM` wartości na jednej osi) —
  świadomie poza zakresem Kana P1.

---

## 12. Pliki

- `karmazyn_lingua.py` — warstwa lingwistyczna (`LinguaLayer`,
  `FeatureOverlay`, `AtomEncoding` w tym `SYMBOLIC_ONEHOT`, `compose`,
  `unbind_position` oracle/blind, `nearest_morpheme`,
  `observe_with_confidence`, `Projector`, `glyph_features_from_unicode`),
  zweryfikowana na prawdziwym `Store`+`Atom`+`HRR`.
- `test_karmazyn_lingua.py` — regresja **35** testów (stan 2026-09-15).
- `kana_corpus.py` / `make_dump_kana.py` / `run_kana_p1.py` — harness
  celu pierwotnego (trening Most vs Softmax na 71 tożsamościach × 3 pisma).
- `doswiadczenie/glify/AUDYT_HOLOGRAM_ZNAKI.md` — audyt postępu.
- `doswiadczenie/glify/LINGUA_P1.md`, `LINGUA_PROBE_MLP.md` — eksperymenty
  Most / dekoder MLP.
- Historyczne testy z toku raportu (sweep pojemności, fazy, は-row, 71
  znaków) — treść w §§9–10; regresja w pliku testowym powyżej.
