# Transkodowanie alfabetów — kanon produktu (Azjateska)

**Status:** D0–D2 + G2 best-seed PASS · poprawki audytu (glued scripts / photo v2 / Softmax framing)  
**Kod:** `transcode/` · CLI: `glyph_transcribe.py`, `make_dump_glyphs.py`  
**Komora SE:** `Most_Lorentz`  
**Doświadczenie:** [`../doswiadczenie/glify/`](../doswiadczenie/glify/)

---

## 1. Po co (teza komercyjna)

Klasyczne AI przy zmianie pisma zwykle **uczy się od nowa**: nowy OCR, nowy BPE, nowy fine-tune. Koszt = **wielokrotna nauka**.

Tu:

```text
dowolne pismo → ten sam kontrakt Φ → ten sam Most
```

Zmienia się segmentacja, słownik `S` i **która warstwa jest jądrem semantycznym** — nie silnik.

Świadomie **nie** oferujemy: full LLM, magicznego OCR, ani „pewniejszego niż Softmax bez dowodu”.

---

## 2. Gate obowiązkowy: nie być kaleką vs klasyka

**Klasyczna metoda** rozpoznawania / nauki na piśmie:

1. tokenizacja **liter / znaków**,
2. **następstwo** w sekwencji,
3. sieć (tu: Softmax-twin na tej sekwencji).

To jest domyślny, sprawdzony tor. Nowa metoda (atomy Φ + Most + `T`) **musi dorównać** klasycznej na tym samym zadaniu — inaczej jest kaleką z ładnym opisem.

| Tor | Co robi | Rola w benchu |
|-----|---------|----------------|
| **Softmax-twin** | ten sam dump Φ, uwaga bez zamka `ΔE` | **świadek** w `--baselines` |
| **ZERO** | Most z wyzerowanym tracerem | kontrola degeneracji |
| **Most** | Φ + `T` + jądro semantyczne per skrypt | kandydat produktu |

`classical_tokens.json` = **inwentarz** kolejności liter/znaków (czytelność / przyszły tor).  
**Nie** jest dziś wejściem treningu Softmax — Softmax w `phimost_learn --baselines` trenuje na **tym samym** loadere Φ co Most (tylko bez Most-attention). Twierdzenie „Softmax = klasyczna sekwencja liter” wymaga osobnego loadera (TODO); do tego czasu mówimy: Softmax-twin-on-Φ.

**Kryterium uczciwości (produkt):**

- Most **nie może** systematycznie przegrywać z Softmax-twinem na gate bez wyjaśnienia.
- Przewaga Mostu musi pochodzić z geometrii `τ` / jądra jednostek — nie z przecieku etykiet do contentu.
- Dopóki multi-seed parity nie jest stabilna, w komunikacji zewnętrznej: **lab + best-seed**, nie „lepsze niż klasyczne OCR/NN”.

Ablacja ZERO = dowód roli τ; Softmax-twin = dolna granica „czy Most w ogóle coś wnosi na tym dumpie”.

---

## 3. Asymetria jednostek (łacina ≠ CJK)

| Rodzina | Gdzie jest semantyka | Litera/znak |
|---------|----------------------|-------------|
| **Łacina** (alfabety) | na **słowie** (potem gramatyka) | struktura pod słowem |
| **Hiragana / katakana** | zwykle w **znaku** (mora); fraza opcjonalnie | = jednostka |
| **Kanji / hanzi** | w **znaku** (morfem); kreski później | = jednostka |

Litera `M` nie znaczy „MOST”. Znak `東` często już niesie sens. Stary dump G0 wiązał oba skrypty tak samo (`word:*` na znaki) — to **luka**, nie kanon produktu.

### Mapowanie depth (zgodne z tracerem `n_dim=6`)

Tracer: depth one-hot na **3 binach** (0..2) + `T` + state + mazur — **bez** depth=3 (zjadłoby kanał `T`).

| Skrypt | depth 0 | depth 1 | depth 2 (drobne) |
|--------|---------|---------|------------------|
| **Łacina** | strona | **słowo = jądro** | litera |
| **CJK / kana** | strona | linia | **znak = jądro** |

Pola: `unit` ∈ `{page, line, word, letter, char}` · `S` z prefiksem skryptu · `T` ≠ XY (bbox tylko w `E`).

Edges: `next` w warstwie sekwencji; `parent` w górę; `contains` słowo→litery; **bez** HOT-bind lat↔hira.

---

## 4. Pipeline

```text
źródło (tekst / zdjęcie)
  → detect_script + semantic_unit(script)
  → segmentacja (spacje → słowa lat; znaki → jądra CJK)
  → atomy Φ + edges
  → Softmax | Most | ZERO   ← zawsze trzy tory przy claimach
```

API: pakiet `transcode/` (`scripts`, `units`, `phi_build`).

---

## 5. Stan i roadmap

| Etap | Treść | Status |
|------|--------|--------|
| G0/G1 | lab glify (stary kontrakt znak=depth2) | PASS |
| **D0** | ten dokument + kontrakt jednostek | **DONE** |
| **D1** | `transcode/` + dump v2 | **DONE** |
| **D2** | testy kontraktu + framing parity | **DONE** |
| **G2** | Softmax\|Most\|ZERO na dumpie v2 | **PASS best-seed** (seed0 ΔSM=+0.022; 1/3 seedów — jak G0; szczegóły w `doswiadczenie/glify/WYNIKI_GLYPHS_G2.md`) |
| G3+ | OCR-propozycje S, stroke kanji, kolejne alfabety | później |

Wyniki lab: [`../doswiadczenie/glify/`](../doswiadczenie/glify/).  
Notatki badawcze: [`GLYPHS.md`](GLYPHS.md).

---

## 6. Zakazy

- Uczenie z framebuffera Lunety / bitmapy jako kanonu  
- Mieszanie toru DESI/JoannaM z alfabetem  
- Twierdzenie „lepsi niż klasyczna NN” bez tabeli Softmax\|Most\|ZERO  
- Jedna klika `word:*` na całą stronę mieszaną  
- `T` = pozycja XY

---

## 7. Metoda 2: hologramy znaków (Holo-Φ)

**Status:** pierwsza próba labowa **FAIL gate** (2026-09-11) — Most > ZERO, ale Softmax-twin ≥ Most.  
**Założenia w jednym:** [`HOLO_ZALOZENIA.md`](../archiwum/holo/HOLO_ZALOZENIA.md) · wyniki: [`../doswiadczenie/glify/HOLOGRAMY.md`](../archiwum/holo/HOLOGRAMY.md).

- litery / kana / kanji jako **fale** \(A e^{i\varphi}\),
- **słowo** = interferencja Σ,
- **rdzeń** (np. **女**) + pochodne = wspólna faza + bind `holo:root:…`,
- kod: `holo_phi.py` · `make_dump_holo.py` · dump `dumps/from_holo`.

Ten sam Most / seal / Softmax-twin. Hologramy **nie zastąpiły** potrzeby uszczelnienia+skali z samplingiem — dają inny nośnik kompozycji, nie darmowy PASS.
