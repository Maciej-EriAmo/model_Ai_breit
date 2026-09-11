# Holo-Φ — założenia (jeden dokument)

**Nazwa robocza metody:** Holo-Φ (metoda 2 obok transkodu glifów / Azjateska)  
**Status empirii (2026-09-11):** pierwsza próba **FAIL gate** Softmax\|Most\|ZERO (best ΔSM≈+0.003)  
**Kod:** `holo_phi.py` · `make_dump_holo.py`  
**Dump / ckpt:** `dumps/from_holo` · `checkpoints/most_holo.pt`  
**Wyniki lab:** [`../doswiadczenie/glify/HOLOGRAMY.md`](../doswiadczenie/glify/HOLOGRAMY.md)  
**Audyt założeń:** [`ocena_holo.txt`](ocena_holo.txt) — audytor 1 + **audytor 2** (8,5/10); przyjęte  
**Powiązane:** [`TRANSCODE.md`](TRANSCODE.md) §7 · [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md) · `most_seal.py`

> Recenzja **nie** mówi „Holo-Φ jest złe”. Mówi: FAIL jest uczciwy, ale eksperyment **jeszcze nie pozwala powiedzieć dlaczego**. Kolejność napraw: **P0→P4** (§8b).

---

## 1. Po co

Klasyczny tor (tokeny liter + Softmax) i tor glifów (`glyph_unit_v2`) uczą **sekwencji / hierarchii**.  
Holo-Φ dodaje **kompozycję falową**:

1. **słowo** = interferencja liter (nie tylko `next`),
2. **znak pochodny** = fala od **rdzenia** (np. 女) + offset — rodzina w jednej fazie bazowej.

**Cel (hipoteza):** jeden Most; bogatsza kompozycja pisma bez OCR-bitmapy.  
**Nie twierdzimy:** Holo bije Softmax ani że to LLM.

---

## 2. Założenia (rdzeń) — po audycie `ocena_holo.txt`

| # | Założenie | Status / dopisek audytu |
|---|-----------|-------------------------|
| Z1 | Znak = fala \(A\,e^{i\varphi}\) (1 complex) | OK |
| Z2 | **`A` = częstość w korpusie lab** (+ boost rdzenia); **`φ` = hash etykiety** | **P0 DONE** — patrz §2b. Seal na `T` z energii fali = zamierzona bramka „słaba fala / brak fokusu”, nie losowy hash. |
| Z3 | Słowo = \(\sum_k A_k e^{i\varphi_k}\) | OK |
| Z4 | Rdzeń ustala `φ_root`; pochodne = `φ_root + α·φ_off` | OK jako **wstrzyknięta** struktura labowa — nie dowód uczenia (patrz §8) |
| Z5 | Ten sam kontrakt Φ / Most co glify | OK dla mechanizmu; **graf nie jest ceteris paribus** (brak klik `script:*`) — porównanie Holo↔glify tylko z tym zastrzeżeniem |
| Z6 | Softmax-twin = świadek na **tym samym** dumpie Φ | OK — silny punkt metody |
| Z7 | `most_seal` na Most | Bramkuje niskie **T** = niską energię fali / TOMB; nie dostaje już losowego A z hasha |
| Z8 | `T ≠ XY` | OK |

### 2b. Definicje A i T (P0 — kanon)

#### A — amplituda fali (energia nośnika)

| | |
|--|--|
| **Sens** | Jak silnie znak wchodzi w interferencję (= waga w Σ) |
| **Wzór** | \(A = A_{\mathrm{floor}} + (1-A_{\mathrm{floor}})\cdot f(c)/f_{\max}\), \(A_{\mathrm{floor}}=0.35\) |
| **\(f(c)\)** | liczba wystąpień znaku w korpusie lab (słowa lat + hira + rodzina rdzenia) |
| **Boost rdzenia** | jeśli \(c \in\) rodzina 女: \(A \leftarrow \max(A, 0.85)\) |
| **Nie jest** | hashem etykiety; saliency spoza korpusu; pozycją XY |
| **Archive** | poza korpusem → nie liczy A do fali; atom archive ma `A=0` |

#### T — ciepło uwagi (wpuszczenie do Most / seal)

| | |
|--|--|
| **Sens** | Czy klucz wolno wpuścić do okna uwagi (`most_seal` tnie niskie T) |
| **Znak @2** | \(T = \mathrm{energy\_to\_T}(A, ref)\) |
| **Compound @1** | \(T = \mathrm{energy\_to\_T}(\|\Sigma\|, ref)\) |
| **Wspólny `ref`** | \(\max\{A(c)\} \cup \{\|\Sigma\|\}\) na całym dumpie — jedna skala depth 1 i 2 |
| **\(\mathrm{energy\_to\_T}\)** | \(T = \mathrm{clip}(0.25 + 0.7\cdot E/ref,\ T_{\mathrm{TOMB}}, 0.98)\), \(T_{\mathrm{TOMB}}=0.08\) |
| **Fokus (≠ A)** | pierwsza litera słowa: \(T \leftarrow \max(T, 0.90)\) — jawna reguła czytania |
| **Archive** | \(T := T_{\mathrm{TOMB}}\) **stałe** — nie z A; seal ma co odciąć |
| **Nie jest** | pozycją XY; osobnym hashem; niezależnym od energii fali (poza fokusem i archive) |

#### φ — faza (osobno)

Nadal **hash(etykieta)** (+ `φ_root` dla rodziny). Faza buduje klaster / interferencję; **nie** steruje seal bezpośrednio (seal patrzy na T).

#### Łańcuch uwagi (jawny)

```text
freq(c) → A → fala A·e^{iφ}
                ↓
         |Σ| / A  →  T (wspólny ref)  →  most_seal
         fokus pierwszej litery / archive T_TOMB
```

**Świadomie:** seal **widzi** saliency częstościową (przez A→T). To nie jest już sztuczna struktura z hasha — to prior „częstszy / głośniejszy w korpusie lab → cieplejszy”. Semantyka językowa spoza korpusu = przyszła wersja A.

Stan P0: **zdefiniowane i w kodzie** (`holo_phi.py`). Dalej: P1 (seedy / ziarnistość / krzywe).

---

## 3. Wymiary

Holo i glif: **ten sam packing** (`d_x=8`, `n_dim=6`, `d_model=32`).  
Fala = 2 liczby rzeczywiste **przed** Φ; potem wciśnięte w `E` / `T` / `mazur`.

| Kanał | Dim | Holo |
|--------|-----|------|
| content `x` | 8 | hash S⊕E (+ blend_holo z binu fazy) |
| tracer `τ` | 6 | depth OH · **T** · state · **mazur=φ/2π** |
| Most hidden | 32 | po Linear(8→32) |

Więcej dim pod Re/Im osobno = zmiana kanonu → nowy bench.

---

## 4. Mapowanie fala → atom + normalizacja `T`

| Pole | Znak @2 | Compound @1 |
|------|---------|-------------|
| `S` | `holo:ch:…` | `holo:word:…` / `holo:root:…` |
| `E` | podpis re/im/mag | j.w. z n części |
| `T` | z amp | z `\|Σ\|` |
| `mazur` | `φ/2π` | `arg(Σ)/2π` |

**Normalizacja `T` (po audycie pkt 4):** jedna wspólna baza `ref = max(wszystkie amp znaków ∪ \|Σ\| compoundów)` na dumpie — **ta sama skala** dla depth 1 i 2, żeby seal widział porównywalne `T` w całej hierarchii. (Wczesny kod liczył znak względem `ref=1.0` osobno — to niespójność; kanon = wspólny `ref`.)

### Graf

- `contains` / `part_of` / `next` / `next_word` / `holo:root:…`
- **Bez** gęstych klik `script:*` (świadomie) → **ograniczenie porównywalności** z torem glifów (inna topologia, nie tylko inne węzły).

---

## 5. Rdzeń 女 — lab, nie etymologia

Rodzina v0.1: 女 安 姿 数 好 如 — **ręcznie** pod wspólną fazą.  
Ablacja „wyłącz wspólną fazę → klaster znika” **mierzy tylko cofnięcie konstrukcji**, nie naukę modelu (audyt pkt 3).  
Prawdziwy test wymaga sygnału **spoza** wstrzykniętej fazy (np. holdout pokrewieństwa, którego nie kodujemy w `φ`).

---

## 6. Protokół empirii

| Element | Reguła |
|---------|--------|
| Tory | Softmax-twin \| Most+seal \| ZERO |
| Gate ΔSM r@1 | **+0.02** |
| Seeds w raporcie | **wszystkie**; kryterium sukcesu ≥4/5 wymaga **≥5 seedów** w eksperymencie (audyt pkt 1 — pierwsza próba miała tylko 3 → nie rozstrzyga §8) |
| `blend_holo` | pierwsza próba **0.4 bez sweepu** — FAIL może być czuły na hiperparametr; przed pętlą 2: sweep 0.0/0.2/0.4/0.6/0.8 |
| Epoki | 16 — **brak krzywej uczenia** w v0.1; FAIL może być niedotrenowaniem (audyt pkt 7) |
| Skala | 73 atomy — r@1 ziarniste; różnice 0.01–0.02 mogą być szumem metryki (audyt pkt 6) |

### Pierwsza próba (3 seedy — nie spełnia kryterium 4/5)

| seed | Most | SM | ZERO | ΔSM |
|------|------|----|------|-----|
| 0 | 0.145 | 0.155 | 0.128 | −0.009 |
| 1 | 0.130 | 0.152 | 0.125 | −0.022 |
| 42 | 0.155 | 0.152 | 0.127 | +0.003 |

FAIL gate. Most>ZERO; SM≥Most.

---

## 7. Zakazy

1. Bitmapa Lunety / OCR jako kanon.  
2. Mieszanie z DESI/JoannaM.  
3. Claim vs Softmax bez tabeli 3 torów + pełnych seedów.  
4. Zmiana `d_x`/`n_dim` bez bencha.  
5. Traktowanie ablacji 女 v0.1 jako dowodu uczenia.  
6. Porównanie Holo↔glify bez wzmianki o różnicy topologii grafu.

---

## 8. Kryteria sukcesu

Gate docelowy: Most > Softmax na ≥4/5 seedów (ΔSM r@1 ≥ +0.02) — **dopiero po P0–P3**.

### 8b. Kolejność przed pętlą 2 (audytor 2 — obowiązująca)

Nie robić od razu dużego eksperymentu / pełnego gate.

| Priorytet | Treść | Dlaczego |
|-----------|--------|----------|
| **P0 — audyt definicji** | Jednoznacznie: co to `A`, co to `T`; wspólna normalizacja depth; czy `seal` nie dostaje sztucznej struktury z hasha | **Fundamentalne** (audytor 2 stawia pkt 2 wyżej): łańcuch `A→T→seal→COLD` przy hashowym `A` może sterować uwagą bez semantyki |
| **P1 — kontrola eksperymentalna** | 5–10 seedów; krzywe train/val; **ziarnistość r@1** (ile przypadków → minimalny krok metryki → dopiero próg +0.02) | Bez tego FAIL/PASS może być szumem lub niedotrenowaniem |
| **P2 — mini-sweep** | `blend_holo ∈ {0.0, 0.2, 0.4, 0.6, 0.8}` | Czy FAIL to metoda, czy hiperparametr |
| **P3 — ablacja 女** | Dopiero gdy test **nie** jest usunięciem ręcznie zakodowanej odpowiedzi w `φ` | Unik tautologii |
| **P4 — pełny gate** | ≥4/5 seedów, ΔSM≥+0.02 | Dopiero tu: czy Most naprawdę > Softmax |

**Najmocniejsze punkty obu audytów (nie gubić):** seedy 3≠5 · ablacja 女 · blend bez sweepu · brak krzywej loss · **`A`/`T` fundamentalne** · ziarnistość r@1 przy N=73.

**P0–P4a DONE.** Wnioski: [`../doswiadczenie/glify/HOLO_WNIOSKI.md`](../doswiadczenie/glify/HOLO_WNIOSKI.md).  
**P4a FAIL:** 1/8 PASS @ blend=0.6 pre-registered ([`HOLO_P4a.md`](../doswiadczenie/glify/HOLO_P4a.md)) — sygnał P2 nie utrzymał się na 8 seedach.  
Dalej opcjonalnie: **P4b** skalowanie alfabetów (bez udawania lokalnego gate).

---

## 9. Komendy

```bat
cd /d C:\Users\drwis\model Ai
python make_dump_holo.py
python phimost_learn.py --src dumps/from_holo --skip-load --no-warm --no-promote --baselines --epochs 16 --seeds 0,1,2,7,42 --blend-holo 0.4 --out-ckpt checkpoints/most_holo.pt
```

---

## 10. Mapa

| Plik | Rola |
|------|------|
| **Ten plik** | założenia + odpowiedzi na audyt |
| [`ocena_holo.txt`](ocena_holo.txt) | ocena zewnętrzna v0.1 |
| [`../doswiadczenie/glify/HOLOGRAMY.md`](../doswiadczenie/glify/HOLOGRAMY.md) | dziennik wyników |
| `holo_phi.py` | implementacja |
