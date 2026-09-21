# Audyt projektu Most_Lorentz (Glify)

**Status:** przyjęty jako kanon wniosków (2026-09-11)  
**Metoda:** [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md)  
**Wyniki:** [`WYNIKI_GLYPHS.md`](WYNIKI_GLYPHS.md) · [`WYNIKI_GLYPHS_G2.md`](WYNIKI_GLYPHS_G2.md)

---

## 1. Stabilność i zależność od inicjalizacji (krytyczna wątpliwość)

Wyniki empiryczne z **G0** (PASS tylko dla seed 42) oraz **G2 Run B** (PASS tylko dla seed 0, skuteczność 1/3) wskazują na fundamentalną niestabilność. Przewaga Mostu nad Softmaxem jest na tym etapie w dużej mierze uzależniona od losowego rozkładu początkowych wag. Przy tak małym korpusie (~27–31 atomów) dowodzi to silnej podatności na overfitting do szumu inicjalizacyjnego.

**Dopisek labu:** po fixach audytu kontraktu (**G2 Run C/C′**) gate jest **FAIL na 0/3 seedów** (Most ≤ Softmax). To **wzmocnia** ten punkt — kruchy PASS nie wrócił po uczciwszym kontrakcie.

## 2. Skuteczność mechaniki bramkującej (luka w architekturze)

Obserwacja z G0: *archive cold (`lat/archive/X`) nadal wchodzi wysoko* obnaża słabość obecnej implementacji Mostu. Skoro oś `T` ma decydować o wpuszczeniu atomu do okna uwagi, wysoki recall dla atomów COLD oznacza, że bramka „przecieka”. Kara termiczna dla stanu zimnego jest matematycznie zbyt słaba, by skutecznie wykluczyć niechciany kontekst.

## 3. Metodologia i czystość założeń (kompromis i plusy)

Rygorystyczne rozdzielenie osi tożsamości/roli (`S`, `E`) od osi termicznej (`T`) to czysty i mierzalny kontrakt (`glyph_unit_v2`: łacina = słowo, CJK = znak). Utrzymanie w dzienniku nieudanych seedów jest poprawną praktyką badawczą — ukazuje faktyczny stan bez upiększania.

G0 opiera się na danych syntetycznych; G1 (photo) jest smoke. Prawdziwa weryfikacja skali: korpus z `glyph_transcribe.py` + stabilny multi-seed. Softmax w `--baselines` = Softmax-twin na tym samym Φ (nie osobny train na `classical_tokens`) — patrz TRANSCODE §2.

---

## Wnioski i kryteria sukcesu dla kolejnych pętli

### 1. Uszczelnienie Mostu (bloker) — **WDROŻONE**

Zimne atomy przebijały filtr. Kod: `most_seal.py` → `apply_t_key_seal` w `bridge.py` + `recall_model.py` (Softmax twin **bez**).

| Param (`CANON`) | Sens |
|-----------------|------|
| `t_attn_floor=0.20` | hard mask: T &lt; floor → waga 0 (TOMB / archive) |
| `t_attn_cold=0.45` | pas COLD: kara `exp(-k·relu(cold−T))` |
| `t_attn_exp_k=12` | siła kary |
| `t_attn_seal=True` | włącznik (ablacja: False) |

CLS zawsze otwarty jako key. Test: `doswiadczenie/testy/test_most_seal.py`.

### 2. Ewaluacja hipotezy: ciągłe `T`

Przejście z grubego {HOT, COLD} na ciągłe  
\(T_{vac} + \Delta T \cdot \exp(-\lambda \cdot age)\)  
może wygładzić gradienty i zmniejszyć rozrzut między seedami.

### 3. Powiększenie próby

Dowodzenie bramki na ~30 atomach daje fałszywe poczucie pewności.  
**Kryterium przed** stroke-grafem kanji / geometrią DESI: stabilny PASS (np. **≥4/5 seedów**) na zbiorze rzędu **500–1000** atomów z `glyph_transcribe.py`.

### 4. Zawieszenie pobocznych torów

DESI / JoannaM / niebo — **izolowane** w `doswiadczenie/desi|joannam/`, bez mieszania z alfabetem, dopóki Most nie udowodni stabilności na piśmie. Stroke depth dla kanji i hologramy (§7 TRANSCODE) — **po** uszczelnieniu osi `T`.

---

## Kolejność operacyjna (z tego audytu)

1. Uszczelnić bramkę `T` / COLD (bloker)  
2. Odzyskać / ustabilizować gate na kontrolowanym dumpie (≥4/5 seed)  
3. Skala 500–1000 atomów z transkrypcji  
4. Dopiero potem: classical Softmax loader, kanji strokes, hologramy; DESI osobno
