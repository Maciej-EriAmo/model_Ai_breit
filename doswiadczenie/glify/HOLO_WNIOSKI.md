# Holo-Φ — wnioski (P0 → P2)

**Data:** 2026-09-11  
**Założenia:** [`../../docs/HOLO_ZALOZENIA.md`](../../docs/HOLO_ZALOZENIA.md)  
**Audyt:** [`../../docs/ocena_holo.txt`](../../docs/ocena_holo.txt) (audytor 1+2)

---

## P0 — definicje A i T

| Symbol | Definicja (kanon) |
|--------|-------------------|
| **A** | Częstość znaku w korpusie lab (+ boost rodziny rdzenia). **Nie** hash. |
| **T** | `energy_to_T(A` lub `|Σ|`, wspólny `ref)` + fokus 1. litery + archive `T_TOMB`. |
| **φ** | Hash etykiety (+ `φ_root` dla rodziny) — faza, nie saliency. |
| **Seal** | Bramkuje niskie **T** = energię fali / TOMB — zamierzone, nie losowy hash. |

**Wniosek:** łańcuch `A→T→seal` jest teraz interpretowalny; P0 domknięte.

---

## P1 — kontrola eksperymentalna

| Fakt | Wartość |
|------|---------|
| Seedy | 8 (0,1,2,7,11,17,23,42) |
| Ziarnistość r@1 | krok **0.0025** (400 queries) → gate +0.02 ≈ **8 kroków** (OK) |
| PASS @ blend=0.4 | **1/8** (seed 23, ΔSM=+0.020) |
| Krzywa Most seed0 | r@1 rośnie do ep14, potem lekki spadek — **nie pełny plateau** |

**Wniosek:** FAIL nie wynika z „zbyt grubego” progu metryki ani z samego braku seedów. Stabilność multi-seed przy 0.4 jest słaba. Szczegóły: [`HOLO_P1.md`](HOLO_P1.md).

---

## P2 — sweep `blend_holo`

| blend | PASS /5 | mean ΔSM |
|------:|--------:|---------:|
| 0.0 | 1/5 | +0.004 |
| 0.2 | 0/5 | −0.009 |
| **0.4** | 1/5 | −0.001 |
| **0.6** | **3/5** | **+0.020** |
| 0.8 | 2/5 | +0.010 |

**Wnioski (po recenzji [`założenia.txt`](założenia.txt)):**

1. **FAIL P1 był częściowo od hiperparametru** — właściwe słowo: *częściowo*. `0.6` nie „naprawia metody” (3/5 < 4/5); pokazuje, że **P1 nie był wystarczającym testem potencjału** (konfiguracja nieoptymalna).  
2. **`blend_holo = 0.6`** = **lab default / pre-registered dla P4**, nie wartość „udowodniona optymalna”. Nie dostrajać dalej 0.55/0.65 przed P4.  
3. **3/5 to sygnał, nie zwycięstwo** — ciekawszy niż 1/5, nadal poniżej gate 4/5.  
4. Z tabeli **bezpośrednio** wynika tylko: przy 0.6 Most osiąga PASS 3/5 i mean ΔSM=+0.020.  
   Zdanie „Most lepiej korzysta z binów fazy w τ” = **hipoteza interpretacyjna**, nie wynik wykazany. Do „dlaczego”: ablacja `content only | phase only | content+phase` (później).  
5. Kształt sweepu (0.0→1, 0.2→0, 0.4→1, **0.6→3**, 0.8→2) osłabia „Holo nie działa” na rzecz: **wynik zależy od proporcji informacji holograficznej; w zakresie best = 0.6**.

**P2 = udany wynik diagnostyczny** — nie dowód przewagi Mostu; znalazł punkt `0.6` do pełnego protokołu.

Szczegóły: [`HOLO_P2.md`](HOLO_P2.md).

---

## P3 — ablacja 女 (bez tautologii)

Design: phase± × binds±; klucz **++ vs −+** (binds stałe). blend=0.6. Szczegóły: [`HOLO_P3.md`](HOLO_P3.md).

| cond | phase | binds | PASS/4 | mean ΔSM |
|------|:-----:|:-----:|-------:|---------:|
| `pp` | + | + | 1/4 | (patrz JSON) |
| `mp` | − | + | 1/4 | |
| `pm` | + | − | **0/4** | najgorzej |
| `mm` | − | − | 1/4 | |

**Efekt fazy na fam@3 (pp−mp):** Most **0.000** · Softmax **0.000** — metryka `family_hit@3` **nasycona (1.0)** we wszystkich warunkach (wspólny parent `root/女` + podobny szablon content i tak spina rodzinę).

**Wnioski P3:**

1. Przy **stałych binds** wyłączenie `φ_root` **nie** psuje fam@3 — klaster w tej metryce **nie zależy od wstrzykniętej fazy** (idzie z grafu/content).  
2. To **odrzuca tautologię audytu** w drugą stronę: nie udowadniamy „Most nauczył się semantyki z fazy”; pokazujemy, że **faza nie jest tu nośnikiem klastra**.  
3. Gate: sam `phase+` bez binds (`pm`) = **0/4 PASS** — wspólna faza **bez** krawędzi nie pomaga Mostowi w ΔSM.  
4. Do P4: nie opierać claimu rodziny na `φ_root`; opierać na binds/content albo zaprojektować ostrzejszą metrykę (np. bez wspólnego parenta w eval).

---

## P4 — dwa poziomy oceny (z [`założenia.txt`](założenia.txt))

### A. Gate lokalny (P4a) — **FAIL**

Pre-registered: `blend_holo=0.6` · 8 seedów · kryterium ≥80% (≥7/8).  
Raport: [`HOLO_P4a.md`](HOLO_P4a.md).

| | |
|--|--|
| PASS | **1/8** (seed 0, ΔSM=+0.035) |
| mean ΔSM | ≈ **0** |
| Werdykt | **P4a FAIL** |

Sygnał P2 (3/5 na podzbiorze) **nie utrzymał się** na pełnych 8 seedach. Bez dostrajania blendu.

### B. Gate skalowania (P4b) — jeszcze nie

1→8 alfabetów: koszt / transfer — osobny protokół; nie mylić z lokalnym r@1.

---

## Jednym zdaniem

A/T OK; P2 znalazł **0.6** jako lab default; P3: φ_root nie spina 女; **P4a FAIL (1/8)** — nie ma lokalnej przewagi Mostu na pełnym protokole; ewentualnie P4b (skala alfabetów).
