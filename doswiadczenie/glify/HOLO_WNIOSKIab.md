# Holo-Φ — wnioski P1→P4a, replikacja (sesja 2026-09-11, popołudnie)

**Data przebiegu:** 2026-09-11, 18:12–18:48
**Skrypty:** `run_holo_p1.py` · `run_holo_p2.py` · `run_holo_p3.py` · `run_holo_p4a.py` (bez zmian w kodzie od czasu wcześniejszego przebiegu tego samego dnia)
**Założenia:** [`../../docs/HOLO_ZALOZENIA.md`](../../docs/HOLO_ZALOZENIA.md)
**Wcześniejszy przebieg (ten sam dzień, wcześniej):** [`HOLO_WNIOSKI.md`](HOLO_WNIOSKI.md)

> Ten dokument to **druga, niezależna replikacja** P1→P4a na identycznym kodzie i identycznych seedach co `HOLO_WNIOSKI.md`. Liczby **różnią się** między przebiegami mimo `torch.manual_seed(seed)` — patrz §5 (reprodukowalność). Werdykt końcowy (FAIL) jest **spójny** w obu przebiegach; konkretne PASS-y na poszczególnych seedach — nie.

---

## 1. P1 — kontrola eksperymentalna (blend=0.4, 8 seedów)

| | |
|--|--|
| Ziarnistość r@1 | krok **0.0025** (400 queries) → gate +0.02 ≈ **8 kroków** |
| PASS | **0/8** |
| best ΔSM | +0.0125 (seed 7) |

| seed | Most | SM | ZERO | ΔSM | gate |
|-----:|-----:|---:|-----:|----:|:----:|
| 0 | 0.163 | 0.160 | 0.160 | +0.0025 | FAIL |
| 1 | 0.152 | 0.140 | 0.138 | +0.0125 | FAIL |
| 2 | 0.130 | 0.138 | 0.138 | −0.0075 | FAIL |
| 7 | 0.122 | 0.110 | 0.120 | +0.0125 | FAIL |
| 11 | 0.177 | 0.170 | 0.135 | +0.0075 | FAIL |
| 17 | 0.133 | 0.155 | 0.142 | −0.0225 | FAIL |
| 23 | 0.180 | 0.185 | 0.165 | −0.0050 | FAIL |
| 42 | 0.113 | 0.115 | 0.133 | −0.0025 | FAIL |

**Wniosek:** ΔSM rozrzucone symetrycznie wokół zera (−0.0225…+0.0125) — brak sygnału, nie kwestia ziarnistości metryki (krok 0.0025 ≪ obserwowany rozrzut). Gorzej niż w poprzednim przebiegu (tam 1/8 PASS).

---

## 2. P2 — sweep `blend_holo` (5 seedów × 5 blendów)

| blend | PASS/5 | mean ΔSM | best ΔSM | worst ΔSM |
|------:|-------:|---------:|---------:|----------:|
| 0.0 | 0/5 | +0.0020 | +0.0200 | −0.0275 |
| 0.2 | 2/5 | +0.0105 | +0.0325 | −0.0025 |
| 0.4 | 0/5 | −0.0030 | +0.0125 | −0.0225 |
| 0.6 | 1/5 | −0.0005 | +0.0275 | −0.0200 |
| **0.8** | **2/5** | **+0.0220** | +0.0500 | +0.0050 |

**Wniosek:** najlepszy punkt tego przebiegu to `blend=0.8` (2/5, mean ΔSM najwyższe i **wyłącznie dodatnie** ΔSM — jedyny blend bez ani jednego ujemnego seeda). W poprzednim przebiegu najlepszy był `blend=0.6` (3/5). Żaden blend w żadnym z dwóch przebiegów nie zbliża się do bramki 4/5 — **FAIL P1 to cecha metody na tym dumpie, nie źle dobrany hiperparametr** (wniosek stabilny mimo różnych "zwycięzców" sweepu między przebiegami).

`blend=0.6` pozostaje pre-registered do P4a **z wcześniejszego** uzasadnienia (nie post-hoc z tego sweepu) — zgodnie z dyscypliną pre-rejestracji z `HOLO_ZALOZENIA.md` §8b, mimo że numerycznie nie jest tu zwycięzcą.

---

## 3. P3 — ablacja rodziny 女 bez tautologii (blend=0.6, 4 seedy × 4 warunki)

| cond | phase | binds | PASS/4 | mean ΔSM | fam@3 Most | fam@3 SM |
|------|:-----:|:-----:|-------:|---------:|-----------:|---------:|
| `pp` (++) | + | + | 1/4 | +0.0044 | 1.00 | 1.00 |
| `mp` (−+) | − | + | 2/4 | +0.0138 | 1.00 | 1.00 |
| `pm` (+−) | + | − | 0/4 | +0.0019 | 1.00 | 1.00 |
| `mm` (−−) | − | − | 1/4 | +0.0075 | 1.00 | 1.00 |

**Efekt fazy na fam@3 (pp−mp, binds stałe):** Most **+0.000** · Softmax **+0.000**.

**Wniosek — powtórzony i wzmocniony względem poprzedniego przebiegu:** `family_hit@3` jest **na suficie (1.00) we wszystkich 4 warunkach, u obu modeli**, niezależnie od tego czy rodzina 女 dzieli wspólną fazę `φ_root`. Wyłączenie fazy (pp→mp) **nic nie zmienia** w tej metryce. Klaster rodziny idzie w całości z krawędzi grafu (`holo:root:女` binds + wspólny `parent`) i/lub z contentu (hash `S`/`E`), **nie** z wstrzykniętej fazy. To jest teraz potwierdzone w dwóch niezależnych przebiegach — nie przypadek jednego seeda.

Efekt uboczny: metryka fam@3 jest **bezużyteczna diagnostycznie** przy obecnym designie (6-elementowa rodzina w 73-atomowym grafie z gęstymi bindami rodziny → top-3 trywialnie trafia w rodzinę). Przyszła iteracja P3 potrzebuje **ostrzejszego** testu (np. rodzina bez wspólnego parenta w eval, albo distractor-y bliżej rodziny w przestrzeni contentu).

---

## 4. P4a — gate lokalny pre-registered (blend=0.6, 8 seedów)

| | |
|--|--|
| Kryterium | PASS ≥ 80% (≥7/8) |
| **Wynik** | **2/8** |
| mean ΔSM | +0.0041 |
| best | +0.0275 @ seed 0 |
| **Werdykt** | **P4a FAIL** |

| seed | Most | SM | ZERO | ΔSM | gate |
|-----:|-----:|---:|-----:|----:|:----:|
| 0 | 0.155 | 0.128 | 0.155 | +0.0275 | PASS |
| 1 | 0.152 | 0.142 | 0.128 | +0.0100 | FAIL |
| 2 | 0.147 | 0.135 | 0.115 | +0.0125 | FAIL |
| 7 | 0.117 | 0.098 | 0.102 | +0.0200 | PASS |
| 11 | 0.152 | 0.150 | 0.168 | +0.0025 | FAIL |
| 17 | 0.135 | 0.155 | 0.128 | −0.0200 | FAIL |
| 23 | 0.165 | 0.177 | 0.170 | −0.0125 | FAIL |
| 42 | 0.102 | 0.110 | 0.128 | −0.0075 | FAIL |

**Wniosek:** 2/8 (25%) — daleko od bramki 80%. Zgodny kierunkowo z poprzednim przebiegiem (tam 1/8), gorszy niż potrzeba o ~5-6 seedów niezależnie od tego, który konkretnie z dwóch przebiegów bierzemy.

---

## 5. Reprodukowalność — nowy wniosek (nieobecny w `HOLO_WNIOSKI.md`)

Ten przebieg użył **dokładnie tego samego kodu i tych samych seedów** co wcześniejszy tego dnia, a mimo to liczby się różnią punktowo (np. P4a seed=0: ΔSM=+0.035 wcześniej vs **+0.0275** tutaj; P1 seed=23: ΔSM=+0.020 wcześniej vs **−0.0050** tutaj — różnica wystarczająca, by zmienić PASS↔FAIL na pojedynczym seedzie). `torch.manual_seed(seed)` nie daje tu bit-for-bit powtarzalności — najbardziej prawdopodobne źródło: niedeterministyczna redukcja zmiennoprzecinkowa przy wielowątkowym CPU-BLAS w PyTorch (kolejność sumowania w matmul/attention zależy od harmonogramu wątków, nie jest pinowana przez `manual_seed`).

**Konsekwencja dla protokołu:** bramki PASS/FAIL na pojedynczym seedzie są **szumem na granicy** (gate ΔSM=0.02 vs krok r@1=0.0025 — 8 kroków, ale run-to-run drift tego samego seeda potrafi być rzędu kilku kroków). Zbiorcze werdykty (n_pass/n_seeds, mean ΔSM) są w obu przebiegach **kierunkowo zgodne** (FAIL wszędzie, żaden blend/warunek nie zbliża się do bramki) — to jest wynik odporny. Ale przywoływanie **konkretnego** seeda jako "PASS" (np. "seed 23 PASS" z pierwszego przebiegu) jest niewiarygodne bez dodatkowego pinningu determinizmu.

**Rekomendacja przed P4b / kolejną iteracją:** dodać na starcie skryptów `torch.set_num_threads(1)` (lub `torch.use_deterministic_algorithms(True)`) i potwierdzić bit-for-bit powtarzalność jednego seeda w dwóch przebiegach, zanim jakikolwiek pojedynczy-seed wynik wejdzie do komunikacji jako dowód.

---

## Jednym zdaniem

Replikacja potwierdza wcześniejszy werdykt **P4a FAIL** (2/8, potrzeba 7/8) inną liczbą PASS-ów niż poprzedni przebieg tego samego kodu/seedów — sam kierunek (Most nie bije Softmax-twin na tym dumpie) jest **stabilny między przebiegami**, ale pojedyncze-seed claimy nie są, bo pipeline nie jest bit-reprodukowalny mimo `manual_seed` (§5). P3 potwierdza (mocniej niż poprzednio, bo fam@3=1.00 wszędzie): klaster rodziny 女 idzie z binds/content, nie z fazy. Przed P4b: (a) opinąć determinizm, (b) jeśli dalej ciągnąć Holo-Φ lokalnie — nowa metryka P3 odporna na sufit, nie kolejny sweep blendu.
