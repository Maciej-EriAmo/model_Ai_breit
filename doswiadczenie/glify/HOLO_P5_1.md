# Holo-Φ P5.1 — kontrola na τ=Unihan (stroke_count), holdout=radykał

**Data:** 2026-09-13
**Plan:** [`../../docs/HOLO_P5_PLAN_IMPLEMENTACJI.md`](../../docs/HOLO_P5_PLAN_IMPLEMENTACJI.md) ·
[`../../docs/HOLO_P5_TOR_DESI.md`](../../docs/HOLO_P5_TOR_DESI.md)
**Kod:** `unihan_tracer.py`, `make_dump_holo_v2.py`, `bench_holo_v2.py`

---

## Setup

- τ (T) = `energy_to_T(stroke_count(ch), ref)` — **prawdziwy pomiar Unihan**
  (Unicode 17.0, `kTotalStrokes`/`kRSUnicode`), nie hash(etykieta). Niezależność
  od hasha zweryfikowana w P5.0 (`test_unihan_tracer.py`: |Pearson| z hashem < 0.3).
- Zadanie/etykieta = **radykał Kangxi** (fakt słownikowy, nie wstrzyknięty ręcznie).
- Split (opcja A, §0 planu): 5 radykałów × 20 znaków. Train = 4 radykały
  (9 人, 38 女, 75 木, 86 火) jako osobne grafy-per-radykał. **Holdout = radykał
  85 (水), CAŁKOWICIE nieobecny w żadnym forward passie treningowym**
  (zweryfikowane: `test_holo_v2_split.py`, 4/4 PASS).
- Determinizm: `torch.set_num_threads(1)` + `use_deterministic_algorithms` +
  reset RNG przed budową danych — zweryfikowany bit-do-bitu
  (`test_holo_v2_determinism.py` PASS) **przed** sweepem seedów, zgodnie z §2.5.
- Gate: Most vs ZERO na holdout r@1 (≥+0.02), 8 seedów, ≥7/8 = PASS metody.

## Wynik (8 seedów, epochs=60 — plateau, patrz niżej)

| seed | Most | SM | ZERO | ΔZERO | ΔSM | train_sanity | gate |
|-----:|-----:|---:|-----:|------:|----:|--------------:|:----:|
| 0 | 0.005 | 0.015 | 0.015 | −0.0100 | −0.0100 | 1.000 | FAIL |
| 1 | 0.020 | 0.020 | 0.010 | +0.0100 | +0.0000 | 1.000 | FAIL |
| 2 | 0.010 | 0.015 | 0.015 | −0.0050 | −0.0050 | 1.000 | FAIL |
| 7 | 0.005 | 0.010 | 0.005 | +0.0000 | −0.0050 | 1.000 | FAIL |
| 11 | 0.005 | 0.005 | 0.000 | +0.0050 | +0.0000 | 1.000 | FAIL |
| 17 | 0.025 | 0.025 | 0.005 | +0.0200 | +0.0000 | 1.000 | PASS |
| 23 | 0.020 | 0.025 | 0.040 | −0.0200 | −0.0050 | 1.000 | FAIL |
| 42 | 0.020 | 0.035 | 0.045 | −0.0250 | −0.0150 | 1.000 | FAIL |

**P5.1 FAIL: 1/8** · mean ΔZERO=**−0.0031** · mean ΔSM=**−0.0050** —
praktycznie zero, w żadną stronę. JSON: `holo_p5_1_results.json`.

## Diagnostyka krytyczna — dlaczego, nie tylko czy

Zanim FAIL policzono jako wynik, sprawdzono krzywą uczenia (wymóg §2 planu:
"krzywa ucząca do plateau"), żeby odróżnić "τ nie pomaga" od "model niczego się
tu nie uczy" (floor effect):

| epoka | holdout r@1 | train r@1 (in-distribution, radykał treningowy) |
|------:|------------:|--------------------------------------------------:|
| 0 | 0.020 | 0.062 |
| 15 | 0.015 | 0.188 |
| 30 | 0.010 | 0.875 |
| 45 | 0.010 | **1.000** |
| 60–119 | 0.005–0.010 | 1.000 (plateau) |

**Model uczy się doskonale** (train r@1 → 100%, strata spada monotonicznie
3.09→1.31 w pierwszych 30 epokach) — to nie jest niedouczony/zepsuty setup.
Ale **holdout r@1 nigdy nie odrywa się od poziomu szansy (~1/51≈0.02) i lekko
SPADA** w miarę jak train-accuracy rośnie do 100% — klasyczny podpis
**przeuczenia na tożsamość konkretnych znaków treningowych**, nie uczenia się
transferowalnego mechanizmu "atenduj po bliskości τ".

## Wniosek

1. **P5.1 FAIL**, ale z innej przyczyny niż P4a. P4a: τ=hash(etykieta) było
   tautologią (audyt: usunięcie fazy nic nie zmieniało — Δ=0.000). P5.1: τ jest
   tu realne i niezależne (Unihan), a mimo to **żaden** z trzech torów
   (Most/Softmax/ZERO) nie generalizuje na holdout — wszystkie płasko na
   poziomie szansy. To sugeruje, że wąskim gardłem nie jest wybór między
   zamkiem Lorentza a Softmaxem, tylko **reprezentacja treści**: `content` =
   `hash_embed(znak)` nie niesie żadnej cechy współdzielonej między znakami z
   różnych radykałów, więc żaden mechanizm uwagi nie ma z czego zbudować
   mostu do znaku, którego nigdy nie widział.
2. Zgodnie z regułą stop/go planu (§2.6, P5.2: *"tylko jeśli P5.1 pokaże
   jakikolwiek sygnał"*) — **nie uruchamiam P5.2** (mini-sweep `blend_holo`).
   1/8 PASS przy ~zerowym mean Δ to szum, nie sygnał do dostrajania.
3. P5.3 (ablacja τ=const) i P5.4 (pełny gate) **nie mają sensu przed naprawą
   punktu 1** — ablacja τ na modelu, który i tak nie generalizuje z żadnym τ,
   niczego nie rozstrzygnie.

## Rekomendacja na przyszłość (nie zaimplementowane w tej sesji)

Żeby test miał w ogóle szansę wykazać przewagę Mostu, `content` musiałby nieść
**jakąś** strukturę współdzieloną między train i holdout — np. dekompozycję
graficzną znaku (komponenty/kreski jako współdzielone tokeny, nie jeden hash
całego znaku) albo IDS (Ideographic Description Sequence) zamiast
`atom_content_vec(hash(S))`. Bez tego żadna architektura (Most czy Softmax)
nie ma z czego zbudować transferu na nigdy niewidziany znak.

---

Jednym zdaniem: **P5.1 pokazuje przeuczenie na tożsamość, nie brak/obecność
mostu Lorentza** — content trzeba przeprojektować, zanim pytanie P4a/P5 (czy τ
pomaga) da się w ogóle uczciwie zadać na holdout.
