# T2 — tiny LM, ta sama informacja dla obu jąder

Zadeklarowane przed seedem 0. Sześć dokumentów ze wspólnym prefiksem `COMMON-PREFIX-0123456789-ABCDEFGHIJ-` i ogonem `aaa`…`fff`. Numer źródła nie jest znakiem w tekście. Margines straty **±0.05**. 20 seedów, 300 kroków, d=64, L=2, jak smoke #1b. Średniego LM nie ma.

Specyfikacja `jadro_t2_spec.json`. Przebiegi `jadro_t2_runs.jsonl`.

| porównanie | Δ straty | 90% CI | w ±0.05? |
|---|---:|---|:-:|
| A  Most·0 − Softmax·0 | +0.0009 | [-0.0005, +0.0022] | tak |
| B  Most·tok − Softmax·tok | -0.0016 | [-0.0040, +0.0006] | tak |
| plugin  Softmax·tok − Softmax·0 | -0.0075 | [-0.0109, -0.0041] | tak |
| C  Most kanoniczny − Softmax·0 (mieszane) | +0.0045 | [+0.0028, +0.0063] | tak |

Równoważność wzoru czytamy z A i z B. C nie jest werdyktem o jądrze. Plugin działa, gdy średnia różnica Softmax·tok − Softmax·0 jest ujemna i cały 90% CI leży pod zerem.

## Strata końcowa (średnia z seedów)

| ramię | strata | czas sumaryczny |
|---|---:|---:|
| A-bw | 0.0506 | 83 s |
| A-sm | 0.0497 | 58 s |
| B-bw | 0.0405 | 84 s |
| B-sm | 0.0421 | 59 s |
| C-bw | 0.0542 | 86 s |

Czas Most / Softmax = 253 s / 117 s = 2.15×.

Każdy przebieg ma skończoną stratę i ostatni krok niższy niż pierwszy.

## Wniosek

Zadeklarowana bramka T2 jest spełniona. Wzory są równoważne w ±0,05 i plugin obniża stratę Softmaxu.

1. **A.** Most bez ΔE i Softmax bez pluginu: Δ = +0,0009, 90% CI [−0,0005, +0,0022]. Cały przedział jest w ±0,05.
2. **B.** Oba jądra z numerem źródła w tokenie: Δ = −0,0016, 90% CI [−0,0040, +0,0006]. Też w ±0,05. Różnica jąder jest mniejsza niż dwie tysięczne.
3. **Plugin.** Softmax z numerem źródła ma stratę 0,0421 wobec 0,0497 bez niego. Różnica −0,0075, 90% CI [−0,0109, −0,0041], cały pod zerem.
4. **C, mieszane.** Most kanoniczny (tracer pozycji przez ΔE) ma stratę wyższą o +0,0045, CI [+0,0028, +0,0063]. Przedział jest nad zerem i zarazem w ±0,05. To nie jest werdykt o samym wzorze.
5. **T3.** Wszystkie 100 przebiegów ma skończoną, schodzącą stratę. Most trwa 2,15 raza dłużej (253 s wobec 117 s).

Korpus jest mały i powtarzalny: strata schodzi do około 0,05. Zgoda ±0,05 jest tu szeroka wobec różnicy jąder rzędu 0,001. Plugin jest prawdziwy na tym zbiorze, nie jest pomiarem dużego języka.

