# Odpowiedź na recenzję — lost-in-the-middle (CPU, ~4M, seq=2048)

**Recenzja miała rację metodologicznie.** Pytanie „softmax = konieczność czy przyzwyczajenie” jest retoryczne; lab pyta: *co przy 100M ma być widać, czego nie widać przy 3M — i dlaczego?* Historia o grubych ogonach Lorentza jest **mierzalna teraz**, zanim prosimy o GPU.

## Co zmierzyliśmy

Protokół: [`../doswiadczenie/glify/LOST_IN_MIDDLE_PROBE.md`](../doswiadczenie/glify/LOST_IN_MIDDLE_PROBE.md)  
Model `medium_long` (~4.3M) · seq=2048 · 400 kroków treningu · Most vs Softmax.

| Metryka | Most | Softmax | Hipoteza „Most lepszy mid” |
|---------|-----:|--------:|:---------------------------:|
| mid_dip (masa uwagi) | **+0.0012** (lekka wklęsłość) | **~0** (płasko) | **FAIL** |
| entropia mid | 4.74 | **6.93** | **FAIL** (SM wyższa) |
| secret_acc (retrieval) | 0.0 wszędzie | 0.0 wszędzie | n/a (zadanie nie nauczyło się) |

`mid_dip = ½(head+tail) − mid`; dodatnie = gorszy środek.

## Werdykt

**FAIL hipotezy na tej skali / tym protokole.**

- Softmax **nie** pokazał głębszego „lost in the middle” niż Most.
- Most **nie** trzymał wyższej entropii mid (wręcz przeciwnie).
- Masa na igle jest u obu mała; krzywe SM praktycznie płaskie; Most lekko gorszy w środku.
- Retrieval sekretu = 0 — krótki trening + bajtowy cue **nie** nauczył odzyskiwania; to limit protokołu, nie dowód na 100M.

## Co to zmienia w planie GPU

| Przed pomiarem | Po pomiarze |
|----------------|-------------|
| GPU: „sprawdźmy, czy Most wygra mid” | GPU: **nie** wynika automatycznie z trendu 3M/2k |
| Retoryka grubych ogonów | **Nie potwierdzona** tu; albo zły protokół, albo zły mechanizm na tej skali |
| Prośba o 100M | Nadal wymaga **innej** hipotezy: *co dokładnie* ma wyjść inaczej @100M |

**GPU wraca do statusu „kosztowna eksploracja / inna hipoteza”**, nie „potwierdźmy już widoczny trend”.

## Update: strojenie gałek zamka

Po FAIL na default knobs zrobiliśmy sweep γ / e_scale / τ ([`../doswiadczenie/glify/MOST_KNOBS_SWEEP.md`](../doswiadczenie/glify/MOST_KNOBS_SWEEP.md)):

- **baseline Most** ent_mid≈3.15 (daleko od SM)
- **ultra_wide** (γ_init=2.5, e_scale=0.1, τ=2) ent_mid≈**6.928** ≥ SM 6.927, dip≤SM — **PASS cienki**
- Wniosek: „grube ogony” **nie wychodzą z samego wzoru przy init=0**; trzeba **świadomie** szerokiego g i słabego ΔE. To też ostrzeżenie: ultra_wide może zbliżać Most do „miękkiego” softplus² bez ostrej selekcji energii.

## Co dalej (uczciwie)

1. Używać `ultra_wide` (lub `fat_tail`) jako knobs w kolejnych mid-probe — nie default.
2. **Wzmocnić protokół retrieval** (secret_acc>0) — nadal 0 we wszystkich wariantach.
3. **100M / NVIDIA** — dopiero po (2) albo gdy ultra_wide pokaże **wyraźną** przewagę mid vs SM (nie ε).

## Spójność z resztą labu

- Parytet r@1 + spowolnienie 1.3–1.7× + hold-out FAIL + ten FAIL mid → **spójny obraz**: Most działa, nie dominuje, nie dowodzi jeszcze unikalnej zalety uwagi na obecnej skali.
- Negatywny wynik **jest** wartością labu — recenzja dostała pomiar, nie slajd.
