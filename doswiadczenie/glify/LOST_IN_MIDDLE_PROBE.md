# Lost-in-the-middle probe — odpowiedź na recenzję

Model medium_long (~3.3M) · seq=**2048** · train_steps=400 · seed=0

**Werdykt:** FAIL hipotezy na tej skali: Softmax nie pokazuje głębszego mid-dipu niż Most (albo Most nie trzyma wyższej entropii mid). Historia „grube ogony → płaski mid” nie potwierdza się przy 3M/2k — albo potrzeba innego protokołu/treningu.

Grube ogony Lorentza → masa uwagi nie zapada się; lost-in-middle płaski dla Most, wklęsły dla Softmax — mierzalne na CPU przed 100M.

mid_dip = 0.5*(head+tail) - mid; dodatnie = wklęsłość (gorszy środek). To nie jest jeszcze Needle 16k (#4); to mechanistyczny proxy na 2k.

## Softmax

### softmax

| rel | attn_mass | secret_acc |
|----:|----------:|-----------:|
| 0.10 | 0.0054 | 0.00 |
| 0.25 | 0.0054 | 0.00 |
| 0.50 | 0.0054 | 0.00 |
| 0.75 | 0.0053 | 0.00 |
| 0.90 | 0.0054 | 0.00 |

mid_dip mass=+0.0000 · mid_dip acc=+0.0000
entropy head/mid/tail=4.333/6.927/7.570

## Most

### most

| rel | attn_mass | secret_acc |
|----:|----------:|-----------:|
| 0.10 | 0.0021 | 0.00 |
| 0.25 | 0.0014 | 0.00 |
| 0.50 | 0.0014 | 0.00 |
| 0.75 | 0.0022 | 0.00 |
| 0.90 | 0.0031 | 0.00 |

mid_dip mass=+0.0012 · mid_dip acc=+0.0000
entropy head/mid/tail=2.776/4.736/5.229

## Hipoteza

- Most mniejszy mid_dip (masa): **False**
- Most wyższa entropia mid: **False**
- secret_acc mid OK: **True**

elapsed=127s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_090530_lost_in_middle_probe`

JSON: `LOST_IN_MIDDLE_PROBE_results.json`
