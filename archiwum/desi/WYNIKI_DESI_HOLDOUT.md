# WYNIKI DESI — duży set + NOWE obiekty (hold-out)

- train=94 val=23 **new_holdout=18**
- epochs=40 augment=3 T=color(g−r) content=luminancja
- ckpt=`C:\Users\drwis\model Ai\checkpoints\most_desi_color.pt`

## Val (widziane rozkładem, nie plikami holdout)
| | Softmax | Most | ZERO |
|--|---------|------|------|
| acc | 0.8696 | 0.8696 | 0.8696 |
| Δ Most−SM=+0.0000  Δ Most−ZERO=+0.0000  gate=0.03

## Nowe obiekty (nigdy w train)
| | Softmax | Most | ZERO |
|--|---------|------|------|
| acc | 0.6667 | 0.7222 | 0.6667 | n=18 |
| **Δ Most−SM=+0.0556**  **Δ Most−ZERO=+0.0556**
| Most−SM new → **PASS** · most vs ZERO → **PASS**

## Per obiekt (Most)
| plik | y | pred | ok | p_cluster | \|dE\| |
|------|---|------|----|-----------|------|
| `demo_desi_new_abell1835_ra210.26_dec2.88` | 1 | 1 | ✓ | 0.98 | 0.660 |
| `demo_desi_new_abell2219_ra250.08_dec46.7` | 1 | 1 | ✓ | 1.00 | 0.569 |
| `demo_desi_new_abell2390_ra328.4_dec17.7_` | 1 | 1 | ✓ | 0.97 | 0.641 |
| `demo_desi_new_coma_far_ra196.5_dec26.5_s` | 1 | 0 | ✗ | 0.02 | 0.470 |
| `demo_desi_new_field_01_ra142.5_dec6.5_s1` | 0 | 1 | ✗ | 0.97 | 0.555 |
| `demo_desi_new_field_02_ra157.5_dec-6.5_s` | 0 | 0 | ✓ | 0.08 | 0.482 |
| `demo_desi_new_field_03_ra167.5_dec14.5_s` | 0 | 1 | ✗ | 0.85 | 0.474 |
| `demo_desi_new_field_04_ra177.5_dec-4.5_s` | 0 | 0 | ✓ | 0.07 | 0.473 |
| `demo_desi_new_field_05_ra187.5_dec19.5_s` | 0 | 0 | ✓ | 0.07 | 0.476 |
| `demo_desi_new_field_06_ra197.5_dec-9.5_s` | 0 | 0 | ✓ | 0.08 | 0.492 |
| `demo_desi_new_field_07_ra207.5_dec8.5_s1` | 0 | 0 | ✓ | 0.05 | 0.501 |
| `demo_desi_new_field_08_ra217.5_dec-2.5_s` | 0 | 0 | ✓ | 0.08 | 0.486 |
| `demo_desi_new_field_09_ra227.5_dec11.5_s` | 0 | 0 | ✓ | 0.07 | 0.510 |
| `demo_desi_new_field_10_ra132.5_dec3.5_s1` | 0 | 0 | ✓ | 0.05 | 0.534 |
| `demo_desi_new_field_11_ra162.5_dec21.5_s` | 0 | 0 | ✓ | 0.07 | 0.482 |
| `demo_desi_new_field_12_ra192.5_dec-12.5_` | 0 | 0 | ✓ | 0.03 | 0.476 |
| `demo_desi_new_leo_far_ra171.8_dec13.5_s1` | 1 | 0 | ✗ | 0.05 | 0.496 |
| `demo_desi_new_virgo_far_ra189.5_dec11.0_` | 1 | 0 | ✗ | 0.05 | 0.472 |
