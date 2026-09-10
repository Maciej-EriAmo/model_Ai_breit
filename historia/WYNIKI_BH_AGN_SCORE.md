# Poszukiwanie śladów czarnych dziur (AGN/kwazar) — score Most

**Uwaga naukowa:** model nie widzi horyzontu. Widzi cutout Legacy; trening był cluster/field. Tu: czy znane AGN wyglądają „gorąco/klastrowo” w barwie→T vs kontrolne pola.

ckpt=`most_desi_color.pt` · n=14

| obiekt | kind | pred | p_cluster | T_mean | |dE| |
|--------|------|------|-----------|--------|------|
| `bh_ngc4151_ra182.6358_dec39.4057_s128` | seyfert/BH/AGN | cluster | 1.000 | 0.538 | 0.587 |
| `bh_m87_ra187.7059_dec12.3911_s128` | smbh_host/BH/AGN | cluster | 0.999 | 0.543 | 0.801 |
| `bh_oj287_ra133.7036_dec20.1085_s128` | blazar_binary_cand/BH/AGN | cluster | 0.995 | 0.675 | 0.527 |
| `bh_3c273_ra187.2779_dec2.0524_s128` | quasar/BH/AGN | cluster | 0.995 | 0.580 | 0.592 |
| `bh_3c279_ra194.0465_dec-5.7893_s128` | blazar/BH/AGN | cluster | 0.994 | 0.542 | 0.525 |
| `bh_cta102_ra338.1517_dec11.7308_s128` | quasar/BH/AGN | cluster | 0.994 | 0.609 | 0.469 |
| `bh_3c48_ra24.4221_dec33.1598_s128` | quasar/BH/AGN | cluster | 0.992 | 0.633 | 0.528 |
| `bh_mrk421_ra166.1138_dec38.2088_s128` | blazar/BH/AGN | cluster | 0.795 | 0.645 | 0.588 |
| `ctrl_near_4151_ra181.0_dec38.0_s128` | field/CTRL | cluster | 0.715 | 0.151 | 0.464 |
| `bh_3c345_ra250.745_dec39.81_s128` | quasar/BH/AGN | field | 0.190 | 0.629 | 0.527 |
| `ctrl_near_3c273_ra188.5_dec3.0_s128` | field/CTRL | field | 0.075 | 0.634 | 0.484 |
| `bh_ton618_ra186.3542_dec31.5261_s128` | quasar_ultramassive/BH/AGN | field | 0.073 | 0.570 | 0.481 |
| `ctrl_near_m87_ra186.0_dec11.0_s128` | field/CTRL | field | 0.038 | 0.581 | 0.487 |
| `bh_cena_ra201.3651_dec-43.0191_s128` | radio_gal/BH/AGN | field | 0.029 | 0.500 | 0.476 |

## Skrót
- AGN/BH targets: mean p_cluster=0.732 (n=11)
- CTRL fields: mean p_cluster=0.276 (n=3)
