# Kana P1 - cel pierwotny raportu hologram-znaki: trening, nie retrieval

Seedy: [0, 1, 17, 23, 42] - epochs=16 - encodings=['hologram', 'symbolic'] - n_train=80 n_val=25

Pierwszy test CELU PIERWOTNEGO z sekcji 1 raportu hologram-znaki (oszczednosc przebiegow treningowych) na realnym, duzym korpusie: 71 tozsamosci dzwiekowych (hiragana/katakana/romaji Hepburn, bez kanji) = 286 atomow, 26x wiecej niz Lingua P1 (3 slowa, 11 atomow), ktory sam siebie oznaczyl jako 'za malo sygnalu'.

## Agregat per encoding

| encoding | atoms | PASS | mean DSM | best DSM | worst DSM | mean Most | mean SM | mean ZERO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hologram | 286 | 0/5 | -0.0075 | +0.0025 | -0.0175 | 0.005 | 0.013 | 0.007 |
| symbolic | 286 | 0/5 | -0.0050 | +0.0025 | -0.0150 | 0.008 | 0.013 | 0.010 |

**Najlepszy encoding (PASS, potem meanD):** **symbolic**

## Pelna tabela

| encoding | seed | Most | SM | ZERO | DSM | gate |
|---|---:|---:|---:|---:|---:|:----:|
| hologram | 0 | 0.003 | 0.013 | 0.005 | -0.0100 | FAIL |
| hologram | 1 | 0.007 | 0.005 | 0.010 | +0.0025 | FAIL |
| hologram | 17 | 0.005 | 0.022 | 0.010 | -0.0175 | FAIL |
| hologram | 23 | 0.010 | 0.018 | 0.007 | -0.0075 | FAIL |
| hologram | 42 | 0.000 | 0.005 | 0.003 | -0.0050 | FAIL |
| symbolic | 0 | 0.010 | 0.007 | 0.010 | +0.0025 | FAIL |
| symbolic | 1 | 0.007 | 0.013 | 0.010 | -0.0050 | FAIL |
| symbolic | 17 | 0.007 | 0.022 | 0.013 | -0.0150 | FAIL |
| symbolic | 23 | 0.013 | 0.018 | 0.015 | -0.0050 | FAIL |
| symbolic | 42 | 0.003 | 0.005 | 0.003 | -0.0025 | FAIL |

**Werdykt:** best encoding=symbolic PASS 0/5 meanDSM=-0.0050; gate = Most r@1 - Softmax r@1 >= CANON.gate_recall_r1 (0.02)

JSON: `kana_p1_results.json`

