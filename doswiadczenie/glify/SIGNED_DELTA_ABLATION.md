# Skierowane ΔE w Moście (energie ze znakiem)

E z Linear może być ujemne już dziś; dE² gubi znak. directed/onesided wstawiają znak z powrotem przy A≥0.

**Werdykt:** PASS vs Softmax: **most:ultra_wide_directed** ent_mid=6.928 dip=-0.0000.

seq=2048 · steps=300 · seed=0

| name | mode | ent_mid | mid_dip | mass@0.5 |
|------|------|--------:|--------:|---------:|
| softmax | — | 6.928 | +0.0001 | 0.0055 |
| most:square | square | 6.412 | +0.0003 | 0.0035 |
| most:directed | directed | 6.377 | +0.0012 | 0.0038 |
| most:onesided | onesided | 6.660 | -0.0002 | 0.0052 |
| most:ultra_wide_directed | directed | 6.928 | -0.0000 | 0.0057 |

elapsed=335s · `C:\Users\drwis\model Ai\runs\20260921_092607_signed_delta_ablation`
