# Strojenie gałek zamka Most (γ / e_scale / τ)

seq=2048 · steps=300 · seed=0

**Werdykt:** PASS strojenia vs Softmax: **most:ultra_wide** ent_mid=6.928≥SM 6.927, dip=-0.0002≤SM +0.0001.

A = g · softplus(S/τ)² / (g² + (e_scale·ΔE)²)

| name | γ_init | e_scale | τ | ent_mid | mid_dip | mass@0.5 |
|------|-------:|--------:|--:|--------:|--------:|---------:|
| softmax | — | — | — | 6.927 | +0.0001 | 0.0060 |
| most:baseline | 0.0 | 1.0 | 1.0 | 3.146 | -0.0055 | 0.0149 |
| most:wide_gamma | 1.5 | 1.0 | 1.0 | 5.046 | +0.0003 | 0.0040 |
| most:soft_energy | 0.0 | 0.25 | 1.0 | 6.777 | -0.0021 | 0.0826 |
| most:fat_tail | 1.5 | 0.25 | 1.5 | 6.913 | -0.0002 | 0.0046 |
| most:ultra_wide | 2.5 | 0.1 | 2.0 | 6.928 | -0.0002 | 0.0079 |

elapsed=338s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_091257_most_knobs_sweep`

JSON: `MOST_KNOBS_SWEEP_results.json`
