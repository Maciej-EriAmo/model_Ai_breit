# Mendelejew FULL118 — Most vs Softmax vs ZERO

dump=`C:\Users\drwis\model Ai\dumps\from_mendelejew_full` · epochs=16 · seeds=[0, 1, 2, 7, 42]
cuda=False · CUDA niedostępne — VRAM N/A; proxy = peak RSS (MB) procesu + wall time.
gate Δ(Most−SM) r@1 ≥ 0.02

**Werdykt:** Most mean r@1=0.568 r@3=0.865 | SM r@1=0.563 r@3=0.882 | ΔSM r@1=+0.005 r@3=-0.016 | gate 0/5 | wall Most/SM=43.6/25.4s | RSS Most/SM=369/369MB

## Tabela seedów

| seed | Most r@1 | Most r@3 | SM r@1 | SM r@3 | ZERO r@1 | ΔSM r@1 | gate | Most s | SM s | Most RSS | SM RSS |
|-----:|---------:|---------:|-------:|-------:|---------:|--------:|:----:|-------:|-----:|---------:|-------:|
| 0 | 0.578 | 0.867 | 0.566 | 0.906 | 0.567 | +0.012 | N | 45.1 | 25.7 | 386 | 387 |
| 1 | 0.584 | 0.884 | 0.569 | 0.897 | 0.600 | +0.016 | N | 43.2 | 25.2 | 369 | 369 |
| 2 | 0.566 | 0.863 | 0.562 | 0.873 | 0.569 | +0.003 | N | 42.3 | 25.1 | 384 | 384 |
| 7 | 0.564 | 0.852 | 0.575 | 0.853 | 0.567 | -0.011 | N | 43.1 | 25.3 | 353 | 353 |
| 42 | 0.548 | 0.861 | 0.544 | 0.878 | 0.537 | +0.005 | N | 44.5 | 25.9 | 353 | 353 |

## Aggregate

- mean Most r@1/r@3 = **0.568** / **0.865**
- mean Softmax r@1/r@3 = **0.563** / **0.882**
- mean ZERO r@1/r@3 = **0.568** / **0.881**
- mean ΔSM r@1/r@3 = **+0.005** / **-0.016**
- gate PASS = **0/5**
- mean wall Most/SM/ZERO = 43.6 / 25.4 / 41.0 s
- mean peak RSS Most/SM = 369 / 369 MB

elapsed_total=579s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_074808_mendelejew_baselines_compare`

JSON: `MENDELEJEW_BASELINES_COMPARE_results.json`
