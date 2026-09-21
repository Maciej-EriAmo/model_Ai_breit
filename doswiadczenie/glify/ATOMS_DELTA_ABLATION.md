# Atomy FULL118 — ablacja ΔE (square / directed / onesided / ultra_wide)

dump=`C:\Users\drwis\model Ai\dumps\from_mendelejew_full` · epochs=16 · seeds=[0, 1]

**Werdykt:** PARTIAL: **onesided** lepszy niż square na FULL118 (r@1 0.591 vs 0.567, ΔSM +0.012 vs -0.012); gate vs SM: 1/2.

## Aggregate

Softmax mean r@1/r@3 = **0.579** / **0.891**

| variant | r@1 | r@3 | ΔSM r@1 | ΔSM r@3 | gate | wall s |
|---------|----:|----:|--------:|--------:|-----:|-------:|
| softmax | 0.579 | 0.891 | — | — | — | — |
| square | 0.567 | 0.875 | -0.012 | -0.016 | 0/2 | 44.3 |
| directed | 0.577 | 0.866 | -0.002 | -0.026 | 0/2 | 51.8 |
| onesided | 0.591 | 0.896 | +0.012 | +0.005 | 1/2 | 64.6 |
| ultra_wide | 0.584 | 0.886 | +0.005 | -0.005 | 0/2 | 57.0 |
| ultra_wide_directed | 0.584 | 0.887 | +0.005 | -0.004 | 0/2 | 59.2 |

## Per seed

### seed 0
SM r@1=0.589 r@3=0.886
- square: r@1=0.550 r@3=0.866 ΔSM=-0.039
- directed: r@1=0.572 r@3=0.855 ΔSM=-0.017
- onesided: r@1=0.586 r@3=0.883 ΔSM=-0.003
- ultra_wide: r@1=0.592 r@3=0.880 ΔSM=+0.003
- ultra_wide_directed: r@1=0.594 r@3=0.881 ΔSM=+0.005

### seed 1
SM r@1=0.569 r@3=0.897
- square: r@1=0.584 r@3=0.884 ΔSM=+0.016
- directed: r@1=0.583 r@3=0.877 ΔSM=+0.014
- onesided: r@1=0.595 r@3=0.909 ΔSM=+0.027
- ultra_wide: r@1=0.577 r@3=0.892 ΔSM=+0.008
- ultra_wide_directed: r@1=0.575 r@3=0.894 ΔSM=+0.006

elapsed=630s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_093754_atoms_delta_ablation`

JSON: `ATOMS_DELTA_ABLATION_results.json`
