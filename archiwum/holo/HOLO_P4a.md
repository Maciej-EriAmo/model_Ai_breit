# Holo-Φ P4a — gate lokalny

**Pre-registered:** `blend_holo=0.6` · seeds=[0, 1, 2, 7, 11, 17, 23, 42] · epochs=16
**Kryterium:** PASS ≥ 80% seedów (≥7/8)

blend=0.6 pre-registered z P2; bez fine-tune. To gate lokalny (P4a), nie gate skalowania alfabetów (P4b).

| seed | Most | SM | ZERO | ΔSM | gate |
|-----:|-----:|---:|-----:|----:|:----:|
| 0 | 0.155 | 0.128 | 0.155 | +0.0275 | PASS |
| 1 | 0.152 | 0.142 | 0.128 | +0.0100 | FAIL |
| 2 | 0.147 | 0.135 | 0.115 | +0.0125 | FAIL |
| 7 | 0.117 | 0.098 | 0.102 | +0.0200 | PASS |
| 11 | 0.152 | 0.150 | 0.168 | +0.0025 | FAIL |
| 17 | 0.135 | 0.155 | 0.128 | -0.0200 | FAIL |
| 23 | 0.165 | 0.177 | 0.170 | -0.0125 | FAIL |
| 42 | 0.102 | 0.110 | 0.128 | -0.0075 | FAIL |

**Wynik:** 2/8 PASS · mean ΔSM=+0.0041

**Werdykt P4a:** P4a FAIL: 2/8 (wymagane ≥7/8 = 80%); meanΔSM=+0.0041 best=+0.0275@seed0

JSON: `holo_p4a_results.json` · ckpt: `checkpoints/most_holo_p4a.pt`

