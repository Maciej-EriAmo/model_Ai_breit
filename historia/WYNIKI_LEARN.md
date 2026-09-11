# phimost_learn

dump=`dumps\from_holo` atoms=73 seed=42
ΔSM=+0.0031 → **FAIL**
ckpt=`checkpoints\most_holo.pt` promote=False

| seed | Most | SM | ZERO | ΔSM | gate |
|------|------|----|------|-----|------|
| 0 | 0.145 | 0.155 | 0.128 | -0.0094 | FAIL |
| 1 | 0.130 | 0.152 | 0.125 | -0.0219 | FAIL |
| 42 | 0.155 | 0.152 | 0.127 | +0.0031 | FAIL |
