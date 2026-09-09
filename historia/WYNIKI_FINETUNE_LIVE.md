# Fine-tune Most — Karmazyn live Φ (seed sweep)

**Dump:** `C:\Users\drwis\model Ai\dumps\from_struktura` atoms=72
**pair_mode:** `mixed`  noise=0.05  epochs=16
**Warm-start:** False
**Wybrany seed:** 0

| seed | Most r@1 | SM r@1 | ZERO r@1 | ΔSM | gate |
|------|----------|--------|----------|-----|------|
| 0 | 0.439 | 0.425 | 0.428 | +0.0141 | FAIL |
| 1 | 0.441 | 0.441 | 0.439 | +0.0000 | FAIL |
| 7 | 0.453 | 0.458 | 0.438 | -0.0047 | FAIL |
| 42 | 0.436 | 0.453 | 0.448 | -0.0172 | FAIL |

**Werdykt:** **FAIL**  ΔSM=+0.0141  clean r@1=0.438
**Ckpt:** `C:\Users\drwis\model Ai\checkpoints\most_struktura_atomow.pt` (bez promote)
