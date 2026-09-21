# BF16 gate — kryterium sukcesu #2

**Werdykt:** **PASS** — Most w autocast `bfloat16` bez NaN/Inf (finite-check na S/C/ΔE/A/w/loss).

dump=`C:\Users\drwis\model Ai\dumps\from_mendelejew_full` · epochs=16 · seed=0
r@1=0.5781 · r@3=0.8281 · mrr=0.7196
elapsed=42.3s · run-dir=`C:\Users\drwis\model Ai\runs\20260921_073445_bf16_gate`

Lab nadal domyślnie FP32; gate #2 = ścieżka `--bf16` / ten skrypt.

JSON: `BF16_GATE_results.json`
