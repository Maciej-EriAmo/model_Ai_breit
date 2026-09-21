# Wyniki Faza D — offline dump Φ

**Dump:** `dumps/textfile_phi` (Plik tekstowy jako przestrzeń Φ (nie Unix .txt))
Atoms=15, bind pairs=23.
Trening na zaszumionych kopiach topologii dumpa; ewal na holdout.

| wariant | r@1 | r@3 | r@5 | MRR |
|---------|-----|-----|-----|-----|
| Softmax | 0.388 | 0.907 | 1.000 | 0.633 |
| Most | 0.410 | 0.890 | 1.000 | 0.643 |
| Most ZERO | 0.420 | 0.890 | 1.000 | 0.646 |
| flat payload | 0.380 | 0.785 | 0.930 | 0.597 |
| Most no-payload | 0.125 | 0.750 | 1.000 | 0.439 |
| Most clean-batch | 0.312 | 0.938 | 1.000 | 0.565 |
| R (depth) | 0.287 | 0.525 | 0.718 | 0.469 |
| cosine | 0.220 | 0.492 | 0.605 | 0.418 |

**Δ Most−Softmax r@1** = +0.0225 → PASS
**Δ Most−R r@1** = +0.1225 → PASS
**Werdykt:** **PASS**

## Sanity / uwagi
- HOT=3 / WARM=10 / COLD=2 w dumpie; WARM bez payload.
- **flat payload** gorszy niż Most → Unix fiction nie wystarcza.
- **Most ZERO ≈ Most** na tym małym dumpie (N=15) — przewaga tracera słabsza niż w Fazie C; gate i tak Most>Softmax i Most>R.
- `Most no-payload` / `clean-batch` = diagnostyka single-batch (wyższa wariancja); werdykt na holdout val.

## Case study
- `dumps/textfile_phi/CASE_TEXT.md`
- `dumps/textfile_phi/CASE_RUNTIME.md`

## Go / no-go Faza E
GO: live read-only Store opcjonalnie; generative head później.
