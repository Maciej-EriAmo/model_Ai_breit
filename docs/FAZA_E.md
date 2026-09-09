# Faza E — Live read-only + generative + szew Mazur

**Status:** PASS (smoke/guardy + `mazur_crystal.test_bridge` 6/6 OK)  
**Root:** `C:\Users\drwis\model Ai`

## Cel

1. Live **read-only** Store/dump → Φ (bez tick/heat w modelu).  
2. Opcjonalny **gen head**: propozycja `E` warunkowana tracerem.  
3. Szew **λ_neural** w Mazur `relevance()` — **OFF domyślnie**.

## Flagii (domyślnie wyłączone)

| Env | Default | Znaczenie |
|-----|---------|-----------|
| `PHIMOST_LIVE` | `0` | HostAPI / NativeStore RO |
| `PHIMOST_GEN` | `0` | `propose_E` API |
| `PHIMOST_NEURAL_LAMBDA` | `0` | mix N w Λ |
| `PHIMOST_LAMBDA_NEURAL` | `0.2` | waga N gdy ON |
| `PHIMOST_ROOT` | `C:\Users\drwis\model Ai` | import hooka Mazur |

## Artefakty

| Plik | Rola |
|------|------|
| `phimost_flags.py` | odczyt flag |
| `live_store.py` | dump / mock / host RO + export PhiJSON |
| `gen_head.py` | PhiGenHead + train_gen / propose_E |
| `mazur_neural.py` | blend_lambda, rank_with_neural |
| Karmazyn `mazur_crystal/neural_lambda.py` | product seam |
| `test_faza_e.py`, `bench_gen.py` | testy |

## Acceptance

- [x] flagi off → brak gen API / brak hosta / klasyczne Λ  
- [x] live RO (dump); host zablokowany bez LIVE  
- [x] gen: MSE ZERO ≥ MSE+tracer (smoke); propose_E za GEN  
- [x] docs  
- [x] test_bridge Mazur zielony (GC nietknięte)  

## Uwagi

- `neural_component_for_pair` dziś = hash-E cosine, **nie** encoder Most — nie włączać λ_neural w produkcji, dopóki nie ma checkpointu ([`STATUS.md`](STATUS.md) P2).  
- Trening gen w labie może biec bez `PHIMOST_GEN`; produktowe API wymaga flagi.

## Wyniki

[`../historia/WYNIKI_FAZA_E.md`](../historia/WYNIKI_FAZA_E.md) · Karmazyn: `Documents/MAZUR_CRYSTAL.md` (sekcja PhiMost).
