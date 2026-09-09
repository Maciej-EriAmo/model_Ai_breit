# Wyniki Faza E — live / gen / λ_neural

## Flagii (domyślne OFF)
- live=False gen=False neural=False

## Live read-only
- source=**dump** atoms=15 read_only=True
- export=`C:\Users\drwis\model Ai\dumps\live_export_e`
- host blocked without LIVE: **True**

## Generative head
- MSE+tracer=0.0001
- MSE ZERO=0.0022
- Δ(ZERO−tr)=+0.0022

- propose_E GEN=1 shape=(8,)
- propose_E GEN=0 blocked: **True**

## λ_neural
- blend OFF=0.5 (assert)
- blend ON=0.5833 (≠0.5)
- mazur.neural_lambda_enabled 0/1 = False/True

**Werdykt:** **PASS**

## Użycie
- `PHIMOST_LIVE=1` + HostAPI → snapshot_from_host
- `PHIMOST_GEN=1` → propose_E
- `PHIMOST_NEURAL_LAMBDA=1` + `PHIMOST_ROOT` → Mazur relevance z N
- Domyślnie OFF — GC/slab nietknięte

