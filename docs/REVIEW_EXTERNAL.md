## Summary

PhiMost is a coherent research lab: softplus×2 Lorentz Most, Softmax/ZERO twins, Φ schema, recall/dump/n-body benches, and Phase-E flags default OFF. The core attention lock in `bridge.py` / `recall_model.py` matches the stated canon and the Softmax twin correctly ignores the tracer. Dominant risks are physical correctness in solar n-body kinematics (giants vs Moon branch), a dump “no-payload” ablation that does not change tensors, and the Mazur neural hook that gates on `_ENCODER` but never runs it. Phase-E live host I/O also fails open to an empty graph.

## Issues

### Issue 1 -- Severity: bug
- File: C:\Users\drwis\model Ai\nbody_data.py:102
- Description: `elif hier == 2` is labeled “Moon: near Earth phase”, but hierarchy 2 is **giant** (Jupiter…Neptune). The Moon is appended with `hier=1` (`nbody_data.py:83`), so it takes the normal Kepler branch at `a=1.0` AU. Giants are incorrectly placed as tiny Earth satellites; the Moon is not. Solar kinematics/energies used for content and tracer dim3 are therefore wrong for the claimed solar-system task (reported +20.6 pkt may still pass via hierarchy one-hots, but it is not the intended physics).
- Suggestion: Branch on `name == "Moon"` (or a dedicated hierarchy/flag) for the Earth-offset path; send `hier == 2` through `_kepler_state(a, mass, phase)` like the rocky planets.
- Status: open

### Issue 2 -- Severity: bug
- File: C:\Users\drwis\model Ai\bench_dump.py:85
- Description: `eval_no_payload_most` calls `strip_all_payloads` then `make_recall_from_graph`. Dump recall content is built only from `topic_content(topics[i], atom_id, …)` in `dump_recall.graph_to_tensors` (lines 107–108); payload/S/E are never read into content or tracer. Stripping payloads is a no-op on the tensors. The reported `Most no-payload` r@1≈0.125 is from a single noisy `seed=42` batch eval on the trained model, not from ablating payloads — yet `WYNIKI_FAZA_D.md` presents it as a payload-strip diagnostic.
- Suggestion: Either build dump content from `atom_content_vec` (S/E/payload) so strip matters, or remove/rename the metric and evaluate on the same val loader as Most. Do not attribute single-batch variance to payload removal.
- Status: open

### Issue 3 -- Severity: bug
- File: C:\Users\drwis\model Ai\mazur_neural.py:131
- Description: `neural_component_for_pair` returns `None` if `_ENCODER is None`, but when the encoder **is** set it still computes cosine of `hash_embed(E)` and never calls `_ENCODER` / `neural_scores`. With `PHIMOST_NEURAL_LAMBDA=1`, the Mazur product hook does not use Most at all — only a hash baseline — while `rank_with_neural` correctly uses the encoder. Flag-on behavior is inconsistent and misrepresents “λ_neural from N”.
- Suggestion: When `_ENCODER` is set, score the pair via the encoder (or document that the hook is hash-only and stop requiring `_ENCODER`). Prefer one code path with `neural_scores` / a two-atom graph.
- Status: open

### Issue 4 -- Severity: bug
- File: C:\Users\drwis\model Ai\live_store.py:122
- Description: Under `PHIMOST_LIVE=1`, `snapshot_from_host` swallows all exceptions from `host.list_atoms`, falls through, and returns a successful `LiveSnapshot` with `atoms=[]` and no edges if both host entry points fail or return empty. Callers cannot distinguish “Store empty” from “API broken / wrong host”.
- Suggestion: Re-raise (or wrap) failures from `list_atoms`; if both paths yield no atoms, raise `RuntimeError` with context rather than returning an empty graph silently.
- Status: open

### Issue 5 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\nbody_data.py:173
- Description: Docstring still says the three-body label is the escaper index (max specific energy). Implementation at lines 270–278 uses intra-layer content-sign interactions (like the Φ-tree). `bench_nbody.py:760` repeats the stale “uciekinier” description while `WYNIKI_NBODY.md` correctly describes layers. Easy to misread the bench.
- Suggestion: Update the module docstring and `bench_nbody` header to match the layer-interaction label; keep escaper energy only as tracer/meta.
- Status: open

### Issue 6 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\nbody_data.py:289
- Description: Tracer dim5 is set from `content[b, i, 0]`, leaking a content channel into the observer that Softmax ignores. That weakens the Softmax|Most|ZERO isolation story for the three-body task (Most can lock on a content-correlated ΔE axis).
- Suggestion: Fill dim5 from mass/rank/mazur-like quantities already computed (e.g. normalized `log m`), never from `content`.
- Status: open

### Issue 7 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\phi_dump.py:71
- Description: `load_atoms_jsonl` mutates `E` by prefixing `[topic:…]`, while `dump_recall.load_raw_records` + `records_to_graph` keep raw `E` and use a separate `topic` list for content. `live_store.snapshot_from_dump` → `load_dump` takes the mutated path; benches take the raw path. Same dump yields different content embeddings depending on loader.
- Suggestion: Keep `E` pristine in the loader; pass `topic` as metadata (or a dedicated field) and let recall/live choose explicitly how to embed.
- Status: open

### Issue 8 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\phimost_flags.py:33
- Description: Default `PHIMOST_ROOT` is a machine-specific `C:\Users\drwis\model Ai`. `bench_gen.py:588` and `test_faza_e.py:100` also hardcode `KarmazynOs\software` on `sys.path`. Fresh clones / other machines break dump discovery and Mazur imports unless env vars are set first — a silent DX footgun for the stated `cd … & python test_*.py` flow.
- Suggestion: Default `phimost_root()` to `Path(__file__).resolve().parent`; resolve Karmazyn via `HOLON_KARMIN_PATH` / `PHIMOST_KARMAZYN_SOFTWARE` (or skip the external import when absent).
- Status: open

### Issue 9 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\recall_model.py:16
- Description: `PhiBridgeEncoder` / `PhiSoftmaxEncoder` duplicate the Lorentz/Softmax stack from `bridge.py` (`BridgeTransformer` / `SoftmaxTransformer`) with only head vs token-output differing. Formula or masking fixes must be applied twice; drift risk is real (already two copies of softplus×2 + diagonal zeroing).
- Suggestion: Share one attention backbone (factory or base module) and swap classification head vs token export.
- Status: open

### Issue 10 -- Severity: suggestion
- File: C:\Users\drwis\model Ai\bridge.py:99
- Description: Sequence `mask` from `PhiGraphBatch` / `RecallBatch` is never applied in attention. Harmless while `pad_to == N0`, but any real padding attends as fake atoms and pollutes ΔE / Softmax weights.
- Suggestion: Mask padded keys (and queries) with zeros / `-inf` before normalizing `A` or `softmax`, using `batch.mask`.
- Status: open

### Issue 11 -- Severity: nit
- File: C:\Users\drwis\model Ai\gen_head.py:75
- Description: Dead `if not gen_enabled(): pass` plus a comment that narrates product-vs-lab policy instead of a constraint the next reader cannot see from `propose_E`.
- Suggestion: Delete the no-op branch; keep the guard only on `propose_E` (already correct).
- Status: open

### Issue 12 -- Severity: nit
- File: C:\Users\drwis\model Ai\bridge.py:163
- Description: Softmax twin applies `softmax` then zeros the diagonal and renormalizes; Most zeros the diagonal on `A` before normalize. Equivalent in spirit, but not the usual “mask before softmax” twin — small numerical asymmetry in the control.
- Suggestion: Prefer `masked_fill(-inf)` on the diagonal before `softmax` for a cleaner Softmax twin, or note the intentional post-mask renorm next to the Most path.
- Status: open
