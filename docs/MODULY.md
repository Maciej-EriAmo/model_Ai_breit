# Mapa modułów PhiMost

Root: `C:\Users\drwis\model Ai`

## Silnik i kontrakt danych

| Plik | Rola |
|------|------|
| `config.py` | `CANON` — softplus×2, n_dim=6, bramki, zakaz EMA-S |
| `phi_schema.py` | `PhiAtom` / `PhiGraph` / `atom_content_vec` / `atom_tracer_vec` / `hash_embed` (SHA256) |
| `bridge.py` | `BridgeTransformer` + `SoftmaxTransformer` + `train_eval` (classify) |
| `recall_model.py` | `PhiBridgeEncoder` / `PhiSoftmaxEncoder` + InfoNCE + r@k |
| `scalar_r.py` | skalarne \(R = g V^2/(g^2+dE^2)\) (baseline Mazur-like) |

## Dane / zadania

| Plik | Rola |
|------|------|
| `synthetic_phi.py` | Φ-drzewo lab (Faza A/B) |
| `recall_data.py` | recall syntetyczny topic+depth (Faza C) |
| `nbody_data.py` | solar + three-body (walidacja fizyczna) |
| `make_dump_textfile.py` | budowa dumpa `dumps/textfile_phi` |
| `phi_dump.py` | loader PhiJSON |
| `dump_recall.py` | recall z bind na dumpie |

## Faza E / produkt

| Plik | Rola |
|------|------|
| `phimost_flags.py` | LIVE / GEN / NEURAL_LAMBDA (default OFF) |
| `live_store.py` | snapshot RO: dump / mock / host |
| `gen_head.py` | gen E-embedding warunkowany τ |
| `mazur_neural.py` | blend Λ + hook pod Mazur |

Karmazyn (poza root): `software/mazur_crystal/neural_lambda.py` — szew `relevance()`.

## P1/P2 — pamięć / fasada / adaptery

| Plik | Rola |
|------|------|
| `checkpoint.py` | M1 save/load (`checkpoints/most_dump_recall.pt`) |
| `memory_bus.py` | ingest → encode → recall + export + episodes |
| `phimost_recall.py` | CLI recall (`--id` / `--text` / `--log-episode`) |
| `smoke_checkpoint.py` | train-if-needed → load → recall bez re-treningu Most |
| `phimost_learn.py` | **uniwersalne** load + nauka (Most-only; `--baselines` opc.) |
| `import_struktura_atomow.py` | CSV/JSON konfiguracja elektronowa → Φ |
| `import_literature.py` | `.md`/`.txt` → sekcje/chunki Φ |
| `qm_orbitals_data.py` | Aufbau syntetyczny → Φ |
| `bubblefs_import.py` | BubbleFS `*.bbl` → PhiGraph / PhiJSON |
| `export_karmazyn_live.py` | żywy Store Mazur → BubbleFS → Φ |
| `finetune_*.py` | cienkie wrapery (preferuj `phimost_learn`) |
| `phimost_day.py` | rytuał dnia: export→encode→top-k→epizod |
| `memory_bus.py` | ingest→encode→recall (`exclude_nucleus`) |
| `gen_head.py` / `phimost_gen.py` | gen E produkt |

## Benchy i testy

| Plik | Co odpala |
|------|-----------|
| `test_not_flat.py` | ontologia A |
| `bench_phi.py` | B classify |
| `bench_recall.py` | C recall |
| `bench_dump.py` | D dump (+ zapis ckpt) |
| `test_faza_e.py` / `bench_gen.py` | E |
| `bench_nbody.py` | solar / three |
| `test_mask.py` | mask InfoNCE / attn |
| `test_memory_bus.py` | MemoryBus happy-path |

## Formuła zamka (przypomnienie)

\[
C = \mathrm{softplus}(QK^\top/\sqrt{d})^2,\quad
A_{ij} = \frac{\gamma\, C_{ij}}{\gamma^2 + \Delta E_{ij}^2},\quad
W = \mathrm{rownorm}(A),\ A_{ii}=0
\]

QKV z **content**; \(\Delta E\) z **tracer**.  
**τ = warunek Mostu** (bez niego zamek nie powstaje / degeneruje) — nie handicap vs Softmax.
