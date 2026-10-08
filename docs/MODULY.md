# Mapa modułów PhiMost

Root: `C:\Users\drwis\model Ai`

## Silnik i kontrakt danych

| Plik | Rola |
|------|------|
| `config.py` | `CANON` — softplus×2, n_dim=6, bramki, zakaz EMA-S |
| `breit_kernel.py` | wzór Most-1: `breit_wigner_rows`, `breit_wigner_weights`. Dostaje gotowe `C` i `ΔE` |
| `phi_schema.py` | `PhiAtom` / `PhiGraph` / `atom_content_vec` / `atom_tracer_vec` / `hash_embed` (SHA256) |
| `bridge.py` | szkielet: `BridgeTransformer` + `SoftmaxTransformer` + `train_eval` (classify) |
| `recall_model.py` | szkielet: `PhiBridgeEncoder` / `PhiSoftmaxEncoder` + InfoNCE + r@k |
| `lm_tiny.py` | szkielet językowy: `_MostCausalBlock` / `_SoftmaxCausalBlock` w `TinyLM` |
| `scalar_r.py` | skalarne \(R = g V^2/(g^2+dE^2)\) (baseline Mazur-like) |

Wzór Most-1 jest w `breit_kernel.py`. Trzy szkielety wołają go z `bridge.py`, `recall_model.py` i `lm_tiny.py`. Seal, `most_seal.py`, jest w tabeli poniżej.

## Dane / zadania

| Plik | Rola |
|------|------|
| `synthetic_phi.py` | Φ-drzewo lab (Faza A/B) |
| `recall_data.py` | recall syntetyczny topic+depth (Faza C) |
| `nbody_data.py` | solar + three-body (walidacja fizyczna) |
| `make_dump_textfile.py` | budowa dumpa `dumps/textfile_phi` |
| `most_seal.py` | uszczelnienie Most: hard mask T + kara COLD na kluczach |
| `transcode/` | **transkodowanie alfabetów** → Φ (`glyph_unit_v2`) |
| `doswiadczenie/testy/test_transcode_contract.py` | kontrakt lat=słowo / CJK=znak + classical tokens |
| `phi_dump.py` | loader PhiJSON |
| `dump_recall.py` | recall z bind na dumpie |

Tory boczne (glify / Holo / DESI / lingua / Mendelejew / `run_*`): [`../archiwum/skrypty/`](../archiwum/skrypty/) — mapa: [`../archiwum/README.md`](../archiwum/README.md).

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
| `phimost_learn.py` | **uniwersalne** load + nauka (Most-only; `--baselines` opc.) + run-dir |
| `runlog.py` | kryterium #5 — `runs/<UTC>_<name>/` (meta, hparams, loss.csv) |
| `precision.py` | kryterium #2 — AMP bf16 + `assert_finite` na zamku |
| `lm_tiny.py` | kryterium #1b — tiny causal LM (Most \| Softmax attn) |
| `import_struktura_atomow.py` | CSV/JSON konfiguracja elektronowa → Φ |
| `import_literature.py` | `.md`/`.txt` → sekcje/chunki Φ |
| `qm_orbitals_data.py` | Aufbau syntetyczny → Φ |
| `bubblefs_import.py` | BubbleFS `*.bbl` → PhiGraph / PhiJSON |
| `phimost_day.py` | rytuał dnia: export→encode→top-k→epizod |
| `memory_bus.py` | ingest→encode→recall (`exclude_nucleus`) |
| `gen_head.py` / `phimost_gen.py` | gen E produkt |

## Glify / alfabety

| Plik | Rola |
|------|------|
| `docs/GLYPHS.md` | notatki badawcze (teza, metoda, G0, roadmap) |
| `archiwum/ZdjęcieW/` | szkice osi T + kontrakt atomu + DESI notes |
| `archiwum/skrypty/bench_desi.py` | smoke Softmax\|Most\|ZERO na cutoutach (opc.) |

## Benchy i testy

| Plik | Co odpala |
|------|-----------|
| `doswiadczenie/testy/test_not_flat.py` | ontologia A |
| `bench_phi.py` | B classify |
| `bench_recall.py` | C recall |
| `bench_dump.py` | D dump (+ zapis ckpt) |
| `doswiadczenie/testy/test_faza_e.py` / `bench_gen.py` | E |
| `bench_nbody.py` | solar / three |
| `doswiadczenie/testy/test_mask.py` | mask InfoNCE / attn |
| `doswiadczenie/testy/test_memory_bus.py` | MemoryBus happy-path |

## Formuła zamka (przypomnienie)

\[
C = \mathrm{softplus}(QK^\top/\sqrt{d})^2,\quad
A_{ij} = \frac{\gamma\, C_{ij}}{\gamma^2 + \Delta E_{ij}^2},\quad
W = \mathrm{rownorm}(A),\ A_{ii}=0
\]

QKV z **content**; \(\Delta E\) z **tracer**.  
**τ = warunek Mostu** (bez niego zamek nie powstaje / degeneruje) — nie handicap vs Softmax.
