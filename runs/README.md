# `runs/` — odtwarzalne katalogi treningu (kryterium sukcesu #5)

Każdy trening, który woła `runlog.RunDir`, dostaje katalog:

```text
runs/<UTC>_<name>/
  meta.json       # start/stop, git sha, host, torch, status
  command.txt     # pełne argv
  hparams.json    # hiperparametry + CANON
  loss.csv        # epoch,loss,r@1,r@3,mrr (gdy jest krzywa)
  metrics.json    # metryki końcowe / raport
  summary.md      # krótki werdykt
  README.md
```

**Źródło kodu:** `runlog.py`  
**Podpięte:** `phimost_learn.py`, `archiwum/skrypty/run_mendelejew_full.py`  
**Mapa:** [`docs/KRYTERIA_SUKCESU_MAPA.md`](../docs/KRYTERIA_SUKCESU_MAPA.md) — warstwa B, krok 1.

Katalogi runów są w `.gitignore` (artefakty lokalne); ten README zostaje w git.
