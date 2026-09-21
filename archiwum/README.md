# Archiwum — rzeczy poza silnikiem kanonicznym

Tu leży to, czego **nie trzeba** mieć w root labu na co dzień.
Silnik A–E, `docs/`, `doswiadczenie/`, `dumps/`, `checkpoints/`, `transcode/` zostają w root.

| Katalog | Zawartość |
|---------|-----------|
| [`historia/`](historia/) | Artefakty faz A–E / n-ciał / learn (wyniki, wykresy) |
| [`ZdjęcieW/`](ZdjęcieW/) | Cutouty DESI, kontrakt glifów, skrypty pobierania |
| [`skrypty/`](skrypty/) | Tory boczne: DESI, Holo, glify, lingua/kana, solar, Mendelejew, one-shoty, `run_*` / `bench_*` torów |
| [`dane/`](dane/) | Luźne CSV/JSON wejściowe (`struktura_atomow*`, `solar_names_data.json`) |

## Importy

Skrypty w `skrypty/` dokładają root labu na `sys.path` (bootstrap).
Testy w `doswiadczenie/testy/` widzą `archiwum/skrypty` przez `_labroot.py`.

```powershell
# przykład toru bocznego
python archiwum/skrypty/run_mendelejew_p0.py
python archiwum/skrypty/bench_desi.py

# nauka ze źródła z archiwum
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote
```
