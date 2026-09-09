# Nauka Most — uniwersalny tor

**CLI:** `phimost_learn.py` (PowerShell-friendly).

## Szybki start

```powershell
cd "C:\Users\drwis\model Ai"

# struktura atomów (CSV/JSON)
python phimost_learn.py --src struktura_atomow.csv --no-warm --no-promote

# literatura (.md / .txt / katalog)
python phimost_learn.py --src .\rozdzial.md --no-warm --no-promote

# gotowy dump PhiJSON
python phimost_learn.py --src dumps/from_struktura --skip-load --no-warm --no-promote

# recall bez treningu
python phimost_learn.py --src dumps/from_struktura --skip-load --skip-train --recall-id C/2p/m-1/up
```

## Presety

| `--preset` | Źródło |
|------------|--------|
| `struktura` | `struktura_atomow.json` / `.csv` |
| `karmazyn` | żywy Store Mazur → BubbleFS → Φ |
| `qm` | Aufbau syntetyczny |
| `textfile` | lab `dumps/textfile_phi` |
| `literature` | `.md`/`.txt` z root projektu |

## Tryby treningu

| Flaga | Zachowanie |
|-------|------------|
| *(domyślnie)* | **tylko Most** — wybór seeda po `r@1` |
| `--baselines` | Most + Softmax + ZERO + gate ΔSM (≥ +2 pkt) — **3× wolniej** |
| `--no-promote` | nie nadpisuj `most_dump_recall.pt` |
| `--promote` | przy PASS (lub fast OK) skopiuj do default ckpt |

## Mapowanie danych → Φ

| Źródło | Atom | Tracer / bind |
|--------|------|----------------|
| Struktura atomów | orbital (n,ℓ,m,spin) | depth, T, mazur=ℓ; bind=podpowłoka |
| Literatura | sekcja / akapit | depth=nagłówek; bind=outline |
| Karmazyn | atom Store | T, mazur holo; bind=hologram |
| BubbleFS | `.bbl` | meta depth/T; bind=`.hgm` |

## Recall

- `same_prefix=True` — tylko ten sam pierwiastek / doc (w CLI learn)
- `exclude_nucleus=True` (domyślnie) — bez `*/nucleus` w top-k orbitali

## Warstwy pamięci (nie mieszać)

- **M1** = `checkpoints/*.pt`
- **M2** = dump Φ (`dumps/…`)
- **M3** = Holon SE (`Most_Lorentz`) — sesja agenta, nie wagi
- **JSONL** `episodes.jsonl` — ślad, nie baza

Zob. [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md).
