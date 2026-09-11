# ZdjęcieW — szkice wejścia wizualnego → Φ

Materiały robocze pod tor **zdjęcie → atomy** oraz cutouty nieba.  
**Kanoniczne notatki badawcze (glify):** [`../docs/GLYPHS.md`](../docs/GLYPHS.md).  
**Hub doświadczeń:** [`../doswiadczenie/README.md`](../doswiadczenie/README.md).

| Plik | Treść |
|------|--------|
| `1d_do_2d.txt` | Kanon: oś litery = T hot/cold, nie bitmapa OCR |
| `GLYPH_ATOMS.md` | Kontrakt pól atomu glifu |
| `demo/demo_lat.png` | Demo G1 — tylko łacina (MOST/BRIDGE) |
| `demo/demo_hira.png` | Demo G1 — tylko hiragana (osobny plik) |
| `demo/desi_*.fits/.png` | Cutouty nieba (lokalnie; gitignore FITS/PNG) |
| `demo/JoannaM.png` | **JoannaM** — nazwany kandydat (RA 120°, Dec +5°) |
| `pobieranie.py` | DESI Legacy → zapis do `demo/` |
| `zdjęcieW.txt` | DESI Legacy — *późniejszy* test geometrii nieba |
| [`../doswiadczenie/joannam/JOANNAM.md`](../doswiadczenie/joannam/JOANNAM.md) | metryka i dedykacja nazwy |

Trening Most na cutoutach z `demo/`:

```bat
cd /d "C:\Users\drwis\model Ai"
python bench_desi.py --demo-only --epochs 20 --augment 8
```

Wynik: `doswiadczenie/desi/WYNIKI_DESI_DEMO.md`.

**Nie mylić:** screenshot Lunety ≠ dane treningowe Mostu.  
Nauka glifów: stan atomu → `dumps/from_glyphs` → `phimost_learn`.
