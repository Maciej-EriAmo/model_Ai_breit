# ZdjęcieW — szkice wejścia wizualnego → Φ

Materiały robocze pod tor **zdjęcie → atomy**.  
**Kanoniczne notatki badawcze (dla potomności):** [`../docs/GLYPHS.md`](../docs/GLYPHS.md).

| Plik | Treść |
|------|--------|
| `1d_do_2d.txt` | Kanon: oś litery = T hot/cold, nie bitmapa OCR |
| `GLYPH_ATOMS.md` | Kontrakt pól atomu glifu |
| `demo/demo_lat.png` | Demo G1 — tylko łacina (MOST/BRIDGE) |
| `demo/demo_hira.png` | Demo G1 — tylko hiragana (osobny plik) |
| `demo/desi_*.fits/.png` | Cutouty nieba z `pobieranie.py` |
| `demo/JoannaM.png` | **JoannaM** — nazwany kandydat (RA 120°, Dec +5°) |
| `pobieranie.py` | DESI Legacy → zapis do `demo/` |
| `../historia/JOANNAM.md` | metryka i dedykacja nazwy |

Trening Most na cutoutach z `demo/`:

```bat
cd /d "C:\Users\drwis\model Ai"
python bench_desi.py --demo-only --epochs 20 --augment 8
```

Wynik: `historia/WYNIKI_DESI_DEMO.md`.
| `zdjęcieW.txt` | DESI Legacy — *późniejszy* test geometrii nieba |
| `README.md` | ten plik |

**Nie mylić:** screenshot Lunety ≠ dane treningowe Mostu.  
Nauka: stan atomu → `dumps/from_glyphs` → `phimost_learn`.
