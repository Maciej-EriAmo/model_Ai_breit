# Doświadczenie — hub empirii terenowej / torów bocznych

**Root lab:** `C:\Users\drwis\model Ai`  
**Komora SE:** `Most_Lorentz`  
**Rola:** osobny katalog na **doświadczenia** (niebo, JoannaM, glify, Mendelejew), żeby nie mieszać z kanonem silnika A–E w `docs/` + `archiwum/historia/`.  
**Aktualizacja:** 2026-09-16

| Tu (`doswiadczenie/`) | Nie tu |
|------------------------|--------|
| Wyniki i notatki torów empirycznych | Kanon Most / Φ (`docs/PAMIEC_I_KARMIENIE.md`) |
| Kandydaci / skany / audyty toru | Fazy A–E + n-ciał (`archiwum/historia/WYNIKI_FAZA_*`) |
| Testy regresji (`testy/`) | Runtime ckpt (`checkpoints/`), dumpy (`dumps/`) |
| Checklist kompletności docs | Silnik `.py` w root labu |

## Mapa

| Tor | Katalog | Status (skrót) |
|-----|---------|----------------|
| **JoannaM** | [`joannam/`](joannam/) | RA120 Dec+5; weryfikacja **DOMKNIĘTA** (flaga=gwiazda); [`CROSSMATCH.md`](joannam/CROSSMATCH.md) |
| **Asia** | [`joannam/ASIA.md`](joannam/ASIA.md) | RA142.5 Dec+6.5; p=0.974→0.80 po masce; [`CROSSMATCH_C2.md`](joannam/CROSSMATCH_C2.md) |
| **DESI / niebo** | [`desi/`](desi/) | bench + holdout + BH/AGN + audyt; **zawieszone** (gate FAIL) |
| **Glify / transkodowanie** | [`glify/`](glify/) | G0+G1 PASS; G2 best-seed; kanon [`docs/TRANSCODE.md`](../docs/TRANSCODE.md) |
| **Holo-Φ** | [`glify/`](glify/) (`HOLO_*`) | P5.1 **FAIL** holdout 0/8 |
| **Lingua / hologram-znaki** | [`glify/`](glify/) + `karmazyn_lingua.py` | warstwa **DONE**; Kana P1 negatyw |
| **Mendelejew** | [`glify/`](glify/) (`MENDELEJEW_*`) | P0 parity · FULL118 **DONE** · hold-out Na/K **FAIL** (brak gen. grupy) |
| **Testy** | [`testy/`](testy/) | regresje silnika / kontraktów (`test_*.py`) |

Mapa plików glify: [`glify/README.md`](glify/README.md).

## Wejście docs

1. Ten plik (hub doświadczenia)  
2. [`CHECKLIST.md`](CHECKLIST.md) — co jest / czego brakuje  
3. [`docs/STATUS.md`](../docs/STATUS.md) — stan labu + kolejka  
4. [`docs/INDEX.md`](../docs/INDEX.md) — indeks całego labu  

## Zakazy

- Nie mylić cutoutów DESI z fundamentem alfabetu (glify).  
- Nie nazywać JoannyM „odkrytą czarną dziurą” bez weryfikacji katalogowej / spektroskopii.  
- Nie mylić **lingua-znaki** (Store+HRR) z **Holo-Φ** (τ Unihan) ani z **Mendelejew** (struktura Φ).  
- Ciężkie FITS/PNG zostają lokalnie w `archiwum/ZdjęcieW/demo/` (gitignore) — tu tylko metadane i wyniki.  
- Softmax w `--baselines` = twin-on-Φ, dopóki nie ma classical loadera (TRANSCODE §2).
