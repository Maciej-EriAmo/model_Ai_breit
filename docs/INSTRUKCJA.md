# Instrukcja obsługi PhiMost

**Dla kogo:** Ty (operator) + agent z komorą `Most_Lorentz`.  
**Gdzie:** `C:\Users\drwis\model Ai`  
**Shell:** PowerShell (nie używaj `REM` z cmd).

---

## 1. Co to jest (jednym zdaniem)

**Most Lorentza** to transformer, w którym uwaga ma zamek  
\(A = \gamma C / (\gamma^2 + \Delta E^2)\) — \(\Delta E\) bierze się z **tracera** (obserwatora Φ), nie z treści.

Uczysz **encodera do recall** (ranking atomów), nie pełnego chat-LLM.

---

## 2. Trzy pamięci (nie mieszać)

| Warstwa | Co | Przykład |
|---------|-----|----------|
| **M1** | wagi sieci | `checkpoints/*.pt` |
| **M2** | przestrzeń Φ (dane) | `dumps/from_struktura` |
| **M3** | sesja agenta | Holon / Karmin_Ae, projekt `Most_Lorentz` |

JSONL `episodes.jsonl` = **ślad**, nie baza wiedzy.

---

## 3. Codzienna obsługa

### 3.1 Nauka z Twoich danych

```powershell
cd "C:\Users\drwis\model_Ai_breit"

# struktura atomów
python phimost_learn.py --src archiwum/dane/struktura_atomow.csv --no-warm --no-promote

# literatura
python phimost_learn.py --src .\moja_ksiazka.md --no-warm --no-promote

# katalog .md/.txt
python phimost_learn.py --src .\literatura\ --no-warm --no-promote
```

- **Domyślnie:** trenuje tylko **Most** (szybko).  
- **Gate vs Softmax:** dodaj `--baselines` (3× wolniej).  
- **Bez `--promote`:** nie nadpisuje ckpt dnia (`most_dump_recall.pt`).

### 3.2 Recall (bez treningu)

```powershell
python phimost_learn.py --src dumps/from_struktura --skip-load --skip-train --recall-id C/2p/m-1/up
```

- Ranking w obrębie pierwiastka / dokumentu (`same_prefix`).  
- Orbital: **bez** `*/nucleus` w top-k (`exclude_nucleus`).

### 3.3 Rytuał dnia (Karmazyn live)

```powershell
python phimost_day.py --query main -k 5
# bez ponownego eksportu Store:
python phimost_day.py --query main -k 5 --skip-export
```

### 3.4 Presety

```powershell
python phimost_learn.py --preset struktura --load-only
python phimost_learn.py --preset karmazyn --no-warm --no-promote
python phimost_learn.py --preset qm --no-warm --no-promote
python phimost_learn.py --preset literature --load-only
python phimost_learn.py --preset textfile --no-warm --no-promote
```

---

## 4. Mapa plików (operator)

| Chcę… | Komenda / plik |
|-------|----------------|
| Wszystko o nauce | `docs/LEARN.md` |
| Kanon pamięci | `docs/PAMIEC_I_KARMIENIE.md` |
| Porównanie z LLM | `docs/POROWNANIE_MODELE.md` |
| Status | `docs/STATUS.md` |
| Wyniki faz | `docs/WYNIKI.md`, `archiwum/historia/` |
| Konfiguracja | `config.py`, `phimost_flags.py` |

---

## 5. Flagi produktowe (OFF = bezpiecznie)

| Env | Default | Znaczenie |
|-----|---------|-----------|
| `PHIMOST_LIVE` | 0 | live Store RO |
| `PHIMOST_GEN` | 0 | generative head |
| `PHIMOST_NEURAL_LAMBDA` | 0 | Most w Mazur Λ |

Nie włączaj na produkcji bez ckpt i testu.

---

## 6. Checkpointy (lokalne)

| Plik | Po co |
|------|--------|
| `most_dump_recall.pt` | default / DAY |
| `most_struktura_atomow.pt` | CSV/JSON atomów |
| `most_karmazyn_live.pt` | Store live |
| `most_literature.pt` | tekst |
| `most_qm_orbitals.pt` | Aufbau lab |
| `most_gen_head.pt` | propose E |

Leżą w `checkpoints/` (gitignore).

---

## 7. Diagnostyka

```powershell
python doswiadczenie/testy/test_not_flat.py
python doswiadczenie/testy/test_mask.py
python doswiadczenie/testy/test_memory_bus.py
python doswiadczenie/testy/test_decorrelate.py
python doswiadczenie/testy/test_bubblefs_import.py
python doswiadczenie/testy/test_review_fixes.py
```

Mazur (KarmazynOs):

```powershell
cd C:\Users\drwis\KarmazynOs
$env:PYTHONPATH="archiwum\kernel_python;software;native;."
$env:KARMAZYN_SUBSTRATE="python"
$env:KARMAZYN_MAZUR="1"
python -m mazur_crystal.test_bridge
```

---

## 8. Czego nie robić

1. Mieszać Holon facts z wagami Mostu.  
2. Uczyć „płaskiego tekstu” bez Φ (S/E/T + tracer).  
3. Traktować Softmax-only jako cel (to świadek).  
4. Kasować `holon_memory.json` / dane SE bez prośby.  
5. Commitować `*.pt` i `dumps/`.

---

## 9. Agent (Karmin_Ae)

```powershell
cd C:\Users\drwis\Karmin_Ae
python agent_boot.py --project Most_Lorentz
```

Potem praca w `model Ai`. Na koniec sesji: `close` w Holonie.
