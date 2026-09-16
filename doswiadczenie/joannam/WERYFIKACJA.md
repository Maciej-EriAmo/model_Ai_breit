# JoannaM — weryfikacja (checklist)

**Status kandydata:** anomalia optyczna według Mostu — **nie** potwierdzona czarna dziura.  
**Współrzędne:** RA 120.000° · Dec +5.000° · cutout `desi_grid_004_ra120.0_dec5.0_s128`  
**Cross-match:** [`CROSSMATCH.md`](CROSSMATCH.md) (2026-09-16)

## Zrobione

- [x] Flaga Most: p_cluster ≈ 0.980 (barwa → T), skan pól/grid
- [x] Zapis kandydata + top lista (`JOANNAM_CANDIDATES.json`)
- [x] Dokumentacja wewnętrzna (`JOANNAM.md`)
- [x] Szkic postu EN na forum (`FORUM_EN.md`) — bez overclaim
- [x] Cross-match katalogowy wokół RA120 Dec+5 (SIMBAD / SDSS Spec / DESI DR1 / LS DR10)
- [x] Sprawdzenie obiektów w promieniu cutoutu (fotometria LS: 37 źródeł; spektroskopia: **0**)
- [x] Notatka werdyktu → [`CROSSMATCH.md`](CROSSMATCH.md) + dopisek w `JOANNAM.md`

## Werdykt spektroskopii (2026-09-16)

| Katalog | Promień | Wynik |
|---------|---------|--------|
| SDSS DR17 SpecObj | 5′ | **0** |
| DESI DR1 zpix | ~20′ | **0** |
| SIMBAD | 60″ | **0** nazwanych |
| LS DR10 tractor | 36″ | **37** fotometrycznych (2 jasne gwiazdy Gaia, reszta słaba) |

→ **NO MATCH spektroskopowy.** Brak publicznego z. BH/AGN **niepotwierdzone**.

## X-ray / radio (2026-09-16) — DONE

| Katalog | ≤5′ | Najbliższe |
|---------|----:|------------|
| eROSITA eRASS1 | **0** | 7.4′ (`1eRASS J080025.8+045621`) |
| 2RXS / Chandra CSC | **0** (≤15′) | — |
| NVSS | **0** | ~5.8′ (3.1 mJy) |
| FIRST | **0** (≤15′) | — |

→ **NO MATCH** X-ray/radio w cutoucie. Szczegóły: [`CROSSMATCH.md`](CROSSMATCH.md) §3.

## Ablacja gwiazdy (2026-09-16) — DONE

Maska Gaia @8.8″ (r=3–4 px) → `p_cluster` **0.797 → 0.40 / 0.24** (Δ≈−0.40 / −0.56).  
Kontrole kątowe Δ≈0. **Gwiazda odpowiedzialna za flagę.**  
Szczegóły: [`ABLATION_STAR_MASK.md`](ABLATION_STAR_MASK.md).

## Świeży cutout (2026-09-16) — DONE

Niezależny download Legacy → **sha identyczny** ze starym FITS · p_cluster=0.797 · ablacja Δ=−0.556.  
[`REPRODUCE_FRESH_CUTOUT.md`](REPRODUCE_FRESH_CUTOUT.md).

## Checklist weryfikacji — domknięta

Opcjonalne follow-upy (nieblokujące): nowy spektrum DESI/SDSS gdy się pojawi; eROSITA DR2 gdy publiczne.

## Kryterium uczciwości

Most widzi **światło i geometrię barwy** w cutoucie.  
Potwierdzenie BH wymaga danych poza tym benchiem — spektroskopia publiczna w tej pozycji **nie istnieje** (stan na 2026-09-16).
