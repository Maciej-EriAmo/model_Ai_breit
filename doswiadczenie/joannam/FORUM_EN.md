# Forum post (English) — JoannaM

Ready to paste:

---

**JoannaM** — optical candidate (working name)

This is **not** a confirmed black hole. It is an **optical anomaly flagged by a Most-based model**: the cutout scores as strongly “cluster-like / hot,” not as proof of an event horizon.

| | |
|--|--|
| **RA** | 120.000° |
| **Dec** | +5.000° |
| Cutout | 128×128, `grz` bands |
| Model score | **p_cluster ≈ 0.980** (content = luminance; **T = g−r color**) |
| Scan | field grid; training treated this grid position as “field” |

Survey imaging shows **light** (galaxies / nucleus / structure), not the black hole itself.

**Catalog cross-match (2026-09-16):** no SDSS DR17 spectrum within 5′, no DESI DR1 spectrum within ~20′, nothing in SIMBAD within 60″. Legacy Surveys DR10 shows ~37 photometric sources in the cutout (including a Gaia star at ~8.8″, r≈17.7). **X-ray/radio:** no eROSITA eRASS1 / NVSS source within 5′ (nearest eRASS1 ~7.4′, nearest NVSS ~5.8′). So: **optical anomaly only — no public redshift, no local X-ray/radio counterpart**.

**Star-mask ablation (2026-09-16):** masking the Gaia star at ~8.8″ (r≈17.7) drops `p_cluster` from ~0.80 to ~0.24 (r=4 px); angle-matched controls do not. So the optical flag is largely **driven by that star**, not by an unseen AGN.

**Fresh cutout (2026-09-16):** independent Legacy download is **byte-identical** to the archived FITS; score 0.797 and star-mask Δ≈−0.56 both reproduce.

Happy to share cutout / method details if useful. Critique welcome.

---
