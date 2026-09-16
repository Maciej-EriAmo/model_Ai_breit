# Most — parity z Softmaxem (szkic pod CV / portfolio)

**Kontekst:** własna architektura Most (Φ / Bridge / tracer) vs klasyczny Softmax
na zadaniu recall po nie-płaskim grafie pojęć (tablica Mendelejewa, mini-set).

**Wynik (Mendelejew P0, 8 pierwiastków → 72 atomy Φ, 64 epoki, 3 seedy):**

| model | mean r@1 |
|-------|----------|
| Softmax (baseline kanoniczny) | ~0.445 |
| **Most (konstrukcja własna)** | **~0.427** |
| Most + płaski payload | ~0.329 |

- Most **dorównuje** Softmaxowi (Δ ≈ −0.02, w szumie bramki).
- Most na strukturze Φ **wyraźnie bije** płaski tekst (nonflat win 3/3, Δ ≈ +0.10).
- Wniosek aplikacyjny: domowy Most + nie-płaski nośnik = **parity z ustaloną linią badawczą**
  na tym protokole; przewaga wynika ze struktury danych, nie z re-implementacji Softmaxa.

**One-liner (EN):** *Designed a custom Φ-geometry encoder (Most) that matched canonical Softmax recall (~0.43 vs ~0.45) on a structured periodic-table graph while beating flat-text baselines.*

**One-liner (PL):** *Własna architektura Most osiągnęła parity z Softmaxem (~0.43 vs ~0.45) na recallu po grafie tablicy Mendelejewa i wyraźnie przebiła płaski baseline.*

Źródła: `MENDELEJEW_P0.md`, `mendelejew_p0_results.json`.
