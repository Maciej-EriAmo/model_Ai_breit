# Doświadczenie: transkodowanie alfabetów

Hub metody (produkt): [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md)

## Gate vs Softmax-twin (ważne)

| Artefakt | Rola |
|----------|------|
| Softmax `--baselines` | Softmax-twin na **tym samym** dumpie Φ (bez ΔE) |
| `classical_tokens.json` | inwentarz liter/znaków (nie wejście treningu Softmax) |
| Most + ZERO | kandydat + degeneracja τ |

Pełny „Softmax na samej sekwencji liter” = TODO (osobny loader).

## Wyniki lab

- G0/G1: [`WYNIKI_GLYPHS.md`](WYNIKI_GLYPHS.md) · [`WYNIKI_GLYPHS_G1.md`](WYNIKI_GLYPHS_G1.md)
- **G2 parity v2:** [`WYNIKI_GLYPHS_G2.md`](WYNIKI_GLYPHS_G2.md) — PASS best-seed0; seed1 Most&lt;SM (bez claimu multi-seed)

## Testy kontraktu

```bat
cd /d "C:\Users\drwis\model Ai"
python test_transcode_contract.py
python make_dump_glyphs.py
```

## Audyt zewnętrzny (2026-09-11)

Naprawione: glued lat+CJK, classical≠archive, Softmax framing (twin-on-Φ), photo→v2, char focus.  
Review scratch: `grok-review-b7ffadea.md`.

## FUTURE — metoda 2 (nie teraz)

Hologramy liter/kany: składanie hologramów → słowa **lub** znaki pochodne od rdzenia (np. onna/女 + rodzina).  
Kanon: [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md) §7. Po stabilnym multi-seed.
