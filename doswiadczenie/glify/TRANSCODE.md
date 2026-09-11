# Doświadczenie: transkodowanie alfabetów

Hub metody (produkt): [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md)

## Gate vs klasyka (ważne)

Rozpoznawanie pisma przez **tokenizację liter + następstwo → NN (Softmax)** to stara, działająca metoda.  
Nowa (Φ + Most + jądra word/char) **musi dorównać** starej na benchu — inaczej jest kaleką.

| Artefakt | Rola |
|----------|------|
| `classical_tokens.json` (po `make_dump_glyphs`) | sekwencja klasyczna |
| Softmax twin | stara metoda |
| Most + ZERO | nowa + kontrola degeneracji |

## Wyniki lab

- G0/G1: [`WYNIKI_GLYPHS.md`](WYNIKI_GLYPHS.md) · [`WYNIKI_GLYPHS_G1.md`](WYNIKI_GLYPHS_G1.md)
- **G2 parity v2:** [`WYNIKI_GLYPHS_G2.md`](WYNIKI_GLYPHS_G2.md) — PASS best-seed0; seed1 Most&lt;SM (bez claimu multi-seed)

## Testy kontraktu

```bat
cd /d "C:\Users\drwis\model Ai"
python test_transcode_contract.py
python make_dump_glyphs.py
```

## FUTURE — metoda 2 (nie teraz)

Hologramy liter/kany: składanie hologramów → słowa **lub** znaki pochodne od rdzenia (np. onna/女 + rodzina).  
Kanon: [`../../docs/TRANSCODE.md`](../../docs/TRANSCODE.md) §7. Po G2 parity.
