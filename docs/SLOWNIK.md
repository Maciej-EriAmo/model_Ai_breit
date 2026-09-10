# Słownik PhiMost / Most / Φ

| Termin | Znaczenie tu |
|--------|----------------|
| **Most** | Transformer z zamkiem Lorentza \(A=\gamma C/(\gamma^2+\Delta E^2)\) zamiast Softmax |
| **Softmax twin** | Ten sam szkielet sieci, wagi = softmax(QK); tracer ignorowany |
| **ZERO** | Most z τ=0 — **degeneracja bez warunku** (≈ Softmax); nie „fair fight” |
| **content `x`** | Kanał treści (S/E/payload) — semantyka |
| **tracer `τ`** | **Część Mostu** — obserwator, który **gwarantuje prawidłowy most** (warunek \(\Delta E\) / zamek Lorentza). Nie handicap, nie opcjonalny addon. Bez τ → degeneracja (ZERO≈Softmax) |
| **Φ / Phi** | Przestrzeń informacji Karmazyn: atomy, bańki, nie Unix FS |
| **Atom** | \(A=(S,E,T)\) (+ state, bubble, depth, payload?) |
| **HOT/WARM/COLD** | Temperatura / dostępność treści (payload zwykle tylko HOT) |
| **Mazur / mazur_tracer** | Sonda energetyczna (dim 5 w τ przy n_dim=6); też pakiet `mazur_crystal` |
| **\(R\)** | Skalarne \(g V^2/(g^2+dE^2)\) — ranking/retencja produktowa |
| **Λ** | \(\lambda_R R + \lambda_M M\) (MRC); opcjonalnie + λ_neural |
| **EMA-S** | Superpozycja jako EMA wag — **odrzucona** |
| **M1 / M2 / M3** | Wagi NN / przestrzeń Φ / sesja Holon SE |
| **PhiJSON** | Dump `atoms.jsonl` + `edges.jsonl` + manifest |
| **Holon SE** | Pamięć agenta (Karmin_Ae), nie wagi modelu |
| **KarmazynOs** | Runtime przestrzeni Φ (Store, slab, shell) |
| **InfoNCE** | Loss recall (pozytyw = bind same-depth) |
| **MemoryBus** | Fasada ingest→encode→recall (`memory_bus.py`) |
| **Glif / glyph** | Znak pisma jako atom Φ (nie bitmapa); tor G0+ |
| **Transkrypcja** | Zdjęcie/skan → `PhiAtom`+edges (nie trening na RGB ekranu) |
| **Oś +1 / T litery** | Hot↔cold: *czy/kiedy* w oknie Mostu; ≠ XY (`1d_do_2d`) |
| **from_glyphs** | Dump M2 `dumps/from_glyphs` — lat+hira, nauka ze stanu |
