# Kanon: pamięć i karmienie PhiMost

**Status:** kanon roboczy (po A–E + audycie + n-ciał)  
**Root:** `C:\Users\drwis\model Ai`  
**Silnik:** sieć neuronowa — Most (transformer + zamek Lorentza). Tego **nie przeskakujemy**.

Nawigacja: [`INDEX.md`](INDEX.md) · [`SLOWNIK.md`](SLOWNIK.md) · [`STATUS.md`](STATUS.md) · [`MODULY.md`](MODULY.md)

---

## 0. Teza w jednym akapicie

PhiMost **je przestrzeń Φ** (atomy, bańki, energie), nie płaskie pliki.  
**Myśli** zamkiem Mostu: treść w `content`, geometria w `tracer` (\(\Delta E\)).  
**Pamięć** nie jest jedną bazą — to **warstwy**, z których tylko część to wagi NN.  
Mazur i Holon SE to **osobne toru**; nie wkładać handoffu agenta do checkpointu ani GC Store do InfoNCE.

> Cudze Softmaxy chwalicie — swego Mostu nie znacie: bez \(\Delta E\) Most degeneruje.  
> Bez NN nie ma Mostu. Bez Φ Most nie ma czego obserwować.

### Tracer ≠ handicap — to **warunek** Mostu

Softmax jest uwagi **bez geometrii obserwatora**.  
Most **powstaje** dopiero gdy jest niezerowe \(\Delta E\) z tracera; bez τ most się **nie formuje** albo jest **niestabilny** (degeneracja do Softmax / szumu).

Dlatego:

| Sformułowanie | Status |
|---------------|--------|
| „Most ma unfair advantage bo dostaje tracer” | **Fałsz** — to nie bonus, to definicja zamka |
| „ZERO ≈ Softmax ⇒ Most oszukuje” | **Fałsz** — ZERO pokazuje *brak mostu*, nie handicap Softmaxu |
| Tracer jako warunek tworzenia / stabilności Mostu | **Kanon** |

Kontrola ZERO zostaje w benchach jako **dowód degeneracji bez warunku**, nie jako „fair fight bez geometrii”.

---

## 1. Dlaczego NN (i „coś podobnego”) — bez skoku w bok

| Podejście | Werdykt |
|-----------|---------|
| Czysty Softmax / LLM na tokenach `.txt` | Odrzucone jako **kanon karmienia** (Unix fiction) |
| Skalarne Mazur \(R\) samo | Produkt GC/recall — **nie** zastępuje encodera |
| HRR / hologram bez gradientu | Pole korelacji Φ — **dodatek**, nie zamiennik Mostu |
| **Most NN (softplus×2 + tracer)** | **Kanon silnika** — empiria B/C/n-ciał |
| EMA-S / superpozycja w NN | **Zakaz** (FAIL 2026-09-08/09) |

**Wniosek:** pamięć i karmienie **opieramy o NN Most**, a wokół dokładamy pamięć przestrzenną Φ i produktową Mazur — podobny kształt do „model + store + session”, nie nowa magia zamiast sieci.

```text
        ┌─────────────────────────────────────────┐
        │  Holon SE (agent)  — handoff, 1 work    │  ← nie wagi modelu
        └─────────────────────────────────────────┘
                          │ kontekst sesji
        ┌─────────────────▼───────────────────────┐
        │  PhiMost NN                             │
        │   content x  ←── E / payload / embed    │
        │   tracer  τ  ←── T, depth, state, mazur │
        │   Most lock  A = γC/(γ²+ΔE²)            │
        │   head: recall | classify | (gen E)     │
        └─────────────────┬───────────────────────┘
                          │ score / embedding
        ┌─────────────────▼───────────────────────┐
        │  Przestrzeń Φ (Karmazyn)                │
        │   atomy HOT/WARM/COLD · bańki · bind    │
        │   Mazur R / Λ  (flaga λ_neural OFF)     │
        └─────────────────────────────────────────┘
```

---

## 2. Trzy pamięci (nie mylić)

| Warstwa | Co trzyma | Trwałość | Kto zapisuje | Kanon ścieżki |
|---------|-----------|----------|--------------|---------------|
| **M1 — Wagi Most** | parametry NN (`state_dict`) | checkpoint plik | trening / fine-tune | `checkpoints/` (docelowo) |
| **M2 — Przestrzeń Φ** | atomy `(S,E,T)`, bind, HOT/WARM/COLD | Store / dump PhiJSON | Karmazyn / eksport RO | `dumps/`, live RO |
| **M3 — Sesja agenta** | 1 work + facts `[Most_Lorentz]` | Holon SE | `agent_boot` / `close` | Karmin_Ae — **osobny tor** |

**Zasada:** M1 uczy się **jak patrzeć**; M2 trzyma **na co patrzeć**; M3 mówi agentowi **co robić w sesji**.  
Mieszanie M3→M1 (wklejanie handoffu do loss) albo M1→GC slab — **zakaz**.

Produktowe Mazur (\(R\), MRC, retencja) żyje **przy M2**, nie w M1.  
`PHIMOST_NEURAL_LAMBDA` domyślnie **OFF** — neural score nie psuje Λ, dopóki encoder nie jest checkpointem.

---

## 3. Karmienie — dwa kanały (nie jeden strumień tokenów)

### 3.1 Content `x` — „co jest powiedziane”

| Źródło | Kiedy | Uwagi |
|--------|-------|-------|
| `E` (emanacja) | zawsze | semantyka lokalna |
| `S` (struktura) | zawsze | tożsamość / typ |
| `payload` | tylko **HOT** + sensowny MIME | bajty tekstu — **opcjonalne** |
| embed topic / hash | lab / bootstrap | docelowo: prawdziwe S/E (P1 audytu) |

**Nie wrzucać do content:** `T`, depth, state, mazur — to obserwator.  
(Przeciek hierarchii do `x` zabija przewagę Mostu — patrz bench n-ciał przed poprawką.)

### 3.2 Tracer `τ` — „skąd / jak głęboko / jak gorąco”

Kanon `n_dim=6` (`config.CANON` + `phi_schema.atom_tracer_vec`):

| dim | Oś |
|-----|-----|
| 0–2 | depth / hierarchia (one-hot) |
| 3 | `T` |
| 4 | state (skalar) |
| **5** | **mazur_tracer** |

Bez niezerowego \(\Delta E\) na τ: Most ≈ Softmax (kontrola ZERO **obowiązkowa** w benchach).

### 3.3 Menu według temperatury Φ

| Stan | Dieta content | Dieta tracer | Rola |
|------|---------------|--------------|------|
| **HOT** | S + E + payload | pełny τ | uczenie / recall „żywy” |
| **WARM** | S + E (**bez** payload) | pełny τ | indeks — model **musi** działać |
| **COLD** | S + E skrócone / hash | τ (T niskie) | archiwum; ostrożnie w train |
| **TOMB** | zwykle pomijać | — | GC / retencja Mazur, nie feed NN |

**Sanity kanoniczne:** strip-all-payloads nie może zabijać Mostu, jeśli S/E/τ żyją.

---

## 4. Rytuały (jak karmić w praktyce)

### 4.1 Trening (lab → produkt)

```text
1. Zbierz graf Φ (dump PhiJSON / synteza / n-body)
2. Zbuduj (x, τ, label|pozytyw bind) — osie rozdzielone
3. Softmax twin + Most + ZERO  (ten sam seed)
4. Gate: classify Δ≥+3 pkt | recall r@1 Δ≥+2 pkt
5. Checkpoint → M1  (nie zapisuj do Store w trakcie train)
```

Zakazy treningu:

- EMA-S / Most+S  
- jeden tor „wszystko w tokenach”  
- brak kontroli ZERO  
- mieszanie komór Holon w batche  

### 4.2 Inferencja / recall (MVP produktowe)

```text
1. Snapshot RO przestrzeni (dump lub PHIMOST_LIVE=1)
2. Encoder Most → embeddingi atomów
3. Rank: neural score  ⊕  (opcjonalnie) skalarne R Mazur
4. Wynik: lista id + score — zapis epizodu tylko jeśli polityka M2 na to pozwala
5. Domyślnie: model NIE woła tick/heat/create (GC poza NN)
```

### 4.3 Sesja agenta (M3)

```text
enter Most_Lorentz → agent_boot
  → praca na kodzie / dumpie / benchu
  → remember fact | set-work
  → close (work+fact)
```

Handoff **nie** jest datasetem do CE — najwyżej wskazuje *który dump / które zadanie*.

---

## 5. Co jest „pamięcią długotrwałą” modelu

| Artefakt | Warstwa | Opis |
|----------|---------|------|
| `checkpoints/most_*.pt` | M1 | wagi encodera (+ head recall/gen) |
| `dumps/<name>/` | M2 mirror | PhiJSON: atoms + edges + case |
| Karmazyn Store | M2 live | prawda operacyjna; Most tylko czyta (flaga) |
| `holon_memory` | M3 | ciągłość **agenta**, nie sieci |
| Wykresy / `historia/WYNIKI_*.md` | meta | empiria — karmienie naukowe, nie runtime |

**Epizod vs fakt (w M2, nie w Softmax-stylu chat log):**

- **Epizod:** jeden przebieg recall / sesja bańki (można logować JSONL obok dumpa)  
- **Fakt Φ:** atom z wysokim \(T\) / bind utrwalony / crystallize po stronie Store  
- **Fakt SE:** `[Most_Lorentz] …` w Holon — decyzje badawcze  

NN **nie musi** pamiętać faktów słowami — musi umieć je **odzyskać** z M2 przy dobrym τ.

---

## 6. Gen head (opcjonalnie) — pamięć generatywna

- Warunkowanie **tracerem** (GEN API za `PHIMOST_GEN`)  
- Cel: propozycja `E` / embed, nie klasyczny next-token bez geometrii  
- Nie zastępuje M2: wygenerowane `E` wraca do Φ tylko przez **świadomy zapis** (osobna polityka, nie default)

---

## 7. Interfejs docelowy (kontrakt, jeszcze nie cały kod)

```text
MemoryBus  (memory_bus.py — zaimplementowane)
  ingest(snapshot: PhiGraph | dump) -> Batch(x, τ)
  encode(batch) -> H          # Most NN z checkpoint
  recall(q, k) -> [(id, score)]
  export_dump(path)           # RO mirror M2
  append_episode(jsonl)       # M2 mirror, nie Holon
```

CLI: `python phimost_recall.py --id __main__ -k 5`.  
Load M1: `checkpoint.load_encoder`.  
**MemoryBus** = sklejenie kanonu, nie nowa ontologia.

---

## 8. Checklist zgodności (przed każdym większym PR)

- [ ] Content ≠ tracer (brak przecieku T/depth/mazur do `x`)  
- [ ] Bench ma Softmax + ZERO  
- [ ] WARM bez payload w diecie testowej  
- [ ] EMA-S nie wraca  
- [ ] M1/M2/M3 nie zlane w jeden plik  
- [ ] Live bez mutacji Store przy flagach OFF  
- [ ] Zmiana `CANON` ⇒ nowy Softmax|Most|ZERO  

---

## 9. Następne kroki implementacyjne (po tym dokumencie)

1. ~~P1 dump `atom_content_vec`~~  
2. ~~`checkpoints/`~~  
3. ~~`MemoryBus`~~  
4. ~~JSONL epizodów~~  
5. ~~λ_neural + BubbleFS + dekorelacja C + gen produkt~~ (P2 zamknięte)  

---

## 10. Jedno zdanie na lodówkę

**Karmimy Most osiami Φ; pamiętamy w trzech szufladach (wagi · przestrzeń · sesja); Softmax zostaje świadkiem, nie kapłanem.**

---

*Kanon spisany po fokusie na wynikach — żeby wyniki miały gdzie wracać.*
