# PhiMost a inne modele — budowa i nauka

**Pytanie:** jeśli chcę się nauczyć *jak robić modele*, czy lepiej na własnym (PhiMost), czy na cudzym (GPT/Transformer tutorial)?

**Odpowiedź krótka:** **lepiej na własnym torze**, mając obok Softmax jako świadka. Cudze LLM uczą Cię *używania API* i skali; własne uczą Cię *kontraktu, ablacji i decyzji architektonicznych*.

---

## 1. Porównanie budowy (warstwa uwagi)

| | Klasyczny Transformer / LLM | PhiMost (Most) |
|--|------------------------------|----------------|
| **Uwaga** | Softmax\((QK^\top)/\sqrt{d}\) | \(A=\gamma C/(\gamma^2+\Delta E^2)\), \(C=\mathrm{softplus}^2\) |
| **Skąd \(\Delta E\)** | brak (tylko treść+pozycja) | **tracer** τ (obserwator Φ) |
| **Treść** | tokeny / embeddingi | kanał **content** `x` (S/E/payload) |
| **Geometria** | learned positional / RoPE | **jawna** w τ (depth, T, mazur, …) |
| **Cel MVP** | next-token / chat | **recall/ranking** atomów |
| **Kontrola** | często brak ZERO-twin | Softmax twin + **ZERO-tracer** (opc. `--baselines`) |
| **Pamięć** | wagi + KV-cache (+ RAG zewnętrzny) | **M1 wagi · M2 Φ · M3 sesja** (rozdzielone) |
| **Dane** | korpus tokenów | graf Φ (atomy, bańki, bindy) |
| **Skala** | miliardy parametrów | lab: `d_model≈32`, 2 warstwy (świadomie małe) |

Wzór Mostu (kanon):

\[
A_{ij}=\frac{\gamma\,C_{ij}}{\gamma^2+\Delta E_{ij}^2}
\]

Bez niezerowego \(\Delta E\) most **degeneruje** — stąd ablacja ZERO: to nie „fair handicap”, tylko wyłączenie warunku geometrii.

---

## 2. Co jest podobne (żeby nie mitologizować)

| Element | Wspólne z LLM |
|---------|----------------|
| Warstwy QKV + residual + FFN | tak (szkielet transformera) |
| Trening gradientowy (Adam) | tak |
| Embedding wejścia → przestrzeń `d` | tak |
| Softmax jako baseline | tak — u nas **świadek**, nie cel |

PhiMost **nie** jest „zupełnie innym Wszechświatem”. To transformer z **innym zamkiem uwagi** i **innym kontraktem danych** (Φ zamiast płaskiego ciągu tokenów).

---

## 3. Co jest inne (tu jest nauka)

1. **Rozdział torów:** treść ≠ geometria (x vs τ).  
2. **Jawny obserwator:** τ musisz *zaprojektować* (depth, energia, klaster holo…).  
3. **Ontologia danych:** atom `(S,E,T)`, bańka, HOT/WARM — nie „wszystko w BPE”.  
4. **Kryterium sukcesu:** Most − Softmax ≥ gate (gdy włączysz baselines), nie tylko loss↓.  
5. **Trzy pamięci:** wiesz *gdzie* siedzi wiedza (wagi / dump / sesja).

To są decyzje, których tutorial „fine-tune LLaMA” **nie wymusza**.

---

## 4. Czy uczyć się na własnym modelu?

| Cel nauki | Lepiej |
|-----------|--------|
| Zrozumieć uwagę, ΔE, ablacje, kontrakt danych | **Własny PhiMost** |
| Szybki produkt chat / skala / RLHF | cudzy stack (API / otwarty LLM) |
| Intuicja „co psuje Most” (ZERO, flat, EMA-S) | **Własny** — masz już FAIL/PASS w `historia/` |
| Publikacja / reprodukcja paperów NLP | cudze baseline’y + Twoje ablacje obok |

**Rekomendacja dla Ciebie:**  
traktuj PhiMost jako **laboratorium budowy modeli**. Softmax twin jest wbudowanym „cudzym modelem w miniaturze”. LLM na zewnątrz bierz jako *narzędzie*, nie jako jedyny podręcznik architektury.

Dlaczego to skuteczniejsze pedagogicznie:

- Widzisz **cały tor**: dane → Φ → loss → ckpt → recall.  
- Możesz **zepsuć** jedną oś (mask, nucleus, blend_holo) i zobaczyć efekt.  
- Skala jest mała → iteracja w minutach, nie w GPU-tygodniach.  
- Masz **kanon zakazów** (EMA-S, mieszanie M1/M2/M3) — to dyscyplina inżynierska.

---

## 5. Mini-mapa „jak rosnąć” na własnym

```text
1. Zmieniasz kontrakt danych (CSV atomów, .md, Store)
2. Uczysz Most (phimost_learn, bez baselines)
3. Oglądasz recall (same_prefix, exclude_nucleus)
4. Co kilka iteracji: --baselines (Most vs Softmax vs ZERO)
5. Dopiero potem: większy d_model / więcej warstw / gen head
```

Nie zaczynaj od kopiowania GPT. Zacznij od pytania:  
**„co jest w tracerze, a co w content?”** — to jest Twój podręcznik.

---

## 6. Słownik skrótów vs branża

| U nas | W branży w przybliżeniu |
|-------|-------------------------|
| Most | attention z Lorentzian / energy-gated kernel |
| Softmax twin | standard multi-head attention |
| Tracer τ | side-channel / bias energii (nie RoPE) |
| Recall InfoNCE | contrastive retrieval (jak w bi-encoderach) |
| PhiJSON dump | structured memory graph (nie plain text corpus) |
| MemoryBus | cienki retrieval API nad embeddingami |

Szczegóły operacyjne: [`INSTRUKCJA.md`](INSTRUKCJA.md) · kanon: [`PAMIEC_I_KARMIENIE.md`](PAMIEC_I_KARMIENIE.md).
