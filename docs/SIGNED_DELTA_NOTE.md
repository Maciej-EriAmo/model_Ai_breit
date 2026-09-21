# Energie ujemne / skierowane ΔE w Moście

## Fakt o starym wzorze

`E = Linear(tracer)` **już może być ujemne**. Problem nie jest „brak ujemnych E”, tylko:

\[
A \propto \frac{1}{g^2 + (\Delta E)^2}
\quad\text{gubi znak:}\quad
\Delta E \equiv -\Delta E.
\]

## Co dodano (`MostKnobs.delta_mode`)

| mode | Zachowanie |
|------|------------|
| `square` | klasyczny Lorentz |
| `directed` | × \(\sigma(\alpha\Delta E)\) — preferuj key o niższej energii (\(E_q>E_k\)) |
| `onesided` | \(g_{\mathrm{eff}}=g(1+\mathrm{softplus}(\alpha\Delta E))\) — szerszy zamek „w dół” |

## Wynik ablacji @2048

Zob. [`../doswiadczenie/glify/SIGNED_DELTA_ABLATION.md`](../doswiadczenie/glify/SIGNED_DELTA_ABLATION.md).

- **directed** sam: ent_mid **6.38** ≤ square **6.41** (nie pomaga)
- **onesided** sam: **6.66** > square (lekka poprawa, wciąż < SM 6.93)
- **ultra_wide_directed**: **6.928** ≈ SM — to w praktyce efekt **szerokiego g / słabego e_scale**, nie cudu znaku

## Update: ten sam wzór na atomach FULL118

[`../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md`](../doswiadczenie/glify/ATOMS_DELTA_ABLATION.md) — 16 ep × seeds 0,1:

| variant | mean r@1 | ΔSM r@1 |
|---------|---------:|--------:|
| Softmax | 0.579 | — |
| square | 0.567 | −0.012 |
| directed | 0.577 | −0.002 |
| **onesided** | **0.591** | **+0.012** |
| ultra_wide | 0.584 | +0.005 |
| ultra_wide_directed | 0.584 | +0.005 |

Na **secie atomów** `onesided` wygrywa square i lekko Softmax (gate 1/2). To **mocniejszy** sygnał niż na LM mid, gdzie pomagało głównie szerokie g.

## Wniosek

- Znak ΔE jest **uzasadniony** i w encoderze atomów **`onesided` warto trzymać jako default eksperymentalny**.
- `directed` (sigmoid) słabszy niż `onesided` (szerszy g „w dół”).
- LM mid ≠ atomy: na tekście wygrywało `ultra_wide`; na Φ-atomach — **onesided**.
- Hold-out **onesided** ([`MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md`](../doswiadczenie/glify/MENDELEJEW_HOLDOUT_FULL_NaK_onesided.md)):
  - **Na:** PARTIAL — Li ranks 10/6 (lepiej niż square 29/2); **K** (mate grupy) @1 i @3
  - **K:** FAIL — Na ranks **57/59** (gorzej niż square 19/27)
- Wniosek: `onesided` **wzmacnia sygnał alkali przy hold-out Na** (często K zamiast Li), ale **nie** naprawia generalizacji K→Na. Nie overclaim „grupa rozwiązana”.
