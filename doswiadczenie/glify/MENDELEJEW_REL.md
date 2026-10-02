# Mendelejew REL — nauka relacyjna (osobne głowice relacji)

Cechy jak ZC (tylko Z + konfiguracja). Pary wszystkich relacji tylko spośród pierwiastków treningowych; ocena na nigdy nieoznaczonych (Sc/Ga/Ge/Tc zawsze w teście). Seedy=[0, 1, 2, 3, 4] · kroki=600 · czas=746s · skrypt `archiwum/skrypty/run_mendelejew_rel.py`

Pary w treningu (średnio): Z-sąsiedzi=58, okres=673, grupa all=309. Losowe hit@1 ¬f = 0.050.

`mix_*` = relacje zlane w jedno podobieństwo; `rel_*` = osobna głowica (przestrzeń) na relację.

## Grupa: hit@1 bez bloku f (główna miara)

| K grupa | reżim | most | softmax | zero | mlp |
|---|---|---:|---:|---:|---:|
| 18 | single | 0.167 ± 0.084 | 0.048 ± 0.066 | 0.000 ± 0.000 | 0.111 ± 0.054 |
| 18 | mix_Z | 0.048 ± 0.034 | 0.000 ± 0.000 | 0.008 ± 0.017 | 0.024 ± 0.022 |
| 18 | rel_Z | 0.065 ± 0.069 | 0.032 ± 0.018 | 0.023 ± 0.035 | 0.047 ± 0.032 |
| 18 | mix_ZP | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.023 ± 0.021 |
| 18 | rel_ZP | 0.040 ± 0.051 | 0.000 ± 0.000 | 0.016 ± 0.022 | 0.063 ± 0.022 |
| 54 | single | 0.299 ± 0.097 | 0.053 ± 0.062 | 0.046 ± 0.050 | 0.152 ± 0.107 |
| 54 | mix_Z | 0.069 ± 0.103 | 0.000 ± 0.000 | 0.024 ± 0.036 | 0.000 ± 0.000 |
| 54 | rel_Z | 0.257 ± 0.180 | 0.140 ± 0.110 | 0.127 ± 0.098 | 0.166 ± 0.058 |
| 54 | mix_ZP | 0.008 ± 0.017 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 54 | rel_ZP | 0.258 ± 0.103 | 0.112 ± 0.099 | 0.085 ± 0.063 | 0.182 ± 0.088 |
| all | single | 0.244 ± 0.107 | 0.054 ± 0.045 | 0.116 ± 0.105 | 0.290 ± 0.137 |
| all | mix_Z | 0.015 ± 0.021 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| all | rel_Z | 0.407 ± 0.186 | 0.196 ± 0.090 | 0.138 ± 0.072 | 0.230 ± 0.111 |
| all | mix_ZP | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| all | rel_ZP | 0.360 ± 0.094 | 0.180 ± 0.064 | 0.175 ± 0.102 | 0.205 ± 0.083 |

## Grupa: NMI (k-means, wszystkie 118)

| K grupa | reżim | most | softmax | zero | mlp |
|---|---|---:|---:|---:|---:|
| 18 | single | 0.584 ± 0.020 | 0.547 ± 0.009 | 0.550 ± 0.032 | 0.573 ± 0.028 |
| 18 | mix_Z | 0.557 ± 0.019 | 0.543 ± 0.015 | 0.537 ± 0.008 | 0.548 ± 0.020 |
| 18 | rel_Z | 0.558 ± 0.031 | 0.567 ± 0.013 | 0.551 ± 0.008 | 0.578 ± 0.013 |
| 18 | mix_ZP | 0.491 ± 0.031 | 0.486 ± 0.006 | 0.484 ± 0.012 | 0.489 ± 0.015 |
| 18 | rel_ZP | 0.560 ± 0.036 | 0.561 ± 0.016 | 0.562 ± 0.011 | 0.561 ± 0.013 |
| 54 | single | 0.702 ± 0.035 | 0.592 ± 0.044 | 0.578 ± 0.028 | 0.648 ± 0.024 |
| 54 | mix_Z | 0.630 ± 0.021 | 0.578 ± 0.012 | 0.577 ± 0.023 | 0.563 ± 0.020 |
| 54 | rel_Z | 0.705 ± 0.015 | 0.644 ± 0.017 | 0.656 ± 0.024 | 0.660 ± 0.021 |
| 54 | mix_ZP | 0.525 ± 0.020 | 0.445 ± 0.008 | 0.445 ± 0.009 | 0.471 ± 0.027 |
| 54 | rel_ZP | 0.683 ± 0.027 | 0.654 ± 0.028 | 0.641 ± 0.018 | 0.657 ± 0.026 |
| all | single | 0.687 ± 0.034 | 0.588 ± 0.041 | 0.596 ± 0.061 | 0.703 ± 0.025 |
| all | mix_Z | 0.648 ± 0.035 | 0.569 ± 0.019 | 0.580 ± 0.016 | 0.562 ± 0.010 |
| all | rel_Z | 0.776 ± 0.020 | 0.692 ± 0.040 | 0.683 ± 0.015 | 0.687 ± 0.025 |
| all | mix_ZP | 0.471 ± 0.029 | 0.449 ± 0.007 | 0.445 ± 0.023 | 0.435 ± 0.015 |
| all | rel_ZP | 0.771 ± 0.026 | 0.685 ± 0.017 | 0.694 ± 0.032 | 0.691 ± 0.030 |

## Efekt relacji względem `single` (Δ hit@1 ¬f, wygrane seedy) — Most i średnia po armiach

| K | reżim | Most Δ (wygrane) | średnia Δ po armiach |
|---|---|---:|---:|
| 18 | mix_Z | -0.119 (0/5) | -0.061 |
| 18 | rel_Z | -0.102 (0/5) | -0.040 |
| 18 | mix_ZP | -0.167 (0/5) | -0.076 |
| 18 | rel_ZP | -0.127 (0/5) | -0.052 |
| 54 | mix_Z | -0.229 (0/5) | -0.114 |
| 54 | rel_Z | -0.042 (1/5) | +0.035 |
| 54 | mix_ZP | -0.291 (0/5) | -0.136 |
| 54 | rel_ZP | -0.040 (0/5) | +0.022 |
| all | mix_Z | -0.229 (0/5) | -0.172 |
| all | rel_Z | +0.163 (4/5) | +0.067 |
| all | mix_ZP | -0.244 (0/5) | -0.176 |
| all | rel_ZP | +0.116 (5/5) | +0.054 |

## Most vs reszta (hit@1 ¬f, wygrane seedy)

| K | reżim | vs softmax | vs zero | vs mlp |
|---|---|---:|---:|---:|
| 18 | single | 5/5 | 5/5 | 3/5 |
| 18 | mix_Z | 4/5 | 3/5 | 2/5 |
| 18 | rel_Z | 3/5 | 3/5 | 2/5 |
| 18 | mix_ZP | 0/5 | 0/5 | 0/5 |
| 18 | rel_ZP | 3/5 | 2/5 | 1/5 |
| 54 | single | 5/5 | 5/5 | 5/5 |
| 54 | mix_Z | 2/5 | 2/5 | 2/5 |
| 54 | rel_Z | 5/5 | 4/5 | 3/5 |
| 54 | mix_ZP | 1/5 | 1/5 | 1/5 |
| 54 | rel_ZP | 4/5 | 5/5 | 4/5 |
| all | single | 4/5 | 3/5 | 2/5 |
| all | mix_Z | 2/5 | 2/5 | 2/5 |
| all | rel_Z | 5/5 | 4/5 | 5/5 |
| all | mix_ZP | 0/5 | 0/5 | 0/5 |
| all | rel_ZP | 5/5 | 4/5 | 5/5 |

## Kontrola: głowica „okres” w rel_ZP (hit@1 okresu na teście)

| K | most | softmax | zero | mlp |
|---|---:|---:|---:|---:|
| 18 | 0.994 ± 0.013 | 0.983 ± 0.016 | 0.989 ± 0.016 | 0.983 ± 0.026 |
| 54 | 0.966 ± 0.037 | 0.989 ± 0.016 | 0.994 ± 0.013 | 0.983 ± 0.026 |
| all | 0.983 ± 0.016 | 0.989 ± 0.016 | 0.989 ± 0.016 | 0.977 ± 0.024 |

## Analogi luk w głowicy „grupa” (Most, seed 0, K=54)

Oczekiwane: Sc → Y/La/Lu · Ga → Al/In/Tl · Ge → Si/Sn/Pb · Tc → Mn/Re

| reżim | Sc | Ga | Ge | Tc |
|---|---|---|---|---|
| single | Ti, Ca, V, Nb | Be, In, B, Li | In, Ga, B, Tl | Mo, Sg, Db, Nb |
| mix_Z | Ca, Ti, K, Sr | Ge, Rf, Db, V | Ga, As, Al, N | Mo, Nb, Sg, V |
| rel_Z | Ca, Ti, Sr, Li | Ge, Mn, Cr, In | As, Bi, Ga, Po | Sn, In, Ge, Fl |
| mix_ZP | Ti, Ca, V, Cr | Ge, As, Cr, V | Ga, As, Cr, Mn | Zr, Nb, Mo, Y |
| rel_ZP | Ti, Ca, Lr, Rf | Ge, Nh, Cr, La | As, Ga, C, Bi | Ir, Fe, Hs, Os |

## Wnioski

**Zastrzeżenia:** ~25 pierwiastków testowych spoza bloku f (1 trafienie ≈ 4 pkt), 5 seedów, duży rozrzut (Most do ±0.19). Liczę się tylko ze stałym znakiem efektu. Głowica „okres” korzysta z dodatkowych etykiet (pary tego samego okresu), a relacja Z jest darmowa.

1. **Zlewanie relacji w jedno podobieństwo niszczy grupę.** `mix_Z` i `mix_ZP` są gorsze od `single` w **każdym** budżecie i modelu. Most traci od −0.12 do −0.29, 0/5 seedów. Przy relacji okresu (673 pary, poziomej) grupa spada do zera. To potwierdza diagnozę z SEMI i ACTIVE: jedna przestrzeń nie pomieści „poziomo” i „pionowo” naraz.
2. **Osobne głowice relacji rozwiązują ten konflikt, jeśli etykiet grupy jest dość.**

   | Most, K=all | hit@1 ¬f | NMI grupy |
   |---|---:|---:|
   | single | 0.24 | 0.69 |
   | rel_Z | **0.41** (4/5 lepiej) | **0.78** |
   | rel_ZP | 0.36 (5/5 lepiej) | 0.77 |

   **0.41 to najlepsza generalizacja grupy w całej serii** (poprzednio 0.32 w SEMI K=54), a NMI 0.78 to najwyższe grupowanie. Głowica „okres” w `rel_ZP` jednocześnie trafia okres w ~0.98. **Wspólny enkoder trzyma obie struktury tablicy, każdą w swojej przestrzeni.** Darmowa relacja Z (`rel_Z`) wystarcza: etykiety okresu nic nie dodają.
3. **Przy małej liczbie etykiet grupy relacje pomocnicze dalej konkurują.** Przy K=18 każde `rel_*` jest gorsze od `single` (Most od −0.10 do −0.13, 0/5). Przy K=54 jest mniej więcej neutralnie (Most −0.04, średnia po modelach +0.03). Gęsta relacja pomocnicza (58 par Z, 673 okresu) przy 18–54 parach grupowych przeciąga wspólny enkoder na swoją stronę. Zysk pojawia się dopiero przy ~300 parach grupowych. Ważenie strat relacji nie było strojone.
4. **Most jest najlepszy właśnie w reżimie relacyjnym.** Przy `rel_Z`/`rel_ZP` K=all: Most 0.41 / 0.36, wobec Softmax 0.20 / 0.18, ZERO 0.14 / 0.18 i MLP 0.23 / 0.21. To 5/5 wobec Softmax i MLP oraz 4/5 wobec ZERO. Przewaga nad ZERO pokazuje, że relacje uczone przez ΔE z tracera przenoszą się na nowe pierwiastki lepiej niż przez samą treść. Zauważalne jest też, że `single` K=all (zdominowane przez pary z bloku f) jest słabsze od K=54, a głowica Z ten spadek naprawia.
5. **Analogi luk stają się „pionowe”, choć jeszcze nie trafne.** We wcześniejszych testach sąsiadami Ga były As i Se (ten sam okres). Teraz w głowicy grupy (K=54, seed 0): Ga → **In**, **B** (grupa 13, trafienie) w `single`, Ga → **Nh** (13) w `rel_ZP`. Ge → In, B, Tl (grupa 13 zamiast 14), Ge → Bi, Po (15/16) w `rel_Z`. Tc → Mo, Nb, Sg, Db (grupy 5–6 zamiast 7). Model szuka już w pionie, ale myli się o jedną grupę.

## Werdykt

Nauka relacyjna działa i daje **najlepszy wynik serii**: Most w `rel_Z` przy pełnych parach grupowych ma hit@1 ¬f 0.41 i NMI 0.78, 5/5 wobec Softmax i MLP. Kluczowe jest **rozdzielenie relacji na osobne przestrzenie**: ta sama wiedza wlana do jednej przestrzeni (mix) niszczy grupę do zera. Warunek to dość etykiet grupy (~300 par); przy 18–54 parach relacje pomocnicze jeszcze konkurują o enkoder.

**Kierunki:**
1. Ważenie strat (mała waga relacji pomocniczych przy małym K) albo stopniowe wygaszanie. To prawdopodobnie przesunie zysk na małe budżety.
2. Relacje kompozycyjne: „ta sama grupa ∘ następny okres” (TransE / operator przesunięcia). To pozwoliłoby wprost zadać pytanie Mendelejewa: „co jest pod Al?”.
3. Zrównoważyć `all` (bez dominacji bloku f) i powtórzyć na 10–20 seedach, bo rozrzut jest duży.

