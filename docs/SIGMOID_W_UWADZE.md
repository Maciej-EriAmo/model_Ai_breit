# Sigmoid zamiast softmaxa. Krótka historia buntu w sercu Transformera

*Podsumowanie popularnonaukowe prac z lat 2023–2025 o uwadze sigmoidalnej i jej krewnych*

---

## 1. O co w ogóle chodzi

Każdy Transformer, od GPT po modele rozpoznające obrazy, ma w środku mechanizm **uwagi**. Działa on mniej więcej tak: każde słowo (token) zadaje pytanie (*query*), każde inne słowo wystawia etykietę (*key*), a model liczy, jak bardzo pytanie pasuje do etykiety. Z tych dopasowań powstają **wagi**, które mówią: „z tego słowa weź dużo, z tamtego mało”.

Od 2017 roku wagi te liczy się funkcją **softmax**. Softmax robi dwie rzeczy naraz:

1. podnosi wynik dopasowania do potęgi liczby *e* (wszystko staje się dodatnie, a różnice się wyostrzają),
2. dzieli przez sumę, tak żeby wszystkie wagi w wierszu sumowały się do **jedynki**.

Ten drugi krok ma ciekawą konsekwencję: tokeny **konkurują** o uwagę. Jeśli jeden dostaje więcej, pozostałe muszą dostać mniej. Uwaga jest jak tort, który zawsze trzeba rozdać w całości, nawet gdy nikt nie jest głodny.

**Sigmoid** działa inaczej. Każda waga liczona jest **osobno**, jako liczba od 0 do 1, bez oglądania się na sąsiadów. Nie ma tortu do podziału: każdy token dostaje tyle, na ile sam zasłużył. Może się zdarzyć, że wszyscy dostaną dużo albo wszyscy prawie nic.

Pytanie, które przez kilka lat zadawali sobie badacze, brzmi: **czy ta konkurencja jest naprawdę potrzebna?**

---

## 2. Pierwsza przymiarka: ReLU dzielone przez długość (2023)

Wcześniejsze próby zastąpienia softmaxa prostymi funkcjami punktowymi kończyły się spadkiem jakości. Zespół z Google DeepMind (Wortsman i współpracownicy, 2023) znalazł prosty powód i proste lekarstwo.

Okazało się, że kluczowe jest **podzielenie wyniku przez długość sekwencji**. Softmax robi to „za darmo”, bo normalizuje wagi do jedynki. Funkcja punktowa tego nie robi, więc przy długich tekstach suma wag rośnie i sieć się rozregulowuje. Po dodaniu dzielenia przez długość uwaga oparta na ReLU w transformerach wizyjnych trenowanych na ImageNet-21k zbliżyła się do softmaxa, a czasem mu dorównała, również pod względem tego, jak jakość rośnie z mocą obliczeniową.

Co ciekawe, autorzy przetestowali cały zestaw funkcji (ReLU, ReLU², GELU, softplus, sigmoid i inne) i **nie znaleźli wyraźnego zwycięzcy**. Wybrali ReLU, bo jest najszybsze.

**Lekcja nr 1:** nie chodzi o kształt funkcji, tylko o **skalę**. Odpowiednio przeskalowana prawie każda rozsądna funkcja działa.

---

## 3. Przełom: Apple i „najlepsze praktyki” (2024, ICLR 2025)

Najważniejszą pracą o samej uwadze sigmoidalnej jest artykuł zespołu Apple (Ramapuram i 10 współautorów, wrzesień 2024, przyjęty na ICLR 2025). Autorzy zrobili trzy rzeczy.

**Teoria.** Udowodnili, że Transformer z uwagą sigmoidalną jest **uniwersalnym aproksymatorem funkcji**, czyli w zasadzie może nauczyć się tego samego co wersja z softmaxem. Pokazali też, że ma lepsze własności regularności (jest „gładszy” matematycznie).

**Diagnoza.** Wcześniejsze próby się wykładały, bo na początku treningu normy uwagi bywają bardzo duże. Kluczem okazało się **ustabilizowanie tych dużych norm na starcie treningu**. Praktycznie robi się to, dodając do argumentu sigmoidy ujemne przesunięcie zależne od długości sekwencji (rzędu −log n). Dzięki temu na starcie każda waga jest mała, a suma wag pozostaje rozsądna, podobnie jak przy softmaxie.

**Inżynieria.** Napisali **FlashSigmoid**, zoptymalizowaną implementację na GPU, która na kartach H100 dała około 17% przyspieszenia jądra względem FlashAttention-2. Sigmoid jest szybszy, bo nie trzeba liczyć sumy po całym wierszu, więc każdy element da się obliczyć niezależnie.

**Wynik.** Poprawnie znormalizowana uwaga sigmoidalna dorównała softmaxowi w zadaniach językowych, wizyjnych i mowy, w wielu skalach.

**Lekcja nr 2:** sigmoid może być pełnoprawnym zamiennikiem, ale tylko z dobrze dobraną inicjalizacją i przesunięciem. „Goła” sigmoida nie wystarczy.

---

## 4. Teoria: czy sigmoid uczy się szybciej? (2025)

W lutym 2025 Yan i współpracownicy spojrzeli na problem od strony teorii statystycznego uczenia. Zauważyli, że każdy wiersz macierzy uwagi można opisać jak **mieszaninę ekspertów** (*mixture of experts*). Każdy token to „ekspert”, a wagi uwagi decydują, ile każdy z nich ma do powiedzenia.

W tym ujęciu wykazali, że eksperci w uwadze sigmoidalnej potrzebują **znacznie mniej danych**, żeby osiągnąć tę samą dokładność aproksymacji co w softmaxie. Intuicja: softmax wymusza konkurencję, która może zawężać uwagę do kilku cech i pomijać inne użyteczne informacje. Sigmoid tej presji nie wywiera.

**Uwaga krytyczna:** to wynik teoretyczny w uproszczonym modelu (głównie jedna głowica). Pełne ustawienie wielogłowicowe pozostaje otwartym pytaniem.

---

## 5. Zagadka „zlewu uwagi” (2024, ICLR 2025)

Modele językowe mają dziwny nawyk: kierują ogromną część uwagi na **pierwszy token** w tekście, nawet gdy jest on semantycznie bez znaczenia. Zjawisko nazwano *attention sink*, czyli „zlew uwagi”.

Gu i współpracownicy (2024) zbadali, skąd się to bierze. Wnioski:

- zlew pojawia się dopiero po skutecznym treningu na dostatecznie dużej ilości danych,
- działa jak **bias klucza**: przechowuje nadmiarową uwagę, która nie wnosi nic do wyniku,
- przynajmniej częściowo wynika z **normalizacji softmaxa**. Skoro wagi muszą sumować się do jedynki, model, który „nie ma na co patrzeć”, musi tę jedynkę gdzieś zrzucić. Wybiera pierwszy token jako śmietnik.

Najbardziej elegancki eksperyment: gdy zastąpić softmax **uwagą sigmoidalną bez normalizacji**, zlew uwagi w ogóle się nie pojawia, i to w modelach aż do 1 mld parametrów. Co więcej, po dodaniu normalizacji z powrotem zlew wraca.

**Lekcja nr 3:** wymóg „suma = 1” nie jest neutralny. Zmusza model do wymyślania obejść.

---

## 6. Sigmoid jako bramka: nagroda NeurIPS 2025

Uwaga: to inna rodzina pomysłów, ale bardzo blisko spokrewniona i dziś najbardziej wpływowa.

Zespół Qwen (Qiu i współpracownicy, 2025) **nie wyrzucił softmaxa**. Zamiast tego dodał **sigmoidalną bramkę** zaraz za wyjściem uwagi. Każda głowica dostaje własny „zawór” od 0 do 1, sterowany treścią wejścia. Jeśli głowica nie znalazła nic sensownego, może po prostu się przymknąć, zamiast zrzucać uwagę na pierwszy token.

Skala eksperymentów była potężna: ponad 30 wariantów, modele gęste 1,7 mld parametrów i modele MoE 15 mld parametrów, trening na 3,5 biliona tokenów. Najlepiej wypadła bramka sigmoidalna, specyficzna dla głowicy, mnożona element po elemencie przez wyjście uwagi. Pokonała m.in. warianty z SiLU i warianty addytywne.

Efekty:

- **zlew uwagi praktycznie znika**: udział uwagi na pierwszym tokenie spadł z około 47% do około 5%,
- trening jest **stabilniejszy** i toleruje wyższe tempo uczenia,
- model lepiej **ekstrapoluje na długi kontekst**,
- bramka często nasyca się blisko zera, tworząc naturalną rzadkość (*sparsity*).

Mechanizm trafił do modeli **Qwen3-Next**, a praca dostała **nagrodę za najlepszy artykuł na NeurIPS 2025**.

**Lekcja nr 4:** czasem nie trzeba wyrzucać softmaxa. Wystarczy dać mu wyjście ewakuacyjne w postaci „mogę nic nie powiedzieć”.

---

## 7. Cień: długie konteksty (2025)

Nie wszystko jest różowe. Praca o uwadze softplus (2025) porównała różne warianty bez softmaxa na długich sekwencjach. Oryginalna uwaga sigmoidalna wyraźnie traciła jakość przy dłuższych tekstach: strata walidacyjna rosła gwałtownie przy 8 tysiącach tokenów.

Ratunkiem okazało się dodanie z powrotem **normalizacji L1**, czyli dzielenia przez sumę wag. Z nią sigmoid wracał do poziomu softmaxa.

**Lekcja nr 5:** przy krótkich tekstach brak normalizacji pomaga (brak zlewu), przy bardzo długich szkodzi (wagi się rozjeżdżają). Rachunek zależy od długości kontekstu.

---

## 8. Szerszy kontekst: softmax nie jest święty

Prace o sigmoidzie to część większego ruchu. W tym samym czasie badano m.in.:

- **jądra wielomianowe**, które działają mimo łamania dodatniości, normalizacji i rzadkości. Autorzy sugerują, że siła softmaxa leży głównie w niejawnej kontroli normy macierzy uwagi, a nie w interpretacji probabilistycznej,
- **jądro Cauchy'ego** (ZETA, 2025) z cięższymi ogonami, lepiej dopasowane do odległości euklidesowej,
- **jądro Laplace'a** (LaplacianFormer, 2026), które mniej tłumi interakcje na średnim dystansie.

Wspólny wniosek z tej literatury: **wiele różnych funkcji działa**, jeśli zadba się o skalę, normalizację i stabilną inicjalizację.

---

## 9. Pięć lekcji w pigułce

| # | Lekcja | Skąd |
|---|--------|------|
| 1 | Skala ważniejsza niż kształt funkcji | Wortsman i in. 2023 |
| 2 | Sigmoid działa, ale potrzebuje przesunięcia i stabilnej inicjalizacji | Ramapuram i in. 2024 (Apple) |
| 3 | Wymóg „suma = 1” rodzi zlew uwagi | Gu i in. 2024 |
| 4 | Bramka sigmoidalna za softmaxem daje stabilność i długi kontekst | Qiu i in. 2025 (Qwen) |
| 5 | Przy bardzo długich tekstach normalizacja wraca do łask | praca o softplus, 2025 |

---

## 10. Co z tego wynika dla kogoś, kto projektuje własne jądro uwagi

- **Parytet z softmaxem to dziś punkt startowy, a nie wynik.** Literatura pokazała, że osiąga go wiele funkcji.
- **Trzeba zawsze raportować, jak normalizujesz.** Bez normalizacji, z normalizacją wierszową, L1, przez długość sekwencji: każda opcja zmienia zachowanie przy długich kontekstach i wpływa na zlew uwagi.
- **Warto mierzyć zlew uwagi** (udział uwagi na pierwszym tokenie). To tani, czytelny wskaźnik, który recenzenci znają.
- **Warto testować na kilku długościach kontekstu**, bo przewaga przy krótkich może zniknąć przy długich.
- **Najmocniejsze argumenty to mechanizm albo zysk w skali**, a nie sam kształt krzywej.

---

## Źródła

- Wortsman i in., *Replacing softmax with ReLU in Vision Transformers* (2023) — https://arxiv.org/abs/2309.08586
- Ramapuram i in., *Theory, Analysis, and Best Practices for Sigmoid Self-Attention* (ICLR 2025) — https://arxiv.org/abs/2409.04431
- Yan i in., *Sigmoid Self-Attention is Better than Softmax Self-Attention: A Mixture-of-Experts Perspective* (2025) — https://arxiv.org/abs/2502.00281
- Gu i in., *When Attention Sink Emerges in Language Models: An Empirical View* (ICLR 2025) — https://arxiv.org/abs/2410.10781
- Qiu i in., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free* (NeurIPS 2025, Best Paper) — https://arxiv.org/abs/2505.06708
- *Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models* (2025) — https://arxiv.org/abs/2501.13428
- *Rethinking Softmax: Self-Attention with Polynomial Activations* — https://arxiv.org/abs/2410.18613
- ZETA, *Leveraging Z-order Curves for Efficient Top-k Attention* (2025) — https://arxiv.org/abs/2501.14577
- Feng i in., *LaplacianFormer: Rethinking Linear Attention with Laplacian Kernel* (2026) — https://arxiv.org/abs/2604.20368
