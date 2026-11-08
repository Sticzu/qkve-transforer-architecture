# QKVE — Transformer Architecture and Value Gating Experiments

[English](#english) · [Polski](#polski)

## English

### Motivation

This project began as a hands-on exploration of how Transformers work. The starting question was: what happens if we extend attention horizontally, by adding parallel learned projections, rather than increasing the number of Transformer blocks?

I explored additional projections and their roles within attention. The most promising direction in my experiments was an extra projection, called **E**, that modulates the values **V**.

### Mechanism

Alongside the usual query, key, and value projections, QKVE introduces a learned projection matrix $W_E$. It produces an input-dependent gate that modulates each component of the value vector through elementwise multiplication:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \tanh(xW_E).
$$

Here, $x$ is the input representation, $W_V$ is the value projection matrix, and $\odot$ denotes elementwise multiplication. The output dimensions of the two projections match.

The gate is applied to the projected values before the attention-weighted aggregation. The name **E** originally came from “emotional” and was later interpreted as “energy”; technically, it is a value-modulating gate, rather than a defined measure of energy.

I also explored an offset variant:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \left(1 + \tanh(xW_E)\right).
$$

These variants have different behavior:

| Gate | Multiplier range | Effect on each value component |
| --- | --- | --- |
| $\tanh(xW_E)$ | $(-1, 1)$ | Attenuates magnitude and can reverse the sign |
| $1 + \tanh(xW_E)$ | $(0, 2)$ | Can attenuate or amplify magnitude while preserving the sign |

These descriptions compare the gated component with the corresponding component of $xW_V$ in the same forward pass. Both projection matrices are learned during training.

A sine-based variant was also explored. In the included pilot, its precise definition is:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \left(\sin(xW_E) + \tanh(xW_E)\right).
$$

In that implementation, $x$ is the normalized input to attention and the gate is applied on the KV heads before aggregation.

### Exploratory observations

In my experiments so far, the E projection was the most promising of the additional projections I tried. The `+1` offset did not provide a worthwhile improvement in loss or perplexity in the runs I compared. A sine-based variant appeared to improve quality, but introduced additional computational cost. These observations motivated further exploration of the plain `tanh` gate.

These are preliminary observations from my own experiments. The separate pilot below provides quantitative comparisons in a short-training setting; it does not establish a hardware speedup or validate the separately trained checkpoint.

### Included pilot benchmarks

The repository includes the complete original [GQA E pilot](benchmarks/README.md): training and evaluation scripts, the Polish tokenizer, data manifests, plots, logs, and raw results. The [full Polish report](benchmarks/gqa_e_pilot/RAPORT.md) describes the experiment and its limitations.

The pilot used approximately 1M-parameter models, three seeds (11, 22, 33), context length 64, and **307,200 training tokens per language-model run**, on CPU. It contains 60 main training runs and six additional exploratory scale-control runs.

Selected test perplexity results, mean ± sample standard deviation across three seeds; lower is better:

| Variant | Wikipedia PL | Polish literature |
| --- | ---: | ---: |
| Baseline GQA | 508.89 ± 10.70 | 331.32 ± 2.05 |
| $V \odot \tanh(xW_E)$ | 453.82 ± 5.40 | 258.78 ± 11.04 |
| No E: parameter-matched FFN and $V \cdot 0.216$ | 445.60 ± 0.41 | 241.26 ± 2.82 |

**Interpretation:** tanh gating improved perplexity relative to baseline GQA in this short-training setting. The fixed-scale control obtained lower mean perplexity on both corpora, but paired confidence intervals for the difference between E and this control included zero. The pilot therefore does not demonstrate a benefit from the learned E projection beyond changing the scale of V. The scale control was added after inspecting initial results and is exploratory.

These measurements do not establish convergence, larger-model performance, or faster training. The recorded CPU wall times include validation and shared CPU use. Synthetic-task results are reported separately in the full report. No new training runs were performed when importing this pilot.

### Model and current status

The model has been trained locally on my own GPU, predominantly on Polish-language data. The currently available local checkpoint uses the older **`1 + tanh`** variant. Results from that checkpoint need to be distinguished from results obtained with the plain `tanh` variant.

Training and evaluation are still in progress. The project also serves as a way to learn about training data, training dynamics, and the steps needed to turn an experimental language model into a useful one.

The repository contains the project description and the pilot implementation and benchmark artifacts under [`benchmarks/gqa_e_pilot/`](benchmarks/gqa_e_pilot/). The separately trained model checkpoint described above is not included. The pilot archive also contains no final model weights; reproducing those models requires retraining.

### Evaluation to document

- A standard-attention baseline and the individual gating variants.
- Model sizes, parameter counts, training-token budgets, data splits, and tokenizer details.
- Training configurations, initialization, random seeds, and checkpoint versions.
- Held-out loss and perplexity, together with any task-specific evaluations.
- Training and inference performance, hardware, precision, and measurement settings.
- Reproduction commands, raw results, and known limitations.

### Related work

[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) investigates learned multiplicative gates at several positions within attention, including after the value projection. It provides a relevant reference for comparison.

This project documents my own exploration and implementation. It does not claim that value gating itself is a new mechanism.

## Polski

### Motywacja

Projekt powstał podczas praktycznej nauki działania Transformerów. Punktem wyjścia było pytanie: co się stanie, jeśli rozszerzymy attention horyzontalnie, dodając równoległe uczone projekcje, zamiast zwiększać liczbę bloków Transformera?

Sprawdzałem dodatkowe projekcje i ich role w mechanizmie attention. Najbardziej obiecującym kierunkiem w moich eksperymentach okazała się dodatkowa projekcja, nazwana **E**, modulująca wartości **V**.

### Mechanizm

Obok standardowych projekcji query, key i value QKVE wprowadza uczoną macierz projekcji $W_E$. Tworzy ona bramkę zależną od wejścia, która moduluje poszczególne składowe wektora wartości przez mnożenie element po elemencie:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \tanh(xW_E).
$$

Tutaj $x$ oznacza reprezentację wejściową, $W_V$ jest macierzą projekcji wartości, a $\odot$ oznacza mnożenie element po elemencie. Wymiary wyjściowe obu projekcji są zgodne.

Bramka jest nakładana na wartości po projekcji, przed agregacją ważoną przez attention. Nazwa **E** początkowo pochodziła od „emotional”, a później była interpretowana jako „energy”; technicznie jest to bramka modulująca wartości, a nie zdefiniowana miara energii.

Sprawdzałem również wariant z przesunięciem:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \left(1 + \tanh(xW_E)\right).
$$

Warianty te zachowują się inaczej:

| Bramka | Zakres mnożnika | Wpływ na każdą składową wartości |
| --- | --- | --- |
| $\tanh(xW_E)$ | $(-1, 1)$ | Tłumi wartość bezwzględną i może odwrócić znak |
| $1 + \tanh(xW_E)$ | $(0, 2)$ | Może tłumić lub wzmacniać wartość bezwzględną, zachowując znak |

Powyższy opis porównuje składową po bramkowaniu z odpowiadającą jej składową $xW_V$ w tym samym przebiegu obliczeń. Obie macierze projekcji są uczone podczas treningu.

Sprawdzałem także wariant zawierający sinus. W dołączonym pilotażu ma on dokładnie postać:

$$
V_{\mathrm{mod}}(x) = (xW_V) \odot \left(\sin(xW_E) + \tanh(xW_E)\right).
$$

W tej implementacji $x$ oznacza znormalizowane wejście attention, a bramka działa na głowach KV przed agregacją.

### Wstępne obserwacje

W dotychczasowych eksperymentach projekcja E była najbardziej obiecującą spośród dodatkowych projekcji, które sprawdzałem. Przesunięcie `+1` nie przyniosło wartej dodatkowego kosztu poprawy funkcji straty ani perplexity w porównywanych przebiegach. Wariant z sinusem wydawał się poprawiać jakość, ale wprowadzał dodatkowy koszt obliczeniowy. Obserwacje te skłoniły mnie do dalszych eksperymentów z samą bramką `tanh`.

Są to wstępne obserwacje z moich własnych eksperymentów. Osobny pilotaż opisany poniżej zawiera porównania liczbowe przy krótkim treningu; nie dowodzi przyspieszenia sprzętowego ani nie stanowi oceny osobno trenowanego checkpointu.

### Dołączone benchmarki pilotażowe

Repozytorium zawiera kompletny oryginalny [pilotaż GQA E](benchmarks/README.md): skrypty treningu i ewaluacji, polski tokenizer, manifesty danych, wykresy, logi i surowe wyniki. [Pełny raport po polsku](benchmarks/gqa_e_pilot/RAPORT.md) opisuje eksperyment i jego ograniczenia.

Pilotaż obejmował modele około 1 mln parametrów, trzy seedy (11, 22, 33), kontekst 64 i **307 200 tokenów treningowych na każdy trening modelu językowego**, na CPU. Zawiera 60 treningów głównych oraz sześć dodatkowych, eksploracyjnych treningów kontroli skali.

Wybrane wyniki testowej perplexity: średnia ± odchylenie standardowe z trzech seedów; mniej oznacza lepiej:

| Wariant | Wikipedia PL | Literatura PL |
| --- | ---: | ---: |
| Bazowe GQA | 508,89 ± 10,70 | 331,32 ± 2,05 |
| $V \odot \tanh(xW_E)$ | 453,82 ± 5,40 | 258,78 ± 11,04 |
| Bez E: FFN o wyrównanej liczbie parametrów i $V \cdot 0{,}216$ | 445,60 ± 0,41 | 241,26 ± 2,82 |

**Interpretacja:** bramka tanh poprawiła perplexity względem bazowego GQA przy tym krótkim treningu. Kontrola ze stałą skalą uzyskała niższe średnie na obu korpusach, ale sparowane przedziały niepewności różnicy między E a tą kontrolą obejmowały zero. Pilotaż nie wykazuje więc korzyści z uczonej projekcji E ponad efekt zmiany skali V. Kontrolę skali dodano po obejrzeniu początkowych wyników i ma ona charakter eksploracyjny.

Pomiary nie dowodzą zbieżności, skuteczności większego modelu ani przyspieszenia treningu. Zapisane czasy CPU obejmują walidację i współdzielenie procesora. Wyniki zadań syntetycznych przedstawiono osobno w pełnym raporcie. Przy dodawaniu pilotażu do repozytorium nie wykonywano nowych treningów.

### Model i obecny stan projektu

Model był trenowany lokalnie na mojej karcie graficznej, przede wszystkim na danych polskojęzycznych. Obecnie dostępny lokalnie checkpoint wykorzystuje starszy wariant **`1 + tanh`**. Wyniki tego checkpointu należy rozróżniać od wyników uzyskanych z samą bramką `tanh`.

Trening i ewaluacja pozostają w toku. Projekt służy również nauce doboru danych, dynamiki treningu oraz kroków potrzebnych do przekształcenia eksperymentalnego modelu językowego w model użyteczny.

Repozytorium zawiera opis projektu oraz implementację i pliki benchmarków pilotażu w [`benchmarks/gqa_e_pilot/`](benchmarks/gqa_e_pilot/). Osobno trenowany checkpoint opisany powyżej nie jest dołączony. Archiwum pilotażu również nie zawiera końcowych wag modeli; ich odtworzenie wymaga ponownego treningu.

### Plan dokumentacji pomiarów

- Model odniesienia ze standardowym attention i poszczególne warianty bramkowania.
- Rozmiary modeli, liczby parametrów, budżety tokenów treningowych, podziały danych i szczegóły tokenizera.
- Konfiguracje treningu, inicjalizacja, ziarna losowości i wersje checkpointów.
- Funkcja straty i perplexity na danych odłożonych do oceny oraz ewentualne testy konkretnych zadań.
- Wydajność treningu i inferencji, sprzęt, precyzja obliczeń i warunki pomiarów.
- Komendy pozwalające powtórzyć pomiary, surowe wyniki i znane ograniczenia.

### Powiązane prace

[Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708) bada uczone bramki multiplikatywne w różnych miejscach attention, w tym po projekcji wartości. Stanowi istotny punkt odniesienia do porównań.

Projekt dokumentuje moje własne eksperymenty i implementację. Nie zakłada, że samo bramkowanie wartości jest nowym mechanizmem.
