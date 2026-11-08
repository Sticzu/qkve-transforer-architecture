# GQA E pilot / Pilotaż GQA E

[English](#english) · [Polski](#polski)

## English

This directory preserves the complete 2026-11-07 pilot from `gqa_E_polskie_benchmarki.zip`. All 101 original files are unchanged. The 100 checksums in the original manifest were verified on import. Archive provenance is recorded in [PILOT_SOURCE.json](PILOT_SOURCE.json).

### Files

| Artifact | Location |
| --- | --- |
| Full report, including limitations and scale control | [gqa_e_pilot/RAPORT.md](gqa_e_pilot/RAPORT.md) |
| Original reproduction instructions | [gqa_e_pilot/README.md](gqa_e_pilot/README.md) |
| Model, training loop, and evaluation | [gqa_e_pilot/bench.py](gqa_e_pilot/bench.py) |
| Pinned data preparation | [gqa_e_pilot/prepare.py](gqa_e_pilot/prepare.py) |
| Main experiment runner | [gqa_e_pilot/run_all.py](gqa_e_pilot/run_all.py) |
| Exploratory scale-control runner | [gqa_e_pilot/run_scale_control.py](gqa_e_pilot/run_scale_control.py) |
| Analysis and report generation | [gqa_e_pilot/analyze.py](gqa_e_pilot/analyze.py) |
| Implementation checks and parameter audit | [gqa_e_pilot/checks.py](gqa_e_pilot/checks.py), [gqa_e_pilot/audit.py](gqa_e_pilot/audit.py) |
| Raw main results: 60 runs plus environment records | [gqa_e_pilot/results_final/](gqa_e_pilot/results_final/) |
| Raw scale-control results: six runs plus environment records | [gqa_e_pilot/scale_results/](gqa_e_pilot/scale_results/) |
| Aggregated main and scale-control results | [gqa_e_pilot/summary.json](gqa_e_pilot/summary.json), [gqa_e_pilot/scale_summary.json](gqa_e_pilot/scale_summary.json) |
| Tokenizer, source revisions, and corpus token counts | [gqa_e_pilot/data/](gqa_e_pilot/data/) |
| Historical dependency versions | [gqa_e_pilot/requirements.txt](gqa_e_pilot/requirements.txt) |
| Original SHA-256 file manifest | [gqa_e_pilot/MANIFEST.json](gqa_e_pilot/MANIFEST.json) |

Training, analysis, scale-control, and check logs are included alongside the scripts. Plots: [final comparisons](gqa_e_pilot/comparison.png) and [learning curves](gqa_e_pilot/learning_curves.png).

### Scope and conclusion

The pilot used approximately 1M parameters, three seeds, and 307,200 tokens per language-model run on CPU. Tanh value gating improved perplexity over baseline GQA, but did not demonstrate a benefit over an exploratory parameter-matched control with V scaled by 0.216. Paired confidence intervals between the gate and that control included zero.

The report is about early learning, not full convergence, scaling to large models, or GPU speed. The synthetic tasks have their own limitations: reversal reached 100% for all variants, while the code-retrieval task did not beat a simple majority-digit heuristic on mean whole-answer accuracy.

The separately trained checkpoint described in the root README was not evaluated by these runs. Neither its weights nor the pilot's final weights are included. Source texts and token arrays must be downloaded or regenerated using the pinned data-preparation script.

### Reproduction

Follow the [original environment instructions](gqa_e_pilot/README.md). The recorded dependency versions describe the original runs; no new training or dependency installation was performed during import.

Run from `gqa_e_pilot/`. To generate fresh results without replacing the saved run directories:

```bash
python checks.py
python prepare.py
python bench.py --tasks recall,reverse,wiki,books --variants baseline,matched,tanh,residual,sin_tanh --seeds 11,22,33 --threads 1 --steps-synth 300 --steps-real 600 --out results_reproduction
python bench.py --tasks wiki,books --variants scaled_matched --seeds 11,22,33 --threads 1 --steps-real 600 --out scale_reproduction
```

These commands preserve the model and training settings while executing seeds sequentially; wall times are not directly comparable with the original concurrent CPU runs. Existing result files are skipped, so use new output-directory names for another independent rerun.

`analyze.py` reads the original `results_final/` and `scale_results/` directories and rewrites the original report, summaries, and plots. Analyze a separate working copy and adjust those input paths when summarizing new runs. The original `run_all.py` and `run_scale_control.py` also target the saved directories and overwrite their runner logs.

## Polski

Katalog zachowuje kompletny pilotaż z 2026-11-07 z paczki `gqa_E_polskie_benchmarki.zip`. Wszystkie 101 oryginalnych plików pozostawiono bez zmian. Przy imporcie sprawdzono 100 sum kontrolnych z oryginalnego manifestu. Pochodzenie paczki zapisano w [PILOT_SOURCE.json](PILOT_SOURCE.json).

Zacznij od [pełnego raportu](gqa_e_pilot/RAPORT.md). Powyższa tabela wskazuje kod, instrukcje odtworzenia, surowe wyniki, tokenizer, manifesty danych i zależności. Dołączono także logi, wykresy oraz osobną kontrolę skali V.

Badanie dotyczyło modeli około 1 mln parametrów, trzech seedów i 307 200 tokenów na trening modelu językowego na CPU. Bramka tanh poprawiła perplexity względem bazowego GQA, ale nie wykazała przewagi nad eksploracyjną kontrolą bez E, ze stałym mnożnikiem V równym 0,216 i wyrównaną liczbą parametrów. Sparowane przedziały niepewności różnicy obejmowały zero.

To wyniki wczesnego uczenia. Nie potwierdzają zbieżności, skuteczności większych modeli ani przyspieszenia GPU. Zadania syntetyczne mają osobne ograniczenia: odwracanie osiągnęło 100% dla wszystkich wariantów, a średnia poprawność całych odpowiedzi w odczycie kodu nie przewyższała prostej heurystyki najczęstszej cyfry.

Wyniki nie są oceną osobno trenowanego checkpointu opisanego w głównym README. Nie dołączono jego wag ani końcowych wag modeli z pilotażu. Teksty i tablice tokenów trzeba pobrać lub odtworzyć przez skrypt przygotowania danych z przypiętymi rewizjami.

Instrukcje środowiska znajdziesz w [oryginalnym README](gqa_e_pilot/README.md). Wersje zależności opisują historyczne uruchomienia; podczas importu nie instalowano zależności ani nie wykonywano treningów. Powyższe komendy uruchamiane w `gqa_e_pilot/` zapisują nowe wyniki w osobnych katalogach. Seedy wykonują się kolejno, więc czasów nie należy porównywać wprost z historycznymi równoległymi przebiegami CPU. Istniejące pliki wyników są pomijane; kolejny niezależny przebieg wymaga nowych nazw katalogów wyjściowych.

`analyze.py` czyta historyczne `results_final/` i `scale_results/`, po czym nadpisuje raport, podsumowania i wykresy. Do analizy nowych przebiegów użyj osobnej kopii roboczej i zmień ścieżki wejściowe. Oryginalne skrypty `run_all.py` i `run_scale_control.py` także wskazują zapisane katalogi i nadpisują swoje logi uruchomienia.
