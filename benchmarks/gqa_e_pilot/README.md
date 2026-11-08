# Eksperyment E w GQA, polski model ~1M

Zacznij od RAPORT.md. Surowe wyniki: results_final/. Wykresy: comparison.png i learning_curves.png.

## Odtworzenie

Python 3.11+, środowisko z PyTorch CPU. Wersje użyte w requirements.txt i results_final/environment_*.json.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python checks.py
python prepare.py
python run_all.py
python run_scale_control.py
python analyze.py
```

Skrypt pomija ukończone treningi w results_final/. Żeby wykonać je ponownie, zmień `--out` w run_all.py i odpowiednią nazwę katalogu w analyze.py albo przenieś results_final/ poza katalog projektu. Nie myl dołączonych wyników z nowym uruchomieniem.

Tokenizer i manifest użytych danych są dołączone; same teksty i tablice tokenów pobiera/przygotowuje skrypt. `python prepare.py` odtwarza podzbiory z przypiętych rewizji Hugging Face; wymaga internetu. `bench.py` tworzy tokenizer/tokeny tylko jeśli nie istnieją — po celowej zmianie danych usuń cache tokenizera i .npy. Nie należy tworzyć innego tokenizera dla każdego wariantu.

Przykład dłuższego treningu, NIE wykonany w raporcie:

```bash
python bench.py --tasks wiki,books --seeds 11,22,33,44,55 --steps-real 40000 --out longer_results --checkpoints
```

To 20,48M tokenów na trening; domyślnie nadal CPU i kontekst 64. Program nie wybiera automatycznie GPU. Do eksperymentu GPU należy przenieść model i batche na urządzenie oraz ponownie zweryfikować deterministyczność i pomiar czasu.

Nie ma w pakiecie wag Qwen ani wag końcowych modeli. Jest kompletny kod treningu i pobierania danych, tokenizer, manifest korpusów, logi, metryki i raport.

Dane: radlab/wikipedia-pl (karta CC BY 4.0), ArielUW/wolnelektury (Wolne Lektury; szczegóły i linki w raporcie). Nazwy autorów, utworów i tłumaczy pozostawiono w źródłowych tekstach. Kod eksperymentu wygenerowany na potrzeby tej analizy.
