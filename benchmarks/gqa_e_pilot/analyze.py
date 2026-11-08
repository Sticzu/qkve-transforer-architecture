import json,pathlib,statistics,math,hashlib,zipfile,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).parent
variants=['baseline','matched','tanh','residual','sin_tanh']; labels={'baseline':'GQA bazowe','matched':'GQA + szerszy FFN','tanh':'V · tanh(E)','residual':'V · (1+tanh(E))','sin_tanh':'V · (sin(E)+tanh(E))'}
tasks=['recall','reverse','wiki','books'];tasknames={'recall':'Odczyt kodu','reverse':'Odwracanie cyfr','wiki':'Wikipedia PL','books':'Literatura PL'}
rows=[json.loads(p.read_text()) for p in (ROOT/'results_final').glob('*.json') if not p.name.startswith('environment')]
assert len(rows)==60, f'Expected 60 completed runs, got {len(rows)}'
by={(r['task'],r['variant'],r['seed']):r for r in rows}
def vals(t,v,k):return [by[t,v,s]['test'][k] for s in [11,22,33]]
def mean_sd(a):return f'{statistics.mean(a):.3f} ± {statistics.stdev(a):.3f}'
summary=[]
for t in tasks:
 for v in variants:
  n=vals(t,v,'nll');r={'task':t,'variant':v,'params':by[t,v,11]['params'],'nll_mean':statistics.mean(n),'nll_sd':statistics.stdev(n),'ppl_mean':statistics.mean(vals(t,v,'ppl')),'ppl_sd':statistics.stdev(vals(t,v,'ppl'))}
  if t in ['recall','reverse']:r.update(exact_mean=statistics.mean(vals(t,v,'greedy_exact')),exact_sd=statistics.stdev(vals(t,v,'greedy_exact')))
  for control in ['baseline','matched']:
   d=[a-b for a,b in zip(n,vals(t,control,'nll'))];mu=statistics.mean(d);err=4.3026527299*statistics.stdev(d)/math.sqrt(3)
   r[f'delta_{control}']=mu;r[f'ci95_{control}']=[mu-err,mu+err]
  summary.append(r)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
colors=['#667085','#b692f6','#f79009','#12b76a','#2e90fa']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,t in zip(axs.flat,tasks):
 metric='greedy_exact' if t in ['recall','reverse'] else 'ppl'
 means=[statistics.mean(vals(t,v,metric)) for v in variants];sd=[statistics.stdev(vals(t,v,metric)) for v in variants]
 if metric=='greedy_exact':means=np.array(means)*100;sd=np.array(sd)*100
 plotlabels=['GQA','GQA\n+FFN','tanh','1+tanh','sin+tanh']
 if t in ['wiki','books']:
  sr=[json.loads(p.read_text()) for p in (ROOT/'scale_results').glob(t+'_*.json')]
  assert len(sr)==3
  sa=[r['test']['ppl'] for r in sr];means=list(means)+[statistics.mean(sa)];sd=list(sd)+[statistics.stdev(sa)];plotlabels+=['stała\n0,216']
 ax.bar(range(len(means)),means,yerr=sd,color=colors+['#ee46bc'],capsize=3)
 if t=='recall':
  ax.axhline(35.9375,color='#344054',linestyle='--',linewidth=1,label='Prosta reguła: 35,94%');ax.legend(loc='lower right',fontsize=8)
 ax.set_xticks(range(len(means)),plotlabels);ax.set_title(tasknames[t]);ax.set_ylabel('Cała odpowiedź poprawna [%] ↑' if metric=='greedy_exact' else 'Perplexity ↓');ax.grid(axis='y',alpha=.15)
fig.suptitle('GQA ~1M: trzy seedy, średnia ± odchylenie standardowe\nPilotaż: 300 kroków syntetycznych / 600 kroków LM',fontsize=14)
fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,t in zip(axs.flat,tasks):
 for v,c in zip(variants,colors):
  curves=[by[t,v,s]['curve'] for s in [11,22,33]];steps=[p['step'] for p in curves[0]];a=np.array([[p['nll'] for p in curve] for curve in curves]);ax.plot(steps,a.mean(0),label=labels[v],color=c);ax.fill_between(steps,a.mean(0)-a.std(0,ddof=1),a.mean(0)+a.std(0,ddof=1),color=c,alpha=.15)
 ax.set_title(tasknames[t]);ax.set_xlabel('Krok treningu');ax.set_ylabel('Walidacyjny NLL ↓');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8);fig.savefig(ROOT/'learning_curves.png',dpi=160);plt.close(fig)
head='''# Czy dodatkowa projekcja E pomaga atencji GQA?

Raport z rzeczywiście wykonanych treningów na CPU. Data: 2026-11-07.

## Zakres i najważniejsze zastrzeżenie

To pilotaż wczesnego uczenia modelu około 1 mln parametrów. **Nie jest to miniaturowa wierna kopia Qwen3.6-27B ani dowód dotyczący skalowania do 27B.** Nie użyto jego wag. Oficjalny Qwen3.6 ma hybrydowy układ 3× Gated DeltaNet + 1× Gated Attention, encoder obrazu i inne elementy, których nie obejmuje ten eksperyment. Badamy klasyczny dekoder z przyczynową atencją GQA, RMSNorm, RoPE, QK RMSNorm i SwiGLU.

Wzór `sin()` jest niekompletny. Przyjęto **sin(xW_E)**; inny argument oznacza inny eksperyment. Bramkowanie znajduje się na V, przed mnożeniem przez wagi atencji. x oznacza tutaj wyjście pre-RMSNorm bloku. Ta bramka zależy od tokenu źródłowego V; różni się od bramkowania wyniku SDPA zależnego od zapytania.

## Konfiguracja

- 3 bloki, d_model=128, 4 głowy Q, 2 głowy KV, head_dim=32; GQA 2:1.
- Kontekst 64 tokeny, batch 8, słownik 4096, wspólna macierz embeddingów i wyjścia.
- FFN 288; kontrola o równej liczbie parametrów: FFN 309/309/310.
- Bez biasów i dropout. Pełne RoPE, theta=10000. E ma wymiar 128×64 na warstwę; bramka współdzielona przez zapytania korzystające z tej samej głowy KV.
- GQA realizowane przez powtórzenie K/V przed SDPA: poprawność matematyczna GQA, ale bez korzyści pamięciowych wyspecjalizowanego kernela GQA. Brak KV cache.
- Bazowe GQA: **1 004 608** parametrów. Każdy wariant E oraz szerszy FFN: **1 029 184**. Różnica 24 576, czyli 2,446%.
- AdamW, lr maks. 0,002, betas=(0,9;0,95), weight decay=0,1, warmup 10%, cosine do 10% lr, clipping normy gradientu do 1.
- Wszystkie macierze N(0;0,02²), wagi RMSNorm=1. E ma tę samą losową inicjalizację w trzech wariantach. Nie testowano osobno zerowej inicjalizacji residual E.
- Seedy 11/22/33. Inicjalizacja zależy od nazwy parametru: wspólne tensory mają dokładnie te same wartości. Te same minibatch'e dla sparowanych seedów.
- 300 kroków/zadanie syntetyczne, 600 kroków/korpus rzeczywisty. **307 200 tokenów treningowych na jeden trening LM**; to znacznie za mało do oceny docelowego potencjału modelu 1M.
- 5 punktów walidacji; oceniamy końcowy checkpoint, bez wyboru po zbiorze testowym. 16 384 losowo próbkowane tokeny testowe na korpus/run. Okna mogą nakładać się; nie jest to ewaluacja całego korpusu.
- 256 syntetycznych przykładów testowych generowanych autoregresyjnie, z wymaganym poprawnym EOS. Zbiór testowy wspólny dla seedów. Trzy seedy mierzą zmienność treningu, nie pełną niepewność wynikającą z doboru danych.
- Trzy procesy CPU po jednym wątku. Czasy wall-clock obejmują walidację i współdzielenie CPU, więc **nie są wiarygodnym benchmarkiem przyspieszenia sprzętowego**.

## Polski tokenizer i rzeczywiste dane

Wspólny byte-level BPE 4096, NFC, wytrenowany wyłącznie na polskojęzycznych częściach treningowych obu korpusów i treningowych szablonach syntetycznych. Test/val nie uczestniczą w uczeniu BPE. To nowy mały polski tokenizer, nie tokenizer Qwena ani gotowy duży tokenizer Bielika. Pełny alfabet bajtowy zapobiega nieznanym polskim znakom. Przy 50 tys. tokenów sama macierz embeddingów o szerokości 128 miałaby 6,4 mln wag.

**Wikipedia:** radlab/wikipedia-pl, dump 2023-04-01. Pierwsze 5000 artykułów o długości >300 znaków, maksymalnie 30 000 znaków/artykuł. Nie jest to reprezentatywne losowanie całej Wikipedii. Exact dedup po znormalizowanych białych znakach, podział dokumentów hashem SHA256 na 80/10/10. Bez deduplikacji semantycznej i near-duplicate.

**Literatura:** ArielUW/wolnelektury. Trening: Alcott, Andersen, Amicis, Verne, Dumas (6 utworów), walidacja: Prus — „Antek”, test: Conrad — „Lord Jim”, oba tomy. Podział według autorów, bez przechodzenia fragmentów tej samej książki między podziałami. Mała próbka, głównie starszy język i tłumaczenia; wyniku nie należy uogólniać na współczesną polszczyznę. Zachowano teksty korpusu z nagłówkami/metadanymi. Repozytoria, konkretne rewizje, nazwy utworów i hashe plików zapisano w data/sources.json.

Nie trenowano najpierw na syntetyce, a potem na korpusie. **Każde zadanie/korpus trenowane od zera**, żeby nie mieszać transferu z efektem architektury.

## Wyniki końcowe

Niżej średnia ± odchylenie standardowe z 3 seedów. Niższe NLL/perplexity oznacza lepszy wynik. Perplexity porównujemy wyłącznie w obrębie tego samego korpusu i wspólnego tokenizera. Dla syntetyki ważniejsza jest poprawność całej swobodnie wygenerowanej odpowiedzi niż token accuracy obejmująca łatwy EOS.

'''
report=head
counts=json.loads((ROOT/'data/token_counts.json').read_text())
report+='Liczba tokenów w przygotowanych korpusach (całe podziały, nie liczba tokenów użytych w jednym treningu):\n\n| Korpus | Train | Val | Test |\n|---|---:|---:|---:|\n'
for domain in ['wiki','books']:
 report+='| '+tasknames[domain]+' | '+' | '.join(str(counts[f'{domain}_{split}']) for split in ['train','val','test'])+' |\n'
report+='\n'
for t in tasks:
 report+=f'### {tasknames[t]}\n\n| Wariant | NLL test | Perplexity test |'+(' Cała odpowiedź [%] |' if t in ['recall','reverse'] else '')+'\n|---|---:|---:|'+('---:|' if t in ['recall','reverse'] else '')+'\n'
 for v in variants:
  report+=f'| {labels[v]} | {mean_sd(vals(t,v,"nll"))} | {mean_sd(vals(t,v,"ppl"))} |'
  if t in ['recall','reverse']:report+=' '+mean_sd([100*x for x in vals(t,v,'greedy_exact')])+' |'
  report+='\n'
 report+='\n'
report+='''![Porównanie końcowe](comparison.png)

![Krzywe walidacyjne](learning_curves.png)

Dodatkowa kontrola syntetyczna: wybieranie najczęstszej cyfry bez rozumienia pytania daje **35,94%** na tym samym teście odczytu kodu (92/256). Wynik na tym poziomie nie świadczy o opanowaniu przypisania osoba→kod. Zwykłe skopiowanie cyfr bez odwrócenia daje 0/256 na użytym teście odwracania.

## Sparowane różnice NLL

Ujemna różnica oznacza poprawę względem wskazanej kontroli. Orientacyjny 95% przedział t-Studenta z 3 sparowanych seedów, df=2; szeroki i wrażliwy na założenie normalności. Bez korekty wielokrotnych porównań. To opis niepewności pilotażu, nie potwierdzenie odkrycia.

| Zadanie | E | Δ vs bazowe [95% CI] | Δ vs równe parametry [95% CI] |
|---|---|---:|---:|
'''
for r in summary:
 if r['variant'] in ['baseline','matched']:continue
 report+=f'| {tasknames[r["task"]]} | {labels[r["variant"]]} |'
 for c in ['baseline','matched']:
  lo,hi=r[f'ci95_{c}'];report+=f' {r[f"delta_{c}"]:+.4f} [{lo:+.4f}, {hi:+.4f}] |'
 report+='\n'
report+='''
## Interpretacja mechanizmu

Dla z=xW_E:

| Bramka | Zakres | g(0) | g′(0) | Konsekwencja |
|---|---|---:|---:|---|
| tanh(z) | (-1,1) | 0 | 1 | może odwracać znak i na starcie tłumi V |
| 1+tanh(z) | (0,2) | 1 | 1 | zachowuje bezpośrednią ścieżkę V; przy E=0 odtwarza bazową atencję |
| sin(z)+tanh(z) | ograniczona przez (-2,2) | 0 | 2 | większe nachylenie przy zerze i niemonotoniczność; brak składnika tożsamościowego |

Nie tylko liczba parametrów odróżnia te funkcje: zmieniają skalę wyjścia i gradientów. Dlatego sam wynik nie izoluje np. „korzyści z sinusoidy”. Potrzebne byłyby ablacje wyrównujące początkową wariancję bramki oraz osobne strojenie learning rate z jednakowym budżetem dla wszystkich wariantów.

Zero-inicjalizacja E jest naturalną dodatkową ablacją dla 1+tanh. Dla tanh i sin+tanh wyzerowałaby początkowo V_mod, blokując w pierwszym kroku gradienty części gałęzi atencji (ale nie samego E). W wykonanych treningach E nie było zerowane.

W literaturze istnieją pozytywne wyniki dla bramkowania **wyjścia SDPA sigmoidem** zależnym od zapytania. Nie dowodzą one przewagi konkretnego bramkowania V z tanh lub sin. Kontrola sigmoid-after-SDPA nie została tutaj uruchomiona.

## Co ten test sprawdza, a czego nie

Wykonano 60 treningów głównych i 6 dodatkowych kontroli skali; sprawdzono kauzalność, przepływ skończonych gradientów, dokładną liczbę parametrów kontroli i tożsamość residual gate przy E=0. Zadania syntetyczne mają rozłączne przykłady train/val/test według hasha całego przykładu. BPE zapisano i użyto wspólnie we wszystkich wariantach.

Nie sprawdzono pełnej zbieżności, modelu hybrydowego Qwena, długiego kontekstu, większego modelu, zmiany proporcji Q/KV, transferu do zadań klasyfikacji, benchmarków wiedzy ani kosztu zoptymalizowanych kerneli GPU. Brak dowodów na konkretny procent przyspieszenia uczenia. Same wyniki po tak małej liczbie tokenów nie uzasadniają wdrożenia E do dużego modelu.

Do mocniejszej weryfikacji: przynajmniej 20–50 mln tokenów na trening LM, większa i losowana próbka polskich tekstów, ≥5 seedów, dłuższy kontekst, wspólne strojenie lr oraz ablacje inicjalizacji/skali i sigmoid-after-SDPA. To propozycja kolejnego eksperymentu, nie wynik wykonany tutaj.

## Powtarzalność

`python prepare.py` pobiera przypięte rewizje danych. Pakiet zawiera tokenizer i manifest danych. Najpierw uruchom `python prepare.py`, następnie `python run_all.py`; tablice tokenów powstaną automatycznie. Same teksty i tablice tokenów nie są dołączone. `checks.py` sprawdza implementację, `analyze.py` generuje raport. `results_final/` zawiera surowe wyniki każdego treningu i parametry środowiska. Brak zapisanych wag końcowych — odtworzenie modeli wymaga ponownego treningu. Nie utożsamiać innych katalogów z wynikami finalnymi; pakiet obejmuje wyłącznie finalny eksperyment.

Źródła:

- https://huggingface.co/Qwen/Qwen3.6-27B — architektura referencyjna; pobrana konfiguracja w qwen_reference_config.json.
- https://huggingface.co/datasets/radlab/wikipedia-pl — karta: CC BY 4.0, dump Wikipedii.
- https://huggingface.co/datasets/ArielUW/wolnelektury — teksty Wolnych Lektur, autorzy/tytuły/tłumacze zachowani w plikach. Zob. warunki użycia https://wolnelektury.pl/media/chunks/attachment/Ksi%C4%99ga_atrybucji_q0K6Cc5.pdf .
- https://arxiv.org/abs/2505.06708 — Gated Attention for Large Language Models, Qiu i in.; odmienny mechanizm bramkowania.
'''
scale_rows=[json.loads(p.read_text()) for p in (ROOT/'scale_results').glob('*.json') if not p.name.startswith('environment')]
assert len(scale_rows)==6, f'Expected 6 scale controls, got {len(scale_rows)}'
scale_by={(r['task'],r['seed']):r for r in scale_rows}
scale_summary=[]
report+='\n## Dodatkowa kontrola amplitudy V (eksploracyjna)\n\nDodana po obejrzeniu wstępnego wyniku tanh na Wikipedii: szeroki FFN o tej samej liczbie parametrów co E, bez E, ze stałym mnożnikiem V=0,216. Ta liczba jest zaokrągleniem początkowego RMS tanh(z)≈0,2158735 dla z~N(0;128·0,02²), a nie wynikiem strojenia na walidacji. Wyrównuje przybliżoną początkową amplitudę, ale nie średnią, znaki ani adaptacyjność bramki. Identyczny protokół 600 kroków, 3 seedy. Nie jest częścią pierwotnie ustalonego zestawu porównań.\n\n| Korpus | PPL stała skala 0,216 | PPL tanh(E) | Sparowane ΔNLL tanh − stała [95% CI] |\n|---|---:|---:|---:|\n'
for t in ['wiki','books']:
 a=[scale_by[t,seed]['test']['ppl'] for seed in [11,22,33]]
 n=[scale_by[t,seed]['test']['nll'] for seed in [11,22,33]]
 d=[u-v for u,v in zip(vals(t,'tanh','nll'),n)];mu=statistics.mean(d);err=4.3026527299*statistics.stdev(d)/math.sqrt(3)
 scale_summary.append({'task':t,'variant':'scaled_matched','ppl_mean':statistics.mean(a),'ppl_sd':statistics.stdev(a),'tanh_delta_nll':mu,'ci95':[mu-err,mu+err]})
 report+=f'| {tasknames[t]} | {mean_sd(a)} | {mean_sd(vals(t,"tanh","ppl"))} | {mu:+.4f} [{mu-err:+.4f}, {mu+err:+.4f}] |\n'
(ROOT/'scale_summary.json').write_text(json.dumps(scale_summary,indent=2))
conclusion='## Wynik w tym budżecie\n\n'
for t in ['wiki','books']:
 winner=min(variants,key=lambda v:statistics.mean(vals(t,v,'nll')))
 reduction=100*(1-statistics.mean(vals(t,winner,'ppl'))/statistics.mean(vals(t,'baseline','ppl')))
 conclusion+=f'- {tasknames[t]}: spośród pięciu wariantów głównych najniższy średni NLL ma **{labels[winner]}**. Średnie PPL jest o **{reduction:.2f}%** niższe niż bazowe GQA.\n'
conclusion+='\n**Stałe osłabienie V bez E uzyskało niższą średnią perplexity niż tanh(E) na obu korpusach.** Sparowane przedziały różnicy obejmują zero, więc nie ogłaszamy statystycznej przewagi kontroli. Wyniki nie pokazują przewagi E nad prostą kontrolą amplitudy; pokazują, że skala gałęzi atencji jest istotną alternatywną przyczyną poprawy.\n'
conclusion+='\nTo wynik wczesnego uczenia przy jednym harmonogramie lr, a nie ranking architektur po pełnym treningu. Kontrola z równą liczbą wag i osobna kontrola amplitudy są przedstawione niżej.\n\n'
report=report.replace('## Zakres i najważniejsze zastrzeżenie',conclusion+'## Zakres i najważniejsze zastrzeżenie')
(ROOT/'RAPORT.md').write_text(report)
print(json.dumps(summary,indent=2))
