import urllib.request as u, json,gzip,hashlib,random,pathlib,concurrent.futures
D=pathlib.Path(__file__).parent/'data'; D.mkdir(exist_ok=True)
def get(url):
 with u.urlopen(url,timeout=120) as r:return r.read()
def wiki():
 repo='radlab/wikipedia-pl';rev='63a2c8deb4a47f144970d1067008c9d244f13cec'
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/corpus.jsonl.gz'
 docs=[]
 with u.urlopen(url,timeout=120) as r:
  with gzip.GzipFile(fileobj=r) as z:
   for line in z:
    o=json.loads(line); t=o.get('text',o.get('content',''))
    if len(t)>300:docs.append({'id':str(o.get('id',len(docs))), 'text':t[:30000]})
    if len(docs)>=5000:break
 print('wiki',len(docs),flush=True)
 return docs,{'repo':repo,'revision':rev,'sampling':'first 5000 records with >300 chars; text capped at 30000 chars'}
def books():
 repo='ArielUW/wolnelektury'; rev='cfc65a4c42613a6c025399240f6b5e4faae55630'
 # Small deliberately selected Polish-language fiction corpus; splits by author.
 groups={'train':['alcott-male-kobietki.txt','andersen-mala-syrenka.txt','andersen-tylko-grajek.txt','amicis-serce.txt','20-000-mil-podmorskiej-zeglugi.txt','aleksander-dumas-trzej-muszkieterowie-tom-2.txt'], 'val':['antek.txt'], 'test':['lord-jim-tom-pierwszy.txt','lord-jim-tom-drugi.txt']}
 out={}
 for split,names in groups.items():
  def fetch(n):return {'id':n,'text':get(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{n}').decode('utf-8-sig')}
  with concurrent.futures.ThreadPoolExecutor(4) as pool:out[split]=list(pool.map(fetch,names))
 print('books', {k:sum(len(x['text']) for x in v) for k,v in out.items()},flush=True)
 return out,{'repo':repo,'revision':rev,'sampling':'selected works; train authors Alcott/Andersen/Amicis/Verne/Dumas; val Prus; test Conrad'}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  fw=pool.submit(wiki);fb=pool.submit(books);wd,wm=fw.result();bd,bm=fb.result()
 seen=set();ws={k:[] for k in ['train','val','test']}
 for doc in wd:
  h=hashlib.sha256(' '.join(doc['text'].split()).encode()).hexdigest()
  if h in seen:continue
  seen.add(h); bucket=int(h[:8],16)%100;split='test' if bucket<10 else 'val' if bucket<20 else 'train';ws[split].append(doc)
 for name,splits in [('wiki',ws),('books',bd)]:
  for split,docs in splits.items():
   (D/f'{name}_{split}.json').write_text(json.dumps(docs,ensure_ascii=False))
 meta={'wiki':wm,'books':bm,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.json') if p.name!='sources.json'}}
 (D/'sources.json').write_text(json.dumps(meta,indent=2)); print('ready',flush=True)
