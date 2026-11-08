import pathlib,json,hashlib,random,collections
from bench import sample_synth,Tokenizer
D=pathlib.Path(__file__).parent/'data';out={}
for domain in ['wiki','books']:
 sets=[];counts={}
 for split in ['train','val','test']:
  docs=json.loads((D/f'{domain}_{split}.json').read_text());sets.append({hashlib.sha256(' '.join(o['text'].split()).encode()).hexdigest() for o in docs});counts[split]=len(docs)
 assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
 out[domain]={'documents':counts,'exact_overlap_between_splits':False}
for task in ['recall','reverse']:
 sets=[]
 for split in ['train','val','test']:
  rng=random.Random(77);sets.append({sample_synth(task,rng,split) for _ in range(3000)})
 assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
 out[task]={'exact_overlap_between_sampled_splits':False,'unique_sample_sizes':[len(x) for x in sets]}
out['tokenizer_sha256']=hashlib.sha256((D/'tokenizer.json').read_bytes()).hexdigest()
tok=Tokenizer.from_file(str(D/'tokenizer.json'));example='Zażółć gęślą jaźń. Programowanie i atencja.'
assert tok.decode(tok.encode(example).ids)==example
out['polish_tokenizer_roundtrip']=True
(pathlib.Path(__file__).parent/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
