"""Small causal GQA ablation, all gate variants act on V before attention.
No pretrained weights; not a faithful Qwen3.6 hybrid reproduction.
"""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,math,time,random,hashlib,argparse,pathlib,platform
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders,normalizers
ROOT=pathlib.Path(__file__).parent; D=ROOT/'data'
VARIANTS=['baseline','matched','tanh','residual','sin_tanh']
class RMS(nn.Module):
 def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
 def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.weight
class Attention(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant
  self.q=nn.Linear(128,128,bias=False);self.k=nn.Linear(128,64,bias=False);self.v=nn.Linear(128,64,bias=False);self.o=nn.Linear(128,128,bias=False)
  self.qnorm=RMS(32);self.knorm=RMS(32)
  if variant not in ['baseline','matched','scaled_matched']:self.e=nn.Linear(128,64,bias=False)
 def rope(self,x):
  t=torch.arange(x.size(-2),device=x.device).float();freq=1/(10000**(torch.arange(0,32,2,device=x.device).float()/32))
  z=torch.outer(t,freq);a,b=x[...,0::2],x[...,1::2]
  return torch.stack((a*z.cos()-b*z.sin(),a*z.sin()+b*z.cos()),-1).flatten(-2)
 def forward(self,x):
  B,T,_=x.shape;q=self.q(x).view(B,T,4,32).transpose(1,2);k=self.k(x).view(B,T,2,32).transpose(1,2);v=self.v(x)
  if self.variant=='scaled_matched':v=v*0.216
  if hasattr(self,'e'):
   e=self.e(x);g=e.tanh()
   if self.variant=='residual':g=1+g
   if self.variant=='sin_tanh':g=e.sin()+g
   v=v*g
  q=self.rope(self.qnorm(q));k=self.rope(self.knorm(k));v=v.view(B,T,2,32).transpose(1,2)
  # Explicit repeat is mathematically GQA, without an optimized GQA kernel.
  z=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),is_causal=True)
  return self.o(z.transpose(1,2).reshape(B,T,128))
class Block(nn.Module):
 def __init__(self,variant,ff):
  super().__init__();self.n1=RMS(128);self.n2=RMS(128);self.attn=Attention(variant)
  self.up=nn.Linear(128,ff,bias=False);self.gate=nn.Linear(128,ff,bias=False);self.down=nn.Linear(ff,128,bias=False)
 def forward(self,x):
  x=x+self.attn(self.n1(x));z=self.n2(x);return x+self.down(F.silu(self.gate(z))*self.up(z))
class Model(nn.Module):
 def __init__(self,variant):
  super().__init__();self.emb=nn.Embedding(4096,128);self.blocks=nn.ModuleList([Block(variant,([309,309,310][i] if variant in ['matched','scaled_matched'] else 288)) for i in range(3)]);self.norm=RMS(128)
 def forward(self,x):
  x=self.emb(x)
  for b in self.blocks:x=b(x)
  return F.linear(self.norm(x),self.emb.weight)
def init_model(variant,seed):
 m=Model(variant)
 # Name-keyed initialization keeps all shared tensors identical across variants.
 for name,p in m.named_parameters():
  if p.ndim==1:nn.init.ones_(p)
  else:
   key=int(hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest()[:8],16)
   g=torch.Generator().manual_seed(key);p.data.normal_(0,.02,generator=g)
 return m
NAMES=['Ala','Ola','Jan','Ewa','Adam','Iga','Leon','Maja']
def synth_text(task,rng):
 if task=='recall':
  names=rng.sample(NAMES,4);vals=[rng.randrange(10) for _ in names];i=rng.randrange(4)
  return ' '.join(f'{n} ma kod {v}.' for n,v in zip(names,vals))+f' Jaki kod ma {names[i]}? ',str(vals[i])
 vals=[rng.randrange(10) for _ in range(4)]
 return 'Odwróć kolejność: '+' '.join(map(str,vals))+'. Odpowiedź: ',' '.join(map(str,vals[::-1]))
def synth_split(prompt,answer):
 bucket=int(hashlib.sha256((prompt+answer).encode()).hexdigest()[:8],16)%100
 return 'test' if bucket<10 else 'val' if bucket<20 else 'train'
def sample_synth(task,rng,split):
 while True:
  prompt,answer=synth_text(task,rng)
  if synth_split(prompt,answer)==split:return prompt,answer
def prep_tokens():
 if not (D/'tokenizer.json').exists():
  tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.normalizer=normalizers.NFC();tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
  texts=[]
  for domain in ['wiki','books']:texts.extend(o['text'] for o in json.loads((D/f'{domain}_train.json').read_text()))
  rng=random.Random(19)
  texts.extend(''.join(sample_synth(task,rng,'train')) for task in ['recall','reverse'] for _ in range(2000))
  tok.train_from_iterator(texts,trainers.BpeTrainer(vocab_size=4096,special_tokens=['<pad>','<unk>','<eos>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False));tok.save(str(D/'tokenizer.json'))
 tok=Tokenizer.from_file(str(D/'tokenizer.json'));assert tok.get_vocab_size()==4096
 stats={}
 for domain in ['wiki','books']:
  for split in ['train','val','test']:
   path=D/f'{domain}_{split}.npy'
   if not path.exists():
    docs=json.loads((D/f'{domain}_{split}.json').read_text());ids=[]
    for doc in docs:ids.extend(tok.encode(doc['text']).ids+[2])
    np.save(path,np.array(ids,dtype=np.int32))
   stats[f'{domain}_{split}']=int(len(np.load(path)))
 (D/'token_counts.json').write_text(json.dumps(stats,indent=2));return tok
class Batcher:
 def __init__(self,task,split,seed,tok,batch,seq):
  self.task=task;self.split=split;self.rng=random.Random(seed);self.tok=tok;self.batch=batch;self.seq=seq
  if task in ['wiki','books']:self.tokens=np.load(D/f'{task}_{split}.npy')
 def next(self):
  if self.task in ['wiki','books']:
   rows=[self.tokens[(p:=self.rng.randrange(len(self.tokens)-self.seq-1)):p+self.seq+1] for _ in range(self.batch)]
   z=torch.tensor(np.stack(rows),dtype=torch.long);return z[:,:-1],z[:,1:]
  rows=[];labels=[]
  for _ in range(self.batch):
   prompt,answer=sample_synth(self.task,self.rng,self.split)
   # Separately encode prompt/answer for explicit supervised boundary.
   a=self.tok.encode(prompt).ids;b=self.tok.encode(answer).ids+[2];ids=a+b
   assert len(ids)<=self.seq+1
   x=ids[:-1];y=[-100]*(len(a)-1)+b
   rows.append(x+[0]*(self.seq-len(x)));labels.append(y+[-100]*(self.seq-len(y)))
  return torch.tensor(rows),torch.tensor(labels)
@torch.no_grad()
def evaluate(m,batcher,n):
 m.eval();loss=correct=count=exact=examples=0
 for _ in range(n):
  x,y=batcher.next();z=m(x);mask=y!=-100;pred=z.argmax(-1)
  loss+=F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum').item();count+=mask.sum().item();correct+=((pred==y)&mask).sum().item()
  exact+=((pred==y)|~mask).all(-1).sum().item();examples+=x.size(0)
 m.train();return {'nll':loss/count,'ppl':math.exp(loss/count),'token_accuracy':correct/count,'teacher_forced_exact':exact/examples,'target_tokens':count}
@torch.no_grad()
def greedy_accuracy(m,task,tok,n=256):
 m.eval();rng=random.Random(999);groups={};correct=0
 for _ in range(n):
  prompt,answer=sample_synth(task,rng,'test');a=tok.encode(prompt).ids;b=tok.encode(answer).ids+[2]
  groups.setdefault((len(a),len(b)),[]).append((a,b))
 for (_,length),rows in groups.items():
  for start in range(0,len(rows),16):
   batch=rows[start:start+16];x=torch.tensor([a for a,b in batch]);target=torch.tensor([b for a,b in batch]);generated=[]
   for _ in range(length):
    nxt=m(x)[:,-1].argmax(-1);generated.append(nxt);x=torch.cat((x,nxt[:,None]),1)
   correct+=(torch.stack(generated,1)==target).all(-1).sum().item()
 m.train();return correct/n
@torch.no_grad()
def gate_diagnostics(m,variant,x):
 records=[];handles=[]
 def hook(module,inputs,e):
  g=e.tanh()
  if variant=='residual':g=1+g
  if variant=='sin_tanh':g=e.sin()+g
  records.append({'mean':g.mean().item(),'std':g.std().item(),'negative_fraction':(g<0).float().mean().item(),'near_zero_fraction':(g.abs()<.05).float().mean().item(),'tanh_saturated_fraction':(e.tanh().abs()>.95).float().mean().item()})
 for b in m.blocks:
  if hasattr(b.attn,'e'):handles.append(b.attn.e.register_forward_hook(hook))
 m(x)
 for h in handles:h.remove()
 return records
def run(args):
 torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);tok=prep_tokens();results=[]
 out=ROOT/args.out;out.mkdir(exist_ok=True)
 (out/f'environment_{args.seeds}.json').write_text(json.dumps({'torch':torch.__version__,'python':platform.python_version(),'cuda':torch.cuda.is_available(),'threads':args.threads,'args':vars(args)},indent=2))
 for task in args.tasks.split(','):
  steps=args.steps_real if task in ['wiki','books'] else args.steps_synth
  for seed in map(int,args.seeds.split(',')):
   for variant in args.variants.split(','):
    path=out/f'{task}_{seed}_{variant}.json'
    if path.exists():continue
    m=init_model(variant,seed);optim=torch.optim.AdamW(m.parameters(),lr=args.lr,weight_decay=.1,betas=(.9,.95))
    train=Batcher(task,'train',10000+seed,tok,args.batch,args.seq);curve=[];start=time.perf_counter();maxgrad=0;target_count=0;clipped=0
    for step in range(steps):
     lr=args.lr*min(1,(step+1)/max(1,int(steps*.1)))*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
     for group in optim.param_groups:group['lr']=lr
     x,y=train.next();target_count+=(y!=-100).sum().item();optim.zero_grad(set_to_none=True);z=m(x);loss=F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100)
     assert torch.isfinite(loss);loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1.0);assert torch.isfinite(gn);maxgrad=max(maxgrad,gn.item());clipped+=int(gn.item()>1);optim.step()
     if step in [0,steps//4-1,steps//2-1,3*steps//4-1,steps-1]:curve.append({'step':step+1,'train_loss':loss.item(),'elapsed_s':time.perf_counter()-start,**evaluate(m,Batcher(task,'val',777,tok,args.batch,args.seq),args.eval_batches)})
    elapsed=time.perf_counter()-start
    test=evaluate(m,Batcher(task,'test',888,tok,args.batch,args.seq),args.test_batches)
    if task in ['recall','reverse']:test['greedy_exact']=greedy_accuracy(m,task,tok)
    result={'task':task,'seed':seed,'variant':variant,'params':sum(p.numel() for p in m.parameters()),'steps':steps,'supervised_tokens':target_count,'input_positions':steps*args.batch*args.seq,'wall_seconds_with_validation':elapsed,'max_gradient_norm':maxgrad,'clipped_steps':clipped,'gate_diagnostics':gate_diagnostics(m,variant,Batcher(task,'test',888,tok,args.batch,args.seq).next()[0]),'test':test,'curve':curve}
    path.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['task','seed','variant','params','wall_seconds_with_validation']}|test),flush=True)
    if args.checkpoints:torch.save(m.state_dict(),out/f'{task}_{seed}_{variant}.pt')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tasks',default='recall,reverse,wiki,books');p.add_argument('--variants',default=','.join(VARIANTS));p.add_argument('--seeds',default='11,22,33');p.add_argument('--steps-synth',type=int,default=300);p.add_argument('--steps-real',type=int,default=600);p.add_argument('--batch',type=int,default=8);p.add_argument('--seq',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--lr',type=float,default=.002);p.add_argument('--eval-batches',type=int,default=8);p.add_argument('--test-batches',type=int,default=32);p.add_argument('--out',default='results');p.add_argument('--checkpoints',action='store_true');run(p.parse_args())
