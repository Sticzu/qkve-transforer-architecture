import torch
from bench import Model,init_model,Attention
# Causality; exact parameter matching; residual identity at zero E;
# all gates finite gradients and GQA dimensions.
torch.set_num_threads(2)
a=init_model('baseline',1);r=init_model('residual',1)
for b in r.blocks:b.attn.e.weight.data.zero_()
x=torch.randint(0,4096,(2,16));assert torch.allclose(a(x),r(x),atol=1e-7)
y=x.clone();y[:,8:]=torch.randint(0,4096,(2,8));assert torch.allclose(a(x)[:,:8],a(y)[:,:8],atol=1e-6)
counts={v:sum(p.numel() for p in Model(v).parameters()) for v in ['baseline','matched','tanh','residual','sin_tanh']}
assert counts['matched']==counts['residual']==counts['tanh']==counts['sin_tanh']
for v in counts:
 m=init_model(v,1);m(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
print('PASS causality, parameter matching, residual identity, gradients',counts)
# Independent dense reference for causal grouped attention, including all gates.
import math
for variant in ['baseline','tanh','residual','sin_tanh']:
 a=init_model(variant,13).blocks[0].attn;x=torch.randn(2,7,128);B,T,_=x.shape
 q=a.rope(a.qnorm(a.q(x).view(B,T,4,32).transpose(1,2)));k=a.rope(a.knorm(a.k(x).view(B,T,2,32).transpose(1,2)));v=a.v(x)
 if hasattr(a,'e'):
  e=a.e(x);g=e.tanh()
  if variant=='residual':g=g+1
  if variant=='sin_tanh':g=g+e.sin()
  v=v*g
 v=v.view(B,T,2,32).transpose(1,2);heads=[]
 for h in range(4):
  score=q[:,h]@k[:,h//2].transpose(-1,-2)/math.sqrt(32)
  score=score.masked_fill(torch.ones(T,T,dtype=torch.bool).triu(1),float('-inf'))
  heads.append(score.softmax(-1)@v[:,h//2])
 ref=a.o(torch.stack(heads,2).reshape(B,T,128));assert torch.allclose(a(x),ref,atol=1e-6)
print('PASS explicit dense causal GQA reference for all gates')
s=init_model('scaled_matched',1);b=init_model('matched',1)
assert sum(p.numel() for p in s.parameters())==counts['tanh']
x=torch.randn(2,8,128)
assert torch.allclose(s.blocks[0].attn(x),b.blocks[0].attn(x)*.216,atol=1e-6)
print('PASS constant-scale control, equal parameter budget')
