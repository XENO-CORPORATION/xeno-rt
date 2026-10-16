"""Pinned CUDA Q/K normalization, both RoPE layouts, and activation boundaries.

All inputs and weights are synthetic. No native result participates in fixture
generation; the hash-pinned Fish modules and actual Torch operators run on CUDA.
"""
import argparse,json,pathlib,types,torch
import torch.nn.functional as F
from drama3_upstream_reference import HASHES,load
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();load(a.source)
from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm,GatedRMSNorm,apply_neox_rope
from fish_speech.models.text2semantic.fish_omni.tower import TowerAttention
generator=torch.Generator().manual_seed(541566)
def sample(shape,scale=1):return (torch.randn(shape,generator=generator)*scale).bfloat16().cuda()
def values(x):return x.float().cpu().flatten().tolist()
rope=[];gates=[]
with torch.inference_mode():
    dim,rotary,theta=256,64,10000000.0
    q=sample((1,1,16,dim));k=sample((1,1,2,dim));qw=sample((dim,),.07);kw=sample((dim,),.07)
    qnorm=ZeroCenteredRMSNorm(dim,1e-6,dtype=torch.bfloat16,device='cuda');qnorm.weight.copy_(qw)
    knorm=ZeroCenteredRMSNorm(dim,1e-6,dtype=torch.bfloat16,device='cuda');knorm.weight.copy_(kw)
    for interleaved in [False,True]:
        for position in [1,33]:
            positions=torch.tensor([position],device='cuda')
            if interleaved:
                owner=types.SimpleNamespace(config=types.SimpleNamespace(rotary_dim=rotary,rope_base=theta))
                eq=TowerAttention.rotate(owner,qnorm(q)[0],positions)
                ek=TowerAttention.rotate(owner,knorm(k)[0],positions)
            else:
                frequency=1.0/(theta**(torch.arange(0,rotary,2,device='cuda').float()/rotary))
                angle=positions[:,None]*frequency[None,:];angle=torch.cat((angle,angle),-1)
                eq=apply_neox_rope(qnorm(q),angle.cos(),angle.sin())
                ek=apply_neox_rope(knorm(k),angle.cos(),angle.sin())
            rope.append(dict(dim=dim,rotary=rotary,theta=theta,position=position,interleaved=interleaved,
                query=values(q),key=values(k),qweight=values(qw),kweight=values(kw),expected=values(torch.cat((eq.flatten(),ek.flatten())))))
    core=sample((32,128));z=sample((32,128),1.5);norm=GatedRMSNorm(128,1e-6,dtype=torch.bfloat16,device='cuda');norm.weight.copy_(sample((128,),.5)+1)
    gated=dict(core=values(core),gate=values(z),weight=values(norm.weight),expected=values(norm(core,z)))
    x=sample((259,),4);up=sample((259,),2)
    silu=dict(gate=values(x),up=values(up),expected=values(F.silu(x)*up))
    value=sample((259,),2);add=sample((259,),2)
    for scalar in [True,False]:
        gate=sample((1,) if scalar else (259,),2)
        for include_add in [False,True]:
            out=value*gate.sigmoid()
            if include_add:out=out+add
            gates.append(dict(value=values(value),gate=values(gate),add=values(add) if include_add else None,expected=values(out)))
report=dict(scope=__doc__,source_hashes=HASHES,torch=torch.__version__,torch_commit=torch.version.git_version,
            cuda=torch.version.cuda,rope=rope,gated=gated,silu=silu,gates=gates)
with a.out.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(report,indent=2)+'\n')
print('Pinned CUDA scalar/rotary fixtures:',len(rope),'rotary cases, gated norm, SiLU and',len(gates),'sigmoid cases')
