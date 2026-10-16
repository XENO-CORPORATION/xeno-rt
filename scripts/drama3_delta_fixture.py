"""Pinned CUDA gated-delta sequence with synthetic BF16 projections/controls."""
import argparse,json,pathlib,torch
import torch.nn.functional as F
from drama3_upstream_reference import HASHES,load
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();load(a.source)
from fish_speech.models.text2semantic.fish_omni.layers import chunk_gated_delta_rule,recurrent_gated_delta_rule
kh,vh,dim,steps=2,4,128,8
generator=torch.Generator().manual_seed(541566)
def values(shape,scale=1):return (torch.randn(shape,generator=generator)*scale).bfloat16().cuda().float()
with torch.inference_mode():
    raw=values((steps,2*kh*dim+vh*dim));aa=values((steps,vh));bb=values((steps,vh));bias=values((vh,),.4);log_a=values((vh,),.3)
    cases=[]
    for tiled in [False,True]:
        state=None;outputs=[]
        for i in range(steps):
            q,k,v=raw[i].split((kh*dim,kh*dim,vh*dim))
            q=q.bfloat16().reshape(1,1,kh,dim);k=k.bfloat16().reshape(1,1,kh,dim);v=v.bfloat16().reshape(1,1,vh,dim)
            if tiled:q=q.repeat(1,1,vh//kh,1);k=k.repeat(1,1,vh//kh,1)
            else:q=q.repeat_interleave(vh//kh,dim=2);k=k.repeat_interleave(vh//kh,dim=2)
            beta=bb[i].sigmoid();g=-log_a.exp()*F.softplus(aa[i]+bias)
            core,state=(chunk_gated_delta_rule if i==0 else recurrent_gated_delta_rule)(q,k,v,g.reshape(1,1,vh),beta.reshape(1,1,vh),state)
            outputs.append(dict(core=core.float().cpu().flatten().tolist(),beta=beta.cpu().tolist(),log_decay=g.cpu().tolist()))
        cases.append(dict(tiled=tiled,outputs=outputs,final_state_value_major=state.transpose(-1,-2).contiguous().cpu().flatten().tolist()))
    report=dict(scope=__doc__,source_hashes=HASHES,torch=torch.__version__,key_heads=kh,value_heads=vh,
        input=raw.cpu().tolist(),a=aa.cpu().tolist(),b=bb.cpu().tolist(),bias=bias.cpu().tolist(),log_a=log_a.cpu().tolist(),cases=cases)
with a.out.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(report,indent=2)+'\n')
print('Pinned gated-delta fixture:',steps,'positions, repeat-interleave and tile')
