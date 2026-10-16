"""Independent pinned CUDA attention; synthetic BF16 Q/K/V, no model weights."""
import argparse
import json
import pathlib
import torch
from drama3_upstream_reference import HASHES, load

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
p.add_argument('--serving',action='store_true',help='compact extended online-attention cases')
a=p.parse_args();load(a.source)
from fish_speech.models.text2semantic.fish_omni.layers import grouped_attention

generator=torch.Generator().manual_seed(541566)
cases=[]
with torch.inference_mode():
    shapes=([(n,256,4,2) for n in [1,2,7,34,129,257]]+[(129,128,4,2),(257,128,4,2)]
        +[(n,d,h,2) for d,h in [(256,16),(128,32)] for n in [2,34,129,257]]) if a.serving else [(1,64,4,2),(2,256,4,2),(7,64,4,2),(34,128,4,2)]
    def encode(x):
        if a.serving:return x.bfloat16().contiguous().cpu().view(torch.uint8).numpy().tobytes().hex()
        return x.float().cpu().flatten().tolist()
    for rows,dim,heads,kv_heads in shapes:
        q=torch.randn((1,heads,1,dim),generator=generator).bfloat16().cuda()
        k=torch.randn((1,kv_heads,rows,dim),generator=generator).bfloat16().cuda()
        v=torch.randn((1,kv_heads,rows,dim),generator=generator).bfloat16().cuda()
        expected=grouped_attention(q,k,v,torch.ones((1,rows),dtype=torch.bool,device='cuda'),scale=dim**-.5)
        expanded_k=k.repeat_interleave(heads//kv_heads,dim=1).float()
        expanded_v=v.repeat_interleave(heads//kv_heads,dim=1).float()
        precise=((q.float()@expanded_k.transpose(-1,-2)*dim**-.5).softmax(-1)@expanded_v).bfloat16()
        cases.append(dict(rows=rows,dim=dim,heads=heads,kv_heads=kv_heads,
            query=encode(q),keys=encode(k.transpose(1,2)),
            values=encode(v.transpose(1,2)),expected=encode(expected),
            fp32_probability_control_relative_rms=float((precise.float()-expected.float()).norm()/expected.float().norm())))
assert max(c['fp32_probability_control_relative_rms'] for c in cases)>.001
with a.out.open('x',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(dict(scope=__doc__,source_hashes=HASHES,torch=torch.__version__,torch_commit=torch.version.git_version,
        encoding='bf16-little-endian-hex' if a.serving else 'float32-json',cases=cases),indent=2)+'\n')
print('Pinned attention fixture:',len(cases),'cases')
