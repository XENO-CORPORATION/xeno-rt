"""Capture a real, trained projection and independent PyTorch outputs.

Reads a completed immutable shard from the development install; verifies its
whole-file SHA-256 before extracting one expert. Outputs stay outside the repo.
This qualifies a projection, never the whole model or the incomplete bundle.
"""
import argparse,hashlib,json,pathlib
import torch
from safetensors import safe_open
from safetensors.torch import save_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--shard',type=pathlib.Path,required=True)
p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
pin=json.loads((pathlib.Path(__file__).resolve().parents[1]/'reference/audio/drama3/provenance.json').read_text(encoding='utf-8-sig'))
record=next(f for f in pin['files'] if f['path']==a.shard.name)
digest=hashlib.file_digest(a.shard.open('rb'),'sha256').hexdigest()
if digest!=record['sha256'] or a.shard.stat().st_size!=record['sizeBytes']:raise ValueError('shard integrity mismatch')
with safe_open(a.shard,framework='pt',device='cpu') as store:
    name=next(k for k in store.keys() if k.endswith('experts.0.gate_proj.weight'))
    scales=name[:-len('weight')]+'weight_scale_inv'
    w=store.get_tensor(name).contiguous();s=store.get_tensor(scales).float().contiguous()
rows,cols=w.shape;expanded=w.float()*s.repeat_interleave(128,0).repeat_interleave(128,1)[:rows,:cols]
x=torch.sin(torch.arange(cols,dtype=torch.float32)*.013)
quantized=[];activation_scales=[]
for group in x.split(128):
    scale=group.abs().max().clamp(min=1e-10)/448
    quantized.append((group/scale).clamp(-448,448).to(torch.float8_e4m3fn).float())
    activation_scales.append(scale)
qx=torch.cat(quantized);result=torch.zeros(rows)
for block in range(cols//128):
    begin,end=block*128,(block+1)*128
    result += (w.float()[:,begin:end]@qx[begin:end])*s[:,block].repeat_interleave(128)[:rows]*activation_scales[block]
save_file({'weight':w,'scale':s,'input':x,'expected_f32':expanded@x,'expected_w8a8':result},str(a.out/'model.safetensors'))
report={'scope':'single trained expert projection','admitted':False,'model_revision':pin['revision'],
        'source_shard':record['path'],'source_sha256':digest,'tensor':name,'torch':torch.__version__,
        'fixture_sha256':hashlib.file_digest((a.out/'model.safetensors').open('rb'),'sha256').hexdigest()}
(a.out/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
