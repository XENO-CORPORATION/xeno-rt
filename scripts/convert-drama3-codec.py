"""Convert the pinned codec's tensor-only checkpoint for native Rust decoding.

Python is a provisioning/reference tool only. The native decoder uses the
SafeTensors output and never invokes Python or unpickles model files.
"""
import argparse,hashlib,json,pathlib
import torch
from safetensors.torch import save_file

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--include-encoder',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
with a.source.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
if digest!='74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f':raise ValueError('not the pinned Drama 3 codec')
state=torch.load(a.source,weights_only=True,map_location='cpu',mmap=True)
out={}
for name,value in state.items():
    if not name.startswith(('encoder.','decoder.','quantizer.')) or name.startswith('quantizer.semantic_predictor_module.'):
        continue
    if not a.include_encoder and name.startswith(('encoder.','quantizer.downsample.','quantizer.pre_module.')):
        continue
    if name.endswith(('.causal_mask','.freqs_cis')):continue
    if '.parametrizations.weight.original0' in name or name.endswith('.weight_g'):continue
    if '.parametrizations.weight.original1' in name or name.endswith('.weight_v'):
        gain_name=name.replace('original1','original0') if 'original1' in name else name[:-len('weight_v')]+'weight_g'
        gain=state[gain_name].float()
        value=value.float();axes=tuple(range(1,value.ndim))
        value=value*(gain/torch.linalg.vector_norm(value,dim=axes,keepdim=True))
        name=name.replace('.parametrizations.weight.original1','.weight').replace('.weight_v','.weight')
    if not torch.isfinite(value).all():raise ValueError('non-finite codec tensor')
    out[name]=value.float().contiguous()
save_file(out,str(a.out/'model.safetensors'))
metadata={'schema':1,'family':'fish-dac-drama3','sample_rate':44100,'frame_samples':2048,
 'semantic_codebook_size':4096,'residual_codebook_size':1024,'codebooks':10,
 'source_sha256':digest,'model_revision':'e0a9218aa48888270f06ef1ed9ea67622c0e17d2',
 'encoder':a.include_encoder, 'weight_norm':'folded-per-dimension-0','precision':'f32','torch':torch.__version__,'tensors':len(out),
 'converted_sha256':hashlib.file_digest((a.out/'model.safetensors').open('rb'),'sha256').hexdigest(),
 'license':'Fish Audio Community License v1.0','admitted':False}
(a.out/'codec.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata))

# Retain the material license and conversion attribution with derived tensors.
notice_root=pathlib.Path(__file__).resolve().parents[1]/'reference'/'audio'/'drama3'
(a.out/'LICENSE').write_bytes((notice_root/'LICENSE').read_bytes())
(a.out/'Notice').write_bytes((notice_root/'Notice').read_bytes())
