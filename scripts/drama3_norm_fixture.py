"""Pinned CUDA RMS normalization with synthetic BF16 inputs and weights."""
import argparse,json,pathlib,torch
from drama3_upstream_reference import HASHES,load
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();load(a.source)
from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm
generator=torch.Generator().manual_seed(541566);cases=[]
with torch.inference_mode():
    for width,rows in [(128,32),(256,16),(2048,1)]:
        norm=ZeroCenteredRMSNorm(width,1e-6,dtype=torch.bfloat16,device='cuda')
        norm.weight.copy_((torch.randn(width,generator=generator)*.07).bfloat16())
        for attempt in range(256):
            x=torch.randn((rows,width),generator=generator).bfloat16().cuda()
            expected=norm(x)[0].float().cpu();first=x[0].float().cpu();weight=norm.weight.float().cpu()
            mean=first.double().square().mean().float()
            old=(first*((mean+1e-6).sqrt().reciprocal())*(1+weight)).bfloat16().float()
            mismatches=int((old!=expected).sum())
            if mismatches or attempt==255:
                cases.append(dict(width=width,batch_rows=rows,epsilon=1e-6,input=first.tolist(),weight=weight.tolist(),
                                  expected=expected.tolist(),cpu_inverse_root_counterfactual_mismatches=mismatches,attempt=attempt))
                break
assert any(c['cpu_inverse_root_counterfactual_mismatches'] for c in cases),'no inverse-root counterfactual found in bounded search'
with a.out.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(scope=__doc__,source_hashes=HASHES,torch=torch.__version__,cases=cases),indent=2)+'\n')
print('Pinned CUDA norm fixtures:',[(c['width'],c['batch_rows'],c['cpu_inverse_root_counterfactual_mismatches']) for c in cases])
