"""Independent actual pinned Triton reference, including partial block padding.
No native output is read. Synthetic weights have no model redistribution issue.
"""
import argparse,json,pathlib
import torch,triton
from drama3_upstream_reference import load,HASHES
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args();load(a.source)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
h,m,e=(2048,512,8) if a.benchmark else (259,129,2);hp=((h+127)//128)*128;mp=((m+127)//128)*128
x=((torch.arange(h,dtype=torch.float32)*.031).sin()*(1.0 if a.benchmark else 1.7)).bfloat16().float()
gate_up=torch.zeros(e,2*mp,hp,dtype=torch.uint8);down=torch.zeros(e,hp,mp,dtype=torch.uint8)
s13=torch.zeros(e,2*(mp//128),hp//128);s2=torch.zeros(e,hp//128,mp//128)
for expert in range(e):
    for projection,rows,cols in [(0,m,h),(1,m,h),(2,h,m)]:
        i=torch.arange(rows*cols);raw=((i*17+expert*13+projection*5)%(96 if a.benchmark else 120)).to(torch.uint8);raw[i%3==0]|=128;raw=raw.reshape(rows,cols)
        count=((rows+127)//128)*((cols+127)//128)
        scales=torch.full((count,),.001) if a.benchmark else .0007+torch.arange(count,dtype=torch.float32)*.000031
        scales=scales.reshape((rows+127)//128,(cols+127)//128)
        if projection<2:
            gate_up[expert,projection*mp:projection*mp+m,:h]=raw
            s13[expert,projection*(mp//128):(projection+1)*(mp//128)]=scales
        else:down[expert,:h,:m]=raw;s2[expert]=scales
with torch.inference_mode():
    input=torch.zeros(1,hp,dtype=torch.bfloat16,device='cuda');input[0,:h]=x.cuda()
    probabilities=torch.full((1,e),1/e,device='cuda') if a.benchmark else torch.tensor([[.333,.679]],device='cuda')/1.012
    actual=fp8_moe_forward(input,torch.arange(e,device='cuda')[None],probabilities.bfloat16(),
        gate_up.view(torch.float8_e4m3fn).cuda(),s13.cuda(),down.view(torch.float8_e4m3fn).cuda(),s2.cuda())
    result=actual[0,:h].float().cpu()
report={'scope':__doc__,'source_hashes':HASHES,'torch':torch.__version__,'triton':triton.__version__,
        'device':torch.cuda.get_device_name(),'hidden':h,'middle':m,'experts':e,'input':x.tolist(),'output':result.tolist()}
a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print('Wrote actual Triton W8A8 expert reference',h,m)
