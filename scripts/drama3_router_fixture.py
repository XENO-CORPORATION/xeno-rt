"""Independent CUDA router softmax fixtures; synthetic BF16 logits only."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();torch.set_num_threads(8)
g=torch.Generator().manual_seed(541566)
cases=[]
for n in [32,64,128,256,512,1024,256,256]:
    logits=(torch.randn(n,generator=g)*3).bfloat16().float().cuda()
    if len(cases)==6:logits.zero_()
    if len(cases)==7:logits[0]=1000;logits[1:]=-1000
    probabilities=logits[None].softmax(-1)
    weights,ids=probabilities.topk(8,-1)
    cases.append(dict(input=logits.cpu().tolist(),probabilities=probabilities.flatten().cpu().tolist(),
        top8=ids.flatten().cpu().tolist(),denominator=weights.sum().item(),
        normalized_bf16=(weights/weights.sum(-1,keepdim=True)).bfloat16().float().flatten().cpu().tolist()))
with a.out.open('x',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(dict(scope=__doc__,torch=torch.__version__,torch_commit=torch.version.git_version,cases=cases),indent=2)+'\n')
print('Pinned CUDA router fixtures:',len(cases))
