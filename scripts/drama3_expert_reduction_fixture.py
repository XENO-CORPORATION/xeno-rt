"""Independent CUDA reduction over eight synthetic routed expert outputs."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
g=torch.Generator().manual_seed(541566)
partial=torch.randn((8,2048),generator=g).cuda()
expected=partial[None].sum(dim=1).flatten()
tree=partial.clone()
for stride in [4,2,1]:tree[:stride]=tree[:stride]+tree[stride:2*stride]
assert (tree[0]!=expected).any(), 'fixture must reject the eight-way tree'
with a.out.open('x',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(dict(scope=__doc__,torch=torch.__version__,torch_commit=torch.version.git_version,
        partial=partial.cpu().flatten().tolist(),expected=expected.cpu().tolist(),
        wrong_tree_mismatches=int((tree[0]!=expected).sum())),indent=2)+'\n')
print('Pinned expert reduction: counterfactual mismatches',int((tree[0]!=expected).sum()))
