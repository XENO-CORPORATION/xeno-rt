"""Generate small independent PyTorch fixtures. No weights or upstream source.
Run from the repository root. Checked-in fixtures carry their generator version.
"""
import json
import pathlib
import torch

destination=pathlib.Path('crates/xrt-kernels/tests/fixtures')
destination.mkdir(parents=True,exist_ok=True)
torch.manual_seed(314159)
rows,cols,edge=17,19,8
raw=((torch.arange(rows*cols,dtype=torch.int64)*13)%126).to(torch.uint8).reshape(rows,cols)
raw[::2] |= 128
weight=raw.view(torch.float8_e4m3fn).float()
scales=torch.linspace(.001,.02,rows//edge+1).view(-1,1).expand(-1,cols//edge+1).contiguous()
row_ids=torch.arange(rows)//edge
col_ids=torch.arange(cols)//edge
expanded=weight*scales[row_ids[:,None],col_ids[None,:]]
activations=torch.linspace(-3.0,2.0,cols)
quantized=[]; activation_scales=[]
for group in activations.split(edge):
    scale=group.abs().max().clamp(min=1e-10)/448.0
    quantized.append((group/scale).clamp(-448,448).to(torch.float8_e4m3fn).float())
    activation_scales.append(float(scale))
quantized=torch.cat(quantized)
w8a8=torch.zeros(rows)
for group in range((cols+edge-1)//edge):
    begin,end=group*edge,min((group+1)*edge,cols)
    w8a8 += ((weight[:,begin:end]@quantized[begin:end])*activation_scales[group])*scales[row_ids,group]

kh,vh,kd,vd,steps=2,4,3,2,5
query=torch.randn(steps,kh,kd); key=torch.randn(steps,kh,kd)
values=torch.randn(steps,vh,vd); decay=-torch.rand(steps,vh); beta=torch.sigmoid(torch.randn(steps,vh))
state=torch.zeros(vh,kd,vd); outputs=[]
for step in range(steps):
    q=query[step]*torch.rsqrt(query[step].square().sum(-1,keepdim=True)+1e-6)
    k=key[step]*torch.rsqrt(key[step].square().sum(-1,keepdim=True)+1e-6)
    q=q.repeat_interleave(vh//kh,dim=0)/(kd**.5); k=k.repeat_interleave(vh//kh,dim=0)
    state=state*decay[step].exp()[:,None,None]
    remembered=(state*k[:,:,None]).sum(1)
    delta=(values[step]-remembered)*beta[step,:,None]
    state=state+k[:,:,None]*delta[:,None,:]
    outputs.append((state*q[:,:,None]).sum(1).flatten().tolist())
fixture={'generator':'scripts/drama3_numeric_oracle.py','torch':torch.__version__,
 'fp8':{'rows':rows,'cols':cols,'edge':edge,'weights':raw.flatten().tolist(),'scales':scales.flatten().tolist(),
 'input':activations.tolist(),'expected':(expanded@activations).tolist(),'w8a8':w8a8.tolist()},
 'delta':{'key_heads':kh,'value_heads':vh,'key_dim':kd,'value_dim':vd,'query':query.flatten().tolist(),
 'key':key.flatten().tolist(),'value':values.flatten().tolist(),'decay':decay.flatten().tolist(),
 'beta':beta.flatten().tolist(),'outputs':outputs,'final_state_value_major':state.transpose(1,2).flatten().tolist()}}
conv_fixtures=[]
for input_channels,output_channels,kernel,stride,dilation,groups,transpose in [(7,5,3,1,2,1,False),(4,6,3,2,1,2,False),(4,4,5,1,1,4,False),(7,5,4,2,1,1,True)]:
    length=5
    x=torch.randn(1,input_channels,length);bias=torch.randn(output_channels)
    shape=(input_channels,output_channels,kernel) if transpose else (output_channels,input_channels//groups,kernel)
    weight=torch.randn(shape)
    if transpose:
        y=torch.nn.functional.conv_transpose1d(x,weight,bias,stride=stride)[...,:length*stride]
    else:
        left=(kernel-1)*dilation+1-stride
        y=torch.nn.functional.conv1d(torch.nn.functional.pad(x,(left,(-length)%stride)),weight,bias,stride=stride,dilation=dilation,groups=groups)
    conv_fixtures.append(dict(input_channels=input_channels,output_channels=output_channels,kernel=kernel,stride=stride,dilation=dilation,groups=groups,transpose=transpose,frames=length,input=x.transpose(1,2).flatten().tolist(),weights=weight.flatten().tolist(),bias=bias.tolist(),expected=y.transpose(1,2).flatten().tolist()))
fixture['causal_conv']=conv_fixtures
(destination/'drama3_numeric.json').write_text(json.dumps(fixture,indent=2)+'\n')
print('Generated independent PyTorch FP8 and recurrent-state fixtures')
