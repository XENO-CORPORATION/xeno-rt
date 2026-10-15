"""Isolate recorded operator arithmetic; this is not whole-model admission.

Each operator receives the captured input. Dense trained weights are read from
the pinned checkpoint; norm vectors are captured operator inputs. FP64 controls
measure backend rounding separately from the real upstream BF16 result.
"""
import argparse
import json
import pathlib
import torch
import torch.nn.functional as F
from drama3_upstream_reference import TensorReader, load, HASHES
from drama3_w8a8_reference import quantize, gemm

p = argparse.ArgumentParser(description=__doc__)
for name in ['model', 'native', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
p.add_argument('--source', type=pathlib.Path)
a = p.parse_args()
if a.source:
    load(a.source)
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
trace = json.loads(a.native.read_text())
bf16 = trace['numerics'] == 'bf16_w8a8'
dtype = torch.bfloat16 if bf16 else torch.float32
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']


def weight(name):
    with TensorReader(a.model / index[name]) as reader:
        return reader.get_tensor(name)


def metrics(actual, expected, rtol=None, atol=None):
    delta = (actual - expected).abs()
    worst = int(delta.argmax())
    return dict(max_absolute_error=float(delta.max()),
                relative_rms=float(delta.norm() / expected.norm().clamp(min=1e-30)),
                exact_fraction=float((actual == expected).float().mean()),
                worst_index=worst, expected_at_worst=float(expected[worst]), actual_at_worst=float(actual[worst]),
                passed=bool(torch.allclose(actual, expected, rtol=rtol if rtol is not None else (.008 if bf16 else 2e-4), atol=atol if atol is not None else (.002 if bf16 else 2e-4))))


results = []
with torch.inference_mode():
    for step in trace['traces']:
        for layer in step['layers']:
            for ordinal, op in enumerate(layer.get('operators', [])):
                x = torch.tensor(op['input'])
                auxiliary = torch.tensor(op['auxiliary'])
                args = op['parameters']
                control = None
                subchecks = None
                if op['kind'] == 'norm':
                    scale = auxiliary + (1.0 if args['zero_centered'] else 0.0)
                    y = x * torch.rsqrt(x.square().mean() + args['epsilon'])
                    expected = (y * scale).to(dtype).float()
                    mean = x.double().square().mean().float()
                    control = (x * torch.rsqrt(mean + args['epsilon']) * scale).to(dtype).float()
                elif op['kind'] == 'silu_mul':
                    if args['routed'] and bf16:
                        expected = F.silu(x.float()) * auxiliary.float()
                    else:
                        expected = (F.silu(x.to(dtype)) * auxiliary.to(dtype)).float()
                elif op['kind'] == 'attention_gate':
                    expected = (x.to(dtype) * auxiliary.to(dtype).sigmoid()).float()
                elif op['kind'] == 'gdn_conv':
                    channels, kernel = args['channels'], args['kernel']
                    history = x[channels:].reshape(channels,kernel-1)
                    padded = torch.cat((history,x[:channels,None]),-1)[None].to(dtype)
                    expected = F.silu(F.conv1d(padded,auxiliary.reshape(channels,1,kernel).to(dtype),groups=channels)).flatten().float()
                elif op['kind'] == 'gdn_controls':
                    a_control,b_control=x.chunk(2)
                    bias,log_a=auxiliary.chunk(2)
                    expected = torch.cat((b_control.sigmoid(),-log_a.exp()*F.softplus(a_control+bias)))
                elif op['kind'] == 'gdn_gate':
                    assert a.source, '--source is required for the pinned GDN reference'
                    from fish_speech.models.text2semantic.fish_omni.layers import GatedRMSNorm
                    width=args['head_dim']
                    norm=GatedRMSNorm(width,args['epsilon'],dtype=dtype)
                    norm.weight.copy_(auxiliary[:width])
                    expected=norm(x.reshape(-1,width).to(dtype),auxiliary[width:].reshape(-1,width).to(dtype)).flatten().float()
                elif op['kind'] == 'gdn_recurrent':
                    assert a.source, '--source is required for the pinned GDN reference'
                    from fish_speech.models.text2semantic.fish_omni.layers import chunk_gated_delta_rule, recurrent_gated_delta_rule
                    kh,vh,kd,vd=(args[k] for k in ['key_heads','value_heads','key_dim','value_dim'])
                    q,k,v=x.split((kh*kd,kh*kd,vh*vd))
                    q=q.reshape(1,1,kh,kd).to(dtype).repeat_interleave(vh//kh,dim=2)
                    k=k.reshape(1,1,kh,kd).to(dtype).repeat_interleave(vh//kh,dim=2)
                    v=v.reshape(1,1,vh,vd).to(dtype)
                    beta,decay,state=auxiliary.split((vh,vh,auxiliary.numel()-2*vh))
                    rule=recurrent_gated_delta_rule if args['position'] else chunk_gated_delta_rule
                    core,state=rule(q,k,v,decay.reshape(1,1,vh),beta.reshape(1,1,vh),
                                    initial_state=state.reshape(1,vh,vd,kd).transpose(-1,-2).contiguous() if args['position'] else None)
                    # Native snapshots are value-major; the pinned PyTorch rule
                    # returns key-major state. Compare the declared layouts.
                    expected=torch.cat((core.flatten().float(),state.transpose(-1,-2).contiguous().flatten().float()))
                    actual=torch.tensor(op['output'])
                    subchecks=dict(core=metrics(actual[:vh*vd],expected[:vh*vd]),
                                   state=metrics(actual[vh*vd:],expected[vh*vd:],rtol=2e-4,atol=2e-4))
                elif op['kind'] in ['linear', 'linear_rows']:
                    w = weight(op['name'])
                    if op['kind'] == 'linear_rows':
                        w = w[args['first']:args['first'] + args['count']]
                    if w.dtype == torch.float8_e4m3fn:
                        scales = weight(op['name'].removesuffix('.weight') + '.weight_scale_inv')
                        if bf16:
                            if not args.get('expert_down', False):
                                x = x.to(dtype).float()
                            q, activation_scale = quantize(x[None])
                            expected = gemm(q, activation_scale, w.float(), scales)[0]
                            if not args.get('expert_down', False):
                                expected = expected.to(dtype).float()
                        else:
                            rows = torch.arange(w.shape[0]) // 128
                            cols = torch.arange(w.shape[1]) // 128
                            expected = F.linear(x, w.float() * scales[rows[:,None],cols[None,:]])
                    else:
                        expected = F.linear(x.to(dtype), w.to(dtype)).float()
                        control = F.linear(x.to(dtype).double(), w.to(dtype).double()).float().to(dtype).float()
                else:
                    raise AssertionError(op['kind'])
                actual = torch.tensor(op['output'])
                result = dict(position=step['position'], layer=layer['layer'], ordinal=ordinal,
                              kind=op['kind'], name=op['name'], **metrics(actual, expected))
                if control is not None:
                    result['upstream_vs_fp64_control'] = metrics(expected, control)
                    result['native_vs_fp64_control'] = metrics(actual, control)
                if subchecks is not None:
                    result['subchecks']=subchecks
                    result['passed']=result['passed'] and all(v['passed'] for v in subchecks.values())
                results.append(result)
report = dict(admitted=False, scope=__doc__, torch=torch.__version__, numerics=trace['numerics'],
              source_hashes=HASHES if a.source else {}, checks=results, passed=bool(results) and all(v['passed'] for v in results))
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(checks=len(results), passed=report['passed'], failed=[v for v in results if not v['passed']])))
assert report['passed'], 'operator audit is empty or has numerical mismatches'
