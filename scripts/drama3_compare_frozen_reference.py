"""Compare a native trace to independently composed pinned reference outputs.

Initial embeddings and teacher-forced identity must match exactly. This never
substitutes native intermediate inputs into the reference graph or widens the
existing BF16 layer and log-probability tolerances.
"""
import argparse
import hashlib
import json
import pathlib
import torch
from drama3_upstream_reference import HASHES

p=argparse.ArgumentParser(description=__doc__)
for name in ['native','reference','out']:
    p.add_argument('--'+name,type=pathlib.Path,required=True)
a=p.parse_args()
torch.set_num_threads(8)
native_bytes=a.native.read_bytes();reference_bytes=a.reference.read_bytes()
n=json.loads(native_bytes);r=json.loads(reference_bytes)
assert n['numerics']=='bf16_w8a8'
assert r['composed'] and r['device']=='cuda' and not r['stable_routing_control']
assert r['source_hashes']==HASHES
traces=n['traces'];assert 1<=len(traces)<=64
assert [s['position'] for s in traces]==list(range(len(traces)))
digest=hashlib.sha256(json.dumps([s['input'] for s in traces],separators=(',',':')).encode()).hexdigest()
assert digest==r['initial_input_sha256'],'initial embeddings differ from frozen reference'
assert n['teacher_forced_sha256']==r['teacher_forced_sha256'] and n['prompt_rows']==r['prompt_rows']
expected={(s['layer'],s['position']):s['values'] for s in r['reference_layer_outputs']}
assert len(expected)==len(r['reference_layer_outputs'])==40*len(traces)
checks=[];outputs=[]
for step in traces:
    position=step['position']
    afters=[l for l in step['layers'] if l['after']]
    assert [l['layer'] for l in afters]==list(range(40))
    for layer in afters:
        actual=torch.tensor(layer['values']);ref=torch.tensor(expected[layer['layer'],position])
        assert actual.shape==ref.shape==(2048,)
        error=(actual-ref).abs()
        checks.append(dict(layer=layer['layer'],position=position,passed=bool(torch.allclose(actual,ref,rtol=.008,atol=.002)),
            relative_rms=float(error.norm()/ref.norm().clamp(min=1e-30)),max_absolute_error=float(error.max()),exact_fraction=float((actual==ref).float().mean())))
    ref=r['reference_outputs'][position];assert ref['position']==position
    actual=torch.tensor(step['semantic_logits']);expected_logits=torch.tensor(ref['semantic_logits'])
    assert actual.shape==expected_logits.shape==(4097,)
    lp=actual.log_softmax(-1);ref_lp=expected_logits.log_softmax(-1);top=ref_lp.topk(20).indices
    outputs.append(dict(position=position,generation_position=position>=n['prompt_rows']-1,
        passed=bool(torch.allclose(lp[top],ref_lp[top],rtol=.02,atol=.02)),
        top20_logprob_max_error=float((lp[top]-ref_lp[top]).abs().max()),
        reference_top1=int(expected_logits.argmax()),native_top1=int(actual.argmax())))
report=dict(admitted=False,scope=__doc__,native_sha256=hashlib.sha256(native_bytes).hexdigest(),
    reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),initial_input_sha256=digest,
    layer_checks=checks,model_outputs=outputs,passed=all(c['passed'] for c in checks+outputs))
with a.out.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(report,indent=2)+'\n')
generation=[c for c in outputs if c['generation_position']]
print(json.dumps(dict(layer_passes=sum(c['passed'] for c in checks),layer_checks=len(checks),
    generation_passes=sum(c['passed'] for c in generation),generation_checks=len(generation),
    generation=generation)))
assert report['passed'],'strict composed-layer or output mismatch; retained diagnostic is not admission'
