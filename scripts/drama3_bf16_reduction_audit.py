"""Measure BF16 layer sensitivity without changing native admission bounds.

The baseline is the actual pinned upstream graph. The control changes only
norm means and dense dot accumulation to FP64, returning to FP32 before the
original BF16 boundary. It is a diagnostic control, not stock serving parity.
"""
import argparse
import json
import pathlib
from unittest.mock import patch
import torch
import torch.nn.functional as F
from drama3_upstream_reference import load, HASHES
from drama3_tower_reference import trained_tower, teacher_forced

p = argparse.ArgumentParser(description=__doc__)
for name in ['source', 'model', 'native', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
a = p.parse_args()
load(a.source)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
config = FishOmniConfig.from_dict(json.loads((a.model / 'config.json').read_text()))
native = json.loads(a.native.read_text())['traces']
decoder = trained_tower(a.model, config, torch.bfloat16)
baseline, control = {}, {}
teacher_forced(decoder, config, native, lambda position, layer, value: baseline.update({(position,layer):value.clone()}))


def controlled_linear(module, x):
    result = F.linear(x.double(), module.weight.double(), None if module.bias is None else module.bias.double())
    return result.float().to(x.dtype)


def controlled_norm(module, x):
    value = x.float()
    mean = value.double().square().mean(-1, keepdim=True).float()
    return (value * torch.rsqrt(mean + module.eps) * (1.0 + module.weight.float())).to(x.dtype)


with patch.object(torch.nn.Linear, 'forward', controlled_linear), patch.object(ZeroCenteredRMSNorm, 'forward', controlled_norm):
    teacher_forced(decoder, config, native, lambda position, layer, value: control.update({(position,layer):value.clone()}))


def comparison(x, expected):
    error = x - expected
    return dict(max_absolute_error=float(error.abs().max()), relative_rms=float(error.norm()/expected.norm().clamp(min=1e-30)),
                exact_fraction=float((x == expected).float().mean()),
                passes_existing_pointwise_bound=bool(torch.allclose(x,expected,rtol=.008,atol=.002)))


checks=[]
for step in native:
    for layer in step['layers']:
        if not layer['after']:
            continue
        key=step['position'],layer['layer']
        value=torch.tensor(layer['values'])
        checks.append(dict(position=key[0],layer=key[1],native_vs_upstream=comparison(value,baseline[key]),
                           native_vs_control=comparison(value,control[key]),upstream_vs_control=comparison(baseline[key],control[key])))
report=dict(admitted=False,scope=__doc__,source_hashes=HASHES,torch=torch.__version__,checks=checks)
a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),upstream_control_failures=sum(not c['upstream_vs_control']['passes_existing_pointwise_bound'] for c in checks),
                     max_upstream_control_rms=max(c['upstream_vs_control']['relative_rms'] for c in checks),
                     max_native_upstream_rms=max(c['native_vs_upstream']['relative_rms'] for c in checks))))
