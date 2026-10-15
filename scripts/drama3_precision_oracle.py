"""Diagnose dense BF16 boundaries independently with actual trained weights."""
import argparse
import json
import pathlib
import torch
from drama3_upstream_reference import TensorReader

p = argparse.ArgumentParser(description=__doc__)
for name in ['model', 'native', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
config = json.loads((a.model / 'config.json').read_text())
checks = []
with torch.inference_mode():
    for case in json.loads(a.native.read_text()):
        with TensorReader(a.model / index[case['name']]) as reader:
            weight = reader.get_tensor(case['name'])
        x = torch.tensor(case['input'], dtype=torch.bfloat16)
        if case['op'] == 'norm':
            y = x.float()
            y *= torch.rsqrt(y.square().mean() + config['audio_decoder_config']['norm_eps'])
            mathematical = y * (1.0 + weight.float())
            expected = mathematical.to(torch.bfloat16).float()
        else:
            mathematical = torch.nn.functional.linear(x.float(), weight.float())
            expected = torch.nn.functional.linear(x, weight).float()
        actual = torch.tensor(case['output'])
        error = actual - expected
        checks.append(dict(name=case['name'], op=case['op'],
                           exact_fraction=float((actual == expected).float().mean()),
                           max_absolute_error=float(error.abs().max()),
                           relative_rms=float(error.norm() / expected.norm().clamp(min=1e-30)),
                           max_error_vs_f32=float((actual - mathematical).abs().max())))
report = dict(admitted=False, scope='BF16 dense operator diagnostics', torch=torch.__version__, checks=checks)
a.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
