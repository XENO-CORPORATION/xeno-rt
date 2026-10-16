"""Generate independent FP8 group-scale/rounding cases with pinned Triton.

Synthetic BF16 inputs contain no trained weights or captured model activations.
The precise-division counterfactual must differ, preventing a vacuous fixture.
"""
import argparse
import json
import pathlib
import torch
from drama3_upstream_reference import HASHES, load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
a = p.parse_args()
load(a.source)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import quantize_groups, quant_kernel

with torch.inference_mode():
    generator = torch.Generator().manual_seed(541566)
    x = (torch.randn((1, 4096), generator=generator) * .31).bfloat16().cuda()
    # Exercise the scale floor, signed zero, and a final partial group too.
    x[0, :128] = 0
    x = x[:, :3987].contiguous()
    q, scale = quantize_groups(quant_kernel, x, x.shape[-1])
    original = x.float().cpu()
    counter_scales = torch.stack([v.abs().amax(-1).clamp(min=1e-10) / 448.0
                                  for v in original.split(128, -1)], -1)
    different = int((counter_scales != scale.cpu()).sum())
    assert different > 0, 'fixture must expose precise-divide scale differences'
    report = dict(scope=__doc__, source_hashes=HASHES, torch=torch.__version__,
                  group=128, input=original.flatten().tolist(),
                  quantized=q.float().cpu().flatten().tolist(),
                  scales=scale.cpu().flatten().tolist(),
                  precise_division_scale_mismatches=different)
with a.out.open('x', encoding='utf-8', newline='\n') as output:
    output.write(json.dumps(report, indent=2) + '\n')
print('Pinned Triton quantization fixture:', original.numel(), 'values;', different, 'scale controls differ')
