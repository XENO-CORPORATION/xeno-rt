"""Check every trained backbone layer against actual pinned Fish CPU/F32 code.

Keeps only one upstream layer resident at a time. Source lives outside the repo;
its hashes are verified before import. This is a reference tool, not inference.
"""
import argparse
import gc
import json
import pathlib
import re
import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=pathlib.Path, required=True)
p.add_argument('--model', type=pathlib.Path, required=True)
p.add_argument('--native', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
p.add_argument("--bf16-w8a8", action="store_true")
a = p.parse_args()
from drama3_upstream_reference import HASHES, TensorReader, load
load(a.source)
hashes = HASHES
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext

torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
config = FishOmniTextConfig.from_dict(json.loads((a.model / 'config.json').read_text())['text_config'])
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
native = json.loads(a.native.read_text())['traces']
assert [step['position'] for step in native] == [0, 1], 'require both successive probe positions'
expert_pattern = re.compile(r'mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$')
results = []
with torch.inference_mode():
    for layer_id in range(config.num_hidden_layers):
        layer = DecoderLayer(config, layer_id, fp8=True, dtype=torch.bfloat16 if a.bf16_w8a8 else torch.float32, device='cpu').eval()
        targets = dict(layer.named_parameters())
        targets.update(dict(layer.named_buffers()))
        prefix = f'model.language_model.layers.{layer_id}.'
        files = {}
        for name, shard in index.items():
            if name.startswith(prefix): files.setdefault(shard, []).append(name)
        filled = set()
        for shard, names in files.items():
            with TensorReader(a.model / shard) as store:
                for name in names:
                    suffix = name[len(prefix):]
                    value = store.get_tensor(name)
                    match = expert_pattern.fullmatch(suffix)
                    if match:
                        expert, projection, kind = match.groups()
                        parameter = f"mlp.experts.{'w2' if projection == 'down_proj' else 'w13'}_{'scale_inv' if kind == 'weight_scale_inv' else 'weight'}"
                        destination = targets[parameter][int(expert)]
                        if projection != 'down_proj':
                            half = destination.shape[0] // 2
                            destination = destination[:half] if projection == 'gate_proj' else destination[half:]
                        assert destination.shape == value.shape
                        destination.copy_(value)
                        filled.add(parameter)
                    else:
                        assert suffix in targets and targets[suffix].shape == value.shape, suffix
                        targets[suffix].copy_(value)
                        filled.add(suffix)
        assert filled == set(targets), set(targets) - filled
        if a.bf16_w8a8:
            from drama3_w8a8_reference import install
            install(layer.mlp.experts)
        dtype = torch.bfloat16 if a.bf16_w8a8 else torch.float32
        if layer.layer_type == 'full_attention':
            layer.self_attn.allocate_cache(1, len(native), dtype, 'cpu')
        else:
            layer.linear_attn.allocate_cache(1, dtype, 'cpu')
        for step in native:
            position = step['position']
            before = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and not v['after'])
            after = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and v['after'])
            rotary = config.rotary_dim
            frequency = 1.0 / (config.rope_theta ** (torch.arange(0, rotary, 2).float() / rotary))
            angles = torch.cat((position * frequency, position * frequency))[None]
            context = LayerContext(input_pos=torch.tensor([position]), cos=angles.cos(), sin=angles.sin(), kv_len=position + 1, continue_state=position > 0, use_cache=True)
            expected = layer(torch.tensor(before,dtype=dtype)[None, None], context).flatten().float()
            actual = torch.tensor(after)
            error = (actual - expected).abs()
            results.append({'layer': layer_id, 'position': position, 'kind': layer.layer_type,
                            'max_absolute_error': float(error.max()),
                            'max_scaled_error': float((error / expected.abs().clamp(min=1)).max()),
                            'relative_rms': float(error.norm() / expected.norm().clamp(min=1e-30)),
                            'exact_fraction': float((actual == expected).float().mean()),
                            'passed': bool(torch.allclose(actual, expected, rtol=0.008 if a.bf16_w8a8 else 2e-4, atol=0.002 if a.bf16_w8a8 else 2e-4))})
        print(json.dumps(results[-2:]), flush=True)
        del layer, targets
        gc.collect()
report = {'admitted': False, 'scope': 'every trained native backbone layer vs actual pinned Fish code',
          'source_commit': '541566932abbaebcb0a40e562a1f846738383f38', 'source_hashes': hashes,
          'torch': torch.__version__, 'numerics': 'BF16 with CPU-emulated W8A8 expert operator' if a.bf16_w8a8 else 'actual upstream FP32 CPU', 'checks': results, 'passed': all(v['passed'] for v in results)}
a.out.write_text(json.dumps(report, indent=2) + '\n')
assert report['passed'], 'trained backbone layer mismatch'
