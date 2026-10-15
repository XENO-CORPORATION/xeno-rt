"""Check all actual trained tower layers against pinned Fish CPU/F32 code."""
import argparse
import json
import pathlib
import torch
from drama3_upstream_reference import HASHES, TensorReader, load

p = argparse.ArgumentParser(description=__doc__)
for name in ['source', 'model', 'native', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
p.add_argument("--bf16-w8a8", action="store_true")
p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
a = p.parse_args()
load(a.source)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, TOWER_CHECKPOINT_PREFIXES

torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
config = FishOmniConfig.from_dict(json.loads((a.model / 'config.json').read_text()))
native = json.loads(a.native.read_text())['traces']
assert [step['position'] for step in native] == [0, 1], 'require both successive probe positions'
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
with torch.inference_mode():
    dtype=torch.bfloat16 if a.bf16_w8a8 else torch.float32
    decoder = FishOmniAudioDecoder(config, dtype=dtype, device=a.device).eval()
    targets = dict(decoder.named_parameters())
    files = {}
    for name, shard in index.items():
        if name.startswith(tuple(TOWER_CHECKPOINT_PREFIXES)):
            files.setdefault(shard, []).append(name)
    filled = set()
    for shard, names in files.items():
        with TensorReader(a.model / shard) as store:
            for name in names:
                target_name = next(new + name[len(old):] for old, new in TOWER_CHECKPOINT_PREFIXES.items() if name.startswith(old))
                value = store.get_tensor(name)
                assert targets[target_name].shape == value.shape and target_name not in filled
                targets[target_name].copy_(value)
                filled.add(target_name)
    assert filled == set(targets), set(targets) - filled
    decoder.setup_caches(len(native))
    layers = [*decoder.shared_layers, *decoder.residual_layers]
    results = []
    for step in native:
        position = step['position']
        frame, next_frame = torch.tensor([step['frame']], device=a.device), torch.tensor([step['next']], device=a.device)
        semantic = decoder.is_semantic(next_frame[:, 0])
        unseen = decoder.embedding_sum(frame, 1 + config.audio_decoder.vq_backbone_codebooks, config.audio_decoder.num_codebooks)
        unseen = torch.where(decoder.is_semantic(frame[:, 0])[:, None], unseen, 0.0)
        for layer_id, layer in enumerate(layers):
            before = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and not v['after'])
            after = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and v['after'])
            shared = layer_id < config.audio_decoder.shared_enrichment_layers
            code_index = 1 if shared else layer_id - config.audio_decoder.shared_enrichment_layers + 1
            expected = layer(torch.tensor(before,dtype=dtype, device=a.device)[None], next_frame[:, code_index], semantic,
                             torch.tensor([position], device=a.device), unseen if layer_id == 0 else None).flatten().float().cpu()
            actual = torch.tensor(after)
            error = (actual - expected).abs()
            results.append({'layer': layer_id, 'position': position, 'shared': shared,
                            'max_absolute_error': float(error.max()),
                            'max_scaled_error': float((error / expected.abs().clamp(min=1)).max()),
                            'relative_rms': float(error.norm() / expected.norm().clamp(min=1e-30)),
                            'exact_fraction': float((actual == expected).float().mean()),
                            'passed': bool(torch.allclose(actual, expected, rtol=0.008 if a.bf16_w8a8 else 2e-4, atol=0.002 if a.bf16_w8a8 else 2e-4))})
    report = {'admitted': False, 'scope': 'all trained tower layers vs actual pinned Fish code',
              'source_commit': '541566932abbaebcb0a40e562a1f846738383f38', 'source_hashes': HASHES,
              'torch': torch.__version__, 'device':a.device, 'numerics':str(dtype),'checks': results, 'passed': all(v['passed'] for v in results)}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    assert report['passed'], 'trained tower layer mismatch'
