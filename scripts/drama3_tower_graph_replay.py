"""Diagnose graph-boundary differences by replaying captured operator outputs.

This is NOT a fidelity oracle: injected outputs deliberately isolate the wiring
between operators. Every upstream input is checked before injecting a value.
"""
import argparse
import json
import pathlib
from unittest.mock import patch
import torch
import torch.nn.functional as F
from drama3_upstream_reference import load, HASHES
from drama3_tower_reference import trained_tower

p = argparse.ArgumentParser(description=__doc__)
for name in ['source', 'model', 'native', 'out']:
    p.add_argument('--' + name, type=pathlib.Path, required=True)
a = p.parse_args()
load(a.source)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm
from fish_speech.models.text2semantic.fish_omni.tower import TowerMLP

torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
config = FishOmniConfig.from_dict(json.loads((a.model / 'config.json').read_text()))
native = json.loads(a.native.read_text())['traces']
decoder = trained_tower(a.model, config, torch.bfloat16)
decoder.setup_caches(len(native))
layers = [*decoder.shared_layers, *decoder.residual_layers]
checks = []
attention_checks = []
original_sdpa = F.scaled_dot_product_attention
ops, cursor, position, layer_id = [], 0, 0, 0


def compare(x, values, kind):
    actual = x.flatten().float()
    expected = torch.tensor(values)
    assert actual.shape == expected.shape
    error = (actual - expected).abs()
    checks.append(dict(position=position, layer=layer_id, ordinal=cursor,
                       kind=kind, exact=bool(torch.equal(actual, expected)),
                       mismatches=int((actual != expected).sum()),
                       max_absolute_error=float(error.max()),
                       relative_rms=float(error.norm()/expected.norm().clamp(min=1e-30))))


def consume(kind):
    global cursor
    op = ops[cursor]
    assert op['kind'] == kind, (cursor, kind, op['kind'])
    cursor += 1
    return op


def norm(module, x):
    outputs = []
    for row in x.reshape(-1, x.shape[-1]):
        op = consume('norm')
        compare(row, op['input'], 'norm.input')
        outputs.append(torch.tensor(op['output'], dtype=x.dtype))
    return torch.stack(outputs).reshape(x.shape)


def linear(module, x):
    if ops[cursor]['kind'] == 'attention_gate':
        # The upstream attention result reaches wo after the gate. Unlike the
        # dense/norm hooks, this checks an actual (non-injected) graph edge.
        op = consume('attention_gate')
        compare(x, op['output'], 'attention_gate.output')
    op = consume('linear')
    compare(x, op['input'], 'linear.input')
    return torch.tensor(op['output'], dtype=x.dtype).reshape(*x.shape[:-1], -1)


def mlp(module, x):
    gate, up = module.w1(x), module.w3(x)
    op = consume('silu_mul')
    compare(gate, op['input'], 'silu.gate')
    compare(up, op['auxiliary'], 'silu.up')
    compare(F.silu(gate)*up, op['output'], 'silu.output')
    hidden = torch.tensor(op['output'], dtype=x.dtype).reshape(gate.shape)
    return module.w2(hidden)


def attention(query, key, value, **kwargs):
    result = original_sdpa(query, key, value, **kwargs)
    scores = query.float() @ key.float().transpose(-1,-2) * query.shape[-1]**-.5
    if kwargs.get('attn_mask') is not None:
        scores = scores.masked_fill(~kwargs['attn_mask'], -torch.inf)
    probabilities = scores.softmax(-1)
    full = (probabilities @ value.float()).to(result.dtype)
    rounded = (probabilities.to(result.dtype).float() @ value.float()).to(result.dtype)
    native_core = torch.tensor(ops[cursor]['input']).to(result.dtype).reshape(result.shape)
    def metric(x,y):
        error = x.float()-y.float()
        return dict(exact_fraction=float((x==y).float().mean()), relative_rms=float(error.norm()/y.float().norm().clamp(min=1e-30)))
    attention_checks.append(dict(position=position,layer=layer_id,
        upstream_vs_full=metric(result,full),upstream_vs_rounded_probabilities=metric(result,rounded),
        native_vs_full=metric(native_core,full),native_vs_rounded_probabilities=metric(native_core,rounded)))
    return result


with torch.inference_mode(), patch.object(ZeroCenteredRMSNorm, 'forward', norm), patch.object(torch.nn.Linear, 'forward', linear), patch.object(TowerMLP, 'forward', mlp), patch.object(F, 'scaled_dot_product_attention', attention):
    for step in native:
        position = step['position']
        frame, next_frame = torch.tensor([step['frame']]), torch.tensor([step['next']])
        semantic = decoder.is_semantic(next_frame[:, 0])
        unseen = decoder.embedding_sum(frame, 1 + config.audio_decoder.vq_backbone_codebooks, config.audio_decoder.num_codebooks)
        unseen = torch.where(decoder.is_semantic(frame[:, 0])[:, None], unseen, 0.0)
        for layer_id, layer in enumerate(layers):
            after = next(v for v in step['layers'] if v['layer'] == layer_id and v['after'])
            ops = after.get('operators', [])
            assert ops, 'record all layers for graph replay'
            cursor = 0
            before = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and not v['after'])
            shared = layer_id < config.audio_decoder.shared_enrichment_layers
            code_index = 1 if shared else layer_id - config.audio_decoder.shared_enrichment_layers + 1
            layer(torch.tensor(before, dtype=torch.bfloat16)[None], next_frame[:,code_index], semantic,
                  torch.tensor([position]), unseen if layer_id == 0 else None)
            assert cursor == len(ops), (cursor, len(ops))
report = dict(admitted=False, scope=__doc__, torch=torch.__version__, source_hashes=HASHES, checks=checks, attention_checks=attention_checks)
a.out.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(checks=len(checks), mismatches=[v for v in checks if not v['exact']])))
