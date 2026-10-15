"""Check every trained backbone layer against actual pinned Fish CPU/F32 code.

Keeps only one upstream layer resident at a time. Source lives outside the repo;
its hashes are verified before import. This is a reference tool, not inference.
"""
import argparse
import gc
import hashlib
import json
import pathlib
import re
import torch
from contextlib import nullcontext
from torch.nn.attention import SDPBackend, sdpa_kernel
from unittest.mock import patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=pathlib.Path, required=True)
p.add_argument('--model', type=pathlib.Path, required=True)
p.add_argument('--native', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
p.add_argument("--bf16-w8a8", action="store_true")
p.add_argument('--sdpa-backend', choices=['default', 'math'], default='default')
p.add_argument('--composed', action='store_true', help='propagate upstream outputs through the full graph, rather than resetting every layer to native inputs')
p.add_argument('--stable-routing', action='store_true', help='diagnostic control: use deterministic expert-index ordering for equal router probabilities')
p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
p.add_argument('--batched-prefill', action='store_true', help='actual upstream whole-prompt prefill, including chunk GDN')
p.add_argument('--capture-reference', action='store_true', help='retain reference outputs for independent backend-variation calibration')
a = p.parse_args()
assert a.device == 'cpu' or a.bf16_w8a8, 'CUDA reference executes the actual BF16/W8A8 serving operator'
assert not a.batched_prefill or a.composed, 'batched prefill requires full composition'
from drama3_upstream_reference import HASHES, TensorReader, load
load(a.source)
hashes = HASHES
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer
from fish_speech.models.text2semantic.fish_omni.layers import LayerContext

torch.set_num_threads(8)
torch.backends.mkldnn.enabled = False
full_config = json.loads((a.model / 'config.json').read_text())
config = FishOmniTextConfig.from_dict(full_config['text_config'])
index = json.loads((a.model / 'model.safetensors.index.json').read_text())['weight_map']
native_bytes = a.native.read_bytes()
native_report = json.loads(native_bytes)
native = native_report['traces']
assert 1 <= len(native) <= 64 and [step['position'] for step in native] == list(range(len(native))), 'require contiguous bounded probe positions'
prompt_rows = native_report.get('prompt_rows', len(native))
assert 0 <= prompt_rows <= len(native)
prefill_rows = prompt_rows or len(native)
assert a.composed or len(native)==2, 'isolated historical oracle requires both successive positions'
expert_pattern = re.compile(r'mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$')
results = []
reference_hidden = {step['position']:torch.tensor(step['input'],dtype=torch.bfloat16 if a.bf16_w8a8 else torch.float32,device=a.device)[None,None] for step in native}
def stable_topk(value,k,dim=-1,largest=True,sorted=True):
    indices=torch.argsort(value,dim=dim,descending=largest,stable=True).narrow(dim,0,k)
    return value.gather(dim,indices),indices

with torch.inference_mode(), (sdpa_kernel(SDPBackend.MATH) if a.sdpa_backend == 'math' else nullcontext()), (patch.object(torch.Tensor,'topk',stable_topk) if a.stable_routing else nullcontext()):
    for layer_id in range(config.num_hidden_layers):
        layer = DecoderLayer(config, layer_id, fp8=True, dtype=torch.bfloat16 if a.bf16_w8a8 else torch.float32, device=a.device).eval()
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
        if a.bf16_w8a8 and a.device == 'cpu':
            from drama3_w8a8_reference import install
            install(layer.mlp.experts)
        dtype = torch.bfloat16 if a.bf16_w8a8 else torch.float32
        if layer.layer_type == 'full_attention':
            layer.self_attn.allocate_cache(1, len(native), dtype, a.device)
        else:
            layer.linear_attn.allocate_cache(1, dtype, a.device)
        batched_output=None
        if a.batched_prefill:
            positions=torch.arange(prefill_rows,device=a.device)
            frequency=1.0/(config.rope_theta**(torch.arange(0,config.rotary_dim,2,device=a.device).float()/config.rotary_dim))
            angles=positions[:,None]*frequency[None,:]
            angles=torch.cat((angles,angles),-1)
            context=LayerContext(input_pos=positions,cos=angles.cos(),sin=angles.sin(),kv_len=prefill_rows,continue_state=False,use_cache=True)
            batched_output=layer(torch.cat([reference_hidden[s['position']] for s in native[:prefill_rows]],dim=1),context)
        for step in native:
            position = step['position']
            before = reference_hidden[position] if a.composed else torch.tensor(next(v['values'] for v in step['layers'] if v['layer'] == layer_id and not v['after']),dtype=dtype,device=a.device)[None,None]
            after = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and v['after'])
            rotary = config.rotary_dim
            frequency = 1.0 / (config.rope_theta ** (torch.arange(0, rotary, 2, device=a.device).float() / rotary))
            angles = torch.cat((position * frequency, position * frequency))[None]
            context = LayerContext(input_pos=torch.tensor([position],device=a.device), cos=angles.cos(), sin=angles.sin(), kv_len=position + 1, continue_state=position > 0, use_cache=True)
            expected = (batched_output[0,position] if batched_output is not None and position < prefill_rows else layer(before, context)).flatten().float()
            reference_hidden[position] = expected.to(dtype)[None,None]
            expected=expected.cpu()
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
model_outputs = []
reference_outputs = []
if a.composed:
    from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm
    import torch.nn.functional as F
    dtype=torch.bfloat16 if a.bf16_w8a8 else torch.float32
    name='model.language_model.norm.weight'
    with TensorReader(a.model/index[name]) as reader: norm_weight=reader.get_tensor(name).to(dtype)
    norm=ZeroCenteredRMSNorm(config.hidden_size,config.rms_norm_eps,dtype=dtype,device=a.device).eval()
    norm.weight.data.copy_(norm_weight)
    begin=full_config['semantic_start_token_id'];end=full_config['semantic_end_token_id']+1;eos=full_config['eos_token_id']
    with TensorReader(a.model/index['lm_head.weight']) as reader:
        head=reader.get_tensor('lm_head.weight')
        selected=torch.cat((head[begin:end],head[eos:eos+1])).to(device=a.device,dtype=dtype)
    with torch.inference_mode():
        for step in native:
            hidden=norm(reference_hidden[step['position']]).flatten()
            expected=F.linear(hidden,selected).float().cpu()
            hidden=hidden.cpu()
            actual=torch.tensor(step['semantic_logits'])
            if a.capture_reference: reference_outputs.append(dict(position=step['position'],hidden=hidden.float().tolist(),semantic_logits=expected.tolist()))
            assert actual.shape==expected.shape
            actual_lp=actual.log_softmax(-1);expected_lp=expected.log_softmax(-1)
            top=expected.topk(20).indices
            model_outputs.append(dict(position=step['position'],generation_position=prompt_rows > 0 and step['position'] >= prompt_rows-1,hidden_relative_rms=float((torch.tensor(step['hidden'])-hidden.float()).norm()/hidden.float().norm().clamp(min=1e-30)),
                logit_relative_rms=float((actual-expected).norm()/expected.norm().clamp(min=1e-30)),
                top20_logprob_max_error=float((actual_lp[top]-expected_lp[top]).abs().max()),
                top20_logprob_rms=float((actual_lp[top]-expected_lp[top]).square().mean().sqrt()),
                reference_top1=int(expected.argmax()),native_top1=int(actual.argmax()),
                kl_reference_to_native=float((expected_lp.exp()*(expected_lp-actual_lp)).sum()),
                top20_logprobs_close=bool(torch.allclose(actual_lp[top],expected_lp[top],rtol=.02,atol=.02))))
report = {'admitted': False, 'scope': 'every trained native backbone layer vs actual pinned Fish code',
          'native_trace_sha256':hashlib.sha256(native_bytes).hexdigest(),'prompt_rows':prompt_rows,'teacher_forced_frames':native_report.get('teacher_forced_frames',0),'teacher_forced_sha256':native_report.get('teacher_forced_sha256'),
          'source_commit': '541566932abbaebcb0a40e562a1f846738383f38', 'source_hashes': hashes,
          'torch': torch.__version__, 'device':a.device,'sdpa_backend':a.sdpa_backend, 'stable_routing_control':a.stable_routing, 'batched_prefill':a.batched_prefill,'composed':a.composed,'model_outputs':model_outputs,'reference_outputs':reference_outputs, 'numerics': ('BF16 with actual pinned Triton W8A8 operator' if a.device=='cuda' else 'BF16 with CPU-emulated W8A8 expert operator') if a.bf16_w8a8 else 'actual upstream FP32 CPU', 'checks': results, 'passed': all(v['passed'] for v in results)}
a.out.write_text(json.dumps(report, indent=2) + '\n')
assert report['passed'], 'trained backbone layer mismatch'
