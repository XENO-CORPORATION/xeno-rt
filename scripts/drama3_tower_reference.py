"""Actual pinned tower loading/teacher forcing shared by qualification tools."""
import json
import torch
from drama3_upstream_reference import TensorReader


def trained_tower(model, config, dtype):
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder, TOWER_CHECKPOINT_PREFIXES
    decoder = FishOmniAudioDecoder(config, dtype=dtype, device='cpu').eval()
    targets = dict(decoder.named_parameters())
    files = {}
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    for name, shard in index.items():
        if name.startswith(tuple(TOWER_CHECKPOINT_PREFIXES)):
            files.setdefault(shard, []).append(name)
    filled = set()
    with torch.inference_mode():
        for shard, names in files.items():
            with TensorReader(model / shard) as store:
                for name in names:
                    target = next(new + name[len(old):] for old, new in TOWER_CHECKPOINT_PREFIXES.items() if name.startswith(old))
                    value = store.get_tensor(name)
                    assert targets[target].shape == value.shape and target not in filled
                    targets[target].copy_(value)
                    filled.add(target)
    assert filled == set(targets), set(targets) - filled
    return decoder


def teacher_forced(decoder, config, native, callback):
    decoder.setup_caches(len(native))
    layers = [*decoder.shared_layers, *decoder.residual_layers]
    dtype = decoder.codebook_embeddings.weight.dtype
    with torch.inference_mode():
        for step in native:
            position = step['position']
            frame, next_frame = torch.tensor([step['frame']]), torch.tensor([step['next']])
            semantic = decoder.is_semantic(next_frame[:, 0])
            unseen = decoder.embedding_sum(frame, 1 + config.audio_decoder.vq_backbone_codebooks, config.audio_decoder.num_codebooks)
            unseen = torch.where(decoder.is_semantic(frame[:, 0])[:, None], unseen, 0.0)
            for layer_id, layer in enumerate(layers):
                before = next(v['values'] for v in step['layers'] if v['layer'] == layer_id and not v['after'])
                shared = layer_id < config.audio_decoder.shared_enrichment_layers
                code_index = 1 if shared else layer_id - config.audio_decoder.shared_enrichment_layers + 1
                expected = layer(torch.tensor(before, dtype=dtype)[None], next_frame[:, code_index], semantic,
                                 torch.tensor([position]), unseen if layer_id == 0 else None).flatten().float()
                callback(position, layer_id, expected)
