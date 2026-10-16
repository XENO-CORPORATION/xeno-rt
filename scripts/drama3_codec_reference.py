"""Independent tensor-graph oracle using ordinary PyTorch operators.

This reconstructs the pinned graph, not Fish's packaged serving executable.
Only provisioning/qualification scripts import it; Rust inference never does.
"""
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class TensorGraph:
    def __init__(self, root):
        self.w = load_file(str(root / 'model.safetensors'))

    def conv(self, x, name, stride=1, dilation=1, groups=1, transpose=False):
        weight, bias = self.w[name + '.weight'], self.w[name + '.bias']
        kernel = weight.shape[-1]
        if transpose:
            out = F.conv_transpose1d(x, weight, bias, stride=stride)
            return out[..., :x.shape[-1] * stride]
        left = (kernel - 1) * dilation + 1 - stride
        return F.conv1d(F.pad(x, (left, (-x.shape[-1]) % stride)), weight, bias,
                        stride=stride, dilation=dilation, groups=groups)

    def linear(self, x, name):
        return F.linear(x, self.w[name + '.weight'], self.w.get(name + '.bias'))

    def snake(self, x, name):
        alpha = self.w[name + '.alpha']
        return x + torch.sin(alpha * x).square() / (alpha + 1e-9)

    def residual(self, x, prefix, dilation):
        y = self.conv(self.snake(x, prefix + '.block.0'), prefix + '.block.1.conv', dilation=dilation)
        return x + self.conv(self.snake(y, prefix + '.block.2'), prefix + '.block.3.conv')

    def convnext(self, x, prefix):
        y = self.conv(x, prefix + '.dwconv.conv', groups=x.shape[1]).transpose(1, 2)
        y = F.layer_norm(y, (y.shape[-1],), self.w[prefix + '.norm.weight'], self.w[prefix + '.norm.bias'], 1e-6)
        y = self.linear(F.gelu(self.linear(y, prefix + '.pwconv1')), prefix + '.pwconv2')
        return x + (y * self.w[prefix + '.gamma']).transpose(1, 2)

    def rms(self, x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.w[name + '.weight']

    def transformer(self, x, module, layers, window):
        x = x.transpose(1, 2)
        length, width = x.shape[1:]
        heads = width // 64
        angles = torch.arange(length).float()[:, None] / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))

        def rotate(t):
            t = t.reshape(1, length, heads, 64)
            even, odd = t[..., ::2], t[..., 1::2]
            cos, sin = angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]
            return torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1).flatten(-2).transpose(1, 2)

        time = torch.arange(length)
        mask = (time[None, :] <= time[:, None]) & (time[None, :] >= time[:, None] - window + 1)
        for layer in range(layers):
            prefix = f'{module}.layers.{layer}'
            q, k, v = self.linear(self.rms(x, prefix + '.attention_norm'), prefix + '.attention.wqkv').chunk(3, -1)
            out = F.scaled_dot_product_attention(rotate(q), rotate(k), v.reshape(1, length, heads, 64).transpose(1, 2), attn_mask=mask)
            x = x + self.linear(out.transpose(1, 2).flatten(-2), prefix + '.attention.wo') * self.w[prefix + '.attention_layer_scale.gamma']
            normalized = self.rms(x, prefix + '.ffn_norm')
            p = prefix + '.feed_forward'
            x = x + self.linear(F.silu(self.linear(normalized, p + '.w1')) * self.linear(normalized, p + '.w3'), p + '.w2') * self.w[prefix + '.ffn_layer_scale.gamma']
        return self.rms(x, module + '.norm').transpose(1, 2)

    def encode(self, samples):
        x = samples[None, None]
        x = F.pad(x, (0, (-x.shape[-1]) % 2048))
        x = self.conv(x, 'encoder.block.0.conv')
        for i, stride in enumerate([2, 4, 8, 8]):
            p = f'encoder.block.{i + 1}.block'
            for j, dilation in enumerate([1, 3, 9]):
                x = self.residual(x, f'{p}.{j}', dilation)
            x = self.conv(self.snake(x, p + '.3'), p + '.4.conv', stride=stride)
            if i == 3:
                x = self.transformer(x, p + '.5', 4, 512)
        x = self.conv(self.snake(x, 'encoder.block.5'), 'encoder.block.6.conv')
        for i in range(2):
            x = self.convnext(self.conv(x, f'quantizer.downsample.{i}.0.conv', stride=2), f'quantizer.downsample.{i}.1')
        residual = self.transformer(x, 'quantizer.pre_module', 8, 128)
        codes = []
        for book in range(10):
            p = 'quantizer.semantic_quantizer.quantizers.0' if book == 0 else f'quantizer.quantizer.quantizers.{book - 1}'
            encodings = F.normalize(self.conv(residual, p + '.in_proj')[0].T)
            raw = self.w[p + '.codebook.weight']
            table = F.normalize(raw)
            distances = encodings.square().sum(1, keepdim=True) - 2 * encodings @ table.T + table.square().sum(1)[None]
            ids = distances.argmin(1)
            codes.append(ids)
            residual = residual - self.conv(F.embedding(ids, raw).T[None], p + '.out_proj')
        return torch.stack(codes, -1)

    def decode(self, codes):
        latent = 0
        for book in range(10):
            p = 'quantizer.semantic_quantizer.quantizers.0' if book == 0 else f'quantizer.quantizer.quantizers.{book - 1}'
            table = self.w[p + '.codebook.weight']
            latent = latent + self.conv(F.embedding(codes[:, book].clamp(max=table.shape[0] - 1), table).T[None], p + '.out_proj')
        x = self.transformer(latent, 'quantizer.post_module', 8, 128)
        for i in range(2):
            x = self.convnext(self.conv(x, f'quantizer.upsample.{i}.0.conv', stride=2, transpose=True), f'quantizer.upsample.{i}.1')
        x = self.conv(x, 'decoder.model.0.conv')
        for i, stride in enumerate([8, 8, 4, 2]):
            p = f'decoder.model.{i + 1}.block'
            x = self.conv(self.snake(x, p + '.0'), p + '.1.conv', stride=stride, transpose=True)
            for j, dilation in enumerate([1, 3, 9]):
                x = self.residual(x, f'{p}.{j + 2}', dilation)
        return torch.tanh(self.conv(self.snake(x, 'decoder.model.5'), 'decoder.model.6.conv')).flatten()
