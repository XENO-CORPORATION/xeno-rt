"""CPU/PyTorch mathematical reference for the pinned Triton W8A8 contract.

The remaining transformer code is the actual upstream implementation. This
operator emulates group quantization/GEMM because CPU torch cannot run Triton.
"""
import types
import torch
import torch.nn.functional as F


def quantize(x):
    groups = x.float().split(128, -1)
    scales = [v.abs().amax(-1).clamp(min=1e-10) / 448.0 for v in groups]
    q = [(v / s[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float() for v, s in zip(groups, scales)]
    return torch.cat(q, -1), torch.stack(scales, -1)


def gemm(q, activation_scale, weight, weight_scale):
    out = torch.zeros(q.shape[0], weight.shape[0])
    row_groups = torch.arange(weight.shape[0]) // 128
    for group, begin in enumerate(range(0, q.shape[-1], 128)):
        dot = q[:, begin:begin + 128] @ weight[:, begin:begin + 128].float().T
        out += dot * weight_scale[row_groups, group][None] * activation_scale[:, group:group + 1]
    return out


def install(experts):
    def forward(self, x, topk_ids, topk_weights):
        output = torch.zeros_like(x, dtype=torch.float32)
        q, scales = quantize(x)
        for token in range(x.shape[0]):
            for slot in range(topk_ids.shape[-1]):
                expert = int(topk_ids[token, slot])
                gate_up = gemm(q[token:token + 1], scales[token:token + 1],
                               self.w13_weight[expert].float(), self.w13_scale_inv[expert]).to(x.dtype)
                gate, up = gate_up.float().chunk(2, -1)
                h, h_scale = quantize(F.silu(gate) * up)
                down = gemm(h, h_scale, self.w2_weight[expert].float(), self.w2_scale_inv[expert])
                output[token] += down[0] * topk_weights[token, slot].float()
        return output.to(x.dtype)
    experts.forward = types.MethodType(forward, experts)
