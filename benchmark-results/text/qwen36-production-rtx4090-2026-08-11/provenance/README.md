---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- dflash
- speculative-decoding
- diffusion
- efficiency
- flash-decoding
- qwen
- diffusion-language-model
---

# Qwen3.6-27B-DFlash
[**Paper**](https://arxiv.org/abs/2602.06036) | [**GitHub**](https://github.com/z-lab/dflash) | [**Blog**](https://z-lab.ai/projects/dflash/)

**This model is still under training, and inference engine support may not be fully available yet due to architectural changes, including causal SWA layers.**

**DFlash** is a novel speculative decoding method that utilizes a lightweight **block diffusion** model for drafting. It enables efficient, high-quality parallel drafting that pushes the limits of inference speed.

This model is the **drafter** component. It must be used in conjunction with the target model `Qwen/Qwen3.6-27B`.

<div align="center">
  <img src="assets/dflash_system.png" alt="DFlash Architecture" width="100%">
</div>

## Quick Start

### Installation

vLLM (We temporarily modify the installation through this PR to support interleaved SWA and ensure correct handling of target hidden states for optimal performance):
```bash
uv pip install vllm
uv pip install -U --torch-backend=auto "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/40898/head"
```

SGLang:
```bash
uv pip install "git+https://github.com/sgl-project/sglang.git@refs/pull/23000/head#subdirectory=python"
```

### Launch Server

vLLM:
```bash
vllm serve Qwen/Qwen3.6-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.6-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

SGLang:
```bash
# Optional: enable schedule overlapping (experimental, may not be stable)
# export SGLANG_ENABLE_SPEC_V2=1
# export SGLANG_ENABLE_DFLASH_SPEC_V2=1
# export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend fa3 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

### Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B",
    messages=[{"role": "user", "content": "Write a quicksort in Python."}],
    max_tokens=4096,
    temperature=0.0
)
print(response.choices[0].message.content)
```

## Benchmark Results

N/A

## Acknowledgements

Special thanks to [David Wang](https://davidwa.ng/) for his outstanding engineering support on this project. We are also grateful to [Modal](https://modal.com/), [InnoMatrix](https://innomatrix.ai), and [Yotta Labs](https://www.yottalabs.ai/) for providing the compute resources used to train this draft model.

## Citation

If you find DFlash useful, please cite our work. To share feedback on DFlash or request new model support, please fill out this form: [DFlash Feedback](https://forms.gle/4YNwfqb4nJdqn6hq9).

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```