# Supported Models

Support depends on all four dimensions: source format, architecture, weight
encoding, and backend. A model family name alone is not sufficient.

## GGUF Architecture Matrix

| GGUF `general.architecture` | CPU | CUDA | Notes |
|---|---|---|---|
| `llama` | Supported | Supported for standard dense layouts | Includes compatible Llama-family geometry |
| `qwen2`, Qwen2.5 aliases | Supported | Supported for standard dense layouts | Includes declared coder aliases |
| `qwen3` | Supported | Supported for standard dense layouts | Dense text models |
| `qwen35`, `qwen3_5`, `qwen3_next` aliases | Supported | Not generally supported | Hybrid/MoE execution remains outside standard CUDA dense path |
| `gemma4`, `gemma_4` | Supported | Beta | Architecture-specific attention, FFN, and KV handling |

Architectures not listed by the loader fail explicitly. In particular,
Qwen Omni and GLM vision-language models require native multimodal stacks that
are not implemented.

## GGUF Weight Types

The GGUF model path recognizes and executes these primary weight types:

| Type | CPU | CUDA dense path |
|---|---|---|
| F32 | Yes | Yes |
| F16 | Yes | Yes |
| BF16 | Yes | Yes |
| Q8_0 | Yes | Yes |
| Q4_0 | Yes | Yes |
| Q4_K | Yes | Yes |
| Q5_K | Yes | Yes |
| Q6_K | Yes | Yes |

Tensor shape, row geometry, model architecture, and required tensor names must
also match. A file using a listed dtype can still be unsupported if its model
layout is unsupported.

## Hugging Face SafeTensors

SafeTensors directory loading is currently CUDA-only. Supported model types are
dense Qwen2 and Qwen3 with these validated storage families:

- standard dense floating-point weights;
- AutoAWQ GEMM and GEMV 4-bit layouts;
- GPTQ v1/v2 GEMM4 layouts accepted by the metadata validator;
- compressed-tensors W4A16 pack-quantized layouts.

The adapters enforce bits, group size, zero encoding, activation ordering,
packing, tensor geometry, and required metadata. Unsupported variants fail
before upload. CPU SafeTensors decode is not implemented.

## Tokenizers and Chat Templates

- GGUF tokenizers are read from model metadata.
- Hugging Face directories use their tokenizer/config assets.
- Chat requests use the model template when supported and fall back to the
  runtime's compatible template path where defined.

Model output quality and tool-call formatting depend on the model and its
template; loading support does not guarantee instruction-tuning quality.

## Vision

The text runtime can load a separate mmproj GGUF for compatible image-text
inputs. This is an experimental path, not support for native Omni/audio stacks.

The `xrt-vision` crate separately serves ONNX image tasks such as background
removal. Those models are not language-model backends and have their own model
file requirements.

## Backend Selection Rules

- `cpu`: requires a compatible GGUF file.
- `cuda`: requires a CUDA-enabled build and a supported GGUF or SafeTensors
  layout; failure is explicit.
- `auto`: may select CUDA for a compatible GGUF model, then fall back to CPU if
  CUDA cannot initialize. Non-CUDA builds use CPU.
- SafeTensors directories do not fall back to CPU.

## Adding Support

A support claim requires:

1. strict metadata and tensor-layout validation;
2. synthetic fixture coverage;
3. CPU reference parity where a CPU path exists;
4. real-model smoke evidence for the target backend;
5. an update to this matrix and the changelog.

CUDA implementation details and prior evidence are recorded in
[GPU_RUNTIME_ACCELERATION_SPEC.md](GPU_RUNTIME_ACCELERATION_SPEC.md).
