mod config;
mod edit;
mod edit_processor;
mod pipeline;
mod prompt;
mod tensors;
mod text_encoder;
mod text_encoder_cpu;
mod transformer;
#[cfg(feature = "cuda")]
mod transformer_cuda;
mod transformer_executor;
mod transformer_gguf;
mod transformer_safetensors;
mod vae;
mod vae_decoder;
mod vision_encoder_cpu;

pub use config::{
    QwenImageBundleConfig, QwenImageTextConfig, QwenImageTransformerConfig, QwenImageVaeConfig,
    QwenImageVisionConfig,
};
pub(crate) use edit::QwenImageEditPipeline;
pub use edit_processor::{QwenImageEditImageBatch, QwenImageEditProcessor, QwenImageVaeSource};
pub(crate) use pipeline::QwenImagePipeline;
pub use prompt::{
    QwenImagePromptTokenizer, QwenImageTokenBatch, QWEN_IMAGE_EDIT_PROMPT_TEMPLATE_DROP_TOKENS,
    QWEN_IMAGE_PROMPT_TEMPLATE_DROP_TOKENS,
};
pub use tensors::{
    expected_transformer_tensors, open_transformer_gguf, open_transformer_safetensors,
    validate_transformer_gguf, validate_transformer_safetensors, ExpectedTensor,
};
#[cfg(feature = "cuda")]
pub use text_encoder::QwenImageCudaTextEncoder;
pub use text_encoder::QwenImagePromptEmbeddings;
pub use text_encoder_cpu::QwenImageCpuTextEncoder;
pub use transformer::{
    pack_latents, qwen_image_rotary_embeddings, qwen_image_rotary_embeddings_for_shapes,
    qwen_image_transformer_block_bf16, qwen_image_transformer_block_f32, qwen_timestep_projection,
    unpack_latents, QwenImageBf16Linear, QwenImageGgufLinear, QwenImageLinear,
    QwenImageRotaryEmbeddings, QwenImageTransformerBlockWeights,
};
#[cfg(feature = "cuda")]
pub use transformer_cuda::QwenImageCudaTransformer;
pub use transformer_gguf::QwenImageGgufTransformer;
pub use transformer_safetensors::QwenImageBf16Transformer;
pub use vae::{expected_vae_tensors, open_vae_safetensors, validate_vae_safetensors};
pub use vae_decoder::{
    load_vae_decoder_f32_weights, load_vae_encoder_f32_weights, qwen_image_vae_decode_f32,
    qwen_image_vae_decode_tiled_f32, qwen_image_vae_decode_tiled_f32_with_control,
    qwen_image_vae_encode_f32, qwen_image_vae_encode_f32_with_control, QwenImageVaeF32Weights,
    QwenImageVaeTiling,
};
pub use vision_encoder_cpu::{
    QwenImageCpuVisionEncoder, QwenImageVisionEmbeddings, QwenImageVisionInput,
};
