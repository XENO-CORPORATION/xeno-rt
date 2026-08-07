use std::{collections::BTreeMap, sync::Arc};

use xrt_core::{DType, XrtError};
use xrt_cuda::{
    CudaBytes, CudaDevice, CudaF32Buffer, CudaQ4KMatrix, CudaQ5KMatrix, CudaQ6KMatrix,
    CudaQ8_0Matrix, CudaTransferStats, GpuF32Tensor,
};
use xrt_gguf::{GgufFile, QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER};
use xrt_runtime::{GpuAllocationClass, GpuAllocationLease, GpuResourceManager};

use crate::ImageError;

use super::{
    qwen_image_rotary_embeddings_for_shapes, qwen_timestep_projection, validate_transformer_gguf,
    QwenImageDistilledProfile, QwenImageLoraAdapter, QwenImagePromptEmbeddings,
    QwenImageTransformerConfig,
};

const TRANSIENT_ACTIVATION_MULTIPLIER: u64 = 24;

enum CudaMatrixStorage {
    Dense(GpuF32Tensor),
    Q8(CudaQ8_0Matrix),
    Q4K(CudaQ4KMatrix),
    Q5K(CudaQ5KMatrix),
    Q6K(CudaQ6KMatrix),
}

struct CudaMatrix {
    input_features: usize,
    output_features: usize,
    storage: CudaMatrixStorage,
}

struct CudaLoraLayer {
    down: GpuF32Tensor,
    up: GpuF32Tensor,
    rank: usize,
    input_features: usize,
    output_features: usize,
    scale: f32,
}

struct CudaLoraAdapter {
    layers: BTreeMap<String, CudaLoraLayer>,
    profile: QwenImageDistilledProfile,
    byte_len: u64,
}

impl CudaLoraAdapter {
    fn upload(device: &CudaDevice, adapter: &QwenImageLoraAdapter) -> Result<Self, ImageError> {
        let mut layers = BTreeMap::new();
        let mut byte_len = 0u64;
        for prefix in adapter.layer_names() {
            let view = adapter.layer_view(prefix)?.ok_or_else(|| {
                ImageError::Internal(format!(
                    "validated Lightning layer `{prefix}` disappeared during CUDA upload"
                ))
            })?;
            let down_name = format!("{prefix}.lora_down.weight");
            let up_name = format!("{prefix}.lora_up.weight");
            let down = device
                .upload_f32_tensor_transposed_2d_bytes(
                    &down_name,
                    view.rank,
                    view.input_features,
                    DType::BF16,
                    view.down,
                )
                .map_err(map_cuda_load_error)?;
            let up = device
                .upload_f32_tensor_transposed_2d_bytes(
                    &up_name,
                    view.output_features,
                    view.rank,
                    DType::BF16,
                    view.up,
                )
                .map_err(map_cuda_load_error)?;
            byte_len = byte_len
                .checked_add(down.byte_len() as u64)
                .and_then(|total| total.checked_add(up.byte_len() as u64))
                .ok_or_else(|| {
                    ImageError::Admission("CUDA Lightning byte count overflowed".to_string())
                })?;
            layers.insert(
                prefix.to_string(),
                CudaLoraLayer {
                    down,
                    up,
                    rank: view.rank,
                    input_features: view.input_features,
                    output_features: view.output_features,
                    scale: view.scale,
                },
            );
        }
        if byte_len != adapter.cuda_bytes() {
            return Err(ImageError::Internal(format!(
                "CUDA Lightning reservation estimated {} bytes but uploaded {byte_len} bytes",
                adapter.cuda_bytes()
            )));
        }
        Ok(Self {
            layers,
            profile: adapter.profile(),
            byte_len,
        })
    }
}

impl CudaMatrix {
    fn byte_len(&self) -> u64 {
        match &self.storage {
            CudaMatrixStorage::Dense(tensor) => tensor.byte_len() as u64,
            CudaMatrixStorage::Q8(matrix) => matrix
                .scale_count()
                .saturating_mul(std::mem::size_of::<f32>())
                .saturating_add(matrix.quant_byte_len())
                as u64,
            CudaMatrixStorage::Q4K(matrix)
            | CudaMatrixStorage::Q5K(matrix)
            | CudaMatrixStorage::Q6K(matrix) => matrix.byte_len() as u64,
        }
    }

    fn forward(
        &self,
        device: &CudaDevice,
        input: &CudaF32Buffer,
        rows: usize,
    ) -> Result<CudaF32Buffer, ImageError> {
        match &self.storage {
            CudaMatrixStorage::Dense(tensor) => device.matmul_resident_rhs_device(
                input,
                rows,
                self.input_features,
                tensor.buffer(),
                self.output_features,
            ),
            CudaMatrixStorage::Q8(matrix) => {
                device.matmul_q8_0_resident_device(matrix, input, rows)
            }
            CudaMatrixStorage::Q4K(matrix) => {
                device.matmul_q4_k_resident_device(matrix, input, rows)
            }
            CudaMatrixStorage::Q5K(matrix) => {
                device.matmul_q5_k_resident_device(matrix, input, rows)
            }
            CudaMatrixStorage::Q6K(matrix) => {
                device.matmul_q6_k_resident_device(matrix, input, rows)
            }
        }
        .map_err(map_cuda_execution_error)
    }
}

/// Resident mixed-format CUDA executor for the Qwen Image denoiser.
///
/// The initial admitted path intentionally keeps prompt encoding and VAE
/// decoding outside this object. Transformer weights remain resident for the
/// lifetime of the executor, while every forward call uploads its inputs once,
/// executes all layers on device, and downloads only the final prediction.
pub struct QwenImageCudaTransformer {
    config: QwenImageTransformerConfig,
    device: CudaDevice,
    resources: Arc<GpuResourceManager>,
    matrices: BTreeMap<String, CudaMatrix>,
    auxiliary: BTreeMap<String, GpuF32Tensor>,
    adapter: Option<CudaLoraAdapter>,
    weight_bytes: u64,
    _weight_lease: GpuAllocationLease,
}

impl std::fmt::Debug for QwenImageCudaTransformer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("QwenImageCudaTransformer")
            .field("config", &self.config)
            .field("matrix_count", &self.matrices.len())
            .field("auxiliary_count", &self.auxiliary.len())
            .field("has_adapter", &self.adapter.is_some())
            .field("weight_bytes", &self.weight_bytes)
            .finish_non_exhaustive()
    }
}

impl QwenImageCudaTransformer {
    pub fn from_file(
        file: GgufFile,
        config: QwenImageTransformerConfig,
        quantization: &str,
        resources: Arc<GpuResourceManager>,
    ) -> Result<Self, ImageError> {
        Self::from_file_with_adapter(file, config, quantization, resources, None)
    }

    pub fn from_file_with_adapter(
        file: GgufFile,
        config: QwenImageTransformerConfig,
        quantization: &str,
        resources: Arc<GpuResourceManager>,
        adapter: Option<QwenImageLoraAdapter>,
    ) -> Result<Self, ImageError> {
        validate_transformer_gguf(&file, &config, quantization)?;
        if config.use_additional_t_cond || config.use_layer3d_rope {
            return Err(ImageError::UnsupportedCapability(
                "CUDA executor does not accept additional timestep conditioning or Layer3D RoPE"
                    .to_string(),
            ));
        }

        let resource_config = resources.config();
        let device =
            CudaDevice::new(resource_config.device_ordinal).map_err(map_cuda_load_error)?;
        configure_resource_budget(&device, &resources)?;
        let estimated_weight_bytes = estimate_resident_weight_bytes(&file)?
            .checked_add(adapter.as_ref().map_or(0, QwenImageLoraAdapter::cuda_bytes))
            .ok_or_else(|| {
                ImageError::Admission("CUDA transformer weight estimate overflowed".to_string())
            })?;
        let weight_lease = resources
            .allocation_arena()
            .reserve(
                GpuAllocationClass::ImageComponentWeights,
                estimated_weight_bytes,
            )
            .map_err(map_cuda_admission_error)?;

        let mut matrices = BTreeMap::new();
        let mut auxiliary = BTreeMap::new();
        for info in file.tensor_infos() {
            if info.name == QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER {
                continue;
            }
            let bytes = file.tensor_data(&info.name).map_err(|error| {
                ImageError::CorruptComponent(format!(
                    "failed to map GGUF transformer tensor `{}`: {error}",
                    info.name
                ))
            })?;
            match info.dimensions.as_slice() {
                [input_features, output_features] => {
                    let storage = match info.dtype {
                        DType::F32 | DType::F16 | DType::BF16 => CudaMatrixStorage::Dense(
                            device
                                .upload_f32_tensor_transposed_2d_bytes(
                                    &info.name,
                                    *output_features,
                                    *input_features,
                                    info.dtype,
                                    bytes,
                                )
                                .map_err(map_cuda_load_error)?,
                        ),
                        DType::Q8_0 => CudaMatrixStorage::Q8(
                            device
                                .upload_q8_0_matrix(bytes, *output_features, *input_features)
                                .map_err(map_cuda_load_error)?,
                        ),
                        DType::Q4_K => CudaMatrixStorage::Q4K(
                            device
                                .upload_q4_k_matrix_packed(bytes, *output_features, *input_features)
                                .map_err(map_cuda_load_error)?,
                        ),
                        DType::Q5_K => CudaMatrixStorage::Q5K(
                            device
                                .upload_q5_k_matrix(bytes, *output_features, *input_features)
                                .map_err(map_cuda_load_error)?,
                        ),
                        DType::Q6_K => CudaMatrixStorage::Q6K(
                            device
                                .upload_q6_k_embedding_matrix_packed(
                                    bytes,
                                    *output_features,
                                    *input_features,
                                )
                                .map_err(map_cuda_load_error)?,
                        ),
                        other => {
                            return Err(ImageError::UnsupportedTensor(format!(
                                "CUDA Qwen Image matrix `{}` cannot use {other:?}",
                                info.name
                            )))
                        }
                    };
                    matrices.insert(
                        info.name.clone(),
                        CudaMatrix {
                            input_features: *input_features,
                            output_features: *output_features,
                            storage,
                        },
                    );
                }
                [_] => {
                    auxiliary.insert(
                        info.name.clone(),
                        device
                            .upload_f32_tensor_bytes(
                                &info.name,
                                &info.dimensions,
                                info.dtype,
                                bytes,
                            )
                            .map_err(map_cuda_load_error)?,
                    );
                }
                dimensions => {
                    return Err(ImageError::UnsupportedShape(format!(
                    "CUDA Qwen Image tensor `{}` has unsupported GGUF dimensions {dimensions:?}",
                    info.name
                )))
                }
            }
        }

        let adapter = adapter
            .as_ref()
            .map(|adapter| CudaLoraAdapter::upload(&device, adapter))
            .transpose()?;
        let actual_weight_bytes = matrices
            .values()
            .try_fold(0u64, |total, matrix| total.checked_add(matrix.byte_len()))
            .and_then(|total| {
                auxiliary.values().try_fold(total, |sum, tensor| {
                    sum.checked_add(tensor.byte_len() as u64)
                })
            })
            .and_then(|total| {
                total.checked_add(adapter.as_ref().map_or(0, |adapter| adapter.byte_len))
            })
            .ok_or_else(|| {
                ImageError::Admission("CUDA transformer weight byte count overflowed".to_string())
            })?;
        if actual_weight_bytes != estimated_weight_bytes {
            return Err(ImageError::Internal(format!(
                "CUDA transformer reservation estimated {estimated_weight_bytes} bytes but uploaded {actual_weight_bytes} bytes"
            )));
        }

        Ok(Self {
            config,
            device,
            resources,
            matrices,
            auxiliary,
            adapter,
            weight_bytes: actual_weight_bytes,
            _weight_lease: weight_lease,
        })
    }

    pub fn config(&self) -> &QwenImageTransformerConfig {
        &self.config
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn distilled_profile(&self) -> Option<QwenImageDistilledProfile> {
        self.adapter.as_ref().map(|adapter| adapter.profile)
    }

    pub fn transfer_stats(&self) -> CudaTransferStats {
        self.device.transfer_stats()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        frames: usize,
        patch_height: usize,
        patch_width: usize,
    ) -> Result<Vec<f32>, ImageError> {
        self.forward_with_control(
            packed_latents,
            prompt,
            timestep,
            frames,
            patch_height,
            patch_width,
            |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        frames: usize,
        patch_height: usize,
        patch_width: usize,
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        self.forward_for_shapes_with_control(
            packed_latents,
            prompt,
            timestep,
            &[[frames, patch_height, patch_width]],
            checkpoint,
        )
    }

    pub fn forward_edit_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        image_shapes: &[[usize; 3]],
        checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        self.forward_for_shapes_with_control(
            packed_latents,
            prompt,
            timestep,
            image_shapes,
            checkpoint,
        )
    }

    fn forward_for_shapes_with_control<F>(
        &self,
        packed_latents: &[f32],
        prompt: &QwenImagePromptEmbeddings,
        timestep: &[f32],
        image_shapes: &[[usize; 3]],
        mut checkpoint: F,
    ) -> Result<Vec<f32>, ImageError>
    where
        F: FnMut(usize) -> Result<(), ImageError>,
    {
        let config = &self.config;
        let batch = prompt.batch_size;
        if (config.zero_cond_t && image_shapes.len() < 2)
            || (!config.zero_cond_t && image_shapes.len() != 1)
        {
            return Err(ImageError::UnsupportedShape(format!(
                "CUDA transformer zero_cond_t={} is incompatible with {} ordered image sequence(s)",
                config.zero_cond_t,
                image_shapes.len()
            )));
        }
        let image_sequence = image_shapes.iter().try_fold(0usize, |total, shape| {
            let sequence = checked_product(shape, "CUDA transformer image sequence")?;
            total.checked_add(sequence).ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "CUDA transformer image sequence overflowed".to_string(),
                )
            })
        })?;
        let expected_latents = checked_product(
            &[batch, image_sequence, config.in_channels],
            "CUDA packed transformer input",
        )?;
        let expected_prompt = checked_product(
            &[batch, prompt.sequence_length, config.joint_attention_dim],
            "CUDA transformer prompt input",
        )?;
        let expected_mask = checked_product(
            &[batch, prompt.sequence_length],
            "CUDA transformer prompt mask",
        )?;
        if batch == 0
            || image_sequence == 0
            || packed_latents.len() != expected_latents
            || prompt.embeddings.len() != expected_prompt
            || prompt.hidden_size != config.joint_attention_dim
            || prompt.attention_mask.len() != expected_mask
            || prompt.retained_lengths.len() != batch
            || timestep.len() != batch
            || timestep.iter().any(|value| !value.is_finite())
            || prompt.attention_mask.iter().any(|value| *value > 1)
            || prompt
                .attention_mask
                .chunks_exact(prompt.sequence_length)
                .any(|row| !row.iter().any(|value| *value != 0))
        {
            return Err(ImageError::UnsupportedShape(
                "Qwen Image CUDA transformer input geometry or mask is invalid".to_string(),
            ));
        }

        let inner = config.inner_dim()?;
        let image_rows = checked_product(&[batch, image_sequence], "CUDA image rows")?;
        let text_rows = checked_product(&[batch, prompt.sequence_length], "CUDA text rows")?;
        let joint_sequence = image_sequence
            .checked_add(prompt.sequence_length)
            .ok_or_else(|| {
                ImageError::UnsupportedShape("CUDA joint sequence overflowed".to_string())
            })?;
        let joint_rows = checked_product(&[batch, joint_sequence], "CUDA joint rows")?;
        let scratch_bytes = u64::try_from(joint_rows)
            .ok()
            .and_then(|rows| rows.checked_mul(inner as u64))
            .and_then(|values| values.checked_mul(TRANSIENT_ACTIVATION_MULTIPLIER))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                ImageError::Admission("CUDA denoiser scratch estimate overflowed".to_string())
            })?;
        let _scratch_lease = self
            .resources
            .allocation_arena()
            .reserve(GpuAllocationClass::DenoiserTransientScratch, scratch_bytes)
            .map_err(map_cuda_admission_error)?;

        let device = &self.device;
        let packed_latents = device
            .upload_f32(packed_latents)
            .map_err(map_cuda_execution_error)?;
        let prompt_values = device
            .upload_f32(&prompt.embeddings)
            .map_err(map_cuda_execution_error)?;
        let normalized_prompt = device
            .rmsnorm_device(
                &prompt_values,
                self.auxiliary("txt_norm.weight")?,
                text_rows,
                config.joint_attention_dim,
                1e-6,
            )
            .map_err(map_cuda_execution_error)?;
        let mut image_states = self.linear(
            "img_in",
            &packed_latents,
            image_rows,
            config.in_channels,
            inner,
        )?;
        let mut text_states = self.linear(
            "txt_in",
            &normalized_prompt,
            text_rows,
            config.joint_attention_dim,
            inner,
        )?;

        let activated_timestep = self.activated_timestep(timestep, inner)?;
        let zero_timestep = config
            .zero_cond_t
            .then(|| self.activated_timestep(&vec![0.0; batch], inner))
            .transpose()?;

        let rope = qwen_image_rotary_embeddings_for_shapes(
            image_shapes,
            prompt.sequence_length,
            &config.axes_dims_rope,
        )?;
        let image_cos = device
            .upload_f32(&rope.image_cos)
            .map_err(map_cuda_execution_error)?;
        let image_sin = device
            .upload_f32(&rope.image_sin)
            .map_err(map_cuda_execution_error)?;
        let text_cos = device
            .upload_f32(&rope.text_cos)
            .map_err(map_cuda_execution_error)?;
        let text_sin = device
            .upload_f32(&rope.text_sin)
            .map_err(map_cuda_execution_error)?;
        let joint_mask = joint_attention_mask(
            &prompt.attention_mask,
            batch,
            prompt.sequence_length,
            image_sequence,
        )?;
        let joint_mask = device
            .upload_bytes(&joint_mask)
            .map_err(map_cuda_execution_error)?;
        let output_sequence = checked_product(
            image_shapes.first().ok_or_else(|| {
                ImageError::UnsupportedShape(
                    "CUDA transformer requires an output image shape".to_string(),
                )
            })?,
            "CUDA output image sequence",
        )?;
        let image_modulation_selectors = if config.zero_cond_t {
            let mut selectors = Vec::with_capacity(image_rows);
            for _ in 0..batch {
                selectors.extend(std::iter::repeat(0u8).take(output_sequence));
                selectors.extend(std::iter::repeat(1u8).take(image_sequence - output_sequence));
            }
            Some(
                device
                    .upload_bytes(&selectors)
                    .map_err(map_cuda_execution_error)?,
            )
        } else {
            None
        };

        let modulation = inner.checked_mul(6).ok_or_else(|| {
            ImageError::UnsupportedShape("CUDA modulation dimension overflowed".to_string())
        })?;
        let feed_forward = inner.checked_mul(4).ok_or_else(|| {
            ImageError::UnsupportedShape("CUDA feed-forward dimension overflowed".to_string())
        })?;

        for layer in 0..config.num_layers {
            checkpoint(layer)?;
            let prefix = format!("transformer_blocks.{layer}");
            let output_image_modulation = self.linear(
                &format!("{prefix}.img_mod.1"),
                &activated_timestep,
                batch,
                inner,
                modulation,
            )?;
            let image_modulation = if let Some(zero_timestep) = &zero_timestep {
                let source_image_modulation = self.linear(
                    &format!("{prefix}.img_mod.1"),
                    zero_timestep,
                    batch,
                    inner,
                    modulation,
                )?;
                device
                    .image_join_streams_device(
                        &output_image_modulation,
                        &source_image_modulation,
                        batch,
                        1,
                        1,
                        modulation,
                    )
                    .map_err(map_cuda_execution_error)?
            } else {
                output_image_modulation
            };
            let text_modulation = self.linear(
                &format!("{prefix}.txt_mod.1"),
                &activated_timestep,
                batch,
                inner,
                modulation,
            )?;

            let mut image_modulated = device
                .image_layer_norm_device(&image_states, image_rows, inner, 1e-6)
                .map_err(map_cuda_execution_error)?;
            self.affine_image_rows(
                &mut image_modulated,
                &image_modulation,
                image_modulation_selectors.as_ref(),
                batch,
                image_sequence,
                inner,
                modulation,
                inner,
                0,
                1.0,
            )?;
            let mut text_modulated = device
                .image_layer_norm_device(&text_states, text_rows, inner, 1e-6)
                .map_err(map_cuda_execution_error)?;
            device
                .image_affine_rows_assign_device(
                    &mut text_modulated,
                    &text_modulation,
                    batch,
                    prompt.sequence_length,
                    inner,
                    modulation,
                    inner,
                    0,
                    1.0,
                )
                .map_err(map_cuda_execution_error)?;

            let mut image_query = self.linear(
                &format!("{prefix}.attn.to_q"),
                &image_modulated,
                image_rows,
                inner,
                inner,
            )?;
            let mut image_key = self.linear(
                &format!("{prefix}.attn.to_k"),
                &image_modulated,
                image_rows,
                inner,
                inner,
            )?;
            let image_value = self.linear(
                &format!("{prefix}.attn.to_v"),
                &image_modulated,
                image_rows,
                inner,
                inner,
            )?;
            let mut text_query = self.linear(
                &format!("{prefix}.attn.add_q_proj"),
                &text_modulated,
                text_rows,
                inner,
                inner,
            )?;
            let mut text_key = self.linear(
                &format!("{prefix}.attn.add_k_proj"),
                &text_modulated,
                text_rows,
                inner,
                inner,
            )?;
            let text_value = self.linear(
                &format!("{prefix}.attn.add_v_proj"),
                &text_modulated,
                text_rows,
                inner,
                inner,
            )?;
            drop((image_modulated, text_modulated));

            image_query = device
                .rmsnorm_device(
                    &image_query,
                    self.auxiliary(&format!("{prefix}.attn.norm_q.weight"))?,
                    image_rows * config.num_attention_heads,
                    config.attention_head_dim,
                    1e-6,
                )
                .map_err(map_cuda_execution_error)?;
            image_key = device
                .rmsnorm_device(
                    &image_key,
                    self.auxiliary(&format!("{prefix}.attn.norm_k.weight"))?,
                    image_rows * config.num_attention_heads,
                    config.attention_head_dim,
                    1e-6,
                )
                .map_err(map_cuda_execution_error)?;
            text_query = device
                .rmsnorm_device(
                    &text_query,
                    self.auxiliary(&format!("{prefix}.attn.norm_added_q.weight"))?,
                    text_rows * config.num_attention_heads,
                    config.attention_head_dim,
                    1e-6,
                )
                .map_err(map_cuda_execution_error)?;
            text_key = device
                .rmsnorm_device(
                    &text_key,
                    self.auxiliary(&format!("{prefix}.attn.norm_added_k.weight"))?,
                    text_rows * config.num_attention_heads,
                    config.attention_head_dim,
                    1e-6,
                )
                .map_err(map_cuda_execution_error)?;

            for values in [&mut image_query, &mut image_key] {
                device
                    .image_complex_rope_assign_device(
                        values,
                        &image_cos,
                        &image_sin,
                        batch,
                        image_sequence,
                        config.num_attention_heads,
                        config.attention_head_dim,
                    )
                    .map_err(map_cuda_execution_error)?;
            }
            for values in [&mut text_query, &mut text_key] {
                device
                    .image_complex_rope_assign_device(
                        values,
                        &text_cos,
                        &text_sin,
                        batch,
                        prompt.sequence_length,
                        config.num_attention_heads,
                        config.attention_head_dim,
                    )
                    .map_err(map_cuda_execution_error)?;
            }

            let joint_query = device
                .image_join_streams_device(
                    &text_query,
                    &image_query,
                    batch,
                    prompt.sequence_length,
                    image_sequence,
                    inner,
                )
                .map_err(map_cuda_execution_error)?;
            let joint_key = device
                .image_join_streams_device(
                    &text_key,
                    &image_key,
                    batch,
                    prompt.sequence_length,
                    image_sequence,
                    inner,
                )
                .map_err(map_cuda_execution_error)?;
            let joint_value = device
                .image_join_streams_device(
                    &text_value,
                    &image_value,
                    batch,
                    prompt.sequence_length,
                    image_sequence,
                    inner,
                )
                .map_err(map_cuda_execution_error)?;
            drop((
                image_query,
                image_key,
                image_value,
                text_query,
                text_key,
                text_value,
            ));

            let joint_attention = device
                .image_attention_device(
                    &joint_query,
                    &joint_key,
                    &joint_value,
                    &joint_mask,
                    batch,
                    joint_sequence,
                    joint_sequence,
                    config.num_attention_heads,
                    config.attention_head_dim,
                )
                .map_err(map_cuda_execution_error)?;
            let (text_attention, image_attention) = device
                .image_split_streams_device(
                    &joint_attention,
                    batch,
                    prompt.sequence_length,
                    image_sequence,
                    inner,
                )
                .map_err(map_cuda_execution_error)?;
            drop((joint_query, joint_key, joint_value, joint_attention));

            let image_attention = self.linear(
                &format!("{prefix}.attn.to_out.0"),
                &image_attention,
                image_rows,
                inner,
                inner,
            )?;
            self.gated_image_residual(
                &mut image_states,
                &image_attention,
                &image_modulation,
                image_modulation_selectors.as_ref(),
                batch,
                image_sequence,
                inner,
                modulation,
                inner * 2,
            )?;
            let text_attention = self.linear(
                &format!("{prefix}.attn.to_add_out"),
                &text_attention,
                text_rows,
                inner,
                inner,
            )?;
            device
                .image_gated_residual_assign_device(
                    &mut text_states,
                    &text_attention,
                    &text_modulation,
                    batch,
                    prompt.sequence_length,
                    inner,
                    modulation,
                    inner * 2,
                )
                .map_err(map_cuda_execution_error)?;
            drop((image_attention, text_attention));

            let mut image_mlp = device
                .image_layer_norm_device(&image_states, image_rows, inner, 1e-6)
                .map_err(map_cuda_execution_error)?;
            self.affine_image_rows(
                &mut image_mlp,
                &image_modulation,
                image_modulation_selectors.as_ref(),
                batch,
                image_sequence,
                inner,
                modulation,
                inner * 4,
                inner * 3,
                1.0,
            )?;
            image_mlp = self.linear(
                &format!("{prefix}.img_mlp.net.0.proj"),
                &image_mlp,
                image_rows,
                inner,
                feed_forward,
            )?;
            device
                .image_gelu_tanh_assign_device(&mut image_mlp)
                .map_err(map_cuda_execution_error)?;
            image_mlp = self.linear(
                &format!("{prefix}.img_mlp.net.2"),
                &image_mlp,
                image_rows,
                feed_forward,
                inner,
            )?;
            self.gated_image_residual(
                &mut image_states,
                &image_mlp,
                &image_modulation,
                image_modulation_selectors.as_ref(),
                batch,
                image_sequence,
                inner,
                modulation,
                inner * 5,
            )?;
            drop((image_mlp, image_modulation));

            let mut text_mlp = device
                .image_layer_norm_device(&text_states, text_rows, inner, 1e-6)
                .map_err(map_cuda_execution_error)?;
            device
                .image_affine_rows_assign_device(
                    &mut text_mlp,
                    &text_modulation,
                    batch,
                    prompt.sequence_length,
                    inner,
                    modulation,
                    inner * 4,
                    inner * 3,
                    1.0,
                )
                .map_err(map_cuda_execution_error)?;
            text_mlp = self.linear(
                &format!("{prefix}.txt_mlp.net.0.proj"),
                &text_mlp,
                text_rows,
                inner,
                feed_forward,
            )?;
            device
                .image_gelu_tanh_assign_device(&mut text_mlp)
                .map_err(map_cuda_execution_error)?;
            text_mlp = self.linear(
                &format!("{prefix}.txt_mlp.net.2"),
                &text_mlp,
                text_rows,
                feed_forward,
                inner,
            )?;
            device
                .image_gated_residual_assign_device(
                    &mut text_states,
                    &text_mlp,
                    &text_modulation,
                    batch,
                    prompt.sequence_length,
                    inner,
                    modulation,
                    inner * 5,
                )
                .map_err(map_cuda_execution_error)?;
        }
        checkpoint(config.num_layers)?;

        let norm_modulation = self.linear(
            "norm_out.linear",
            &activated_timestep,
            batch,
            inner,
            inner * 2,
        )?;
        let mut image_states = device
            .image_layer_norm_device(&image_states, image_rows, inner, 1e-6)
            .map_err(map_cuda_execution_error)?;
        device
            .image_affine_rows_assign_device(
                &mut image_states,
                &norm_modulation,
                batch,
                image_sequence,
                inner,
                inner * 2,
                0,
                inner,
                1.0,
            )
            .map_err(map_cuda_execution_error)?;
        let output_features = checked_product(
            &[config.out_channels, config.patch_size, config.patch_size],
            "CUDA transformer output features",
        )?;
        let output = self.linear(
            "proj_out",
            &image_states,
            image_rows,
            inner,
            output_features,
        )?;
        let output = device
            .download_f32(&output)
            .map_err(map_cuda_execution_error)?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(ImageError::Numerical {
                component: "cuda_transformer",
                step: 0,
            });
        }
        Ok(output)
    }

    fn activated_timestep(
        &self,
        timestep: &[f32],
        inner: usize,
    ) -> Result<CudaF32Buffer, ImageError> {
        let rows = timestep.len();
        let projection = qwen_timestep_projection(timestep)?;
        let projection = self
            .device
            .upload_f32(&projection)
            .map_err(map_cuda_execution_error)?;
        let stage_1 = self.linear(
            "time_text_embed.timestep_embedder.linear_1",
            &projection,
            rows,
            256,
            inner,
        )?;
        let stage_1 = self
            .device
            .silu_device(&stage_1)
            .map_err(map_cuda_execution_error)?;
        let stage_2 = self.linear(
            "time_text_embed.timestep_embedder.linear_2",
            &stage_1,
            rows,
            inner,
            inner,
        )?;
        self.device
            .silu_device(&stage_2)
            .map_err(map_cuda_execution_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn affine_image_rows(
        &self,
        values: &mut CudaF32Buffer,
        conditioning: &CudaF32Buffer,
        row_selectors: Option<&CudaBytes>,
        batch: usize,
        sequence: usize,
        width: usize,
        conditioning_stride: usize,
        scale_offset: usize,
        shift_offset: usize,
        scale_bias: f32,
    ) -> Result<(), ImageError> {
        let result = if let Some(row_selectors) = row_selectors {
            self.device.image_affine_rows_indexed_assign_device(
                values,
                conditioning,
                row_selectors,
                batch,
                sequence,
                width,
                conditioning_stride,
                2,
                scale_offset,
                shift_offset,
                scale_bias,
            )
        } else {
            self.device.image_affine_rows_assign_device(
                values,
                conditioning,
                batch,
                sequence,
                width,
                conditioning_stride,
                scale_offset,
                shift_offset,
                scale_bias,
            )
        };
        result.map_err(map_cuda_execution_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_image_residual(
        &self,
        states: &mut CudaF32Buffer,
        update: &CudaF32Buffer,
        conditioning: &CudaF32Buffer,
        row_selectors: Option<&CudaBytes>,
        batch: usize,
        sequence: usize,
        width: usize,
        conditioning_stride: usize,
        gate_offset: usize,
    ) -> Result<(), ImageError> {
        let result = if let Some(row_selectors) = row_selectors {
            self.device.image_gated_residual_indexed_assign_device(
                states,
                update,
                conditioning,
                row_selectors,
                batch,
                sequence,
                width,
                conditioning_stride,
                2,
                gate_offset,
            )
        } else {
            self.device.image_gated_residual_assign_device(
                states,
                update,
                conditioning,
                batch,
                sequence,
                width,
                conditioning_stride,
                gate_offset,
            )
        };
        result.map_err(map_cuda_execution_error)
    }

    fn auxiliary(&self, name: &str) -> Result<&CudaF32Buffer, ImageError> {
        self.auxiliary
            .get(name)
            .map(GpuF32Tensor::buffer)
            .ok_or_else(|| {
                ImageError::UnsupportedTensor(format!(
                    "missing CUDA Qwen Image auxiliary tensor `{name}`"
                ))
            })
    }

    fn linear(
        &self,
        prefix: &str,
        input: &CudaF32Buffer,
        rows: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<CudaF32Buffer, ImageError> {
        let name = format!("{prefix}.weight");
        let matrix = self.matrices.get(&name).ok_or_else(|| {
            ImageError::UnsupportedTensor(format!("missing CUDA Qwen Image matrix `{name}`"))
        })?;
        if matrix.input_features != input_features || matrix.output_features != output_features {
            return Err(ImageError::UnsupportedShape(format!(
                "CUDA matrix `{name}` has [{}, {}] input/output features, expected [{input_features}, {output_features}]",
                matrix.input_features, matrix.output_features
            )));
        }
        let mut output = matrix.forward(&self.device, input, rows)?;
        self.device
            .image_bias_add_assign_device(
                &mut output,
                self.auxiliary(&format!("{prefix}.bias"))?,
                rows,
                output_features,
            )
            .map_err(map_cuda_execution_error)?;
        if let Some(layer) = self
            .adapter
            .as_ref()
            .and_then(|adapter| adapter.layers.get(prefix))
        {
            if layer.input_features != input_features || layer.output_features != output_features {
                return Err(ImageError::UnsupportedShape(format!(
                    "CUDA Lightning layer `{prefix}` has [{}, {}] input/output features, expected [{input_features}, {output_features}]",
                    layer.input_features, layer.output_features
                )));
            }
            let hidden = self
                .device
                .matmul_resident_rhs_device(
                    input,
                    rows,
                    input_features,
                    layer.down.buffer(),
                    layer.rank,
                )
                .map_err(map_cuda_execution_error)?;
            let delta = self
                .device
                .matmul_resident_rhs_device(
                    &hidden,
                    rows,
                    layer.rank,
                    layer.up.buffer(),
                    output_features,
                )
                .map_err(map_cuda_execution_error)?;
            self.device
                .scaled_row_add_assign_device(&mut output, &delta, 0, layer.scale)
                .map_err(map_cuda_execution_error)?;
        }
        Ok(output)
    }
}

fn configure_resource_budget(
    device: &CudaDevice,
    resources: &GpuResourceManager,
) -> Result<(), ImageError> {
    let arena = resources.allocation_arena();
    if arena.snapshot().budget_bytes.is_some() {
        return Ok(());
    }
    let (free_bytes, total_bytes) = device.memory_info().map_err(map_cuda_load_error)?;
    let config = resources.config();
    let fraction_limit = ((total_bytes as f64) * f64::from(config.memory_fraction)).floor() as u64;
    let budget = free_bytes
        .min(fraction_limit)
        .saturating_sub(config.reserved_bytes());
    arena
        .configure_budget(budget)
        .map_err(map_cuda_admission_error)
}

fn estimate_resident_weight_bytes(file: &GgufFile) -> Result<u64, ImageError> {
    file.tensor_infos()
        .iter()
        .filter(|info| info.name != QWEN_IMAGE_EDIT_TIMESTEP_ZERO_MARKER)
        .try_fold(0u64, |total, info| {
            let elements = info.dimensions.iter().try_fold(1u64, |count, dimension| {
                count.checked_mul(*dimension as u64).ok_or_else(|| {
                    ImageError::Admission(format!(
                        "CUDA tensor `{}` element count overflowed",
                        info.name
                    ))
                })
            })?;
            let bytes = match (info.dimensions.len(), info.dtype) {
                (1 | 2, DType::F32 | DType::F16 | DType::BF16) => elements.checked_mul(4),
                (2, DType::Q8_0) => expanded_block_bytes(elements, 32, 36),
                (2, DType::Q4_K) => expanded_block_bytes(elements, 256, 148),
                (2, DType::Q5_K) => expanded_block_bytes(elements, 256, 180),
                (2, DType::Q6_K) => expanded_block_bytes(elements, 256, 214),
                _ => None,
            }
            .ok_or_else(|| {
                ImageError::UnsupportedTensor(format!(
                "cannot estimate CUDA storage for tensor `{}` with dimensions {:?} and dtype {:?}",
                info.name, info.dimensions, info.dtype
            ))
            })?;
            total.checked_add(bytes).ok_or_else(|| {
                ImageError::Admission("CUDA transformer weight estimate overflowed".to_string())
            })
        })
}

fn expanded_block_bytes(elements: u64, block: u64, physical_bytes: u64) -> Option<u64> {
    (elements % block == 0)
        .then(|| (elements / block).checked_mul(physical_bytes))
        .flatten()
}

fn joint_attention_mask(
    text_mask: &[u8],
    batch: usize,
    text_sequence: usize,
    image_sequence: usize,
) -> Result<Vec<u8>, ImageError> {
    let joint_sequence = text_sequence.checked_add(image_sequence).ok_or_else(|| {
        ImageError::UnsupportedShape("CUDA joint mask sequence overflowed".to_string())
    })?;
    let mut output = vec![1u8; checked_product(&[batch, joint_sequence], "CUDA joint mask")?];
    for batch_index in 0..batch {
        let source = batch_index * text_sequence;
        let destination = batch_index * joint_sequence;
        output[destination..destination + text_sequence]
            .copy_from_slice(&text_mask[source..source + text_sequence]);
    }
    Ok(output)
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, ImageError> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| ImageError::UnsupportedShape(format!("{label} overflow")))
    })
}

fn map_cuda_load_error(error: XrtError) -> ImageError {
    ImageError::UnsupportedBackend(format!("CUDA Qwen Image load failed: {error}"))
}

fn map_cuda_admission_error(error: XrtError) -> ImageError {
    ImageError::InsufficientMemory(error.to_string())
}

fn map_cuda_execution_error(error: XrtError) -> ImageError {
    match error {
        XrtError::Shape(message) | XrtError::InvalidTensor(message) => {
            ImageError::UnsupportedShape(message)
        }
        other => ImageError::Execution(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_resident_storage_estimates_include_expanded_scales() {
        assert_eq!(expanded_block_bytes(256, 32, 36), Some(288));
        assert_eq!(expanded_block_bytes(256, 256, 148), Some(148));
        assert_eq!(expanded_block_bytes(256, 256, 180), Some(180));
        assert_eq!(expanded_block_bytes(256, 256, 214), Some(214));
        assert_eq!(expanded_block_bytes(255, 256, 148), None);
    }
}
