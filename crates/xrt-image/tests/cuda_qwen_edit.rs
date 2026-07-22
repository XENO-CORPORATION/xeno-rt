#![cfg(feature = "cuda")]

use std::{fs, path::Path, sync::Arc};

use half::f16;
use xrt_core::DType;
use xrt_cuda::CudaDevice;
use xrt_gguf::GgufFile;
use xrt_image::models::qwen_image::{
    expected_transformer_tensors, QwenImageCudaTransformer, QwenImageGgufTransformer,
    QwenImagePromptEmbeddings, QwenImageTransformerConfig,
};
use xrt_kernels::scaled_dot_product_attention;
use xrt_runtime::{GpuResourceConfig, GpuResourceManager};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_ALIGNMENT: usize = 32;

struct TensorFixture {
    name: String,
    dimensions: Vec<usize>,
    dtype: DType,
    data: Vec<u8>,
}

#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn tiny_generation_cuda_stays_in_cpu_parity() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen-image/operators-diffusers-0.39.json"
    ))
    .unwrap();
    let fixture = &fixture["full_transformer"];
    let config = QwenImageTransformerConfig::from_json_bytes(
        serde_json::to_string(&fixture["config"])
            .unwrap()
            .as_bytes(),
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tiny-generation-q8_0.gguf");
    write_tiny_q8_fixture(&path, &config);
    let cpu =
        QwenImageGgufTransformer::from_file(GgufFile::open(&path).unwrap(), config.clone(), "Q8_0")
            .unwrap();
    let cuda = QwenImageCudaTransformer::from_file(
        GgufFile::open(&path).unwrap(),
        config,
        "Q8_0",
        Arc::new(GpuResourceManager::new(GpuResourceConfig::default())),
    )
    .unwrap();
    let packed_latents = (0..16)
        .map(|index| ((index % 9) as f32 - 4.0) * 0.07)
        .collect::<Vec<_>>();
    let prompt = QwenImagePromptEmbeddings {
        embeddings: (0..16)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.05)
            .collect(),
        attention_mask: vec![1, 0],
        retained_lengths: vec![1],
        batch_size: 1,
        sequence_length: 2,
        hidden_size: 8,
    };
    let expected = cpu
        .forward(&packed_latents, &prompt, &[0.125], 1, 2, 2)
        .unwrap();
    let before = cuda.transfer_stats();
    let actual = cuda
        .forward(&packed_latents, &prompt, &[0.125], 1, 2, 2)
        .unwrap();
    let transfers = cuda.transfer_stats().saturating_sub(before);
    assert_cuda_parity(&actual, &expected, 2e-4);
    assert_eq!(transfers.host_to_device_calls, 8);
    assert_eq!(transfers.device_to_host_calls, 1);
}

#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn tiny_zero_conditioned_edit_cuda_matches_cpu() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/qwen-image/operators-diffusers-0.39.json"
    ))
    .unwrap();
    let fixture = &fixture["edit_transformer"];
    let config = QwenImageTransformerConfig::from_json_bytes(
        serde_json::to_string(&fixture["config"])
            .unwrap()
            .as_bytes(),
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tiny-edit-q8_0.gguf");
    write_tiny_q8_fixture(&path, &config);

    let cpu =
        QwenImageGgufTransformer::from_file(GgufFile::open(&path).unwrap(), config.clone(), "Q8_0")
            .unwrap();
    let resources = Arc::new(GpuResourceManager::new(GpuResourceConfig::default()));
    let cuda = QwenImageCudaTransformer::from_file(
        GgufFile::open(&path).unwrap(),
        config,
        "Q8_0",
        resources,
    )
    .unwrap();
    let packed_latents = (0..24)
        .map(|index| ((index % 9) as f32 - 4.0) * 0.07)
        .collect::<Vec<_>>();
    let prompt = QwenImagePromptEmbeddings {
        embeddings: (0..16)
            .map(|index| ((index % 7) as f32 - 3.0) * 0.05)
            .collect(),
        attention_mask: vec![1, 0],
        retained_lengths: vec![1],
        batch_size: 1,
        sequence_length: 2,
        hidden_size: 8,
    };
    let shapes = [[1, 2, 2], [1, 1, 2]];
    let expected = cpu
        .forward_edit_with_control(&packed_latents, &prompt, &[0.125], &shapes, |_| Ok(()))
        .unwrap();
    let before = cuda.transfer_stats();
    let actual = cuda
        .forward_edit_with_control(&packed_latents, &prompt, &[0.125], &shapes, |_| Ok(()))
        .unwrap();
    let transfers = cuda.transfer_stats().saturating_sub(before);

    assert_cuda_parity(&actual, &expected, 2e-4);
    assert_eq!(transfers.host_to_device_calls, 10);
    assert_eq!(transfers.device_to_host_calls, 1);
    assert_eq!(transfers.device_to_host_bytes, actual.len() as u64 * 4);
}

#[test]
#[ignore = "requires a CUDA-capable device and driver"]
fn tiled_attention_above_portable_shared_memory_matches_cpu() {
    const BATCH: usize = 1;
    const QUERY_SEQUENCE: usize = 3;
    const KEY_SEQUENCE: usize = 12_289;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 6;

    assert!(KEY_SEQUENCE * std::mem::size_of::<f32>() > 48 * 1024);
    let query = (0..BATCH * QUERY_SEQUENCE * HEADS * HEAD_DIM)
        .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
        .collect::<Vec<_>>();
    let key = (0..BATCH * KEY_SEQUENCE * HEADS * HEAD_DIM)
        .map(|index| ((index % 29) as f32 - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let value = (0..key.len())
        .map(|index| ((index % 23) as f32 - 11.0) * 0.0234375)
        .collect::<Vec<_>>();
    let mut mask = vec![1u8; BATCH * KEY_SEQUENCE];
    for index in (97..KEY_SEQUENCE).step_by(97) {
        mask[index] = 0;
    }
    let mut expected = vec![0.0f32; query.len()];
    scaled_dot_product_attention(
        &query,
        &key,
        &value,
        BATCH,
        QUERY_SEQUENCE,
        KEY_SEQUENCE,
        HEADS,
        HEAD_DIM,
        Some(&mask),
        &mut expected,
    )
    .unwrap();

    let device = CudaDevice::new(0).unwrap();
    let query_device = device.upload_f32(&query).unwrap();
    let key_device = device.upload_f32(&key).unwrap();
    let value_device = device.upload_f32(&value).unwrap();
    let mask_device = device.upload_bytes(&mask).unwrap();
    let output_device = device
        .image_attention_device(
            &query_device,
            &key_device,
            &value_device,
            &mask_device,
            BATCH,
            QUERY_SEQUENCE,
            KEY_SEQUENCE,
            HEADS,
            HEAD_DIM,
        )
        .unwrap();
    let actual = device.download_f32(&output_device).unwrap();
    assert_cuda_parity(&actual, &expected, 2e-4);
}

fn assert_cuda_parity(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs <= tolerance, "CUDA max_abs={max_abs}");
}

fn write_tiny_q8_fixture(path: &Path, config: &QwenImageTransformerConfig) {
    let mut saw_q8 = false;
    let mut tensors = expected_transformer_tensors(config)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(parameter_index, tensor)| {
            let values = (0..tensor.shape.iter().product::<usize>())
                .map(|flat_index| {
                    ((flat_index % 19) as f32 - 9.0) * 0.004 + (parameter_index + 1) as f32 * 0.0001
                })
                .collect::<Vec<_>>();
            let use_q8 = !saw_q8 && tensor.shape.len() == 2 && tensor.shape[1] % 32 == 0;
            let (dtype, data) = if use_q8 {
                saw_q8 = true;
                (
                    DType::Q8_0,
                    q8_0_tensor_bytes(&values, tensor.shape[0], tensor.shape[1]),
                )
            } else {
                (
                    DType::F32,
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                )
            };
            TensorFixture {
                name: tensor.name,
                dimensions: tensor.shape.into_iter().rev().collect(),
                dtype,
                data,
            }
        })
        .collect::<Vec<_>>();
    assert!(saw_q8);
    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    let mut bytes = Vec::new();
    write_u32(&mut bytes, GGUF_MAGIC);
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, tensors.len() as u64);
    write_u64(&mut bytes, 2);
    write_string(&mut bytes, "general.architecture");
    write_u32(&mut bytes, 8);
    write_string(&mut bytes, "qwen_image");
    write_string(&mut bytes, "general.quantization_version");
    write_u32(&mut bytes, 4);
    write_u32(&mut bytes, 2);

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut next_offset = 0usize;
    for tensor in &tensors {
        let offset = align_up(next_offset, GGUF_ALIGNMENT);
        offsets.push(offset);
        next_offset = offset.checked_add(tensor.data.len()).unwrap();
    }
    for (tensor, offset) in tensors.iter().zip(&offsets) {
        write_string(&mut bytes, &tensor.name);
        write_u32(&mut bytes, tensor.dimensions.len() as u32);
        for dimension in &tensor.dimensions {
            write_u64(&mut bytes, *dimension as u64);
        }
        bytes.extend_from_slice(&tensor.dtype.ggml_type_id().to_le_bytes());
        write_u64(&mut bytes, *offset as u64);
    }
    let data_offset = align_up(bytes.len(), GGUF_ALIGNMENT);
    bytes.resize(data_offset, 0);
    for (tensor, offset) in tensors.iter().zip(offsets) {
        let start = data_offset + offset;
        bytes.resize(start, 0);
        bytes.extend_from_slice(&tensor.data);
    }
    fs::write(path, bytes).unwrap();
}

fn q8_0_tensor_bytes(values: &[f32], rows: usize, columns: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * columns);
    assert_eq!(columns % 32, 0);
    let mut bytes = Vec::with_capacity(rows * (columns / 32) * 34);
    for row in values.chunks_exact(columns) {
        for block in row.chunks_exact(32) {
            let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend(block.iter().map(|value| {
                let quantized = (value / scale).round().clamp(-127.0, 127.0);
                quantized as i8 as u8
            }));
        }
    }
    bytes
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.checked_add(alignment - 1).unwrap() / alignment * alignment
}

fn write_string(bytes: &mut Vec<u8>, value: &str) {
    write_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

fn write_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
