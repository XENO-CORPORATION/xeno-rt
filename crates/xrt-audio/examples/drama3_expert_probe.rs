//! Verify a real extracted FP8 projection against independent PyTorch outputs.
use xrt_safetensors::{SafeTensorDType, SafeTensorStore};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let root = args.first().ok_or("fixture directory required")?;
    let tensors = SafeTensorStore::open(root)?;
    let weights = tensors.require_tensor("weight")?;
    if weights.info.dtype != SafeTensorDType::F8E4M3 || weights.info.shape.len() != 2 {
        return Err("invalid FP8 projection fixture".into());
    }
    let read = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let v = tensors.require_tensor(name)?;
        if v.info.dtype != SafeTensorDType::F32 {
            return Err("expected F32 fixture tensor".into());
        }
        Ok(v.data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let scales = read("scale")?;
    let input = read("input")?;
    let rows = weights.info.shape[0];
    let cols = weights.info.shape[1];
    let matrix =
        xrt_kernels::cpu::fp8::BlockFp8Matrix::new(weights.data, &scales, rows, cols, 128, 128)?;
    for quantized in [false, true] {
        let expected = read(if quantized {
            "expected_w8a8"
        } else {
            "expected_f32"
        })?;
        let mut actual = vec![0.0; rows];
        if quantized {
            matrix.matvec_w8a8(&input, &mut actual)?;
        } else {
            matrix.matvec(&input, &mut actual)?;
        }
        if actual.len() != expected.len() {
            return Err("fixture output dimensions differ".into());
        }
        let mut max_error = 0.0f32;
        for (a, b) in actual.iter().zip(&expected) {
            max_error = max_error.max((a - b).abs());
            if (a - b).abs() > 1e-3 * b.abs().max(1.0) {
                return Err(format!("CPU trained projection mismatch: {a} vs {b}").into());
            }
        }
        println!(
            "PASS trained FP8 CPU projection w8a8={quantized}, max absolute error {max_error}"
        );
        #[cfg(feature = "cuda")]
        {
            let device = xrt_cuda::CudaDevice::new(0)?;
            let gpu = device.upload_block_fp8(
                weights.data,
                &scales,
                rows,
                cols,
                128,
                128,
                matrix.resident_bytes() as u64,
            )?;
            let actual = device.matvec_block_fp8(&gpu, &input, quantized)?;
            let mut max_error = 0.0f32;
            for (a, b) in actual.iter().zip(&expected) {
                max_error = max_error.max((a - b).abs());
                if (a - b).abs() > 1e-3 * b.abs().max(1.0) {
                    return Err(format!("CUDA trained projection mismatch: {a} vs {b}").into());
                }
            }
            println!(
                "PASS trained FP8 CUDA projection w8a8={quantized}, max absolute error {max_error}"
            );
        }
    }
    Ok(())
}
