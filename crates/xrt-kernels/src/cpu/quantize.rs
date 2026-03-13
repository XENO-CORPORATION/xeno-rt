use bytemuck::{pod_read_unaligned, Pod, Zeroable};
use half::f16;
use xrt_core::{Result, XrtError};

const QK4_0: usize = 32;
const QK8_0: usize = 32;
const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ4_0 {
    d: f16,
    qs: [u8; QK4_0 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ8_0 {
    d: f16,
    qs: [i8; QK8_0],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ4_K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ5_K {
    d: f16,
    dmin: f16,
    scales: [u8; K_SCALE_SIZE],
    qh: [u8; QK_K / 8],
    qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlockQ6_K {
    ql: [u8; QK_K / 2],
    qh: [u8; QK_K / 4],
    scales: [i8; QK_K / 16],
    d: f16,
}

pub fn dequantize_q8_0(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elements];
    dequantize_q8_0_row(bytes, &mut output)?;
    Ok(output)
}

pub fn dequantize_q4_0(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elements];
    dequantize_q4_0_row(bytes, &mut output)?;
    Ok(output)
}

pub fn dequantize_q4_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elements];
    dequantize_q4_k_row(bytes, &mut output)?;
    Ok(output)
}

pub fn dequantize_q5_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elements];
    dequantize_q5_k_row(bytes, &mut output)?;
    Ok(output)
}

pub fn dequantize_q6_k(bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; elements];
    dequantize_q6_k_row(bytes, &mut output)?;
    Ok(output)
}

fn get_scale_min_k4(index: usize, packed: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 0x3f, packed[index + 4] & 0x3f)
    } else {
        (
            ((packed[index + 4] & 0x0f) | ((packed[index - 4] >> 6) << 4)) & 0x3f,
            ((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)) & 0x3f,
        )
    }
}

pub fn dequantize_q8_0_row(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    if output.len() % QK8_0 != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "q8_0 row length {} is not divisible by {QK8_0}",
            output.len()
        )));
    }
    let expected = (output.len() / QK8_0) * std::mem::size_of::<BlockQ8_0>();
    if bytes.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "q8_0 row bytes {} do not match expected size {expected}",
            bytes.len()
        )));
    }

    for (block_index, chunk) in bytes
        .chunks_exact(std::mem::size_of::<BlockQ8_0>())
        .enumerate()
    {
        let block: BlockQ8_0 = pod_read_unaligned(chunk);
        let scale = block.d.to_f32();
        let dst = &mut output[block_index * QK8_0..(block_index + 1) * QK8_0];
        for (dst, quant) in dst.iter_mut().zip(block.qs.iter()) {
            *dst = scale * *quant as f32;
        }
    }
    Ok(())
}

pub fn dequantize_q4_0_row(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    if output.len() % QK4_0 != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "q4_0 row length {} is not divisible by {QK4_0}",
            output.len()
        )));
    }
    let expected = (output.len() / QK4_0) * std::mem::size_of::<BlockQ4_0>();
    if bytes.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "q4_0 row bytes {} do not match expected size {expected}",
            bytes.len()
        )));
    }

    for (block_index, chunk) in bytes
        .chunks_exact(std::mem::size_of::<BlockQ4_0>())
        .enumerate()
    {
        let block: BlockQ4_0 = pod_read_unaligned(chunk);
        let scale = block.d.to_f32();
        let dst = &mut output[block_index * QK4_0..(block_index + 1) * QK4_0];
        for value_index in 0..QK4_0 / 2 {
            let packed = block.qs[value_index];
            let low = (packed & 0x0f) as i32 - 8;
            let high = ((packed >> 4) & 0x0f) as i32 - 8;
            dst[value_index] = scale * low as f32;
            dst[value_index + QK4_0 / 2] = scale * high as f32;
        }
    }
    Ok(())
}

pub fn dequantize_q4_k_row(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    if output.len() % QK_K != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "q4_k row length {} is not divisible by {QK_K}",
            output.len()
        )));
    }
    let expected = (output.len() / QK_K) * std::mem::size_of::<BlockQ4_K>();
    if bytes.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "q4_k row bytes {} do not match expected size {expected}",
            bytes.len()
        )));
    }

    for (block_index, chunk) in bytes
        .chunks_exact(std::mem::size_of::<BlockQ4_K>())
        .enumerate()
    {
        let block: BlockQ4_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let dst = &mut output[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 64 {
            let q = &block.qs[group * 32..(group + 1) * 32];
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            let base = group * 64;

            for lane in 0..32 {
                dst[base + lane] = d1 * (q[lane] & 0x0f) as f32 - min1;
                dst[base + 32 + lane] = d2 * (q[lane] >> 4) as f32 - min2;
            }
        }
    }

    Ok(())
}

pub fn dequantize_q5_k_row(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    if output.len() % QK_K != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "q5_k row length {} is not divisible by {QK_K}",
            output.len()
        )));
    }
    let expected = (output.len() / QK_K) * std::mem::size_of::<BlockQ5_K>();
    if bytes.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "q5_k row bytes {} do not match expected size {expected}",
            bytes.len()
        )));
    }

    for (block_index, chunk) in bytes
        .chunks_exact(std::mem::size_of::<BlockQ5_K>())
        .enumerate()
    {
        let block: BlockQ5_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let dst = &mut output[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 64 {
            let ql = &block.qs[group * 32..(group + 1) * 32];
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            let high_mask_low = 1u8 << (group * 2);
            let high_mask_high = 1u8 << (group * 2 + 1);
            let base = group * 64;

            for lane in 0..32 {
                let low = (ql[lane] & 0x0f) as i32
                    + if (block.qh[lane] & high_mask_low) != 0 {
                        16
                    } else {
                        0
                    };
                let high = (ql[lane] >> 4) as i32
                    + if (block.qh[lane] & high_mask_high) != 0 {
                        16
                    } else {
                        0
                    };
                dst[base + lane] = d1 * low as f32 - min1;
                dst[base + 32 + lane] = d2 * high as f32 - min2;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Fused quantized dot products — compute dot(dequant(row), input) in one pass
// without allocating a scratch buffer for the dequantized row.
// ---------------------------------------------------------------------------

pub fn dot_q8_0(row: &[u8], input: &[f32]) -> f32 {
    let block_size = std::mem::size_of::<BlockQ8_0>();
    let mut sum = 0.0f32;
    for (block_index, chunk) in row.chunks_exact(block_size).enumerate() {
        let block: BlockQ8_0 = pod_read_unaligned(chunk);
        let scale = block.d.to_f32();
        let inp = &input[block_index * QK8_0..(block_index + 1) * QK8_0];
        let mut block_sum = 0.0f32;
        for (quant, &x) in block.qs.iter().zip(inp.iter()) {
            block_sum += *quant as f32 * x;
        }
        sum += scale * block_sum;
    }
    sum
}

pub fn dot_q4_0(row: &[u8], input: &[f32]) -> f32 {
    let block_size = std::mem::size_of::<BlockQ4_0>();
    let mut sum = 0.0f32;
    for (block_index, chunk) in row.chunks_exact(block_size).enumerate() {
        let block: BlockQ4_0 = pod_read_unaligned(chunk);
        let scale = block.d.to_f32();
        let inp = &input[block_index * QK4_0..(block_index + 1) * QK4_0];
        let mut block_sum = 0.0f32;
        for value_index in 0..QK4_0 / 2 {
            let packed = block.qs[value_index];
            let low = (packed & 0x0f) as i32 - 8;
            let high = ((packed >> 4) & 0x0f) as i32 - 8;
            block_sum += low as f32 * inp[value_index];
            block_sum += high as f32 * inp[value_index + QK4_0 / 2];
        }
        sum += scale * block_sum;
    }
    sum
}

pub fn dot_q4_k(row: &[u8], input: &[f32]) -> f32 {
    let block_size = std::mem::size_of::<BlockQ4_K>();
    let mut sum = 0.0f32;
    for (block_index, chunk) in row.chunks_exact(block_size).enumerate() {
        let block: BlockQ4_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let inp = &input[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 64 {
            let q = &block.qs[group * 32..(group + 1) * 32];
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            let base = group * 64;

            for lane in 0..32 {
                sum += (d1 * (q[lane] & 0x0f) as f32 - min1) * inp[base + lane];
                sum += (d2 * (q[lane] >> 4) as f32 - min2) * inp[base + 32 + lane];
            }
        }
    }
    sum
}

pub fn dot_q5_k(row: &[u8], input: &[f32]) -> f32 {
    let block_size = std::mem::size_of::<BlockQ5_K>();
    let mut sum = 0.0f32;
    for (block_index, chunk) in row.chunks_exact(block_size).enumerate() {
        let block: BlockQ5_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let dmin = block.dmin.to_f32();
        let inp = &input[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 64 {
            let ql = &block.qs[group * 32..(group + 1) * 32];
            let (sc1, m1) = get_scale_min_k4(group * 2, &block.scales);
            let (sc2, m2) = get_scale_min_k4(group * 2 + 1, &block.scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let min1 = dmin * m1 as f32;
            let min2 = dmin * m2 as f32;
            let high_mask_low = 1u8 << (group * 2);
            let high_mask_high = 1u8 << (group * 2 + 1);
            let base = group * 64;

            for lane in 0..32 {
                let low = (ql[lane] & 0x0f) as i32
                    + if (block.qh[lane] & high_mask_low) != 0 { 16 } else { 0 };
                let high = (ql[lane] >> 4) as i32
                    + if (block.qh[lane] & high_mask_high) != 0 { 16 } else { 0 };
                sum += (d1 * low as f32 - min1) * inp[base + lane];
                sum += (d2 * high as f32 - min2) * inp[base + 32 + lane];
            }
        }
    }
    sum
}

pub fn dot_q6_k(row: &[u8], input: &[f32]) -> f32 {
    let block_size = std::mem::size_of::<BlockQ6_K>();
    let mut sum = 0.0f32;
    for (block_index, chunk) in row.chunks_exact(block_size).enumerate() {
        let block: BlockQ6_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let inp = &input[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 128 {
            let ql = &block.ql[group * 64..(group + 1) * 64];
            let qh = &block.qh[group * 32..(group + 1) * 32];
            let scales = &block.scales[group * 8..(group + 1) * 8];
            let gi = &inp[group * 128..(group + 1) * 128];

            for lane in 0..32 {
                let scale_index = lane / 16;
                let q1 = ((ql[lane] & 0x0f) | ((qh[lane] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) as i32 - 32;

                sum += d * scales[scale_index] as f32 * q1 as f32 * gi[lane];
                sum += d * scales[scale_index + 2] as f32 * q2 as f32 * gi[lane + 32];
                sum += d * scales[scale_index + 4] as f32 * q3 as f32 * gi[lane + 64];
                sum += d * scales[scale_index + 6] as f32 * q4 as f32 * gi[lane + 96];
            }
        }
    }
    sum
}

pub fn dequantize_q6_k_row(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    if output.len() % QK_K != 0 {
        return Err(XrtError::InvalidTensor(format!(
            "q6_k row length {} is not divisible by {QK_K}",
            output.len()
        )));
    }
    let expected = (output.len() / QK_K) * std::mem::size_of::<BlockQ6_K>();
    if bytes.len() != expected {
        return Err(XrtError::InvalidTensor(format!(
            "q6_k row bytes {} do not match expected size {expected}",
            bytes.len()
        )));
    }

    for (block_index, chunk) in bytes
        .chunks_exact(std::mem::size_of::<BlockQ6_K>())
        .enumerate()
    {
        let block: BlockQ6_K = pod_read_unaligned(chunk);
        let d = block.d.to_f32();
        let dst = &mut output[block_index * QK_K..(block_index + 1) * QK_K];

        for group in 0..QK_K / 128 {
            let ql = &block.ql[group * 64..(group + 1) * 64];
            let qh = &block.qh[group * 32..(group + 1) * 32];
            let scales = &block.scales[group * 8..(group + 1) * 8];
            let group_dst = &mut dst[group * 128..(group + 1) * 128];

            for lane in 0..32 {
                let scale_index = lane / 16;
                let q1 = ((ql[lane] & 0x0f) | ((qh[lane] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 0x03) << 4)) as i32 - 32;

                group_dst[lane] = d * scales[scale_index] as f32 * q1 as f32;
                group_dst[lane + 32] = d * scales[scale_index + 2] as f32 * q2 as f32;
                group_dst[lane + 64] = d * scales[scale_index + 4] as f32 * q3 as f32;
                group_dst[lane + 96] = d * scales[scale_index + 6] as f32 * q4 as f32;
            }
        }
    }

    Ok(())
}
