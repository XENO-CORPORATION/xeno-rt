//! Warm, synthetic single-token expert execution, not a model throughput claim.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xrt_cuda::{CudaDevice, GpuFp8Cache, PreparedFp8Matrix};

fn bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
}

fn routed(c: &mut Criterion) {
    let device = CudaDevice::new(0).expect("requires NVIDIA driver and NVRTC");
    let baseline = device.allocation_stats().current_bytes;
    let (hidden, middle) = (2048, 512);
    let input: Vec<_> = (0..hidden)
        .map(|i| bf16((i as f32 * 0.031).sin()))
        .collect();
    let sources: Vec<_> = (0..8)
        .map(|expert| {
            [(middle, hidden), (middle, hidden), (hidden, middle)]
                .into_iter()
                .enumerate()
                .map(|(projection, (rows, cols))| {
                    let bytes: Vec<u8> = (0..rows * cols)
                        .map(|i| {
                            ((i * 17 + expert * 13 + projection * 5) % 96) as u8
                                | if i % 3 == 0 { 128 } else { 0 }
                        })
                        .collect();
                    PreparedFp8Matrix::new(
                        bytes,
                        vec![0.001; rows / 128 * (cols / 128)],
                        rows,
                        cols,
                        128,
                        128,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let experts: Vec<_> = sources.iter().map(|s| [&s[0], &s[1], &s[2]]).collect();
    let probabilities = [0.125; 8];
    let mut cache = GpuFp8Cache::new(device.clone(), 32 * 1024 * 1024);
    let unfused = |cache: &mut GpuFp8Cache| {
        let mut result = vec![0.0; hidden];
        for expert in &sources {
            let gate = cache.matvec_prepared(&expert[0], &input, true).unwrap();
            let up = cache.matvec_prepared(&expert[1], &input, true).unwrap();
            let activation: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| {
                    let g = bf16(*g);
                    g / (1.0 + (-g).exp()) * bf16(u)
                })
                .collect();
            let down = cache
                .matvec_prepared(&expert[2], &activation, true)
                .unwrap();
            for (sum, value) in result.iter_mut().zip(down) {
                *sum += value * 0.125;
            }
        }
        result
    };
    let expected = unfused(&mut cache);
    let actual = cache
        .routed_mlp_w8a8_prepared(&experts, &input, &probabilities, 1.0)
        .unwrap();
    for (a, b) in actual.iter().zip(&expected) {
        assert!((a - b).abs() <= 4e-5 * b.abs().max(1.0));
    }
    let mut group = c.benchmark_group("warm_fp8_w8a8_8x2048x512");
    group.sample_size(20);
    group.bench_function("unfused", |b| b.iter(|| black_box(unfused(&mut cache))));
    group.bench_function("fused", |b| {
        b.iter(|| {
            black_box(
                cache
                    .routed_mlp_w8a8_prepared(&experts, &input, &probabilities, 1.0)
                    .unwrap(),
            )
        })
    });
    group.finish();
    cache.clear();
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}

criterion_group!(benches, routed);
criterion_main!(benches);
