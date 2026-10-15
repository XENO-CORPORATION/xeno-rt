//! Warm encoded dense projection; excludes weight upload and model orchestration.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use xrt_cuda::{CudaDevice, GpuDenseWorkspace};
fn rounded(value: f32) -> f32 {
    let b = value.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff0000)
}
fn dense(c: &mut Criterion) {
    let device = CudaDevice::new(0).expect("requires NVIDIA driver and CUDA-12 cuBLAS");
    let baseline = device.allocation_stats().current_bytes;
    let mut group = c.benchmark_group("warm_dense_bf16_activation");
    group.sample_size(20);
    for (rows, cols) in [(8192usize, 2048usize), (2048, 8192)] {
        let bytes: Vec<u8> = (0..rows * cols)
            .flat_map(|i| {
                ((rounded((i as f32 * 0.037).sin() * 0.1).to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let matrix = device
            .upload_dense_bf16_matrix(&bytes, rows, cols, bytes.len() as u64)
            .unwrap();
        let input: Vec<_> = (0..cols)
            .map(|i| rounded((i as f32 * 0.031).cos()))
            .collect();
        let mut workspace = GpuDenseWorkspace::new(device.clone(), 4 * 1024 * 1024);
        let expected = workspace.matvec_bf16(&matrix, &input).unwrap();
        let actual = workspace.matvec_bf16_activation(&matrix, &input).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 0.016 * b.abs().max(1.0));
        }
        group.bench_with_input(
            BenchmarkId::new("custom_fp32_then_bf16", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut output = workspace.matvec_bf16(&matrix, black_box(&input)).unwrap();
                    for v in &mut output {
                        *v = rounded(*v);
                    }
                    black_box(output)
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("cublas_bf16", format!("{rows}x{cols}")),
            &(),
            |b, _| {
                b.iter(|| {
                    black_box(
                        workspace
                            .matvec_bf16_activation(&matrix, black_box(&input))
                            .unwrap(),
                    )
                })
            },
        );
        workspace.clear();
    }
    group.finish();
    device.synchronize().unwrap();
    assert_eq!(device.allocation_stats().current_bytes, baseline);
}
criterion_group!(benches, dense);
criterion_main!(benches);
