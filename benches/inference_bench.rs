use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use half::f16;
use xrt_kernels::cpu::{dequantize_q4_0_row, dequantize_q4_k_row};
use xrt_kernels::{apply_rmsnorm, apply_rotary, matmul, silu_inplace, softmax_inplace};

const VECTOR_DIMS: [usize; 4] = [128, 512, 2048, 4096];
const ROPE_DIMS: [usize; 3] = [128, 512, 2048];
const SEQ_LENS: [usize; 3] = [128, 512, 2048];
const DEQUANT_ELEMENTS: usize = 4096;

fn benchmark_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/rmsnorm");

    for &dim in &VECTOR_DIMS {
        let input = make_f32_data(dim);
        let weight = make_weight_data(dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter_batched_ref(
                || vec![0.0f32; dim],
                |output| {
                    apply_rmsnorm(
                        black_box(&input),
                        black_box(&weight),
                        black_box(1e-5),
                        black_box(output),
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/rope");

    for &dim in &ROPE_DIMS {
        let n_heads = 8usize;
        let head_dim = dim / n_heads;
        let tensor = make_f32_data(dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter_batched_ref(
                || tensor.clone(),
                |tensor| {
                    apply_rotary(
                        black_box(tensor),
                        n_heads,
                        head_dim,
                        black_box(128),
                        head_dim,
                        black_box(10_000.0),
                        black_box(1.0),
                    )
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/softmax");

    for &seq_len in &SEQ_LENS {
        let values = make_f32_data(seq_len);
        group.throughput(Throughput::Elements(seq_len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter_batched_ref(
                || values.clone(),
                |values| softmax_inplace(black_box(values)),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_silu(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/silu");

    for &dim in &VECTOR_DIMS {
        let values = make_f32_data(dim);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter_batched_ref(
                || values.clone(),
                |values| silu_inplace(black_box(values)),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/matmul_f32");
    let sizes = [
        (128usize, 128usize, 128usize),
        (512, 512, 512),
        (1024, 1024, 64),
    ];

    for &(m, k, n) in &sizes {
        let lhs = make_f32_data(m * k);
        let rhs = make_f32_data(k * n);
        let label = format!("{m}x{k}x{n}");
        group.throughput(Throughput::Elements((m * k * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &(m, k, n), |b, _| {
            b.iter_batched_ref(
                || vec![0.0f32; m * n],
                |output| matmul(black_box(&lhs), m, k, black_box(&rhs), n, black_box(output)),
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference/dequantize");
    let q4_0 = make_q4_0_bytes(DEQUANT_ELEMENTS);
    let q4_k = make_q4_k_bytes(DEQUANT_ELEMENTS);

    group.throughput(Throughput::Elements(DEQUANT_ELEMENTS as u64));
    group.bench_function("q4_0_4096", |b| {
        b.iter_batched_ref(
            || vec![0.0f32; DEQUANT_ELEMENTS],
            |output| {
                dequantize_q4_0_row(black_box(&q4_0), black_box(output))
                    .expect("q4_0 dequantization should succeed")
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("q4_k_4096", |b| {
        b.iter_batched_ref(
            || vec![0.0f32; DEQUANT_ELEMENTS],
            |output| {
                dequantize_q4_k_row(black_box(&q4_k), black_box(output))
                    .expect("q4_k dequantization should succeed")
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn make_f32_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index % 251) as f32) - 125.0) / 37.0)
        .collect()
}

fn make_weight_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| 0.5 + (index % 127) as f32 / 127.0)
        .collect()
}

fn make_q4_0_bytes(elements: usize) -> Vec<u8> {
    assert_eq!(elements % 32, 0);
    let blocks = elements / 32;
    let mut bytes = Vec::with_capacity(blocks * 18);

    for block in 0..blocks {
        let scale = 0.125 + block as f32 * 0.0001;
        bytes.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());

        for lane in 0..16 {
            let low = ((block + lane) & 0x0f) as u8;
            let high = ((block + lane + 7) & 0x0f) as u8;
            bytes.push(low | (high << 4));
        }
    }

    bytes
}

fn make_q4_k_bytes(elements: usize) -> Vec<u8> {
    assert_eq!(elements % 256, 0);
    let blocks = elements / 256;
    let mut bytes = Vec::with_capacity(blocks * 144);

    for block in 0..blocks {
        let d = 0.0625 + block as f32 * 0.0001;
        let dmin = 0.015625 + block as f32 * 0.0001;
        bytes.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        bytes.extend_from_slice(&f16::from_f32(dmin).to_bits().to_le_bytes());

        for scale_index in 0..12 {
            bytes.push(((block + scale_index) % 63) as u8);
        }

        for lane in 0..128 {
            let low = ((block + lane) & 0x0f) as u8;
            let high = ((block * 3 + lane) & 0x0f) as u8;
            bytes.push(low | (high << 4));
        }
    }

    bytes
}

criterion_group!(
    inference_benches,
    benchmark_rmsnorm,
    benchmark_rope,
    benchmark_softmax,
    benchmark_silu,
    benchmark_matmul,
    benchmark_dequantize
);
criterion_main!(inference_benches);
