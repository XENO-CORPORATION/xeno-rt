#[path = "../tests/common/tokenizer_fixture.rs"]
mod common;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use xrt_gguf::GgufFile;
use xrt_tokenizer::Tokenizer;

const TOKENIZER_LENS: [usize; 3] = [10, 100, 1000];

fn benchmark_bpe_encode(c: &mut Criterion) {
    let (fixture, _) = common::build_bpe_tokenizer_fixture().expect("fixture should be created");
    let gguf = GgufFile::open(fixture.path()).expect("GGUF should parse");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer should load");
    let mut group = c.benchmark_group("tokenizer/bpe_encode");

    for &len in &TOKENIZER_LENS {
        let text = build_bench_text(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                let encoded = tokenizer
                    .encode_with_options(black_box(text.as_str()), false, false)
                    .expect("encoding should succeed");
                black_box(encoded);
            });
        });
    }

    group.finish();
}

fn benchmark_decode(c: &mut Criterion) {
    let (fixture, spec) = common::build_bpe_tokenizer_fixture().expect("fixture should be created");
    let gguf = GgufFile::open(fixture.path()).expect("GGUF should parse");
    let tokenizer = Tokenizer::from_gguf(&gguf).expect("tokenizer should load");
    let mut group = c.benchmark_group("tokenizer/decode");

    for &len in &TOKENIZER_LENS {
        let tokens = build_bench_tokens(len, &spec);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                let decoded = tokenizer
                    .decode(black_box(&tokens), true)
                    .expect("decoding should succeed");
                black_box(decoded);
            });
        });
    }

    group.finish();
}

fn build_bench_text(len: usize) -> String {
    let words = ["hello", "world"];
    let mut text = String::with_capacity(len);
    let mut word_index = 0usize;

    while text.len() < len {
        if !text.is_empty() && text.len() < len {
            text.push(' ');
        }

        for ch in words[word_index % words.len()].chars() {
            if text.len() == len {
                break;
            }
            text.push(ch);
        }

        word_index += 1;
    }

    text
}

fn build_bench_tokens(len: usize, spec: &common::TokenizerFixtureSpec) -> Vec<u32> {
    let pattern = [spec.hello_id, spec.world_id, spec.bang_id];
    let mut tokens = Vec::with_capacity(len);

    while tokens.len() < len {
        tokens.extend(pattern);
    }

    tokens.truncate(len);
    tokens
}

criterion_group!(tokenizer_benches, benchmark_bpe_encode, benchmark_decode);
criterion_main!(tokenizer_benches);
