use rand::{rngs::StdRng, Rng, SeedableRng};
use xrt_runtime::{Sampler, SamplerConfig};

fn first_draw(seed: u64) -> f32 {
    StdRng::seed_from_u64(seed).random::<f32>()
}

fn find_seed_in_range(min_inclusive: f32, max_exclusive: f32) -> u64 {
    (0..100_000)
        .find(|seed| {
            let draw = first_draw(*seed);
            draw >= min_inclusive && draw < max_exclusive
        })
        .expect("a deterministic seed should exist in the requested range")
}

fn sample_with_seed(seed: u64, logits: &[f32], history: &[u32], config: SamplerConfig) -> u32 {
    Sampler::new(Some(seed))
        .sample(logits, history, config)
        .expect("sampling should succeed")
}

#[test]
fn temperature_scaling_changes_the_sampled_token() {
    let logits = [2.0, 1.0, 0.0];
    let cold_cfg = SamplerConfig {
        temperature: 0.5,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: None,
    };
    let hot_cfg = SamplerConfig {
        temperature: 2.0,
        ..cold_cfg
    };

    let cold_p0 = (4.0f32.exp()) / (4.0f32.exp() + 2.0f32.exp() + 1.0);
    let hot_p0 = 1.0f32.exp() / (1.0f32.exp() + 0.5f32.exp() + 1.0);
    let hot_p1 = 0.5f32.exp() / (1.0f32.exp() + 0.5f32.exp() + 1.0);
    let seed = find_seed_in_range(hot_p0, hot_p0 + hot_p1);

    assert_eq!(sample_with_seed(seed, &logits, &[], cold_cfg), 0);
    assert_eq!(sample_with_seed(seed, &logits, &[], hot_cfg), 1);
    assert!(first_draw(seed) < cold_p0);
}

#[test]
fn top_k_filters_out_lower_ranked_tokens() {
    let logits = [4.0, 3.0, 2.0, 1.0];
    let full_cfg = SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: None,
    };
    let top_k_cfg = SamplerConfig {
        top_k: 2,
        ..full_cfg
    };

    let p0 = 4.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let p1 = 3.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let p2 = 2.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let top2_p0 = 4.0f32.exp() / (4.0f32.exp() + 3.0f32.exp());
    let seed = find_seed_in_range(p0 + p1, p0 + p1 + p2);

    assert_eq!(sample_with_seed(seed, &logits, &[], full_cfg), 2);
    assert_eq!(sample_with_seed(seed, &logits, &[], top_k_cfg), 1);
    assert!(first_draw(seed) > top2_p0);
}

#[test]
fn top_p_keeps_only_the_nucleus() {
    let logits = [4.0, 3.0, 2.0, 1.0];
    let full_cfg = SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: None,
    };
    let nucleus_cfg = SamplerConfig {
        top_p: 0.85,
        ..full_cfg
    };

    let p0 = 4.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let p1 = 3.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let p2 = 2.0f32.exp() / (4.0f32.exp() + 3.0f32.exp() + 2.0f32.exp() + 1.0f32.exp());
    let nucleus_p0 = 4.0f32.exp() / (4.0f32.exp() + 3.0f32.exp());
    let seed = find_seed_in_range(p0 + p1, p0 + p1 + p2);

    assert_eq!(sample_with_seed(seed, &logits, &[], full_cfg), 2);
    assert_eq!(sample_with_seed(seed, &logits, &[], nucleus_cfg), 1);
    assert!(first_draw(seed) > nucleus_p0);
}

#[test]
fn repetition_penalty_changes_the_greedy_choice() {
    let logits = [1.2, 1.5];
    let history = [1];
    let cfg = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 2.0,
        seed: None,
    };

    assert_eq!(sample_with_seed(7, &logits, &history, cfg), 0);
}

#[test]
fn greedy_sampling_always_returns_the_argmax() {
    let logits = [-1.0, 0.5, 3.25, 2.0];
    let cfg = SamplerConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 0.1,
        repetition_penalty: 1.0,
        seed: None,
    };

    assert_eq!(sample_with_seed(1234, &logits, &[], cfg), 2);
}
