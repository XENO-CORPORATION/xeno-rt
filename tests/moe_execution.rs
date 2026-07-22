mod common;

use serde::Deserialize;
#[cfg(feature = "cuda")]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::time::Instant;

use sha2::{Digest, Sha256};
#[cfg(feature = "cuda")]
use xrt_gguf::GgufFile;
#[cfg(feature = "cuda")]
use xrt_models::LlamaConfig;
#[cfg(feature = "cuda")]
use xrt_runtime::{
    moe_config_sha256, CudaGraphMode, GpuResourceConfig, MoeAcceleration, MoePlacementPolicy,
};
use xrt_runtime::{BackendKind, KvCacheMode, MoeRuntimeConfig, Runtime};
#[cfg(feature = "cuda")]
use xrt_tokenizer::ChatMessage;

const SYNTHETIC_MOE_FIXTURE_SHA256: &str =
    "e0c12e81eb82cfa2aa4d38ff720be8582603c0d2b46c127b5880bf7a31e482a1";
const SYNTHETIC_QWEN35_HYBRID_MOE_FIXTURE_SHA256: &str =
    "bd297ebae52b930a1abec24d7c0b32a184d478b5b636013d1a87df3dab09e156";
const MOE_QUALITY_PROMPTS_SHA256: &str =
    "94619055554e553a2935f90d13e05d80fb85de6ab7192f18fe4f1161fdfffc8b";
const MOE_QUALITY_PROMPTS_JSON: &str = include_str!("common/moe-quality-prompts.json");

#[derive(Debug, Deserialize)]
struct MoeQualityPromptSuite {
    schema_version: u32,
    title: String,
    source: MoeQualityPromptSource,
    short_prompts: Vec<MoeQualityPrompt>,
    multi_turn_prompts: Vec<MoeQualityPrompt>,
    long_context_prompts: Vec<MoeLongQualityPrompt>,
    generated_256_prompt_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MoeQualityPromptSource {
    origin: String,
    license: String,
    private_user_content: bool,
    purpose: String,
}

#[derive(Debug, Deserialize)]
struct MoeQualityPrompt {
    id: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct MoeLongQualityPrompt {
    id: String,
    repeat_text: String,
    question: String,
}

#[cfg(feature = "cuda")]
const REAL_MOE_DEFAULT_EXPERT_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[cfg(feature = "cuda")]
fn real_moe_budget_bytes() -> u64 {
    std::env::var("XRT_REAL_MOE_GPU_EXPERT_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(REAL_MOE_DEFAULT_EXPERT_BUDGET_BYTES)
}

#[cfg(feature = "cuda")]
fn real_moe_parity_tokens() -> usize {
    std::env::var("XRT_REAL_MOE_PARITY_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(1, 32))
        .unwrap_or(2)
}

#[cfg(feature = "cuda")]
fn real_moe_placement() -> MoePlacementPolicy {
    match std::env::var("XRT_REAL_MOE_PLACEMENT")
        .unwrap_or_else(|_| "uniform".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "uniform" => MoePlacementPolicy::Uniform,
        "adaptive" => MoePlacementPolicy::Adaptive,
        other => panic!(
            "XRT_REAL_MOE_PLACEMENT must be uniform or adaptive for this gate, received {other:?}"
        ),
    }
}

#[cfg(feature = "cuda")]
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("real MoE logits must not be empty")
}

#[cfg(feature = "cuda")]
fn real_logit_metrics(actual: &[f32], expected: &[f32]) -> (f64, f64, f64) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut expected_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite(), "CUDA produced a non-finite logit");
        assert!(expected.is_finite(), "CPU produced a non-finite logit");
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let error = actual - expected;
        max_abs = max_abs.max(error.abs());
        dot += actual * expected;
        actual_norm += actual * actual;
        expected_norm += expected * expected;
        squared_error += error * error;
    }
    let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt()).max(f64::MIN_POSITIVE);
    let normalized_rms = (squared_error / actual.len() as f64).sqrt()
        / (expected_norm / expected.len() as f64)
            .sqrt()
            .max(f64::MIN_POSITIVE);
    (max_abs, cosine, normalized_rms)
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn real_logit_failure_diagnostics(actual: &[f32], expected: &[f32]) -> (f64, f64, f64, f64) {
    fn top_two_margin(values: &[f32]) -> f64 {
        let mut first = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        for &value in values {
            if value > first {
                second = first;
                first = value;
            } else if value > second {
                second = value;
            }
        }
        f64::from(first - second)
    }

    let expected_mean_square = expected
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    let error_mean_square = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| f64::from(actual - expected).powi(2))
        .sum::<f64>()
        / expected.len() as f64;
    (
        expected_mean_square.sqrt(),
        error_mean_square.sqrt(),
        top_two_margin(expected),
        top_two_margin(actual),
    )
}

#[cfg(feature = "cuda")]
fn assert_real_recurrent_state_close(
    actual: &xrt_runtime::backend::BackendStateSnapshot,
    expected: &xrt_runtime::backend::BackendStateSnapshot,
) {
    assert_eq!(actual.position(), expected.position());
    assert_eq!(actual.descriptor(), expected.descriptor());
    let report_layers =
        std::env::var("XRT_REAL_MOE_LAYER_DIAGNOSTICS").is_ok_and(|value| value.trim() == "1");
    let mut worst_conv = (0usize, 0.0f32);
    let mut worst_recurrent = (0usize, 0.0f32);
    for (layer, (actual, expected)) in actual.layers().iter().zip(expected.layers()).enumerate() {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                let max_conv_error = actual
                    .conv_state_f32()
                    .iter()
                    .zip(expected.conv_state_f32())
                    .map(|(&actual, &expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                let max_recurrent_error = actual
                    .recurrent_state_f32()
                    .iter()
                    .zip(expected.recurrent_state_f32())
                    .map(|(&actual, &expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                if report_layers {
                    eprintln!(
                        "real_moe_recurrent_layer_parity: layer={layer}, max_conv_abs={max_conv_error:.9}, max_recurrent_abs={max_recurrent_error:.9}"
                    );
                }
                if max_conv_error > worst_conv.1 {
                    worst_conv = (layer, max_conv_error);
                }
                if max_recurrent_error > worst_recurrent.1 {
                    worst_recurrent = (layer, max_recurrent_error);
                }
            }
            (None, None) => {}
            _ => panic!("real hybrid-MoE layer {layer} recurrent-state presence differs"),
        }
    }
    eprintln!(
        "real_moe_recurrent_parity: position={}, worst_conv_layer={}, worst_conv_abs={:.9}, worst_recurrent_layer={}, worst_recurrent_abs={:.9}",
        actual.position(),
        worst_conv.0,
        worst_conv.1,
        worst_recurrent.0,
        worst_recurrent.1,
    );
    assert!(
        worst_conv.1 <= 5e-4,
        "real hybrid-MoE layer {} convolution state diverged: {}",
        worst_conv.0,
        worst_conv.1
    );
    assert!(
        worst_recurrent.1 <= 2e-3,
        "real hybrid-MoE layer {} recurrent state diverged: {}",
        worst_recurrent.0,
        worst_recurrent.1
    );
}

#[cfg(feature = "cuda")]
fn run_real_moe_cpu_cuda_parity(model_path: &Path, expect_hybrid: bool) {
    let budget_bytes = real_moe_budget_bytes();
    let parity_tokens = real_moe_parity_tokens();
    let placement = real_moe_placement();
    assert!(
        std::env::var("XRT_CPU_FLOAT_ACTIVATION_REFERENCE")
            .is_ok_and(|value| value.trim() == "1"),
        "real MoE parity requires XRT_CPU_FLOAT_ACTIVATION_REFERENCE=1 so CPU and CUDA are compared in the same F32 activation domain"
    );
    let cpu_load_started = Instant::now();
    let cpu = Runtime::load_with_backend(model_path, BackendKind::Cpu)
        .expect("real MoE CPU runtime should load");
    let cpu_load = cpu_load_started.elapsed();
    assert!(
        cpu.backend().config().is_moe(),
        "fixture must be an MoE model"
    );
    assert_eq!(
        cpu.backend().config().is_hybrid(),
        expect_hybrid,
        "fixture hybrid architecture classification differs from the gate"
    );

    let cuda_load_started = Instant::now();
    let cuda = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(budget_bytes),
            placement,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("real MoE hybrid CUDA runtime should load");
    let cuda_load = cuda_load_started.elapsed();
    let status_before = cuda.moe_status();
    assert_eq!(status_before.effective_mode, "hybrid");
    assert!(status_before.gpu_expert_slots > 0);
    assert!(status_before.gpu_expert_bytes > 0);
    assert!(status_before.gpu_expert_bytes <= budget_bytes);

    let prompt = "Explain why sparse mixture-of-experts models can be efficient.";
    let encoded = cpu
        .tokenizer()
        .encode_with_options(prompt, true, true)
        .expect("real MoE parity prompt should tokenize");
    assert!(
        !encoded.is_empty(),
        "real MoE parity prompt produced no tokens"
    );
    let tokens = encoded
        .into_iter()
        .cycle()
        .take(parity_tokens)
        .collect::<Vec<_>>();
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, 16);
    let mut cuda_session = cuda.backend().new_session(KvCacheMode::F32, 16);
    cpu.backend()
        .prepare_session_state(&mut cpu_session)
        .expect("real CPU recurrent state should prepare");
    cuda.backend()
        .prepare_session_state(&mut cuda_session)
        .expect("real CUDA recurrent state should prepare");

    let transfers_before = cuda
        .gpu_transfer_stats()
        .expect("real CUDA transfer counters should exist");
    let mut worst_max_abs = 0.0f64;
    let mut worst_cosine = 1.0f64;
    let mut worst_normalized_rms = 0.0f64;
    let execution_started = Instant::now();
    for (position, token) in tokens.into_iter().enumerate() {
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        cpu.backend()
            .forward_token(token, position, &mut cpu_session, &mut expected)
            .expect("real CPU MoE token should execute");
        cuda.backend()
            .forward_token(token, position, &mut cuda_session, &mut actual)
            .expect("real hybrid CUDA MoE token should execute");
        let (max_abs, cosine, normalized_rms) = real_logit_metrics(&actual, &expected);
        eprintln!(
            "real_moe_token_parity: position={position}, max_abs={max_abs:.9}, cosine={cosine:.9}, normalized_rms={normalized_rms:.9}, expected_argmax={}, actual_argmax={}",
            argmax(&expected),
            argmax(&actual)
        );
        worst_max_abs = worst_max_abs.max(max_abs);
        worst_cosine = worst_cosine.min(cosine);
        worst_normalized_rms = worst_normalized_rms.max(normalized_rms);
        assert_eq!(
            argmax(&actual),
            argmax(&expected),
            "real MoE greedy token differs at position {position}"
        );
        assert!(
            cosine >= 0.99999,
            "real MoE cosine similarity {cosine} is below 0.99999 at position {position}"
        );
        assert!(
            normalized_rms <= 1e-3,
            "real MoE normalized RMS error {normalized_rms} exceeds 1e-3 at position {position}"
        );
    }
    let execution = execution_started.elapsed();

    let recurrent_snapshots = expect_hybrid.then(|| {
        let cpu_state = cpu
            .backend()
            .save_state(&cpu_session)
            .expect("real CPU recurrent snapshot should save")
            .expect("real hybrid CPU recurrent state should exist");
        let cuda_state = cuda
            .backend()
            .save_state(&cuda_session)
            .expect("real CUDA recurrent snapshot should save")
            .expect("real hybrid CUDA recurrent state should exist");
        (cuda_state, cpu_state)
    });

    let status_after = cuda.moe_status();
    assert!(status_after.cpu_expert_calls > 0);
    assert!(status_after.gpu_expert_calls > 0);
    assert_eq!(status_after.coordinator_failures, 0);
    assert_eq!(
        cuda.gpu_resource_status()
            .arena_allocations
            .expert_weight_bytes,
        status_before.gpu_expert_bytes,
        "real decode changed resident expert-weight accounting"
    );
    let transfers_after = cuda
        .gpu_transfer_stats()
        .expect("real CUDA transfer counters should exist");
    let transfer_delta = transfers_after.saturating_sub(&transfers_before);
    assert!(
        transfer_delta.host_to_device_bytes < status_before.gpu_expert_bytes,
        "real decode transferred expert-sized host data; resident weights may be moving per token"
    );
    eprintln!(
        "real_moe_parity: path={}, hybrid={}, placement={}, parity_tokens={}, budget_bytes={}, gpu_expert_slots={}, gpu_expert_bytes={}, cpu_load_ms={:.3}, cuda_load_ms={:.3}, execution_ms={:.3}, worst_max_abs={worst_max_abs:.9}, worst_cosine={worst_cosine:.9}, worst_normalized_rms={worst_normalized_rms:.9}, h2d_bytes={}, d2h_bytes={}, graph_captures={}, graph_replays={}",
        model_path.display(),
        expect_hybrid,
        placement.as_str(),
        parity_tokens,
        budget_bytes,
        status_before.gpu_expert_slots,
        status_before.gpu_expert_bytes,
        cpu_load.as_secs_f64() * 1000.0,
        cuda_load.as_secs_f64() * 1000.0,
        execution.as_secs_f64() * 1000.0,
        transfer_delta.host_to_device_bytes,
        transfer_delta.device_to_host_bytes,
        status_after.graph_captures,
        status_after.graph_replays,
    );
    if let Some((cuda_state, cpu_state)) = recurrent_snapshots {
        assert_real_recurrent_state_close(&cuda_state, &cpu_state);
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct PerplexityScore {
    negative_log_likelihood: f64,
    predicted_tokens: usize,
    perplexity: f64,
    windows: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
struct Gsm8kTaskCase {
    id: String,
    question: String,
    target: String,
}

#[cfg(feature = "cuda")]
fn score_perplexity_windows(
    runtime: &Runtime,
    tokens: &[u32],
    window_tokens: usize,
) -> PerplexityScore {
    assert!(
        tokens.len() >= 2,
        "perplexity corpus must contain at least two tokens"
    );
    assert!(window_tokens > 0, "perplexity window must be non-zero");

    let vocab_size = runtime.backend().config().vocab_size;
    let mut negative_log_likelihood = 0.0f64;
    let mut predicted_tokens = 0usize;
    let mut windows = 0usize;

    for start in (0..tokens.len() - 1).step_by(window_tokens) {
        let input_end = (start + window_tokens).min(tokens.len() - 1);
        let inputs = &tokens[start..input_end];
        if inputs.is_empty() {
            continue;
        }
        let target = tokens[input_end] as usize;
        assert!(
            target < vocab_size,
            "perplexity target token {target} exceeds vocabulary size {vocab_size}"
        );

        let mut session = runtime
            .backend()
            .new_session(KvCacheMode::F32, inputs.len());
        runtime
            .backend()
            .prepare_session_state(&mut session)
            .expect("perplexity session state should prepare");
        let window_started = Instant::now();
        eprintln!(
            "real_moe_perplexity_window_start: backend={}, window_index={windows}, corpus_start={start}, input_tokens={}",
            runtime.active_backend(),
            inputs.len()
        );
        let logits = runtime
            .backend()
            .forward_batch(inputs, 0, &mut session)
            .expect("perplexity window should execute");
        assert_eq!(
            logits.len(),
            vocab_size,
            "perplexity window returned an unexpected logit count"
        );
        let max_logit = logits
            .iter()
            .copied()
            .reduce(f32::max)
            .expect("perplexity logit row must not be empty");
        assert!(max_logit.is_finite(), "perplexity logits must be finite");
        let exp_sum = logits
            .iter()
            .map(|&logit| {
                assert!(logit.is_finite(), "perplexity logits must be finite");
                f64::from(logit - max_logit).exp()
            })
            .sum::<f64>();
        let log_sum_exp = f64::from(max_logit) + exp_sum.ln();
        negative_log_likelihood += log_sum_exp - f64::from(logits[target]);
        predicted_tokens += 1;
        eprintln!(
            "real_moe_perplexity_window_done: backend={}, window_index={windows}, corpus_start={start}, input_tokens={}, wall_ms={:.3}",
            runtime.active_backend(),
            inputs.len(),
            window_started.elapsed().as_secs_f64() * 1000.0
        );
        windows += 1;
    }

    assert!(predicted_tokens > 0, "perplexity run predicted no tokens");
    let perplexity = (negative_log_likelihood / predicted_tokens as f64).exp();
    assert!(perplexity.is_finite(), "perplexity must be finite");
    PerplexityScore {
        negative_log_likelihood,
        predicted_tokens,
        perplexity,
        windows,
    }
}

#[cfg(feature = "cuda")]
fn tokenize_perplexity_prefix(
    runtime: &Runtime,
    corpus: &str,
    max_tokens: usize,
) -> (Vec<u32>, usize, usize) {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut source_bytes = 0usize;
    let mut segments = 0usize;
    for segment in corpus.split_inclusive('\n') {
        if segment.is_empty() {
            continue;
        }
        source_bytes = source_bytes
            .checked_add(segment.len())
            .expect("perplexity source-byte count overflowed");
        segments += 1;
        tokens.extend(
            runtime
                .tokenizer()
                .encode_with_options(segment, false, false)
                .expect("perplexity corpus segment should tokenize"),
        );
        if tokens.len() >= max_tokens {
            tokens.truncate(max_tokens);
            break;
        }
    }
    (tokens, source_bytes, segments)
}

#[cfg(feature = "cuda")]
fn run_real_qwen3_moe_perplexity(model_path: &Path, corpus_path: &Path) {
    const MAX_RELATIVE_PERPLEXITY_CHANGE: f64 = 0.001;

    let activation_mode = if std::env::var("XRT_CPU_FLOAT_ACTIVATION_REFERENCE")
        .is_ok_and(|value| value.trim() == "1")
    {
        "f32_reference"
    } else {
        "production"
    };
    let expected_sha256 = std::env::var("XRT_REAL_MOE_PERPLEXITY_TEXT_SHA256")
        .expect("set XRT_REAL_MOE_PERPLEXITY_TEXT_SHA256 to pin the exact corpus bytes")
        .trim()
        .to_ascii_lowercase();
    assert_eq!(
        expected_sha256.len(),
        64,
        "perplexity corpus SHA-256 must contain 64 hexadecimal characters"
    );
    assert!(
        expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "perplexity corpus SHA-256 must be hexadecimal"
    );
    let corpus_bytes = fs::read(corpus_path).expect("perplexity corpus should be readable");
    let actual_sha256 = format!("{:x}", Sha256::digest(&corpus_bytes));
    assert_eq!(
        actual_sha256, expected_sha256,
        "perplexity corpus bytes do not match the pinned SHA-256"
    );
    let corpus = std::str::from_utf8(&corpus_bytes).expect("perplexity corpus must be UTF-8");

    let max_tokens = std::env::var("XRT_REAL_MOE_PERPLEXITY_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1025);
    let window_tokens = std::env::var("XRT_REAL_MOE_PERPLEXITY_WINDOW")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32);
    assert!(
        max_tokens >= 2,
        "perplexity max token count must be at least two"
    );
    assert!(window_tokens > 0, "perplexity window must be non-zero");

    let cpu_only =
        std::env::var("XRT_REAL_MOE_PERPLEXITY_CPU_ONLY").is_ok_and(|value| value.trim() == "1");
    let cuda_only =
        std::env::var("XRT_REAL_MOE_PERPLEXITY_CUDA_ONLY").is_ok_and(|value| value.trim() == "1");
    assert!(
        !(cpu_only && cuda_only),
        "perplexity CPU-only and CUDA-only modes are mutually exclusive"
    );
    if cfg!(feature = "moe-router-exact-reference") {
        assert!(
            cpu_only && !cuda_only,
            "moe-router-exact-reference is a scalar CPU semantic control and must use XRT_REAL_MOE_PERPLEXITY_CPU_ONLY=1"
        );
    }

    let cpu_load_started = Instant::now();
    let cpu = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::Cpu,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Cpu,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig::default(),
    )
    .expect("real optimized-CPU MoE runtime should load for perplexity");
    let cpu_load_ms = cpu_load_started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        cpu.backend().config().is_moe(),
        "fixture must be an MoE model"
    );
    assert_eq!(cpu.moe_status().effective_mode, "cpu");
    let tokenization_started = Instant::now();
    let (tokens, corpus_prefix_bytes, corpus_prefix_segments) =
        tokenize_perplexity_prefix(&cpu, corpus, max_tokens);
    let tokenization_ms = tokenization_started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        tokens.len() >= 2,
        "perplexity corpus produced fewer than two scored tokens"
    );

    let cpu_score = if cuda_only {
        None
    } else {
        let cpu_started = Instant::now();
        let score = score_perplexity_windows(&cpu, &tokens, window_tokens);
        let cpu_wall_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "real_moe_perplexity_cpu: router={}, activation_mode={activation_mode}, tokenization=line_bounded_prefix, sampling=window_endpoint, path={}, corpus_sha256={}, corpus_bytes={}, corpus_prefix_bytes={corpus_prefix_bytes}, corpus_prefix_segments={corpus_prefix_segments}, tokenized_tokens={}, predicted_tokens={}, window_tokens={}, windows={}, negative_log_likelihood={:.9}, perplexity={:.9}, load_ms={cpu_load_ms:.3}, tokenization_ms={tokenization_ms:.3}, score_wall_ms={cpu_wall_ms:.3}",
            if cfg!(feature = "moe-router-exact-reference") {
                "exact"
            } else {
                "boundary_band_1e-5"
            },
            model_path.display(),
            actual_sha256,
            corpus_bytes.len(),
            tokens.len(),
            score.predicted_tokens,
            window_tokens,
            score.windows,
            score.negative_log_likelihood,
            score.perplexity,
        );
        Some(score)
    };
    if cpu_only {
        return;
    }
    drop(cpu);

    let reference_perplexity = match cpu_score {
        Some(score) => score.perplexity,
        None => std::env::var("XRT_REAL_MOE_PERPLEXITY_REFERENCE")
            .expect("CUDA-only perplexity requires XRT_REAL_MOE_PERPLEXITY_REFERENCE from the matching CPU-only run")
            .parse::<f64>()
            .expect("XRT_REAL_MOE_PERPLEXITY_REFERENCE must be a finite positive number"),
    };
    assert!(
        reference_perplexity.is_finite() && reference_perplexity > 0.0,
        "perplexity reference must be finite and positive"
    );

    let budget_bytes = real_moe_budget_bytes();
    let placement = real_moe_placement();
    let cuda_load_started = Instant::now();
    let cuda = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(budget_bytes),
            placement,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("real MoE hybrid CUDA runtime should load for perplexity");
    let cuda_load_ms = cuda_load_started.elapsed().as_secs_f64() * 1000.0;
    let cuda_started = Instant::now();
    let cuda_score = score_perplexity_windows(&cuda, &tokens, window_tokens);
    let cuda_wall_ms = cuda_started.elapsed().as_secs_f64() * 1000.0;
    let expected_windows = (tokens.len() - 1).div_ceil(window_tokens);
    assert_eq!(cuda_score.predicted_tokens, expected_windows);
    assert_eq!(cuda_score.windows, expected_windows);
    let relative_change = (cuda_score.perplexity - reference_perplexity) / reference_perplexity;
    eprintln!(
        "real_moe_perplexity_cuda: activation_mode={activation_mode}, sampling=window_endpoint, placement={}, budget_bytes={}, reference_source={}, reference_perplexity={reference_perplexity:.9}, predicted_tokens={}, windows={}, negative_log_likelihood={:.9}, perplexity={:.9}, relative_change={relative_change:.9}, absolute_relative_change={:.9}, load_ms={cuda_load_ms:.3}, score_wall_ms={cuda_wall_ms:.3}",
        placement.as_str(),
        budget_bytes,
        if cuda_only {
            "matching_cpu_only_run"
        } else {
            "paired_cpu"
        },
        cuda_score.predicted_tokens,
        cuda_score.windows,
        cuda_score.negative_log_likelihood,
        cuda_score.perplexity,
        relative_change.abs(),
    );
    assert!(
        relative_change.abs() <= MAX_RELATIVE_PERPLEXITY_CHANGE,
        "hybrid CUDA perplexity changed by {:.6}%, exceeding the 0.1% gate",
        relative_change.abs() * 100.0
    );
}

#[cfg(feature = "cuda")]
fn load_gsm8k_task_cases(path: &Path, expected_sha256: &str) -> Vec<Gsm8kTaskCase> {
    let bytes = fs::read(path).expect("GSM8K task fixture should be readable");
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_sha256,
        expected_sha256.trim().to_ascii_lowercase(),
        "GSM8K task fixture bytes do not match the pinned SHA-256"
    );
    let text = std::str::from_utf8(&bytes).expect("GSM8K task fixture must be UTF-8");
    let cases = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("GSM8K task fixture row should parse"))
        .collect::<Vec<_>>();
    assert_eq!(
        cases.len(),
        16,
        "pinned GSM8K projection must contain 16 rows"
    );
    cases
}

#[cfg(feature = "cuda")]
fn normalize_last_integer(text: &str) -> Option<String> {
    let mut last = None;
    let mut current = String::new();
    for character in text.chars().chain(std::iter::once(' ')) {
        if character.is_ascii_digit()
            || (character == '-' && current.is_empty())
            || (character == ',' && !current.is_empty())
        {
            current.push(character);
        } else if !current.is_empty() {
            let normalized = current.replace(',', "");
            if normalized != "-" {
                last = Some(normalized);
            }
            current.clear();
        }
    }
    last
}

#[cfg(feature = "cuda")]
fn generate_gsm8k_answer(
    runtime: &Runtime,
    question: &str,
    max_output_tokens: usize,
) -> (Option<String>, String, usize) {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: format!(
            "Solve this grade-school math problem and return only the final integer after ####. {question} /no_think"
        ),
    }];
    let prompt = runtime
        .tokenizer()
        .format_chat(&messages, true)
        .expect("GSM8K chat prompt should format");
    let prompt_tokens = runtime
        .tokenizer()
        .encode_with_options(&prompt, false, true)
        .expect("GSM8K prompt should tokenize");
    assert!(!prompt_tokens.is_empty(), "GSM8K prompt tokenized empty");
    let context_tokens = prompt_tokens
        .len()
        .checked_add(max_output_tokens)
        .expect("GSM8K context token count overflowed");
    let mut session = runtime
        .backend()
        .new_session(KvCacheMode::F32, context_tokens);
    runtime
        .backend()
        .prepare_session_state(&mut session)
        .expect("GSM8K session state should prepare");
    let mut logits = runtime
        .backend()
        .forward_batch(&prompt_tokens, 0, &mut session)
        .expect("GSM8K prompt should execute");
    let eos = runtime.tokenizer().special_tokens().eos;
    let mut output_tokens = Vec::with_capacity(max_output_tokens);
    for output_index in 0..max_output_tokens {
        let next = argmax(&logits) as u32;
        if Some(next) == eos {
            break;
        }
        output_tokens.push(next);
        if output_index + 1 < max_output_tokens {
            runtime
                .backend()
                .forward_token(
                    next,
                    prompt_tokens.len() + output_index,
                    &mut session,
                    &mut logits,
                )
                .expect("GSM8K generated token should execute");
        }
    }
    let output = runtime
        .tokenizer()
        .decode(&output_tokens, true)
        .expect("GSM8K output should decode");
    (
        normalize_last_integer(&output),
        format!("{:x}", Sha256::digest(output.as_bytes())),
        output_tokens.len(),
    )
}

#[cfg(feature = "cuda")]
fn paired_bootstrap_task_interval(cpu: &[bool], cuda: &[bool]) -> (f64, f64) {
    const RESAMPLES: usize = 10_000;
    assert_eq!(cpu.len(), cuda.len());
    assert!(!cpu.is_empty());
    let mut state = 20_260_720u64;
    let mut estimates = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut difference = 0.0f64;
        for _ in 0..cpu.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let index = state as usize % cpu.len();
            difference += f64::from(u8::from(cuda[index])) - f64::from(u8::from(cpu[index]));
        }
        estimates.push(difference / cpu.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    (estimates[249], estimates[9_750])
}

#[cfg(feature = "cuda")]
fn run_real_qwen3_moe_gsm8k_task(model_path: &Path, fixture_path: &Path) {
    let fixture_sha256 = std::env::var("XRT_REAL_MOE_GSM8K_SHA256")
        .expect("set XRT_REAL_MOE_GSM8K_SHA256 to pin the projected task fixture");
    let cases = load_gsm8k_task_cases(fixture_path, &fixture_sha256);
    let case_count = std::env::var("XRT_REAL_MOE_GSM8K_CASES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, cases.len());
    let max_output_tokens = std::env::var("XRT_REAL_MOE_GSM8K_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128)
        .clamp(1, 512);

    let cpu = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::Cpu,
        MoeRuntimeConfig::optimized_cpu(),
        GpuResourceConfig::default(),
    )
    .expect("optimized CPU runtime should load for GSM8K");
    assert_eq!(cpu.moe_status().effective_mode, "cpu");
    let budget_bytes = real_moe_budget_bytes();
    let placement = real_moe_placement();
    let cuda = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(budget_bytes),
            placement,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("hybrid CUDA runtime should load for GSM8K");
    assert_eq!(cuda.moe_status().effective_mode, "hybrid");

    let started = Instant::now();
    let mut cpu_outcomes = Vec::with_capacity(case_count);
    let mut cuda_outcomes = Vec::with_capacity(case_count);
    for case in cases.iter().take(case_count) {
        let case_started = Instant::now();
        let (cpu_answer, cpu_output_sha256, cpu_output_tokens) =
            generate_gsm8k_answer(&cpu, &case.question, max_output_tokens);
        let (cuda_answer, cuda_output_sha256, cuda_output_tokens) =
            generate_gsm8k_answer(&cuda, &case.question, max_output_tokens);
        let cpu_correct = cpu_answer.as_deref() == Some(case.target.as_str());
        let cuda_correct = cuda_answer.as_deref() == Some(case.target.as_str());
        cpu_outcomes.push(cpu_correct);
        cuda_outcomes.push(cuda_correct);
        eprintln!(
            "real_moe_gsm8k_case: id={}, cpu_correct={cpu_correct}, cuda_correct={cuda_correct}, cpu_output_tokens={cpu_output_tokens}, cuda_output_tokens={cuda_output_tokens}, cpu_output_sha256={cpu_output_sha256}, cuda_output_sha256={cuda_output_sha256}, wall_ms={:.3}",
            case.id,
            case_started.elapsed().as_secs_f64() * 1000.0
        );
    }
    let cpu_correct = cpu_outcomes.iter().filter(|&&correct| correct).count();
    let cuda_correct = cuda_outcomes.iter().filter(|&&correct| correct).count();
    let cpu_score = cpu_correct as f64 / case_count as f64;
    let cuda_score = cuda_correct as f64 / case_count as f64;
    let difference = cuda_score - cpu_score;
    let (ci_lower, ci_upper) = paired_bootstrap_task_interval(&cpu_outcomes, &cuda_outcomes);
    eprintln!(
        "real_moe_gsm8k_suite: cases={case_count}, max_output_tokens={max_output_tokens}, cpu_correct={cpu_correct}, cuda_correct={cuda_correct}, cpu_exact_match={cpu_score:.9}, cuda_exact_match={cuda_score:.9}, difference={difference:.9}, paired_bootstrap_ci95=[{ci_lower:.9},{ci_upper:.9}], resamples=10000, seed=20260720, wall_ms={:.3}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    assert!(
        ci_lower >= 0.0,
        "hybrid CUDA GSM8K exact-match is inferior to optimized CPU: difference={difference}, ci95=[{ci_lower},{ci_upper}]"
    );
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MoeQualityProfile {
    Smoke,
    Full,
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
impl MoeQualityProfile {
    fn from_env() -> Self {
        match std::env::var("XRT_REAL_MOE_QUALITY_PROFILE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "smoke" => Self::Smoke,
            "full" => Self::Full,
            other => panic!(
                "XRT_REAL_MOE_QUALITY_PROFILE must be explicitly set to smoke or full, received {other:?}"
            ),
        }
    }

    fn long_context_tokens(self) -> usize {
        match self {
            Self::Smoke => 128,
            Self::Full => 256,
        }
    }
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
struct MoeQualityCase {
    id: String,
    kind: &'static str,
    tokens: Vec<u32>,
    generated_tokens: usize,
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn parse_moe_quality_prompt_suite() -> MoeQualityPromptSuite {
    serde_json::from_str(MOE_QUALITY_PROMPTS_JSON)
        .expect("pinned MoE quality prompt suite should parse")
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn encode_quality_prompt(runtime: &Runtime, text: &str) -> Vec<u32> {
    runtime
        .tokenizer()
        .encode_with_options(text, true, true)
        .expect("quality prompt should tokenize")
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn encode_long_quality_prompt(
    runtime: &Runtime,
    prompt: &MoeLongQualityPrompt,
    minimum_tokens: usize,
) -> Vec<u32> {
    let repeat_tokens = runtime
        .tokenizer()
        .encode_with_options(&prompt.repeat_text, false, true)
        .expect("long-context repeat text should tokenize");
    assert!(
        !repeat_tokens.is_empty(),
        "long-context repeat text is empty"
    );
    let question_tokens = runtime
        .tokenizer()
        .encode_with_options(&prompt.question, false, true)
        .expect("long-context question should tokenize");
    assert!(
        !question_tokens.is_empty(),
        "long-context question is empty"
    );

    let mut tokens = encode_quality_prompt(runtime, &prompt.repeat_text);
    while tokens.len() < minimum_tokens {
        tokens.extend_from_slice(&repeat_tokens);
    }
    tokens.extend_from_slice(&question_tokens);
    tokens
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn build_moe_quality_cases(
    runtime: &Runtime,
    suite: &MoeQualityPromptSuite,
    profile: MoeQualityProfile,
    case_filter: Option<&str>,
    long_context_tokens: usize,
) -> Vec<MoeQualityCase> {
    let generated_256 = suite
        .generated_256_prompt_ids
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let short_count = match profile {
        MoeQualityProfile::Smoke => 1,
        MoeQualityProfile::Full => suite.short_prompts.len(),
    };
    let multi_count = match profile {
        MoeQualityProfile::Smoke => 1,
        MoeQualityProfile::Full => suite.multi_turn_prompts.len(),
    };
    let long_count = match profile {
        MoeQualityProfile::Smoke => 1,
        MoeQualityProfile::Full => suite.long_context_prompts.len(),
    };
    let mut cases = Vec::with_capacity(short_count + multi_count + long_count);
    for prompt in suite
        .short_prompts
        .iter()
        .take(short_count)
        .filter(|prompt| match case_filter {
            Some(filter) => prompt.id == filter,
            None => true,
        })
    {
        let generated_tokens = match profile {
            MoeQualityProfile::Smoke => 8,
            MoeQualityProfile::Full if generated_256.contains(prompt.id.as_str()) => 256,
            MoeQualityProfile::Full => 1,
        };
        cases.push(MoeQualityCase {
            id: prompt.id.clone(),
            kind: "short",
            tokens: encode_quality_prompt(runtime, &prompt.text),
            generated_tokens,
        });
    }
    for prompt in suite
        .multi_turn_prompts
        .iter()
        .take(multi_count)
        .filter(|prompt| match case_filter {
            Some(filter) => prompt.id == filter,
            None => true,
        })
    {
        cases.push(MoeQualityCase {
            id: prompt.id.clone(),
            kind: "multi-turn",
            tokens: encode_quality_prompt(runtime, &prompt.text),
            generated_tokens: 1,
        });
    }
    for prompt in suite
        .long_context_prompts
        .iter()
        .take(long_count)
        .filter(|prompt| match case_filter {
            Some(filter) => prompt.id == filter,
            None => true,
        })
    {
        cases.push(MoeQualityCase {
            id: prompt.id.clone(),
            kind: "long-context",
            tokens: encode_long_quality_prompt(runtime, prompt, long_context_tokens),
            generated_tokens: 1,
        });
    }
    cases
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
const MOE_ROUTER_PAIRWISE_AMBIGUITY_MAX: f32 = 4.0e-4;

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
const MOE_AMBIGUOUS_SUBSTITUTION_RATE_DENOMINATOR: usize = 10_000;

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
const MOE_LONG_CONTEXT_ROUTE_AGREEMENT_DENOMINATOR: usize = 100;

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
#[derive(Debug, Clone, Copy, Default)]
struct MoeRouteParitySummary {
    route_entries: usize,
    ambiguous_boundary_substitutions: usize,
    long_context_route_entries: usize,
    long_context_route_divergences: usize,
    max_long_context_symmetric_difference: usize,
    max_cpu_boundary_gap: f32,
    max_cuda_boundary_gap: f32,
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
impl MoeRouteParitySummary {
    fn accumulate(&mut self, other: Self) {
        self.route_entries += other.route_entries;
        self.ambiguous_boundary_substitutions += other.ambiguous_boundary_substitutions;
        self.long_context_route_entries += other.long_context_route_entries;
        self.long_context_route_divergences += other.long_context_route_divergences;
        self.max_long_context_symmetric_difference = self
            .max_long_context_symmetric_difference
            .max(other.max_long_context_symmetric_difference);
        self.max_cpu_boundary_gap = self.max_cpu_boundary_gap.max(other.max_cpu_boundary_gap);
        self.max_cuda_boundary_gap = self.max_cuda_boundary_gap.max(other.max_cuda_boundary_gap);
    }
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn canonical_moe_route_trace_order(trace: &xrt_runtime::MoeRouteTrace) -> Vec<(usize, usize)> {
    let layer_count = trace
        .entries()
        .iter()
        .map(|entry| entry.layer_index())
        .max()
        .map_or(0, |layer| layer + 1);
    let mut layer_occurrences = vec![0usize; layer_count];
    let mut order = Vec::with_capacity(trace.entries().len());
    for (entry_index, entry) in trace.entries().iter().enumerate() {
        let layer = entry.layer_index();
        let token_ordinal = layer_occurrences[layer];
        layer_occurrences[layer] += 1;
        order.push((token_ordinal, entry_index));
    }
    order.sort_unstable_by_key(|&(token_ordinal, entry_index)| {
        (token_ordinal, trace.entries()[entry_index].layer_index())
    });
    order
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn assert_moe_route_trace_identity(
    case_id: &str,
    expected: &xrt_runtime::MoeRouteTrace,
    actual: &xrt_runtime::MoeRouteTrace,
) -> MoeRouteParitySummary {
    assert!(
        !expected.overflowed(),
        "CPU route trace overflowed for {case_id}"
    );
    assert!(
        !actual.overflowed(),
        "CUDA route trace overflowed for {case_id}"
    );
    assert_eq!(
        actual.entries().len(),
        expected.entries().len(),
        "CPU/CUDA route trace lengths differ for {case_id}"
    );
    let expected_order = canonical_moe_route_trace_order(expected);
    let actual_order = canonical_moe_route_trace_order(actual);
    assert_eq!(
        actual_order.len(),
        expected_order.len(),
        "CPU/CUDA canonical route trace lengths differ for {case_id}"
    );
    let allow_long_context_substitution = case_id.starts_with("long-");
    let mut summary = MoeRouteParitySummary::default();
    for (index, ((expected_token, expected_index), (actual_token, actual_index))) in
        expected_order.iter().zip(&actual_order).enumerate()
    {
        let expected = &expected.entries()[*expected_index];
        let actual = &actual.entries()[*actual_index];
        summary.route_entries += 1;
        if allow_long_context_substitution {
            summary.long_context_route_entries += 1;
        }
        assert_eq!(
            actual_token, expected_token,
            "CPU/CUDA route token ordinals differ for {case_id} at canonical entry {index}"
        );
        assert_eq!(
            actual.layer_index(),
            expected.layer_index(),
            "CPU/CUDA route layers differ for {case_id} at trace entry {index}"
        );
        let mut actual_ids = actual.logical_ids().to_vec();
        let mut expected_ids = expected.logical_ids().to_vec();
        actual_ids.sort_unstable();
        expected_ids.sort_unstable();
        if actual_ids == expected_ids {
            continue;
        }

        let cpu_only = expected_ids
            .iter()
            .copied()
            .filter(|id| !actual_ids.contains(id))
            .collect::<Vec<_>>();
        let cuda_only = actual_ids
            .iter()
            .copied()
            .filter(|id| !expected_ids.contains(id))
            .collect::<Vec<_>>();
        let (cpu_selected, cpu_selected_logit, cpu_excluded, cpu_excluded_logit) =
            expected.boundary_diagnostic();
        let (cuda_selected, cuda_selected_logit, cuda_excluded, cuda_excluded_logit) =
            actual.boundary_diagnostic();
        let cpu_gap = (cpu_selected_logit - cpu_excluded_logit).abs();
        let cuda_gap = (cuda_selected_logit - cuda_excluded_logit).abs();
        let is_registered_boundary_substitution = cpu_only.len() == 1
            && cuda_only.len() == 1
            && cpu_selected == cpu_only[0]
            && cpu_excluded == cuda_only[0]
            && cuda_selected == cuda_only[0]
            && cuda_excluded == cpu_only[0]
            && cpu_gap.is_finite()
            && cuda_gap.is_finite()
            && cpu_gap <= MOE_ROUTER_PAIRWISE_AMBIGUITY_MAX
            && cuda_gap <= MOE_ROUTER_PAIRWISE_AMBIGUITY_MAX;
        if allow_long_context_substitution {
            assert_eq!(
                actual_ids.len(),
                expected_ids.len(),
                "CPU/CUDA long-context top-k widths differ for {case_id} at trace entry {index}"
            );
            summary.long_context_route_divergences += 1;
            summary.max_long_context_symmetric_difference = summary
                .max_long_context_symmetric_difference
                .max(cpu_only.len() + cuda_only.len());
        } else {
            assert!(
                is_registered_boundary_substitution,
                "CPU/CUDA logical expert sets differ outside the registered boundary ambiguity for {case_id} at trace entry {index}, layer {}; CPU route {:?} boundary {:?}, CUDA route {:?} boundary {:?}",
                expected.layer_index(),
                expected.logical_ids(),
                expected.boundary_diagnostic(),
                actual.logical_ids(),
                actual.boundary_diagnostic()
            );
            summary.ambiguous_boundary_substitutions += 1;
        }
        summary.max_cpu_boundary_gap = summary.max_cpu_boundary_gap.max(cpu_gap);
        summary.max_cuda_boundary_gap = summary.max_cuda_boundary_gap.max(cuda_gap);
    }
    summary
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn take_moe_route_traces(
    case_id: &str,
    cpu: &Runtime,
    cuda: &Runtime,
) -> (
    xrt_runtime::MoeRouteTrace,
    xrt_runtime::MoeRouteTrace,
    MoeRouteParitySummary,
) {
    let cpu_trace = cpu
        .backend()
        .take_moe_route_trace()
        .expect("CPU route trace should finish")
        .expect("CPU route trace should exist");
    let cuda_trace = cuda
        .backend()
        .take_moe_route_trace()
        .expect("CUDA route trace should finish")
        .expect("CUDA route trace should exist");
    let parity = assert_moe_route_trace_identity(case_id, &cpu_trace, &cuda_trace);
    (cpu_trace, cuda_trace, parity)
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
fn run_real_qwen3_moe_quality_suite(model_path: &Path, profile: MoeQualityProfile) {
    assert!(
        std::env::var("XRT_CPU_FLOAT_ACTIVATION_REFERENCE").is_ok_and(|value| value.trim() == "1"),
        "real MoE quality parity requires XRT_CPU_FLOAT_ACTIVATION_REFERENCE=1"
    );
    let suite = parse_moe_quality_prompt_suite();
    let budget_bytes = real_moe_budget_bytes();
    let placement = real_moe_placement();
    let cpu_load_started = Instant::now();
    let cpu = Runtime::load_with_backend(model_path, BackendKind::Cpu)
        .expect("real Qwen3 MoE CPU runtime should load");
    let cpu_load_ms = cpu_load_started.elapsed().as_secs_f64() * 1000.0;
    assert!(cpu.backend().config().is_moe());
    assert!(!cpu.backend().config().is_hybrid());

    let cuda_load_started = Instant::now();
    let cuda = Runtime::load_with_backend_configs(
        model_path,
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(budget_bytes),
            placement,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("real Qwen3 MoE hybrid CUDA runtime should load");
    let cuda_load_ms = cuda_load_started.elapsed().as_secs_f64() * 1000.0;
    let case_filter = std::env::var("XRT_REAL_MOE_QUALITY_CASE")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let long_context_tokens = std::env::var("XRT_REAL_MOE_QUALITY_LONG_TOKENS")
        .ok()
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("XRT_REAL_MOE_QUALITY_LONG_TOKENS must be an integer")
        })
        .unwrap_or_else(|| profile.long_context_tokens());
    assert!(
        (128..=cpu.backend().config().context_length).contains(&long_context_tokens),
        "XRT_REAL_MOE_QUALITY_LONG_TOKENS must be in 128..={} ",
        cpu.backend().config().context_length
    );
    if std::env::var_os("XRT_REAL_MOE_QUALITY_LONG_TOKENS").is_some() {
        eprintln!("real_moe_quality_long_override: minimum_tokens={long_context_tokens}");
    }
    let mut cases = build_moe_quality_cases(
        &cpu,
        &suite,
        profile,
        case_filter.as_deref(),
        long_context_tokens,
    );
    if let Some(case_filter) = case_filter {
        assert_eq!(
            cases.len(),
            1,
            "XRT_REAL_MOE_QUALITY_CASE {case_filter:?} did not select exactly one case"
        );
        eprintln!("real_moe_quality_filter: case_id={case_filter}");
    }
    if let Some(output_tokens) = std::env::var("XRT_REAL_MOE_QUALITY_OUTPUT_TOKENS")
        .ok()
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("XRT_REAL_MOE_QUALITY_OUTPUT_TOKENS must be an integer")
        })
    {
        assert!(
            (1..=256).contains(&output_tokens),
            "XRT_REAL_MOE_QUALITY_OUTPUT_TOKENS must be in 1..=256"
        );
        for case in &mut cases {
            case.generated_tokens = output_tokens;
        }
        eprintln!("real_moe_quality_output_override: generated_tokens={output_tokens}");
    }
    let transfers_before = cuda
        .gpu_transfer_stats()
        .expect("quality CUDA transfer counters should exist");

    let mut metric_count = 0usize;
    let mut cosine_sum = 0.0f64;
    let mut worst_max_abs = 0.0f64;
    let mut worst_cosine = 1.0f64;
    let mut worst_normalized_rms = 0.0f64;
    let mut route_parity = MoeRouteParitySummary::default();
    let suite_started = Instant::now();
    for case in &cases {
        assert!(!case.tokens.is_empty(), "quality case {} is empty", case.id);
        let executed_tokens = case
            .tokens
            .len()
            .checked_add(case.generated_tokens.saturating_sub(1))
            .expect("quality token count overflowed");
        assert!(
            executed_tokens <= cpu.backend().config().context_length,
            "quality case {} needs {executed_tokens} tokens but context is {}",
            case.id,
            cpu.backend().config().context_length
        );
        let trace_capacity = executed_tokens
            .checked_mul(cpu.backend().config().block_count)
            .expect("quality route trace capacity overflowed");
        assert!(
            trace_capacity <= 1_000_000,
            "quality case {} route trace needs {trace_capacity} entries, exceeding the hard limit",
            case.id
        );
        cpu.backend()
            .start_moe_route_trace(trace_capacity)
            .expect("CPU route trace should start");
        cuda.backend()
            .start_moe_route_trace(trace_capacity)
            .expect("CUDA route trace should start");
        cpu.backend()
            .prepare_request()
            .expect("CPU quality request should prepare");
        cuda.backend()
            .prepare_request()
            .expect("CUDA quality request should prepare");
        let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, 16);
        let mut cuda_session = cuda.backend().new_session(KvCacheMode::F32, 16);
        cpu.backend()
            .prepare_session_state(&mut cpu_session)
            .expect("CPU quality session should prepare");
        cuda.backend()
            .prepare_session_state(&mut cuda_session)
            .expect("CUDA quality session should prepare");

        let case_started = Instant::now();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        if case.kind == "long-context" {
            expected = cpu
                .backend()
                .forward_batch(&case.tokens, 0, &mut cpu_session)
                .expect("CPU long-context quality prompt batch should execute");
            actual = cuda
                .backend()
                .forward_batch(&case.tokens, 0, &mut cuda_session)
                .expect("CUDA long-context quality prompt batch should execute");
        } else {
            for (position, &token) in case.tokens.iter().enumerate() {
                cpu.backend()
                    .forward_token(token, position, &mut cpu_session, &mut expected)
                    .expect("CPU quality prompt token should execute");
                cuda.backend()
                    .forward_token(token, position, &mut cuda_session, &mut actual)
                    .expect("CUDA quality prompt token should execute");
            }
        }

        let mut generated = Vec::with_capacity(case.generated_tokens);
        for output_index in 0..case.generated_tokens {
            let (max_abs, cosine, normalized_rms) = real_logit_metrics(&actual, &expected);
            let expected_token = argmax(&expected) as u32;
            let actual_token = argmax(&actual) as u32;
            if actual_token != expected_token || cosine < 0.99999 || normalized_rms > 1e-3 {
                let (expected_rms, error_rms, cpu_top_margin, cuda_top_margin) =
                    real_logit_failure_diagnostics(&actual, &expected);
                let (cpu_trace, _, partial_route_parity) =
                    take_moe_route_traces(&case.id, &cpu, &cuda);
                eprintln!(
                    "real_moe_quality_failure: id={}, output_index={output_index}, argmax_identity={}, max_abs={max_abs:.9}, cosine={cosine:.9}, normalized_rms={normalized_rms:.9}, expected_rms={expected_rms:.9}, error_rms={error_rms:.9}, cpu_top_margin={cpu_top_margin:.9}, cuda_top_margin={cuda_top_margin:.9}, partial_route_entries={}, ambiguous_boundary_substitutions={}, long_context_route_divergences={}, route_gate=true",
                    case.id,
                    actual_token == expected_token,
                    cpu_trace.entries().len(),
                    partial_route_parity.ambiguous_boundary_substitutions,
                    partial_route_parity.long_context_route_divergences
                );
            }
            assert_eq!(
                actual_token, expected_token,
                "CPU/CUDA greedy token differs for {} at output {output_index}",
                case.id
            );
            assert!(
                cosine >= 0.99999,
                "CPU/CUDA cosine {cosine} is below 0.99999 for {} at output {output_index}",
                case.id
            );
            assert!(
                normalized_rms <= 1e-3,
                "CPU/CUDA normalized RMS {normalized_rms} exceeds 1e-3 for {} at output {output_index}",
                case.id
            );
            metric_count += 1;
            cosine_sum += cosine;
            worst_max_abs = worst_max_abs.max(max_abs);
            worst_cosine = worst_cosine.min(cosine);
            worst_normalized_rms = worst_normalized_rms.max(normalized_rms);
            generated.push(expected_token);
            if output_index + 1 < case.generated_tokens {
                let position = case.tokens.len() + output_index;
                cpu.backend()
                    .forward_token(expected_token, position, &mut cpu_session, &mut expected)
                    .expect("CPU quality generated token should execute");
                cuda.backend()
                    .forward_token(expected_token, position, &mut cuda_session, &mut actual)
                    .expect("CUDA quality generated token should execute");
            }
        }

        let (cpu_trace, _, case_route_parity) = take_moe_route_traces(&case.id, &cpu, &cuda);
        route_parity.accumulate(case_route_parity);
        assert_eq!(
            cpu_trace.entries().len(),
            trace_capacity,
            "quality case {} did not instrument every token/layer route",
            case.id
        );
        let mut generated_hasher = Sha256::new();
        for token in &generated {
            generated_hasher.update(token.to_le_bytes());
        }
        eprintln!(
            "real_moe_quality_case: id={}, kind={}, prompt_tokens={}, generated_tokens={}, route_entries={}, ambiguous_boundary_substitutions={}, long_context_route_divergences={}, generated_token_sha256={:x}, wall_ms={:.3}",
            case.id,
            case.kind,
            case.tokens.len(),
            generated.len(),
            cpu_trace.entries().len(),
            case_route_parity.ambiguous_boundary_substitutions,
            case_route_parity.long_context_route_divergences,
            generated_hasher.finalize(),
            case_started.elapsed().as_secs_f64() * 1000.0
        );
    }

    assert!(
        metric_count > 0,
        "quality suite produced no logit comparisons"
    );
    let mean_cosine = cosine_sum / metric_count as f64;
    assert!(
        mean_cosine >= 0.99999,
        "quality-suite mean cosine {mean_cosine} is below 0.99999"
    );
    assert!(
        route_parity
            .ambiguous_boundary_substitutions
            .checked_mul(MOE_AMBIGUOUS_SUBSTITUTION_RATE_DENOMINATOR)
            .is_some_and(|scaled| scaled <= route_parity.route_entries),
        "ambiguous CPU/CUDA boundary substitutions {} exceed 0.01% of {} traced routes",
        route_parity.ambiguous_boundary_substitutions,
        route_parity.route_entries
    );
    assert!(
        route_parity
            .long_context_route_divergences
            .checked_mul(MOE_LONG_CONTEXT_ROUTE_AGREEMENT_DENOMINATOR)
            .is_some_and(|scaled| scaled <= route_parity.long_context_route_entries),
        "long-context CPU/CUDA route divergences {} exceed 1% of {} traced long-context routes",
        route_parity.long_context_route_divergences,
        route_parity.long_context_route_entries
    );
    let ambiguous_substitution_rate =
        route_parity.ambiguous_boundary_substitutions as f64 / route_parity.route_entries as f64;
    let long_context_route_divergence_rate = if route_parity.long_context_route_entries == 0 {
        0.0
    } else {
        route_parity.long_context_route_divergences as f64
            / route_parity.long_context_route_entries as f64
    };
    let transfers_after = cuda
        .gpu_transfer_stats()
        .expect("quality CUDA transfer counters should exist");
    let transfer_delta = transfers_after.saturating_sub(&transfers_before);
    eprintln!(
        "real_moe_quality_suite: profile={profile:?}, cases={}, logit_comparisons={metric_count}, mean_cosine={mean_cosine:.9}, worst_cosine={worst_cosine:.9}, worst_max_abs={worst_max_abs:.9}, worst_normalized_rms={worst_normalized_rms:.9}, route_entries={}, ambiguous_boundary_substitutions={}, ambiguous_substitution_rate={ambiguous_substitution_rate:.9}, long_context_route_entries={}, long_context_route_divergences={}, long_context_route_divergence_rate={long_context_route_divergence_rate:.9}, max_long_context_symmetric_difference={}, max_cpu_boundary_gap={:.9}, max_cuda_boundary_gap={:.9}, cpu_load_ms={cpu_load_ms:.3}, cuda_load_ms={cuda_load_ms:.3}, suite_wall_ms={:.3}, h2d_bytes={}, d2h_bytes={} ",
        cases.len(),
        route_parity.route_entries,
        route_parity.ambiguous_boundary_substitutions,
        route_parity.long_context_route_entries,
        route_parity.long_context_route_divergences,
        route_parity.max_long_context_symmetric_difference,
        route_parity.max_cpu_boundary_gap,
        route_parity.max_cuda_boundary_gap,
        suite_started.elapsed().as_secs_f64() * 1000.0,
        transfer_delta.host_to_device_bytes,
        transfer_delta.device_to_host_bytes
    );
}

#[test]
fn synthetic_moe_fixture_sha256_is_pinned() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let digest = format!("{:x}", Sha256::digest(&fixture.bytes));
    assert_eq!(digest, SYNTHETIC_MOE_FIXTURE_SHA256);
}

#[test]
fn moe_quality_prompt_suite_is_pinned_and_complete() {
    let digest = format!("{:x}", Sha256::digest(MOE_QUALITY_PROMPTS_JSON.as_bytes()));
    assert_eq!(digest, MOE_QUALITY_PROMPTS_SHA256);
    let suite: MoeQualityPromptSuite = serde_json::from_str(MOE_QUALITY_PROMPTS_JSON)
        .expect("pinned MoE quality prompt suite should parse");
    assert_eq!(suite.schema_version, 1);
    assert_eq!(
        suite.title,
        "XENO RT public MoE CPU/CUDA parity prompt suite"
    );
    assert_eq!(suite.source.origin, "repository-authored synthetic prompts");
    assert_eq!(suite.source.license, "Apache-2.0");
    assert!(!suite.source.private_user_content);
    assert_eq!(
        suite.source.purpose,
        "deterministic implementation parity, not capability scoring"
    );
    assert_eq!(suite.short_prompts.len(), 20);
    assert_eq!(suite.multi_turn_prompts.len(), 10);
    assert_eq!(suite.long_context_prompts.len(), 5);
    assert_eq!(suite.generated_256_prompt_ids.len(), 5);

    let mut ids = std::collections::HashSet::new();
    for prompt in suite
        .short_prompts
        .iter()
        .chain(suite.multi_turn_prompts.iter())
    {
        assert!(!prompt.id.trim().is_empty());
        assert!(!prompt.text.trim().is_empty());
        assert!(
            ids.insert(prompt.id.as_str()),
            "duplicate prompt ID {}",
            prompt.id
        );
    }
    for prompt in &suite.long_context_prompts {
        assert!(!prompt.id.trim().is_empty());
        assert!(!prompt.repeat_text.trim().is_empty());
        assert!(!prompt.question.trim().is_empty());
        assert!(
            ids.insert(prompt.id.as_str()),
            "duplicate prompt ID {}",
            prompt.id
        );
    }
    let short_ids = suite
        .short_prompts
        .iter()
        .map(|prompt| prompt.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    for id in &suite.generated_256_prompt_ids {
        assert!(
            short_ids.contains(id.as_str()),
            "unknown 256-token prompt ID {id}"
        );
    }
}

#[test]
fn canonical_moe_execution_matches_single_and_batch_paths() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    let backend = runtime.backend();
    let tokens = [3, 4, 5, 6];

    let mut sequential = backend.new_session(KvCacheMode::F32, 4);
    backend
        .prepare_session_state(&mut sequential)
        .expect("dense MoE session preparation should succeed");
    let mut expected = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        let mut logits = Vec::new();
        backend
            .forward_token(token, position, &mut sequential, &mut logits)
            .expect("single-token MoE forward should succeed");
        expected.extend_from_slice(&logits);
    }

    let mut batched = backend.new_session(KvCacheMode::F32, 4);
    backend
        .prepare_session_state(&mut batched)
        .expect("dense MoE session preparation should succeed");
    let actual = backend
        .forward_batch_all_logits(&tokens, 0, &mut batched)
        .expect("batched MoE forward should succeed");

    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "logit {index} differs: batch={actual}, sequential={expected}"
        );
    }

    let optimized = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::optimized_cpu(),
    )
    .expect("optimized CPU MoE runtime should load");
    assert_eq!(optimized.moe_status().requested_mode, "cpu");
    assert_eq!(optimized.moe_status().effective_mode, "cpu");
    let optimized_backend = optimized.backend();
    let mut optimized_session = optimized_backend.new_session(KvCacheMode::F32, 4);
    let optimized_logits = optimized_backend
        .forward_batch_all_logits(&tokens, 0, &mut optimized_session)
        .expect("grouped optimized MoE prefill should succeed");
    assert_eq!(optimized_logits.len(), expected.len());
    for (index, (&actual, &expected)) in optimized_logits.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "optimized logit {index} differs: batch={actual}, sequential={expected}"
        );
    }
}

#[test]
fn grouped_cpu_moe_matches_legacy_above_the_rollout_threshold() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_benchmark_fixture()
        .expect("benchmark fixture should build");
    let legacy =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("legacy should load");
    let optimized = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::Cpu,
        MoeRuntimeConfig::optimized_cpu(),
    )
    .expect("optimized should load");
    let tokens = (0..16).map(|index| 3 + index % 24).collect::<Vec<_>>();

    let mut legacy_session = legacy.backend().new_session(KvCacheMode::F32, 4);
    let expected = legacy
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut legacy_session)
        .expect("legacy batch should run");
    let mut optimized_session = optimized.backend().new_session(KvCacheMode::F32, 4);
    let actual = optimized
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut optimized_session)
        .expect("grouped batch should run");

    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "grouped logit {index} differs: optimized={actual}, legacy={expected}"
        );
    }
    let status = optimized.moe_status();
    assert_eq!(status.grouped_batches, 2);
    assert_eq!(status.grouped_tokens, 32);
    assert_eq!(status.routed_tokens, 32);
    assert_eq!(status.selected_expert_calls, 64);
    assert_eq!(status.expert_call_counts.iter().sum::<u64>(), 64);
    assert_eq!(status.worker_failures, 0);
}

#[test]
fn canonical_qwen3moe_packed_expert_tensor_names_load_and_execute() {
    let (fixture, _) =
        common::build_synthetic_qwen3moe_packed_fixture().expect("packed fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    assert_eq!(runtime.model_architecture(), "qwen3moe");
    assert!(runtime.backend().config().is_moe());
    let mut session = runtime.backend().new_session(KvCacheMode::F32, 4);
    let mut logits = Vec::new();
    runtime
        .backend()
        .forward_token(3, 0, &mut session, &mut logits)
        .expect("packed expert model should execute");
    assert_eq!(logits.len(), 32);
    assert!(logits.iter().all(|value| value.is_finite()));
}

#[test]
fn sigmoid_gated_shared_expert_matches_single_and_batch_execution() {
    let (fixture, _) = common::build_synthetic_qwen3moe_shared_expert_fixture()
        .expect("shared-expert fixture should build");
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    assert_eq!(
        runtime.backend().config().expert_shared_feed_forward_length,
        Some(12)
    );
    let tokens = [3, 7, 11, 15];

    let mut sequential = runtime.backend().new_session(KvCacheMode::F32, 4);
    let mut expected = Vec::new();
    for (position, token) in tokens.into_iter().enumerate() {
        let mut logits = Vec::new();
        runtime
            .backend()
            .forward_token(token, position, &mut sequential, &mut logits)
            .expect("shared-expert token should execute");
        expected.extend_from_slice(&logits);
    }

    let mut batched = runtime.backend().new_session(KvCacheMode::F32, 4);
    let actual = runtime
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut batched)
        .expect("shared-expert batch should execute");
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "shared-expert logit {index} differs: batch={actual}, sequential={expected}"
        );
    }
}

#[test]
fn qwen35_hybrid_moe_fixture_executes_with_transactional_cpu_state() {
    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_moe_fixture().expect("fixture should build");
    assert_eq!(
        format!("{:x}", Sha256::digest(&fixture.bytes)),
        SYNTHETIC_QWEN35_HYBRID_MOE_FIXTURE_SHA256
    );
    let runtime =
        Runtime::load_with_backend(fixture.path(), BackendKind::Cpu).expect("runtime should load");
    assert_eq!(runtime.model_architecture(), "qwen3_5_moe");
    let backend = runtime.backend();
    let tokens = [3, 4, 5, 6];
    let mut session = backend.new_session(KvCacheMode::F32, tokens.len());
    backend
        .prepare_session_state(&mut session)
        .expect("hybrid-MoE recurrent state should prepare");
    let logits = backend
        .forward_batch_all_logits(&tokens, 0, &mut session)
        .expect("hybrid-MoE CPU reference should execute");
    assert_eq!(logits.len(), tokens.len() * 32);
    let state = backend
        .save_state(&session)
        .expect("hybrid-MoE state should save")
        .expect("hybrid-MoE model should expose recurrent state");
    assert_eq!(state.position(), tokens.len() as u64);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_QWEN3_MOE_GGUF plus a CUDA-capable device and driver"]
fn cuda_real_qwen3_moe_short_decode_matches_cpu() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN3_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN3_MOE_GGUF to run real Qwen3 MoE parity");
        return;
    };
    run_real_moe_cpu_cuda_parity(&model_path, false);
}

#[cfg(all(feature = "cuda", feature = "moe-route-trace"))]
#[test]
#[ignore = "requires XRT_REAL_QWEN3_MOE_GGUF, an explicit quality profile, and a CUDA-capable device and driver"]
fn cuda_real_qwen3_moe_quality_suite_matches_cpu() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN3_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN3_MOE_GGUF to run the real Qwen3 MoE quality suite");
        return;
    };
    run_real_qwen3_moe_quality_suite(&model_path, MoeQualityProfile::from_env());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_QWEN3_MOE_GGUF, a SHA-pinned XRT_REAL_MOE_PERPLEXITY_TEXT corpus, and a CUDA-capable device unless CPU-only"]
fn cuda_real_qwen3_moe_wikitext_perplexity_matches_cpu() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN3_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN3_MOE_GGUF to run real Qwen3 MoE perplexity");
        return;
    };
    let corpus_path = std::env::var_os("XRT_REAL_MOE_PERPLEXITY_TEXT")
        .map(std::path::PathBuf::from)
        .expect("set XRT_REAL_MOE_PERPLEXITY_TEXT to the pinned UTF-8 corpus");
    run_real_qwen3_moe_perplexity(&model_path, &corpus_path);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_QWEN3_MOE_GGUF, a SHA-pinned XRT_REAL_MOE_GSM8K_FIXTURE, and a CUDA-capable device"]
fn cuda_real_qwen3_moe_gsm8k_is_non_inferior_to_cpu() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN3_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN3_MOE_GGUF to run real Qwen3 MoE GSM8K");
        return;
    };
    let fixture_path = std::env::var_os("XRT_REAL_MOE_GSM8K_FIXTURE")
        .map(std::path::PathBuf::from)
        .expect("set XRT_REAL_MOE_GSM8K_FIXTURE to the pinned JSONL projection");
    run_real_qwen3_moe_gsm8k_task(&model_path, &fixture_path);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_QWEN35_MOE_GGUF plus a CUDA-capable device and driver"]
fn cuda_real_qwen35_hybrid_moe_short_decode_matches_cpu_and_state() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN35_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN35_MOE_GGUF to run real Qwen3.5 hybrid-MoE parity");
        return;
    };
    run_real_moe_cpu_cuda_parity(&model_path, true);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires XRT_REAL_QWEN35_MOE_GGUF"]
fn inspect_real_qwen35_hybrid_moe_tensor_profile() {
    let Some(model_path) =
        std::env::var_os("XRT_REAL_QWEN35_MOE_GGUF").map(std::path::PathBuf::from)
    else {
        eprintln!("set XRT_REAL_QWEN35_MOE_GGUF to inspect the real Qwen3.5 fixture");
        return;
    };
    let gguf = GgufFile::open(&model_path).expect("real Qwen3.5 GGUF should parse");
    let config = LlamaConfig::from_gguf(&gguf).expect("real Qwen3.5 config should parse");
    let mut dtype_counts = std::collections::BTreeMap::<String, usize>::new();
    let mut dtype_bytes = std::collections::BTreeMap::<String, u64>::new();
    let mut recurrent_projection_dtypes = std::collections::BTreeMap::<String, usize>::new();
    let mut mxfp4_tensors = Vec::new();
    let mut q5_k_tensors = Vec::new();
    let mut q6_k_tensors = Vec::new();
    for tensor in gguf.tensor_infos() {
        let dtype = format!("{:?}", tensor.dtype);
        *dtype_counts.entry(dtype.clone()).or_default() += 1;
        *dtype_bytes.entry(dtype).or_default() += tensor.nbytes as u64;
        if tensor.dtype == xrt_core::DType::MXFP4 {
            mxfp4_tensors.push(format!(
                "{}:{:?}:{}",
                tensor.name, tensor.dimensions, tensor.nbytes
            ));
        }
        if tensor.dtype == xrt_core::DType::Q5_K {
            q5_k_tensors.push(format!("{}:{:?}", tensor.name, tensor.dimensions));
        }
        if tensor.dtype == xrt_core::DType::Q6_K {
            q6_k_tensors.push(format!("{}:{:?}", tensor.name, tensor.dimensions));
        }
        if tensor.name.ends_with(".attn_qkv.weight") {
            *recurrent_projection_dtypes
                .entry(format!("{:?}", tensor.dtype))
                .or_default() += 1;
        }
    }
    eprintln!(
        "real_qwen35_tensor_profile: path={}, architecture={}, blocks={}, experts={:?}, selected_experts={:?}, hybrid={}, tensor_count={}, dtype_counts={dtype_counts:?}, dtype_bytes={dtype_bytes:?}, recurrent_projection_dtypes={recurrent_projection_dtypes:?}, layer37_attn_qkv={:?}, mxfp4_tensors={mxfp4_tensors:?}, q5_k_tensors={q5_k_tensors:?}, q6_k_tensors={q6_k_tensors:?}",
        model_path.display(),
        config.architecture,
        config.block_count,
        config.expert_count,
        config.expert_used_count,
        config.is_hybrid(),
        gguf.tensor_infos().len(),
        gguf.tensor_info("blk.37.attn_qkv.weight")
            .map(|tensor| (tensor.dtype, tensor.dimensions.clone(), tensor.nbytes)),
    );
    assert!(config.is_moe(), "fixture must be an MoE model");
    assert!(config.is_hybrid(), "fixture must be a hybrid model");
    assert!(
        !mxfp4_tensors.is_empty(),
        "Q4_K_S fixture should contain MXFP4 tensors"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_qwen35_hybrid_moe_combines_recurrent_state_and_exact_expert_placement() {
    use xrt_runtime::backend::CudaLayerKvStore;

    let (fixture, _) =
        common::build_synthetic_qwen35_hybrid_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let adaptive_error = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(8 * 1024),
            placement: MoePlacementPolicy::Adaptive,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig::default(),
    )
    .err()
    .expect("hybrid recurrent adaptive placement should remain explicitly gated");
    assert!(adaptive_error.to_string().contains("adaptive placement"));
    let layerwise_error = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(8 * 1024),
            layerwise_prefill: true,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig::default(),
    )
    .err()
    .expect("hybrid recurrent layerwise prefill should remain explicitly gated");
    assert!(layerwise_error.to_string().contains("layerwise prefill"));
    let hybrid = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(8 * 1024),
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("Qwen3.5 hybrid-MoE CUDA runtime should load");

    let tokens = [3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6];
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut hybrid_session = hybrid.backend().new_session(KvCacheMode::F32, tokens.len());
    cpu.backend()
        .prepare_session_state(&mut cpu_session)
        .expect("CPU recurrent state should prepare");
    hybrid
        .backend()
        .prepare_session_state(&mut hybrid_session)
        .expect("CUDA recurrent state should prepare");
    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU Qwen3.5 hybrid-MoE reference should execute");
    let actual = hybrid
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut hybrid_session)
        .expect("CUDA Qwen3.5 hybrid-MoE path should execute");
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-3,
        "Qwen3.5 hybrid-MoE CUDA logits diverged from CPU: max_abs={max_abs}"
    );
    for token_index in 0..tokens.len() {
        let offset = token_index * 32;
        let expected_token = expected[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
        let actual_token = actual[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
        assert_eq!(
            actual_token, expected_token,
            "Qwen3.5 hybrid-MoE changed the top token at batch index {token_index}"
        );
    }

    let cpu_state = cpu
        .backend()
        .save_state(&cpu_session)
        .expect("CPU recurrent state should save")
        .expect("CPU recurrent state should exist");
    let cuda_state = hybrid
        .backend()
        .save_state(&hybrid_session)
        .expect("CUDA recurrent state should save")
        .expect("CUDA recurrent state should exist");
    assert_eq!(cuda_state.position(), tokens.len() as u64);
    assert_eq!(cuda_state.descriptor(), cpu_state.descriptor());
    let mut max_conv_error = 0.0f32;
    let mut max_recurrent_error = 0.0f32;
    for (layer, (actual, expected)) in cuda_state
        .layers()
        .iter()
        .zip(cpu_state.layers())
        .enumerate()
    {
        match (actual, expected) {
            (Some(actual), Some(expected)) => {
                let conv_error = actual
                    .conv_state_f32()
                    .iter()
                    .zip(expected.conv_state_f32())
                    .map(|(&actual, &expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                let recurrent_error = actual
                    .recurrent_state_f32()
                    .iter()
                    .zip(expected.recurrent_state_f32())
                    .map(|(&actual, &expected)| (actual - expected).abs())
                    .fold(0.0f32, f32::max);
                max_conv_error = max_conv_error.max(conv_error);
                max_recurrent_error = max_recurrent_error.max(recurrent_error);
                assert!(
                    conv_error <= 5e-4,
                    "layer {layer} convolution state diverged: {conv_error}"
                );
                assert!(
                    recurrent_error <= 2e-3,
                    "layer {layer} recurrent state diverged: {recurrent_error}"
                );
            }
            (None, None) => {}
            _ => panic!("layer {layer} recurrent-state presence differs"),
        }
    }

    let mut cpu_decode_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut cuda_decode_session = hybrid.backend().new_session(KvCacheMode::F32, tokens.len());
    cpu.backend()
        .prepare_session_state(&mut cpu_decode_session)
        .expect("CPU decode state should prepare");
    hybrid
        .backend()
        .prepare_session_state(&mut cuda_decode_session)
        .expect("CUDA decode state should prepare");
    for (position, &token) in tokens.iter().enumerate() {
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        cpu.backend()
            .forward_token(token, position, &mut cpu_decode_session, &mut expected)
            .expect("CPU hybrid-MoE decode token should execute");
        hybrid
            .backend()
            .forward_token(token, position, &mut cuda_decode_session, &mut actual)
            .expect("CUDA hybrid-MoE decode token should execute");
        let decode_max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            decode_max_abs <= 2e-3,
            "Qwen3.5 hybrid-MoE decode diverged at position {position}: {decode_max_abs}"
        );
    }

    let status = hybrid.moe_status();
    assert_eq!(status.effective_mode, "hybrid");
    assert!(status.cpu_expert_calls > 0);
    assert!(status.gpu_expert_calls > 0);
    assert_eq!(status.coordinator_failures, 0);
    assert!(status.graph_captures > 0);
    assert!(status.graph_replays > 0);
    assert_eq!(status.graph_fallbacks, 0);
    let resources = hybrid.gpu_resource_status();
    assert!(resources.arena_allocations.expert_weight_bytes > 0);
    assert!(resources.arena_allocations.recurrent_state_bytes > 0);
    eprintln!(
        "qwen35_hybrid_moe_diagnostic: max_logit_abs={max_abs}, max_conv_abs={max_conv_error}, max_recurrent_abs={max_recurrent_error}, cpu_expert_calls={}, gpu_expert_calls={}, graph_captures={}, graph_replays={}, expert_bytes={}, recurrent_bytes={}",
        status.cpu_expert_calls,
        status.gpu_expert_calls,
        status.graph_captures,
        status.graph_replays,
        resources.arena_allocations.expert_weight_bytes,
        resources.arena_allocations.recurrent_state_bytes,
    );

    let mut failed_session = hybrid.backend().new_session(KvCacheMode::F32, 4);
    hybrid
        .backend()
        .prepare_session_state(&mut failed_session)
        .expect("failure-probe recurrent state should prepare");
    let mut first_logits = Vec::new();
    hybrid
        .backend()
        .forward_token(3, 0, &mut failed_session, &mut first_logits)
        .expect("failure-probe first token should execute");
    let before_failure = hybrid
        .backend()
        .save_state(&failed_session)
        .expect("pre-failure state should save")
        .expect("pre-failure state should exist");
    if let xrt_runtime::backend::BackendSession::Cuda {
        device,
        layer_caches,
        ..
    } = &mut failed_session
    {
        let capacity = match &layer_caches[3] {
            CudaLayerKvStore::F32(cache) => cache.capacity(),
            _ => panic!("Qwen3.5 hybrid-MoE test session should use f32 KV"),
        };
        let mut invalid = device
            .alloc_paged_layer_kv_cache(capacity, 1, 4)
            .expect("invalid test cache should allocate");
        let key = device
            .upload_f32(&[0.0])
            .expect("invalid test key should upload");
        let value = device
            .upload_f32(&[0.0])
            .expect("invalid test value should upload");
        device
            .append_layer_kv(&mut invalid, &key, &value)
            .expect("invalid test cache should reach position one");
        layer_caches[3] = CudaLayerKvStore::F32(invalid);
    } else {
        panic!("CUDA runtime returned a non-CUDA session");
    }
    let failure = hybrid
        .backend()
        .forward_token(4, 1, &mut failed_session, &mut first_logits)
        .expect_err("invalid full-attention cache should fail after pending recurrent work");
    assert!(
        failure.to_string().contains("width") || failure.to_string().contains("length"),
        "unexpected injected hybrid-MoE CUDA error: {failure}"
    );
    let after_failure = hybrid
        .backend()
        .save_state(&failed_session)
        .expect("failed state should remain snapshotable")
        .expect("failed state should remain present");
    assert_eq!(after_failure, before_failure);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_hybrid_moe_matches_cpu_and_reports_resident_experts() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let hybrid = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("hybrid CUDA runtime should load");

    let status = hybrid.moe_status();
    assert_eq!(status.requested_mode, "hybrid");
    assert_eq!(status.effective_mode, "hybrid");
    assert_eq!(status.placement_generation, 1);
    assert!(status.gpu_expert_slots > 0);
    assert!(status.gpu_expert_slots < 8);
    assert!(status.gpu_expert_bytes > 0);
    assert!(status.gpu_expert_bytes <= 4 * 1024);

    let resources = hybrid.gpu_resource_status();
    assert_eq!(
        resources.arena_allocations.expert_weight_bytes,
        status.gpu_expert_bytes
    );
    assert!(resources.arena_allocations.model_weight_bytes > 0);
    assert_eq!(
        resources.arena_allocated_bytes,
        resources
            .arena_allocations
            .model_weight_bytes
            .saturating_add(resources.arena_allocations.expert_weight_bytes)
    );

    let tokens = [3, 4, 5, 6];
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut hybrid_session = hybrid.backend().new_session(KvCacheMode::F32, tokens.len());
    let transfers_before = hybrid
        .gpu_transfer_stats()
        .expect("hybrid CUDA transfer counters should exist");
    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU MoE reference should execute");
    let actual = hybrid
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut hybrid_session)
        .expect("hybrid CUDA MoE should execute");
    let transfers_after = hybrid
        .gpu_transfer_stats()
        .expect("hybrid CUDA transfer counters should exist");
    let decode_h2d_bytes = transfers_after
        .host_to_device_bytes
        .saturating_sub(transfers_before.host_to_device_bytes);
    let maximum_cpu_result_bytes = (tokens.len() * 2 * 2 * 8 * std::mem::size_of::<f32>()) as u64;
    assert!(
        decode_h2d_bytes <= maximum_cpu_result_bytes,
        "decode transferred {decode_h2d_bytes} H2D bytes, exceeding the activation/result bound {maximum_cpu_result_bytes}; expert weights may have moved per token"
    );
    let live_resources = hybrid.gpu_resource_status();
    assert!(live_resources.arena_allocations.staging_bytes > 0);
    assert_eq!(
        live_resources.arena_allocations.staging_bytes,
        hybrid_session.cuda_staging_allocated_bytes()
    );
    let live_moe = hybrid.moe_status();
    assert!(live_moe.cpu_expert_calls > 0);
    assert!(live_moe.gpu_expert_calls > 0);
    assert_eq!(
        live_moe.cpu_expert_calls + live_moe.gpu_expert_calls,
        (tokens.len() * 2 * 2) as u64
    );
    assert_eq!(live_moe.gpu_placement_hits, live_moe.gpu_expert_calls);
    assert_eq!(live_moe.gpu_placement_misses, live_moe.cpu_expert_calls);
    assert!(live_moe.activation_d2h_bytes > 0);
    assert_eq!(
        live_moe.result_h2d_bytes,
        live_moe.cpu_expert_calls * 8 * std::mem::size_of::<f32>() as u64
    );
    assert_eq!(live_moe.coordinator_failures, 0);

    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-4,
        "hybrid CUDA logits diverged from CPU: max_abs={max_abs}"
    );
    for token_index in 0..tokens.len() {
        let offset = token_index * 32;
        let expected_token = expected[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
        let actual_token = actual[offset..offset + 32]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index);
        assert_eq!(
            actual_token, expected_token,
            "hybrid CUDA changed the top token at batch index {token_index}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_fixed_placement_moe_expert_graphs_replay_for_gpu_and_hybrid_modes() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let gpu = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Gpu,
            gpu_expert_budget_bytes: Some(64 * 1024),
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Auto,
            ..GpuResourceConfig::default()
        },
    )
    .expect("full-GPU MoE runtime should load");
    let hybrid = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("hybrid MoE runtime should load");

    let tokens = [3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6];
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut gpu_session = gpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut hybrid_session = hybrid.backend().new_session(KvCacheMode::F32, tokens.len());
    for (position, token) in tokens.into_iter().enumerate() {
        let mut expected = Vec::new();
        let mut gpu_actual = Vec::new();
        let mut hybrid_actual = Vec::new();
        cpu.backend()
            .forward_token(token, position, &mut cpu_session, &mut expected)
            .expect("CPU MoE reference should execute");
        gpu.backend()
            .forward_token(token, position, &mut gpu_session, &mut gpu_actual)
            .expect("full-GPU graph path should execute");
        hybrid
            .backend()
            .forward_token(token, position, &mut hybrid_session, &mut hybrid_actual)
            .expect("hybrid GPU-subgraph path should execute");
        for (label, actual) in [("gpu", &gpu_actual), ("hybrid", &hybrid_actual)] {
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_abs <= 2e-4,
                "{label} MoE graph logits diverged at position {position}: max_abs={max_abs}"
            );
        }
    }

    for (label, runtime, session) in [
        ("gpu", &gpu, &gpu_session),
        ("hybrid", &hybrid, &hybrid_session),
    ] {
        let status = runtime.moe_status();
        assert!(
            status.graph_captures > 0,
            "{label} mode did not capture an expert graph"
        );
        assert!(
            status.graph_replays > 0,
            "{label} mode did not replay an expert graph"
        );
        assert_eq!(status.graph_fallbacks, 0, "{label} graph path fell back");
        assert_eq!(session.cuda_graph_capture_status(), Some("captured"));
        assert_eq!(session.cuda_graph_last_error(), None);
        assert!(
            runtime.gpu_resource_status().arena_allocations.graph_bytes > 0,
            "{label} graph executables were not charged to the central arena"
        );
    }
    assert!(hybrid.moe_status().cpu_expert_calls > 0);
    assert!(hybrid.moe_status().gpu_expert_calls > 0);

    let hybrid_auto = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Auto,
            ..GpuResourceConfig::default()
        },
    )
    .expect("auto-gated hybrid MoE runtime should load");
    let mut auto_session = hybrid_auto.backend().new_session(KvCacheMode::F32, 4);
    let mut auto_logits = Vec::new();
    hybrid_auto
        .backend()
        .forward_token(3, 0, &mut auto_session, &mut auto_logits)
        .expect("auto-gated hybrid eager token should execute");
    let auto_status = hybrid_auto.moe_status();
    assert_eq!(auto_status.graph_captures, 0);
    assert_eq!(auto_status.graph_replays, 0);
    assert!(auto_status.graph_eager_expert_calls > 0);
    assert_eq!(
        auto_session.cuda_graph_capture_status(),
        Some("not-captured")
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_profiled_moe_manifest_loads_before_upload_and_matches_cpu() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let gguf = GgufFile::open(fixture.path()).expect("fixture GGUF should parse");
    let config = LlamaConfig::from_gguf(&gguf).expect("fixture config should parse");
    let config_sha256 = moe_config_sha256(&config);
    let manifest_root = tempfile::tempdir().expect("manifest tempdir should exist");
    let manifest_path = manifest_root.path().join("placement.json");
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 1,\n",
            "  \"model_sha256\": \"{}\",\n",
            "  \"config_sha256\": \"{}\",\n",
            "  \"architecture\": \"qwen3\",\n",
            "  \"quantization\": \"f32\",\n",
            "  \"layer_count\": 2,\n",
            "  \"expert_count\": 4,\n",
            "  \"gpu_expert_budget_bytes\": 4096,\n",
            "  \"expert_bytes\": 3072,\n",
            "  \"layers\": [\n",
            "    {{\"layer_index\": 1, \"gpu_experts\": [1]}},\n",
            "    {{\"layer_index\": 0, \"gpu_experts\": [3]}}\n",
            "  ]\n",
            "}}\n"
        ),
        SYNTHETIC_MOE_FIXTURE_SHA256, config_sha256
    );
    fs::write(&manifest_path, manifest.as_bytes()).expect("manifest should be written");
    let manifest_sha256 = format!("{:x}", Sha256::digest(manifest.as_bytes()));

    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let profiled = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4096),
            placement: MoePlacementPolicy::Profiled,
            placement_manifest: Some(manifest_path),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("validated profiled runtime should load");

    let status = profiled.moe_status();
    assert_eq!(status.placement, "profiled");
    assert_eq!(status.placement_generation, 1);
    assert_eq!(status.gpu_expert_slots, 2);
    assert_eq!(status.gpu_expert_bytes, 3072);
    assert_eq!(
        status.placement_manifest_sha256.as_deref(),
        Some(manifest_sha256.as_str())
    );

    let tokens = [3, 4, 5, 6];
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let mut profiled_session = profiled
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU reference should execute");
    let actual = profiled
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut profiled_session)
        .expect("profiled CUDA execution should succeed");
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-4,
        "profiled CUDA logits diverged from CPU: max_abs={max_abs}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_adaptive_moe_publishes_only_at_request_boundary_and_preserves_logits() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let adaptive = Runtime::load_with_backend_configs(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            placement: MoePlacementPolicy::Adaptive,
            placement_update_tokens: 2,
            ..MoeRuntimeConfig::default()
        },
        GpuResourceConfig {
            cuda_graph_mode: CudaGraphMode::Enabled,
            ..GpuResourceConfig::default()
        },
    )
    .expect("adaptive CUDA runtime should load");
    let tokens = [3, 4, 5, 6];
    let mut first_session = adaptive
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let mut observation_logits = Vec::new();
    for (position, token) in tokens.into_iter().enumerate() {
        adaptive
            .backend()
            .forward_token(token, position, &mut first_session, &mut observation_logits)
            .expect("first adaptive observation token should execute");
    }
    let observed = adaptive.moe_status();
    assert_eq!(observed.placement, "adaptive");
    assert_eq!(observed.placement_generation, 1);
    assert_eq!(observed.placement_updates, 0);
    assert!(observed.graph_captures > 0);

    adaptive
        .backend()
        .prepare_request()
        .expect("safe request-boundary update should succeed");
    let updated = adaptive.moe_status();
    assert_eq!(updated.placement_generation, 2);
    assert_eq!(updated.placement_evaluations, 1);
    assert_eq!(updated.placement_updates, 1);
    assert!(updated.placement_moves > 0);
    assert!(updated.placement_moves <= 4);
    assert_eq!(
        updated.placement_upload_bytes,
        updated.placement_moves * 1536
    );
    assert!(updated.placement_last_update_micros > 0);

    let mut stale_epoch_was_rejected = false;
    for (offset, token) in [3, 4, 5, 6].into_iter().enumerate() {
        let before = adaptive.moe_status();
        adaptive
            .backend()
            .forward_token(
                token,
                tokens.len() + offset,
                &mut first_session,
                &mut observation_logits,
            )
            .expect("post-placement-update token should execute");
        let after = adaptive.moe_status();
        if after.gpu_expert_calls > before.gpu_expert_calls {
            assert_eq!(
                after.graph_replays, before.graph_replays,
                "the first GPU expert call in a new placement epoch must not replay an old graph"
            );
            assert!(
                after.graph_captures > before.graph_captures,
                "the new placement epoch should capture a replacement expert graph"
            );
            stale_epoch_was_rejected = true;
            break;
        }
    }
    assert!(
        stale_epoch_was_rejected,
        "the post-update probe never selected a resident expert"
    );

    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU reference should execute");
    let mut updated_session = adaptive
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let actual = adaptive
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut updated_session)
        .expect("updated adaptive placement should execute");
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-4,
        "adaptive CUDA logits diverged after placement update: max_abs={max_abs}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_layerwise_moe_prefill_double_buffers_cold_experts_and_preserves_logits() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");
    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let layerwise = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(4 * 1024),
            layerwise_prefill: true,
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("layerwise CUDA runtime should load");
    let tokens = [3, 4, 5, 6];
    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, tokens.len());
    let expected = cpu
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut cpu_session)
        .expect("CPU reference should execute");
    let mut layerwise_session = layerwise
        .backend()
        .new_session(KvCacheMode::F32, tokens.len());
    let actual = layerwise
        .backend()
        .forward_batch_all_logits(&tokens, 0, &mut layerwise_session)
        .expect("layerwise CUDA prefill should execute");

    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-4,
        "layerwise CUDA logits diverged from CPU: max_abs={max_abs}"
    );
    let status = layerwise.moe_status();
    assert!(status.layerwise_prefill_enabled);
    assert_eq!(status.layerwise_prefill_batches, 1);
    assert_eq!(status.layerwise_prefill_tokens, tokens.len() as u64);
    assert!(status.layerwise_prefill_weight_upload_bytes > 0);
    assert_eq!(
        status.layerwise_prefill_repack_bytes,
        status.layerwise_prefill_weight_upload_bytes
    );
    assert!(status.layerwise_prefill_micros > 0);
    assert_eq!(status.placement_generation, 1);
    assert_eq!(status.gpu_expert_slots, 2);
    assert_eq!(status.gpu_expert_bytes, 3072);

    let resources = layerwise.gpu_resource_status();
    assert_eq!(
        resources.arena_allocations.staging_bytes,
        layerwise_session.cuda_staging_allocated_bytes(),
        "temporary layerwise double-buffer reservation must be released after prefill"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn cuda_moe_budget_admission_and_full_gpu_semantics_are_exact() {
    let (fixture, _) = common::build_synthetic_qwen3_moe_fixture().expect("fixture should build");

    let hybrid_error = match Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Hybrid,
            gpu_expert_budget_bytes: Some(1),
            ..MoeRuntimeConfig::default()
        },
    ) {
        Ok(_) => panic!("one byte must not admit one expert per layer"),
        Err(error) => error.to_string(),
    };
    assert!(hybrid_error.contains("cannot place one expert"));

    let gpu_error = match Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Gpu,
            gpu_expert_budget_bytes: Some(4 * 1024),
            ..MoeRuntimeConfig::default()
        },
    ) {
        Ok(_) => panic!("partial residency must not satisfy explicit gpu mode"),
        Err(error) => error.to_string(),
    };
    assert!(gpu_error.contains("requires every expert resident"));

    let cpu = Runtime::load_with_backend(fixture.path(), BackendKind::Cpu)
        .expect("CPU runtime should load");
    let gpu = Runtime::load_with_backend_and_moe_config(
        fixture.path(),
        BackendKind::CudaResident,
        MoeRuntimeConfig {
            acceleration: MoeAcceleration::Gpu,
            gpu_expert_budget_bytes: Some(64 * 1024),
            ..MoeRuntimeConfig::default()
        },
    )
    .expect("full-GPU MoE runtime should load");
    let status = gpu.moe_status();
    assert_eq!(status.effective_mode, "gpu");
    assert_eq!(status.gpu_expert_slots, 8);
    assert!(status.gpu_expert_bytes > 4 * 1024);
    assert!(status.gpu_expert_bytes <= 64 * 1024);

    let mut cpu_session = cpu.backend().new_session(KvCacheMode::F32, 4);
    let mut gpu_session = gpu.backend().new_session(KvCacheMode::F32, 4);
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    cpu.backend()
        .forward_token(3, 0, &mut cpu_session, &mut expected)
        .expect("CPU reference should execute");
    gpu.backend()
        .forward_token(3, 0, &mut gpu_session, &mut actual)
        .expect("full-GPU MoE should execute");
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2e-4,
        "full-GPU MoE logits diverged from CPU: max_abs={max_abs}"
    );
}
