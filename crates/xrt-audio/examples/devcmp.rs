//! Isolate which stage differs between CPU and CUDA.
//! `devcmp <ref.wav> <text> <out_dir>`: generates ONE take on CPU, then renders
//! the SAME speech tokens with the CPU decoder and the CUDA decoder, and runs
//! the CUDA LM on the same text + seed to compare token streams.
use xrt_audio::chatterbox::{ChatterboxModel, Device, GenerateParams, ModelPaths};
use xrt_audio::sampling::SamplingParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (r, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let text = xrt_audio::tokenizer::punc_norm(&a[1]);
    let out = std::path::Path::new(&a[2]);
    let paths = ModelPaths::from_dir(std::path::Path::new(&std::env::var("XRT_AUDIO_MODEL_DIR")?));
    let p = GenerateParams {
        exaggeration: 0.5,
        sampling: SamplingParams {
            cfg_weight: 0.5,
            temperature: 0.8,
            repetition_penalty: 2.0,
            min_p: 0.05,
            top_p: 1.0,
        },
        max_speech_tokens: 4096,
        seed: 42,
    };
    let cpu = ChatterboxModel::load(&paths, Device::Cpu)?;
    let gpu = ChatterboxModel::load(&paths, Device::Cuda(0))?;
    println!("gpu provider: {}", gpu.provider);

    let v_cpu = cpu.prepare_voice(&r, rate)?;
    let v_gpu = gpu.prepare_voice(&r, rate)?;
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    println!("speech_encoder  cond_emb max|Δ| {:.3e}  spk_emb max|Δ| {:.3e}  feat max|Δ| {:.3e}  prompt tokens equal {}",
        diff(v_cpu.cond_emb.as_slice().unwrap(), v_gpu.cond_emb.as_slice().unwrap()),
        diff(v_cpu.speaker_embedding.as_slice().unwrap(), v_gpu.speaker_embedding.as_slice().unwrap()),
        diff(v_cpu.speaker_features.as_slice().unwrap(), v_gpu.speaker_features.as_slice().unwrap()),
        v_cpu.prompt_tokens == v_gpu.prompt_tokens);

    let g_cpu = cpu.generate(&text, "en", &v_cpu, &p)?;
    let g_gpu = gpu.generate(&text, "en", &v_gpu, &p)?;
    let same = g_cpu
        .token_ids
        .iter()
        .zip(&g_gpu.token_ids)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "LM tokens: cpu {} gpu {}  identical prefix {}",
        g_cpu.token_ids.len(),
        g_gpu.token_ids.len(),
        same
    );

    // Same tokens, same voice conditioning, two decoders.
    let d_cpu = cpu.decode(&g_cpu.token_ids, &v_cpu)?;
    let d_gpu = gpu.decode(&g_cpu.token_ids, &v_cpu)?;
    let n = d_cpu.len().min(d_gpu.len());
    let (mut e, mut s) = (0f64, 0f64);
    for i in 0..n {
        e += ((d_cpu[i] - d_gpu[i]) as f64).powi(2);
        s += (d_cpu[i] as f64).powi(2);
    }
    println!(
        "decoder on SAME tokens: len cpu {} gpu {}  SNR {:.1} dB  max|Δ| {:.3}",
        d_cpu.len(),
        d_gpu.len(),
        10.0 * (s / e.max(1e-20)).log10(),
        diff(&d_cpu[..n], &d_gpu[..n])
    );
    for (name, x) in [
        ("A_cpu_lm_cpu_dec", &d_cpu),
        ("B_cpu_lm_gpu_dec", &d_gpu),
        ("C_gpu_lm_gpu_dec", &g_gpu.samples),
    ] {
        std::fs::write(
            out.join(format!("{name}.wav")),
            xrt_audio::audio::write_wav(x, 24_000),
        )?;
    }
    Ok(())
}
