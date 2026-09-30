//! Teacher-forced LM comparison: CPU vs CUDA(TF32) vs CUDA(FP32), all fed the
//! SAME voice conditioning and the SAME speech tokens (a CPU take). Reports,
//! per backend, the largest logit difference and how often the argmax and the
//! sampling-relevant candidate set (min-p 0.05 after temperature 0.8) differ.
//!
//! `lmcmp <ref.wav> <text>`
use xrt_audio::chatterbox::{ChatterboxModel, CudaPrecision, Device, GenerateParams, ModelPaths};
use xrt_audio::sampling::SamplingParams;

fn candidate_set(l: &[f32], temp: f32, min_p: f32) -> Vec<usize> {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = l.iter().map(|x| ((x - m) / temp).exp()).collect();
    let top = p.iter().copied().fold(0.0, f32::max);
    (0..p.len()).filter(|&i| p[i] >= min_p * top).collect()
}

fn argmax(l: &[f32]) -> usize {
    l.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (r, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let text = xrt_audio::tokenizer::punc_norm(&a[1]);
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
    // One conditioning shared by every backend, so only the LM differs.
    let voice = cpu.prepare_voice(&r, rate)?;
    let take = cpu.generate(&text, "en", &voice, &p)?;
    let forced = take.token_ids.clone();
    println!("forced tokens: {}", forced.len());
    let t_cpu = cpu.trace_logits(&text, "en", &voice, &p, &forced)?;
    drop(cpu);

    for (label, prec) in [
        ("cuda TF32", CudaPrecision::Tf32),
        ("cuda FP32", CudaPrecision::Fp32),
    ] {
        let gpu = ChatterboxModel::load_with(&paths, Device::Cuda(0), prec)?;
        let t = gpu.trace_logits(&text, "en", &voice, &p, &forced)?;
        let (mut maxd, mut worst) = (0f32, 0usize);
        let (mut arg_mis, mut set_mis) = (0usize, 0usize);
        let mut first_mis = None;
        for (k, (c, g)) in t_cpu.iter().zip(&t).enumerate() {
            let d = max_abs_diff(c, g);
            if d > maxd {
                maxd = d;
                worst = k;
            }
            if argmax(c) != argmax(g) {
                arg_mis += 1;
                first_mis.get_or_insert(k);
            }
            if candidate_set(c, 0.8, 0.05) != candidate_set(g, 0.8, 0.05) {
                set_mis += 1;
            }
        }
        let early = t_cpu
            .iter()
            .zip(&t)
            .take(20)
            .map(|(c, g)| max_abs_diff(c, g))
            .fold(0.0, f32::max);
        println!(
            "{label}: max|dlogit| {maxd:.4} (step {worst})  first-20 {early:.4}  argmax differs {arg_mis}/{}  first at {first_mis:?}  candidate-set differs {set_mis}",
            t.len()
        );
    }
    Ok(())
}
