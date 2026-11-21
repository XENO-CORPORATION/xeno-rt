//! Pipeline audit: generate ONE take and write the output of every stage, so
//! any artefact can be attributed to the stage that introduces it.
//! `stages <ref.wav> <text-file> <out-dir>` (preset from XRT_TTS_PRESET).
use xrt_audio::chatterbox::{ChatterboxModel, Device, GenerateParams, ModelPaths};
use xrt_audio::sampling::SamplingParams;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (r, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let text = xrt_audio::tokenizer::punc_norm(&std::fs::read_to_string(&a[1])?);
    let out = std::path::Path::new(&a[2]);
    std::fs::create_dir_all(out)?;
    let mut opts = xrt_audio::SpeechOptions::default();
    if let Ok(p) = std::env::var("XRT_TTS_PRESET") {
        xrt_audio::Preset::parse(&p)
            .ok_or("bad preset")?
            .apply(&mut opts);
    }
    let model = ChatterboxModel::load(&ModelPaths::from_dir(&opts.model_dir), Device::Cuda(0))?;
    let voice = model.prepare_voice(&r, rate)?;
    let p = GenerateParams {
        exaggeration: opts.exaggeration,
        sampling: SamplingParams {
            cfg_weight: opts.cfg_weight,
            temperature: opts.temperature,
            repetition_penalty: opts.repetition_penalty,
            min_p: opts.min_p,
            top_p: opts.top_p,
        },
        max_speech_tokens: opts.max_speech_tokens,
        seed: opts.seed,
    };
    let g = model.generate(&text, &opts.language, &voice, &p)?;
    let write = |name: &str, x: &[f32]| -> std::io::Result<()> {
        std::fs::write(out.join(name), xrt_audio::audio::write_wav(x, 24_000))
    };
    write("0-raw.wav", &g.samples)?;
    let t = xrt_audio::prosody::trim_to_speech(&g.samples, 24_000);
    write("1-trimmed.wav", &t)?;
    // Word-safe shaping on this one take, exactly as the pipeline does it.
    let asr = xrt_audio::whisper::Recognizer::load(&opts.asr_dir, Device::Cuda(0))?;
    let heard = asr.transcribe_long(&t, 24_000, &opts.language)?;
    let spans = xrt_audio::wordsafe::protected_spans(&t, 24_000, &heard);
    let (b, n) =
        xrt_audio::wordsafe::soften_between_words(&t, 24_000, &spans, opts.breath_reduction_db);
    write("2-breaths.wav", &b)?;
    let changed_in_words: usize = spans
        .iter()
        .map(|&(a, e)| (a..e).filter(|&k| t[k] != b[k]).count())
        .sum();
    eprintln!("stopped={} tokens={} raw {:.2}s trimmed {:.2}s  {} words heard, {} gaps softened, {} word samples changed",
        g.stopped, g.speech_tokens, g.samples.len() as f32 / 24e3, t.len() as f32 / 24e3, heard.len(), n, changed_in_words);
    Ok(())
}
