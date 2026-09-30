//! `debreath <in.wav> <out.wav> [db]` — apply only word-safe breath softening
//! to an existing take (transcribes it first), so its effect can be judged on
//! real audio in isolation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let db: f32 = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(15.0);
    let asr = xrt_audio::whisper::Recognizer::load(
        &xrt_audio::whisper::Recognizer::default_dir(),
        xrt_audio::chatterbox::Device::Cuda(0),
    )?;
    let words = asr.transcribe_long(&x, rate, "en")?;
    let spans = xrt_audio::wordsafe::protected_spans(&x, rate, &words);
    let (y, n) = xrt_audio::wordsafe::soften_between_words(&x, rate, &spans, db);
    let changed = x
        .iter()
        .zip(&y)
        .filter(|(p, q)| (*p - *q).abs() > 1e-7)
        .count();
    std::fs::write(&a[1], xrt_audio::audio::write_wav(&y, rate))?;
    eprintln!(
        "{} words, {n} gaps softened, {:.1}% of samples attenuated",
        words.len(),
        100.0 * changed as f32 / x.len() as f32
    );
    Ok(())
}
