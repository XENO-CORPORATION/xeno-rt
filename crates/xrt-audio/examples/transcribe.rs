//! `transcribe <wav> [cpu|cuda]` — print words with timings (any length).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let device = if a.get(1).map(String::as_str) == Some("cpu") {
        xrt_audio::chatterbox::Device::Cpu
    } else {
        xrt_audio::chatterbox::Device::Cuda(0)
    };
    let r = xrt_audio::whisper::Recognizer::load(
        &xrt_audio::whisper::Recognizer::default_dir(),
        device,
    )?;
    let t = std::time::Instant::now();
    let words = r.transcribe_long(&x, rate, "en")?;
    for w in &words {
        println!(
            "{:6.2}-{:6.2}  p={:.2}  {}",
            w.start, w.end, w.probability, w.text
        );
    }
    eprintln!(
        "provider {}  {} words in {:.2}s",
        r.provider,
        words.len(),
        t.elapsed().as_secs_f32()
    );
    Ok(())
}
