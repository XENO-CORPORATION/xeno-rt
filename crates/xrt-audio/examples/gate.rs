//! `gate <wav> <text_tokens> <chars>` — run the chunk validator on real audio.
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0]).unwrap()).unwrap();
    let x = xrt_audio::audio::resample(&x, rate, 24_000);
    let x = xrt_audio::audio::trim_silence(&x, 24_000, 0.015);
    let c = xrt_audio::chunking::Chunk::new(
        "x".repeat(a[2].parse().unwrap()),
        a[1].parse().unwrap(),
        false,
    );
    use xrt_audio::speech::ChunkValidator;
    let gap = xrt_audio::speech::longest_quiet_gap(&x, 24_000, 0.08);
    let r = xrt_audio::speech::SignalValidator::default().check(&c, &x);
    println!(
        "{:>6.1}s  gap {gap:.2}s  {}",
        x.len() as f32 / 24_000.0,
        match r {
            Ok(()) => "ACCEPT".into(),
            Err(e) => format!("REJECT: {e}"),
        }
    );
}
