//! `pauses <take.wav> <script-file>` — show, for one take, every boundary
//! between words: the gap the recognizer measured, the gap left after the
//! word-safe protection, and whether a pause could be inserted there.
//! Diagnoses "why was this pause not lengthened".
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (x, rate) = xrt_audio::audio::read_wav(&std::fs::read(&a[0])?)?;
    let script = std::fs::read_to_string(&a[1])?;
    let asr = xrt_audio::whisper::Recognizer::load(
        &xrt_audio::whisper::Recognizer::default_dir(),
        xrt_audio::chatterbox::Device::Cuda(0),
    )?;
    let heard = asr.transcribe_long(&x, rate, "en")?;
    let c = xrt_audio::wordcheck::check(&script, &heard, &[], Default::default());
    let words = xrt_audio::wordcheck::script_word_times(&script, &c, &heard);
    let spans = xrt_audio::wordsafe::protected_spans(&x, rate, &heard);
    let gaps = xrt_audio::wordsafe::gaps(&spans, x.len());
    eprintln!(
        "{} words, {} spans, {} gaps, rejected={:?}",
        words.len(),
        spans.len(),
        gaps.len(),
        c.rejected
    );
    for w in words.windows(2) {
        let t = w[0].text.trim_end_matches(['"', '\'', '\u{201d}']);
        if !t.ends_with(['.', ',', '!', '?', ';', ':']) {
            continue;
        }
        let mid =
            |w: &xrt_audio::wordcheck::TimedWord| ((w.start + w.end) * 0.5 * rate as f32) as usize;
        let g = gaps
            .iter()
            .find(|&&(ga, gb)| ga >= mid(&w[0]) && gb <= mid(&w[1]));
        let s = rate as f32;
        println!(
            "{:>14} | {:.2}->{:.2}  heard gap {:.2}s  protected gap {}",
            w[0].text,
            w[0].end,
            w[1].start,
            w[1].start - w[0].end,
            g.map_or("NONE".to_string(), |&(ga, gb)| format!(
                "{:.2}-{:.2} ({:.2}s)",
                ga as f32 / s,
                gb as f32 / s,
                (gb - ga) as f32 / s
            )),
        );
    }
    Ok(())
}
