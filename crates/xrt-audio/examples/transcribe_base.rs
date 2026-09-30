//! Verify the preserved Whisper-base adapter against a real recording.
use xrt_audio::whisper::WhisperModel;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: transcribe_base <audio.wav>".into());
    }
    let (audio, rate) = xrt_audio::audio::read_wav(&std::fs::read(&args[0])?)?;
    let mut model = WhisperModel::load_from_registry()?;
    let transcript = model.transcribe(&audio, rate)?;
    println!("{}", transcript.text);
    if transcript.text.trim().is_empty() {
        return Err("empty transcription".into());
    }
    for segment in &transcript.segments {
        if segment.end > audio.len() as f32 / rate as f32 + 0.001 {
            return Err("segment exceeds audio duration".into());
        }
    }
    Ok(())
}
