//! `chunks <script-file>` — print how a script is chunked: each chunk's
//! token count, whether it ends a paragraph, and its paragraph parts.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let script = std::fs::read_to_string(&a[0])?;
    let dir = xrt_audio::speech::default_model_dir();
    let tok = xrt_audio::tokenizer::ChatterboxTokenizer::from_file(&dir.join("tokenizer.json"))?;
    for (i, c) in xrt_audio::chunking::chunk_script(&script, "en", &tok, Default::default())?
        .iter()
        .enumerate()
    {
        println!(
            "chunk {i}: {} tokens, ends_paragraph={}, {} parts",
            c.tokens,
            c.ends_paragraph,
            c.parts.len()
        );
        for p in &c.parts {
            println!("   | {p}");
        }
    }
    Ok(())
}
