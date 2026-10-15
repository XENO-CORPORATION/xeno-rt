//! Compare native prompt rows against independent canonical-tokenizer fixtures.
use serde::Deserialize;
use xrt_audio::drama3::{
    config::Config,
    prompt::{self, JsonTokenizer, PriorTake, Reference},
};
#[derive(Deserialize)]
struct Ref {
    transcript: String,
    frames: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
struct History {
    text: String,
    frames: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    text: String,
    references: Vec<Ref>,
    history: Vec<History>,
    expected: Vec<Vec<u32>>,
}
#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: drama3_prompt_probe <tokenizer.json> <oracle.json>".into());
    }
    let config = Config::parse(include_bytes!(
        "../../../reference/audio/drama3/config.json"
    ))?;
    let tokenizer = JsonTokenizer::open(std::path::Path::new(&args[0]))?;
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    for case in fixture.cases {
        let references: Vec<_> = case
            .references
            .iter()
            .map(|r| Reference {
                transcript: &r.transcript,
                frames: &r.frames,
            })
            .collect();
        let history: Vec<_> = case
            .history
            .iter()
            .map(|r| PriorTake {
                text: &r.text,
                frames: &r.frames,
            })
            .collect();
        let result = prompt::build(&config, &tokenizer, &case.text, &references, &history, 8192)?;
        let expected: Vec<_> = case.expected.into_iter().flatten().collect();
        if result.data != expected {
            return Err(format!("native prompt rows differ for {}", case.name).into());
        }
        println!(
            "PASS canonical prompt rows: {} ({} rows)",
            case.name,
            result.len()
        );
    }
    Ok(())
}
