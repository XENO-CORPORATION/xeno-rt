use serde::Deserialize;
use std::{
    io::{self, Read},
    process::ExitCode,
};
use xrt_runtime::Grammar;

#[derive(Deserialize)]
struct Case {
    text: String,
    accepted: bool,
}

#[derive(Deserialize)]
struct Corpus {
    grammar: String,
    cases: Vec<Case>,
}

fn main() -> ExitCode {
    let mut input = String::new();
    if let Err(error) = io::stdin().take(4 * 1024 * 1024).read_to_string(&mut input) {
        eprintln!("corpus input failed: {error}");
        return ExitCode::FAILURE;
    }
    let result = (|| -> Result<(), String> {
        let corpus: Corpus = serde_json::from_str(&input).map_err(|error| error.to_string())?;
        if corpus.cases.is_empty() {
            return Err("empty grammar corpus".to_string());
        }
        let grammar = Grammar::parse(&corpus.grammar)?;
        for (index, case) in corpus.cases.iter().enumerate() {
            let accepted = grammar
                .advance(&grammar.start(), &case.text)
                .is_some_and(|state| grammar.is_complete(&state));
            if accepted != case.accepted {
                return Err(format!(
                    "grammar corpus case {index}: expected {}, observed {accepted}",
                    case.accepted
                ));
            }
        }
        println!("PASS {} generated Patch grammar cases", corpus.cases.len());
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
