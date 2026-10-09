//! GBNF compatibility compiled into the bounded llguidance token matcher.

use llguidance::{
    api::{GrammarWithLexer, StopReason, TopLevelGrammar},
    Matcher, ParserFactory,
};
use schoolmarm::{
    parse,
    types::{Element, ElementType},
};
use std::sync::{Arc, Mutex};
use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};
use xrt_core::{Result, XrtError};
use xrt_tokenizer::Tokenizer;

const MAX_GRAMMAR_BYTES: usize = 64 * 1024;
const MAX_RULES: usize = 4096;
const MAX_ELEMENTS: usize = 32 * 1024;

#[derive(Debug, Clone)]
pub struct Grammar {
    compiled: TopLevelGrammar,
}

#[derive(Clone)]
pub struct GrammarState {
    matcher: Matcher,
    valid: bool,
}

impl std::fmt::Debug for GrammarState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarState")
            .field("valid", &self.valid)
            .finish()
    }
}

impl Grammar {
    pub fn parse(input: &str) -> std::result::Result<Self, String> {
        Self::from_lark(Self::gbnf_rules(input, "g")?, Vec::new())
    }

    /// Compile GBNF rules under a private prefix for a composite constraint.
    pub fn gbnf_rules(input: &str, prefix: &str) -> std::result::Result<String, String> {
        if input.len() > MAX_GRAMMAR_BYTES || input.contains('\0') {
            return Err("grammar exceeds its 64 KiB bound or contains NUL".to_string());
        }
        if prefix.is_empty()
            || !prefix
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        {
            return Err("grammar rule prefix must be lowercase alphanumeric".to_string());
        }
        // Bound expansion before the compatibility parser allocates rules.
        let input = normalize_continuations(input)?;
        preflight_expansion(&input)?;
        let parsed = parse::parse(&input).map_err(|error| error.to_string())?;
        let root = parsed
            .root_index()
            .ok_or("grammar must define a root rule")?;
        if parsed.rules.len() > MAX_RULES
            || parsed.rules.iter().map(Vec::len).sum::<usize>() > MAX_ELEMENTS
        {
            return Err("grammar exceeds its compiled rule/element bound".to_string());
        }
        let mut regular_cache = vec![None; parsed.rules.len()];
        let mut visiting = vec![false; parsed.rules.len()];
        if let Some(regex) = regular_rule(root, &parsed.rules, &mut visiting, &mut regular_cache, 0)
        {
            let terminal = format!("{}ROOT", prefix.to_ascii_uppercase());
            return Ok(format!(
                "{prefix}start: {terminal}\n{terminal}: /{regex}/\n"
            ));
        }
        let mut lark = format!("{prefix}start: {prefix}r{root}\n");
        for (rule_id, rule) in parsed.rules.iter().enumerate() {
            if let Some(regex) =
                regular_rule(rule_id, &parsed.rules, &mut visiting, &mut regular_cache, 0)
            {
                let terminal = format!("{}R{rule_id}", prefix.to_ascii_uppercase());
                lark.push_str(&format!(
                    "{prefix}r{rule_id}: {terminal}\n{terminal}: /{regex}/\n"
                ));
                continue;
            }
            lark.push_str(&format!("{prefix}r{rule_id}: "));
            let mut index = 0;
            while index < rule.len() {
                let element = rule[index];
                match element.etype {
                    ElementType::End => break,
                    ElementType::Alt => lark.push_str(" | "),
                    ElementType::RuleRef => lark.push_str(&format!("{prefix}r{} ", element.value)),
                    ElementType::CharAny => lark.push_str("/[\\s\\S]/ "),
                    ElementType::Char | ElementType::CharNot => {
                        let mut class = String::from("/[");
                        if element.etype == ElementType::CharNot {
                            class.push('^');
                        }
                        class.push_str(&format!("\\x{{{:X}}}", element.value));
                        while index + 1 < rule.len() {
                            let next = rule[index + 1];
                            if next.etype == ElementType::CharRngUpper {
                                class.push_str(&format!("-\\x{{{:X}}}", next.value));
                            } else if next.etype == ElementType::CharAlt {
                                class.push_str(&format!("\\x{{{:X}}}", next.value));
                            } else {
                                break;
                            }
                            index += 1;
                        }
                        class.push_str("]/ ");
                        lark.push_str(&class);
                    }
                    _ => return Err("invalid compiled GBNF character class".to_string()),
                }
                index += 1;
            }
            lark.push('\n');
        }
        Ok(lark)
    }

    /// Composite server constraints retain full JSON schemas as subgrammars.
    pub fn from_lark(
        lark: String,
        schemas: Vec<(String, serde_json::Value)>,
    ) -> std::result::Result<Self, String> {
        if lark.len() > 4 * MAX_GRAMMAR_BYTES || schemas.len() > 256 {
            return Err("composite grammar exceeds its size bound".to_string());
        }
        let lark = if schemas.is_empty() && lark.starts_with("gstart:") {
            format!("start: gstart\n{lark}")
        } else {
            lark
        };
        let mut compiled = TopLevelGrammar::from_lark(lark);
        for (name, schema) in schemas {
            if serde_json::to_vec(&schema)
                .map_err(|error| error.to_string())?
                .len()
                > MAX_GRAMMAR_BYTES
            {
                return Err("tool schema exceeds its 64 KiB bound".to_string());
            }
            let mut entry = GrammarWithLexer::from_json_schema(schema);
            entry.name = Some(name);
            compiled.grammars.push(entry);
        }
        let grammar = Self { compiled };
        grammar
            .matcher(&byte_factory()?)
            .map_err(|error| error.to_string())?;
        Ok(grammar)
    }

    fn matcher(&self, factory: &ParserFactory) -> Result<Matcher> {
        let mut matcher = Matcher::new(factory.create_parser(self.compiled.clone()));
        if let Some(error) = matcher.get_error() {
            return Err(XrtError::Runtime(format!("invalid grammar: {error}")));
        }
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(XrtError::Runtime(format!(
                "unsupported grammar/schema constraint: {}",
                warnings.join("; ")
            )));
        }
        Ok(matcher)
    }

    /// Recover parser-owned boundaries from completed output, not text markers.
    pub fn captures(&self, text: &str) -> std::result::Result<Vec<(String, String)>, String> {
        if text.len() > 4 * 1024 * 1024 {
            return Err("captured output exceeds its 4 MiB bound".to_string());
        }
        llguidance::panic_utils::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let factory = byte_factory()
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
            let mut parser = factory.create_parser(self.compiled.clone())?;
            parser.start_without_prompt();
            for byte in text.bytes() {
                parser.consume_token(u32::from(byte))?;
            }
            parser.check_stop()?;
            if !parser.is_accepting() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "captured output is incomplete",
                )
                .into());
            }
            parser
                .parser
                .captures()
                .iter()
                .map(|(name, bytes)| {
                    let text = std::str::from_utf8(bytes)?.to_string();
                    Ok((name.clone(), text))
                })
                .collect()
        }))
        .map_err(|error| error.to_string())
    }

    /// Compatibility byte state. Inference uses the model-specific token trie.
    pub fn start(&self) -> GrammarState {
        let factory = byte_factory().expect("fixed byte tokenizer factory");
        GrammarState {
            matcher: self.matcher(&factory).expect("grammar validated at parse"),
            valid: true,
        }
    }

    pub fn allowed_bytes(&self, state: &GrammarState) -> [bool; 256] {
        let mut matcher = state.matcher.clone();
        let mut allowed = [false; 256];
        if let Ok(mask) = matcher.compute_mask() {
            for (index, value) in allowed.iter_mut().enumerate() {
                *value = state.valid && mask.is_allowed(index as u32);
            }
        }
        allowed
    }

    pub fn advance(&self, state: &GrammarState, piece: &str) -> Option<GrammarState> {
        let mut next = state.clone();
        for byte in piece.as_bytes() {
            let token = u32::from(*byte);
            if !next.valid || !next.matcher.compute_mask().ok()?.is_allowed(token) {
                return None;
            }
            next.matcher.consume_token(token).ok()?;
        }
        Some(next)
    }

    pub fn is_complete(&self, state: &GrammarState) -> bool {
        let mut matcher = state.matcher.clone();
        state.valid && matcher.is_accepting().unwrap_or(false)
    }

    pub fn token_mask(&self, state: &GrammarState, vocab: &[String]) -> Vec<bool> {
        vocab
            .iter()
            .map(|piece| {
                if piece.is_empty() {
                    self.is_complete(state)
                } else {
                    self.advance(state, piece).is_some()
                }
            })
            .collect()
    }
}

// GBNF's expanded character repetitions are regular. Keeping those as lexer
// expressions avoids an Earley item per byte across a real model's vocabulary.
fn regular_rule(
    id: usize,
    rules: &[Vec<Element>],
    visiting: &mut [bool],
    cache: &mut [Option<String>],
    depth: usize,
) -> Option<String> {
    if let Some(regex) = &cache[id] {
        return Some(regex.clone());
    }
    if visiting[id] || depth > 128 {
        return None;
    }
    visiting[id] = true;
    let result = (|| {
        let mut bases = Vec::new();
        let mut repeats = Vec::new();
        let rule = &rules[id];
        let end = rule
            .iter()
            .position(|element| element.etype == ElementType::End)?;
        for branch in rule[..end].split(|element| element.etype == ElementType::Alt) {
            let recursive = branch.last().is_some_and(|element| {
                element.etype == ElementType::RuleRef && element.value as usize == id
            });
            let atoms = if recursive {
                &branch[..branch.len() - 1]
            } else {
                branch
            };
            let mut regex = String::new();
            let mut index = 0;
            while index < atoms.len() {
                let element = atoms[index];
                if element.etype == ElementType::RuleRef {
                    let nested =
                        regular_rule(element.value as usize, rules, visiting, cache, depth + 1)?;
                    regex.push_str(&format!("(?:{nested})"));
                } else {
                    regex.push_str(&character_regex(atoms, &mut index)?);
                }
                if regex.len() > MAX_GRAMMAR_BYTES {
                    return None;
                }
                index += 1;
            }
            if recursive {
                if regex.is_empty() {
                    return None;
                }
                repeats.push(regex);
            } else {
                bases.push(regex);
            }
        }
        if bases.is_empty() {
            return None;
        }
        let base = format!("(?:{})", bases.join("|"));
        let regex = if repeats.is_empty() {
            base
        } else {
            format!("(?:{})*{base}", repeats.join("|"))
        };
        (regex.len() <= MAX_GRAMMAR_BYTES).then_some(regex)
    })();
    visiting[id] = false;
    if let Some(regex) = &result {
        if cache.iter().flatten().map(String::len).sum::<usize>() + regex.len()
            <= 4 * MAX_GRAMMAR_BYTES
        {
            cache[id] = Some(regex.clone());
        }
    }
    result
}

fn character_regex(atoms: &[Element], index: &mut usize) -> Option<String> {
    let element = atoms[*index];
    if element.etype == ElementType::CharAny {
        return Some("[\\s\\S]".to_string());
    }
    if !matches!(element.etype, ElementType::Char | ElementType::CharNot) {
        return None;
    }
    let mut class = if element.etype == ElementType::CharNot {
        "[^"
    } else {
        "["
    }
    .to_string();
    class.push_str(&format!("\\x{{{:X}}}", element.value));
    while let Some(next) = atoms.get(*index + 1) {
        if next.etype == ElementType::CharRngUpper {
            class.push_str(&format!("-\\x{{{:X}}}", next.value));
        } else if next.etype == ElementType::CharAlt {
            class.push_str(&format!("\\x{{{:X}}}", next.value));
        } else {
            break;
        }
        *index += 1;
    }
    class.push(']');
    Some(class)
}

fn preflight_expansion(input: &str) -> std::result::Result<(), String> {
    let bytes = input.as_bytes();
    let mut index = 0;
    let mut groups = vec![(0usize, 0usize)];
    let mut total = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'#' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        let mut cost = 0;
        if byte == b'"' || byte == b'[' {
            let end = if byte == b'"' { b'"' } else { b']' };
            index += 1;
            while index < bytes.len() && bytes[index] != end {
                if bytes[index] == b'\\' {
                    index += 1;
                }
                index += 1;
                cost += 1;
            }
            cost = cost.max(1);
        } else if byte == b'(' {
            if groups.len() >= 128 {
                return Err("grammar nesting exceeds its bound".to_string());
            }
            groups.push((0, 0));
        } else if byte == b')' && groups.len() > 1 {
            cost = groups.pop().expect("nested group").0.max(1);
        } else if byte == b'{' {
            let end = input[index..]
                .find('}')
                .map(|offset| index + offset)
                .ok_or("unterminated grammar repetition")?;
            let bounds = &input[index + 1..end];
            let (min, max) = match bounds.split_once(',') {
                Some((min, max)) => (min.trim(), max.trim()),
                None => (bounds.trim(), bounds.trim()),
            };
            let min = min
                .parse::<usize>()
                .map_err(|_| "invalid grammar repetition")?;
            let max = if max.is_empty() {
                min.saturating_add(1)
            } else {
                max.parse::<usize>()
                    .map_err(|_| "invalid grammar repetition")?
            };
            if max < min || max > 2000 {
                return Err("grammar repetition bounds are invalid".to_string());
            }
            let group = groups.last_mut().expect("root group");
            let repeated = group
                .1
                .checked_mul(max.max(1))
                .ok_or("grammar repetition overflow")?;
            group.0 = group.0.saturating_add(repeated.saturating_sub(group.1));
            group.1 = repeated;
            index = end;
        } else if matches!(byte, b'*' | b'+' | b'?') {
            let group = groups.last_mut().expect("root group");
            group.0 = group
                .0
                .saturating_add(group.1.saturating_mul(2).saturating_add(4));
            group.1 = group.1.saturating_mul(3).saturating_add(4);
        } else if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
            while index + 1 < bytes.len()
                && (bytes[index + 1].is_ascii_alphanumeric()
                    || matches!(bytes[index + 1], b'_' | b'-'))
            {
                index += 1;
            }
            cost = 1;
        } else if byte == b'\n' && groups.len() == 1 {
            total = total.saturating_add(groups[0].0);
            groups[0] = (0, 0);
        }
        let group = groups.last_mut().expect("root group");
        if cost > 0 {
            group.0 = group.0.saturating_add(cost);
            group.1 = cost;
        }
        if total.saturating_add(group.0) > MAX_ELEMENTS {
            return Err("grammar expansion exceeds its bound".to_string());
        }
        index += 1;
    }
    Ok(())
}

fn normalize_continuations(input: &str) -> std::result::Result<String, String> {
    let mut lines: Vec<String> = Vec::new();
    let mut rule_line: Option<usize> = None;
    let mut declarations = std::collections::HashSet::new();
    for line in input.lines() {
        let mut quoted = false;
        let mut class = false;
        let mut escaped = false;
        let mut end = line.len();
        for (index, ch) in line.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' && !class {
                quoted = !quoted;
            }
            if ch == '[' && !quoted {
                class = true;
            }
            if ch == ']' && !quoted {
                class = false;
            }
            if ch == '#' && !quoted && !class {
                end = index;
                break;
            }
        }
        let line = &line[..end];
        let trimmed = line.trim();
        if trimmed.starts_with('|') {
            let index = rule_line.ok_or("grammar alternative has no preceding rule")?;
            lines[index].push(' ');
            lines[index].push_str(trimmed);
        } else {
            if let Some((name, _)) = trimmed.split_once("::=") {
                if !declarations.insert(name.trim().to_string()) {
                    return Err("duplicate grammar rule declaration".to_string());
                }
                rule_line = Some(lines.len());
            }
            lines.push(line.to_string());
        }
    }
    Ok(lines.join("\n"))
}

fn bounded_factory(environment: &TokEnv) -> std::result::Result<ParserFactory, String> {
    let mut factory = ParserFactory::new_simple(environment).map_err(|error| error.to_string())?;
    factory.quiet();
    let limits = factory.limits_mut();
    limits.verbose_errors = false;
    limits.max_grammar_size = MAX_ELEMENTS;
    limits.max_items_in_row = 2048;
    limits.step_max_items = 50_000;
    limits.max_lexer_states = 32_768;
    Ok(factory)
}

fn byte_factory() -> std::result::Result<ParserFactory, String> {
    bounded_factory(&ApproximateTokEnv::single_byte_env())
}

pub(crate) struct GrammarTokenEnvironment {
    factory: Mutex<ParserFactory>,
    environment: TokEnv,
}

impl GrammarTokenEnvironment {
    pub fn new(tokenizer: &Tokenizer) -> Result<Arc<Self>> {
        let eos = tokenizer.special_tokens().eos.ok_or_else(|| {
            XrtError::Runtime("constrained decoding requires an EOS token".to_string())
        })?;
        let mut words = (0..tokenizer.vocab_size())
            .map(|token| tokenizer.token_bytes(token as u32))
            .collect::<Result<Vec<_>>>()?;
        let eos_word = words.get_mut(eos as usize).ok_or_else(|| {
            XrtError::Runtime("constrained decoding EOS is outside the vocabulary".to_string())
        })?;
        *eos_word = b"\xff<|xeno_eos|>".to_vec();
        let info = TokRxInfo::new(words.len() as u32, eos);
        let environment: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
        let factory = bounded_factory(&environment).map_err(XrtError::Runtime)?;
        Ok(Arc::new(Self {
            factory: Mutex::new(factory),
            environment,
        }))
    }

    pub fn matcher(&self, grammar: &Grammar) -> Result<GrammarTokenMatcher> {
        let factory = self
            .factory
            .lock()
            .map_err(|_| XrtError::Runtime("grammar factory lock poisoned".to_string()))?;
        Ok(GrammarTokenMatcher {
            matcher: grammar.matcher(&factory)?,
            environment: self.environment.clone(),
        })
    }
}

pub(crate) struct GrammarTokenMatcher {
    matcher: Matcher,
    environment: TokEnv,
}

impl GrammarTokenMatcher {
    pub fn mask(&mut self) -> Result<Vec<bool>> {
        if self.matcher.is_stopped()
            && !matches!(
                self.matcher.stop_reason(),
                StopReason::NoExtension | StopReason::NoExtensionBias | StopReason::EndOfSentence
            )
        {
            return Err(XrtError::Runtime(format!(
                "grammar stopped without completion: {}",
                self.matcher.stop_reason()
            )));
        }
        let mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|error| XrtError::Runtime(format!("grammar token mask: {error}")))?;
        Ok((0..self.environment.tok_trie().vocab_size())
            .map(|token| mask.is_allowed(token as u32))
            .collect())
    }

    pub fn consume(&mut self, token: u32) -> Result<()> {
        self.matcher
            .consume_token(token)
            .map_err(|error| XrtError::Runtime(format!("grammar token rejected: {error}")))
    }

    pub fn complete(&mut self) -> Result<bool> {
        self.matcher
            .is_accepting()
            .map_err(|error| XrtError::Runtime(format!("grammar completion: {error}")))
    }
}

#[cfg(test)]
mod qualification_tests {
    use super::Grammar;

    #[test]
    fn split_literals_keep_the_consumed_prefix() {
        let grammar = Grammar::parse("root ::= \"hello\"").unwrap();
        let state = grammar.advance(&grammar.start(), "hel").unwrap();
        assert!(grammar.advance(&state, "lo").is_some());
        assert!(grammar.advance(&state, "hel").is_none());
    }

    #[test]
    fn repetition_really_repeats() {
        let grammar = Grammar::parse("root ::= \"a\"+ \"!\"").unwrap();
        let state = grammar.advance(&grammar.start(), "aaa!").unwrap();
        assert!(grammar.is_complete(&state));
    }

    #[test]
    fn malformed_grammars_are_refused() {
        assert!(Grammar::parse("root ::= \"unclosed").is_err());
        assert!(Grammar::parse("root ::= [a-z").is_err());
        assert!(Grammar::parse("root ::= \"a\"{5,2}").is_err());
        assert!(Grammar::parse("root ::= (\"a\"{2000}){2000}").is_err());
        assert!(Grammar::parse("root ::= missing").is_err());
    }

    #[test]
    fn groups_unicode_and_continuation_lines_are_supported() {
        let grammar = Grammar::parse(
            "root ::= (\"hello\" | \"world\") suffix+\nsuffix ::= \"!\"\n | \"\u{03b3}\"\n",
        )
        .unwrap();
        let state = grammar
            .advance(&grammar.start(), "hello!\u{03b3}!")
            .unwrap();
        assert!(grammar.is_complete(&state));
    }

    #[test]
    fn continuation_after_comment_keeps_the_alternative_and_literal_hashes() {
        let grammar =
            Grammar::parse("root ::= \"hello#\" # inline comment\n | \"world\"\n").unwrap();
        for text in ["hello#", "world"] {
            let state = grammar.advance(&grammar.start(), text).unwrap();
            assert!(grammar.is_complete(&state));
        }
    }

    #[test]
    fn duplicate_rule_declarations_are_refused() {
        assert!(Grammar::parse("root ::= \"old\"\nroot ::= \"new\"\n").is_err());
    }

    #[test]
    fn recursive_nonregular_language_keeps_balanced_boundaries() {
        let grammar = Grammar::parse("root ::= \"(\" root? \")\"").unwrap();
        for text in ["()", "(())", "((()))"] {
            let state = grammar.advance(&grammar.start(), text).unwrap();
            assert!(grammar.is_complete(&state), "{text}");
        }
        for text in ["(", "(()", "())", "()(())", ")("] {
            let complete = grammar.advance(&grammar.start(), text).is_some_and(|state| grammar.is_complete(&state));
            assert!(!complete, "{text}");
        }
    }

    #[test]
    fn compiled_language_matches_the_compatibility_parser_on_a_bounded_corpus() {
        let definitions = ["root ::= [ab]{1,3} \"!\"", "root ::= (\"a\" | \"b\")+ \"!\"",
            "root ::= \"a\" root \"b\" | \"!\"", "root ::= \"a\"* \"b\"? \"!\""];
        for definition in definitions {
            let grammar = Grammar::parse(definition).unwrap();
            let reference = schoolmarm::Grammar::new(definition).unwrap();
            let mut corpus = vec![String::new()];
            for _ in 0..5 {
                let next = corpus.iter().flat_map(|text| ['a', 'b', '!'].map(|ch| format!("{text}{ch}"))).collect::<Vec<_>>();
                corpus.extend(next);
                corpus.sort(); corpus.dedup();
            }
            for text in &corpus {
                let mut state = schoolmarm::GrammarState::new(reference.clone()).unwrap();
                let expected = state.accept_token(text).is_ok() && state.is_accepting();
                let actual = grammar.advance(&grammar.start(), text).is_some_and(|state| grammar.is_complete(&state));
                assert_eq!(actual, expected, "{definition}: {text:?}");
            }
        }
    }
}
