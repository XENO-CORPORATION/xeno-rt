//! Grammar-constrained decoding using GBNF (GGML BNF) grammars.
//!
//! A GBNF grammar defines which token sequences are valid. During decoding,
//! the grammar state machine determines which tokens are allowed at each step,
//! and the sampler masks out disallowed tokens before sampling.

use std::collections::HashMap;

/// A parsed GBNF grammar ready for constrained decoding.
#[derive(Debug, Clone)]
pub struct Grammar {
    rules: Vec<Rule>,
    rule_map: HashMap<String, usize>,
}

/// A grammar rule: name -> list of alternatives (each is a sequence of elements).
#[derive(Debug, Clone)]
struct Rule {
    alternatives: Vec<Vec<GrammarElement>>,
}

/// An element in a grammar production.
#[derive(Debug, Clone)]
enum GrammarElement {
    /// Literal string that must match exactly.
    Literal(String),
    /// Character range [a-z] or character set [abc].
    CharRange(Vec<(char, char)>),
    /// Negated character range [^a-z].
    CharRangeNeg(Vec<(char, char)>),
    /// Reference to another rule.
    RuleRef(usize),
    /// Optional element (e?).
    Optional(Box<GrammarElement>),
    /// Repeat zero or more times (e*).
    Repeat(Box<GrammarElement>),
    /// Repeat one or more times (e+).
    RepeatOne(Box<GrammarElement>),
}

/// The state of grammar-constrained decoding, tracking position within the grammar.
#[derive(Debug, Clone)]
pub struct GrammarState {
    /// Stack of (rule_index, alternative_index, position_in_alternative).
    /// Multiple entries represent nested rule references.
    stacks: Vec<Vec<(usize, usize, usize)>>,
}

impl Grammar {
    /// Parse a GBNF grammar string.
    ///
    /// GBNF format:
    /// ```text
    /// root ::= "hello" | "world"
    /// digit ::= [0-9]
    /// number ::= digit+
    /// ```
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut rules = Vec::new();
        let mut rule_map = HashMap::new();

        // First pass: collect rule names
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(sep_pos) = line.find("::=") {
                let name = line[..sep_pos].trim().to_string();
                if !rule_map.contains_key(&name) {
                    rule_map.insert(name.clone(), rules.len());
                    rules.push(Rule {
                        alternatives: Vec::new(),
                    });
                }
            }
        }

        if !rule_map.contains_key("root") {
            return Err("grammar must define a 'root' rule".to_string());
        }

        // Second pass: parse rule bodies
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(sep_pos) = line.find("::=") {
                let name = line[..sep_pos].trim();
                let body = line[sep_pos + 3..].trim();
                let rule_idx = rule_map[name];

                let alternatives = Self::parse_alternatives(body, &rule_map)?;
                rules[rule_idx].alternatives = alternatives;
            }
        }

        Ok(Grammar { rules, rule_map })
    }

    fn parse_alternatives(
        body: &str,
        rule_map: &HashMap<String, usize>,
    ) -> Result<Vec<Vec<GrammarElement>>, String> {
        let mut alternatives = Vec::new();
        // Split on '|' but not inside quotes or brackets
        let mut current = Vec::new();
        let mut chars = body.chars().peekable();

        while chars.peek().is_some() {
            Self::skip_whitespace(&mut chars);
            if chars.peek().is_none() {
                break;
            }

            match chars.peek() {
                Some('|') => {
                    chars.next();
                    if !current.is_empty() {
                        alternatives.push(std::mem::take(&mut current));
                    }
                }
                Some('"') => {
                    chars.next();
                    let mut s = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == '"' {
                            chars.next();
                            break;
                        }
                        if c == '\\' {
                            chars.next();
                            match chars.next() {
                                Some('n') => s.push('\n'),
                                Some('t') => s.push('\t'),
                                Some('r') => s.push('\r'),
                                Some('\\') => s.push('\\'),
                                Some('"') => s.push('"'),
                                Some(c) => {
                                    s.push('\\');
                                    s.push(c);
                                }
                                None => return Err("unterminated escape in string".to_string()),
                            }
                        } else {
                            s.push(c);
                            chars.next();
                        }
                    }
                    let mut elem = GrammarElement::Literal(s);
                    elem = Self::parse_quantifier(&mut chars, elem);
                    current.push(elem);
                }
                Some('[') => {
                    chars.next();
                    let negated = chars.peek() == Some(&'^');
                    if negated {
                        chars.next();
                    }
                    let mut ranges = Vec::new();
                    while let Some(&c) = chars.peek() {
                        if c == ']' {
                            chars.next();
                            break;
                        }
                        let start = c;
                        chars.next();
                        if chars.peek() == Some(&'-') {
                            chars.next();
                            if let Some(&end) = chars.peek() {
                                chars.next();
                                ranges.push((start, end));
                            }
                        } else {
                            ranges.push((start, start));
                        }
                    }
                    let mut elem = if negated {
                        GrammarElement::CharRangeNeg(ranges)
                    } else {
                        GrammarElement::CharRange(ranges)
                    };
                    elem = Self::parse_quantifier(&mut chars, elem);
                    current.push(elem);
                }
                Some(c) if c.is_alphabetic() || *c == '_' => {
                    let mut name = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' || c == '-' {
                            name.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if let Some(&idx) = rule_map.get(&name) {
                        let mut elem = GrammarElement::RuleRef(idx);
                        elem = Self::parse_quantifier(&mut chars, elem);
                        current.push(elem);
                    } else {
                        return Err(format!("unknown rule reference: {name}"));
                    }
                }
                Some(c) => return Err(format!("unexpected character in grammar: {c}")),
                None => break,
            }
        }

        if !current.is_empty() {
            alternatives.push(current);
        }

        Ok(alternatives)
    }

    fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars>) {
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() {
                chars.next();
            } else {
                break;
            }
        }
    }

    fn parse_quantifier(
        chars: &mut std::iter::Peekable<std::str::Chars>,
        elem: GrammarElement,
    ) -> GrammarElement {
        match chars.peek() {
            Some('?') => {
                chars.next();
                GrammarElement::Optional(Box::new(elem))
            }
            Some('*') => {
                chars.next();
                GrammarElement::Repeat(Box::new(elem))
            }
            Some('+') => {
                chars.next();
                GrammarElement::RepeatOne(Box::new(elem))
            }
            _ => elem,
        }
    }

    /// Create a new decoding state starting at the root rule.
    pub fn start(&self) -> GrammarState {
        let root_idx = self.rule_map["root"];
        let mut stacks = Vec::new();
        for (alt_idx, _) in self.rules[root_idx].alternatives.iter().enumerate() {
            stacks.push(vec![(root_idx, alt_idx, 0)]);
        }
        GrammarState { stacks }
    }

    /// Compute which byte values are allowed at the current grammar position.
    /// Returns a 256-element bitmask (one bit per byte value).
    pub fn allowed_bytes(&self, state: &GrammarState) -> [bool; 256] {
        let mut allowed = [false; 256];
        for stack in &state.stacks {
            self.allowed_bytes_for_stack(stack, &mut allowed);
        }
        allowed
    }

    fn allowed_bytes_for_stack(&self, stack: &[(usize, usize, usize)], allowed: &mut [bool; 256]) {
        if stack.is_empty() {
            // Completed: no more bytes needed. Allow EOS-like behavior.
            return;
        }
        let &(rule_idx, alt_idx, pos) = stack.last().unwrap();
        let alt = &self.rules[rule_idx].alternatives[alt_idx];
        if pos >= alt.len() {
            // This alternative is fully consumed — pop and continue parent
            let mut parent_stack: Vec<(usize, usize, usize)> = stack[..stack.len() - 1].to_vec();
            if let Some(last) = parent_stack.last_mut() {
                last.2 += 1;
            }
            self.allowed_bytes_for_stack(&parent_stack, allowed);
            return;
        }
        self.allowed_bytes_for_element(&alt[pos], allowed, stack);
    }

    fn allowed_bytes_for_element(
        &self,
        elem: &GrammarElement,
        allowed: &mut [bool; 256],
        stack: &[(usize, usize, usize)],
    ) {
        match elem {
            GrammarElement::Literal(s) => {
                if let Some(first_byte) = s.as_bytes().first() {
                    allowed[*first_byte as usize] = true;
                }
            }
            GrammarElement::CharRange(ranges) => {
                for &(lo, hi) in ranges {
                    for b in (lo as u8)..=(hi as u8) {
                        allowed[b as usize] = true;
                    }
                }
            }
            GrammarElement::CharRangeNeg(ranges) => {
                for b in 0u8..=255 {
                    let excluded = ranges.iter().any(|&(lo, hi)| {
                        let c = b as char;
                        c >= lo && c <= hi
                    });
                    if !excluded {
                        allowed[b as usize] = true;
                    }
                }
            }
            GrammarElement::RuleRef(idx) => {
                for (alt_idx, _) in self.rules[*idx].alternatives.iter().enumerate() {
                    let mut new_stack = stack.to_vec();
                    new_stack.push((*idx, alt_idx, 0));
                    self.allowed_bytes_for_stack(&new_stack, allowed);
                }
            }
            GrammarElement::Optional(inner) | GrammarElement::Repeat(inner) => {
                // Can match the element or skip it
                self.allowed_bytes_for_element(inner, allowed, stack);
                // Can also skip: advance position
                let mut skip_stack = stack.to_vec();
                if let Some(last) = skip_stack.last_mut() {
                    last.2 += 1;
                }
                self.allowed_bytes_for_stack(&skip_stack, allowed);
            }
            GrammarElement::RepeatOne(inner) => {
                // Must match at least once
                self.allowed_bytes_for_element(inner, allowed, stack);
            }
        }
    }

    /// Advance the grammar state after consuming a token piece (string of bytes).
    /// Returns the new state, or None if the piece is not allowed.
    pub fn advance(&self, state: &GrammarState, piece: &str) -> Option<GrammarState> {
        let mut new_stacks = Vec::new();
        for stack in &state.stacks {
            self.advance_stack(stack, piece.as_bytes(), 0, &mut new_stacks);
        }
        if new_stacks.is_empty() {
            None
        } else {
            Some(GrammarState { stacks: new_stacks })
        }
    }

    fn advance_stack(
        &self,
        stack: &[(usize, usize, usize)],
        bytes: &[u8],
        byte_pos: usize,
        results: &mut Vec<Vec<(usize, usize, usize)>>,
    ) {
        if byte_pos >= bytes.len() {
            results.push(stack.to_vec());
            return;
        }
        if stack.is_empty() {
            return;
        }

        let &(rule_idx, alt_idx, pos) = stack.last().unwrap();
        let alt = &self.rules[rule_idx].alternatives[alt_idx];
        if pos >= alt.len() {
            // Pop this rule frame and continue with parent
            let mut parent = stack[..stack.len() - 1].to_vec();
            if let Some(last) = parent.last_mut() {
                last.2 += 1;
            }
            self.advance_stack(&parent, bytes, byte_pos, results);
            return;
        }

        let elem = &alt[pos];
        self.advance_element(elem, stack, bytes, byte_pos, results);
    }

    fn advance_element(
        &self,
        elem: &GrammarElement,
        stack: &[(usize, usize, usize)],
        bytes: &[u8],
        byte_pos: usize,
        results: &mut Vec<Vec<(usize, usize, usize)>>,
    ) {
        let b = bytes[byte_pos];
        match elem {
            GrammarElement::Literal(s) => {
                let s_bytes = s.as_bytes();
                // Check if remaining bytes in this literal match
                // We consume one byte at a time from the literal
                if !s_bytes.is_empty() && s_bytes[0] == b {
                    if s_bytes.len() == 1 {
                        // Literal fully consumed, advance position
                        let mut new_stack = stack.to_vec();
                        if let Some(last) = new_stack.last_mut() {
                            last.2 += 1;
                        }
                        self.advance_stack(&new_stack, bytes, byte_pos + 1, results);
                    } else {
                        // Replace with remaining literal
                        // This is tricky with the current representation.
                        // For simplicity, we only support single-byte-at-a-time matching
                        // by checking if the full literal prefix matches.
                        let remaining = &bytes[byte_pos..];
                        if remaining.len() >= s_bytes.len() && remaining.starts_with(s_bytes) {
                            let mut new_stack = stack.to_vec();
                            if let Some(last) = new_stack.last_mut() {
                                last.2 += 1;
                            }
                            self.advance_stack(
                                &new_stack,
                                bytes,
                                byte_pos + s_bytes.len(),
                                results,
                            );
                        } else if s_bytes.starts_with(remaining) {
                            // Partial match: the token piece ends mid-literal
                            // We can't advance past this, but we accept it as partial
                            // This is a simplification; full impl would track literal offset
                            results.push(stack.to_vec());
                        }
                    }
                }
            }
            GrammarElement::CharRange(ranges) => {
                let c = b as char;
                let matches = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
                if matches {
                    let mut new_stack = stack.to_vec();
                    if let Some(last) = new_stack.last_mut() {
                        last.2 += 1;
                    }
                    self.advance_stack(&new_stack, bytes, byte_pos + 1, results);
                }
            }
            GrammarElement::CharRangeNeg(ranges) => {
                let c = b as char;
                let excluded = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
                if !excluded {
                    let mut new_stack = stack.to_vec();
                    if let Some(last) = new_stack.last_mut() {
                        last.2 += 1;
                    }
                    self.advance_stack(&new_stack, bytes, byte_pos + 1, results);
                }
            }
            GrammarElement::RuleRef(idx) => {
                for (alt_idx, _) in self.rules[*idx].alternatives.iter().enumerate() {
                    let mut new_stack = stack.to_vec();
                    new_stack.push((*idx, alt_idx, 0));
                    self.advance_stack(&new_stack, bytes, byte_pos, results);
                }
            }
            GrammarElement::Optional(inner) | GrammarElement::Repeat(inner) => {
                // Try matching the element
                self.advance_element(inner, stack, bytes, byte_pos, results);
                // Try skipping it
                let mut skip_stack = stack.to_vec();
                if let Some(last) = skip_stack.last_mut() {
                    last.2 += 1;
                }
                self.advance_stack(&skip_stack, bytes, byte_pos, results);
            }
            GrammarElement::RepeatOne(inner) => {
                // Must match at least once, then can repeat
                self.advance_element(inner, stack, bytes, byte_pos, results);
            }
        }
    }

    /// Check if any stack has completed (all rules consumed).
    pub fn is_complete(&self, state: &GrammarState) -> bool {
        state.stacks.iter().any(|stack| {
            if stack.is_empty() {
                return true;
            }
            let &(rule_idx, alt_idx, pos) = stack.last().unwrap();
            let alt = &self.rules[rule_idx].alternatives[alt_idx];
            pos >= alt.len() && stack.len() == 1
        })
    }

    /// Build a token-level allow mask from the grammar state and vocabulary.
    /// Returns a Vec<bool> of size vocab_size where true = allowed.
    pub fn token_mask(&self, state: &GrammarState, vocab: &[String]) -> Vec<bool> {
        let allowed_bytes = self.allowed_bytes(state);
        let mut mask = vec![false; vocab.len()];

        for (idx, piece) in vocab.iter().enumerate() {
            if piece.is_empty() {
                // Empty tokens (padding, special) — allow if grammar is complete
                mask[idx] = self.is_complete(state);
                continue;
            }
            // Check if the first byte of this token is allowed
            if let Some(&first_byte) = piece.as_bytes().first() {
                if allowed_bytes[first_byte as usize] {
                    // Quick check passed; verify full token is consumable
                    if self.advance(state, piece).is_some() {
                        mask[idx] = true;
                    }
                }
            }
        }

        // Always allow EOS if grammar is complete
        if self.is_complete(state) {
            // EOS tokens typically have empty pieces or special markers
            // The caller handles EOS separately
        }

        mask
    }
}
