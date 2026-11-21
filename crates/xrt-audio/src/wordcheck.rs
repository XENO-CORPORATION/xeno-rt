//! The word check: does a generated take say its script?
//!
//! Signal gates (`speech::SignalValidator`) cannot see a fluent sentence that
//! dropped a phrase, repeated one, or slurred a word into another. This
//! compares the script with what the recognizer heard and returns the
//! word-by-word alignment, which the pipeline also uses to place pauses.
//!
//! The comparison has to tolerate the recognizer, not just the model:
//! - **numbers** — the script says "One Hundred and Twenty-Five", Whisper
//!   writes "125". Supported English integer phrases retain their VALUE.
//!   Unsupported number forms remain literal, never wildcard matches.
//! - **names** — caller-listed names may match approximate ASR spellings.
//!   Ordinary words are exact after punctuation/case normalization; numbers
//!   and negations never use fuzzy equality. ASR remains heuristic evidence.
//!
//! What it rejects is what a listener hears as broken: a phrase that is
//! missing, words that were repeated or invented, a garbled stretch. A single
//! isolated mismatch is recognizer noise far more often than a model fault,
//! so it is reported but tolerated.

use serde::Serialize;

use crate::whisper::Word;

#[derive(Debug, Clone, Copy)]
pub struct WordCheckLimits {
    /// Longest tolerated run of consecutive mismatches (of any kind).
    pub max_bad_run: usize,
    /// Longest tolerated run of script words that were not heard at all.
    pub max_missing_run: usize,
    /// Longest tolerated run of heard words that are not in the script.
    pub max_extra_run: usize,
    /// Mismatches per script word above which the take is rejected.
    pub max_error_rate: f32,
}

impl Default for WordCheckLimits {
    fn default() -> Self {
        Self {
            max_bad_run: 2,
            max_missing_run: 1,
            max_extra_run: 1,
            max_error_rate: 0.15,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Script word `.0` was heard as recognizer word `.1`.
    Match(usize, usize),
    /// Script word `.0` was heard as a different word `.1`.
    Substitute(usize, usize),
    /// Script word `.0` was not heard.
    Missing(usize),
    /// Recognizer word `.0` is not in the script.
    Extra(usize),
}

/// A script normalised for comparison. Token `k` came from the original
/// whitespace-separated words `source[k] ..= last_source[k]` (a collapsed
/// number such as "One Hundred and Twenty-Five" spans several words), so the
/// pipeline can read the words' punctuation and give every word a time.
#[derive(Debug, Clone, Default)]
pub struct Normalised {
    pub tokens: Vec<String>,
    pub source: Vec<usize>,
    pub last_source: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WordCheck {
    /// Mismatches / script tokens.
    pub error_rate: f32,
    pub script_tokens: usize,
    pub heard_tokens: usize,
    /// Human-readable mismatches, e.g. `missing "the proper"`.
    pub problems: Vec<String>,
    /// `None` when the take passed.
    pub rejected: Option<String>,
    #[serde(skip)]
    pub ops: Vec<Op>,
    #[serde(skip)]
    pub script: Normalised,
    #[serde(skip)]
    pub heard: Normalised,
}

const NUMBER_WORDS: [&str; 32] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
];

fn number_value(words: &[(String, usize)]) -> Option<u64> {
    let (mut total, mut group) = (0u64, 0u64);
    let mut previous_small: Option<u64> = None;
    let mut previous_scale = u64::MAX;
    for (word, _) in words {
        if word == "and" {
            continue;
        }
        let index = NUMBER_WORDS.iter().position(|w| w == word)?;
        if index < 28 {
            let value = if index < 20 {
                index as u64
            } else {
                (index as u64 - 18) * 10
            };
            if let Some(previous) = previous_small {
                if !(previous >= 20 && previous % 10 == 0 && (1..10).contains(&value)) {
                    return None;
                }
            }
            group = group.checked_add(value)?;
            previous_small = Some(value);
        } else if word == "hundred" {
            if group == 0 || group >= 100 {
                return None;
            }
            group = group.checked_mul(100)?;
            previous_small = None;
        } else {
            let scale = match word.as_str() {
                "thousand" => 1_000,
                "million" => 1_000_000,
                "billion" => 1_000_000_000,
                _ => return None,
            };
            if group == 0 || scale >= previous_scale {
                return None;
            }
            total = total.checked_add(group.checked_mul(scale)?)?;
            group = 0;
            previous_small = None;
            previous_scale = scale;
        }
    }
    total.checked_add(group)
}

fn critical(word: &str) -> bool {
    word.starts_with('#')
        || word.chars().any(|c| c.is_ascii_digit())
        || matches!(word, "not" | "no" | "never" | "neither" | "nor" | "without")
        || word.ends_with("nt")
}

/// Normalize case/punctuation and supported English integer phrases while
/// preserving numerical values, decimal points and signs.
pub fn normalise(text: &str) -> Normalised {
    normalise_words(text.split_whitespace())
}

/// As [`normalise`], with the words already separated (a recognizer's word
/// list); `source` indexes this list.
pub fn normalise_words<'a>(words: impl IntoIterator<Item = &'a str>) -> Normalised {
    let mut raw: Vec<(String, usize)> = Vec::new();
    for (wi, word) in words.into_iter().enumerate() {
        // "1,250" and "3.5" stay one number; any other punctuation separates.
        let signed_number = word
            .strip_prefix(['-', '+'])
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
        for part in word.split(|c: char| {
            c.is_whitespace()
                || (c == '-' && !signed_number)
                || c == '\u{2014}'
                || c == '\u{2013}'
                || c == '/'
        }) {
            let numeric = part.trim_matches(|c: char| {
                matches!(
                    c,
                    '"' | '\'' | '(' | ')' | '[' | ']' | ':' | ';' | '!' | '?' | '.' | ','
                )
            });
            let t: String = if numeric.chars().any(|c| c.is_ascii_digit())
                && numeric
                    .chars()
                    .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | '-' | '+'))
            {
                // Keep decimal points; removing them would equate 3.5 and 35.
                // Commas stay literal unless this is valid English grouping.
                let groups: Vec<_> = numeric.split(',').collect();
                let grouped = groups.len() > 1
                    && (1..=3).contains(&groups[0].len())
                    && groups[0].bytes().all(|b| b.is_ascii_digit())
                    && groups[1..]
                        .iter()
                        .all(|g| g.len() == 3 && g.bytes().all(|b| b.is_ascii_digit()));
                if grouped {
                    groups.concat()
                } else {
                    numeric.to_string()
                }
            } else {
                part.chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect()
            };
            if !t.is_empty() {
                raw.push((t, wi));
            }
        }
    }
    let mut out = Normalised::default();
    let mut k = 0;
    while k < raw.len() {
        let begin = k;
        let mut end = k + 1;
        let mut token = raw[k].0.clone();
        if NUMBER_WORDS.contains(&token.as_str()) {
            while end < raw.len()
                && (NUMBER_WORDS.contains(&raw[end].0.as_str())
                    || (raw[end].0 == "and"
                        && raw
                            .get(end + 1)
                            .is_some_and(|(word, _)| NUMBER_WORDS.contains(&word.as_str()))))
            {
                end += 1;
            }
            if let Some(value) = number_value(&raw[begin..end]) {
                token = format!("#{value}");
            } else {
                end = begin + 1; // uncertain grammar stays literal, never wildcard
            }
        } else if token.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(value) = token.parse::<u64>() {
                token = format!("#{value}");
            }
        }
        out.tokens.push(token);
        out.source.push(raw[begin].1);
        out.last_source.push(raw[end - 1].1);
        k = end;
    }
    out
}

/// Light phonetic folding: silent initial letters, `ph`, `ck`, doubled letters.
fn fold(w: &str) -> String {
    let mut s = w.to_string();
    for (a, b) in [
        ("kn", "n"),
        ("gn", "n"),
        ("wr", "r"),
        ("ps", "s"),
        ("wh", "w"),
    ] {
        if s.starts_with(a) && s.len() > a.len() {
            s = format!("{b}{}", &s[a.len()..]);
        }
    }
    let s = s.replace("ph", "f").replace("ck", "k");
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if !out.ends_with(c) {
            out.push(c);
        }
    }
    out
}

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Would a listener accept `heard` as the script word `said`?
pub fn same_word(said: &str, heard: &str, is_name: bool) -> bool {
    if said == heard {
        return true;
    }
    if critical(said) || critical(heard) {
        return false;
    }
    // Approximate equality is restricted to caller-declared names. Ordinary
    // edits can reverse meaning (not/now, can/cant) and are not homophones.
    if !is_name {
        return false;
    }
    let (a, b) = (fold(said), fold(heard));
    if a == b {
        return true;
    }
    let (ca, cb): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let d = edit_distance(&ca, &cb);
    let (short, long) = (ca.len().min(cb.len()), ca.len().max(cb.len()));
    if is_name {
        return short >= 2 && d * 2 <= long + 1;
    }
    // One edit on words that start alike ("know"/"no", "scales"/"scale"), or
    // up to a third of the letters on longer words ("ammut"/"ammit").
    (d == 1 && short >= 2 && ca[0] == cb[0]) || (short >= 4 && d * 3 <= long)
}

/// Compare `script` with the recognizer's `heard` words.
pub fn check(script: &str, heard: &[Word], names: &[String], limits: WordCheckLimits) -> WordCheck {
    let script = normalise(script);
    let heard_n = normalise_words(heard.iter().map(|w| w.text.as_str()));
    let names: Vec<String> = names.iter().flat_map(|n| normalise(n).tokens).collect();
    let (s, h) = (&script.tokens, &heard_n.tokens);
    let (n, m) = (s.len(), h.len());

    // Levenshtein alignment with tolerant equality.
    let eq: Vec<Vec<bool>> = s
        .iter()
        .map(|a| {
            h.iter()
                .map(|b| same_word(a, b, names.contains(a)))
                .collect()
        })
        .collect();
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in cost.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, value) in cost[0].iter_mut().enumerate() {
        *value = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let diag = cost[i - 1][j - 1] + usize::from(!eq[i - 1][j - 1]);
            cost[i][j] = diag.min(cost[i - 1][j] + 1).min(cost[i][j - 1] + 1);
        }
    }
    let mut ops = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + usize::from(!eq[i - 1][j - 1]) {
            ops.push(if eq[i - 1][j - 1] {
                Op::Match(i - 1, j - 1)
            } else {
                Op::Substitute(i - 1, j - 1)
            });
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            ops.push(Op::Missing(i - 1));
            i -= 1;
        } else {
            ops.push(Op::Extra(j - 1));
            j -= 1;
        }
    }
    ops.reverse();

    // Runs of mismatches, reported as phrases a person can find.
    let mut problems = Vec::new();
    let mut rejected: Option<String> = None;
    let reject = |why: String, rejected: &mut Option<String>| {
        if rejected.is_none() {
            *rejected = Some(why);
        }
    };
    let mut k = 0;
    let mut errors = 0usize;
    while k < ops.len() {
        if matches!(ops[k], Op::Match(..)) {
            k += 1;
            continue;
        }
        let start = k;
        while k < ops.len() && !matches!(ops[k], Op::Match(..)) {
            k += 1;
        }
        let run = &ops[start..k];
        errors += run.len();
        let said: Vec<&str> = run
            .iter()
            .filter_map(|o| match o {
                Op::Substitute(a, _) | Op::Missing(a) => Some(s[*a].as_str()),
                _ => None,
            })
            .collect();
        let got: Vec<&str> = run
            .iter()
            .filter_map(|o| match o {
                Op::Substitute(_, b) | Op::Extra(b) => Some(h[*b].as_str()),
                _ => None,
            })
            .collect();
        let desc = format!("said \"{}\", heard \"{}\"", said.join(" "), got.join(" "));
        let longest = |f: fn(&Op) -> bool| {
            let (mut best, mut cur) = (0, 0);
            for o in run {
                cur = if f(o) { cur + 1 } else { 0 };
                best = best.max(cur);
            }
            best
        };
        let missing = longest(|o| matches!(o, Op::Missing(_)));
        let extra = longest(|o| matches!(o, Op::Extra(_)));
        if said.iter().chain(got.iter()).any(|word| critical(word)) {
            reject(format!("number or negation differs: {desc}"), &mut rejected);
        } else if missing > limits.max_missing_run {
            reject(format!("dropped words: {desc}"), &mut rejected);
        } else if extra > limits.max_extra_run {
            reject(format!("repeated or invented words: {desc}"), &mut rejected);
        } else if run.len() > limits.max_bad_run {
            reject(format!("garbled: {desc}"), &mut rejected);
        }
        problems.push(desc);
    }
    let error_rate = if n == 0 {
        1.0
    } else {
        errors as f32 / n as f32
    };
    if error_rate > limits.max_error_rate {
        reject(
            format!("{:.0}% of words wrong", error_rate * 100.0),
            &mut rejected,
        );
    }
    WordCheck {
        error_rate,
        script_tokens: n,
        heard_tokens: m,
        problems,
        rejected,
        ops,
        script,
        heard: heard_n,
    }
}

/// A script word (or a run of words the check treats as one, like a spelled
/// number) with the time it was heard at.
type TokenGroup = (usize, usize, Option<(f32, f32)>);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TimedWord {
    /// The SCRIPT's text, punctuation included — what a caption should show.
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// False when the recognizer did not hear it and the time was
    /// interpolated from its neighbours.
    pub heard: bool,
}

/// Give every word of `script` a time, using the alignment in `check` (made
/// against the same `script` and `heard`). Script words that normalise to
/// nothing ("-", "—") are attached to the word before them, so the texts
/// concatenate back to the script.
pub fn script_word_times(script: &str, check: &WordCheck, heard: &[Word]) -> Vec<TimedWord> {
    let src: Vec<&str> = script.split_whitespace().collect();
    let s = &check.script;
    if s.tokens.is_empty() {
        return Vec::new();
    }
    // Groups: consecutive tokens covering overlapping word ranges
    // ("jackal-headed" is two tokens from one word).
    let mut groups: Vec<TokenGroup> = Vec::new();
    let mut group_of_token = vec![0usize; s.tokens.len()];
    for (i, group) in group_of_token.iter_mut().enumerate() {
        match groups.last_mut() {
            Some(g) if s.source[i] <= g.1 => g.1 = g.1.max(s.last_source[i]),
            _ => groups.push((s.source[i], s.last_source[i], None)),
        }
        *group = groups.len() - 1;
    }
    let mut uncertain = vec![false; groups.len()];
    for op in &check.ops {
        match *op {
            Op::Missing(i) | Op::Substitute(i, _) => uncertain[group_of_token[i]] = true,
            _ => {}
        }
        if let Op::Match(i, j) | Op::Substitute(i, j) = *op {
            let h = &heard[check.heard.source[j]];
            let end = heard[check.heard.last_source[j]].end;
            let g = &mut groups[group_of_token[i]].2;
            *g = Some(match *g {
                Some((a, b)) => (a.min(h.start), b.max(end)),
                None => (h.start, end),
            });
        }
    }
    let total_end = heard.last().map_or(0.0, |w| w.end);
    let mut out: Vec<TimedWord> = Vec::with_capacity(groups.len());
    for (k, &(first, last, time)) in groups.iter().enumerate() {
        // Token-less words before the first group belong to it; any others
        // to the group before them.
        let from = if k == 0 { 0 } else { first };
        let to = (groups.get(k + 1).map_or(src.len(), |g| g.0) - 1).max(last);
        let text = src[from..=to].join(" ");
        let (start, end, heard) = match time {
            Some((a, b)) => (a, b, !uncertain[k]),
            None => {
                let prev = out.last().map_or(0.0, |w| w.end);
                let next = groups[k + 1..]
                    .iter()
                    .find_map(|g| g.2.map(|t| t.0))
                    .unwrap_or(total_end.max(prev));
                (prev, next.max(prev), false)
            }
        };
        out.push(TimedWord {
            text,
            start,
            end,
            heard,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heard(text: &str) -> Vec<Word> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, w)| Word {
                text: w.to_string(),
                start: i as f32 * 0.3,
                end: i as f32 * 0.3 + 0.25,
                probability: 0.9,
            })
            .collect()
    }

    fn verdict(script: &str, got: &str) -> Option<String> {
        check(script, &heard(got), &[], WordCheckLimits::default()).rejected
    }

    #[test]
    fn numbers_spelled_and_in_digits_match() {
        assert_eq!(
            verdict(
                "But Spell One Hundred and Twenty-Five stages something else.",
                "But spell 125 stages something else."
            ),
            None
        );
        assert_eq!(
            verdict(
                "It cost 1,250 coins.",
                "It cost twelve hundred and fifty coins."
            ),
            None
        );
    }

    #[test]
    fn fuzzy_spelling_does_not_silently_erase_differences() {
        let script = "The Egyptians called that order Ma'at. You KNOW the nomes, and Ammut waits.";
        let got = "The Egyptians called that order mart. You no the gnomes, and Ammit waits.";
        let c = check(script, &heard(got), &[], WordCheckLimits::default());
        assert!(!c.problems.is_empty());
        assert!(c.rejected.is_some());
        assert!(!same_word("not", "now", false));
        assert!(!same_word("can", "cant", false));
    }

    #[test]
    fn wrong_numbers_and_missing_negations_never_pass_as_isolated_noise() {
        for (script, got) in [
            (
                "The measurement of the bridge is 125 feet in total.",
                "The measurement of the bridge is 126 feet in total.",
            ),
            (
                "The measurement of the bridge is 3.5 feet in total.",
                "The measurement of the bridge is 35 feet in total.",
            ),
            (
                "The speaker did not steal the gold from the temple.",
                "The speaker did steal the gold from the temple.",
            ),
            (
                "The speaker never stole the gold from the temple.",
                "The speaker stole the gold from the temple.",
            ),
        ] {
            assert!(verdict(script, got).is_some(), "{script} / {got}");
        }
    }

    #[test]
    fn a_dropped_phrase_is_rejected() {
        let script = "Ma'at meant truth, justice, balance, and the proper shape of life.";
        let r = verdict(script, "Ma'at meant truth, justice, balance, and life.");
        assert!(r.as_ref().is_some_and(|r| r.contains("dropped")), "{r:?}");
    }

    #[test]
    fn a_repeated_phrase_is_rejected() {
        let script = "Then the heart faces the scales. Egyptians did not confess.";
        let r = verdict(script, "Then the heart faces the scales. The heart faces the scales. Egyptians did not confess.");
        assert!(r.as_ref().is_some_and(|r| r.contains("repeated")), "{r:?}");
    }

    #[test]
    fn a_garbled_stretch_is_rejected() {
        let script = "A feather could carry all that weight across the hall.";
        let r = verdict(
            script,
            "A feather could marry bold flat bait across the hall.",
        );
        assert!(r.is_some(), "{r:?}");
    }

    #[test]
    fn one_isolated_miss_is_tolerated_and_reported() {
        let script = "The dead person must speak while the evidence remains inside his own chest.";
        let c = check(
            script,
            &heard("The dead person must speak while the evidence remains inside his own guest."),
            &[],
            WordCheckLimits::default(),
        );
        assert_eq!(c.rejected, None);
        assert_eq!(c.problems.len(), 1, "{:?}", c.problems);
    }

    #[test]
    fn listed_names_get_a_looser_match() {
        let script = "Then Thoth records the verdict.";
        let got = heard("Then tot records the verdict.");
        assert!(
            check(script, &got, &["Thoth".into()], WordCheckLimits::default())
                .problems
                .is_empty()
        );
    }

    #[test]
    fn signs_decimals_and_alphanumeric_identifiers_stay_distinct() {
        for (a, b) in [("-5", "5"), ("3.5", "35"), ("B52", "52"), ("125", "126")] {
            let a = normalise(a);
            let b = normalise(b);
            assert_ne!(a.tokens, b.tokens);
        }
    }

    #[test]
    fn collapsed_number_timing_includes_every_recognized_word() {
        let h = heard("It cost twelve hundred and fifty coins.");
        let c = check("It cost 1250 coins.", &h, &[], WordCheckLimits::default());
        let timed = script_word_times("It cost 1250 coins.", &c, &h);
        assert_eq!(timed[2].start, h[2].start);
        assert_eq!(timed[2].end, h[5].end);
    }

    #[test]
    fn normalisation_keeps_the_source_word() {
        let n = normalise("A jackal-headed attendant, 1,250 of them.");
        assert_eq!(
            n.tokens,
            ["a", "jackal", "headed", "attendant", "#1250", "of", "them"]
        );
        assert_eq!(n.source, [0, 1, 1, 2, 3, 4, 5]);
        let n = normalise("Spell One Hundred and Twenty-Five stages");
        assert_eq!(n.tokens, ["spell", "#125", "stages"]);
        assert_eq!((n.source[1], n.last_source[1]), (1, 4));
    }

    #[test]
    fn every_script_word_gets_a_time_and_the_text_survives() {
        let script = "But Spell One Hundred and Twenty-Five stages - something else.";
        let got = heard("But spell 125 stages something else.");
        let c = check(script, &got, &[], WordCheckLimits::default());
        let t = script_word_times(script, &c, &got);
        let texts: Vec<&str> = t.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "But",
                "Spell",
                "One Hundred and Twenty-Five",
                "stages -",
                "something",
                "else."
            ]
        );
        assert_eq!(texts.join(" "), script);
        assert!(t.iter().all(|w| w.heard));
        assert!(t.windows(2).all(|p| p[0].start <= p[1].start));
    }

    #[test]
    fn a_word_that_was_not_heard_is_interpolated() {
        let script = "Truth, justice, balance, and life.";
        let got = heard("Truth, balance, and life.");
        let c = check(script, &got, &[], WordCheckLimits::default());
        let t = script_word_times(script, &c, &got);
        let justice = t.iter().find(|w| w.text == "justice,").unwrap();
        assert!(!justice.heard);
        let truth = t.iter().find(|w| w.text == "Truth,").unwrap();
        let balance = t.iter().find(|w| w.text == "balance,").unwrap();
        assert!(
            justice.start >= truth.end && justice.end <= balance.start,
            "{t:?}"
        );
    }
}
