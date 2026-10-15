use super::error::{DirsError, ErrorCode, Result};

/// One path segment below an area. The strictest of the three platforms on purpose (a layout is portable).
/// Returns the reason, or None when valid. Mirrors `segmentProblem` in the TypeScript implementation.
pub fn segment_problem(s: &str) -> Option<&'static str> {
    if s.is_empty() {
        return Some("empty");
    }
    if s.chars().count() > 255 {
        return Some("longer than 255 characters");
    }
    if s == "." || s == ".." {
        return Some("relative reference");
    }
    if s.chars().any(|c| (c as u32) < 0x20 || c as u32 == 0x7f || "<>:\"|?*\\/".contains(c)) {
        return Some("contains a separator, drive, stream or reserved character");
    }
    if s.ends_with('.') || s.ends_with(' ') {
        return Some("ends in a dot or space");
    }
    let stem = s.split('.').next().unwrap_or(s).trim_end().to_ascii_lowercase();
    if is_device_name(&stem) {
        return Some("Windows device name");
    }
    None
}

fn is_device_name(stem: &str) -> bool {
    if matches!(stem, "con" | "prn" | "aux" | "nul" | "conin$" | "conout$") {
        return true;
    }
    for prefix in ["com", "lpt"] {
        if let Some(rest) = stem.strip_prefix(prefix) {
            let mut chars = rest.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                if c.is_ascii_digit() || matches!(c, '\u{b9}' | '\u{b2}' | '\u{b3}') {
                    return true;
                }
            }
        }
    }
    false
}

pub fn assert_segment(s: &str, what: &str) -> Result<()> {
    match segment_problem(s) {
        // The value itself is never echoed: a segment can be a secret's name.
        Some(p) => Err(DirsError::new(ErrorCode::InvalidSegment, format!("invalid {what}: {p}"))),
        None => Ok(()),
    }
}
