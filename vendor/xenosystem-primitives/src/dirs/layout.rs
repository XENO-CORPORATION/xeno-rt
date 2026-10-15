use super::json;
use super::registry::inputs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutState {
    Missing,
    Ok(u32),
    Newer(u32),
    Invalid(String),
}

/// The layout version this build understands (`data/layout.json`, spec section 8).
pub fn supported_layout() -> u32 {
    inputs().layout
}

/// Pure parser for the contents of layout.json (None = the file does not exist).
pub fn parse_layout(text: Option<&str>, supported: u32) -> LayoutState {
    let Some(text) = text else { return LayoutState::Missing };
    let v = match json::parse(text) {
        Ok(v) => v,
        Err(_) => return LayoutState::Invalid("not valid JSON".into()),
    };
    if !matches!(v, json::Value::Obj(_)) {
        return LayoutState::Invalid("not an object".into());
    }
    match v.get("layout").and_then(json::Value::as_i64) {
        Some(n) if n >= 1 => {
            let n = n as u32;
            if n > supported { LayoutState::Newer(n) } else { LayoutState::Ok(n) }
        }
        _ => LayoutState::Invalid("layout is not a positive integer".into()),
    }
}
