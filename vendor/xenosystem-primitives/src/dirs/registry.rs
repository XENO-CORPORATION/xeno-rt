//! The embedded contract: `contract/registry.json` and `contract/inputs.json`, copied into this crate by
//! `scripts/sync-contract.mjs` (a CI gate fails on drift). Parsed once.

use super::error::{DirsError, ErrorCode, Result};
use super::json::{self, Value};
use super::roots::{Kind, KINDS};
use super::segments::assert_segment;
use std::sync::OnceLock;

const REGISTRY_JSON: &str = include_str!("../../contract/registry.json");
const INPUTS_JSON: &str = include_str!("../../contract/inputs.json");

#[derive(Debug, Clone)]
pub struct AreaEntry {
    pub area: String,
    pub owner: String,
    pub uses: Vec<Kind>,
    /// For `apps/<slug>`: the max slug length (the contract pattern is `^[a-z][a-z0-9-]{0,N}$`).
    slug_max_extra: Option<usize>,
    prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InputValue<T> {
    pub value: T,
    pub status: String,
    pub decided: String,
}

#[derive(Debug, Clone)]
pub struct Inputs {
    pub layout: u32,
    pub macos_bundle_id: String,
    pub macos_bundle_id_input: InputValue<String>,
    pub legacy_fallback_releases: InputValue<u32>,
    pub legacy_homes: Vec<String>,
    pub windows_vendor_dir: String,
    pub linux_dir_name: String,
    pub secrets_key_service: String,
    pub secrets_key_account: String,
}

pub struct Registry {
    pub areas: Vec<AreaEntry>,
    pub lock_suffix: String,
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_else(|| panic!("contract: missing string {k}")).to_string()
}

/// Parses `^[a-z][a-z0-9-]{0,N}$`; anything else is a contract this build does not implement, and fails loudly.
fn parse_slug_pattern(p: &str) -> usize {
    let head = "^[a-z][a-z0-9-]{0,";
    let tail = "}$";
    let n = p.strip_prefix(head).and_then(|r| r.strip_suffix(tail)).and_then(|n| n.parse::<usize>().ok());
    n.unwrap_or_else(|| panic!("contract: unsupported slugPattern {p:?}"))
}

pub fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| {
        let v = json::parse(REGISTRY_JSON).expect("contract/registry.json must parse");
        let mut areas = Vec::new();
        for a in v.get("areas").and_then(Value::as_array).expect("areas") {
            let name = s(a, "area");
            let uses = a
                .get("uses")
                .and_then(Value::as_array)
                .expect("uses")
                .iter()
                .map(|k| Kind::parse(k.as_str().unwrap_or("")).unwrap_or_else(|| panic!("contract: unknown kind in {name}")))
                .collect();
            let (slug_max_extra, prefix) = match a.get("slugPattern").and_then(Value::as_str) {
                Some(p) => {
                    let cut = name.find('<').expect("slug area must contain <slug>");
                    (Some(parse_slug_pattern(p)), Some(name[..cut].to_string()))
                }
                None => (None, None),
            };
            areas.push(AreaEntry { area: name, owner: s(a, "owner"), uses, slug_max_extra, prefix });
        }
        let lock_suffix = v.get("locks").map(|l| s(l, "suffix")).expect("locks.suffix");
        Registry { areas, lock_suffix }
    })
}

pub fn inputs() -> &'static Inputs {
    static I: OnceLock<Inputs> = OnceLock::new();
    I.get_or_init(|| {
        let v = json::parse(INPUTS_JSON).expect("contract/inputs.json must parse");
        let inp = v.get("inputs").expect("inputs");
        let iv = |name: &str| -> (&Value, String, String) {
            let x = inp.get(name).unwrap_or_else(|| panic!("contract: missing input {name}"));
            (x.get("value").expect("value"), s(x, "status"), s(x, "decided"))
        };
        let (b, bs, bd) = iv("macosBundleId");
        let (f, fs, fd) = iv("legacyFallbackReleases");
        let bundle = b.as_str().expect("bundle id").to_string();
        Inputs {
            layout: v.get("layout").and_then(Value::as_i64).expect("layout") as u32,
            macos_bundle_id: bundle.clone(),
            macos_bundle_id_input: InputValue { value: bundle, status: bs, decided: bd },
            legacy_fallback_releases: InputValue { value: f.as_i64().expect("releases") as u32, status: fs, decided: fd },
            legacy_homes: v.get("legacyHomes").and_then(Value::as_array).expect("legacyHomes").iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            windows_vendor_dir: s(&v, "windowsVendorDir"),
            linux_dir_name: s(&v, "linuxDirName"),
            secrets_key_service: s(&v, "secretsKeyService"),
            secrets_key_account: s(&v, "secretsKeyAccount"),
        }
    })
}

/// Finds the registered entry for a concrete area name. `apps/<slug>` matches by slug pattern.
pub fn find_area(area: &str) -> Option<&'static AreaEntry> {
    registry().areas.iter().find(|e| match (&e.prefix, e.slug_max_extra) {
        (Some(prefix), Some(extra)) => area.strip_prefix(prefix.as_str()).is_some_and(|slug| {
            let b = slug.as_bytes();
            !b.is_empty()
                && b.len() <= extra + 1
                && b[0].is_ascii_lowercase()
                && b[1..].iter().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'-')
        }),
        _ => e.area == area,
    })
}

/// Resolves (kind, area, segments) against the registry; returns the path relative to the kind root with `/`.
pub fn registered_relative_path(kind: &str, area: &str, segments: &[&str]) -> Result<String> {
    let kind = Kind::parse(kind).ok_or_else(|| DirsError::new(ErrorCode::UnregisteredArea, "unknown kind"))?;
    let entry = find_area(area).ok_or_else(|| DirsError::new(ErrorCode::UnregisteredArea, format!("area is not registered: {area:?}")))?;
    if !entry.uses.contains(&kind) {
        return Err(if kind == Kind::Run {
            DirsError::new(ErrorCode::LocksOnly, format!("run/{area} holds lock files only; use lock_path"))
        } else {
            DirsError::new(ErrorCode::KindNotUsed, format!("area {area} does not use kind {}", kind.as_str()))
        });
    }
    for seg in segments {
        assert_segment(seg, "segment")?;
    }
    let mut out = area.to_string();
    for seg in segments {
        out.push('/');
        out.push_str(seg);
    }
    Ok(out)
}

/// Relative path of a lock file under the run root. Every registered area may hold locks (spec section 8).
pub fn lock_relative_path(area: &str, name: &str) -> Result<String> {
    find_area(area).ok_or_else(|| DirsError::new(ErrorCode::UnregisteredArea, format!("area is not registered: {area:?}")))?;
    assert_segment(name, "lock name")?;
    Ok(format!("{area}/{name}{}", registry().lock_suffix))
}

/// Kinds in declaration order, for callers that iterate.
pub fn all_kinds() -> [Kind; 6] {
    KINDS
}
