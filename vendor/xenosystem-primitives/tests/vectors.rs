//! The shared contract vectors (../../contract/vectors), parsed with serde_json as an oracle independent of the
//! crate's own JSON reader. A vector the TypeScript implementation passes and this one fails is a contract defect.
#![cfg(feature = "dirs")]

use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use xenosystem_primitives::dirs::*;

fn load(name: &str) -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contract/vectors").join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn code(r: Result<impl std::fmt::Debug>) -> String {
    match r {
        Ok(v) => format!("OK({v:?})"),
        Err(e) => e.code.as_str().to_string(),
    }
}

#[test]
fn roots_vectors() {
    let v = load("roots.json");
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 25);
    for c in cases {
        let id = c["id"].as_str().unwrap();
        let env: HashMap<String, String> = c["input"]["env"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
        let input = ResolveInput { platform: Platform::parse(c["input"]["platform"].as_str().unwrap()).unwrap(), home: c["input"]["home"].as_str().unwrap().to_string(), env };
        let expect = &c["expect"];
        if let Some(e) = expect.get("error") {
            assert_eq!(code(resolve_roots(&input)), e.as_str().unwrap(), "{id}");
            continue;
        }
        let got = resolve_roots(&input).unwrap_or_else(|e| panic!("{id}: {e}"));
        for k in KINDS {
            assert_eq!(got.root(k), expect["roots"][k.as_str()].as_str().unwrap(), "{id} {}", k.as_str());
        }
        let want: BTreeSet<&str> = expect["warnings"].as_array().unwrap().iter().map(|w| w.as_str().unwrap()).collect();
        let have: BTreeSet<&str> = got.warnings.iter().map(String::as_str).collect();
        assert_eq!(have, want, "{id} warnings");
    }
}

#[test]
fn path_vectors() {
    let v = load("paths.json");
    let cases = v["cases"].as_array().unwrap();
    assert!(cases.len() >= 30);
    for c in cases {
        let id = c["id"].as_str().unwrap();
        let segs: Vec<&str> = c["segments"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
        let r = registered_relative_path(c["kind"].as_str().unwrap(), c["area"].as_str().unwrap(), &segs);
        match (c["expect"].get("ok"), c["expect"].get("error")) {
            (Some(ok), _) => assert_eq!(r.unwrap_or_else(|e| panic!("{id}: {e}")), ok.as_str().unwrap(), "{id}"),
            (_, Some(err)) => assert_eq!(code(r), err.as_str().unwrap(), "{id}"),
            _ => panic!("{id}: malformed vector"),
        }
    }
}

#[test]
fn lock_vectors() {
    let v = load("locks.json");
    for c in v["cases"].as_array().unwrap() {
        let id = c["id"].as_str().unwrap();
        let r = lock_relative_path(c["area"].as_str().unwrap(), c["name"].as_str().unwrap());
        match (c["expect"].get("ok"), c["expect"].get("error")) {
            (Some(ok), _) => assert_eq!(r.unwrap(), ok.as_str().unwrap(), "{id}"),
            (_, Some(err)) => assert_eq!(code(r), err.as_str().unwrap(), "{id}"),
            _ => panic!("{id}: malformed vector"),
        }
    }
}

#[test]
fn layout_vectors() {
    let v = load("layout.json");
    let supported = v["supported"].as_u64().unwrap() as u32;
    for c in v["cases"].as_array().unwrap() {
        let id = c["id"].as_str().unwrap();
        let state = parse_layout(c["file"].as_str(), supported);
        let name = match state {
            LayoutState::Missing => "missing",
            LayoutState::Ok(_) => "ok",
            LayoutState::Newer(_) => "newer",
            LayoutState::Invalid(_) => "invalid",
        };
        assert_eq!(name, c["expect"].as_str().unwrap(), "{id}");
    }
}

#[test]
fn contract_inputs_are_the_owner_decisions() {
    let i = inputs();
    assert_eq!(i.macos_bundle_id, "ai.xenosystem");
    assert_eq!(i.legacy_fallback_releases.value, 3);
    assert_eq!(i.legacy_fallback_releases.status, "locked");
    assert_eq!(i.layout, supported_layout());
}
