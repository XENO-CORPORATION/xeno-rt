//! `xeno home doctor`: lists every entry under each root that the registry does not allow (spec section 8, check 2).
//! Registration is by area; what is inside an area belongs to its owner and is not inspected. A root that does not
//! exist yet is fine. Reads only; never changes anything.

use super::layout::LayoutState;
use super::registry::{find_area, inputs, registry};
use super::roots::{Kind, KINDS};
use super::xdirs::XenoDirs;
use std::fs;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unregistered {
    pub kind: Kind,
    /// Path relative to the kind root, `/` separated.
    pub path: String,
    pub entry_type: &'static str,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct LegacyPresence {
    pub name: String,
    pub path: String,
    pub present: bool,
}

#[derive(Debug)]
pub struct DoctorReport {
    pub ok: bool,
    pub layout: LayoutState,
    pub unregistered: Vec<Unregistered>,
    pub legacy: Vec<LegacyPresence>,
}

fn type_of(p: &str) -> &'static str {
    match fs::symlink_metadata(p) {
        Ok(m) if m.file_type().is_symlink() => "link",
        Ok(m) if m.is_dir() => "directory",
        Ok(m) if m.is_file() => "file",
        _ => "other",
    }
}

fn list(p: &str) -> Vec<String> {
    fs::read_dir(p).map(|it| it.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default()
}

fn inside(parent: &str, child: &str, sep: char) -> bool {
    let norm = |x: &str| if sep == '\\' { x.to_lowercase() } else { x.to_string() };
    norm(child).starts_with(&format!("{}{sep}", norm(parent)))
}

pub fn doctor(dirs: &XenoDirs, home: Option<&str>) -> DoctorReport {
    let sep = dirs.platform().sep();
    let mut unregistered = Vec::new();
    for kind in KINDS {
        let root = dirs.root(kind).to_string();
        // Roots nested in another root (Linux secrets under data; the run fallback under state) are allowed there.
        let mut top_allowed: Vec<String> = KINDS
            .iter()
            .filter(|k| **k != kind && inside(&root, dirs.root(**k), sep))
            .map(|k| dirs.root(*k)[root.len() + 1..].split(sep).next().unwrap().to_string())
            .collect();
        if kind == Kind::Data {
            top_allowed.push("layout.json".into());
        }
        let areas_of_kind: Vec<_> = registry().areas.iter().filter(|a| kind == Kind::Run || a.uses.contains(&kind)).collect();
        for name in list(&root) {
            let abs = format!("{root}{sep}{name}");
            if top_allowed.contains(&name) {
                continue;
            }
            let top: Vec<_> = areas_of_kind.iter().filter(|a| a.area.split('/').next() == Some(name.as_str())).collect();
            if top.is_empty() {
                let elsewhere = registry().areas.iter().any(|a| a.area.split('/').next() == Some(name.as_str()));
                unregistered.push(Unregistered {
                    kind,
                    path: name.clone(),
                    entry_type: type_of(&abs),
                    reason: if elsewhere { format!("area {name} is not registered for kind {}", kind.as_str()) } else { "not a registered area".into() },
                });
                continue;
            }
            if !top.iter().any(|a| a.area.contains('/')) {
                continue;
            }
            for child in list(&abs) {
                let full = format!("{name}/{child}");
                match find_area(&full) {
                    Some(e) if kind == Kind::Run || e.uses.contains(&kind) => {}
                    other => unregistered.push(Unregistered {
                        kind,
                        path: full.clone(),
                        entry_type: type_of(&format!("{abs}{sep}{child}")),
                        reason: if other.is_none() { "not a registered area".into() } else { format!("area {full} is not registered for kind {}", kind.as_str()) },
                    }),
                }
            }
        }
    }
    unregistered.sort_by(|a, b| format!("{}{}", a.kind.as_str(), a.path).cmp(&format!("{}{}", b.kind.as_str(), b.path)));
    let legacy = home
        .map(|h| {
            inputs()
                .legacy_homes
                .iter()
                .map(|name| {
                    let path = format!("{h}{sep}{name}");
                    let present = fs::symlink_metadata(&path).is_ok();
                    LegacyPresence { name: name.clone(), path, present }
                })
                .collect()
        })
        .unwrap_or_default();
    DoctorReport { ok: unregistered.is_empty(), layout: dirs.read_layout(), unregistered, legacy }
}
