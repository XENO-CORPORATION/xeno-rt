use super::error::{DirsError, ErrorCode, Result};
use super::registry::inputs;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    MacOs,
    Linux,
}

impl Platform {
    pub fn parse(s: &str) -> Option<Platform> {
        match s {
            "windows" => Some(Platform::Windows),
            "macos" => Some(Platform::MacOs),
            "linux" => Some(Platform::Linux),
            _ => None,
        }
    }
    pub fn host() -> Platform {
        if cfg!(windows) {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Linux
        }
    }
    pub fn sep(self) -> char {
        if self == Platform::Windows { '\\' } else { '/' }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Config,
    Data,
    Secrets,
    State,
    Cache,
    Run,
}

pub const KINDS: [Kind; 6] = [Kind::Config, Kind::Data, Kind::Secrets, Kind::State, Kind::Cache, Kind::Run];

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Config => "config",
            Kind::Data => "data",
            Kind::Secrets => "secrets",
            Kind::State => "state",
            Kind::Cache => "cache",
            Kind::Run => "run",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        KINDS.iter().copied().find(|k| k.as_str() == s)
    }
}

#[derive(Debug, Clone)]
pub struct ResolveInput {
    pub platform: Platform,
    pub home: String,
    pub env: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedRoots {
    pub platform: Platform,
    roots: [String; 6],
    /// Sorted, `code` or `code:detail`.
    pub warnings: Vec<String>,
    pub overridden: bool,
}

impl ResolvedRoots {
    pub fn root(&self, kind: Kind) -> &str {
        &self.roots[KINDS.iter().position(|k| *k == kind).unwrap()]
    }
}

pub fn is_absolute(platform: Platform, p: &str) -> bool {
    if platform == Platform::Windows {
        let b = p.as_bytes();
        let drive = b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/');
        let unc = b.len() >= 3 && (b[0] == b'\\' || b[0] == b'/') && (b[1] == b'\\' || b[1] == b'/') && b[2] != b'\\' && b[2] != b'/';
        drive || unc
    } else {
        p.starts_with('/')
    }
}

/// Normalises separators and drops trailing ones, so `base + sep + name` is always well formed.
pub fn normalize_base(platform: Platform, p: &str) -> String {
    if platform == Platform::Windows {
        let unc = p.starts_with(['\\', '/']) && p[1..].starts_with(['\\', '/']);
        let mut out = String::new();
        let mut prev_sep = false;
        for c in p.chars() {
            let is_sep = c == '\\' || c == '/';
            if is_sep {
                if !prev_sep {
                    out.push('\\');
                }
            } else {
                out.push(c);
            }
            prev_sep = is_sep;
        }
        if unc {
            out.insert(0, '\\');
        }
        out.trim_end_matches('\\').to_string()
    } else {
        p.trim_end_matches('/').to_string()
    }
}

pub fn join_path(platform: Platform, base: &str, parts: &[&str]) -> String {
    let sep = platform.sep().to_string();
    let mut v: Vec<&str> = vec![base];
    v.extend_from_slice(parts);
    v.join(&sep)
}

fn get_env(platform: Platform, env: &HashMap<String, String>, name: &str) -> Option<String> {
    if let Some(v) = env.get(name) {
        return Some(v.clone());
    }
    if platform == Platform::Windows {
        return env.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone());
    }
    None
}

pub fn resolve_roots(input: &ResolveInput) -> Result<ResolvedRoots> {
    let platform = input.platform;
    let mut warnings: Vec<String> = Vec::new();
    let sep = platform.sep();
    let get = |n: &str| get_env(platform, &input.env, n);
    let abs = |p: &str| is_absolute(platform, p);

    let need_home = || -> Result<String> {
        if !input.home.is_empty() && abs(&input.home) {
            Ok(normalize_base(platform, &input.home))
        } else {
            Err(DirsError::new(ErrorCode::HomeUnresolvable, "the home directory is unknown or relative, and no XENO_HOME override is set"))
        }
    };

    if let Some(o) = get("XENO_HOME").filter(|v| !v.is_empty()) {
        if abs(&o) {
            let base = normalize_base(platform, &o);
            let roots = KINDS.map(|k| format!("{base}{sep}{}", k.as_str()));
            return Ok(ResolvedRoots { platform, roots, warnings: vec![], overridden: true });
        }
        warnings.push("xeno-home-relative-ignored".into());
    }

    // An absolute variable is used; a relative one is ignored with a warning; an empty one is unset.
    let mut dir_var = |name: &str, warn_code: &str| -> Option<String> {
        let v = get(name).filter(|v| !v.is_empty())?;
        if abs(&v) {
            Some(normalize_base(platform, &v))
        } else {
            warnings.push(format!("{warn_code}:{name}"));
            None
        }
    };

    let inp = inputs();
    let roots: [String; 6];
    match platform {
        Platform::Windows => {
            let roaming = match dir_var("APPDATA", "platform-dir-relative-ignored") {
                Some(v) => v,
                None => format!("{}\\AppData\\Roaming", need_home()?),
            };
            let local = match dir_var("LOCALAPPDATA", "platform-dir-relative-ignored") {
                Some(v) => v,
                None => format!("{}\\AppData\\Local", need_home()?),
            };
            let v = &inp.windows_vendor_dir;
            roots = [
                format!("{roaming}\\{v}\\config"),
                format!("{roaming}\\{v}\\data"),
                format!("{roaming}\\{v}\\secrets"),
                format!("{local}\\{v}\\state"),
                format!("{local}\\{v}\\cache"),
                format!("{local}\\{v}\\run"),
            ];
        }
        Platform::MacOs => {
            let h = need_home()?;
            let id = &inp.macos_bundle_id;
            let support = format!("{h}/Library/Application Support/{id}");
            let run = match get("TMPDIR").filter(|t| !t.is_empty() && abs(t)) {
                Some(t) => format!("{}/{id}", normalize_base(platform, &t)),
                None => {
                    warnings.push("run-dir-fallback:TMPDIR".into());
                    format!("{support}/state/run")
                }
            };
            roots = [
                format!("{support}/config"),
                format!("{support}/data"),
                format!("{support}/secrets"),
                format!("{support}/state"),
                format!("{h}/Library/Caches/{id}"),
                run,
            ];
        }
        Platform::Linux => {
            let n = &inp.linux_dir_name;
            let mut xdg = |name: &str, rel: &str| -> Result<String> {
                let base = match dir_var(name, "xdg-relative-ignored") {
                    Some(v) => v,
                    None => format!("{}/{rel}", need_home()?),
                };
                Ok(format!("{base}/{n}"))
            };
            let config = xdg("XDG_CONFIG_HOME", ".config")?;
            let data = xdg("XDG_DATA_HOME", ".local/share")?;
            let state = xdg("XDG_STATE_HOME", ".local/state")?;
            let cache = xdg("XDG_CACHE_HOME", ".cache")?;
            let rt = dir_var("XDG_RUNTIME_DIR", "xdg-relative-ignored");
            let run = match rt {
                Some(r) => format!("{r}/{n}"),
                None => {
                    warnings.push("run-dir-fallback:XDG_RUNTIME_DIR".into());
                    format!("{state}/run")
                }
            };
            let secrets = format!("{data}/secrets");
            roots = [config, data, secrets, state, cache, run];
        }
    }
    warnings.sort();
    Ok(ResolvedRoots { platform, roots, warnings, overridden: false })
}
