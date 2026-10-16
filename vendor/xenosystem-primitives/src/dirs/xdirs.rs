use super::atomic::atomic_write_file;
use super::error::{DirsError, ErrorCode, Result};
use super::layout::{parse_layout, supported_layout, LayoutState};
use super::lock::{acquire_lock_file, AcquireOptions, LockHandle};
use super::registry::{lock_relative_path, registered_relative_path};
use super::roots::{join_path, resolve_roots, Kind, Platform, ResolveInput, ResolvedRoots};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The resolver. Every XENO path comes from here; nothing else may build one. It refuses anything the registry does
/// not list, and refuses to write when the layout on disk is newer than this build understands.
#[derive(Debug, Clone)]
pub struct XenoDirs {
    resolved: ResolvedRoots,
}

#[derive(Default)]
pub struct CreateDirsOptions {
    pub platform: Option<Platform>,
    pub home: Option<String>,
    pub env: Option<HashMap<String, String>>,
}

fn host_home() -> String {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    // The one place in the ecosystem allowed to ask the OS environment for the home directory.
    std::env::var(var).unwrap_or_default()
}

impl XenoDirs {
    pub fn new(resolved: ResolvedRoots) -> Self {
        XenoDirs { resolved }
    }
    pub fn platform(&self) -> Platform {
        self.resolved.platform
    }
    pub fn warnings(&self) -> &[String] {
        &self.resolved.warnings
    }
    pub fn overridden(&self) -> bool {
        self.resolved.overridden
    }
    pub fn root(&self, kind: Kind) -> &str {
        self.resolved.root(kind)
    }

    /// Path relative to the kind root, `/`-separated, after registry validation.
    pub fn relative(&self, kind: &str, area: &str, segments: &[&str]) -> Result<String> {
        registered_relative_path(kind, area, segments)
    }

    /// Absolute path for (kind, area, segments). Errors for anything unregistered or invalid.
    pub fn path(&self, kind: Kind, area: &str, segments: &[&str]) -> Result<PathBuf> {
        let rel = registered_relative_path(kind.as_str(), area, segments)?;
        let parts: Vec<&str> = rel.split('/').collect();
        Ok(PathBuf::from(join_path(self.platform(), self.root(kind), &parts)))
    }

    /// `run/<area>/<name>.lock`.
    pub fn lock_path(&self, area: &str, name: &str) -> Result<PathBuf> {
        let rel = lock_relative_path(area, name)?;
        let parts: Vec<&str> = rel.split('/').collect();
        Ok(PathBuf::from(join_path(self.platform(), self.root(Kind::Run), &parts)))
    }

    /// `data/layout.json`.
    pub fn layout_path(&self) -> PathBuf {
        PathBuf::from(join_path(self.platform(), self.root(Kind::Data), &["layout.json"]))
    }

    /// `<kind>/<area>/area.json`, the optional per-area schema marker.
    pub fn area_meta_path(&self, kind: Kind, area: &str) -> Result<PathBuf> {
        self.path(kind, area, &["area.json"])
    }

    pub fn read_layout(&self) -> LayoutState {
        match std::fs::read_to_string(self.layout_path()) {
            Ok(t) => parse_layout(Some(&t), supported_layout()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => LayoutState::Missing,
            Err(_) => LayoutState::Invalid("unreadable".into()),
        }
    }

    /// Errors unless this build may write: a newer or unreadable layout marker refuses (spec section 8).
    pub fn assert_writable(&self) -> Result<LayoutState> {
        match self.read_layout() {
            LayoutState::Newer(n) => Err(DirsError::new(
                ErrorCode::LayoutNewer,
                format!("the XENO layout on disk is version {n}; this build understands up to {}. Update the product before it writes.", supported_layout()),
            )),
            LayoutState::Invalid(r) => Err(DirsError::new(ErrorCode::LayoutInvalid, format!("data/layout.json is not readable ({r}); refusing to write"))),
            s => Ok(s),
        }
    }

    /// Creates data/layout.json at the supported version if absent; otherwise checks it.
    pub fn ensure_layout(&self) -> Result<LayoutState> {
        let s = self.assert_writable()?;
        if s == LayoutState::Missing {
            atomic_write_file(&self.layout_path(), format!("{{\"layout\":{}}}\n", supported_layout()).as_bytes(), None)?;
            return Ok(LayoutState::Ok(supported_layout()));
        }
        Ok(s)
    }

    /// The only sanctioned way to write below the roots: registry check, layout check, atomic replace.
    pub fn write_file(&self, kind: Kind, area: &str, segments: &[&str], data: &[u8]) -> Result<PathBuf> {
        let target = self.path(kind, area, segments)?;
        self.ensure_layout()?;
        atomic_write_file(&target, data, if kind == Kind::Secrets { Some(0o600) } else { None })?;
        Ok(target)
    }

    pub fn read_file(&self, kind: Kind, area: &str, segments: &[&str]) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path(kind, area, segments)?) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Acquires `run/<area>/<name>.lock` (spec section 8).
    pub fn acquire_lock(&self, area: &str, name: &str, options: AcquireOptions) -> Result<LockHandle> {
        acquire_lock_file(&self.lock_path(area, name)?, options)
    }
}

/// Builds a resolver from the host (default) or from explicit inputs (tests, vectors, a fixture root).
pub fn create_dirs(options: CreateDirsOptions) -> Result<XenoDirs> {
    let input = ResolveInput {
        platform: options.platform.unwrap_or_else(Platform::host),
        home: options.home.unwrap_or_else(host_home),
        env: options.env.unwrap_or_else(|| std::env::vars().collect()),
    };
    Ok(XenoDirs::new(resolve_roots(&input)?))
}

/// A resolver rooted entirely under one directory via XENO_HOME; the supported way to isolate tests and fixtures.
pub fn create_dirs_at(xeno_home: &Path) -> Result<XenoDirs> {
    let mut env = HashMap::new();
    env.insert("XENO_HOME".to_string(), xeno_home.to_string_lossy().to_string());
    Ok(XenoDirs::new(resolve_roots(&ResolveInput { platform: Platform::host(), home: String::new(), env })?))
}
