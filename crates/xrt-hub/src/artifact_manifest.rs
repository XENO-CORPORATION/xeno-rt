//! Modality-neutral, content-identified bundles. Schema 2 extends the existing
//! installer; schema 1 image bundles retain their own validation and recovery.
use crate::{BundleArtifact, BundleImportArtifact, BundleImportPlan, BundleInstallPlan};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub id: String,
    pub revision: String,
    /// `model` or `native-runtime`. Native executables require distribution
    /// authorization in addition to content hashes; this type grants none.
    pub kind: String,
    pub family: String,
    pub tasks: Vec<String>,
    pub minimum_runtime: String,
    pub platforms: Vec<String>,
    pub backends: Vec<String>,
    /// Roles -> declared relative file paths, e.g. encoder -> onnx/encoder.onnx.
    pub entrypoints: BTreeMap<String, String>,
    pub dependencies: Vec<ArtifactDependency>,
    pub license: ArtifactLicense,
    pub files: Vec<ArtifactFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDependency {
    pub id: String,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactLicense {
    pub spdx: String,
    pub evidence: String,
    pub files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// Immutable HTTPS origin, with URL grants supplied separately when a
    /// catalog uses expiring links. Transport tokens are not bundle identity.
    pub source: String,
}

fn bad(message: impl Into<String>) -> XrtError {
    XrtError::Runtime(message.into())
}
fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn id_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}
fn path_valid(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && !value.contains(['\\', ':', '\0'])
        && value.split('/').all(|p| {
            !p.is_empty()
                && p != "."
                && p != ".."
                && !p.ends_with(['.', ' '])
                && !p.chars().any(|c| c.is_control())
                && !matches!(
                    p.split('.')
                        .next()
                        .unwrap_or("")
                        .to_ascii_uppercase()
                        .as_str(),
                    "CON"
                        | "PRN"
                        | "AUX"
                        | "NUL"
                        | "COM1"
                        | "COM2"
                        | "COM3"
                        | "COM4"
                        | "COM5"
                        | "COM6"
                        | "COM7"
                        | "COM8"
                        | "COM9"
                        | "LPT1"
                        | "LPT2"
                        | "LPT3"
                        | "LPT4"
                        | "LPT5"
                        | "LPT6"
                        | "LPT7"
                        | "LPT8"
                        | "LPT9"
                )
        })
}

impl ArtifactManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 1024 * 1024 {
            return Err(bad("bundle manifest exceeds 1 MiB"));
        }
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|e| bad(format!("bundle manifest: {e}")))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 2
            || !id_valid(&self.id)
            || self.revision.trim().is_empty()
            || !matches!(self.kind.as_str(), "model" | "native-runtime")
            || self.family.trim().is_empty()
            || self.tasks.is_empty()
            || self.platforms.is_empty()
            || self.backends.is_empty()
        {
            return Err(bad("invalid bundle identity, kind or capabilities"));
        }
        if self.minimum_runtime.split('.').count() != 3
            || self
                .minimum_runtime
                .split('.')
                .any(|n| n.parse::<u32>().is_err())
        {
            return Err(bad(
                "minimum_runtime must be a three-component release version",
            ));
        }
        if self.files.is_empty()
            || self.files.len() > 512
            || self.license.spdx.is_empty()
            || self.license.evidence.is_empty()
            || self.license.files.is_empty()
        {
            return Err(bad(
                "bundle requires bounded artifacts and license evidence files",
            ));
        }
        let mut paths = BTreeSet::new();
        let mut total = 0u64;
        for file in &self.files {
            if !path_valid(&file.path)
                || file.path == "xrt.bundle.json"
                || !paths.insert(file.path.to_ascii_lowercase())
                || file.size_bytes == 0
                || !hash_valid(&file.sha256)
            {
                return Err(bad(format!(
                    "unsafe, duplicate or unverifiable bundle artifact: {}",
                    file.path
                )));
            }
            let url = url::Url::parse(&file.source).map_err(|_| bad("invalid artifact URL"))?;
            if url.scheme() != "https"
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(bad("artifact origin must be credential-free immutable HTTPS, not an expiring grant"));
            }
            total = total
                .checked_add(file.size_bytes)
                .ok_or_else(|| bad("bundle size overflow"))?;
        }
        for path in self.entrypoints.values().chain(self.license.files.iter()) {
            if !self.files.iter().any(|f| &f.path == path) {
                return Err(bad(format!("undeclared entrypoint/license: {path}")));
            }
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &self.dependencies {
            if !id_valid(&dependency.id)
                || dependency.id == self.id
                || !hash_valid(&dependency.digest)
                || !dependencies.insert(&dependency.id)
            {
                return Err(bad("invalid or duplicate bundle dependency"));
            }
        }
        Ok(())
    }

    /// Compares release components only: a release candidate of the required
    /// version is accepted (`0.4.0-rc.1` satisfies `0.4.0`) so candidates can be
    /// qualified against the exact bundles they will ship with.
    pub fn check_runtime(&self, version: &str, platform: &str) -> Result<()> {
        let parse = |v: &str| -> Result<Vec<u32>> {
            let release = v.split('-').next().unwrap_or(v);
            let numbers = release
                .split('.')
                .map(str::parse::<u32>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| bad("invalid runtime version"))?;
            if numbers.len() != 3 {
                return Err(bad("invalid runtime version"));
            }
            Ok(numbers)
        };
        if parse(version)? < parse(&self.minimum_runtime)? {
            return Err(bad(format!(
                "bundle requires runtime {}, installed {version}",
                self.minimum_runtime
            )));
        }
        if !self.platforms.iter().any(|p| p == "any" || p == platform) {
            return Err(bad(format!("bundle does not support {platform}")));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.files.sort_by(|a, b| a.path.cmp(&b.path));
        canonical.dependencies.sort_by(|a, b| a.id.cmp(&b.id));
        canonical.tasks.sort();
        canonical.platforms.sort();
        canonical.backends.sort();
        canonical.license.files.sort();
        let value = serde_json::to_value(canonical).map_err(|e| bad(e.to_string()))?;
        // serde_json's default object representation is an ordered map.
        serde_json::to_vec(&value).map_err(|e| bad(e.to_string()))
    }

    pub fn digest(&self) -> Result<String> {
        Ok(format!("{:x}", Sha256::digest(self.canonical_bytes()?)))
    }

    pub fn install_plan(&self, allowed_hosts: Vec<String>, cap: u64) -> Result<BundleInstallPlan> {
        Ok(BundleInstallPlan::new(
            &self.id,
            self.digest()?,
            self.canonical_bytes()?,
            self.files
                .iter()
                .map(|f| BundleArtifact {
                    path: f.path.clone(),
                    size_bytes: f.size_bytes,
                    sha256: f.sha256.clone(),
                    source: f.source.clone(),
                })
                .collect(),
            allowed_hosts,
            cap,
        ))
    }

    pub fn import_plan(&self, cap: u64) -> Result<BundleImportPlan> {
        Ok(BundleImportPlan::new(
            &self.id,
            self.digest()?,
            self.canonical_bytes()?,
            self.files
                .iter()
                .map(|f| BundleImportArtifact {
                    path: f.path.clone(),
                    size_bytes: f.size_bytes,
                    sha256: f.sha256.clone(),
                })
                .collect(),
            cap,
        ))
    }

    pub fn verify_directory(&self, root: &Path) -> Result<()> {
        self.validate()?;
        for file in &self.files {
            let mut path = root.to_path_buf();
            for part in file.path.split('/') {
                path.push(part);
                let metadata = std::fs::symlink_metadata(&path)?;
                #[cfg(windows)]
                {
                    use std::os::windows::fs::MetadataExt;
                    if metadata.file_attributes() & 0x400 != 0 {
                        return Err(bad("bundle contains a reparse point"));
                    }
                }
                if metadata.file_type().is_symlink() {
                    return Err(bad("bundle contains a symlink"));
                }
            }
            if !path.is_file() || path.metadata()?.len() != file.size_bytes {
                return Err(bad(format!("missing/incorrect size: {}", file.path)));
            }
            use std::io::Read;
            let mut reader = std::fs::File::open(path)?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0; 1024 * 1024];
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            if format!("{:x}", hash.finalize()) != file.sha256 {
                return Err(bad(format!("SHA-256 mismatch: {}", file.path)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ArtifactManifest {
        ArtifactManifest {
            schema_version: 2,
            id: "audio-fixture".into(),
            revision: "immutable-rev".into(),
            kind: "model".into(),
            family: "test".into(),
            tasks: vec!["speech".into()],
            minimum_runtime: "0.3.0".into(),
            platforms: vec!["any".into()],
            backends: vec!["cpu".into()],
            entrypoints: BTreeMap::from([("model".into(), "model.onnx".into())]),
            dependencies: vec![],
            license: ArtifactLicense {
                spdx: "MIT".into(),
                evidence: "pinned source license".into(),
                files: vec!["LICENSE".into()],
            },
            files: vec![
                ArtifactFile {
                    path: "model.onnx".into(),
                    size_bytes: 4,
                    sha256: "a".repeat(64),
                    source: "https://example.com/v1/model.onnx".into(),
                },
                ArtifactFile {
                    path: "LICENSE".into(),
                    size_bytes: 1,
                    sha256: "b".repeat(64),
                    source: "https://example.com/v1/LICENSE".into(),
                },
            ],
        }
    }
    #[test]
    fn identity_is_order_independent_but_covers_every_artifact() {
        let a = fixture();
        let mut b = a.clone();
        b.files.reverse();
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        b.files[0].sha256 = "c".repeat(64);
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }
    #[test]
    fn unsafe_paths_and_missing_members_are_rejected() {
        for path in [
            "../model",
            "C:/model",
            "a\\b",
            "CON",
            "model.onnx ",
            "xrt.bundle.json",
        ] {
            let mut m = fixture();
            m.files[0].path = path.into();
            assert!(m.validate().is_err());
        }
        let mut m = fixture();
        m.files.push(m.files[0].clone());
        assert!(m.validate().is_err());
        let mut m = fixture();
        m.license.files = vec!["missing".into()];
        assert!(m.validate().is_err());
        let mut m = fixture();
        m.schema_version = 3;
        assert!(m.validate().is_err());
    }
    #[test]
    fn runtime_and_platform_requirements_are_enforced() {
        let mut m = fixture();
        m.minimum_runtime = "0.4.0".into();
        m.platforms = vec!["windows-x86_64".into()];
        assert!(m.check_runtime("0.4.0-rc.1", "windows-x86_64").is_ok());
        assert!(m.check_runtime("0.4.1", "windows-x86_64").is_ok());
        assert!(m.check_runtime("0.3.9", "windows-x86_64").is_err());
        assert!(m.check_runtime("0.4.0", "linux-x86_64").is_err());
        assert!(m.check_runtime("latest", "windows-x86_64").is_err());
    }
}
