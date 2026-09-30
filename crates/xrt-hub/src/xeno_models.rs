//! Resolving models XENO publishes, from `updates.xenostudio.ai/models`.
//!
//! `ModelHub`'s existing downloader targets Hugging Face. This adds the second
//! source — our own registry — WITHOUT a second downloader: a model set is
//! turned into a [`BundleInstallPlan`] and handed to the bundle installer that
//! already exists, so it inherits per-artifact sha256 verification, host
//! allow-listing, size caps, redirect limits, staging and locking.
//!
//! # A model is not "the weights"
//!
//! 🔴 The first Whisper publish shipped `encoder`, `decoder` and
//! `tokenizer.json`, and every one of them resolved with HTTP 200 — yet no
//! machine could load the model. `WhisperModel::load` needs `config.json` for
//! its dimensions, and `xrt-tokenizer::from_hf_dir` reads `vocab.json` and
//! `merges.txt` and does **not** read `tokenizer.json`. Three files present,
//! three files verified, and a model that could not be assembled.
//!
//! That is why a SET is the unit here rather than a file. The bundle digest is
//! computed over every member's name, size and hash, so "the whisper-base set"
//! has one identity that changes if any member does — and a partially published
//! set fails at resolve time instead of at load time on a user's machine.
//!
//! # Unverifiable artifacts are refused
//!
//! As of 2026-08-29 the published manifest has 40 entries and **33 of them
//! carry `sha256: ""`** — they were hand-written and never generated from real
//! artifacts. An empty hash is not "no opinion", it is *we do not know what
//! these bytes should be*, and installing unverifiable weights is precisely the
//! supply-chain hole the bundle installer exists to close. Such an entry is
//! refused by name.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::bundle::{BundleArtifact, BundleInstallPlan, InstalledBundle};
use crate::ModelHub;
use xrt_core::{Result, XrtError};

/// Where XENO publishes model artifacts.
pub const XENO_MODEL_BASE_URL: &str = "https://updates.xenostudio.ai/models";
/// The only host a XENO model set may be fetched from.
pub const XENO_MODEL_HOST: &str = "updates.xenostudio.ai";

/// Generous ceiling for one set. whisper-base is ~292 MB; large-v3 is ~3 GB.
const MAX_SET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct RegistryEntry {
    file: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Registry {
    models: BTreeMap<String, RegistryEntry>,
}

/// One member of a model set: the id in the published manifest, and the
/// filename the consuming loader expects on disk.
///
/// The two differ on purpose. The registry is a flat namespace shared by every
/// product (`whisper-base-vocab.json`), while a loader wants the conventional
/// name its library looks for (`vocab.json`). Mapping here keeps the registry
/// flat without making every consumer rename files.
#[derive(Debug, Clone, Copy)]
pub struct SetMember {
    pub registry_id: &'static str,
    pub local_name: &'static str,
}

impl ModelHub {
    /// Installs a set of XENO-published artifacts and returns the directory
    /// holding them under their `local_name`s.
    ///
    /// Idempotent: an already-installed set with the same digest is returned
    /// without a network round trip, and `was_cached` says which happened.
    pub fn install_xeno_model_set(
        &self,
        id: &str,
        members: &[SetMember],
    ) -> Result<InstalledBundle> {
        if members.is_empty() {
            return Err(XrtError::Runtime(
                "a XENO model set must name at least one member".to_string(),
            ));
        }
        let registry = self.fetch_xeno_registry()?;

        let mut artifacts = Vec::with_capacity(members.len());
        for m in members {
            let entry = registry.models.get(m.registry_id).ok_or_else(|| {
                XrtError::Runtime(format!(
                    "`{}` is not in the XENO model registry at {XENO_MODEL_BASE_URL}/manifest.json",
                    m.registry_id
                ))
            })?;
            if entry.sha256.trim().is_empty() {
                return Err(XrtError::Runtime(format!(
                    "`{}` is published without a sha256, so its bytes cannot be verified. \
                     Republish it through xeno-platform's publish-onnx-models.mjs, which \
                     computes the hash from the artifact it uploads.",
                    m.registry_id
                )));
            }
            artifacts.push(BundleArtifact {
                path: m.local_name.to_string(),
                size_bytes: entry.size,
                sha256: entry.sha256.clone(),
                source: format!("{XENO_MODEL_BASE_URL}/{}", entry.file),
            });
        }

        // The set's identity: every member's local name, size and hash. Adding,
        // removing or re-hashing a member changes it, so a stale install can
        // never satisfy a changed set.
        let manifest = serde_json::json!({
            "id": id,
            "source": XENO_MODEL_BASE_URL,
            "artifacts": artifacts.iter().map(|a| serde_json::json!({
                "path": a.path, "size_bytes": a.size_bytes, "sha256": a.sha256,
            })).collect::<Vec<_>>(),
        });
        let manifest_bytes = serde_json::to_vec(&manifest)
            .map_err(|e| XrtError::Runtime(format!("could not encode set manifest: {e}")))?;
        let digest = hex(&Sha256::digest(&manifest_bytes));

        if let Ok(path) = self.resolve_installed_bundle(id, Some(&digest)) {
            return Ok(InstalledBundle {
                id: id.to_string(),
                digest,
                path,
                was_cached: true,
            });
        }

        let plan = BundleInstallPlan::new(
            id,
            digest,
            manifest_bytes,
            artifacts,
            vec![XENO_MODEL_HOST.to_string()],
            MAX_SET_BYTES,
        );
        self.install_bundle(&plan)
    }

    /// Returns the installed directory for a set if it is already present,
    /// without touching the network. Lets a caller answer "is this capability
    /// available offline?" — which is a different question from "install it".
    pub fn xeno_model_set_if_installed(&self, id: &str) -> Option<PathBuf> {
        self.resolve_installed_bundle(id, None).ok()
    }

    fn fetch_xeno_registry(&self) -> Result<Registry> {
        let url = format!("{XENO_MODEL_BASE_URL}/manifest.json");
        let body = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| XrtError::Runtime(format!("could not read the XENO model registry: {e}")))?
            .into_string()
            .map_err(|e| XrtError::Runtime(format!("XENO model registry was unreadable: {e}")))?;
        serde_json::from_str(&body)
            .map_err(|e| XrtError::Runtime(format!("XENO model registry is not valid JSON: {e}")))
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_member_list_is_refused() {
        let hub =
            ModelHub::with_cache_dir(std::env::temp_dir().join("xrt-hub-test-empty")).unwrap();
        let err = hub.install_xeno_model_set("whisper-base", &[]).unwrap_err();
        assert!(err.to_string().contains("at least one member"), "{err}");
    }

    /// The registry is the only source, and it is pinned to one host. A model
    /// set that could name an arbitrary `source` would let a compromised
    /// manifest redirect a download anywhere.
    #[test]
    fn the_allowed_host_is_pinned_to_the_xeno_cdn() {
        assert_eq!(XENO_MODEL_HOST, "updates.xenostudio.ai");
        assert!(XENO_MODEL_BASE_URL.starts_with(&format!("https://{XENO_MODEL_HOST}/")));
    }

    #[test]
    fn hex_encodes_lowercase_fixed_width() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }
}
