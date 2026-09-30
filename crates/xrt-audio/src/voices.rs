//! Voice library: a cloned voice is registered ONCE and referenced by id.
//!
//! Pipelines narrate many scripts in the same voice; re-uploading the clip on
//! every request is wasteful and makes "which voice was this?" unanswerable
//! after the fact. So a voice is stored on disk as its reference clip plus a
//! small manifest, and speech requests name it by id.
//!
//! Layout (versioned, so a future change can migrate rather than orphan):
//!
//! ```text
//! <root>/voices/v1/<id>/reference.wav   24 kHz mono f32, trimmed to 10 s
//! <root>/voices/v1/<id>/voice.json      { id, name, created, seconds, sha256 }
//! ```
//!
//! The clip — not derived conditioning tensors — is what is stored, because
//! the tensors belong to one model export; the clip stays valid across model
//! updates and is re-encoded (a second or two) when used.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::audio::{self, SAMPLE_RATE};
use crate::AudioError;

const LAYOUT: &str = "v1";
/// The model conditions on at most 10 s; storing more buys nothing.
const MAX_SECONDS: f32 = 10.0;
const MIN_SECONDS: f32 = 3.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    /// Unix seconds.
    pub created: u64,
    pub seconds: f32,
    /// SHA-256 of the stored reference.wav, so a caller can tell two voices
    /// apart even when they share a name.
    pub sha256: String,
}

pub struct VoiceLibrary {
    dir: PathBuf,
}

impl VoiceLibrary {
    /// `$XRT_AUDIO_VOICES_DIR`, else `~/.xeno`; `voices/v1` is appended.
    pub fn default_root() -> PathBuf {
        if let Some(d) = std::env::var_os("XRT_AUDIO_VOICES_DIR") {
            return PathBuf::from(d);
        }
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .unwrap_or_default();
        PathBuf::from(home).join(".xeno")
    }

    pub fn open(root: &Path) -> Self {
        Self {
            dir: root.join("voices").join(LAYOUT),
        }
    }

    /// Validate an id: it becomes a directory name, so only a safe charset.
    pub fn check_id(id: &str) -> Result<(), AudioError> {
        let ok = !id.is_empty()
            && id.len() <= 64
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            && !id.starts_with('-');
        let upper = id.to_ascii_uppercase();
        let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || ["COM", "LPT"].iter().any(|p| {
                upper
                    .strip_prefix(p)
                    .is_some_and(|n| n.len() == 1 && matches!(n.as_bytes()[0], b'1'..=b'9'))
            });
        if ok && !reserved {
            Ok(())
        } else {
            Err(AudioError::InvalidRequest(format!(
                "voice id `{id}` must be 1-64 chars of [A-Za-z0-9_-], not starting with '-'"
            )))
        }
    }

    /// Register a voice from a WAV. Refuses to overwrite an existing id.
    pub fn create(&self, id: &str, name: &str, wav: &[u8]) -> Result<VoiceInfo, AudioError> {
        Self::check_id(id)?;
        let target = self.dir.join(id);
        if target.exists() {
            return Err(AudioError::InvalidRequest(format!(
                "voice `{id}` already exists; delete it first"
            )));
        }
        let (x, rate) = audio::read_wav(wav)?;
        let mut x = audio::resample(&x, rate, SAMPLE_RATE);
        x.truncate((MAX_SECONDS * SAMPLE_RATE as f32) as usize);
        let seconds = x.len() as f32 / SAMPLE_RATE as f32;
        if seconds < MIN_SECONDS {
            return Err(AudioError::InvalidReference(format!(
                "reference is {seconds:.1}s; at least {MIN_SECONDS} s of clean speech is needed (6-10 s recommended)"
            )));
        }
        if x.iter().fold(0.0f32, |a, s| a.max(s.abs())) < 1e-3 {
            return Err(AudioError::InvalidReference("reference is silent".into()));
        }
        let bytes = audio::write_wav(&x, SAMPLE_RATE);
        let info = VoiceInfo {
            id: id.to_string(),
            name: if name.trim().is_empty() {
                id.to_string()
            } else {
                name.trim().to_string()
            },
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            seconds,
            sha256: sha256_hex(&bytes),
        };
        // Write into a temp dir and rename, so a crash never leaves a voice
        // directory with a clip but no manifest (or the reverse).
        let io = |e: std::io::Error| AudioError::Inference(format!("voice store: {e}"));
        std::fs::create_dir_all(&self.dir).map_err(io)?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| AudioError::Inference(e.to_string()))?
            .as_nanos();
        let tmp = self
            .dir
            .join(format!(".{id}.tmp-{}-{epoch}-{nonce}", std::process::id()));
        // Never remove a pre-existing staging directory: it can belong to a
        // concurrent creator or be a junction. Collision is a refusal.
        std::fs::create_dir(&tmp).map_err(io)?;
        std::fs::write(tmp.join("reference.wav"), &bytes).map_err(io)?;
        let manifest =
            serde_json::to_vec_pretty(&info).map_err(|e| AudioError::Inference(e.to_string()))?;
        std::fs::write(tmp.join("voice.json"), manifest).map_err(io)?;
        std::fs::rename(&tmp, &target).map_err(io)?;
        Ok(info)
    }

    pub fn list(&self) -> Result<Vec<VoiceInfo>, AudioError> {
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(AudioError::Inference(format!("voice store: {e}"))),
        };
        for e in entries {
            let e = e.map_err(|e| AudioError::Inference(format!("voice store: {e}")))?;
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            out.push(self.info(&name)?);
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    pub fn info(&self, id: &str) -> Result<VoiceInfo, AudioError> {
        Self::check_id(id)?;
        let raw = std::fs::read(self.dir.join(id).join("voice.json")).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                AudioError::InvalidRequest(format!("voice `{id}` not found"))
            } else {
                AudioError::Inference(format!("voice `{id}` manifest: {e}"))
            }
        })?;
        let info: VoiceInfo = serde_json::from_slice(&raw)
            .map_err(|e| AudioError::Inference(format!("voice `{id}` manifest: {e}")))?;
        if info.id != id
            || !info.seconds.is_finite()
            || !(MIN_SECONDS..=MAX_SECONDS).contains(&info.seconds)
            || info.sha256.len() != 64
            || !info.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(AudioError::Inference(format!(
                "voice `{id}` manifest identity is invalid"
            )));
        }
        Ok(info)
    }

    /// The stored reference as mono samples at 24 kHz, integrity-checked.
    pub fn load(&self, id: &str) -> Result<(Vec<f32>, u32), AudioError> {
        let info = self.info(id)?;
        let bytes = std::fs::read(self.dir.join(id).join("reference.wav"))
            .map_err(|e| AudioError::Inference(format!("voice `{id}` clip: {e}")))?;
        if sha256_hex(&bytes) != info.sha256 {
            return Err(AudioError::Inference(format!(
                "voice `{id}` clip does not match its manifest checksum"
            )));
        }
        audio::read_wav(&bytes)
    }

    pub fn delete(&self, id: &str) -> Result<(), AudioError> {
        Self::check_id(id)?;
        let dir = self.dir.join(id);
        if !dir.join("voice.json").is_file() {
            return Err(AudioError::InvalidRequest(format!(
                "voice `{id}` not found"
            )));
        }
        // Remove the two known files, then the (now empty) directory. Never a
        // recursive delete: a voice directory holds exactly these two files.
        let io = |e: std::io::Error| AudioError::Inference(format!("voice store: {e}"));
        std::fs::remove_file(dir.join("voice.json")).map_err(io)?;
        let _ = std::fs::remove_file(dir.join("reference.wav"));
        std::fs::remove_dir(&dir).map_err(io)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(secs: f32, rate: u32) -> Vec<u8> {
        let x: Vec<f32> = (0..(secs * rate as f32) as usize)
            .map(|i| 0.3 * (i as f32 * 0.05).sin())
            .collect();
        audio::write_wav(&x, rate)
    }

    fn lib(tag: &str) -> (VoiceLibrary, PathBuf) {
        let root =
            std::env::temp_dir().join(format!("xrt-voices-test-{tag}-{}", std::process::id()));
        (VoiceLibrary::open(&root), root)
    }

    fn cleanup(root: &Path) {
        // Test roots hold only voice dirs of two plain files each.
        let v = root.join("voices").join(LAYOUT);
        if let Ok(es) = std::fs::read_dir(&v) {
            for e in es.flatten() {
                for f in ["voice.json", "reference.wav"] {
                    let _ = std::fs::remove_file(e.path().join(f));
                }
                let _ = std::fs::remove_dir(e.path());
            }
        }
        let _ = std::fs::remove_dir(&v);
        let _ = std::fs::remove_dir(root.join("voices"));
        let _ = std::fs::remove_dir(root);
    }

    #[test]
    fn create_list_load_delete() {
        let (lib, root) = lib("crud");
        let info = lib
            .create("inanna-teach", "Inanna (teaching)", &clip(12.0, 48_000))
            .unwrap();
        assert_eq!(info.seconds, 10.0, "trimmed to the model's 10 s window");
        assert_eq!(lib.list().unwrap(), vec![info.clone()]);
        let (x, rate) = lib.load("inanna-teach").unwrap();
        assert_eq!(rate, SAMPLE_RATE);
        assert_eq!(x.len(), 10 * SAMPLE_RATE as usize);
        assert!(
            lib.create("inanna-teach", "again", &clip(5.0, 24_000))
                .is_err(),
            "no silent overwrite"
        );
        lib.delete("inanna-teach").unwrap();
        assert!(lib.list().unwrap().is_empty());
        assert!(lib.load("inanna-teach").is_err());
        cleanup(&root);
    }

    #[test]
    fn rejects_unsafe_ids_and_bad_clips() {
        let (lib, root) = lib("bad");
        for id in [
            "",
            "../x",
            "a/b",
            "a b",
            "-x",
            "CON",
            "nul",
            "COM1",
            "LPT9",
            &"x".repeat(65),
        ] {
            assert!(
                lib.create(id, "n", &clip(5.0, 24_000)).is_err(),
                "id {id:?} accepted"
            );
        }
        assert!(
            lib.create("short", "n", &clip(1.0, 24_000)).is_err(),
            "1 s clip accepted"
        );
        assert!(lib
            .create(
                "silent",
                "n",
                &audio::write_wav(&vec![0.0; 24_000 * 5], 24_000)
            )
            .is_err());
        assert!(
            lib.list().unwrap().is_empty(),
            "failed creates leave nothing behind"
        );
        cleanup(&root);
    }

    #[test]
    fn corrupt_manifest_is_not_hidden_as_an_empty_library() {
        let (lib, root) = lib("manifest");
        lib.create("v", "v", &clip(5.0, 24_000)).unwrap();
        let path = root
            .join("voices")
            .join(LAYOUT)
            .join("v")
            .join("voice.json");
        let mut info = lib.info("v").unwrap();
        info.id = "different-voice".into();
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec(&info).unwrap()).unwrap();
        std::fs::rename(temporary, path).unwrap();
        assert!(lib.info("v").is_err());
        assert!(lib.list().is_err());
        cleanup(&root);
    }

    #[test]
    fn a_tampered_clip_is_refused() {
        let (lib, root) = lib("tamper");
        lib.create("v", "v", &clip(5.0, 24_000)).unwrap();
        let p = root
            .join("voices")
            .join(LAYOUT)
            .join("v")
            .join("reference.wav");
        let mut b = std::fs::read(&p).unwrap();
        let last = b.len() - 1;
        b[last] ^= 0xff;
        std::fs::write(&p, b).unwrap();
        assert!(lib.load("v").is_err());
        cleanup(&root);
    }
}
