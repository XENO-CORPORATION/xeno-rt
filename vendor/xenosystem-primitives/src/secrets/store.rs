use super::envelope::{open_secret, random, seal_secret, EnvelopeFile, SealOptions, KEY_BYTES};
use super::error::{Result, SecretsError, SecretsErrorCode as C};
use super::keystore::{KeyStore, Protection};
use crate::dirs::registry::registry;
use crate::dirs::{assert_segment, AcquireOptions, Kind, XenoDirs};
use std::sync::Mutex;
use std::time::Duration;

const SUFFIX: &str = ".secret";

impl crate::dirs::SecretSink for SecretStore {
    fn set(&self, area: &str, name: &str, value: &[u8]) -> std::result::Result<(), String> {
        SecretStore::set(self, area, name, value).map_err(|e| e.to_string())
    }
    fn get(&self, area: &str, name: &str) -> std::result::Result<Option<Vec<u8>>, String> {
        SecretStore::get(self, area, name).map_err(|e| e.to_string())
    }
}

/// Envelope-encrypted secrets at `secrets/<area>/<name>.secret` (XENO LOCAL HOME - SPEC.md section 6). Refuses outright
/// when the key store is not OS-protected, and never creates a new master key while secrets sealed with an old one exist.
pub struct SecretStore {
    dirs: XenoDirs,
    key_store: Box<dyn KeyStore>,
    key: Mutex<Option<Vec<u8>>>,
}

pub struct Status {
    pub available: bool,
    pub key_store: String,
    pub reason: Option<String>,
}

impl SecretStore {
    pub fn new(dirs: XenoDirs, key_store: Box<dyn KeyStore>) -> SecretStore {
        SecretStore { dirs, key_store, key: Mutex::new(None) }
    }

    /// Whether this machine can keep secrets, and if not, why. Safe to show to a person.
    pub fn status(&self) -> Status {
        match self.key_store.protection() {
            Protection::OsProtected => Status { available: true, key_store: self.key_store.id().into(), reason: None },
            Protection::Unprotected => Status {
                available: false,
                key_store: self.key_store.id().into(),
                reason: Some(self.key_store.reason().unwrap_or("no protected store is available").into()),
            },
        }
    }

    fn require_protected(&self) -> Result<()> {
        if self.key_store.protection() != Protection::OsProtected {
            return Err(SecretsError::new(C::UnprotectedStore, self.key_store.reason().unwrap_or("this machine offers only an unprotected store; secrets are not saved")));
        }
        Ok(())
    }

    fn file(&self, area: &str, name: &str) -> Result<std::path::PathBuf> {
        assert_segment(name, "secret name")?;
        Ok(self.dirs.path(Kind::Secrets, area, &[&format!("{name}{SUFFIX}")])?)
    }

    fn any_secret_exists(&self) -> Result<bool> {
        for a in &registry().areas {
            if !a.uses.contains(&Kind::Secrets) || a.area.contains('<') {
                continue;
            }
            match std::fs::read_dir(self.dirs.path(Kind::Secrets, &a.area, &[])?) {
                Ok(it) => {
                    if it.flatten().any(|e| e.file_name().to_string_lossy().ends_with(SUFFIX)) {
                        return Ok(true);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(false)
    }

    fn master_key(&self, create: bool) -> Result<Option<Vec<u8>>> {
        if let Some(k) = self.key.lock().unwrap().clone() {
            return Ok(Some(k));
        }
        self.require_protected()?;
        if let Some(found) = self.key_store.get_key()? {
            if found.len() != KEY_BYTES {
                return Err(SecretsError::new(C::KeyUnavailable, "the stored master key has the wrong length"));
            }
            *self.key.lock().unwrap() = Some(found.clone());
            return Ok(Some(found));
        }
        if !create {
            return Ok(None);
        }
        let lock = self.dirs.acquire_lock("platform", "secrets-key", AcquireOptions { owner: "secrets master key".into(), timeout: Some(Duration::from_secs(60)), ..Default::default() })?;
        let result = (|| -> Result<Vec<u8>> {
            if let Some(again) = self.key_store.get_key()? {
                return Ok(again);
            }
            if self.any_secret_exists()? {
                return Err(SecretsError::new(C::KeyUnavailable, "secrets exist but the master key is missing from the OS store; refusing to create a new key that would orphan them"));
            }
            let fresh = random::<KEY_BYTES>()?.to_vec();
            self.key_store.put_key(&fresh)?;
            match self.key_store.get_key()? {
                Some(check) if check == fresh => Ok(fresh),
                _ => Err(SecretsError::new(C::KeyUnavailable, "the OS store did not keep the master key")),
            }
        })();
        let _ = lock.release();
        let key = result?;
        *self.key.lock().unwrap() = Some(key.clone());
        Ok(Some(key))
    }

    pub fn set(&self, area: &str, name: &str, value: &[u8]) -> Result<()> {
        self.file(area, name)?; // validates area, kind and name before anything is created
        self.require_protected()?;
        self.dirs.assert_writable()?; // a newer layout must refuse before a master key is created
        let key = self.master_key(true)?.expect("created");
        let sealed = seal_secret(&key, area, name, value, SealOptions::default())?;
        self.dirs.write_file(Kind::Secrets, area, &[&format!("{name}{SUFFIX}")], (sealed.to_json() + "\n").as_bytes())?;
        Ok(())
    }

    /// The secret's bytes, or None when it does not exist.
    pub fn get(&self, area: &str, name: &str) -> Result<Option<Vec<u8>>> {
        let path = self.file(area, name)?;
        self.require_protected()?;
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let Some(key) = self.master_key(false)? else {
            return Err(SecretsError::new(C::KeyUnavailable, format!("{area}/{name} exists but the master key is missing from the OS store")));
        };
        Ok(Some(open_secret(&key, &EnvelopeFile::parse(&text)?, area, name)?))
    }

    pub fn has(&self, area: &str, name: &str) -> Result<bool> {
        Ok(self.file(area, name)?.exists())
    }

    pub fn delete(&self, area: &str, name: &str) -> Result<bool> {
        match std::fs::remove_file(self.file(area, name)?) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Names (never values) stored in an area.
    pub fn list(&self, area: &str) -> Result<Vec<String>> {
        match std::fs::read_dir(self.dirs.path(Kind::Secrets, area, &[])?) {
            Ok(it) => {
                let mut v: Vec<String> = it.flatten().filter_map(|e| e.file_name().to_string_lossy().strip_suffix(SUFFIX).map(String::from)).collect();
                v.sort();
                Ok(v)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
            Err(e) => Err(e.into()),
        }
    }

    /// Drops the in-memory copy of the master key.
    pub fn forget_key(&self) {
        if let Some(k) = self.key.lock().unwrap().as_mut() {
            k.fill(0);
        }
        *self.key.lock().unwrap() = None;
    }
}
