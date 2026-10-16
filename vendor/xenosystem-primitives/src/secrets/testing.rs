//! Test-only. A key store that lives in memory. It claims `OsProtected` so the secret store accepts it, which is why it
//! sits behind the `testing` feature and identifies itself as `memory-test-only` in `status()`.

use super::error::Result;
use super::keystore::{KeyStore, Protection};
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct MemoryKeyStore {
    key: Arc<Mutex<Option<Vec<u8>>>>,
}

impl MemoryKeyStore {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_key(key: &[u8]) -> Self {
        MemoryKeyStore { key: Arc::new(Mutex::new(Some(key.to_vec()))) }
    }
    pub fn snapshot(&self) -> Option<Vec<u8>> {
        self.key.lock().unwrap().clone()
    }
}

impl KeyStore for MemoryKeyStore {
    fn id(&self) -> &str {
        "memory-test-only"
    }
    fn protection(&self) -> Protection {
        Protection::OsProtected
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>> {
        Ok(self.snapshot())
    }
    fn put_key(&self, key: &[u8]) -> Result<()> {
        *self.key.lock().unwrap() = Some(key.to_vec());
        Ok(())
    }
    fn delete_key(&self) -> Result<()> {
        *self.key.lock().unwrap() = None;
        Ok(())
    }
}
