//! `secrets`: envelope-encrypted credentials (XENO LOCAL HOME - SPEC.md section 6). The OS store holds one master key
//! per person per machine; each secret has its own data key, wrapped by the master key; AES-256-GCM throughout.
//! There is no plaintext fallback: a store that is not OS-protected refuses every operation. No value or key appears
//! in any error message.

pub mod envelope;
pub mod error;
pub mod keystore;
pub mod store;
#[cfg(windows)]
pub mod windows;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use envelope::{keyid_of, open_secret, seal_secret, EnvelopeFile, SealOptions};
pub use error::{Result, SecretsError, SecretsErrorCode};
pub use keystore::{create_os_key_store, unprotected_key_store, KeyStore, Protection};
pub use store::SecretStore;
