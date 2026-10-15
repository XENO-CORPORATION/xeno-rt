//! Explicit operator signing. A dedicated OS-protected seed never leaves the
//! signer, enters argv/environment, or shares the renderer-tool signing identity.
use clap::{Args, Subcommand};
use ring::{
    rand::{SecureRandom, SystemRandom},
    signature::{Ed25519KeyPair, KeyPair},
};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use xenosystem_primitives::{
    dirs::{create_dirs, AcquireOptions, CreateDirsOptions},
    secrets::{create_os_key_store, keystore::OsStoreOptions, KeyStore, Protection},
};
use xrt_hub::{
    artifact_signature::{self, ManifestSignature, TrustedManifestKey},
    ArtifactManifest,
};
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SERVICE: &str = "XENO.RT.ArtifactPublisher.v1";
const ACCOUNT: &str = "native-bundle-manifests";

#[derive(Args)]
pub struct PublisherArgs {
    #[command(subcommand)]
    command: PublisherCommand,
}
#[derive(Subcommand)]
enum PublisherCommand {
    /// Create/reuse the dedicated protected publisher identity; never rotate.
    Init {
        #[arg(long)]
        trusted_key: PathBuf,
        #[arg(long)]
        confirm: bool,
    },
    /// Read the public identity and protection provider; never exports the seed.
    Info,
    /// Sign only the explicitly pinned canonical manifest with this identity.
    Sign {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        digest: String,
        #[arg(long)]
        expected_key_id: String,
        #[arg(long)]
        out: PathBuf,
    },
}

fn seed(store: &dyn KeyStore, create: bool) -> Result<(Zeroizing<Vec<u8>>, bool)> {
    if store.protection() != Protection::OsProtected {
        return Err(
            "publisher signing requires an OS-protected key store; no plaintext fallback".into(),
        );
    }
    if let Some(found) = store.get_key()? {
        let found = Zeroizing::new(found);
        if found.len() != 32 {
            return Err("stored publisher seed is invalid; refusing replacement".into());
        }
        return Ok((found, false));
    }
    if !create {
        return Err("publisher identity is not provisioned; run publisher init explicitly".into());
    }
    let mut fresh = Zeroizing::new(vec![0; 32]);
    SystemRandom::new()
        .fill(&mut fresh)
        .map_err(|_| "publisher key generation failed")?;
    store.put_key(&fresh)?;
    let check = Zeroizing::new(
        store
            .get_key()?
            .ok_or("protected store did not retain publisher identity")?,
    );
    if *check != *fresh {
        return Err("publisher key readback differs; no signature produced".into());
    }
    Ok((fresh, true))
}
fn pair(seed: &[u8]) -> Result<Ed25519KeyPair> {
    Ed25519KeyPair::from_seed_unchecked(seed).map_err(|_| "invalid protected publisher seed".into())
}
fn encode(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}
fn public(pair: &Ed25519KeyPair) -> TrustedManifestKey {
    let bytes = pair.public_key().as_ref();
    TrustedManifestKey {
        schema_version: 1,
        algorithm: "ed25519".into(),
        key_id: format!("{:x}", Sha256::digest(bytes)),
        public_key: encode(bytes),
    }
}
fn new_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}
fn read_manifest(path: &Path) -> Result<ArtifactManifest> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    std::io::Read::by_ref(&mut file)
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("manifest exceeds 1 MiB".into());
    }
    Ok(ArtifactManifest::parse(&bytes)?)
}
pub fn run(args: PublisherArgs) -> Result<()> {
    let store = create_os_key_store(OsStoreOptions {
        service: Some(SERVICE.into()),
        account: Some(ACCOUNT.into()),
    });
    let dirs = create_dirs(CreateDirsOptions::default())?;
    // Serializes cooperating processes across init/read/sign. This uses its own
    // named lock and cannot break another application's key or signing task.
    let _lock = dirs.acquire_lock(
        "platform",
        "rt-artifact-publisher",
        AcquireOptions {
            owner: "XENO RT artifact publisher".into(),
            timeout: Some(Duration::from_secs(10)),
            ..Default::default()
        },
    )?;
    match args.command {
        PublisherCommand::Init {
            trusted_key,
            confirm,
        } => {
            if !confirm {
                return Err("publisher init requires --confirm; no identity was created".into());
            }
            if trusted_key.exists() {
                return Err("refusing to overwrite public trust material".into());
            }
            let (secret, created) = seed(store.as_ref(), true)?;
            let key = public(&pair(&secret)?);
            new_json(&trusted_key, &key)?;
            println!(
                "{}",
                serde_json::json!({"key_id":key.key_id,"protected":true,"provider":store.id(),"created":created,"published":false})
            );
        }
        PublisherCommand::Info => {
            let (secret, _) = seed(store.as_ref(), false)?;
            println!(
                "{}",
                serde_json::json!({"identity":public(&pair(&secret)?),"protected":true,"provider":store.id()})
            );
        }
        PublisherCommand::Sign {
            manifest,
            digest,
            expected_key_id,
            out,
        } => {
            let manifest = read_manifest(&manifest)?;
            if manifest.digest()? != digest {
                return Err("manifest differs from explicitly pinned digest".into());
            }
            let (secret, _) = seed(store.as_ref(), false)?;
            let pair = pair(&secret)?;
            let key = public(&pair);
            if key.key_id != expected_key_id {
                return Err("protected publisher differs from pinned identity".into());
            }
            let signature = ManifestSignature {
                schema_version: 1,
                algorithm: "ed25519".into(),
                key_id: key.key_id.clone(),
                manifest_digest: digest.clone(),
                signature: encode(
                    pair.sign(&artifact_signature::signing_message(&digest)?)
                        .as_ref(),
                ),
            };
            artifact_signature::verify(
                &manifest,
                &digest,
                &serde_json::to_vec(&signature)?,
                &serde_json::to_vec(&key)?,
            )?;
            new_json(&out, &signature)?;
            println!(
                "{}",
                serde_json::json!({"key_id":key.key_id,"manifest_digest":digest,"protected":true,"signature_written":true,"published":false})
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    struct Store {
        value: Mutex<Option<Vec<u8>>>,
        protected: bool,
    }
    impl KeyStore for Store {
        fn id(&self) -> &str {
            "test-only"
        }
        fn protection(&self) -> Protection {
            if self.protected {
                Protection::OsProtected
            } else {
                Protection::Unprotected
            }
        }
        fn get_key(&self) -> xenosystem_primitives::secrets::Result<Option<Vec<u8>>> {
            Ok(self.value.lock().unwrap().clone())
        }
        fn put_key(&self, key: &[u8]) -> xenosystem_primitives::secrets::Result<()> {
            *self.value.lock().unwrap() = Some(key.to_vec());
            Ok(())
        }
        fn delete_key(&self) -> xenosystem_primitives::secrets::Result<()> {
            unreachable!()
        }
    }
    #[test]
    fn provisioning_never_rotates_or_accepts_unprotected_or_corrupt_material() {
        let store = Store {
            value: Mutex::new(None),
            protected: true,
        };
        assert!(seed(&store, false).is_err());
        let (first, created) = seed(&store, true).unwrap();
        assert!(created);
        let (again, created) = seed(&store, true).unwrap();
        assert!(!created);
        assert_eq!(*first, *again);
        assert_eq!(
            public(&pair(&first).unwrap()).key_id,
            public(&pair(&again).unwrap()).key_id
        );
        *store.value.lock().unwrap() = Some(vec![7; 31]);
        assert!(seed(&store, true).is_err());
        assert_eq!(store.value.lock().unwrap().as_ref().unwrap().len(), 31);
        let unprotected = Store {
            value: Mutex::new(None),
            protected: false,
        };
        assert!(seed(&unprotected, true).is_err());
        assert!(unprotected.value.lock().unwrap().is_none());
    }
}
