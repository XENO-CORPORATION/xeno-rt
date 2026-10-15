//! Publisher authorization for an explicitly pinned bundle manifest.
//!
//! Trust keys come from the caller's trusted installation/configuration, never
//! from a downloaded signature. The expected manifest digest remains mandatory:
//! this is pinned delivery, not an autonomous update feed or a TUF replacement.
use crate::ArtifactManifest;
use ring::signature::{UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use xrt_core::{Result, XrtError};

const DOMAIN: &[u8] = b"xeno-rt.artifact-manifest.v1\0";
fn invalid(message: &str) -> XrtError {
    XrtError::InvalidMetadata(message.into())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestSignature {
    pub schema_version: u32,
    pub algorithm: String,
    pub key_id: String,
    pub manifest_digest: String,
    pub signature: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedManifestKey {
    pub schema_version: u32,
    pub algorithm: String,
    pub key_id: String,
    pub public_key: String,
}

fn hex<const N: usize>(text: &str) -> Result<[u8; N]> {
    if text.len() != N * 2
        || !text
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid(
            "signature fields require exact lowercase hex lengths",
        ));
    }
    let mut bytes = [0; N];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| invalid("invalid signature hex"))?;
    }
    Ok(bytes)
}

pub fn signing_message(digest: &str) -> Result<Vec<u8>> {
    let _ = hex::<32>(digest)?;
    Ok([DOMAIN, digest.as_bytes()].concat())
}

/// Verify publisher authority and content identity before planning any writes.
/// Receipts may retain the public key id, digest and signature, never key material.
pub fn verify(
    manifest: &ArtifactManifest,
    expected_digest: &str,
    signature_bytes: &[u8],
    trusted_key_bytes: &[u8],
) -> Result<String> {
    if signature_bytes.len() > 4096 || trusted_key_bytes.len() > 4096 {
        return Err(invalid("signature or trusted public key exceeds 4 KiB"));
    }
    let signature: ManifestSignature = serde_json::from_slice(signature_bytes)
        .map_err(|_| invalid("invalid manifest signature document"))?;
    let key: TrustedManifestKey = serde_json::from_slice(trusted_key_bytes)
        .map_err(|_| invalid("invalid trusted public key document"))?;
    let digest = manifest.digest()?;
    if signature.schema_version != 1
        || key.schema_version != 1
        || signature.algorithm != "ed25519"
        || key.algorithm != "ed25519"
        || signature.manifest_digest != digest
        || digest != expected_digest
    {
        return Err(invalid(
            "signature schema, algorithm or pinned manifest digest mismatch",
        ));
    }
    let public = hex::<32>(&key.public_key)?;
    let actual_id = format!("{:x}", Sha256::digest(public));
    if key.key_id != actual_id || signature.key_id != actual_id {
        return Err(invalid("manifest signer is not the supplied trusted key"));
    }
    UnparsedPublicKey::new(&ED25519, public)
        .verify(
            &signing_message(&digest)?,
            &hex::<64>(&signature.signature)?,
        )
        .map_err(|_| invalid("manifest signature verification failed"))?;
    Ok(actual_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};
    fn fixture() -> (ArtifactManifest, String, Vec<u8>, Vec<u8>) {
        // Public deterministic TEST seed, never used for a distributed artifact.
        let pair = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
        let manifest = ArtifactManifest::parse(include_bytes!(
            "../../../reference/audio/drama3/bundle.dev.json"
        ))
        .unwrap();
        let digest = manifest.digest().unwrap();
        let public = pair.public_key().as_ref();
        let key_id = format!("{:x}", Sha256::digest(public));
        let encode = |bytes: &[u8]| bytes.iter().map(|v| format!("{v:02x}")).collect::<String>();
        let signature = ManifestSignature {
            schema_version: 1,
            algorithm: "ed25519".into(),
            key_id: key_id.clone(),
            manifest_digest: digest.clone(),
            signature: encode(pair.sign(&signing_message(&digest).unwrap()).as_ref()),
        };
        let key = TrustedManifestKey {
            schema_version: 1,
            algorithm: "ed25519".into(),
            key_id,
            public_key: encode(public),
        };
        (
            manifest,
            digest,
            serde_json::to_vec(&signature).unwrap(),
            serde_json::to_vec(&key).unwrap(),
        )
    }
    #[test]
    fn pinned_manifest_verifies_and_rejects_modified_content_key_and_signature() {
        let (mut manifest, digest, signature, key) = fixture();
        assert!(verify(&manifest, &digest, &signature, &key).is_ok());
        manifest.family.push_str("-changed");
        assert!(verify(&manifest, &digest, &signature, &key).is_err());
        manifest.family.truncate(manifest.family.len() - 8);
        let mut signed: ManifestSignature = serde_json::from_slice(&signature).unwrap();
        signed.signature.replace_range(0..2, "ff");
        assert!(verify(
            &manifest,
            &digest,
            &serde_json::to_vec(&signed).unwrap(),
            &key
        )
        .is_err());
        assert!(verify(&manifest, &"0".repeat(64), &signature, &key).is_err());
        let mut unknown: TrustedManifestKey = serde_json::from_slice(&key).unwrap();
        unknown.key_id = "0".repeat(64);
        assert!(verify(
            &manifest,
            &digest,
            &signature,
            &serde_json::to_vec(&unknown).unwrap()
        )
        .is_err());
        assert!(verify(&manifest, &digest, &vec![0; 4097], &key).is_err());
    }
    #[test]
    fn signatures_are_domain_separated_and_malformed_fields_are_refused() {
        let (manifest, digest, signature, key) = fixture();
        let pair = Ed25519KeyPair::from_seed_unchecked(&[7; 32]).unwrap();
        let mut signed: ManifestSignature = serde_json::from_slice(&signature).unwrap();
        signed.signature = pair
            .sign(digest.as_bytes())
            .as_ref()
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        assert!(verify(
            &manifest,
            &digest,
            &serde_json::to_vec(&signed).unwrap(),
            &key
        )
        .is_err());
        assert!(signing_message(&"A".repeat(64)).is_err());
        assert!(signing_message("short").is_err());
    }
}
