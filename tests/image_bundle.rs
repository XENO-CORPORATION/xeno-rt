#![cfg(feature = "image-generation-tests")]

use std::{collections::BTreeMap, fs, path::Path};

use sha2::{Digest, Sha256};
use xrt_image::{
    BundleComponent, BundleFile, BundleLicense, BundleLimits, BundleManifest, ComponentFormat,
    ComponentRole, ImageCapability, ImageErrorKind, ImageModelBundle, ManifestMode,
};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_fixture(root: &Path) -> BundleManifest {
    let mut components = Vec::new();
    for role in [
        ComponentRole::Transformer,
        ComponentRole::TextEncoder,
        ComponentRole::Tokenizer,
        ComponentRole::Vae,
        ComponentRole::Scheduler,
    ] {
        let path = format!("{}/fixture.bin", role.as_str());
        let full = root.join(&path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        let bytes = [role.as_str().len() as u8];
        fs::write(&full, bytes).unwrap();
        components.push(BundleComponent {
            role,
            format: ComponentFormat::Json,
            optional: false,
            files: vec![BundleFile {
                path,
                size_bytes: 1,
                sha256: sha256(&bytes),
                source: None,
                source_kind: Some("local".to_string()),
            }],
        });
    }
    let manifest = BundleManifest {
        schema_version: 1,
        id: "local-fixture".to_string(),
        family: "qwen-image".to_string(),
        revision: "fixture-revision".to_string(),
        source_revisions: BTreeMap::new(),
        capabilities: vec![ImageCapability::Generate],
        license: BundleLicense {
            spdx: "Apache-2.0".to_string(),
            evidence: "https://example.invalid/model/blob/fixture-revision/README.md".to_string(),
            files: Vec::new(),
        },
        quantization: "Q4_K_M".to_string(),
        components,
        limits: BundleLimits {
            max_sequence_length: 512,
            max_width: 64,
            max_height: 64,
            max_pixels: 4_096,
        },
    };
    fs::write(
        root.join("xrt.bundle.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    manifest
}

#[test]
fn opens_a_complete_hash_verified_local_bundle() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = write_fixture(directory.path());
    let bundle = ImageModelBundle::open(directory.path()).unwrap();
    assert_eq!(bundle.manifest(), &manifest);
    assert_eq!(bundle.digest(), manifest.digest().unwrap());
}

#[test]
fn rejects_artifact_tampering_before_runtime_load() {
    let directory = tempfile::tempdir().unwrap();
    write_fixture(directory.path());
    fs::write(directory.path().join("vae/fixture.bin"), [99]).unwrap();
    let error = ImageModelBundle::open(directory.path()).unwrap_err();
    assert_eq!(error.kind(), ImageErrorKind::Checksum);
}

#[test]
fn phase0_qwen_q4_manifest_matches_the_runtime_schema() {
    let manifest = BundleManifest::from_json_bytes(
        include_bytes!("../reference/image/qwen/manifests/qwen-image-2512-q4_k_m.json"),
        ManifestMode::Catalog,
    )
    .unwrap();
    assert_eq!(manifest.id, "qwen-image-2512-q4_k_m");
    assert_eq!(manifest.capabilities, vec![ImageCapability::Generate]);
    assert_eq!(manifest.digest().unwrap().len(), 64);
}
