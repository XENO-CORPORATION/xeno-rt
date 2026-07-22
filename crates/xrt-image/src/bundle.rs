use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{ImageCapability, ImageError};

const MANIFEST_FILE: &str = "xrt.bundle.json";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const HASH_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestMode {
    Catalog,
    LocalImport,
    Installed,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentRole {
    Pipeline,
    Transformer,
    TextEncoder,
    Tokenizer,
    Processor,
    Vae,
    Scheduler,
    VisionEncoder,
    VisionProjection,
    PreviewDecoder,
    Metadata,
    Other(String),
}

impl ComponentRole {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pipeline => "pipeline",
            Self::Transformer => "transformer",
            Self::TextEncoder => "text_encoder",
            Self::Tokenizer => "tokenizer",
            Self::Processor => "processor",
            Self::Vae => "vae",
            Self::Scheduler => "scheduler",
            Self::VisionEncoder => "vision_encoder",
            Self::VisionProjection => "vision_projection",
            Self::PreviewDecoder => "preview_decoder",
            Self::Metadata => "metadata",
            Self::Other(value) => value,
        }
    }

    fn from_string(value: String) -> Self {
        match value.as_str() {
            "pipeline" => Self::Pipeline,
            "transformer" => Self::Transformer,
            "text_encoder" => Self::TextEncoder,
            "tokenizer" => Self::Tokenizer,
            "processor" => Self::Processor,
            "vae" => Self::Vae,
            "scheduler" => Self::Scheduler,
            "vision_encoder" => Self::VisionEncoder,
            "vision_projection" => Self::VisionProjection,
            "preview_decoder" => Self::PreviewDecoder,
            "metadata" => Self::Metadata,
            _ => Self::Other(value),
        }
    }

    fn is_known(&self) -> bool {
        !matches!(self, Self::Other(_))
    }
}

impl Serialize for ComponentRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ComponentRole {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if !valid_identifier(&value) {
            return Err(D::Error::custom("component role is not a safe identifier"));
        }
        Ok(Self::from_string(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentFormat {
    Gguf,
    SafeTensors,
    HuggingFaceJson,
    Json,
    Other(String),
}

impl ComponentFormat {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gguf => "gguf",
            Self::SafeTensors => "safetensors",
            Self::HuggingFaceJson => "huggingface-json",
            Self::Json => "json",
            Self::Other(value) => value,
        }
    }

    fn from_string(value: String) -> Self {
        match value.as_str() {
            "gguf" => Self::Gguf,
            "safetensors" => Self::SafeTensors,
            "huggingface-json" => Self::HuggingFaceJson,
            "json" => Self::Json,
            _ => Self::Other(value),
        }
    }
}

impl Serialize for ComponentFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ComponentFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if !valid_identifier(&value) {
            return Err(D::Error::custom(
                "component format is not a safe identifier",
            ));
        }
        Ok(Self::from_string(value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleFile {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleComponent {
    pub role: ComponentRole,
    pub format: ComponentFormat,
    #[serde(default)]
    pub optional: bool,
    pub files: Vec<BundleFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleLicense {
    pub spdx: String,
    pub evidence: String,
    #[serde(default)]
    pub files: Vec<BundleFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleLimits {
    pub max_sequence_length: usize,
    pub max_width: u32,
    pub max_height: u32,
    pub max_pixels: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub id: String,
    pub family: String,
    pub revision: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub source_revisions: BTreeMap<String, String>,
    pub capabilities: Vec<ImageCapability>,
    pub license: BundleLicense,
    pub quantization: String,
    pub components: Vec<BundleComponent>,
    pub limits: BundleLimits,
}

impl BundleManifest {
    pub fn from_json_bytes(bytes: &[u8], mode: ManifestMode) -> Result<Self, ImageError> {
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(ImageError::Manifest(format!(
                "manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit"
            )));
        }
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| ImageError::Manifest(format!("invalid JSON: {error}")))?;
        manifest.validate(mode)?;
        Ok(manifest)
    }

    pub fn validate(&self, mode: ManifestMode) -> Result<(), ImageError> {
        if self.schema_version != 1 {
            return Err(ImageError::Manifest(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        if !valid_identifier(&self.id) || !valid_identifier(&self.family) {
            return Err(ImageError::Manifest(
                "id and family must be non-empty lowercase-safe identifiers".to_string(),
            ));
        }
        if self.revision.trim().is_empty() || self.revision.eq_ignore_ascii_case("main") {
            return Err(ImageError::Manifest(
                "revision must be immutable and non-empty".to_string(),
            ));
        }
        for (source, revision) in &self.source_revisions {
            if source.trim().is_empty()
                || revision.len() != 40
                || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return Err(ImageError::Manifest(
                    "source_revisions must map non-empty source IDs to 40-hex revisions"
                        .to_string(),
                ));
            }
        }
        if self.quantization.trim().is_empty() || self.quantization.len() > 64 {
            return Err(ImageError::Manifest(
                "quantization must be a non-empty bounded label".to_string(),
            ));
        }
        if self.capabilities.is_empty() {
            return Err(ImageError::Manifest(
                "at least one image capability is required".to_string(),
            ));
        }
        let capabilities = self.capabilities.iter().copied().collect::<BTreeSet<_>>();
        if capabilities.len() != self.capabilities.len() {
            return Err(ImageError::Manifest("duplicate capability".to_string()));
        }
        if self.license.spdx.trim().is_empty() || self.license.evidence.trim().is_empty() {
            return Err(ImageError::Manifest(
                "license SPDX identifier and evidence URL are required".to_string(),
            ));
        }
        validate_catalog_url(&self.license.evidence, "license evidence")?;
        validate_limits(&self.limits)?;
        if self.components.is_empty() {
            return Err(ImageError::Manifest(
                "components must not be empty".to_string(),
            ));
        }

        let mut roles = BTreeSet::new();
        let mut role_paths = HashSet::new();
        for component in &self.components {
            if component.files.is_empty() {
                return Err(ImageError::Manifest(format!(
                    "component `{}` has no files",
                    component.role.as_str()
                )));
            }
            if !component.role.is_known() && !component.optional {
                return Err(ImageError::Manifest(format!(
                    "unknown required component role `{}`",
                    component.role.as_str()
                )));
            }
            if matches!(component.format, ComponentFormat::Other(_)) && !component.optional {
                return Err(ImageError::Manifest(format!(
                    "unknown required format `{}`",
                    component.format.as_str()
                )));
            }
            roles.insert(component.role.clone());
            for file in &component.files {
                validate_file(file, mode)?;
                if !role_paths.insert((component.role.clone(), file.path.clone())) {
                    return Err(ImageError::Manifest(format!(
                        "duplicate component file ({}, {})",
                        component.role.as_str(),
                        file.path
                    )));
                }
            }
        }
        for file in &self.license.files {
            validate_file(file, mode)?;
        }
        for required in required_roles(&capabilities) {
            if !roles.contains(&required) {
                return Err(ImageError::MissingComponent(required.as_str().to_string()));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ImageError> {
        self.validate(ManifestMode::Installed)?;
        let mut normalized = self.clone();
        normalized.components.sort_by(|left, right| {
            left.role
                .as_str()
                .cmp(right.role.as_str())
                .then_with(|| first_path(left).cmp(first_path(right)))
        });
        for component in &mut normalized.components {
            component
                .files
                .sort_by(|left, right| left.path.cmp(&right.path));
        }
        normalized
            .license
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));
        let value = serde_json::to_value(normalized)
            .map_err(|error| ImageError::Manifest(error.to_string()))?;
        let mut output = Vec::new();
        write_canonical_value(&value, &mut output)?;
        Ok(output)
    }

    pub fn digest(&self) -> Result<String, ImageError> {
        let mut hash = Sha256::new();
        hash.update(b"xrt-bundle-v1\0");
        hash.update(self.canonical_bytes()?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

#[derive(Debug)]
pub struct ImageModelBundle {
    root: PathBuf,
    manifest: BundleManifest,
    digest: String,
}

impl ImageModelBundle {
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ImageError> {
        let root = fs::canonicalize(root.as_ref())?;
        if !root.is_dir() {
            return Err(ImageError::Manifest(format!(
                "bundle root `{}` is not a directory",
                root.display()
            )));
        }
        let manifest_path = root.join(MANIFEST_FILE);
        reject_symlink(&manifest_path)?;
        let metadata = fs::metadata(&manifest_path)?;
        if metadata.len() > MAX_MANIFEST_BYTES {
            return Err(ImageError::Manifest(format!(
                "manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit"
            )));
        }
        let bytes = fs::read(&manifest_path)?;
        let manifest = BundleManifest::from_json_bytes(&bytes, ManifestMode::Installed)?;
        Self::open_validated(root, manifest)
    }

    /// Open an explicit local-import manifest without requiring it to have
    /// already been written into the source directory. This keeps raw model
    /// discovery read-only while applying the same path, size, and digest
    /// verification used for installed bundles.
    pub fn open_local_import(
        root: impl AsRef<Path>,
        manifest_bytes: &[u8],
    ) -> Result<Self, ImageError> {
        let root = fs::canonicalize(root.as_ref())?;
        if !root.is_dir() {
            return Err(ImageError::Manifest(format!(
                "bundle root `{}` is not a directory",
                root.display()
            )));
        }
        let manifest = BundleManifest::from_json_bytes(manifest_bytes, ManifestMode::LocalImport)?;
        Self::open_validated(root, manifest)
    }

    fn open_validated(root: PathBuf, manifest: BundleManifest) -> Result<Self, ImageError> {
        verify_files(&root, &manifest)?;
        let digest = manifest.digest()?;
        Ok(Self {
            root,
            manifest,
            digest,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    #[cfg(feature = "test-util")]
    pub(crate) fn synthetic(manifest: BundleManifest) -> Result<Self, ImageError> {
        manifest.validate(ManifestMode::Installed)?;
        let digest = manifest.digest()?;
        Ok(Self {
            root: PathBuf::new(),
            manifest,
            digest,
        })
    }
}

fn required_roles(capabilities: &BTreeSet<ImageCapability>) -> BTreeSet<ComponentRole> {
    let mut roles = BTreeSet::new();
    if capabilities.contains(&ImageCapability::Generate)
        || capabilities.contains(&ImageCapability::Edit)
        || capabilities.contains(&ImageCapability::Inpaint)
    {
        roles.extend([
            ComponentRole::Transformer,
            ComponentRole::TextEncoder,
            ComponentRole::Tokenizer,
            ComponentRole::Vae,
            ComponentRole::Scheduler,
        ]);
    }
    if capabilities.contains(&ImageCapability::Edit) {
        roles.insert(ComponentRole::Processor);
    }
    roles
}

fn validate_limits(limits: &BundleLimits) -> Result<(), ImageError> {
    if limits.max_sequence_length == 0
        || limits.max_width == 0
        || limits.max_height == 0
        || limits.max_pixels == 0
    {
        return Err(ImageError::Manifest(
            "all bundle limits must be positive".to_string(),
        ));
    }
    let rectangular_max = u64::from(limits.max_width)
        .checked_mul(u64::from(limits.max_height))
        .ok_or_else(|| ImageError::Manifest("bundle dimension limit overflow".to_string()))?;
    if limits.max_pixels > rectangular_max {
        return Err(ImageError::Manifest(
            "max_pixels exceeds max_width * max_height".to_string(),
        ));
    }
    Ok(())
}

fn validate_file(file: &BundleFile, mode: ManifestMode) -> Result<(), ImageError> {
    validate_relative_path(&file.path)?;
    if file.size_bytes == 0 {
        return Err(ImageError::Manifest(format!(
            "file `{}` has zero size",
            file.path
        )));
    }
    if file.sha256.len() != 64
        || !file
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ImageError::Manifest(format!(
            "file `{}` does not have a lowercase 64-hex SHA-256",
            file.path
        )));
    }
    match mode {
        ManifestMode::Catalog => {
            if file.source_kind.is_some() {
                return Err(ImageError::Manifest(format!(
                    "catalog file `{}` cannot set source_kind",
                    file.path
                )));
            }
            validate_catalog_url(
                file.source.as_deref().ok_or_else(|| {
                    ImageError::Manifest(format!("catalog file `{}` is missing source", file.path))
                })?,
                "component source",
            )?;
        }
        ManifestMode::LocalImport => {
            if file.source.is_some() || file.source_kind.as_deref() != Some("local") {
                return Err(ImageError::Manifest(format!(
                    "local file `{}` must use source_kind=local without a source URL",
                    file.path
                )));
            }
        }
        ManifestMode::Installed => match (file.source.as_deref(), file.source_kind.as_deref()) {
            (Some(source), None) => validate_catalog_url(source, "component source")?,
            (None, Some("local")) => {}
            _ => {
                return Err(ImageError::Manifest(format!(
                    "file `{}` has an invalid installed source declaration",
                    file.path
                )))
            }
        },
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), ImageError> {
    if value.is_empty() || value.contains('\\') {
        return Err(ImageError::Manifest(format!(
            "unsafe bundle path `{value}`"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ImageError::Manifest(format!(
            "unsafe bundle path `{value}`"
        )));
    }
    Ok(())
}

fn validate_catalog_url(value: &str, label: &str) -> Result<(), ImageError> {
    let Some(remainder) = value.strip_prefix("https://") else {
        return Err(ImageError::Manifest(format!("{label} must use HTTPS")));
    };
    let authority = remainder.split('/').next().unwrap_or_default();
    if authority.is_empty()
        || authority.contains('@')
        || value.contains('?')
        || value.contains('#')
        || value.contains("/resolve/main/")
        || value.contains("/blob/main/")
    {
        return Err(ImageError::Manifest(format!(
            "{label} must be credential-free and revision-pinned"
        )));
    }
    Ok(())
}

fn verify_files(root: &Path, manifest: &BundleManifest) -> Result<(), ImageError> {
    for file in manifest
        .components
        .iter()
        .flat_map(|component| component.files.iter())
        .chain(manifest.license.files.iter())
    {
        let path = root.join(&file.path);
        reject_symlink(&path)?;
        let canonical = fs::canonicalize(&path)
            .map_err(|error| ImageError::MissingComponent(format!("{}: {error}", file.path)))?;
        if !canonical.starts_with(root) || !canonical.is_file() {
            return Err(ImageError::CorruptComponent(format!(
                "file `{}` escapes the bundle root",
                file.path
            )));
        }
        let actual_size = fs::metadata(&canonical)?.len();
        if actual_size != file.size_bytes {
            return Err(ImageError::CorruptComponent(format!(
                "file `{}` is {actual_size} bytes, expected {}",
                file.path, file.size_bytes
            )));
        }
        let actual_hash = sha256_file(&canonical)?;
        if actual_hash != file.sha256 {
            return Err(ImageError::Checksum(file.path.clone()));
        }
    }
    Ok(())
}

fn reject_symlink(path: &Path) -> Result<(), ImageError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| ImageError::MissingComponent(format!("{}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() {
        return Err(ImageError::CorruptComponent(format!(
            "symlink is not allowed: {}",
            path.display()
        )));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, ImageError> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut hash = Sha256::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn write_canonical_value(value: &Value, output: &mut Vec<u8>) -> Result<(), ImageError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(value) => output.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(value) => output.extend_from_slice(
            serde_json::to_string(value)
                .map_err(|error| ImageError::Manifest(error.to_string()))?
                .as_bytes(),
        ),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_value(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(
                    serde_json::to_string(key)
                        .map_err(|error| ImageError::Manifest(error.to_string()))?
                        .as_bytes(),
                );
                output.push(b':');
                write_canonical_value(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn first_path(component: &BundleComponent) -> &str {
    component
        .files
        .iter()
        .map(|file| file.path.as_str())
        .min()
        .unwrap_or("")
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str) -> BundleFile {
        BundleFile {
            path: path.to_string(),
            size_bytes: 1,
            sha256: "00".repeat(32),
            source: Some(format!(
                "https://huggingface.co/x/y/resolve/0123456789abcdef/{path}"
            )),
            source_kind: None,
        }
    }

    fn manifest() -> BundleManifest {
        BundleManifest {
            schema_version: 1,
            id: "fixture-q4_k_m".to_string(),
            family: "qwen-image".to_string(),
            revision: "0123456789abcdef".to_string(),
            source_revisions: BTreeMap::new(),
            capabilities: vec![ImageCapability::Generate],
            license: BundleLicense {
                spdx: "Apache-2.0".to_string(),
                evidence: "https://huggingface.co/x/y/blob/0123456789abcdef/README.md".to_string(),
                files: Vec::new(),
            },
            quantization: "Q4_K_M".to_string(),
            components: [
                ComponentRole::Vae,
                ComponentRole::Scheduler,
                ComponentRole::Tokenizer,
                ComponentRole::TextEncoder,
                ComponentRole::Transformer,
            ]
            .into_iter()
            .map(|role| BundleComponent {
                format: if role == ComponentRole::Transformer {
                    ComponentFormat::Gguf
                } else {
                    ComponentFormat::Json
                },
                files: vec![file(&format!("{}/asset.bin", role.as_str()))],
                role,
                optional: false,
            })
            .collect(),
            limits: BundleLimits {
                max_sequence_length: 512,
                max_width: 64,
                max_height: 64,
                max_pixels: 4_096,
            },
        }
    }

    #[test]
    fn canonical_digest_is_independent_of_component_order() {
        let first = manifest();
        let mut second = first.clone();
        second.components.reverse();
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn catalog_rejects_mutable_or_credentialed_sources() {
        let mut candidate = manifest();
        candidate.components[0].files[0].source =
            Some("https://token@example.com/x/resolve/main/model.gguf".to_string());
        assert_eq!(
            candidate
                .validate(ManifestMode::Catalog)
                .unwrap_err()
                .kind(),
            crate::ImageErrorKind::Manifest
        );
    }

    #[test]
    fn manifest_rejects_path_traversal() {
        let mut candidate = manifest();
        candidate.components[0].files[0].path = "../escape".to_string();
        assert!(candidate.validate(ManifestMode::Catalog).is_err());
    }

    #[test]
    fn local_import_manifest_opens_without_mutating_the_source_directory() {
        let directory = tempfile::tempdir().unwrap();
        let mut candidate = manifest();
        let payload = b"local-component";
        let digest = format!("{:x}", Sha256::digest(payload));
        for file in candidate
            .components
            .iter_mut()
            .flat_map(|component| component.files.iter_mut())
        {
            file.size_bytes = payload.len() as u64;
            file.sha256 = digest.clone();
            file.source = None;
            file.source_kind = Some("local".to_string());
            let path = directory.path().join(&file.path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, payload).unwrap();
        }
        let bytes = serde_json::to_vec_pretty(&candidate).unwrap();
        let bundle = ImageModelBundle::open_local_import(directory.path(), &bytes).unwrap();
        assert_eq!(bundle.manifest(), &candidate);
        assert!(!directory.path().join(MANIFEST_FILE).exists());
    }
}
