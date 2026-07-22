use std::{
    fs::{self, File},
    io::Read,
    path::Path,
    sync::Arc,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use xrt_core::{Result, XrtError};
use xrt_models::LlamaConfig;

use crate::{expert_placement::ExpertPlacementSnapshot, moe_config::MoeAcceleration};

pub const MOE_PLACEMENT_MANIFEST_VERSION: u32 = 1;
pub const MAX_MOE_PLACEMENT_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MoePlacementManifest {
    schema_version: u32,
    model_sha256: String,
    config_sha256: String,
    architecture: String,
    quantization: String,
    layer_count: usize,
    expert_count: usize,
    gpu_expert_budget_bytes: u64,
    expert_bytes: u64,
    layers: Vec<MoePlacementManifestLayer>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MoePlacementManifestLayer {
    layer_index: usize,
    gpu_experts: Vec<usize>,
}

pub(crate) struct MoePlacementManifestContext<'a> {
    pub model_sha256: &'a str,
    pub config_sha256: &'a str,
    pub architecture: &'a str,
    pub quantization: &'a str,
    pub layer_count: usize,
    pub expert_count: usize,
    pub gpu_expert_budget_bytes: u64,
    pub acceleration: MoeAcceleration,
    pub expert_costs: &'a [Vec<u64>],
}

pub(crate) struct ValidatedMoePlacementManifest {
    pub placements: Vec<Arc<ExpertPlacementSnapshot>>,
    pub expert_bytes: u64,
    pub expert_slots: usize,
    pub manifest_sha256: String,
}

pub(crate) fn load_moe_placement_manifest(
    path: &Path,
    context: &MoePlacementManifestContext<'_>,
) -> Result<ValidatedMoePlacementManifest> {
    validate_context(context)?;
    let bytes = read_bounded_manifest(path)?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let manifest: MoePlacementManifest = serde_json::from_slice(&bytes).map_err(|error| {
        XrtError::Runtime(format!(
            "failed to parse MoE placement manifest `{}`: {error}",
            path.display()
        ))
    })?;

    if manifest.schema_version != MOE_PLACEMENT_MANIFEST_VERSION {
        return Err(XrtError::Runtime(format!(
            "unsupported MoE placement manifest schema version {}; expected {}",
            manifest.schema_version, MOE_PLACEMENT_MANIFEST_VERSION
        )));
    }
    validate_sha256("model_sha256", &manifest.model_sha256)?;
    validate_sha256("config_sha256", &manifest.config_sha256)?;
    if !manifest
        .model_sha256
        .eq_ignore_ascii_case(context.model_sha256)
    {
        return Err(XrtError::Runtime(
            "MoE placement manifest model_sha256 does not match the exact GGUF".to_string(),
        ));
    }
    if !manifest
        .config_sha256
        .eq_ignore_ascii_case(context.config_sha256)
    {
        return Err(XrtError::Runtime(
            "MoE placement manifest config_sha256 does not match the loaded model geometry"
                .to_string(),
        ));
    }
    for (field, actual, expected) in [
        (
            "architecture",
            manifest.architecture.as_str(),
            context.architecture,
        ),
        (
            "quantization",
            manifest.quantization.as_str(),
            context.quantization,
        ),
    ] {
        if actual != expected {
            return Err(XrtError::Runtime(format!(
                "MoE placement manifest {field} {actual:?} does not match {expected:?}"
            )));
        }
    }
    for (field, actual, expected) in [
        ("layer_count", manifest.layer_count, context.layer_count),
        ("expert_count", manifest.expert_count, context.expert_count),
    ] {
        if actual != expected {
            return Err(XrtError::Runtime(format!(
                "MoE placement manifest {field} {actual} does not match {expected}"
            )));
        }
    }
    if manifest.gpu_expert_budget_bytes != context.gpu_expert_budget_bytes {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest GPU expert budget {} does not match configured budget {}",
            manifest.gpu_expert_budget_bytes, context.gpu_expert_budget_bytes
        )));
    }
    if manifest.layers.len() != context.layer_count {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest contains {} layer entries for {} layers",
            manifest.layers.len(),
            context.layer_count
        )));
    }

    let mut by_layer = vec![None; context.layer_count];
    for layer in manifest.layers {
        if layer.layer_index >= context.layer_count {
            return Err(XrtError::Runtime(format!(
                "MoE placement manifest layer {} is outside 0..{}",
                layer.layer_index, context.layer_count
            )));
        }
        if by_layer[layer.layer_index].is_some() {
            return Err(XrtError::Runtime(format!(
                "MoE placement manifest contains layer {} more than once",
                layer.layer_index
            )));
        }
        if layer.gpu_experts.is_empty() {
            return Err(XrtError::Runtime(format!(
                "MoE placement manifest layer {} assigns no GPU experts",
                layer.layer_index
            )));
        }
        match context.acceleration {
            MoeAcceleration::Hybrid if layer.gpu_experts.len() >= context.expert_count => {
                return Err(XrtError::Runtime(format!(
                    "hybrid MoE placement layer {} must retain at least one CPU expert",
                    layer.layer_index
                )));
            }
            MoeAcceleration::Gpu if layer.gpu_experts.len() != context.expert_count => {
                return Err(XrtError::Runtime(format!(
                    "GPU MoE placement layer {} must contain all {} logical experts",
                    layer.layer_index, context.expert_count
                )));
            }
            MoeAcceleration::Hybrid | MoeAcceleration::Gpu => {}
            other => {
                return Err(XrtError::Runtime(format!(
                    "profiled placement cannot serve MoE acceleration mode {}",
                    other.as_str()
                )));
            }
        }
        by_layer[layer.layer_index] = Some(Arc::new(ExpertPlacementSnapshot::from_gpu_experts(
            layer.layer_index,
            context.expert_count,
            1,
            &layer.gpu_experts,
        )?));
    }

    let placements = by_layer
        .into_iter()
        .enumerate()
        .map(|(layer, placement)| {
            placement.ok_or_else(|| {
                XrtError::Runtime(format!("MoE placement manifest is missing layer {layer}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let expert_bytes = placement_bytes(&placements, context.expert_costs)?;
    if expert_bytes != manifest.expert_bytes {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest expert_bytes {} does not match computed resident bytes {expert_bytes}",
            manifest.expert_bytes
        )));
    }
    if expert_bytes > context.gpu_expert_budget_bytes {
        return Err(XrtError::Runtime(format!(
            "profiled MoE placement requires {expert_bytes} expert bytes, exceeding configured budget {}",
            context.gpu_expert_budget_bytes
        )));
    }
    let expert_slots = placements.iter().try_fold(0usize, |total, placement| {
        total
            .checked_add(placement.gpu_slot_count())
            .ok_or_else(|| {
                XrtError::Runtime("profiled MoE placement slot count overflowed".to_string())
            })
    })?;

    Ok(ValidatedMoePlacementManifest {
        placements,
        expert_bytes,
        expert_slots,
        manifest_sha256,
    })
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| {
        XrtError::Runtime(format!(
            "failed to open model `{}` for placement identity hashing: {error}",
            path.display()
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            XrtError::Runtime(format!(
                "failed to hash model `{}` for placement identity: {error}",
                path.display()
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute the canonical geometry fingerprint required by a placement manifest.
///
/// This intentionally excludes filesystem paths and runtime-only tuning. It is
/// public so offline profiling tools can bind a manifest to the exact model
/// geometry using the same implementation as the loader.
pub fn moe_config_sha256(config: &LlamaConfig) -> String {
    let mut hasher = Sha256::new();
    hash_string(&mut hasher, &config.architecture);
    for value in [
        config.vocab_size,
        config.context_length,
        config.embedding_length,
        config.feed_forward_length,
        config.block_count,
        config.attention_head_count,
        config.attention_head_count_kv,
        config.rope_dimension_count,
        config.expert_count.unwrap_or(usize::MAX),
        config.expert_used_count.unwrap_or(usize::MAX),
        config
            .expert_shared_feed_forward_length
            .unwrap_or(usize::MAX),
        config.ssm_conv_kernel.unwrap_or(usize::MAX),
        config.ssm_state_size.unwrap_or(usize::MAX),
        config.ssm_group_count.unwrap_or(usize::MAX),
        config.ssm_inner_size.unwrap_or(usize::MAX),
        config.ssm_dt_rank.unwrap_or(usize::MAX),
    ] {
        hasher.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }
    for value in [
        config.rms_norm_eps,
        config.rope_freq_base,
        config.rope_freq_scale,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_string(hasher: &mut Sha256, value: &str) {
    hasher.update(u64::try_from(value.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn validate_context(context: &MoePlacementManifestContext<'_>) -> Result<()> {
    validate_sha256("expected model SHA-256", context.model_sha256)?;
    validate_sha256("expected config SHA-256", context.config_sha256)?;
    if context.layer_count == 0
        || context.expert_count == 0
        || context.expert_costs.len() != context.layer_count
        || context
            .expert_costs
            .iter()
            .any(|costs| costs.len() != context.expert_count)
    {
        return Err(XrtError::Runtime(
            "MoE manifest validation context has inconsistent layer/expert geometry".to_string(),
        ));
    }
    Ok(())
}

fn placement_bytes(
    placements: &[Arc<ExpertPlacementSnapshot>],
    expert_costs: &[Vec<u64>],
) -> Result<u64> {
    placements.iter().try_fold(0u64, |total, placement| {
        placement
            .gpu_slots_to_logical()
            .iter()
            .try_fold(total, |subtotal, &logical_expert| {
                let bytes = expert_costs
                    .get(placement.layer_index())
                    .and_then(|costs| costs.get(usize::from(logical_expert)))
                    .copied()
                    .ok_or_else(|| {
                        XrtError::Runtime(format!(
                            "MoE placement cost is missing layer {} expert {logical_expert}",
                            placement.layer_index()
                        ))
                    })?;
                subtotal.checked_add(bytes).ok_or_else(|| {
                    XrtError::Runtime("MoE placement byte count overflowed".to_string())
                })
            })
    })
}

fn validate_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest {field} must be a 64-character hexadecimal SHA-256"
        )));
    }
    Ok(())
}

fn read_bounded_manifest(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path).map_err(|error| {
        XrtError::Runtime(format!(
            "failed to inspect MoE placement manifest `{}`: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest `{}` is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_MOE_PLACEMENT_MANIFEST_BYTES {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest `{}` is {} bytes, exceeding the {}-byte limit",
            path.display(),
            metadata.len(),
            MAX_MOE_PLACEMENT_MANIFEST_BYTES
        )));
    }
    let file = File::open(path).map_err(|error| {
        XrtError::Runtime(format!(
            "failed to open MoE placement manifest `{}`: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_MOE_PLACEMENT_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            XrtError::Runtime(format!(
                "failed to read MoE placement manifest `{}`: {error}",
                path.display()
            ))
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_MOE_PLACEMENT_MANIFEST_BYTES {
        return Err(XrtError::Runtime(format!(
            "MoE placement manifest `{}` grew beyond the {}-byte limit while being read",
            path.display(),
            MAX_MOE_PLACEMENT_MANIFEST_BYTES
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn context<'a>(
        model_sha256: &'a str,
        config_sha256: &'a str,
        expert_costs: &'a [Vec<u64>],
    ) -> MoePlacementManifestContext<'a> {
        MoePlacementManifestContext {
            model_sha256,
            config_sha256,
            architecture: "qwen3",
            quantization: "f32",
            layer_count: 2,
            expert_count: 4,
            gpu_expert_budget_bytes: 100,
            acceleration: MoeAcceleration::Hybrid,
            expert_costs,
        }
    }

    #[test]
    fn profiled_manifest_is_exactly_bound_and_sorted_by_layer() {
        let root = tempdir().unwrap();
        let path = root.path().join("placement.json");
        let model_hash = "a".repeat(64);
        let config_hash = "b".repeat(64);
        let manifest = json!({
            "schema_version": 1,
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "architecture": "qwen3",
            "quantization": "f32",
            "layer_count": 2,
            "expert_count": 4,
            "gpu_expert_budget_bytes": 100,
            "expert_bytes": 50,
            "layers": [
                {"layer_index": 1, "gpu_experts": [1]},
                {"layer_index": 0, "gpu_experts": [0, 2]}
            ]
        });
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let costs = vec![vec![10, 20, 20, 40], vec![20, 20, 30, 40]];
        let validated =
            load_moe_placement_manifest(&path, &context(&model_hash, &config_hash, &costs))
                .unwrap();

        assert_eq!(validated.expert_bytes, 50);
        assert_eq!(validated.expert_slots, 3);
        assert_eq!(validated.placements[0].gpu_slots_to_logical(), &[0, 2]);
        assert_eq!(validated.placements[1].gpu_slots_to_logical(), &[1]);
        assert_eq!(validated.manifest_sha256.len(), 64);
    }

    #[test]
    fn profiled_manifest_rejects_unknown_duplicate_and_mismatched_data() {
        let root = tempdir().unwrap();
        let path = root.path().join("placement.json");
        let model_hash = "a".repeat(64);
        let config_hash = "b".repeat(64);
        let costs = vec![vec![10, 20, 20, 40], vec![20, 20, 30, 40]];
        let validation = context(&model_hash, &config_hash, &costs);
        for manifest in [
            json!({
                "schema_version": 1,
                "model_sha256": model_hash,
                "config_sha256": config_hash,
                "architecture": "qwen3",
                "quantization": "f32",
                "layer_count": 2,
                "expert_count": 4,
                "gpu_expert_budget_bytes": 100,
                "expert_bytes": 30,
                "unknown": true,
                "layers": [
                    {"layer_index": 0, "gpu_experts": [0]},
                    {"layer_index": 1, "gpu_experts": [0]}
                ]
            }),
            json!({
                "schema_version": 1,
                "model_sha256": model_hash,
                "config_sha256": config_hash,
                "architecture": "qwen3",
                "quantization": "f32",
                "layer_count": 2,
                "expert_count": 4,
                "gpu_expert_budget_bytes": 100,
                "expert_bytes": 30,
                "layers": [
                    {"layer_index": 0, "gpu_experts": [0, 0]},
                    {"layer_index": 1, "gpu_experts": [0]}
                ]
            }),
            json!({
                "schema_version": 1,
                "model_sha256": "c".repeat(64),
                "config_sha256": config_hash,
                "architecture": "qwen3",
                "quantization": "f32",
                "layer_count": 2,
                "expert_count": 4,
                "gpu_expert_budget_bytes": 100,
                "expert_bytes": 30,
                "layers": [
                    {"layer_index": 0, "gpu_experts": [0]},
                    {"layer_index": 1, "gpu_experts": [0]}
                ]
            }),
        ] {
            fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            assert!(load_moe_placement_manifest(&path, &validation).is_err());
        }
    }

    #[test]
    fn manifest_reader_enforces_a_hard_size_limit() {
        let root = tempdir().unwrap();
        let path = root.path().join("placement.json");
        fs::write(
            &path,
            vec![b' '; (MAX_MOE_PLACEMENT_MANIFEST_BYTES + 1) as usize],
        )
        .unwrap();
        assert!(read_bounded_manifest(&path).is_err());
    }
}
