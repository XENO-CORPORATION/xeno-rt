//! Experimental Qwen3.6 DFlash drafter admission and resident weights.
//!
//! The target model remains the ordinary XRT Qwen3.6 runtime. This module owns
//! only the auxiliary block drafter selected with
//! `XRT_QWEN_DFLASH_DRAFT_MODEL`; it does not alter GGUF or public API
//! contracts when the opt-in is absent.

use super::*;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

const DFLASH_ARCHITECTURE: &str = "qwen35-dflash-draft";
const UPSTREAM_DFLASH_ARCHITECTURE: &str = "dflash";
const DFLASH_TARGET_LAYER_IDS: [usize; 5] = [1, 16, 31, 46, 61];
const DFLASH_DRAFT_MODEL_ENV: &str = "XRT_QWEN_DFLASH_DRAFT_MODEL";
const DFLASH_CONTEXT_ENV: &str = "XRT_QWEN_DFLASH_CONTEXT";
const DFLASH_TOP4_DIAGNOSTIC_ENV: &str = "XRT_QWEN_DFLASH_TOP4_DIAGNOSTIC";
const DFLASH_TREE_DEPTH_BONUS_DEFAULT: f32 = 0.3;
const DSPARK_CONFIDENCE_MIN_ENV: &str = "XRT_QWEN_DSPARK_CONFIDENCE_MIN";
const DSPARK_DRAFT_PROFILE_US_ENV: &str = "XRT_QWEN_DSPARK_DRAFT_PROFILE_US";
const DSPARK_VERIFY_PROFILE_US_ENV: &str = "XRT_QWEN_DSPARK_VERIFY_PROFILE_US";
const DSPARK_CONFIDENCE_TEMPERATURES_ENV: &str = "XRT_QWEN_DSPARK_CONFIDENCE_TEMPERATURES";
const MTP_MAX_DRAFT_TOKENS_ENV: &str = "XRT_QWEN_MTP_MAX_DRAFT_TOKENS";
const DFLASH_CONTEXT_DEFAULT: usize = 2_048;
const DFLASH_CONTEXT_MAX: usize = 4_096;

#[derive(Debug, Clone, Copy)]
struct DFlashTreeCandidate {
    parent: usize,
    depth: usize,
    rank: usize,
    log_weight: f32,
    score: f32,
    insertion_order: usize,
}

#[derive(Debug, Clone, Copy)]
struct DFlashTreeNode {
    token: u32,
    parent: usize,
    depth: usize,
    rank: usize,
}

/// Builds a prefix-closed draft tree from the block drafter's per-depth top-k
/// distributions. The node budget excludes the target boundary/root row.
///
/// The priority score is cumulative log probability plus a small depth bonus.
/// The latter counteracts the tendency of a fixed 15-node budget to spend all
/// work on shallow siblings. A 0.3 bonus was selected from the checked-in
/// production-suite diagnostic without changing token selection semantics;
/// target verification still decides the only committed path.
fn build_dflash_draft_tree<Row>(topk_rows: &[Row], node_budget: usize) -> Result<MtpDraftTree>
where
    Row: AsRef<[(u32, f32)]>,
{
    if topk_rows.is_empty() || node_budget == 0 {
        return Err(XrtError::Shape(
            "DFlash tree construction requires rows and a non-zero node budget".to_string(),
        ));
    }
    if node_budget >= QWEN35_VERIFY_MAX_ROWS {
        return Err(XrtError::Shape(format!(
            "DFlash tree node budget {node_budget} exceeds the verifier maximum {}",
            QWEN35_VERIFY_MAX_ROWS - 1
        )));
    }
    if topk_rows.iter().any(|row| row.as_ref().is_empty()) {
        return Err(XrtError::Shape(
            "DFlash tree construction received an empty top-k row".to_string(),
        ));
    }

    let first_log_weight = topk_rows[0].as_ref()[0].1;
    let mut next_insertion_order = 1usize;
    let mut frontier = vec![DFlashTreeCandidate {
        parent: 0,
        depth: 1,
        rank: 0,
        log_weight: first_log_weight,
        score: first_log_weight + DFLASH_TREE_DEPTH_BONUS_DEFAULT,
        insertion_order: 0,
    }];
    let mut selected = Vec::with_capacity(node_budget);

    while !frontier.is_empty() && selected.len() < node_budget {
        let mut best = 0usize;
        for index in 1..frontier.len() {
            let ordering = frontier[index]
                .score
                .total_cmp(&frontier[best].score)
                .then_with(|| {
                    // Earlier insertions win exact score ties, matching the
                    // deterministic top-1-before-sibling traversal.
                    frontier[best]
                        .insertion_order
                        .cmp(&frontier[index].insertion_order)
                });
            if ordering.is_gt() {
                best = index;
            }
        }
        let candidate = frontier.swap_remove(best);
        let row = selected.len() + 1;
        let row_candidates = topk_rows[candidate.depth - 1].as_ref();
        let token = row_candidates[candidate.rank].0;
        selected.push(DFlashTreeNode {
            token,
            parent: candidate.parent,
            depth: candidate.depth,
            rank: candidate.rank,
        });

        if candidate.rank + 1 < row_candidates.len() {
            let sibling_log_weight = candidate.log_weight - row_candidates[candidate.rank].1
                + row_candidates[candidate.rank + 1].1;
            frontier.push(DFlashTreeCandidate {
                parent: candidate.parent,
                depth: candidate.depth,
                rank: candidate.rank + 1,
                log_weight: sibling_log_weight,
                score: sibling_log_weight
                    + DFLASH_TREE_DEPTH_BONUS_DEFAULT * candidate.depth as f32,
                insertion_order: next_insertion_order,
            });
            next_insertion_order = next_insertion_order.saturating_add(1);
        }
        if candidate.depth < topk_rows.len() {
            let child_depth = candidate.depth + 1;
            let child_log_weight = candidate.log_weight + topk_rows[candidate.depth].as_ref()[0].1;
            frontier.push(DFlashTreeCandidate {
                parent: row,
                depth: child_depth,
                rank: 0,
                log_weight: child_log_weight,
                score: child_log_weight + DFLASH_TREE_DEPTH_BONUS_DEFAULT * child_depth as f32,
                insertion_order: next_insertion_order,
            });
            next_insertion_order = next_insertion_order.saturating_add(1);
        }
    }

    // Reorder the selected shape into depth-first pre-order. The shape and
    // target-visible hypotheses are unchanged, but the recurrent CUDA kernel
    // can keep state in registers whenever a child immediately follows its
    // parent. Rank zero is visited first to maximize that fast path.
    let mut children = vec![Vec::<usize>::new(); selected.len() + 1];
    for (index, node) in selected.iter().enumerate() {
        children[node.parent].push(index + 1);
    }
    for child_rows in &mut children {
        child_rows.sort_by_key(|&row| (selected[row - 1].rank, row));
    }
    fn visit(row: usize, children: &[Vec<usize>], order: &mut Vec<usize>) {
        for &child in &children[row] {
            order.push(child);
            visit(child, children, order);
        }
    }
    let mut order = Vec::with_capacity(selected.len());
    visit(0, &children, &mut order);
    let mut remap = vec![0usize; selected.len() + 1];
    for (new_index, &old_row) in order.iter().enumerate() {
        remap[old_row] = new_index + 1;
    }
    let tree = MtpDraftTree {
        tokens: order.iter().map(|&row| selected[row - 1].token).collect(),
        parents: order
            .iter()
            .map(|&row| remap[selected[row - 1].parent])
            .collect(),
        depths: order.iter().map(|&row| selected[row - 1].depth).collect(),
    };
    tree.validate()?;
    Ok(tree)
}

#[derive(Debug, Clone, PartialEq)]
struct DSparkHardwareProfile {
    draft_micros: f32,
    /// Target verification time for prefix lengths `0..=N`. Entry zero is
    /// the ordinary one-token target step; entry N verifies N proposals plus
    /// the target boundary row.
    verify_micros: Vec<f32>,
}

impl DSparkHardwareProfile {
    fn select_prefix(&self, conditional_acceptance: &[f32]) -> usize {
        let mut best_prefix = 0usize;
        let mut best_throughput = 1.0 / (self.draft_micros + self.verify_micros[0]);
        let mut prefix_survival = 1.0f32;
        let mut expected_advance = 1.0f32;
        for (index, &conditional) in conditional_acceptance.iter().enumerate() {
            prefix_survival *= conditional.clamp(0.0, 1.0);
            expected_advance += prefix_survival;
            let prefix = index + 1;
            let throughput = expected_advance / (self.draft_micros + self.verify_micros[prefix]);
            if throughput > best_throughput {
                best_throughput = throughput;
                best_prefix = prefix;
            }
        }
        best_prefix
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DFlashDraftConfig {
    pub(super) embedding_length: usize,
    pub(super) block_count: usize,
    pub(super) feed_forward_length: usize,
    pub(super) attention_head_count: usize,
    pub(super) attention_head_count_kv: usize,
    pub(super) head_dim: usize,
    /// Target-model vocabulary used by the boundary token, embeddings, and
    /// target verification.
    pub(super) vocab_size: usize,
    /// Logical vocabulary predicted by the auxiliary drafter. Reduced-vocab
    /// DSpark artifacts map these indices back into target token IDs with the
    /// resident `d2t` table before the next Markov step.
    pub(super) draft_vocab_size: usize,
    /// Physical output width of the draft head. Q8_0 Marlin artifacts may pad
    /// this to a 64-row tile while argmax remains bounded to
    /// `draft_vocab_size`.
    pub(super) draft_vocab_stride: usize,
    pub(super) uses_draft_vocab: bool,
    pub(super) rms_norm_eps: f32,
    pub(super) rope_freq_base: f32,
    /// Maximum block width encoded by the admitted artifact.
    pub(super) block_size: usize,
    /// Fixed row width executed by this backend instance. When the process
    /// explicitly pins an MTP depth, sizing the draft scratch to that width
    /// prevents DSpark from running and discarding the remainder of its
    /// trained 15-row block. Without an explicit pin, retain the full artifact
    /// width so callers may still select a session depth dynamically.
    pub(super) draft_rows: usize,
    pub(super) mask_token_id: u32,
    pub(super) target_layer_ids: Vec<usize>,
    pub(super) is_dspark: bool,
    pub(super) markov_rank: usize,
}

impl DFlashDraftConfig {
    fn block_size_suffix(is_dspark: bool) -> &'static str {
        if is_dspark {
            "block_size"
        } else {
            "dflash.block_size"
        }
    }

    fn fixed_draft_rows(block_size: usize, is_dspark: bool, max_draft_tokens: usize) -> usize {
        let proposals = max_draft_tokens.clamp(1, 15);
        if is_dspark {
            proposals.min(block_size)
        } else {
            // The original DFlash layout includes the boundary/anchor row in
            // its block, whereas DSpark predicts from every backbone row.
            proposals.saturating_add(1).min(block_size)
        }
    }

    fn draft_rows_from_env(block_size: usize, is_dspark: bool) -> usize {
        env::var(MTP_MAX_DRAFT_TOKENS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .map_or(block_size, |max_draft_tokens| {
                Self::fixed_draft_rows(block_size, is_dspark, max_draft_tokens)
            })
    }

    pub(super) fn capture_rows(&self) -> usize {
        self.draft_rows + usize::from(self.is_dspark)
    }

    pub(super) fn context_capacity_from_env() -> usize {
        env::var(DFLASH_CONTEXT_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DFLASH_CONTEXT_DEFAULT)
            .clamp(16, DFLASH_CONTEXT_MAX)
    }

    fn required_usize(gguf: &GgufFile, prefix: &str, suffix: &str) -> Result<usize> {
        let key = format!("{prefix}.{suffix}");
        gguf.metadata_usize(&key).ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "DFlash draft GGUF is missing required metadata `{key}`"
            ))
        })
    }

    fn required_f32(gguf: &GgufFile, prefix: &str, suffix: &str) -> Result<f32> {
        let key = format!("{prefix}.{suffix}");
        gguf.metadata_f32(&key).ok_or_else(|| {
            XrtError::InvalidMetadata(format!(
                "DFlash draft GGUF is missing required metadata `{key}`"
            ))
        })
    }

    pub(super) fn from_gguf(gguf: &GgufFile, target: &LlamaConfig) -> Result<Self> {
        let architecture = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| {
                XrtError::InvalidMetadata(
                    "DFlash draft GGUF is missing `general.architecture`".to_string(),
                )
            })?;
        if architecture != DFLASH_ARCHITECTURE && architecture != UPSTREAM_DFLASH_ARCHITECTURE {
            return Err(XrtError::Unsupported(format!(
                "DFlash draft architecture `{architecture}` is not an admitted `{DFLASH_ARCHITECTURE}` or upstream `{UPSTREAM_DFLASH_ARCHITECTURE}` contract"
            )));
        }
        let is_dspark = architecture == UPSTREAM_DFLASH_ARCHITECTURE
            && gguf.tensor_info("markov_w1.weight").is_some()
            && gguf.tensor_info("markov_w2.weight").is_some();
        let prefix = if is_dspark {
            UPSTREAM_DFLASH_ARCHITECTURE
        } else {
            DFLASH_ARCHITECTURE
        };
        let target_layer_ids = if is_dspark {
            gguf.metadata_array("dflash.target_layers")
                .and_then(|values| values.as_u32_vec())
                .ok_or_else(|| {
                    XrtError::InvalidMetadata(
                        "DSpark GGUF is missing `dflash.target_layers`".to_string(),
                    )
                })?
                .into_iter()
                .map(|value| {
                    usize::try_from(value)
                        .ok()
                        .and_then(|value| value.checked_sub(1))
                        .ok_or_else(|| {
                            XrtError::InvalidMetadata(format!(
                                "DSpark target layer {value} cannot be converted to XRT indexing"
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let n_target_layers = Self::required_usize(gguf, prefix, "dflash.n_target_layers")?;
            let values = gguf
                .metadata_array(&format!("{prefix}.dflash.target_layer_ids"))
                .and_then(|values| values.as_u32_vec())
                .map(|values| values.into_iter().map(|value| value as usize).collect())
                .unwrap_or_else(|| DFLASH_TARGET_LAYER_IDS.to_vec());
            if values.len() != n_target_layers {
                return Err(XrtError::InvalidMetadata(format!(
                    "DFlash target-layer id count {} does not match declared count {n_target_layers}",
                    values.len()
                )));
            }
            values
        };
        let mask_key = if is_dspark {
            "tokenizer.ggml.mask_token_id"
        } else {
            "qwen35-dflash-draft.dflash.mask_token_id"
        };
        let mask_token_id = gguf
            .metadata_usize(mask_key)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                XrtError::InvalidMetadata(format!(
                    "DFlash draft GGUF is missing a valid `{mask_key}`"
                ))
            })?;
        let markov_rank = if is_dspark {
            gguf.require_tensor("markov_w1.weight")?.row_len()
        } else {
            0
        };
        let has_draft_vocab_map = gguf.tensor_info("d2t").is_some();
        let has_draft_output_head = gguf.tensor_info("lm_head.weight").is_some();
        if has_draft_vocab_map != has_draft_output_head {
            return Err(XrtError::InvalidTensor(
                "DSpark reduced-vocabulary artifacts require both `d2t` and `lm_head.weight`"
                    .to_string(),
            ));
        }
        let draft_vocab_size = gguf
            .tensor_info("d2t")
            .map(|info| info.numel())
            .unwrap_or(target.vocab_size);
        let draft_vocab_stride = gguf
            .tensor_info("lm_head.weight")
            .map(|info| info.rows())
            .unwrap_or(target.vocab_size);
        let block_size = Self::required_usize(gguf, prefix, Self::block_size_suffix(is_dspark))?;
        let config = Self {
            embedding_length: Self::required_usize(gguf, prefix, "embedding_length")?,
            block_count: Self::required_usize(gguf, prefix, "block_count")?,
            feed_forward_length: Self::required_usize(gguf, prefix, "feed_forward_length")?,
            attention_head_count: Self::required_usize(gguf, prefix, "attention.head_count")?,
            attention_head_count_kv: Self::required_usize(gguf, prefix, "attention.head_count_kv")?,
            head_dim: Self::required_usize(gguf, prefix, "attention.key_length")?,
            vocab_size: if is_dspark {
                target.vocab_size
            } else {
                Self::required_usize(gguf, prefix, "vocab_size")?
            },
            draft_vocab_size,
            draft_vocab_stride,
            uses_draft_vocab: has_draft_vocab_map,
            rms_norm_eps: Self::required_f32(gguf, prefix, "attention.layer_norm_rms_epsilon")?,
            rope_freq_base: Self::required_f32(gguf, prefix, "rope.freq_base")?,
            block_size,
            draft_rows: Self::draft_rows_from_env(block_size, is_dspark),
            mask_token_id,
            target_layer_ids,
            is_dspark,
            markov_rank,
        };
        config.validate_target(target)?;
        Ok(config)
    }

    fn validate_target(&self, target: &LlamaConfig) -> Result<()> {
        if !target.is_qwen35_family() || !target.is_hybrid() || target.is_moe() {
            return Err(XrtError::Unsupported(
                "the admitted DFlash draft path requires a dense Qwen3.6 hybrid target".to_string(),
            ));
        }
        // DFlash is an auxiliary transformer, not a copy of the target
        // attention geometry. Its 32 query / 8 KV heads at width 128 are part
        // of the admitted drafter contract below; only the shared hidden and
        // vocabulary spaces must agree with the target.
        if self.embedding_length != target.embedding_length || self.vocab_size != target.vocab_size
        {
            return Err(XrtError::Unsupported(format!(
                "DFlash drafter spaces do not match the target: draft dim/vocab={}/{}, target={}/{}",
                self.embedding_length,
                self.vocab_size,
                target.embedding_length,
                target.vocab_size,
            )));
        }
        if self.block_count != 5
            || (!self.is_dspark && self.block_size != 16)
            || (self.is_dspark
                && (self.markov_rank != 256
                    || (!self.uses_draft_vocab && self.block_size != 15)
                    || (self.uses_draft_vocab && self.block_size != 8)))
            || self.target_layer_ids != DFLASH_TARGET_LAYER_IDS
            || self.feed_forward_length != 17_408
            || self.embedding_length != 5_120
            || self.attention_head_count != 32
            || self.attention_head_count_kv != 8
            || self.head_dim != 128
        {
            return Err(XrtError::Unsupported(format!(
                "DFlash drafter does not match the admitted Qwen3.6-27B DFlash/DSpark contract: {self:?}"
            )));
        }
        if self.draft_vocab_size == 0
            || self.draft_vocab_size > self.vocab_size
            || self.draft_vocab_stride < self.draft_vocab_size
            || (!self.uses_draft_vocab
                && (self.draft_vocab_size != self.vocab_size
                    || self.draft_vocab_stride != self.vocab_size))
        {
            return Err(XrtError::Unsupported(format!(
                "DFlash draft vocabulary is invalid: logical={} stride={} target={} mapped={}",
                self.draft_vocab_size,
                self.draft_vocab_stride,
                self.vocab_size,
                self.uses_draft_vocab,
            )));
        }
        if self.mask_token_id as usize >= self.vocab_size {
            return Err(XrtError::InvalidMetadata(format!(
                "DFlash mask token {} exceeds vocabulary size {}",
                self.mask_token_id, self.vocab_size
            )));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_freq_base.is_finite()
            || self.rope_freq_base <= 0.0
        {
            return Err(XrtError::InvalidMetadata(
                "DFlash normalization/RoPE metadata must be finite and positive".to_string(),
            ));
        }
        if self.rope_freq_base != target.rope_freq_base {
            return Err(XrtError::Unsupported(format!(
                "DFlash RoPE base {} does not match target RoPE base {}; reconvert the draft artifact from the matching model config",
                self.rope_freq_base, target.rope_freq_base
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) struct CudaQwen35DFlashLayerCache {
    pub(super) key: CudaF32Buffer,
    pub(super) value: CudaF32Buffer,
}

#[derive(Debug)]
pub(super) struct CudaQwen35DFlashState {
    pub(super) capacity: usize,
    pub(super) len: usize,
    pub(super) projection_streams: Vec<CudaExecutionStream>,
    pub(super) feature_rows: CudaF32Buffer,
    pub(super) feature_compact: CudaF32Buffer,
    pub(super) projected_rows: CudaF32Buffer,
    pub(super) normed_rows: CudaF32Buffer,
    pub(super) k_update: CudaF32Buffer,
    pub(super) v_update: CudaF32Buffer,
    pub(super) cache: Vec<CudaQwen35DFlashLayerCache>,
    pub(super) noise_a: CudaF32Buffer,
    pub(super) noise_b: CudaF32Buffer,
    pub(super) noise_normed: CudaF32Buffer,
    pub(super) query: CudaF32Buffer,
    pub(super) noise_key: CudaF32Buffer,
    pub(super) noise_value: CudaF32Buffer,
    pub(super) attention: CudaF32Buffer,
    pub(super) ffn_gate: CudaF32Buffer,
    pub(super) ffn_up: CudaF32Buffer,
    pub(super) hidden_temp: CudaF32Buffer,
    pub(super) logits: CudaF32Buffer,
    pub(super) argmax_indices: CudaF32Buffer,
    pub(super) top4: CudaF32Buffer,
    pub(super) markov_input: Option<CudaF32Buffer>,
    pub(super) markov_bias: Option<CudaF32Buffer>,
    pub(super) markov_argmax: Option<CudaF32Buffer>,
    pub(super) markov_draft_argmax: Option<CudaF32Buffer>,
    pub(super) confidence_input: Option<CudaF32Buffer>,
    pub(super) confidence_logit: Option<CudaF32Buffer>,
    pub(super) confidence_logits: Option<CudaF32Buffer>,
}

impl CudaQwen35DFlashState {
    pub(super) fn device_bytes(config: &DFlashDraftConfig, capacity: usize) -> Result<u64> {
        let rows = config.draft_rows;
        let dim = config.embedding_length;
        let q_width = config
            .attention_head_count
            .checked_mul(config.head_dim)
            .ok_or_else(|| XrtError::Shape("DFlash query width overflowed".to_string()))?;
        let kv_width = config
            .attention_head_count_kv
            .checked_mul(config.head_dim)
            .ok_or_else(|| XrtError::Shape("DFlash KV width overflowed".to_string()))?;
        let feature_width = dim
            .checked_mul(config.target_layer_ids.len())
            .ok_or_else(|| XrtError::Shape("DFlash feature width overflowed".to_string()))?;
        let per_row = feature_width
            .checked_add(dim.checked_mul(5).ok_or_else(|| {
                XrtError::Shape("DFlash hidden scratch width overflowed".to_string())
            })?)
            .and_then(|value| value.checked_add(q_width.checked_mul(2)?))
            .and_then(|value| value.checked_add(kv_width.checked_mul(4)?))
            .and_then(|value| value.checked_add(config.feed_forward_length.checked_mul(2)?))
            .and_then(|value| value.checked_add(config.draft_vocab_stride))
            .and_then(|value| value.checked_add(9))
            .ok_or_else(|| XrtError::Shape("DFlash row scratch size overflowed".to_string()))?;
        let cache_elements = config
            .block_count
            .checked_mul(2)
            .and_then(|value| value.checked_mul(capacity))
            .and_then(|value| value.checked_mul(kv_width))
            .ok_or_else(|| XrtError::Shape("DFlash cache size overflowed".to_string()))?;
        let markov_elements = if config.is_dspark {
            config
                .markov_rank
                .checked_add(config.draft_vocab_stride)
                .and_then(|value| value.checked_add(2))
                .ok_or_else(|| XrtError::Shape("DSpark Markov scratch overflowed".to_string()))?
        } else {
            0
        };
        let confidence_elements = if config.is_dspark {
            config
                .embedding_length
                .checked_add(config.markov_rank)
                .and_then(|value| value.checked_add(1))
                .and_then(|value| value.checked_add(rows))
                .ok_or_else(|| {
                    XrtError::Shape("DSpark confidence scratch overflowed".to_string())
                })?
        } else {
            0
        };
        let tree_feature_elements = feature_width
            .checked_mul(config.capture_rows())
            .ok_or_else(|| XrtError::Shape("DFlash tree feature staging overflowed".to_string()))?;
        let capture_extra = tree_feature_elements
            .checked_add(if config.is_dspark {
                feature_width
                    .checked_add(dim.checked_mul(2).ok_or_else(|| {
                        XrtError::Shape("DSpark capture hidden scratch overflowed".to_string())
                    })?)
                    .and_then(|value| value.checked_add(kv_width.checked_mul(2)?))
                    .ok_or_else(|| {
                        XrtError::Shape("DSpark capture scratch overflowed".to_string())
                    })?
            } else {
                0
            })
            .ok_or_else(|| XrtError::Shape("DFlash capture scratch overflowed".to_string()))?;
        per_row
            .checked_mul(rows)
            .and_then(|value| value.checked_add(cache_elements))
            .and_then(|value| value.checked_add(markov_elements))
            .and_then(|value| value.checked_add(confidence_elements))
            .and_then(|value| value.checked_add(capture_extra))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| XrtError::Shape("DFlash device byte count overflowed".to_string()))
    }

    pub(super) fn allocate(
        device: &CudaDevice,
        config: &DFlashDraftConfig,
        capacity: usize,
    ) -> Result<Self> {
        let rows = config.draft_rows;
        let capture_rows = config.capture_rows();
        let dim = config.embedding_length;
        let q_width = config.attention_head_count * config.head_dim;
        let kv_width = config.attention_head_count_kv * config.head_dim;
        let feature_width = dim * config.target_layer_ids.len();
        let sized = |width: usize, label: &str| {
            rows.checked_mul(width)
                .ok_or_else(|| XrtError::Shape(format!("DFlash {label} size overflowed")))
        };
        let sized_capture = |width: usize, label: &str| {
            capture_rows
                .checked_mul(width)
                .ok_or_else(|| XrtError::Shape(format!("DFlash {label} size overflowed")))
        };
        let mut cache = Vec::with_capacity(config.block_count);
        for _ in 0..config.block_count {
            let elements = capacity
                .checked_mul(kv_width)
                .ok_or_else(|| XrtError::Shape("DFlash layer cache size overflowed".to_string()))?;
            cache.push(CudaQwen35DFlashLayerCache {
                key: device.zeros_f32(elements)?,
                value: device.zeros_f32(elements)?,
            });
        }
        Ok(Self {
            capacity,
            len: 0,
            projection_streams: (0..2)
                .map(|_| device.create_execution_stream())
                .collect::<Result<Vec<_>>>()?,
            feature_rows: device.zeros_f32(sized_capture(feature_width, "feature rows")?)?,
            feature_compact: device
                .zeros_f32(sized_capture(feature_width, "compact feature rows")?)?,
            projected_rows: device.zeros_f32(sized_capture(dim, "projected rows")?)?,
            normed_rows: device.zeros_f32(sized_capture(dim, "normalized rows")?)?,
            k_update: device.zeros_f32(sized_capture(kv_width, "key update")?)?,
            v_update: device.zeros_f32(sized_capture(kv_width, "value update")?)?,
            cache,
            noise_a: device.zeros_f32(sized(dim, "noise A")?)?,
            noise_b: device.zeros_f32(sized(dim, "noise B")?)?,
            noise_normed: device.zeros_f32(sized(dim, "noise norm")?)?,
            query: device.zeros_f32(sized(q_width, "query")?)?,
            noise_key: device.zeros_f32(sized(kv_width, "noise key")?)?,
            noise_value: device.zeros_f32(sized(kv_width, "noise value")?)?,
            attention: device.zeros_f32(sized(q_width, "attention")?)?,
            ffn_gate: device.zeros_f32(sized(config.feed_forward_length, "FFN gate")?)?,
            ffn_up: device.zeros_f32(sized(config.feed_forward_length, "FFN up")?)?,
            hidden_temp: device.zeros_f32(sized(dim, "hidden temporary")?)?,
            logits: device.zeros_f32(sized(config.draft_vocab_stride, "logits")?)?,
            argmax_indices: device.zeros_f32(rows)?,
            top4: device.zeros_f32(sized(8, "top-4 output")?)?,
            markov_input: config
                .is_dspark
                .then(|| device.zeros_f32(config.markov_rank))
                .transpose()?,
            markov_bias: config
                .is_dspark
                .then(|| device.zeros_f32(config.draft_vocab_stride))
                .transpose()?,
            markov_argmax: config.is_dspark.then(|| device.zeros_f32(1)).transpose()?,
            markov_draft_argmax: config
                .uses_draft_vocab
                .then(|| device.zeros_f32(1))
                .transpose()?,
            confidence_input: config
                .is_dspark
                .then(|| device.zeros_f32(config.embedding_length + config.markov_rank))
                .transpose()?,
            confidence_logit: config.is_dspark.then(|| device.zeros_f32(1)).transpose()?,
            confidence_logits: config
                .is_dspark
                .then(|| device.zeros_f32(rows))
                .transpose()?,
        })
    }

    pub(super) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
    }
}

pub(super) struct DFlashDraftPlan {
    pub(super) path: PathBuf,
    pub(super) gguf: Arc<GgufFile>,
    pub(super) config: DFlashDraftConfig,
    pub(super) resident_bytes: u64,
}

impl DFlashDraftPlan {
    pub(super) fn from_env(target: &LlamaConfig) -> Result<Option<Self>> {
        let Some(path) = env::var_os(DFLASH_DRAFT_MODEL_ENV)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
        else {
            return Ok(None);
        };
        let gguf = Arc::new(GgufFile::open(&path).map_err(|error| {
            XrtError::Runtime(format!(
                "failed to open DFlash draft GGUF `{}`: {error}",
                path.display()
            ))
        })?);
        let config = DFlashDraftConfig::from_gguf(&gguf, target)?;
        validate_tensor_contract(&gguf, &config)?;
        let resident_bytes = resident_weight_bytes(&gguf, &config)?;
        Ok(Some(Self {
            path,
            gguf,
            config,
            resident_bytes,
        }))
    }
}

enum ResidentDFlashMatrix {
    Q8_0(Arc<CudaQ8_0Matrix>),
    F16 {
        weights: CudaBytes,
        input_f16: Mutex<CudaBytes>,
        rows: usize,
        cols: usize,
    },
}

impl ResidentDFlashMatrix {
    fn upload(device: &CudaDevice, source: &impl ResidentTensorSource, name: &str) -> Result<Self> {
        let info = source.require_tensor(name)?;
        match info.dtype {
            DType::Q8_0 => {
                let marlin_enabled =
                    env::var_os("XRT_CUDA_DFLASH_Q8_0_MARLIN").is_some_and(|value| {
                        let value = value.to_string_lossy();
                        value == "1" || value.eq_ignore_ascii_case("true")
                    });
                let matrix = if marlin_enabled {
                    device.upload_q8_0_marlin_matrix(
                        source.tensor_data(name)?,
                        info.rows,
                        info.cols,
                    )?
                } else {
                    device.upload_q8_0_matrix(source.tensor_data(name)?, info.rows, info.cols)?
                };
                Ok(Self::Q8_0(Arc::new(matrix)))
            }
            DType::F16 | DType::BF16 => {
                let weights = device.upload_f16_tensor_2d_bytes(
                    name,
                    info.rows,
                    info.cols,
                    info.dtype,
                    source.tensor_data(name)?,
                )?;
                let scratch_bytes = info
                    .cols
                    .checked_mul(16)
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| {
                        XrtError::Shape(format!("DFlash F16 scratch for `{name}` overflowed"))
                    })?;
                Ok(Self::F16 {
                    weights,
                    input_f16: Mutex::new(device.zeros_bytes(scratch_bytes)?),
                    rows: info.rows,
                    cols: info.cols,
                })
            }
            dtype => Err(XrtError::Unsupported(format!(
                "DFlash linear `{name}` requires Q8_0, F16, or BF16 storage, found {dtype:?}"
            ))),
        }
    }

    fn upload_row_major_q8_or_16(
        device: &CudaDevice,
        source: &impl ResidentTensorSource,
        name: &str,
    ) -> Result<Self> {
        let info = source.require_tensor(name)?;
        if info.dtype == DType::Q8_0 {
            return Ok(Self::Q8_0(Arc::new(device.upload_q8_0_matrix(
                source.tensor_data(name)?,
                info.rows,
                info.cols,
            )?)));
        }
        Self::upload(device, source, name)
    }

    fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Q8_0(matrix) => (matrix.rows(), matrix.cols()),
            Self::F16 { rows, cols, .. } => (*rows, *cols),
        }
    }

    fn graph_kind(&self) -> &'static str {
        match self {
            Self::Q8_0(_) => "q8_0",
            Self::F16 { .. } => "f16_cublas",
        }
    }
}

pub(super) struct ResidentQwen35DFlashLayerWeights {
    pub(super) attn_norm: GpuF32Tensor,
    attn_q: ResidentDFlashMatrix,
    attn_k: ResidentDFlashMatrix,
    attn_v: ResidentDFlashMatrix,
    attn_output: ResidentDFlashMatrix,
    pub(super) attn_q_norm: GpuF32Tensor,
    pub(super) attn_k_norm: GpuF32Tensor,
    pub(super) ffn_norm: GpuF32Tensor,
    ffn_gate: ResidentDFlashMatrix,
    ffn_up: ResidentDFlashMatrix,
    ffn_down: ResidentDFlashMatrix,
    pub(super) sliding_window: bool,
}

pub(super) struct ResidentQwen35DFlashWeights {
    pub(super) config: DFlashDraftConfig,
    feature_projection: ResidentDFlashMatrix,
    pub(super) hidden_norm: GpuF32Tensor,
    pub(super) output_norm: GpuF32Tensor,
    pub(super) layers: Vec<ResidentQwen35DFlashLayerWeights>,
    draft_output: Option<ResidentDFlashMatrix>,
    draft_to_target: Option<CudaF32Buffer>,
    markov: Option<ResidentDSparkMarkovWeights>,
    confidence: Option<ResidentDSparkConfidenceWeights>,
}

struct ResidentDSparkConfidenceWeights {
    projection: ResidentDFlashMatrix,
    bias: f32,
}

enum ResidentDSparkMarkovEmbedding {
    F16 {
        weights: CudaBytes,
        rows: usize,
        cols: usize,
    },
    Q8_0(Arc<CudaQ8_0Matrix>),
}

struct ResidentDSparkMarkovWeights {
    embedding: ResidentDSparkMarkovEmbedding,
    projection: ResidentDFlashMatrix,
}

impl ResidentDSparkMarkovWeights {
    fn load_embedding_from_device_token(
        &self,
        device: &CudaDevice,
        token_id: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        match &self.embedding {
            ResidentDSparkMarkovEmbedding::Q8_0(table) => {
                device.embed_q8_0_resident_device_f32_token_into(table, token_id, output)
            }
            ResidentDSparkMarkovEmbedding::F16 {
                weights,
                rows,
                cols,
            } => device
                .embed_f16_resident_device_f32_token_into(weights, *rows, *cols, token_id, output),
        }
    }
}

impl ResidentQwen35DFlashWeights {
    pub(super) fn load(device: &CudaDevice, plan: &DFlashDraftPlan) -> Result<Self> {
        let source = GgufResidentTensorSource::new(&plan.gguf);
        let feature_projection_name = if plan.config.is_dspark {
            "fc.weight"
        } else {
            "dflash.fc.weight"
        };
        let hidden_norm_name = if plan.config.is_dspark {
            "enc.output_norm.weight"
        } else {
            "dflash.hidden_norm.weight"
        };
        let mut layers = Vec::with_capacity(plan.config.block_count);
        for layer in 0..plan.config.block_count {
            layers.push(ResidentQwen35DFlashLayerWeights {
                attn_norm: upload_resident_f32_tensor(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_norm.weight"),
                )?,
                attn_q: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_q.weight"),
                )?,
                attn_k: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_k.weight"),
                )?,
                attn_v: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_v.weight"),
                )?,
                attn_output: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_output.weight"),
                )?,
                attn_q_norm: upload_resident_f32_tensor(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_resident_f32_tensor(
                    device,
                    &source,
                    &format!("blk.{layer}.attn_k_norm.weight"),
                )?,
                ffn_norm: upload_resident_f32_tensor(
                    device,
                    &source,
                    &format!("blk.{layer}.ffn_norm.weight"),
                )?,
                ffn_gate: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.ffn_gate.weight"),
                )?,
                ffn_up: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.ffn_up.weight"),
                )?,
                ffn_down: ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    &format!("blk.{layer}.ffn_down.weight"),
                )?,
                sliding_window: layer + 1 < plan.config.block_count,
            });
        }
        let markov = if plan.config.is_dspark {
            let embedding = source.require_tensor("markov_w1.weight")?;
            let embedding = match embedding.dtype {
                // Markov W1 is an embedding table, not a linear projection.
                // Preserve its row-major layout even when the optional
                // DFlash Marlin projection path is enabled.
                DType::Q8_0 => {
                    ResidentDSparkMarkovEmbedding::Q8_0(Arc::new(device.upload_q8_0_matrix(
                        source.tensor_data("markov_w1.weight")?,
                        embedding.rows,
                        embedding.cols,
                    )?))
                }
                DType::F16 | DType::BF16 => ResidentDSparkMarkovEmbedding::F16 {
                    weights: device.upload_f16_tensor_2d_bytes(
                        "markov_w1.weight",
                        embedding.rows,
                        embedding.cols,
                        embedding.dtype,
                        source.tensor_data("markov_w1.weight")?,
                    )?,
                    rows: embedding.rows,
                    cols: embedding.cols,
                },
                dtype => {
                    return Err(XrtError::Unsupported(format!(
                        "DSpark Markov embedding requires Q8_0, F16, or BF16 storage, found {dtype:?}"
                    )));
                }
            };
            Some(ResidentDSparkMarkovWeights {
                embedding,
                projection: ResidentDFlashMatrix::upload(device, &source, "markov_w2.weight")?,
            })
        } else {
            None
        };
        let confidence = if plan.config.is_dspark
            && source.tensor_info("conf_proj.weight").is_some()
            && source.tensor_info("conf_proj.bias").is_some()
        {
            let bias_info = source.require_tensor("conf_proj.bias")?;
            if bias_info.dtype != DType::F32 || bias_info.numel != 1 {
                return Err(XrtError::InvalidTensor(format!(
                    "DSpark confidence bias must be one F32 value, found dtype={:?} elements={}",
                    bias_info.dtype, bias_info.numel
                )));
            }
            let bias_bytes = source.tensor_data("conf_proj.bias")?;
            let bias = f32::from_le_bytes(bias_bytes.try_into().map_err(|_| {
                XrtError::InvalidTensor(format!(
                    "DSpark confidence bias requires 4 bytes, found {}",
                    bias_bytes.len()
                ))
            })?);
            if !bias.is_finite() {
                return Err(XrtError::InvalidTensor(
                    "DSpark confidence bias must be finite".to_string(),
                ));
            }
            Some(ResidentDSparkConfidenceWeights {
                projection: ResidentDFlashMatrix::upload_row_major_q8_or_16(
                    device,
                    &source,
                    "conf_proj.weight",
                )?,
                bias,
            })
        } else {
            None
        };
        let (draft_output, draft_to_target) = if plan.config.uses_draft_vocab {
            let mapping = decode_dspark_d2t(&source, &plan.config)?;
            (
                Some(ResidentDFlashMatrix::upload(
                    device,
                    &source,
                    "lm_head.weight",
                )?),
                Some(device.upload_f32(&mapping)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            config: plan.config.clone(),
            feature_projection: ResidentDFlashMatrix::upload(
                device,
                &source,
                feature_projection_name,
            )?,
            hidden_norm: upload_resident_f32_tensor(device, &source, hidden_norm_name)?,
            output_norm: upload_resident_f32_tensor(device, &source, "output_norm.weight")?,
            layers,
            draft_output,
            draft_to_target,
            markov,
            confidence,
        })
    }
}

fn validate_tensor_contract(gguf: &GgufFile, config: &DFlashDraftConfig) -> Result<()> {
    let source = GgufResidentTensorSource::new(gguf);
    let dim = config.embedding_length;
    let feature_width = dim
        .checked_mul(config.target_layer_ids.len())
        .ok_or_else(|| XrtError::Shape("DFlash feature width overflowed".to_string()))?;
    let q_width = config
        .attention_head_count
        .checked_mul(config.head_dim)
        .ok_or_else(|| XrtError::Shape("DFlash query width overflowed".to_string()))?;
    let kv_width = config
        .attention_head_count_kv
        .checked_mul(config.head_dim)
        .ok_or_else(|| XrtError::Shape("DFlash KV width overflowed".to_string()))?;

    let feature_projection_name = if config.is_dspark {
        "fc.weight"
    } else {
        "dflash.fc.weight"
    };
    let hidden_norm_name = if config.is_dspark {
        "enc.output_norm.weight"
    } else {
        "dflash.hidden_norm.weight"
    };
    require_linear(&source, feature_projection_name, dim, feature_width)?;
    for name in [hidden_norm_name, "output_norm.weight"] {
        require_vector(&source, name, dim)?;
    }
    for layer in 0..config.block_count {
        for name in ["attn_norm", "ffn_norm"] {
            require_vector(&source, &format!("blk.{layer}.{name}.weight"), dim)?;
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            require_vector(
                &source,
                &format!("blk.{layer}.{name}.weight"),
                config.head_dim,
            )?;
        }
        for (name, rows, cols) in [
            ("attn_q", q_width, dim),
            ("attn_k", kv_width, dim),
            ("attn_v", kv_width, dim),
            ("attn_output", dim, q_width),
            ("ffn_gate", config.feed_forward_length, dim),
            ("ffn_up", config.feed_forward_length, dim),
            ("ffn_down", dim, config.feed_forward_length),
        ] {
            require_linear(&source, &format!("blk.{layer}.{name}.weight"), rows, cols)?;
        }
    }
    if config.is_dspark {
        require_linear(
            &source,
            "markov_w1.weight",
            config.vocab_size,
            config.markov_rank,
        )?;
        require_linear(
            &source,
            "markov_w2.weight",
            config.draft_vocab_stride,
            config.markov_rank,
        )?;
        if config.uses_draft_vocab {
            require_vector(&source, "d2t", config.draft_vocab_size)?;
            if source.require_tensor("d2t")?.dtype != DType::F32 {
                return Err(XrtError::InvalidTensor(
                    "DSpark `d2t` must use F32 exact-integer storage".to_string(),
                ));
            }
            require_linear(&source, "lm_head.weight", config.draft_vocab_stride, dim)?;
            decode_dspark_d2t(&source, config)?;
        }
        match (
            source.tensor_info("conf_proj.weight"),
            source.tensor_info("conf_proj.bias"),
        ) {
            (Some(_), Some(bias)) => {
                require_linear(
                    &source,
                    "conf_proj.weight",
                    1,
                    dim.checked_add(config.markov_rank).ok_or_else(|| {
                        XrtError::Shape("DSpark confidence width overflowed".to_string())
                    })?,
                )?;
                if bias.dtype != DType::F32 || bias.numel != 1 {
                    return Err(XrtError::InvalidTensor(format!(
                        "DSpark confidence bias must be one F32 value, found dtype={:?} elements={}",
                        bias.dtype, bias.numel
                    )));
                }
            }
            (None, None) => {}
            _ => {
                return Err(XrtError::InvalidTensor(
                    "DSpark confidence head requires both `conf_proj.weight` and `conf_proj.bias`"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn decode_dspark_d2t(
    source: &impl ResidentTensorSource,
    config: &DFlashDraftConfig,
) -> Result<Vec<f32>> {
    let info = source.require_tensor("d2t")?;
    if info.dtype != DType::F32 || info.numel != config.draft_vocab_size {
        return Err(XrtError::InvalidTensor(format!(
            "DSpark `d2t` must contain {} F32 values, found dtype={:?} elements={}",
            config.draft_vocab_size, info.dtype, info.numel
        )));
    }
    let bytes = source.tensor_data("d2t")?;
    if bytes.len() != info.numel.saturating_mul(std::mem::size_of::<f32>()) {
        return Err(XrtError::InvalidTensor(format!(
            "DSpark `d2t` byte length {} does not match {} F32 values",
            bytes.len(),
            info.numel
        )));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte d2t chunk")))
        .collect::<Vec<_>>();
    let mut seen = vec![false; config.vocab_size];
    for (draft_id, &target_id) in values.iter().enumerate() {
        if !target_id.is_finite()
            || target_id < 0.0
            || target_id.fract() != 0.0
            || target_id >= config.vocab_size as f32
        {
            return Err(XrtError::InvalidTensor(format!(
                "DSpark `d2t` entry {draft_id} is not a target token ID: {target_id}"
            )));
        }
        let target_id = target_id as usize;
        if std::mem::replace(&mut seen[target_id], true) {
            return Err(XrtError::InvalidTensor(format!(
                "DSpark `d2t` maps more than one draft token to target token {target_id}"
            )));
        }
    }
    Ok(values)
}

fn require_vector(source: &impl ResidentTensorSource, name: &str, expected: usize) -> Result<()> {
    let info = source.require_tensor(name)?;
    if !is_supported_resident_float_tensor(&info) || info.numel != expected {
        return Err(XrtError::InvalidTensor(format!(
            "DFlash tensor `{name}` must be a supported {expected}-element float vector, found dtype={:?} storage={:?} shape={}x{}",
            info.dtype, info.storage, info.rows, info.cols
        )));
    }
    Ok(())
}

fn require_linear(
    source: &impl ResidentTensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let info = source.require_tensor(name)?;
    if info.storage != ResidentTensorStorage::Dense
        || !matches!(info.dtype, DType::Q8_0 | DType::F16 | DType::BF16)
        || info.rows != rows
        || info.cols != cols
    {
        return Err(XrtError::InvalidTensor(format!(
            "DFlash tensor `{name}` must be a dense Q8_0, F16, or BF16 {rows}x{cols} linear, found dtype={:?} storage={:?} shape={}x{}",
            info.dtype, info.storage, info.rows, info.cols
        )));
    }
    Ok(())
}

fn resident_weight_bytes(gguf: &GgufFile, config: &DFlashDraftConfig) -> Result<u64> {
    let source = GgufResidentTensorSource::new(gguf);
    source
        .tensor_infos()
        .into_iter()
        .try_fold(0u64, |total, info| {
            let bytes = if config.is_dspark && info.name == "conf_proj.bias" {
                0
            } else if info.storage == ResidentTensorStorage::Dense
                && matches!(info.dtype, DType::F16 | DType::BF16)
                && info.rows > 1
                && info.cols > 1
            {
                u64::try_from(info.numel)
                    .ok()
                    .and_then(|elements| elements.checked_mul(2))
                    .ok_or_else(|| {
                        XrtError::Runtime(format!(
                            "DFlash 16-bit tensor `{}` resident byte count overflowed",
                            info.name
                        ))
                    })?
            } else if is_supported_resident_float_tensor(&info) {
                cuda_resident_f32_tensor_bytes(&info)?
            } else {
                cuda_matrix_resident_tensor_bytes(&info)?
            };
            total.checked_add(bytes).ok_or_else(|| {
                XrtError::Runtime("DFlash resident weight byte count overflowed".to_string())
            })
        })
}

impl CudaResidentBackend {
    fn dspark_csv_f32_from_env(name: &str) -> Result<Option<Vec<f32>>> {
        let Some(value) = env::var_os(name).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        let text = value.to_string_lossy();
        let values = text
            .split(',')
            .enumerate()
            .map(|(index, value)| {
                let value = value.trim().parse::<f32>().map_err(|error| {
                    XrtError::InvalidMetadata(format!(
                        "invalid `{name}` entry {index} `{}`: {error}",
                        value.trim()
                    ))
                })?;
                if !value.is_finite() || value <= 0.0 {
                    return Err(XrtError::InvalidMetadata(format!(
                        "`{name}` entry {index} must be finite and positive, found {value}"
                    )));
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()?;
        if values.is_empty() {
            return Err(XrtError::InvalidMetadata(format!(
                "`{name}` must contain at least one value"
            )));
        }
        Ok(Some(values))
    }

    fn dspark_confidence_min_from_env() -> Result<Option<f32>> {
        let Some(value) = env::var_os(DSPARK_CONFIDENCE_MIN_ENV).filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let text = value.to_string_lossy();
        let threshold = text.parse::<f32>().map_err(|error| {
            XrtError::InvalidMetadata(format!(
                "invalid `{DSPARK_CONFIDENCE_MIN_ENV}` value `{text}`: {error}"
            ))
        })?;
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(XrtError::InvalidMetadata(format!(
                "`{DSPARK_CONFIDENCE_MIN_ENV}` must be finite and between 0 and 1, found {threshold}"
            )));
        }
        Ok((threshold > 0.0).then_some(threshold))
    }

    fn dspark_hardware_profile_from_env(
        max_draft_tokens: usize,
    ) -> Result<Option<DSparkHardwareProfile>> {
        let draft = env::var_os(DSPARK_DRAFT_PROFILE_US_ENV).filter(|value| !value.is_empty());
        let verify = Self::dspark_csv_f32_from_env(DSPARK_VERIFY_PROFILE_US_ENV)?;
        let (draft, verify_micros) = match (draft, verify) {
            (Some(draft), Some(verify_micros)) => (draft, verify_micros),
            (None, None) => return Ok(None),
            _ => {
                return Err(XrtError::InvalidMetadata(format!(
                    "`{DSPARK_DRAFT_PROFILE_US_ENV}` and `{DSPARK_VERIFY_PROFILE_US_ENV}` must be set together"
                )));
            }
        };
        let draft_text = draft.to_string_lossy();
        let draft_micros = draft_text.parse::<f32>().map_err(|error| {
            XrtError::InvalidMetadata(format!(
                "invalid `{DSPARK_DRAFT_PROFILE_US_ENV}` value `{draft_text}`: {error}"
            ))
        })?;
        if !draft_micros.is_finite() || draft_micros < 0.0 {
            return Err(XrtError::InvalidMetadata(format!(
                "`{DSPARK_DRAFT_PROFILE_US_ENV}` must be finite and non-negative, found {draft_micros}"
            )));
        }
        let required = max_draft_tokens.checked_add(1).ok_or_else(|| {
            XrtError::Shape("DSpark hardware-profile length overflowed".to_string())
        })?;
        if verify_micros.len() < required {
            return Err(XrtError::InvalidMetadata(format!(
                "`{DSPARK_VERIFY_PROFILE_US_ENV}` requires at least {required} entries for prefix lengths 0..={max_draft_tokens}, found {}",
                verify_micros.len()
            )));
        }
        Ok(Some(DSparkHardwareProfile {
            draft_micros,
            verify_micros,
        }))
    }

    fn dspark_confidence_temperatures_from_env(
        max_draft_tokens: usize,
    ) -> Result<Option<Vec<f32>>> {
        let temperatures = Self::dspark_csv_f32_from_env(DSPARK_CONFIDENCE_TEMPERATURES_ENV)?;
        if let Some(values) = temperatures.as_ref() {
            if values.len() < max_draft_tokens {
                return Err(XrtError::InvalidMetadata(format!(
                    "`{DSPARK_CONFIDENCE_TEMPERATURES_ENV}` requires at least {max_draft_tokens} entries, found {}",
                    values.len()
                )));
            }
        }
        Ok(temperatures)
    }

    fn dflash_parallel_projections_enabled() -> bool {
        env::var("XRT_CUDA_DFLASH_PARALLEL_PROJECTIONS").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty()
                && value != "0"
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("off")
        })
    }

    fn dflash_bidirectional_noise_enabled() -> bool {
        env::var("XRT_CUDA_DFLASH_BIDIRECTIONAL_NOISE").is_ok_and(|value| {
            let value = value.trim();
            !value.is_empty()
                && value != "0"
                && !value.eq_ignore_ascii_case("false")
                && !value.eq_ignore_ascii_case("off")
        })
    }

    fn dflash_parallel_projection_supported(matrix: &ResidentDFlashMatrix) -> bool {
        matches!(matrix, ResidentDFlashMatrix::Q8_0(matrix) if matrix.uses_marlin_layout())
    }

    fn trace_dflash_buffer(
        &self,
        label: &str,
        buffer: &CudaF32Buffer,
        elements: usize,
    ) -> Result<()> {
        if env::var_os("XRT_QWEN_DFLASH_TRACE").is_none() {
            return Ok(());
        }
        let values = self.device.download_f32(buffer)?;
        let values = &values[..elements.min(values.len())];
        let first = values.iter().take(8).copied().collect::<Vec<_>>();
        let sum = values.iter().map(|&value| value as f64).sum::<f64>();
        let l2 = values
            .iter()
            .map(|&value| (value as f64) * (value as f64))
            .sum::<f64>()
            .sqrt();
        let max_abs = values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        tracing::warn!(
            target: "xrt_runtime::dflash",
            label,
            elements = values.len(),
            sum,
            l2,
            max_abs,
            first = ?first,
            "DFlash parity trace"
        );
        Ok(())
    }

    fn dflash_q8_prefix_matmul(
        &self,
        matrix: &ResidentDFlashMatrix,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.dflash_q8_prefix_matmul_on_stream(matrix, input, rows, output, None)
    }

    fn dflash_q8_prefix_matmul_on_stream(
        &self,
        matrix: &ResidentDFlashMatrix,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
        stream: Option<&CudaExecutionStream>,
    ) -> Result<()> {
        let profile = env::var_os("XRT_QWEN_DFLASH_PROFILE").is_some();
        if profile {
            self.device.synchronize()?;
        }
        let started = profile.then(std::time::Instant::now);
        let result = match matrix {
            ResidentDFlashMatrix::Q8_0(matrix) => self
                .device
                .dflash_matmul_q8_0_batch16_device_prefix_into_on_stream(
                    matrix, input, rows, output, stream,
                ),
            ResidentDFlashMatrix::F16 {
                weights,
                input_f16,
                rows: weight_rows,
                cols: weight_cols,
            } => {
                if stream.is_some() {
                    return Err(XrtError::Unsupported(
                        "parallel DFlash projections are not admitted for F16 cuBLAS weights"
                            .to_string(),
                    ));
                }
                let mut input_f16 = input_f16.lock().map_err(|_| {
                    XrtError::Runtime("DFlash F16 input scratch lock was poisoned".to_string())
                })?;
                self.device.matmul_f16_resident_device_into(
                    input,
                    rows,
                    *weight_cols,
                    weights,
                    *weight_rows,
                    &mut input_f16,
                    output,
                )
            }
        };
        if profile {
            self.device.synchronize()?;
            tracing::warn!(
                target: "xrt_runtime::dflash",
                weight_rows = matrix.dimensions().0,
                weight_cols = matrix.dimensions().1,
                batch_rows = rows,
                weight_kind = matrix.graph_kind(),
                elapsed_micros = started.expect("profile timer exists").elapsed().as_micros(),
                "DFlash projection profile"
            );
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_dspark_confidence_logit(
        &self,
        confidence: &ResidentDSparkConfidenceWeights,
        hidden: &CudaF32Buffer,
        hidden_row: usize,
        hidden_dim: usize,
        markov_input: &CudaF32Buffer,
        confidence_input: &mut CudaF32Buffer,
        confidence_logit: &mut CudaF32Buffer,
        confidence_logits: &mut CudaF32Buffer,
    ) -> Result<()> {
        self.device.copy_f32_device_subrange(
            hidden,
            hidden_row
                .checked_mul(hidden_dim)
                .ok_or_else(|| XrtError::Shape("DSpark confidence row overflowed".to_string()))?,
            confidence_input,
            0,
            hidden_dim,
        )?;
        self.device.copy_f32_device_subrange(
            markov_input,
            0,
            confidence_input,
            hidden_dim,
            markov_input.len(),
        )?;
        self.dflash_q8_prefix_matmul(
            &confidence.projection,
            confidence_input,
            1,
            confidence_logit,
        )?;
        self.device
            .copy_f32_device_subrange(confidence_logit, 0, confidence_logits, hidden_row, 1)
    }

    pub(super) fn capture_qwen35_dflash_target_layer(
        &self,
        weights: &ResidentQwen35DFlashWeights,
        state: &mut CudaQwen35DFlashState,
        layer_output: &CudaF32Buffer,
        rows: usize,
        target_layer: usize,
    ) -> Result<()> {
        let Some(capture_index) = weights
            .config
            .target_layer_ids
            .iter()
            .position(|&layer| layer == target_layer)
        else {
            return Ok(());
        };
        self.device.dflash_capture_features_device(
            layer_output,
            &mut state.feature_rows,
            rows,
            weights.config.embedding_length,
            weights.config.embedding_length * weights.config.target_layer_ids.len(),
            capture_index,
        )
    }

    pub(super) fn update_qwen35_dflash_target_cache(
        &self,
        weights: &ResidentQwen35DFlashWeights,
        state: &mut CudaQwen35DFlashState,
        start_position: usize,
        rows: usize,
    ) -> Result<()> {
        let max_rows = weights.config.capture_rows();
        if rows == 0 || rows > max_rows {
            return Err(XrtError::Shape(format!(
                "DFlash target-cache update requires 1..={} rows, found {rows}",
                max_rows
            )));
        }
        if state.len > start_position {
            state.truncate(start_position);
        }
        if state.len != start_position {
            return Err(XrtError::Runtime(format!(
                "DFlash target-cache update expected position {}, found {}",
                state.len, start_position
            )));
        }
        let config = &weights.config;
        if start_position == 0 {
            self.trace_dflash_buffer(
                "captured_target_features",
                &state.feature_rows,
                rows * config.embedding_length * config.target_layer_ids.len(),
            )?;
        }
        self.dflash_q8_prefix_matmul(
            &weights.feature_projection,
            &state.feature_rows,
            rows,
            &mut state.projected_rows,
        )?;
        self.device.rmsnorm_device_prefix_into(
            &state.projected_rows,
            weights.hidden_norm.buffer(),
            rows,
            config.embedding_length,
            config.rms_norm_eps,
            &mut state.normed_rows,
        )?;
        if start_position == 0 {
            self.trace_dflash_buffer(
                "fused_target_features",
                &state.normed_rows,
                rows * config.embedding_length,
            )?;
        }
        for (layer, cache) in weights.layers.iter().zip(state.cache.iter_mut()) {
            // Qwen3.6 DFlash has context_kv_layer_norm=false: the fused target
            // feature feeds each layer's context K/V projections directly.
            // `attn_norm` belongs only to that layer's noise-block input.
            let parallel = Self::dflash_parallel_projections_enabled()
                && Self::dflash_parallel_projection_supported(&layer.attn_k)
                && Self::dflash_parallel_projection_supported(&layer.attn_v);
            if parallel {
                let dependency = self.device.record_event()?;
                state.projection_streams[0].wait_for_event(&dependency)?;
                self.dflash_q8_prefix_matmul_on_stream(
                    &layer.attn_v,
                    &state.normed_rows,
                    rows,
                    &mut state.v_update,
                    Some(&state.projection_streams[0]),
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.attn_k,
                    &state.normed_rows,
                    rows,
                    &mut state.k_update,
                )?;
                let completion = state.projection_streams[0].record_event()?;
                self.device.wait_for_event(&completion)?;
            } else {
                self.dflash_q8_prefix_matmul(
                    &layer.attn_k,
                    &state.normed_rows,
                    rows,
                    &mut state.k_update,
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.attn_v,
                    &state.normed_rows,
                    rows,
                    &mut state.v_update,
                )?;
            }
            self.device.dflash_norm_rope_device(
                &mut state.k_update,
                layer.attn_k_norm.buffer(),
                rows,
                config.attention_head_count_kv,
                config.head_dim,
                start_position,
                config.head_dim,
                config.rms_norm_eps,
                config.rope_freq_base,
            )?;
            let kv_width = config.attention_head_count_kv * config.head_dim;
            self.device.dflash_store_ring_rows_device(
                &state.k_update,
                &mut cache.key,
                rows,
                kv_width,
                start_position,
                state.capacity,
            )?;
            self.device.dflash_store_ring_rows_device(
                &state.v_update,
                &mut cache.value,
                rows,
                kv_width,
                start_position,
                state.capacity,
            )?;
        }
        state.len = start_position.saturating_add(rows);
        Ok(())
    }

    pub(super) fn draft_qwen35_dflash_proposal(
        &self,
        weights: &ResidentQwen35DFlashWeights,
        output_weights: &ResidentQ8_0ProbeWeights,
        next_token_id: u32,
        max_draft_tokens: usize,
        tree_budget: Option<usize>,
        state: &mut CudaQwen35DFlashState,
    ) -> Result<MtpDraftProposal> {
        let config = &weights.config;
        if next_token_id as usize >= config.vocab_size {
            return Err(XrtError::Model(format!(
                "DFlash boundary token {next_token_id} exceeds vocabulary size {}",
                config.vocab_size
            )));
        }
        let rows = config.draft_rows;
        let take = max_draft_tokens.min(if config.is_dspark {
            rows
        } else {
            rows.saturating_sub(1)
        });
        let mut token_ids = vec![config.mask_token_id; rows];
        token_ids[0] = next_token_id;
        self.embed_probe_tokens_into(output_weights, &token_ids, &mut state.noise_a)?;
        self.trace_dflash_buffer(
            "noise_embedding",
            &state.noise_a,
            rows * config.embedding_length,
        )?;

        let context_rows = state.len.min(state.capacity);
        let full_context_start = state.len.saturating_sub(context_rows);
        let mut input_is_a = true;
        for (layer_index, (layer, cache)) in
            weights.layers.iter().zip(state.cache.iter()).enumerate()
        {
            let (input, output) = if input_is_a {
                (&state.noise_a, &mut state.noise_b)
            } else {
                (&state.noise_b, &mut state.noise_a)
            };
            self.device.rmsnorm_device_into(
                input,
                layer.attn_norm.buffer(),
                rows,
                config.embedding_length,
                config.rms_norm_eps,
                &mut state.noise_normed,
            )?;
            let parallel_attention = Self::dflash_parallel_projections_enabled()
                && [&layer.attn_q, &layer.attn_k, &layer.attn_v]
                    .into_iter()
                    .all(Self::dflash_parallel_projection_supported);
            if parallel_attention {
                let dependency = self.device.record_event()?;
                for stream in &state.projection_streams {
                    stream.wait_for_event(&dependency)?;
                }
                self.dflash_q8_prefix_matmul_on_stream(
                    &layer.attn_k,
                    &state.noise_normed,
                    rows,
                    &mut state.noise_key,
                    Some(&state.projection_streams[0]),
                )?;
                self.dflash_q8_prefix_matmul_on_stream(
                    &layer.attn_v,
                    &state.noise_normed,
                    rows,
                    &mut state.noise_value,
                    Some(&state.projection_streams[1]),
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.attn_q,
                    &state.noise_normed,
                    rows,
                    &mut state.query,
                )?;
                for stream in &state.projection_streams {
                    let completion = stream.record_event()?;
                    self.device.wait_for_event(&completion)?;
                }
            } else {
                self.dflash_q8_prefix_matmul(
                    &layer.attn_q,
                    &state.noise_normed,
                    rows,
                    &mut state.query,
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.attn_k,
                    &state.noise_normed,
                    rows,
                    &mut state.noise_key,
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.attn_v,
                    &state.noise_normed,
                    rows,
                    &mut state.noise_value,
                )?;
            }
            self.device.dflash_norm_rope_device(
                &mut state.query,
                layer.attn_q_norm.buffer(),
                rows,
                config.attention_head_count,
                config.head_dim,
                state.len,
                config.head_dim,
                config.rms_norm_eps,
                config.rope_freq_base,
            )?;
            self.device.dflash_norm_rope_device(
                &mut state.noise_key,
                layer.attn_k_norm.buffer(),
                rows,
                config.attention_head_count_kv,
                config.head_dim,
                state.len,
                config.head_dim,
                config.rms_norm_eps,
                config.rope_freq_base,
            )?;
            let layer_context_rows = if layer.sliding_window {
                context_rows.min(2_048)
            } else {
                context_rows
            };
            let context_start =
                full_context_start + context_rows.saturating_sub(layer_context_rows);
            let causal_noise = layer.sliding_window && !Self::dflash_bidirectional_noise_enabled();
            self.device.dflash_block_attention_device(
                &state.query,
                &cache.key,
                &cache.value,
                &state.noise_key,
                &state.noise_value,
                &mut state.attention,
                rows,
                config.attention_head_count,
                config.attention_head_count_kv,
                config.head_dim,
                layer_context_rows,
                context_start,
                state.capacity,
                causal_noise,
                1.0 / (config.head_dim as f32).sqrt(),
            )?;
            self.dflash_q8_prefix_matmul(
                &layer.attn_output,
                &state.attention,
                rows,
                &mut state.hidden_temp,
            )?;
            self.device
                .add_device_into(input, &state.hidden_temp, output)?;
            self.trace_dflash_buffer(
                &format!("draft_layer_{}_attention_residual", layer_index),
                output,
                rows * config.embedding_length,
            )?;
            self.device.rmsnorm_device_into(
                output,
                layer.ffn_norm.buffer(),
                rows,
                config.embedding_length,
                config.rms_norm_eps,
                &mut state.noise_normed,
            )?;
            self.trace_dflash_buffer(
                &format!("draft_layer_{}_ffn_input", layer_index),
                &state.noise_normed,
                rows * config.embedding_length,
            )?;
            let parallel_ffn = Self::dflash_parallel_projections_enabled()
                && Self::dflash_parallel_projection_supported(&layer.ffn_gate)
                && Self::dflash_parallel_projection_supported(&layer.ffn_up);
            if parallel_ffn {
                let dependency = self.device.record_event()?;
                state.projection_streams[0].wait_for_event(&dependency)?;
                self.dflash_q8_prefix_matmul_on_stream(
                    &layer.ffn_up,
                    &state.noise_normed,
                    rows,
                    &mut state.ffn_up,
                    Some(&state.projection_streams[0]),
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.ffn_gate,
                    &state.noise_normed,
                    rows,
                    &mut state.ffn_gate,
                )?;
                let completion = state.projection_streams[0].record_event()?;
                self.device.wait_for_event(&completion)?;
            } else {
                self.dflash_q8_prefix_matmul(
                    &layer.ffn_gate,
                    &state.noise_normed,
                    rows,
                    &mut state.ffn_gate,
                )?;
                self.dflash_q8_prefix_matmul(
                    &layer.ffn_up,
                    &state.noise_normed,
                    rows,
                    &mut state.ffn_up,
                )?;
            }
            self.device.silu_assign_device(&mut state.ffn_gate)?;
            self.device
                .mul_assign_device(&mut state.ffn_gate, &state.ffn_up)?;
            self.dflash_q8_prefix_matmul(
                &layer.ffn_down,
                &state.ffn_gate,
                rows,
                &mut state.hidden_temp,
            )?;
            self.device.add_assign_device(output, &state.hidden_temp)?;
            self.trace_dflash_buffer(
                &format!("draft_layer_{}_output", layer_index),
                output,
                rows * config.embedding_length,
            )?;
            input_is_a = !input_is_a;
        }
        let final_hidden = if input_is_a {
            &state.noise_a
        } else {
            &state.noise_b
        };
        self.trace_dflash_buffer(
            "final_draft_hidden",
            final_hidden,
            rows * config.embedding_length,
        )?;
        self.device.rmsnorm_device_into(
            final_hidden,
            weights.output_norm.buffer(),
            rows,
            config.embedding_length,
            config.rms_norm_eps,
            &mut state.noise_normed,
        )?;
        self.trace_dflash_buffer(
            "normalized_draft_hidden",
            &state.noise_normed,
            rows * config.embedding_length,
        )?;
        let profile = env::var_os("XRT_QWEN_DFLASH_PROFILE").is_some();
        if profile {
            self.device.synchronize()?;
        }
        let output_head_started = std::time::Instant::now();
        if let Some(draft_output) = weights.draft_output.as_ref() {
            self.dflash_q8_prefix_matmul(
                draft_output,
                &state.noise_normed,
                rows,
                &mut state.logits,
            )?;
        } else {
            self.matmul_quant_verify_resident_device_into(
                &output_weights.output,
                &state.noise_normed,
                rows,
                &mut state.logits,
            )?;
        }
        if profile {
            self.device.synchronize()?;
            tracing::warn!(
                target: "xrt_runtime::dflash",
                rows,
                target_vocab_size = config.vocab_size,
                draft_vocab_size = config.draft_vocab_size,
                draft_vocab_stride = config.draft_vocab_stride,
                elapsed_micros = output_head_started.elapsed().as_micros(),
                "DFlash output-head profile"
            );
        }
        if let Some(markov) = weights.markov.as_ref() {
            if profile {
                self.device.synchronize()?;
            }
            let markov_started = std::time::Instant::now();
            let confidence_min = Self::dspark_confidence_min_from_env()?;
            let hardware_profile = Self::dspark_hardware_profile_from_env(take)?;
            if confidence_min.is_some() && hardware_profile.is_some() {
                return Err(XrtError::InvalidMetadata(format!(
                    "`{DSPARK_CONFIDENCE_MIN_ENV}` cannot be combined with `{DSPARK_VERIFY_PROFILE_US_ENV}`"
                )));
            }
            let confidence_temperatures = Self::dspark_confidence_temperatures_from_env(take)?;
            let confidence_required = confidence_min.is_some() || hardware_profile.is_some();
            if confidence_temperatures.is_some() && !confidence_required {
                return Err(XrtError::InvalidMetadata(format!(
                    "`{DSPARK_CONFIDENCE_TEMPERATURES_ENV}` requires a confidence scheduler"
                )));
            }
            let confidence = match (confidence_required, weights.confidence.as_ref()) {
                (true, Some(confidence)) => Some(confidence),
                (true, None) => {
                    return Err(XrtError::Unsupported(format!(
                        "DSpark confidence scheduling requires an artifact with `conf_proj.weight` and `conf_proj.bias`"
                    )));
                }
                (false, _) => None,
            };
            let (
                markov_input,
                markov_bias,
                markov_argmax,
                mut markov_draft_argmax,
                confidence_input,
                confidence_logit,
                confidence_logits,
            ) = match (
                state.markov_input.as_mut(),
                state.markov_bias.as_mut(),
                state.markov_argmax.as_mut(),
                state.markov_draft_argmax.as_mut(),
                state.confidence_input.as_mut(),
                state.confidence_logit.as_mut(),
                state.confidence_logits.as_mut(),
            ) {
                (
                    Some(input),
                    Some(bias),
                    Some(argmax),
                    draft_argmax,
                    Some(conf_input),
                    Some(conf_logit),
                    Some(conf_logits),
                ) => (
                    input,
                    bias,
                    argmax,
                    draft_argmax,
                    conf_input,
                    conf_logit,
                    conf_logits,
                ),
                _ => {
                    return Err(XrtError::Runtime(
                        "DSpark state is missing Markov/confidence scratch buffers".to_string(),
                    ));
                }
            };
            self.device
                .upload_f32_into(&[next_token_id as f32], markov_argmax)?;
            for row in 0..take {
                markov.load_embedding_from_device_token(
                    &self.device,
                    markov_argmax,
                    markov_input,
                )?;
                if let Some(confidence) = confidence {
                    self.capture_dspark_confidence_logit(
                        confidence,
                        &state.noise_normed,
                        row,
                        config.embedding_length,
                        markov_input,
                        confidence_input,
                        confidence_logit,
                        confidence_logits,
                    )?;
                }
                self.dflash_q8_prefix_matmul(&markov.projection, markov_input, 1, markov_bias)?;
                let logits_offset =
                    row.checked_mul(config.draft_vocab_stride).ok_or_else(|| {
                        XrtError::Shape("DSpark logits row offset overflowed".to_string())
                    })?;
                self.device.add_assign_device_subrange(
                    &mut state.logits,
                    logits_offset,
                    markov_bias,
                )?;
                if let (Some(mapping), Some(draft_argmax)) = (
                    weights.draft_to_target.as_ref(),
                    markov_draft_argmax.as_deref_mut(),
                ) {
                    self.device.argmax_first_f32_rows_device_subrange_into(
                        &state.logits,
                        logits_offset,
                        1,
                        config.draft_vocab_size,
                        draft_argmax,
                    )?;
                    self.device.lookup_f32_table_device_index_into(
                        mapping,
                        config.draft_vocab_size,
                        draft_argmax,
                        markov_argmax,
                    )?;
                } else {
                    self.device.argmax_first_f32_rows_device_subrange_into(
                        &state.logits,
                        logits_offset,
                        1,
                        config.draft_vocab_size,
                        markov_argmax,
                    )?;
                }
                self.device.copy_f32_device_subrange(
                    markov_argmax,
                    0,
                    &mut state.argmax_indices,
                    row,
                    1,
                )?;
            }
            let mut predictions: Vec<u32> = self
                .device
                .download_argmax_first_f32_rows(&state.argmax_indices, config.vocab_size)?
                .into_iter()
                .take(take)
                .collect();
            let mut confidence_scores = Vec::new();
            if let Some(confidence) = confidence {
                confidence_scores = self
                    .device
                    .download_f32_range(confidence_logits, 0, take)?
                    .into_iter()
                    .enumerate()
                    .map(|(index, logit)| {
                        let temperature = confidence_temperatures
                            .as_ref()
                            .map_or(1.0, |values| values[index]);
                        let calibrated_logit = (logit + confidence.bias) / temperature;
                        1.0 / (1.0 + (-calibrated_logit).exp())
                    })
                    .collect();
                let selected_rows = if let Some(profile) = hardware_profile.as_ref() {
                    profile.select_prefix(&confidence_scores)
                } else {
                    let threshold = confidence_min.expect("confidence threshold is present");
                    confidence_scores
                        .iter()
                        .position(|score| *score < threshold)
                        .unwrap_or(confidence_scores.len())
                };
                predictions.truncate(selected_rows);
            }
            if profile {
                self.device.synchronize()?;
                tracing::warn!(
                    target: "xrt_runtime::dflash",
                    rows = take,
                    selected_rows = predictions.len(),
                    device_token_chain = true,
                    confidence_min,
                    hardware_scheduled = hardware_profile.is_some(),
                    confidence_scores = ?confidence_scores,
                    elapsed_micros = markov_started.elapsed().as_micros(),
                    "DSpark Markov chain profile"
                );
            }
            return Ok(MtpDraftProposal::Linear(predictions));
        }

        let argmax_started = std::time::Instant::now();
        self.device.argmax_first_f32_rows_device_into(
            &state.logits,
            rows,
            config.vocab_size,
            &mut state.argmax_indices,
        )?;
        if profile {
            self.device.synchronize()?;
            tracing::warn!(
                target: "xrt_runtime::dflash",
                rows,
                vocab_size = config.vocab_size,
                elapsed_micros = argmax_started.elapsed().as_micros(),
                "DFlash argmax profile"
            );
        }
        let predictions = self
            .device
            .download_argmax_first_f32_rows(&state.argmax_indices, config.vocab_size)?;
        let top4 = if tree_budget.is_some() || env::var_os(DFLASH_TOP4_DIAGNOSTIC_ENV).is_some() {
            Some(self.device.top4_first_f32_rows_into(
                &state.logits,
                rows,
                config.vocab_size,
                &mut state.top4,
            )?)
        } else {
            None
        };
        if env::var_os(DFLASH_TOP4_DIAGNOSTIC_ENV).is_some() {
            let diagnostic_rows = top4
                .as_ref()
                .expect("top-4 diagnostic requested a top-4 readback")
                .iter()
                .skip(1)
                .take(take)
                .map(|row| {
                    row.iter()
                        .map(|&(token, log_probability)| {
                            serde_json::json!([token, log_probability])
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            eprintln!(
                "XRT_DFLASH_TOP4_DIAGNOSTIC {}",
                serde_json::json!({
                    "context_len": state.len,
                    "rows": diagnostic_rows,
                })
            );
        }
        if let Some(tree_budget) = tree_budget {
            let tree_rows = top4
                .expect("tree proposal requested a top-4 readback")
                .into_iter()
                .skip(1)
                .take(take)
                .collect::<Vec<_>>();
            return build_dflash_draft_tree(&tree_rows, tree_budget).map(MtpDraftProposal::Tree);
        }
        Ok(MtpDraftProposal::Linear(
            predictions.into_iter().skip(1).take(take).collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admitted_target_layer_ids_are_stable() {
        assert_eq!(DFLASH_TARGET_LAYER_IDS, [1, 16, 31, 46, 61]);
    }

    #[test]
    fn draft_families_use_their_declared_block_size_metadata_keys() {
        assert_eq!(
            DFlashDraftConfig::block_size_suffix(false),
            "dflash.block_size"
        );
        assert_eq!(DFlashDraftConfig::block_size_suffix(true), "block_size");
    }

    #[test]
    fn dspark_fixed_depth_executes_only_requested_backbone_rows() {
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(15, true, 1), 1);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(15, true, 6), 6);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(15, true, 15), 15);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(15, true, 99), 15);
    }

    #[test]
    fn original_dflash_fixed_depth_retains_its_anchor_row() {
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(16, false, 1), 2);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(16, false, 6), 7);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(16, false, 15), 16);
        assert_eq!(DFlashDraftConfig::fixed_draft_rows(16, false, 99), 16);
    }

    #[test]
    fn dflash_tree_is_prefix_closed_topological_and_budgeted() {
        let rows = vec![
            vec![(10, -0.10), (11, -0.40), (12, -1.20), (13, -2.0)],
            vec![(20, -0.05), (21, -0.60), (22, -1.40), (23, -2.2)],
            vec![(30, -0.08), (31, -0.80), (32, -1.60), (33, -2.4)],
            vec![(40, -0.12), (41, -0.90), (42, -1.70), (43, -2.5)],
        ];
        let tree = build_dflash_draft_tree(&rows, 7).unwrap();
        assert_eq!(tree.tokens.len(), 7);
        tree.validate().unwrap();
        for (index, &parent) in tree.parents.iter().enumerate() {
            assert!(parent < index + 1);
        }
        assert_eq!(tree.child_row(0, 10), Some(1));
        assert!(tree.depths.iter().copied().max().unwrap() >= 3);
    }

    #[test]
    fn dflash_tree_keeps_alternative_root_hypotheses() {
        let rows = vec![
            vec![(10, -0.30), (11, -0.31), (12, -3.0), (13, -4.0)],
            vec![(20, -0.20), (21, -0.50), (22, -1.0), (23, -2.0)],
            vec![(30, -0.20), (31, -0.50), (32, -1.0), (33, -2.0)],
        ];
        let tree = build_dflash_draft_tree(&rows, 5).unwrap();
        assert_eq!(tree.child_row(0, 10), Some(1));
        assert!(tree.child_row(0, 11).is_some());
    }

    #[test]
    fn dspark_hardware_profile_keeps_profitable_high_confidence_suffix() {
        let profile = DSparkHardwareProfile {
            draft_micros: 1.0,
            verify_micros: vec![1.0, 1.0, 1.0, 1.0],
        };
        assert_eq!(profile.select_prefix(&[1.0, 1.0, 1.0]), 3);
    }

    #[test]
    fn dspark_hardware_profile_prunes_expensive_low_return_suffix() {
        let profile = DSparkHardwareProfile {
            draft_micros: 1.0,
            verify_micros: vec![1.0, 1.0, 10.0],
        };
        assert_eq!(profile.select_prefix(&[0.9, 0.9]), 1);
    }

    #[test]
    fn dspark_hardware_profile_can_choose_target_only_fallback() {
        let profile = DSparkHardwareProfile {
            draft_micros: 1.0,
            verify_micros: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(profile.select_prefix(&[0.0, 0.0]), 0);
    }
}
