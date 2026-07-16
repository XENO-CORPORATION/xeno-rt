//! LoRA (Low-Rank Adaptation) support for applying lightweight adapters to base models.
//!
//! LoRA adapters are stored as GGUF files containing low-rank matrices A and B
//! for each adapted layer. During inference: output = W*x + scale * B(A*x)

use std::collections::HashMap;
use std::sync::Arc;
use xrt_core::{DType, Result, XrtError};
use xrt_gguf::GgufFile;
use xrt_kernels::cpu::matvec_quantized;

/// A loaded LoRA adapter with pre-resolved weight pairs.
pub struct LoraAdapter {
    gguf: Arc<GgufFile>,
    /// Map from base tensor name → (A weight, B weight) with pre-resolved offsets.
    weights: HashMap<String, LoraWeightPair>,
    /// Global scaling factor (alpha / rank).
    scale: f32,
}

/// A pair of LoRA matrices (A: rank×in_dim, B: out_dim×rank).
#[derive(Debug, Clone, Copy)]
struct LoraWeightPair {
    a_offset: usize,
    a_nbytes: usize,
    a_rows: usize,
    a_cols: usize,
    a_dtype: DType,
    b_offset: usize,
    b_nbytes: usize,
    b_rows: usize,
    b_cols: usize,
    b_dtype: DType,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a GGUF file.
    ///
    /// LoRA GGUF tensor naming convention:
    /// - `blk.{i}.attn_q.weight.lora_a` / `blk.{i}.attn_q.weight.lora_b`
    /// - Or: `blk.{i}.attn_q.lora_a` / `blk.{i}.attn_q.lora_b`
    pub fn load(path: &str) -> Result<Self> {
        let gguf = Arc::new(GgufFile::open(path)?);

        // Read LoRA alpha from metadata (default: rank value)
        let alpha = gguf.metadata_f32("adapter.lora.alpha");

        // Discover all LoRA weight pairs by scanning tensor names
        let mut pairs: HashMap<String, (Option<usize>, Option<usize>)> = HashMap::new();

        for name in gguf.tensor_names() {
            let base_name = if let Some(base) = name.strip_suffix(".lora_a") {
                base.to_string()
            } else if let Some(base) = name.strip_suffix(".lora_b") {
                base.to_string()
            } else if let Some(base) = name.strip_suffix(".loraA") {
                base.to_string()
            } else if let Some(base) = name.strip_suffix(".loraB") {
                base.to_string()
            } else {
                continue;
            };

            let entry = pairs.entry(base_name).or_default();
            if name.ends_with("lora_a") || name.ends_with("loraA") {
                let idx = gguf.tensor_infos().iter().position(|t| t.name == name);
                entry.0 = idx;
            } else {
                let idx = gguf.tensor_infos().iter().position(|t| t.name == name);
                entry.1 = idx;
            }
        }

        let mut weights = HashMap::new();
        let mut rank = None;

        for (base_name, (a_idx, b_idx)) in &pairs {
            let (Some(a_idx), Some(b_idx)) = (a_idx, b_idx) else {
                continue; // Incomplete pair
            };

            let a_info = &gguf.tensor_infos()[*a_idx];
            let b_info = &gguf.tensor_infos()[*b_idx];

            if rank.is_none() {
                rank = Some(a_info.rows()); // rank = rows of A matrix
            }

            weights.insert(
                base_name.clone(),
                LoraWeightPair {
                    a_offset: a_info.offset,
                    a_nbytes: a_info.nbytes,
                    a_rows: a_info.rows(),
                    a_cols: a_info.row_len(),
                    a_dtype: a_info.dtype,
                    b_offset: b_info.offset,
                    b_nbytes: b_info.nbytes,
                    b_rows: b_info.rows(),
                    b_cols: b_info.row_len(),
                    b_dtype: b_info.dtype,
                },
            );
        }

        let r = rank.unwrap_or(16) as f32;
        let scale = alpha.unwrap_or(r) / r;

        tracing::info!(
            "loaded LoRA adapter with {} weight pairs, rank={}, scale={:.3}",
            weights.len(),
            rank.unwrap_or(16),
            scale
        );

        Ok(Self {
            gguf,
            weights,
            scale,
        })
    }

    /// Check if this adapter has a LoRA pair for the given base weight name.
    pub fn has_weight(&self, base_name: &str) -> bool {
        self.weights.contains_key(base_name)
    }

    /// Apply LoRA delta: output += scale * B(A*input).
    /// `base_name` is the original weight tensor name (e.g., "blk.0.attn_q.weight").
    pub fn apply(&self, base_name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        let pair = self
            .weights
            .get(base_name)
            .ok_or_else(|| XrtError::Runtime(format!("no LoRA weights for {base_name}")))?;

        // Step 1: intermediate = A * input (rank-dimensional)
        let mut intermediate = vec![0.0f32; pair.a_rows];
        let a_bytes = self.gguf.tensor_data_raw(pair.a_offset, pair.a_nbytes);
        matvec_quantized(
            a_bytes,
            pair.a_rows,
            pair.a_cols,
            pair.a_dtype,
            input,
            &mut intermediate,
        )?;

        // Step 2: delta = B * intermediate (output-dimensional)
        let mut delta = vec![0.0f32; pair.b_rows];
        let b_bytes = self.gguf.tensor_data_raw(pair.b_offset, pair.b_nbytes);
        matvec_quantized(
            b_bytes,
            pair.b_rows,
            pair.b_cols,
            pair.b_dtype,
            &intermediate,
            &mut delta,
        )?;

        // Step 3: output += scale * delta
        let scale = self.scale;
        for (o, d) in output.iter_mut().zip(delta.iter()) {
            *o += scale * d;
        }

        Ok(())
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }
}
