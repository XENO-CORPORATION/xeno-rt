//! Native causal codebook tower; Q/K normalization is always zero-centered,
//! including released configurations whose attention_qk_norm field is false.
use super::checkpoint::Checkpoint;
use crate::sampling::Rng;
use xrt_core::{Result, XrtError};
use xrt_kernels::cpu::softmax_inplace;

#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 30,
            top_p: 0.7,
        }
    }
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || self.top_k == 0
            || !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
        {
            return Err(XrtError::InvalidMetadata(
                "invalid Fish sampling parameters".into(),
            ));
        }
        Ok(())
    }
}
pub fn sample(logits: &[f32], sampling: Sampling, rng: &mut Rng) -> Result<usize> {
    sampling.validate()?;
    if logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || !sampling.temperature.is_finite()
        || sampling.temperature < 0.0
        || sampling.top_k == 0
        || !sampling.top_p.is_finite()
        || !(0.0..=1.0).contains(&sampling.top_p)
    {
        return Err(XrtError::InvalidTensor(
            "invalid Fish sampling parameters/logits".into(),
        ));
    }
    let mut order: Vec<_> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    if sampling.temperature == 0.0 {
        return Ok(order[0]);
    }
    let mut probabilities: Vec<_> = order
        .iter()
        .map(|i| (logits[*i] - logits[order[0]]) / sampling.temperature)
        .collect();
    softmax_inplace(&mut probabilities);
    let mut cumulative = 0.0;
    let mut candidates = Vec::new();
    for (rank, p) in probabilities.iter().enumerate() {
        cumulative += p;
        if rank == 0 || (rank < sampling.top_k && cumulative <= sampling.top_p) {
            candidates.push((order[rank], *p));
        }
    }
    let total = candidates.iter().map(|(_, p)| *p).sum::<f32>();
    let mut best = (0, f64::NEG_INFINITY);
    for (id, p) in candidates {
        let noise = -rng.next_f64().max(f64::MIN_POSITIVE).ln();
        let score = (p / total) as f64 / noise;
        if score > best.1 {
            best = (id, score);
        }
    }
    Ok(best.0)
}

struct Cache {
    keys: Vec<f32>,
    values: Vec<f32>,
}
pub struct TowerSession<'a> {
    store: &'a Checkpoint,
    caches: Vec<Cache>,
    max_rows: usize,
}
impl<'a> TowerSession<'a> {
    pub fn new(store: &'a Checkpoint, max_rows: usize, budget_bytes: u64) -> Result<Self> {
        if store.config.state_bytes(max_rows)? > budget_bytes {
            return Err(XrtError::Runtime(
                "Fish tower state exceeds session budget".into(),
            ));
        }
        let a = &store.config.audio_decoder_config;
        let elements = max_rows
            .checked_mul(a.n_local_heads)
            .and_then(|n| n.checked_mul(a.head_dim))
            .ok_or_else(|| XrtError::Shape("Fish tower cache size overflow".into()))?;
        Ok(Self {
            store,
            caches: (0..a.shared_enrichment_layers + a.num_codebooks - 1)
                .map(|_| {
                    Ok(Cache {
                        keys: super::backbone::reserve_cache(elements)?,
                        values: super::backbone::reserve_cache(elements)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            max_rows,
        })
    }
    pub fn reset(&mut self) {
        for c in &mut self.caches {
            c.keys.clear();
            c.values.clear();
        }
    }
    fn semantic(&self, token: u32) -> bool {
        (self.store.config.semantic_start_token_id..=self.store.config.semantic_end_token_id)
            .contains(&token)
    }
    fn sum_embeddings(&self, frame: &[u32], first: usize, end: usize) -> Result<Vec<f32>> {
        let a = &self.store.config.audio_decoder_config;
        if frame.len() != a.num_codebooks + 1 || end > a.num_codebooks || first > end {
            return Err(XrtError::Shape("Fish frame embedding dimensions".into()));
        }
        let mut out = vec![0.0; a.dim];
        for codebook in first..end {
            if frame[1 + codebook] as usize >= a.vocab_size {
                return Err(XrtError::Shape("Fish codebook id out of bounds".into()));
            }
            let row = self.store.row(
                "audio_decoder.codebook_embeddings.weight",
                codebook * a.vocab_size + frame[1 + codebook] as usize,
            )?;
            if row.len() != out.len() {
                return Err(XrtError::Shape("Fish codebook embedding width".into()));
            }
            for (o, v) in out.iter_mut().zip(row) {
                *o += v;
            }
        }
        self.store.numerics.round_slice(&mut out);
        Ok(out)
    }
    pub fn backbone_embedding(&self, frame: &[u32], text_embedding: &[f32]) -> Result<Vec<f32>> {
        if frame.len() != self.store.config.audio_decoder_config.num_codebooks + 1 {
            return Err(XrtError::Shape("Fish backbone frame width".into()));
        }
        if !self.semantic(frame[0]) {
            return Ok(text_embedding.to_vec());
        }
        let n = self.store.config.audio_decoder_config.vq_backbone_codebooks;
        let extra = self.sum_embeddings(frame, 1, 1 + n)?;
        if text_embedding.len() != extra.len() {
            return Err(XrtError::Shape("Fish backbone embedding width".into()));
        }
        Ok(text_embedding
            .iter()
            .zip(extra)
            .map(|(x, e)| {
                self.store
                    .numerics
                    .round(self.store.numerics.round(x + e) / ((1 + n) as f32).sqrt())
            })
            .collect())
    }
    fn run_shared(
        &mut self,
        hidden: &[f32],
        code: u32,
        semantic: bool,
        position: usize,
        frame: &[u32],
    ) -> Result<Vec<f32>> {
        self.run_shared_traced(hidden, code, semantic, position, frame, &mut |_, _, _| {})
    }
    #[allow(clippy::too_many_arguments)]
    fn run_shared_traced(
        &mut self,
        hidden: &[f32],
        code: u32,
        semantic: bool,
        position: usize,
        frame: &[u32],
        trace: &mut impl FnMut(usize, bool, &[f32]),
    ) -> Result<Vec<f32>> {
        let a = &self.store.config.audio_decoder_config;
        if frame.len() != a.num_codebooks + 1 {
            return Err(XrtError::Shape("Fish shared tower frame width".into()));
        }
        let unseen = if self.semantic(frame[0]) {
            self.sum_embeddings(frame, 1 + a.vq_backbone_codebooks, a.num_codebooks)?
        } else {
            vec![0.0; a.dim]
        };
        let mut hidden = hidden.to_vec();
        for layer in 0..a.shared_enrichment_layers {
            trace(layer, false, &hidden);
            hidden = self.layer(
                &format!("audio_decoder_mtp_shared.layers.{layer}"),
                layer,
                &hidden,
                code,
                semantic,
                position,
                true,
                if layer == 0 { Some(&unseen) } else { None },
            )?;
            trace(layer, true, &hidden);
        }
        Ok(hidden)
    }
    pub fn prefill(
        &mut self,
        hidden: &[f32],
        position: usize,
        frame: &[u32],
        next: &[u32],
    ) -> Result<()> {
        self.prefill_traced(hidden, position, frame, next, |_, _, _| {})
    }
    pub fn prefill_traced(
        &mut self,
        hidden: &[f32],
        position: usize,
        frame: &[u32],
        next: &[u32],
        mut trace: impl FnMut(usize, bool, &[f32]),
    ) -> Result<()> {
        let a = &self.store.config.audio_decoder_config;
        if next.len() != a.num_codebooks + 1 {
            return Err(XrtError::Shape("Fish next frame dimensions".into()));
        }
        let semantic = self.semantic(next[0]);
        let mut hidden =
            self.run_shared_traced(hidden, next[1], semantic, position, frame, &mut trace)?;
        for i in 0..a.num_codebooks - 1 {
            trace(a.shared_enrichment_layers + i, false, &hidden);
            hidden = self.layer(
                &format!("audio_decoder_mtp.layers.{i}"),
                a.shared_enrichment_layers + i,
                &hidden,
                next[i + 1],
                semantic,
                position,
                false,
                None,
            )?;
            trace(a.shared_enrichment_layers + i, true, &hidden);
        }
        Ok(())
    }
    pub fn decode(
        &mut self,
        logits: &[f32],
        hidden: &[f32],
        position: usize,
        frame: &[u32],
        history: &[u32],
        sampling: Sampling,
        rng: &mut Rng,
    ) -> Result<Vec<u32>> {
        let cfg = &self.store.config;
        let a = &cfg.audio_decoder_config;
        if logits.len() != cfg.text_config.vocab_size {
            return Err(XrtError::Shape("Fish main logits vocabulary".into()));
        }
        let mut candidates = vec![cfg.eos_token_id];
        candidates.extend(cfg.semantic_start_token_id..=cfg.semantic_end_token_id);
        candidates.sort_unstable();
        let values: Vec<_> = candidates.iter().map(|i| logits[*i as usize]).collect();
        let first = candidates[sample(&values, sampling, rng)?];
        let retry = candidates[sample(
            &values,
            Sampling {
                temperature: 1.0,
                top_k: 30,
                top_p: 0.95,
            },
            rng,
        )?];
        let main = if self.semantic(first) && history.iter().rev().take(16).any(|i| *i == first) {
            retry
        } else {
            first
        };
        let c0 = main
            .saturating_sub(cfg.semantic_start_token_id)
            .min(a.vocab_size as u32 - 1);
        let semantic = self.semantic(main);
        let mut hidden = self.run_shared(hidden, c0, semantic, position, frame)?;
        let mut codes = vec![main, c0];
        for i in 0..a.num_codebooks - 1 {
            hidden = self.layer(
                &format!("audio_decoder_mtp.layers.{i}"),
                a.shared_enrichment_layers + i,
                &hidden,
                *codes.last().unwrap(),
                semantic,
                position,
                false,
                None,
            )?;
            let logits = self
                .store
                .linear(&format!("audio_decoder_mtp.heads.{i}.weight"), &hidden)?;
            if logits.len() != a.vocab_size {
                return Err(XrtError::Shape("Fish codebook head vocabulary".into()));
            }
            codes.push(sample(&logits, sampling, rng)? as u32);
        }
        Ok(codes)
    }
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        prefix: &str,
        cache_id: usize,
        hidden: &[f32],
        code: u32,
        semantic: bool,
        position: usize,
        shared: bool,
        extra: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let a = &self.store.config.audio_decoder_config;
        if position >= self.max_rows || hidden.len() != a.dim || code as usize >= a.vocab_size {
            return Err(XrtError::Shape("Fish tower input/context bounds".into()));
        }
        let mut embedding = if semantic {
            self.store.row(
                &format!(
                    "{prefix}.{}.weight",
                    if shared {
                        "semantic_embedding"
                    } else {
                        "codebook_embedding"
                    }
                ),
                code as usize,
            )?
        } else {
            vec![0.0; a.dim]
        };
        if let Some(extra) = extra {
            if extra.len() != embedding.len() {
                return Err(XrtError::Shape("Fish shared embedding width".into()));
            }
            for (e, x) in embedding.iter_mut().zip(extra) {
                *e += x;
            }
        }
        self.store.numerics.round_slice(&mut embedding);
        let mut mixed = self.store.norm(
            &embedding,
            &self.store.vector(&format!("{prefix}.enorm.weight"))?,
            a.norm_eps,
            true,
        )?;
        mixed.extend(self.store.norm(
            hidden,
            &self.store.vector(&format!("{prefix}.hnorm.weight"))?,
            a.norm_eps,
            true,
        )?);
        let mixed = self
            .store
            .linear(&format!("{prefix}.eh_proj.weight"), &mixed)?;
        let transformer = format!("{prefix}.transformer_layer");
        let normalized = self.store.norm(
            &mixed,
            &self
                .store
                .vector(&format!("{transformer}.attention_norm.weight"))?,
            a.norm_eps,
            true,
        )?;
        let attention = format!("{transformer}.attention");
        let packed = self
            .store
            .linear(&format!("{attention}.wqkv.weight"), &normalized)?;
        let qw = a.n_head * a.head_dim;
        let kw = a.n_local_heads * a.head_dim;
        if packed.len() != 2 * qw + 2 * kw {
            return Err(XrtError::Shape("Fish tower Q/Gate/K/V packing".into()));
        }
        let mut query = packed[..qw].to_vec();
        let gate = &packed[qw..2 * qw];
        let mut key = packed[2 * qw..2 * qw + kw].to_vec();
        let value = &packed[2 * qw + kw..];
        let qnorm = self.store.vector(&format!("{attention}.q_norm.weight"))?;
        let knorm = self.store.vector(&format!("{attention}.k_norm.weight"))?;
        let rotary = (a.head_dim as f32 * a.rotary_percent) as usize;
        if !self.store.normalize_rotate_qk(
            &mut query,
            &mut key,
            &qnorm,
            &knorm,
            xrt_cuda::Bf16RopeSpec {
                head_dim: a.head_dim,
                rotary_dim: rotary,
                theta: a.rope_base,
                position,
                interleaved: true,
                epsilon: a.norm_eps,
            },
        )? {
            for q in query.chunks_exact_mut(a.head_dim) {
                q.copy_from_slice(
                    &self
                        .store
                        .norm_with_rows(q, &qnorm, a.norm_eps, true, a.n_head)?,
                );
            }
            for k in key.chunks_exact_mut(a.head_dim) {
                k.copy_from_slice(&self.store.norm_with_rows(
                    k,
                    &knorm,
                    a.norm_eps,
                    true,
                    a.n_local_heads,
                )?);
            }
            for x in [&mut query, &mut key] {
                for head in x.chunks_exact_mut(a.head_dim) {
                    for pair in 0..rotary / 2 {
                        let angle =
                            position as f32 / a.rope_base.powf((2 * pair) as f32 / rotary as f32);
                        let (s, c) = angle.sin_cos();
                        let even = head[2 * pair];
                        let odd = head[2 * pair + 1];
                        head[2 * pair] = even * c - odd * s;
                        head[2 * pair + 1] = odd * c + even * s;
                    }
                }
            }
        }
        self.store.numerics.round_slice(&mut query);
        self.store.numerics.round_slice(&mut key);
        let cache = &mut self.caches[cache_id];
        if cache.keys.len() != position * kw {
            return Err(XrtError::Runtime(
                "Fish tower cache position is not contiguous".into(),
            ));
        }
        cache.keys.extend(key);
        cache.values.extend_from_slice(value);
        let mut out = self.store.attention_core(
            &query,
            &cache.keys,
            &cache.values,
            a.n_head,
            a.n_local_heads,
            a.head_dim,
        )?;
        self.store.attention_gate(&mut out, gate)?;
        let attention = self.store.linear(&format!("{attention}.wo.weight"), &out)?;
        if attention.len() != mixed.len() {
            return Err(XrtError::Shape(
                "Fish tower attention residual width".into(),
            ));
        }
        let residual: Vec<_> = mixed
            .iter()
            .zip(attention)
            .map(|(x, a)| self.store.numerics.round(x + a))
            .collect();
        let normalized = self.store.norm(
            &residual,
            &self
                .store
                .vector(&format!("{transformer}.ffn_norm.weight"))?,
            a.norm_eps,
            true,
        )?;
        let p = format!("{transformer}.feed_forward");
        let gate = self.store.linear(&format!("{p}.w1.weight"), &normalized)?;
        let up = self.store.linear(&format!("{p}.w3.weight"), &normalized)?;
        if gate.len() != up.len() {
            return Err(XrtError::Shape("Fish tower MLP dimensions".into()));
        }
        let activated: Vec<_> = if self.store.numerics == super::numerics::Numerics::Fp32Reference {
            gate.iter()
                .zip(up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect()
        } else {
            self.store.mlp_hidden(&gate, &up, false)?
        };
        let ffn = self.store.linear(&format!("{p}.w2.weight"), &activated)?;
        if ffn.len() != residual.len() {
            return Err(XrtError::Shape("Fish tower FFN residual width".into()));
        }
        let residual: Vec<_> = residual
            .iter()
            .zip(ffn)
            .map(|(x, f)| self.store.numerics.round(x + f))
            .collect();
        self.store.norm(
            &residual,
            &self
                .store
                .vector(&format!("{prefix}.final_layernorm.weight"))?,
            a.norm_eps,
            true,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nucleus_rule_excludes_crossing_token_and_is_seeded() {
        let p = Sampling {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.7,
        };
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            let first = sample(&[3.0, 2.0, 1.0], p, &mut a).unwrap();
            assert_eq!(first, sample(&[3.0, 2.0, 1.0], p, &mut b).unwrap());
            assert_eq!(first, 0);
        }
        assert!(sample(&[f32::NAN], p, &mut a).is_err());
    }
}
