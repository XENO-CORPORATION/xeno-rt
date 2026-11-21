//! Native causal codebook tower; Q/K normalization is always zero-centered,
//! including released configurations whose attention_qk_norm field is false.
use super::checkpoint::Checkpoint;
use crate::sampling::Rng;
use xrt_core::{Result, XrtError};
use xrt_kernels::cpu::softmax_inplace;

/// Codes per residual codebook of the Drama 3 codec (`codebook_size` in upstream
/// `modded_dac_vq.yaml`). The tower's residual heads are wider (4,096, the semantic
/// codebook); reference sampling never draws a residual code the codec cannot hold.
pub const RESIDUAL_CODEBOOK_SIZE: usize = 1024;
/// Reference sampling's main-token epsilon cutoff when a request sets none
/// (upstream `REFERENCE_EPSILON_CUTOFF`).
pub const REFERENCE_EPSILON_CUTOFF: f32 = 1e-3;
const RAS_WINDOW_FRAMES: usize = 16;
const MIN_TEMPERATURE: f32 = 1e-5;

/// What replaces a main token that repeats one of the last 16 main tokens under
/// reference sampling (upstream `ReferenceSamplingFallback`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceFallback {
    /// The most probable candidate. Upstream's default.
    Argmax,
    /// A draw at temperature 1.0, top-p 0.95, top-k 30.
    Ras,
}

/// Which sampler draws each frame.
///
/// `Default` is upstream's original Drama 3 sampler: nucleus sampling, and only a
/// repeated semantic token is redrawn. `Reference` is Fish Audio's reference sampler
/// for Drama 3 (fish-speech `0f8a12e8`, `reference_sample` in `fish_omni/tower.py`):
/// an epsilon cutoff on the main token, every repeated main token replaced (end of
/// speech included), and residual codes kept inside the codec's codebooks. Upstream
/// recommends it with temperature 1.0, top-p 1.0 and top-k 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplerMode {
    Default,
    Reference {
        fallback: ReferenceFallback,
        epsilon_cutoff: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    /// Under reference sampling 0 removes the cap.
    pub top_k: usize,
    pub top_p: f32,
    pub mode: SamplerMode,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 30,
            top_p: 0.7,
            mode: SamplerMode::Default,
        }
    }
}
impl Sampling {
    /// Upstream's recommended reference-sampling configuration.
    pub fn reference() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            mode: SamplerMode::Reference {
                fallback: ReferenceFallback::Argmax,
                epsilon_cutoff: REFERENCE_EPSILON_CUTOFF,
            },
        }
    }
    pub fn validate(&self) -> Result<()> {
        let reference_ok = match self.mode {
            SamplerMode::Default => self.top_k != 0,
            SamplerMode::Reference { epsilon_cutoff, .. } => {
                epsilon_cutoff.is_finite() && (0.0..1.0).contains(&epsilon_cutoff)
            }
        };
        if !self.temperature.is_finite()
            || self.temperature < 0.0
            || !reference_ok
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

/// One id under Fish Audio's reference sampling rules (upstream `reference_sample`).
///
/// Rank r stays when (top_k < 1 or r < top_k) and (top_p >= 1 or cum[r] <= top_p)
/// and p[r] >= epsilon_cutoff; the most probable id (or, with `legal`, the most
/// probable legal id) always stays, and ids at or above `legal` never do. The draw is
/// argmax of p / E with E ~ Exp(1), the same Gumbel-style draw as [`sample`].
pub fn reference_sample(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    epsilon_cutoff: f32,
    legal: Option<usize>,
    rng: &mut Rng,
) -> Result<usize> {
    if logits.is_empty()
        || logits.iter().any(|x| !x.is_finite())
        || !temperature.is_finite()
        || temperature < 0.0
        || !top_p.is_finite()
        || !(0.0..=1.0).contains(&top_p)
        || !epsilon_cutoff.is_finite()
        || !(0.0..1.0).contains(&epsilon_cutoff)
        || legal == Some(0)
    {
        return Err(XrtError::InvalidTensor(
            "invalid Fish reference sampling parameters/logits".into(),
        ));
    }
    let temperature = temperature.max(MIN_TEMPERATURE);
    let mut order: Vec<_> = (0..logits.len()).collect();
    // Stable descending sort, as torch.sort(stable=True): ties keep id order.
    order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b)));
    let mut probabilities: Vec<_> = order
        .iter()
        .map(|i| (logits[*i] - logits[order[0]]) / temperature)
        .collect();
    softmax_inplace(&mut probabilities);
    let mut kept = vec![false; order.len()];
    let mut cumulative = 0.0f32;
    for (rank, p) in probabilities.iter().enumerate() {
        cumulative += p;
        let within_k = top_k == 0 || rank < top_k;
        let within_p = top_p >= 1.0 || cumulative <= top_p;
        kept[rank] = within_k && within_p && *p >= epsilon_cutoff;
    }
    match legal {
        None => kept[0] = true,
        Some(limit) => {
            for (rank, id) in order.iter().enumerate() {
                if *id >= limit {
                    kept[rank] = false;
                }
            }
            let first_legal = order
                .iter()
                .position(|id| *id < limit)
                .ok_or_else(|| XrtError::InvalidTensor("no legal Fish sampling id".into()))?;
            kept[first_legal] = true;
        }
    }
    let total: f32 = probabilities
        .iter()
        .zip(&kept)
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .sum();
    let mut best = (order[0], f64::NEG_INFINITY);
    for ((id, p), k) in order.iter().zip(&probabilities).zip(&kept) {
        if !*k {
            continue;
        }
        let noise = -rng.next_f64().max(f64::MIN_POSITIVE).ln();
        let score = (*p / total) as f64 / noise;
        if score > best.1 {
            best = (*id, score);
        }
    }
    Ok(best.0)
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
        let main = match sampling.mode {
            SamplerMode::Default => {
                let first = candidates[sample(&values, sampling, rng)?];
                let retry = candidates[sample(
                    &values,
                    Sampling {
                        temperature: 1.0,
                        top_k: 30,
                        top_p: 0.95,
                        mode: SamplerMode::Default,
                    },
                    rng,
                )?];
                // Only audio tokens are resampled: the window starts with prompt rows,
                // which hold the user turn's eos, and a stop must not be redrawn.
                if self.semantic(first)
                    && history
                        .iter()
                        .rev()
                        .take(RAS_WINDOW_FRAMES)
                        .any(|i| *i == first)
                {
                    retry
                } else {
                    first
                }
            }
            SamplerMode::Reference {
                fallback,
                epsilon_cutoff,
            } => {
                let first = candidates[reference_sample(
                    &values,
                    sampling.temperature,
                    sampling.top_k,
                    sampling.top_p,
                    epsilon_cutoff,
                    None,
                    rng,
                )?];
                // Neither fallback applies the epsilon cutoff. Both are drawn every
                // frame, as upstream does, so the random stream does not depend on
                // whether a repeat occurred.
                let retry = match fallback {
                    ReferenceFallback::Argmax => {
                        let best = (0..values.len())
                            .max_by(|a, b| values[*a].total_cmp(&values[*b]).then(b.cmp(a)))
                            .ok_or_else(|| XrtError::Shape("empty Fish candidates".into()))?;
                        candidates[best]
                    }
                    ReferenceFallback::Ras => {
                        candidates[reference_sample(&values, 1.0, 30, 0.95, 0.0, None, rng)?]
                    }
                };
                // Any repeated first draw is replaced, eos included.
                if history
                    .iter()
                    .rev()
                    .take(RAS_WINDOW_FRAMES)
                    .any(|i| *i == first)
                {
                    retry
                } else {
                    first
                }
            }
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
            let code = match sampling.mode {
                SamplerMode::Default => sample(&logits, sampling, rng)?,
                // No epsilon cutoff, and only codes the codec's residual codebooks hold.
                SamplerMode::Reference { .. } => reference_sample(
                    &logits,
                    sampling.temperature,
                    sampling.top_k,
                    sampling.top_p,
                    0.0,
                    Some(RESIDUAL_CODEBOOK_SIZE),
                    rng,
                )?,
            };
            codes.push(code as u32);
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
            mode: SamplerMode::Default,
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

    fn draws(
        logits: &[f32],
        eps: f32,
        legal: Option<usize>,
        top_k: usize,
        top_p: f32,
    ) -> Vec<usize> {
        let mut rng = Rng::new(7);
        let mut seen = vec![0usize; logits.len()];
        for _ in 0..4000 {
            let id = reference_sample(logits, 1.0, top_k, top_p, eps, legal, &mut rng).unwrap();
            seen[id] += 1;
        }
        seen
    }

    #[test]
    fn reference_epsilon_cutoff_drops_improbable_ids() {
        // softmax([0, 0, -12]) puts about 3e-6 on the last id, below 1e-3.
        let logits = [0.0, 0.0, -12.0];
        let without = draws(&logits, 0.0, None, 0, 1.0);
        let with = draws(&logits, REFERENCE_EPSILON_CUTOFF, None, 0, 1.0);
        assert!(without[0] > 0 && without[1] > 0);
        assert_eq!(with[2], 0, "epsilon cutoff must remove the 3e-6 id");
        assert!(with[0] > 1500 && with[1] > 1500);
    }

    #[test]
    fn reference_legal_limit_never_returns_out_of_codebook_ids() {
        // The best id is illegal; the most probable legal one must still be kept.
        let mut logits = vec![0.0f32; 8];
        logits[6] = 10.0;
        logits[2] = 5.0;
        let seen = draws(&logits, 0.0, Some(4), 0, 1.0);
        assert!(seen[4..].iter().all(|n| *n == 0), "{seen:?}");
        assert!(seen[2] > 0);
        // Even when every kept rule would drop it, the top legal id survives.
        let seen = draws(&logits, 0.0, Some(4), 1, 0.01);
        assert_eq!(seen[2], 4000, "{seen:?}");
    }

    #[test]
    fn reference_top_k_zero_and_top_p_one_disable_their_rules() {
        let logits = [1.0, 0.9, 0.8, 0.7];
        let seen = draws(&logits, 0.0, None, 0, 1.0);
        assert!(seen.iter().all(|n| *n > 400), "{seen:?}");
        let seen = draws(&logits, 0.0, None, 2, 1.0);
        assert_eq!(seen[2] + seen[3], 0, "{seen:?}");
    }

    #[test]
    fn reference_mode_validation_allows_uncapped_top_k_only_there() {
        assert!(Sampling::reference().validate().is_ok());
        let default_uncapped = Sampling {
            top_k: 0,
            ..Sampling::default()
        };
        assert!(default_uncapped.validate().is_err());
        let bad = Sampling {
            mode: SamplerMode::Reference {
                fallback: ReferenceFallback::Argmax,
                epsilon_cutoff: 1.0,
            },
            ..Sampling::reference()
        };
        assert!(bad.validate().is_err());
    }
}
