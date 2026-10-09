use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

#[cfg(test)]
mod mask_qualification_tests {
    use super::{Sampler, SamplerConfig};

    #[test]
    fn greedy_sampling_refuses_an_empty_mask() {
        let mut sampler = Sampler::new(Some(1));
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(sampler
            .sample_with_mask(&[3.0, 2.0], &[], config, Some(&[false, false]))
            .is_err());
    }

    #[test]
    fn a_short_mask_is_not_an_unrestricted_suffix() {
        let mut sampler = Sampler::new(Some(1));
        let config = SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(sampler
            .sample_with_mask(&[3.0, 9.0], &[], config, Some(&[true]))
            .is_err());
    }

    #[test]
    fn temperature_sampling_refuses_nonfinite_candidates() {
        let mut sampler = Sampler::new(Some(1));
        let config = SamplerConfig::default();
        for logits in [
            [f32::NEG_INFINITY, f32::NEG_INFINITY],
            [f32::NAN, f32::INFINITY],
        ] {
            assert!(sampler
                .sample_with_mask(&logits, &[], config, Some(&[true, true]))
                .is_err());
        }
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

pub struct Sampler {
    rng: StdRng,
    /// Reusable buffer to avoid allocation per sample call
    candidates: Vec<(u32, f32)>,
    /// Reusable set for repetition penalty lookups (O(1) instead of O(n))
    seen_tokens: HashSet<u32>,
}

impl Sampler {
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(rand::random);
        Self {
            rng: StdRng::seed_from_u64(seed),
            candidates: Vec::new(),
            seen_tokens: HashSet::new(),
        }
    }

    pub fn reseed(&mut self, seed: Option<u64>) {
        let seed = seed.unwrap_or_else(rand::random);
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        history: &[u32],
        config: SamplerConfig,
    ) -> Result<u32> {
        self.sample_with_mask(logits, history, config, None)
    }

    /// Sample with an optional grammar token mask.
    /// When `mask` is Some, only tokens where mask[i] == true are considered.
    pub fn sample_with_mask(
        &mut self,
        logits: &[f32],
        history: &[u32],
        config: SamplerConfig,
        mask: Option<&[bool]>,
    ) -> Result<u32> {
        if logits.is_empty() {
            return Err(XrtError::Runtime(
                "cannot sample from an empty logits vector".to_string(),
            ));
        }
        if mask.is_some_and(|mask| mask.len() != logits.len()) {
            return Err(XrtError::Runtime(
                "grammar mask length does not match logits".to_string(),
            ));
        }
        if mask.is_some_and(|mask| !mask.iter().any(|allowed| *allowed)) {
            return Err(XrtError::Runtime("grammar admits no token".to_string()));
        }

        // Greedy (temperature ≈ 0): single pass to find argmax
        if config.temperature <= 1e-5 {
            return self.sample_greedy(logits, history, config.repetition_penalty, mask);
        }

        // Temperature sampling with top-k + top-p
        self.sample_temperature(logits, history, config, mask)
    }

    fn sample_greedy(
        &mut self,
        logits: &[f32],
        history: &[u32],
        rep_penalty: f32,
        mask: Option<&[bool]>,
    ) -> Result<u32> {
        let use_penalty = rep_penalty > 1.0 && !history.is_empty();
        if use_penalty {
            self.seen_tokens.clear();
            self.seen_tokens.extend(history.iter().copied());
        }

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            if let Some(m) = mask {
                if i < m.len() && !m[i] {
                    continue;
                }
            }
            if !logit.is_finite() {
                continue;
            }
            let adjusted = if use_penalty && self.seen_tokens.contains(&(i as u32)) {
                if logit > 0.0 {
                    logit / rep_penalty
                } else {
                    logit * rep_penalty
                }
            } else {
                logit
            };
            if adjusted > best_val {
                best_val = adjusted;
                best_idx = i as u32;
            }
        }

        if !best_val.is_finite() {
            return Err(XrtError::Runtime(
                "no finite logit is allowed by the sampler".to_string(),
            ));
        }
        Ok(best_idx)
    }

    fn sample_temperature(
        &mut self,
        logits: &[f32],
        history: &[u32],
        config: SamplerConfig,
        mask: Option<&[bool]>,
    ) -> Result<u32> {
        let top_k = if config.top_k > 0 {
            config.top_k
        } else {
            logits.len()
        };
        let inv_temp = 1.0 / config.temperature;

        let use_penalty = config.repetition_penalty > 1.0 && !history.is_empty();
        if use_penalty {
            self.seen_tokens.clear();
            self.seen_tokens.extend(history.iter().copied());
        }

        // Step 1: Find top-k candidates in O(n) using a min-heap of size k.
        // For small k (40) and large n (152K), this is much faster than sorting.
        self.candidates.clear();

        for (i, &logit) in logits.iter().enumerate() {
            if let Some(m) = mask {
                if i < m.len() && !m[i] {
                    continue;
                }
            }
            if !logit.is_finite() {
                continue;
            }
            let adjusted = if use_penalty && self.seen_tokens.contains(&(i as u32)) {
                if logit > 0.0 {
                    logit / config.repetition_penalty
                } else {
                    logit * config.repetition_penalty
                }
            } else {
                logit
            };

            if self.candidates.len() < top_k {
                self.candidates.push((i as u32, adjusted));
                if self.candidates.len() == top_k {
                    // Build min-heap: smallest element at [0]
                    self.heapify_min();
                }
            } else if adjusted > self.candidates[0].1 {
                // Replace the smallest element in the heap
                self.candidates[0] = (i as u32, adjusted);
                self.sift_down_min(0);
            }
        }

        if self.candidates.is_empty() {
            return Err(XrtError::Runtime("no candidates after top-k".to_string()));
        }

        // Sort candidates by logit descending (only k elements, k is small)
        self.candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Step 2: Apply temperature and compute softmax
        let max_logit = self.candidates[0].1;
        let mut sum = 0.0f32;
        for (_, logit) in &mut self.candidates {
            let scaled = (*logit - max_logit) * inv_temp;
            let prob = scaled.exp();
            *logit = prob;
            sum += prob;
        }

        if !sum.is_finite() || sum <= 0.0 {
            return Err(XrtError::Runtime("softmax underflow".to_string()));
        }

        let inv_sum = 1.0 / sum;
        for (_, prob) in &mut self.candidates {
            *prob *= inv_sum;
        }

        // Step 3: Apply top-p (nucleus sampling)
        let mut keep = self.candidates.len();
        if config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            for (i, &(_, prob)) in self.candidates.iter().enumerate() {
                cumulative += prob;
                if cumulative >= config.top_p {
                    keep = (i + 1).max(1);
                    break;
                }
            }
            // Renormalize
            let sub_sum: f32 = self.candidates[..keep].iter().map(|(_, p)| p).sum();
            let inv_sub = 1.0 / sub_sum;
            for (_, prob) in &mut self.candidates[..keep] {
                *prob *= inv_sub;
            }
        }

        // Step 4: Sample from the distribution
        let r = self.rng.random::<f32>();
        let mut cumulative = 0.0f32;
        for &(idx, prob) in &self.candidates[..keep] {
            cumulative += prob;
            if r <= cumulative {
                return Ok(idx);
            }
        }

        Ok(self.candidates[0].0)
    }

    /// Build a min-heap on self.candidates (smallest logit at index 0).
    fn heapify_min(&mut self) {
        let n = self.candidates.len();
        for i in (0..n / 2).rev() {
            self.sift_down_min(i);
        }
    }

    /// Sift down element at `idx` in a min-heap.
    fn sift_down_min(&mut self, mut idx: usize) {
        let n = self.candidates.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < n && self.candidates[left].1 < self.candidates[smallest].1 {
                smallest = left;
            }
            if right < n && self.candidates[right].1 < self.candidates[smallest].1 {
                smallest = right;
            }

            if smallest == idx {
                break;
            }
            self.candidates.swap(idx, smallest);
            idx = smallest;
        }
    }
}
