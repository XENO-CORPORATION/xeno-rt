use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
        }
    }
}

pub struct Sampler {
    rng: StdRng,
    /// Reusable buffer to avoid allocation per sample call
    candidates: Vec<(u32, f32)>,
    /// Reusable history counts for repetition, presence, and frequency penalties.
    token_counts: HashMap<u32, u32>,
}

impl Sampler {
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(rand::random);
        Self {
            rng: StdRng::seed_from_u64(seed),
            candidates: Vec::new(),
            token_counts: HashMap::new(),
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

        // Greedy (temperature ≈ 0): single pass to find argmax
        if config.temperature <= 1e-5 {
            return self.sample_greedy(logits, history, config, mask);
        }

        // Temperature sampling with top-k + top-p
        self.sample_temperature(logits, history, config, mask)
    }

    fn sample_greedy(
        &mut self,
        logits: &[f32],
        history: &[u32],
        config: SamplerConfig,
        mask: Option<&[bool]>,
    ) -> Result<u32> {
        let use_penalty = Self::penalties_enabled(config) && !history.is_empty();
        if use_penalty {
            self.count_history(history);
        }

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            if let Some(m) = mask {
                if i < m.len() && !m[i] {
                    continue;
                }
            }
            let adjusted = if use_penalty {
                Self::apply_penalties(logit, self.token_counts.get(&(i as u32)).copied(), config)
            } else {
                logit
            };
            if adjusted > best_val {
                best_val = adjusted;
                best_idx = i as u32;
            }
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

        let use_penalty = Self::penalties_enabled(config) && !history.is_empty();
        if use_penalty {
            self.count_history(history);
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
            let adjusted = if use_penalty {
                Self::apply_penalties(logit, self.token_counts.get(&(i as u32)).copied(), config)
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

        if sum == 0.0 {
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

    fn penalties_enabled(config: SamplerConfig) -> bool {
        config.repetition_penalty > 1.0
            || config.presence_penalty != 0.0
            || config.frequency_penalty != 0.0
    }

    fn count_history(&mut self, history: &[u32]) {
        self.token_counts.clear();
        for &token in history {
            *self.token_counts.entry(token).or_insert(0) += 1;
        }
    }

    fn apply_penalties(logit: f32, count: Option<u32>, config: SamplerConfig) -> f32 {
        let Some(count) = count else {
            return logit;
        };
        let mut adjusted = if config.repetition_penalty > 1.0 {
            if logit > 0.0 {
                logit / config.repetition_penalty
            } else {
                logit * config.repetition_penalty
            }
        } else {
            logit
        };
        adjusted -= config.presence_penalty;
        adjusted -= config.frequency_penalty * count as f32;
        adjusted
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_presence_penalty_discourages_a_seen_token_once() {
        let mut sampler = Sampler::new(Some(1));
        let token = sampler
            .sample(
                &[2.0, 1.5],
                &[0, 0, 0],
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.0,
                    presence_penalty: 1.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn greedy_frequency_penalty_scales_with_occurrence_count() {
        let mut sampler = Sampler::new(Some(1));
        let token = sampler
            .sample(
                &[3.0, 2.0],
                &[0, 0, 0],
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.0,
                    frequency_penalty: 0.5,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn negative_presence_penalty_can_promote_a_seen_token() {
        let mut sampler = Sampler::new(Some(1));
        let token = sampler
            .sample(
                &[1.0, 1.5],
                &[0],
                SamplerConfig {
                    temperature: 0.0,
                    repetition_penalty: 1.0,
                    presence_penalty: -1.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(token, 0);
    }

    #[test]
    fn temperature_path_applies_presence_penalty_before_top_k() {
        let mut sampler = Sampler::new(Some(1));
        let token = sampler
            .sample(
                &[2.0, 1.5],
                &[0],
                SamplerConfig {
                    temperature: 1.0,
                    top_k: 1,
                    repetition_penalty: 1.0,
                    presence_penalty: 1.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(token, 1);
    }
}
