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
}

impl Sampler {
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(rand::random);
        Self {
            rng: StdRng::seed_from_u64(seed),
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
        if logits.is_empty() {
            return Err(XrtError::Runtime(
                "cannot sample from an empty logits vector".to_string(),
            ));
        }

        let mut adjusted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        let seen: HashSet<u32> = history.iter().copied().collect();
        if config.repetition_penalty > 1.0 {
            for (token, logit) in &mut adjusted {
                if seen.contains(&(*token as u32)) {
                    if *logit > 0.0 {
                        *logit /= config.repetition_penalty;
                    } else {
                        *logit *= config.repetition_penalty;
                    }
                }
            }
        }

        if config.temperature <= 1e-5 {
            let best = adjusted
                .iter()
                .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
                .map(|(index, _)| *index)
                .ok_or_else(|| XrtError::Runtime("no logits available after argmax".to_string()))?;
            return Ok(best as u32);
        }

        for (_, logit) in &mut adjusted {
            *logit /= config.temperature;
        }
        adjusted.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));

        if config.top_k > 0 && adjusted.len() > config.top_k {
            adjusted.truncate(config.top_k);
        }

        let max_logit = adjusted
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = adjusted
            .iter()
            .map(|(index, logit)| (*index, (*logit - max_logit).exp()))
            .collect();
        let mut sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
        if sum == 0.0 {
            return Err(XrtError::Runtime(
                "softmax underflow during sampling".to_string(),
            ));
        }
        for (_, prob) in &mut probs {
            *prob /= sum;
        }

        if config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut keep = 0usize;
            for (_, prob) in &probs {
                cumulative += *prob;
                keep += 1;
                if cumulative >= config.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
            sum = probs.iter().map(|(_, prob)| *prob).sum();
            for (_, prob) in &mut probs {
                *prob /= sum;
            }
        }

        let sample = self.rng.random::<f32>();
        let mut cumulative = 0.0f32;
        for (index, prob) in probs {
            cumulative += prob;
            if sample <= cumulative {
                return Ok(index as u32);
            }
        }

        Ok(adjusted[0].0 as u32)
    }
}
