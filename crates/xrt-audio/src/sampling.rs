//! Speech-token sampling, in the reference order:
//! CFG → repetition penalty → temperature → min-p → top-p → multinomial.
//!
//! Greedy decoding is what made early spikes metronomic; sampling at the
//! reference temperature is part of the delivery, not noise.

/// Deterministic, seedable RNG (SplitMix64). A request with a seed must be
/// reproducible on the same backend.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub cfg_weight: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
}

/// Classifier-free guidance: `cond + w * (cond - uncond)`.
pub fn apply_cfg(cond: &[f32], uncond: &[f32], w: f32) -> Vec<f32> {
    cond.iter()
        .zip(uncond)
        .map(|(c, u)| c + w * (c - u))
        .collect()
}

/// HF `RepetitionPenaltyLogitsProcessor`: each distinct previously generated
/// token is penalised once — negative logits multiplied, positive divided.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[i64], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in history {
        if t >= 0 && (t as usize) < logits.len() && seen.insert(t) {
            let s = &mut logits[t as usize];
            *s = if *s < 0.0 { *s * penalty } else { *s / penalty };
        }
    }
}

/// Sample one token. Returns the argmax when `temperature <= 0`.
pub fn sample(mut logits: Vec<f32>, history: &[i64], p: &SamplingParams, rng: &mut Rng) -> i64 {
    apply_repetition_penalty(&mut logits, history, p.repetition_penalty);
    if p.temperature <= 0.0 {
        return argmax(&logits) as i64;
    }
    for l in logits.iter_mut() {
        *l /= p.temperature;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits.iter().map(|&l| ((l - max) as f64).exp()).collect();
    let sum: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|x| *x /= sum);

    // min-p: keep tokens with prob >= min_p * top prob.
    let top = probs.iter().copied().fold(0.0f64, f64::max);
    let floor = p.min_p as f64 * top;
    let mut idx: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] >= floor).collect();
    idx.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // top-p (after min-p renormalisation, as the HF warpers chain).
    if p.top_p < 1.0 {
        let kept: f64 = idx.iter().map(|&i| probs[i]).sum();
        let mut acc = 0.0;
        let mut cut = idx.len();
        for (n, &i) in idx.iter().enumerate() {
            acc += probs[i] / kept;
            if acc >= p.top_p as f64 {
                cut = n + 1;
                break;
            }
        }
        idx.truncate(cut.max(1));
    }

    let total: f64 = idx.iter().map(|&i| probs[i]).sum();
    let r = rng.next_f64() * total;
    let mut acc = 0.0;
    for &i in &idx {
        acc += probs[i];
        if acc >= r {
            return i as i64;
        }
    }
    *idx.last().expect("min-p keeps at least the top token") as i64
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SamplingParams {
        SamplingParams {
            cfg_weight: 0.5,
            temperature: 0.8,
            repetition_penalty: 1.2,
            min_p: 0.05,
            top_p: 1.0,
        }
    }

    #[test]
    fn cfg_matches_formula() {
        assert_eq!(apply_cfg(&[2.0, -1.0], &[1.0, 1.0], 0.5), vec![2.5, -2.0]);
    }

    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        let mut l = vec![2.4, -1.0, 3.0];
        apply_repetition_penalty(&mut l, &[0, 0, 1], 1.2);
        assert!(
            (l[0] - 2.0).abs() < 1e-6,
            "positive divided once, not twice"
        );
        assert!((l[1] + 1.2).abs() < 1e-6, "negative multiplied");
        assert_eq!(l[2], 3.0);
    }

    #[test]
    fn min_p_excludes_tail() {
        // token 1 is ~0.0000454x the top: always below min_p 0.05.
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            assert_eq!(sample(vec![10.0, 0.0], &[], &params(), &mut rng), 0);
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let l: Vec<f32> = (0..50).map(|i| (i as f32 * 0.37).sin()).collect();
        let run = |seed| {
            let mut rng = Rng::new(seed);
            (0..20)
                .map(|_| sample(l.clone(), &[], &params(), &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let p = SamplingParams {
            temperature: 0.0,
            ..params()
        };
        assert_eq!(sample(vec![0.1, 0.9, 0.3], &[], &p, &mut Rng::new(1)), 1);
    }
}
