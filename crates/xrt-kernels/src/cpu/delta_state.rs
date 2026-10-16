//! Bounded recurrent gated-delta execution with an explicit head-mapping rule.
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadExpansion {
    RepeatInterleave,
    Tile,
}

#[derive(Debug, Clone)]
pub struct GatedDeltaState {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    expansion: HeadExpansion,
    state: Vec<f32>,
}
impl GatedDeltaState {
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        expansion: HeadExpansion,
        budget_bytes: usize,
    ) -> Result<Self> {
        if key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || value_heads % key_heads != 0
        {
            return Err(XrtError::Shape("invalid gated-delta head geometry".into()));
        }
        let count = value_heads
            .checked_mul(key_dim)
            .and_then(|v| v.checked_mul(value_dim))
            .ok_or_else(|| XrtError::Shape("gated-delta state size overflow".into()))?;
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| XrtError::Shape("gated-delta bytes overflow".into()))?;
        if bytes > budget_bytes {
            return Err(XrtError::Runtime(format!(
                "gated-delta state needs {bytes} bytes, budget {budget_bytes}"
            )));
        }
        let mut state = Vec::new();
        state
            .try_reserve_exact(count)
            .map_err(|e| XrtError::Runtime(format!("gated-delta state allocation: {e}")))?;
        state.resize(count, 0.0);
        Ok(Self {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            expansion,
            state,
        })
    }
    pub fn bytes(&self) -> usize {
        self.state.len() * 4
    }
    /// Value-major within each head, matching the shared native delta kernel.
    pub fn snapshot(&self) -> &[f32] {
        &self.state
    }
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
    pub fn step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.step_impl(query, key, value, log_decay, beta, output, false)
    }
    /// Preserve the eager graph's FP32 operation boundaries: decay each state
    /// element before remembering, and scale each query before its state dot.
    /// The default fused kernel remains available for its original contract.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ordered(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        self.step_impl(query, key, value, log_decay, beta, output, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn step_impl(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        output: &mut [f32],
        ordered: bool,
    ) -> Result<()> {
        if query.len() != self.key_heads * self.key_dim
            || key.len() != query.len()
            || value.len() != self.value_heads * self.value_dim
            || output.len() != value.len()
            || log_decay.len() != self.value_heads
            || beta.len() != self.value_heads
        {
            return Err(XrtError::Shape("gated-delta step geometry mismatch".into()));
        }
        if query
            .iter()
            .chain(key)
            .chain(value)
            .chain(log_decay)
            .chain(beta)
            .any(|x| !x.is_finite())
            || log_decay.iter().any(|g| *g > 0.0)
            || beta.iter().any(|b| !(0.0..=1.0).contains(b))
        {
            return Err(XrtError::InvalidTensor("invalid gated-delta inputs".into()));
        }
        let mut normalized_query = query.to_vec();
        let mut normalized_key = key.to_vec();
        for (q, k) in normalized_query
            .chunks_exact_mut(self.key_dim)
            .zip(normalized_key.chunks_exact_mut(self.key_dim))
        {
            super::l2_normalize(q, 1e-6);
            super::l2_normalize(k, 1e-6);
            if ordered {
                let scale = 1.0 / (self.key_dim as f32).sqrt();
                for value in q {
                    *value *= scale;
                }
            }
        }
        let head_state = self.key_dim * self.value_dim;
        for head in 0..self.value_heads {
            let source_head = match self.expansion {
                HeadExpansion::RepeatInterleave => head / (self.value_heads / self.key_heads),
                HeadExpansion::Tile => head % self.key_heads,
            };
            let q = &normalized_query[source_head * self.key_dim..][..self.key_dim];
            let k = &normalized_key[source_head * self.key_dim..][..self.key_dim];
            let v = &value[head * self.value_dim..][..self.value_dim];
            let state = &mut self.state[head * head_state..][..head_state];
            let out = &mut output[head * self.value_dim..][..self.value_dim];
            if ordered {
                let decay = log_decay[head].exp();
                for (vi, row) in state.chunks_exact_mut(self.key_dim).enumerate() {
                    for s in &mut *row {
                        *s *= decay;
                    }
                    let remembered = row.iter().zip(k).map(|(s, k)| s * k).sum::<f32>();
                    let delta = (v[vi] - remembered) * beta[head];
                    for (s, k) in row.iter_mut().zip(k) {
                        *s += k * delta;
                    }
                    out[vi] = row.iter().zip(q).map(|(s, q)| s * q).sum();
                }
                continue;
            }
            // SAFETY: all head slices are dimension-checked and disjoint.
            // The shared kernel owns no pointers beyond this call.
            unsafe {
                super::delta_rule_group(
                    state,
                    k.as_ptr(),
                    q.as_ptr(),
                    v.as_ptr(),
                    out.as_mut_ptr(),
                    self.value_dim,
                    self.key_dim,
                    log_decay[head].exp(),
                    beta[head],
                    1.0 / (self.key_dim as f32).sqrt(),
                );
            }
        }
        if output.iter().chain(&self.state).any(|x| !x.is_finite()) {
            self.reset();
            return Err(XrtError::InvalidTensor(
                "gated-delta state overflow; state cleared".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapping_is_explicit_and_invalid_input_cannot_mutate_state() {
        let mut interleave =
            GatedDeltaState::new(2, 4, 2, 1, HeadExpansion::RepeatInterleave, 32).unwrap();
        let mut tile = GatedDeltaState::new(2, 4, 2, 1, HeadExpansion::Tile, 32).unwrap();
        let q = [1.0, 0.0, 0.0, 1.0];
        let k = [1.0, 0.0, 1.0, 0.0];
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        interleave
            .step(&q, &k, &[1.0; 4], &[0.0; 4], &[1.0; 4], &mut first)
            .unwrap();
        tile.step(&q, &k, &[1.0; 4], &[0.0; 4], &[1.0; 4], &mut second)
            .unwrap();
        assert!(first[1] > 0.7 && first[2] == 0.0);
        assert!(second[1] == 0.0 && second[2] > 0.7);
        let snapshot = interleave.snapshot().to_vec();
        assert!(interleave
            .step(&q, &k, &[1.0; 4], &[0.0; 4], &[2.0; 4], &mut first)
            .is_err());
        assert_eq!(interleave.snapshot(), snapshot);
        interleave.reset();
        assert!(interleave.snapshot().iter().all(|x| *x == 0.0));
    }
    #[test]
    fn state_budget_and_overflow_are_checked_before_allocation() {
        assert!(
            GatedDeltaState::new(16, 32, 128, 128, HeadExpansion::RepeatInterleave, 1024).is_err()
        );
        assert!(GatedDeltaState::new(
            1,
            usize::MAX,
            128,
            128,
            HeadExpansion::RepeatInterleave,
            usize::MAX
        )
        .is_err());
    }
}
