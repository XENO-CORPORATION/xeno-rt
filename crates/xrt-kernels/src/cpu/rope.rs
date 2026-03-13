/// Precomputed frequency table for rotary position embeddings.
/// Caches inv_freq values so we only compute powf once at model load,
/// then just sin_cos per position (which is unavoidable).
pub struct RopeFreqs {
    inv_freq: Vec<f32>,
    scale: f32,
}

impl RopeFreqs {
    pub fn new(rope_dim: usize, base: f32, scale: f32) -> Self {
        let half = rope_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|pair| base.powf((2.0 * pair as f32) / rope_dim as f32))
            .collect();
        Self { inv_freq, scale }
    }

    pub fn apply(&self, tensor: &mut [f32], n_heads: usize, head_dim: usize, position: usize) {
        assert_eq!(tensor.len(), n_heads * head_dim);
        let half_width = self.inv_freq.len().min(head_dim / 2);

        for head in 0..n_heads {
            let head_slice = &mut tensor[head * head_dim..(head + 1) * head_dim];
            for pair_index in 0..half_width {
                let angle = position as f32 * self.scale / self.inv_freq[pair_index];
                let (sin, cos) = angle.sin_cos();
                let lhs = head_slice[pair_index];
                let rhs = head_slice[pair_index + half_width];
                head_slice[pair_index] = lhs * cos - rhs * sin;
                head_slice[pair_index + half_width] = lhs * sin + rhs * cos;
            }
        }
    }

    pub fn apply_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        position: usize,
    ) {
        self.apply(q, q_heads, head_dim, position);
        self.apply(k, kv_heads, head_dim, position);
    }
}

pub fn apply_rotary(
    tensor: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    position: usize,
    rope_dim: usize,
    base: f32,
    scale: f32,
) {
    assert_eq!(tensor.len(), n_heads * head_dim);
    let rotary_width = rope_dim.min(head_dim);
    let half_width = rotary_width / 2;

    for head in 0..n_heads {
        let head_slice = &mut tensor[head * head_dim..(head + 1) * head_dim];
        for pair_index in 0..half_width {
            let theta = base.powf((2.0 * pair_index as f32) / rotary_width as f32);
            let angle = position as f32 * scale / theta;
            let (sin, cos) = angle.sin_cos();
            let lhs = head_slice[pair_index];
            let rhs = head_slice[pair_index + half_width];
            head_slice[pair_index] = lhs * cos - rhs * sin;
            head_slice[pair_index + half_width] = lhs * sin + rhs * cos;
        }
    }
}

pub fn apply_rotary_qk(
    q: &mut [f32],
    k: &mut [f32],
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    position: usize,
    rope_dim: usize,
    base: f32,
    scale: f32,
) {
    apply_rotary(q, q_heads, head_dim, position, rope_dim, base, scale);
    apply_rotary(k, kv_heads, head_dim, position, rope_dim, base, scale);
}
