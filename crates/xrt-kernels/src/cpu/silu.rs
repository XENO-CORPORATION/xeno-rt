pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn silu_inplace(values: &mut [f32]) {
    for value in values {
        *value = silu(*value);
    }
}

pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());

    #[cfg(target_arch = "x86_64")]
    {
        if super::simd::has_avx2_fma() {
            unsafe { swiglu_avx2(gate, up) };
            return;
        }
    }

    for (gate, up) in gate.iter_mut().zip(up.iter()) {
        *gate = silu(*gate) * up;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_avx2(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::*;

    let n = gate.len();
    let chunks = n / 8;

    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    for i in 0..chunks {
        let base = i * 8;
        let g = _mm256_loadu_ps(gate.as_ptr().add(base));
        let u = _mm256_loadu_ps(up.as_ptr().add(base));

        // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
        let neg_g = _mm256_mul_ps(g, neg_one);
        let exp_neg_g = super::simd::fast_exp_avx2(neg_g);
        let sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_g));
        let silu = _mm256_mul_ps(g, sigmoid);
        let result = _mm256_mul_ps(silu, u);

        _mm256_storeu_ps(gate.as_mut_ptr().add(base), result);
    }

    // Remainder
    for i in (chunks * 8)..n {
        gate[i] = silu(gate[i]) * up[i];
    }
}
