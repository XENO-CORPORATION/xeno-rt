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
    for (gate, up) in gate.iter_mut().zip(up.iter()) {
        *gate = silu(*gate) * up;
    }
}
