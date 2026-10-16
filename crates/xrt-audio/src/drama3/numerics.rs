//! Model activation precision is distinct from encoded weight storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Numerics {
    #[default]
    Fp32Reference,
    Bf16W8a8,
}
impl Numerics {
    pub fn round(self, value: f32) -> f32 {
        if self == Self::Bf16W8a8 {
            half::bf16::from_f32(value).to_f32()
        } else {
            value
        }
    }
    pub fn round_slice(self, values: &mut [f32]) {
        if self == Self::Bf16W8a8 {
            for v in values {
                *v = self.round(*v);
            }
        }
    }
    pub fn round_checked(self, values: &mut [f32]) -> xrt_core::Result<()> {
        self.round_slice(values);
        if values.iter().any(|v| !v.is_finite()) {
            return Err(xrt_core::XrtError::InvalidTensor(
                "native activation exceeds the selected precision range".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding_checks_bf16_overflow_after_conversion() {
        let mut finite_f32 = [f32::MAX];
        assert!(Numerics::Fp32Reference
            .round_checked(&mut finite_f32)
            .is_ok());
        assert!(Numerics::Bf16W8a8.round_checked(&mut finite_f32).is_err());
        assert!(Numerics::Fp32Reference
            .round_checked(&mut [f32::NAN])
            .is_err());
    }
}
