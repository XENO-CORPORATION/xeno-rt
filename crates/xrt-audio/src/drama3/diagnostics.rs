//! Explicit, bounded numerical recording for qualification builds only.
//! Production features compile no tensor recorder. Captures may contain user
//! data; enabling one is the caller's explicit diagnostic action.
use xrt_core::{Result, XrtError};

#[derive(serde::Serialize)]
pub struct Operator {
    pub kind: &'static str,
    pub name: String,
    pub input: Vec<f32>,
    pub auxiliary: Vec<f32>,
    pub output: Vec<f32>,
    pub parameters: serde_json::Value,
}
pub(crate) struct Recorder {
    pub operators: Vec<Operator>,
    bytes: usize,
    limit: usize,
}
impl Recorder {
    pub fn new(limit: usize) -> Result<Self> {
        if !(1024..=64 * 1024 * 1024).contains(&limit) {
            return Err(XrtError::Runtime(
                "operator trace limit must be 1 KiB..=64 MiB".into(),
            ));
        }
        Ok(Self {
            operators: Vec::new(),
            bytes: 0,
            limit,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        kind: &'static str,
        name: &str,
        input: &[f32],
        auxiliary: &[f32],
        output: &[f32],
        parameters: serde_json::Value,
    ) -> Result<()> {
        if input
            .iter()
            .chain(auxiliary)
            .chain(output)
            .any(|v| !v.is_finite())
        {
            return Err(XrtError::InvalidTensor(
                "non-finite diagnostic activation".into(),
            ));
        }
        let bytes = input
            .len()
            .checked_add(auxiliary.len())
            .and_then(|v| v.checked_add(output.len()))
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| v.checked_add(name.len() + 512))
            .and_then(|v| v.checked_add(self.bytes))
            .ok_or_else(|| XrtError::Runtime("operator trace size overflow".into()))?;
        if bytes > self.limit || self.operators.len() >= 4096 {
            return Err(XrtError::Runtime(
                "operator trace budget exhausted; capture refused".into(),
            ));
        }
        self.bytes = bytes;
        self.operators.push(Operator {
            kind,
            name: name.into(),
            input: input.to_vec(),
            auxiliary: auxiliary.to_vec(),
            output: output.to_vec(),
            parameters,
        });
        Ok(())
    }
}
