use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageTimings {
    pub prompt_encoding_ms: f64,
    pub source_encoding_ms: f64,
    pub denoising_ms: f64,
    pub vae_decode_ms: f64,
    pub encoding_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageBatchTimings {
    pub admission_ms: f64,
    pub queue_ms: f64,
    pub execution_ms: f64,
    pub total_ms: f64,
}
