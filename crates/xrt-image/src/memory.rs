use serde::{Deserialize, Serialize};

use crate::{ImageBackendKind, ImageOffloadPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageRequestKind {
    Generate,
    Edit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedImageOutput {
    pub output_index: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageExecutionPlan {
    pub request_kind: ImageRequestKind,
    pub model: String,
    pub bundle_digest: String,
    pub backend: ImageBackendKind,
    pub offload: ImageOffloadPolicy,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub outputs: Vec<PlannedImageOutput>,
    pub estimated_host_bytes: u64,
    pub estimated_device_bytes: u64,
}
