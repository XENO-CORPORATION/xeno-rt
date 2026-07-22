use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageQuality {
    Auto,
    Low,
    Medium,
    High,
    Standard,
    Hd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageBackground {
    Auto,
    Opaque,
    Transparent,
}

/// Values admitted by the pinned synchronous image response schema. Request
/// compatibility additionally accepts `auto`, `standard`, and `hd`, so the
/// request enum must not be reused when serializing responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageResponseQuality {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageResponseBackground {
    Opaque,
    Transparent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiImageResponseSize {
    #[serde(rename = "1024x1024")]
    Square1024,
    #[serde(rename = "1024x1536")]
    Portrait1024x1536,
    #[serde(rename = "1536x1024")]
    Landscape1536x1024,
}

impl OpenAiImageResponseSize {
    pub const fn from_dimensions(width: u32, height: u32) -> Option<Self> {
        match (width, height) {
            (1024, 1024) => Some(Self::Square1024),
            (1024, 1536) => Some(Self::Portrait1024x1536),
            (1536, 1024) => Some(Self::Landscape1536x1024),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageResponseFormat {
    B64Json,
    Url,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiImageInputFidelity {
    Low,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiXenoImageBackend {
    Auto,
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiXenoImageOffload {
    None,
    Sequential,
    Balanced,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiXenoResizePolicy {
    Reject,
    RoundDown,
}

/// XENO-only controls remain under one strict namespace so misspellings do
/// not silently change an expensive image job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiXenoImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_cfg_scale: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offload: Option<OpenAiXenoImageOffload>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<OpenAiXenoImageBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resize_policy: Option<OpenAiXenoResizePolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview_interval_steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_noop: Option<bool>,
}

/// JSON body for `POST /v1/images/generations` at the pinned OpenAI schema
/// revision. Model-specific support is validated after parsing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageGenerationRequest {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiImageResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moderation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_xeno: Option<OpenAiXenoImageOptions>,
}

/// Scalar fields from the standard multipart image-edit request. Uploaded
/// image and mask parts remain ordered byte streams owned by the server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageEditFields {
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fidelity: Option<OpenAiImageInputFidelity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiImageResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moderation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_xeno: Option<OpenAiXenoImageOptions>,
}

/// One image reference in the current JSON image-edit request contract.
/// The server validates that exactly one reference kind is present and applies
/// resolver policy before any bytes reach the image runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageReference {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

/// JSON body for `POST /v1/images/edits` in the current pinned OpenAI server
/// schema. Multipart uploads use [`OpenAiImageEditFields`] plus binary parts;
/// this form uses ordered image references instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageEditJsonRequest {
    pub images: Vec<OpenAiImageReference>,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<OpenAiImageReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_fidelity: Option<OpenAiImageInputFidelity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_images: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moderation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_xeno: Option<OpenAiXenoImageOptions>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiImageData {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiImageTokenDetails {
    pub image_tokens: u64,
    pub text_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiImageUsage {
    pub input_tokens: u64,
    pub input_tokens_details: OpenAiImageTokenDetails,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiImageResponse {
    pub created: u64,
    pub data: Vec<OpenAiImageData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageResponseQuality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<OpenAiImageResponseSize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageResponseBackground>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiImageUsage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAiImageStreamEventType {
    #[serde(rename = "image_generation.partial_image")]
    GenerationPartialImage,
    #[serde(rename = "image_generation.completed")]
    GenerationCompleted,
    #[serde(rename = "image_edit.partial_image")]
    EditPartialImage,
    #[serde(rename = "image_edit.completed")]
    EditCompleted,
}

impl OpenAiImageStreamEventType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GenerationPartialImage => "image_generation.partial_image",
            Self::GenerationCompleted => "image_generation.completed",
            Self::EditPartialImage => "image_edit.partial_image",
            Self::EditCompleted => "image_edit.completed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiImageStreamEvent {
    #[serde(rename = "type")]
    pub event_type: OpenAiImageStreamEventType,
    pub b64_json: String,
    pub created_at: u64,
    pub output_format: OpenAiImageFormat,
    pub quality: OpenAiImageResponseQuality,
    pub size: OpenAiImageResponseSize,
    pub background: OpenAiImageResponseBackground,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_image_index: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAiImageUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiErrorEnvelope {
    pub error: OpenAiErrorBody,
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    const FIXTURE_ROOT: &str = "../../../tests/fixtures/openai/images";

    #[test]
    fn generation_request_matches_pinned_sdk_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai/images/generation-request.json"
        ))
        .unwrap();
        let request: OpenAiImageGenerationRequest =
            serde_json::from_value(fixture["body"].clone()).unwrap();
        assert_eq!(request.n, Some(2));
        assert_eq!(request.quality, Some(OpenAiImageQuality::High));
        assert_eq!(request.output_format, Some(OpenAiImageFormat::Png));
        assert_eq!(
            request.x_xeno.as_ref().and_then(|options| options.seed),
            Some(424_242)
        );
        assert_eq!(
            request.x_xeno.as_ref().and_then(|options| options.backend),
            Some(OpenAiXenoImageBackend::Cuda)
        );
    }

    #[test]
    fn responses_and_stream_events_round_trip_pinned_fixtures() {
        let response_value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai/images/generation-response.json"
        ))
        .unwrap();
        let response: OpenAiImageResponse = serde_json::from_value(response_value.clone()).unwrap();
        assert_eq!(serde_json::to_value(response).unwrap(), response_value);

        let event_value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai/images/edit-stream-events.json"
        ))
        .unwrap();
        let events: Vec<OpenAiImageStreamEvent> =
            serde_json::from_value(event_value.clone()).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(serde_json::to_value(events).unwrap(), event_value);
    }

    #[test]
    fn response_domains_reject_request_only_and_local_values() {
        for (field, value) in [
            ("quality", "auto"),
            ("quality", "standard"),
            ("quality", "hd"),
            ("background", "auto"),
            ("size", "32x32"),
        ] {
            let mut response = serde_json::json!({"created": 1, "data": []});
            response[field] = Value::String(value.to_string());
            assert!(
                serde_json::from_value::<OpenAiImageResponse>(response).is_err(),
                "{field}={value} must not enter the pinned response domain"
            );
        }
    }

    #[test]
    fn json_edit_request_matches_pinned_server_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai/images/edit-request-json.json"
        ))
        .unwrap();
        let request: OpenAiImageEditJsonRequest =
            serde_json::from_value(fixture["body"].clone()).unwrap();
        assert_eq!(request.images.len(), 2);
        assert!(request
            .images
            .iter()
            .all(|reference| reference.image_url.is_some() && reference.file_id.is_none()));
        assert!(request.mask.is_some());
        assert_eq!(request.quality, Some(OpenAiImageQuality::High));
        assert_eq!(request.moderation.as_deref(), Some("auto"));
        assert_eq!(
            request.x_xeno.as_ref().and_then(|options| options.seed),
            Some(515_151)
        );
    }

    #[test]
    fn json_edit_contract_is_closed_and_represents_current_fields() {
        let request: OpenAiImageEditJsonRequest = serde_json::from_value(serde_json::json!({
            "images": [{"file_id": "file_fixture"}],
            "prompt": "fixture",
            "moderation": "auto"
        }))
        .unwrap();
        assert_eq!(request.moderation.as_deref(), Some("auto"));
        assert_eq!(request.images[0].file_id.as_deref(), Some("file_fixture"));

        let error = serde_json::from_value::<OpenAiImageEditJsonRequest>(serde_json::json!({
            "images": [{"image_url": "data:image/png;base64,AA=="}],
            "prompt": "fixture",
            "response_format": "b64_json"
        }))
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `response_format`"));
    }

    #[test]
    fn stream_event_type_exposes_the_raw_sse_event_name() {
        assert_eq!(
            OpenAiImageStreamEventType::GenerationPartialImage.as_str(),
            "image_generation.partial_image"
        );
        assert_eq!(
            OpenAiImageStreamEventType::EditCompleted.as_str(),
            "image_edit.completed"
        );
    }

    #[test]
    fn unknown_xeno_controls_fail_closed() {
        let error = serde_json::from_value::<OpenAiImageGenerationRequest>(serde_json::json!({
            "prompt": "test",
            "x_xeno": { "sead": 42 }
        }))
        .unwrap_err();
        assert!(error.to_string().contains("unknown field `sead`"));
    }

    #[test]
    fn unsupported_parameter_error_matches_fixture() {
        let value: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/openai/images/unsupported-parameter-error.json"
        ))
        .unwrap();
        let envelope: OpenAiErrorEnvelope = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(
            envelope.error.code.as_deref(),
            Some("unsupported_parameter")
        );
        assert_eq!(serde_json::to_value(envelope).unwrap(), value);
    }

    #[test]
    fn fixture_path_constant_stays_relative_to_workspace() {
        assert_eq!(FIXTURE_ROOT, "../../../tests/fixtures/openai/images");
    }
}
