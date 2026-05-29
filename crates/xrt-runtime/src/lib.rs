pub mod grammar;
pub mod kv_cache;
pub mod policy;
pub mod sampler;
pub mod session;

use std::{path::Path, sync::Arc};
use xrt_core::Result;
use xrt_gguf::GgufFile;
use xrt_models::{LlamaModel, VisionEncoder};
use xrt_tokenizer::Tokenizer;

pub use grammar::Grammar;
pub use kv_cache::{
    KeyQ4ValueQ8PagedKvCache, KvCacheMode, PagedKvCache, QuantizedPagedKvCache, SessionKvCache,
};
pub use policy::{CachePolicyKind, PromptSpan, PromptSpanKind, SessionPolicy};
pub use sampler::{Sampler, SamplerConfig};
pub use session::{GenerateRequest, Session};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionPromptLayout {
    pub patch_token_piece: String,
    pub patch_token_id: u32,
    pub start_token_piece: Option<String>,
    pub start_token_id: Option<u32>,
    pub end_token_piece: Option<String>,
    pub end_token_id: Option<u32>,
    pub patches_per_image: usize,
}

impl VisionPromptLayout {
    pub fn prompt_fragment(&self) -> String {
        let mut fragment = String::new();
        if let Some(start) = &self.start_token_piece {
            fragment.push_str(start);
        }
        for _ in 0..self.patches_per_image {
            fragment.push_str(&self.patch_token_piece);
        }
        if let Some(end) = &self.end_token_piece {
            fragment.push_str(end);
        }
        fragment
    }
}

pub struct Runtime {
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    vision: Option<Arc<VisionEncoder>>,
}

impl Runtime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let gguf = Arc::new(GgufFile::open(model_path)?);
        Self::from_gguf(gguf)
    }

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Arc<Self>> {
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model = Arc::new(LlamaModel::from_gguf(gguf)?);
        Ok(Arc::new(Self {
            model,
            tokenizer,
            vision: None,
        }))
    }

    /// Load a multimodal projection (mmproj) GGUF for vision support.
    pub fn load_vision(self: &Arc<Self>, mmproj_path: &str) -> Result<Arc<Self>> {
        let encoder = VisionEncoder::load(mmproj_path)?;
        Ok(Arc::new(Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            vision: Some(Arc::new(encoder)),
        }))
    }

    pub fn model(&self) -> &LlamaModel {
        self.model.as_ref()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        self.tokenizer.as_ref()
    }

    pub fn model_name(&self) -> &str {
        self.model.model_name()
    }

    pub fn model_architecture(&self) -> &str {
        &self.model.config().architecture
    }

    pub fn vision(&self) -> Option<&VisionEncoder> {
        self.vision.as_deref()
    }

    pub fn vision_prompt_layout(&self) -> Option<VisionPromptLayout> {
        let vision = self.vision()?;
        let tokenizer = self.tokenizer();

        for (patch_piece, start_piece, end_piece) in [
            (
                "<|image_pad|>",
                Some("<|vision_start|>"),
                Some("<|vision_end|>"),
            ),
            ("<image>", None, None),
            ("<|image|>", None, None),
            ("<image_pad>", None, None),
        ] {
            let patch_token_id = match tokenizer.token_id_for_piece(patch_piece) {
                Some(id) => id,
                None => continue,
            };
            let start_token_id = start_piece.and_then(|piece| tokenizer.token_id_for_piece(piece));
            let end_token_id = end_piece.and_then(|piece| tokenizer.token_id_for_piece(piece));
            let use_wrappers = start_piece.is_some()
                && end_piece.is_some()
                && start_token_id.is_some()
                && end_token_id.is_some();

            return Some(VisionPromptLayout {
                patch_token_piece: patch_piece.to_string(),
                patch_token_id,
                start_token_piece: use_wrappers.then(|| start_piece.unwrap().to_string()),
                start_token_id: use_wrappers.then_some(start_token_id.unwrap()),
                end_token_piece: use_wrappers.then(|| end_piece.unwrap().to_string()),
                end_token_id: use_wrappers.then_some(end_token_id.unwrap()),
                patches_per_image: vision.config().patch_count,
            });
        }

        None
    }

    pub fn new_session(self: &Arc<Self>) -> Session {
        Session::new(self.clone())
    }

    pub fn new_session_with_cache_mode(self: &Arc<Self>, mode: KvCacheMode) -> Session {
        Session::new_with_cache_mode(self.clone(), mode)
    }
}
