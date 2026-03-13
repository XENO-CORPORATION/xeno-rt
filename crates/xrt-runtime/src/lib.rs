pub mod kv_cache;
pub mod sampler;
pub mod session;

use std::{path::Path, sync::Arc};
use xrt_core::Result;
use xrt_gguf::GgufFile;
use xrt_models::LlamaModel;
use xrt_tokenizer::Tokenizer;

pub use kv_cache::PagedKvCache;
pub use sampler::{Sampler, SamplerConfig};
pub use session::{GenerateRequest, Session};

pub struct Runtime {
    model: Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
}

impl Runtime {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let gguf = Arc::new(GgufFile::open(model_path)?);
        Self::from_gguf(gguf)
    }

    pub fn from_gguf(gguf: Arc<GgufFile>) -> Result<Arc<Self>> {
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model = Arc::new(LlamaModel::from_gguf(gguf)?);
        Ok(Arc::new(Self { model, tokenizer }))
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

    pub fn new_session(self: &Arc<Self>) -> Session {
        Session::new(self.clone())
    }
}
