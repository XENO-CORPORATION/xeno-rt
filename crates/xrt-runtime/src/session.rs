use crate::{PagedKvCache, Runtime, Sampler, SamplerConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use xrt_core::{KvCache, Result, XrtError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 128,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

pub struct Session {
    runtime: Arc<Runtime>,
    cache: PagedKvCache,
    sampler: Sampler,
    tokens: Vec<u32>,
}

impl Session {
    pub(crate) fn new(runtime: Arc<Runtime>) -> Self {
        let config = runtime.model().config();
        let block_count = config.block_count;
        let kv_width = config.kv_width();
        Self {
            runtime,
            cache: PagedKvCache::new(block_count, kv_width, 32),
            sampler: Sampler::new(None),
            tokens: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.cache.clear();
        self.tokens.clear();
    }

    pub fn generate(&mut self, request: &GenerateRequest) -> Result<String> {
        let mut output = String::new();
        self.generate_stream(request, |piece| output.push_str(piece))?;
        Ok(output)
    }

    pub fn generate_stream<F>(&mut self, request: &GenerateRequest, mut on_token: F) -> Result<()>
    where
        F: FnMut(&str),
    {
        self.reset();
        self.sampler.reseed(request.seed);

        let tokenizer = self.runtime.tokenizer();
        let mut prompt_tokens = tokenizer.encode_with_options(&request.prompt, true, true)?;
        if prompt_tokens.is_empty() {
            if let Some(bos) = tokenizer.special_tokens().bos {
                prompt_tokens.push(bos);
            } else {
                return Err(XrtError::Runtime(
                    "empty prompt and tokenizer has no BOS token".to_string(),
                ));
            }
        }
        if prompt_tokens.len() > self.runtime.model().config().context_length {
            return Err(XrtError::Runtime(format!(
                "prompt length {} exceeds model context length {}",
                prompt_tokens.len(),
                self.runtime.model().config().context_length
            )));
        }

        // Batch prefill: process all prompt tokens in a single forward pass.
        // This reads each weight matrix once instead of once per token.
        let mut logits = self
            .runtime
            .model()
            .forward_batch(&prompt_tokens, 0, &mut self.cache)?;
        self.tokens.extend_from_slice(&prompt_tokens);

        let sampler_config = SamplerConfig {
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            repetition_penalty: request.repetition_penalty,
            seed: request.seed,
        };

        for _ in 0..request.max_tokens {
            let next = self.sampler.sample(&logits, &self.tokens, sampler_config)?;
            if Some(next) == tokenizer.special_tokens().eos {
                break;
            }

            if self.tokens.len() >= self.runtime.model().config().context_length {
                break;
            }

            self.tokens.push(next);
            let piece = tokenizer.decode(&[next], true)?;
            if !piece.is_empty() {
                on_token(&piece);
            }

            self.runtime.model().forward_token(
                next,
                self.tokens.len() - 1,
                &mut self.cache,
                &mut logits,
            )?;
        }

        Ok(())
    }
}
