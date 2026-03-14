use crate::{PagedKvCache, Runtime, Sampler, SamplerConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use xrt_core::{KvCache, Result, XrtError};

/// N-gram order for prompt lookup decoding.
/// Looks for the last NGRAM_ORDER tokens somewhere earlier in the context.
const NGRAM_ORDER: usize = 3;

/// Maximum number of draft tokens to propose from an n-gram match.
const MAX_DRAFT: usize = 8;

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

        let eos = tokenizer.special_tokens().eos;
        let ctx_len = self.runtime.model().config().context_length;
        let vocab_size = self.runtime.model().config().vocab_size;
        let mut generated = 0usize;

        while generated < request.max_tokens {
            let next = self.sampler.sample(&logits, &self.tokens, sampler_config)?;
            if Some(next) == eos {
                break;
            }
            if self.tokens.len() >= ctx_len {
                break;
            }

            self.tokens.push(next);
            generated += 1;
            let piece = tokenizer.decode(&[next], true)?;
            if !piece.is_empty() {
                on_token(&piece);
            }

            // Try prompt lookup: find n-gram match and draft continuation tokens
            let draft = self.ngram_draft(request.max_tokens - generated);

            if draft.is_empty() {
                // No draft — standard single-token decode
                self.runtime.model().forward_token(
                    next,
                    self.tokens.len() - 1,
                    &mut self.cache,
                    &mut logits,
                )?;
            } else {
                // Speculative decode: run [next, draft...] through the model
                let mut batch_tokens = Vec::with_capacity(1 + draft.len());
                batch_tokens.push(next);
                batch_tokens.extend_from_slice(&draft);

                let start_pos = self.tokens.len() - 1; // position of `next`
                let all_logits = self.runtime.model().forward_batch_all_logits(
                    &batch_tokens,
                    start_pos,
                    &mut self.cache,
                )?;

                // Verify draft tokens greedily (argmax)
                let mut accepted = 0;
                for i in 0..draft.len() {
                    // logits at position i predict the next token after batch_tokens[i]
                    let pos_logits = &all_logits[i * vocab_size..(i + 1) * vocab_size];
                    let predicted = argmax(pos_logits);
                    if predicted == draft[i] {
                        accepted += 1;
                        self.tokens.push(draft[i]);
                        generated += 1;
                        let piece = tokenizer.decode(&[draft[i]], true)?;
                        if !piece.is_empty() {
                            on_token(&piece);
                        }
                        if Some(draft[i]) == eos || self.tokens.len() >= ctx_len || generated >= request.max_tokens {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Roll back KV cache for rejected draft tokens
                let cache_target = self.tokens.len();
                self.cache.truncate(cache_target);

                // The logits for the last accepted position become our current logits
                let last_logit_idx = accepted; // logits[accepted] predicts the next token
                logits.resize(vocab_size, 0.0);
                logits.copy_from_slice(&all_logits[last_logit_idx * vocab_size..(last_logit_idx + 1) * vocab_size]);

                // If we hit EOS or context limit during draft acceptance, stop
                if generated >= request.max_tokens || self.tokens.len() >= ctx_len {
                    break;
                }
                if accepted > 0 && Some(self.tokens[self.tokens.len() - 1]) == eos {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Search for an n-gram match in the token history and return draft continuation tokens.
    /// Returns empty slice if no match found.
    fn ngram_draft(&self, max_tokens: usize) -> Vec<u32> {
        let n = NGRAM_ORDER;
        let tokens = &self.tokens;
        if tokens.len() < n + 1 {
            return Vec::new();
        }

        let max_draft = MAX_DRAFT.min(max_tokens);
        if max_draft == 0 {
            return Vec::new();
        }

        // The n-gram to search for: last N tokens
        let needle = &tokens[tokens.len() - n..];
        let search_end = tokens.len() - n; // don't match the needle itself

        // Search backwards for most recent match (more likely to be relevant)
        let mut best_pos = None;
        for start in (0..search_end).rev() {
            if start + n > search_end {
                continue;
            }
            if tokens[start..start + n] == *needle {
                best_pos = Some(start + n);
                break;
            }
        }

        if let Some(continuation_start) = best_pos {
            let draft_len = max_draft.min(tokens.len() - continuation_start);
            if draft_len > 0 {
                return tokens[continuation_start..continuation_start + draft_len].to_vec();
            }
        }

        Vec::new()
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
