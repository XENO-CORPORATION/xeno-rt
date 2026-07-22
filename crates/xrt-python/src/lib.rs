//! Python bindings for xrt-runtime via PyO3.
//!
//! Usage:
//! ```python
//! import xrt
//!
//! rt = xrt.Runtime("model.gguf")
//! print(rt.model_name)
//!
//! sess = rt.session()
//! output = sess.generate("Hello!", max_tokens=128, temperature=0.8)
//! print(output)
//!
//! # Streaming
//! for piece in sess.stream("Hello!", max_tokens=128):
//!     print(piece, end="", flush=True)
//!
//! # Chat formatting
//! messages = [
//!     {"role": "system", "content": "You are helpful."},
//!     {"role": "user", "content": "Hi!"},
//! ]
//! prompt = rt.format_chat(messages)
//! output = sess.generate(prompt, max_tokens=256)
//! ```

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex, MutexGuard};

use xrt_runtime::{GenerateRequest, Runtime as InnerRuntime, Session as InnerSession};
use xrt_tokenizer::ChatMessage;

/// A loaded model runtime.
#[pyclass]
struct Runtime {
    inner: Arc<InnerRuntime>,
}

#[pymethods]
impl Runtime {
    /// Load a GGUF model file.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let inner =
            InnerRuntime::load(model_path).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// The model name from GGUF metadata.
    #[getter]
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.tokenizer().vocab_size()
    }

    /// Create a new inference session.
    fn session(&self) -> Session {
        Session {
            inner: Mutex::new(self.inner.new_session()),
            runtime: self.inner.clone(),
        }
    }

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner
            .tokenizer()
            .encode(text)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Decode token IDs to text.
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.inner
            .tokenizer()
            .decode(&tokens, true)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Format chat messages using the model's chat template.
    /// Each message should be a dict with "role" and "content" keys.
    fn format_chat(
        &self,
        messages: Vec<Bound<'_, PyDict>>,
        add_generation_prompt: Option<bool>,
    ) -> PyResult<String> {
        let mut chat_messages = Vec::with_capacity(messages.len());
        for msg in &messages {
            let role: String = msg
                .get_item("role")?
                .ok_or_else(|| PyRuntimeError::new_err("message missing 'role' key"))?
                .extract()?;
            let content: String = msg
                .get_item("content")?
                .ok_or_else(|| PyRuntimeError::new_err("message missing 'content' key"))?
                .extract()?;
            chat_messages.push(ChatMessage { role, content });
        }
        self.inner
            .tokenizer()
            .format_chat(&chat_messages, add_generation_prompt.unwrap_or(true))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

/// An inference session with KV cache state.
#[pyclass]
struct Session {
    inner: Mutex<InnerSession>,
    /// Kept alive to prevent the runtime from being dropped while the session exists.
    #[allow(dead_code)]
    runtime: Arc<InnerRuntime>,
}

impl Session {
    fn lock_inner(&self) -> PyResult<MutexGuard<'_, InnerSession>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("inference session lock is poisoned"))
    }
}

#[pymethods]
impl Session {
    /// Reset the session (clear KV cache).
    fn reset(&self) -> PyResult<()> {
        self.lock_inner()?.reset();
        Ok(())
    }

    /// Generate text from a prompt. Returns the full output string.
    #[pyo3(signature = (prompt, *, max_tokens=128, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.1, seed=None))]
    fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> PyResult<String> {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            ..Default::default()
        };
        self.lock_inner()?
            .generate(&request)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Stream tokens from a prompt. Returns a list of string pieces.
    /// (For true iterator-based streaming, use generate_stream callback pattern.)
    #[pyo3(signature = (prompt, *, max_tokens=128, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.1, seed=None))]
    fn stream(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> PyResult<Vec<String>> {
        let request = GenerateRequest {
            prompt: prompt.to_string(),
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            ..Default::default()
        };
        let mut pieces = Vec::new();
        self.lock_inner()?
            .generate_stream(&request, |piece| {
                pieces.push(piece.to_string());
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(pieces)
    }
}

/// xrt - XENO Runtime Python bindings
#[pymodule]
fn xrt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runtime>()?;
    m.add_class::<Session>()?;
    Ok(())
}
