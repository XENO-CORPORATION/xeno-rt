//! Request-scoped cancellation shared by every ONNX stage in a task.
//! Cancellation is sticky: a cancelled request must never start another run.
use crate::AudioError;
use ort::session::{run_options::RunOptions, Session, SessionInputs, SessionOutputs};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

#[derive(Default)]
pub struct InferenceControl {
    cancelled: AtomicBool,
    options: OnceLock<RunOptions>,
}

impl InferenceControl {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(options) = self.options.get() {
            if let Err(error) = options.terminate() {
                tracing::warn!(%error, "ONNX termination signal failed; cooperative cancellation remains set");
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check(&self) -> Result<(), AudioError> {
        if self.is_cancelled() {
            Err(AudioError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub(crate) fn run<'r, 's: 'r, 'i, 'v: 'i, const N: usize>(
        &'r self,
        session: &'s Session,
        inputs: impl Into<SessionInputs<'i, 'v, N>>,
    ) -> Result<SessionOutputs<'r, 's>, AudioError> {
        self.check()?;
        if self.options.get().is_none() {
            // Initialization is lazy so cancelling before model loading needs
            // no native runtime. Concurrent initialization is harmless.
            let _ = self.options.set(RunOptions::new()?);
        }
        self.check()?;
        let options = self.options.get().expect("initialized run options");
        let result = session.run_with_options(inputs, options);
        // A cancel racing with completion is still cancellation, never a
        // successful result that the caller could accidentally publish.
        self.check()?;
        result.map_err(AudioError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_is_sticky_and_needs_no_onnx_runtime() {
        let c = InferenceControl::default();
        assert!(c.check().is_ok());
        c.cancel();
        c.cancel();
        assert!(c.is_cancelled());
        assert!(matches!(c.check(), Err(AudioError::Cancelled)));
    }
    #[test]
    fn cancellation_crosses_threads() {
        let c = std::sync::Arc::new(InferenceControl::default());
        let worker = c.clone();
        std::thread::spawn(move || worker.cancel()).join().unwrap();
        assert!(matches!(c.check(), Err(AudioError::Cancelled)));
    }
}
