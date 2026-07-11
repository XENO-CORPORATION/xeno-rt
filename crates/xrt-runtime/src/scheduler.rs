use std::{
    env, fmt,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use xrt_core::{Result, XrtError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_active_sequences: usize,
    pub max_queued_sequences: usize,
    pub stream_buffer_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_active_sequences: 1,
            max_queued_sequences: 32,
            stream_buffer_capacity: 32,
        }
    }
}

impl SchedulerConfig {
    pub fn new(
        max_active_sequences: usize,
        max_queued_sequences: usize,
        stream_buffer_capacity: usize,
    ) -> Result<Self> {
        if max_active_sequences == 0 {
            return Err(XrtError::Runtime(
                "scheduler max_active_sequences must be at least 1".to_string(),
            ));
        }
        if stream_buffer_capacity == 0 {
            return Err(XrtError::Runtime(
                "scheduler stream_buffer_capacity must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            max_active_sequences,
            max_queued_sequences,
            stream_buffer_capacity,
        })
    }

    pub fn from_env() -> Self {
        let default = Self::default();
        Self::new(
            parse_positive_usize("XRT_MAX_ACTIVE_SEQUENCES")
                .unwrap_or(default.max_active_sequences),
            parse_usize("XRT_MAX_QUEUED_SEQUENCES").unwrap_or(default.max_queued_sequences),
            parse_positive_usize("XRT_STREAM_BUFFER_CAPACITY")
                .unwrap_or(default.stream_buffer_capacity),
        )
        .unwrap_or(default)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SchedulerStatus {
    pub max_active_sequences: usize,
    pub max_queued_sequences: usize,
    pub stream_buffer_capacity: usize,
    pub active_sequences: usize,
    pub queued_sequences: usize,
    pub admitted_total: u64,
    pub rejected_total: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAcquireError {
    QueueFull,
    Closed,
}

impl fmt::Display for SchedulerAcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => f.write_str("inference scheduler queue is full"),
            Self::Closed => f.write_str("inference scheduler is closed"),
        }
    }
}

impl std::error::Error for SchedulerAcquireError {}

#[derive(Debug)]
pub struct RequestScheduler {
    config: SchedulerConfig,
    permits: Arc<Semaphore>,
    active_sequences: AtomicUsize,
    queued_sequences: AtomicUsize,
    admitted_total: AtomicU64,
    rejected_total: AtomicU64,
}

impl RequestScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(config.max_active_sequences)),
            config,
            active_sequences: AtomicUsize::new(0),
            queued_sequences: AtomicUsize::new(0),
            admitted_total: AtomicU64::new(0),
            rejected_total: AtomicU64::new(0),
        }
    }

    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    pub fn status(&self) -> SchedulerStatus {
        SchedulerStatus {
            max_active_sequences: self.config.max_active_sequences,
            max_queued_sequences: self.config.max_queued_sequences,
            stream_buffer_capacity: self.config.stream_buffer_capacity,
            active_sequences: self.active_sequences.load(Ordering::Acquire),
            queued_sequences: self.queued_sequences.load(Ordering::Acquire),
            admitted_total: self.admitted_total.load(Ordering::Relaxed),
            rejected_total: self.rejected_total.load(Ordering::Relaxed),
        }
    }

    pub async fn acquire(
        self: &Arc<Self>,
    ) -> std::result::Result<SchedulerPermit, SchedulerAcquireError> {
        if let Ok(permit) = self.permits.clone().try_acquire_owned() {
            return Ok(self.activate(permit));
        }

        if self
            .queued_sequences
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < self.config.max_queued_sequences).then_some(queued + 1)
            })
            .is_err()
        {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            return Err(SchedulerAcquireError::QueueFull);
        }

        let mut queued = QueuedRegistration {
            scheduler: self.clone(),
            registered: true,
        };
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| SchedulerAcquireError::Closed)?;
        queued.promote();
        Ok(self.activate(permit))
    }

    fn activate(self: &Arc<Self>, permit: OwnedSemaphorePermit) -> SchedulerPermit {
        self.active_sequences.fetch_add(1, Ordering::AcqRel);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        SchedulerPermit {
            scheduler: self.clone(),
            _permit: permit,
        }
    }
}

struct QueuedRegistration {
    scheduler: Arc<RequestScheduler>,
    registered: bool,
}

impl QueuedRegistration {
    fn promote(&mut self) {
        if self.registered {
            self.scheduler
                .queued_sequences
                .fetch_sub(1, Ordering::AcqRel);
            self.registered = false;
        }
    }
}

impl Drop for QueuedRegistration {
    fn drop(&mut self) {
        self.promote();
    }
}

pub struct SchedulerPermit {
    scheduler: Arc<RequestScheduler>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for SchedulerPermit {
    fn drop(&mut self) {
        self.scheduler
            .active_sequences
            .fetch_sub(1, Ordering::AcqRel);
    }
}

fn parse_usize(name: &str) -> Option<usize> {
    env::var(name).ok()?.trim().parse().ok()
}

fn parse_positive_usize(name: &str) -> Option<usize> {
    parse_usize(name).filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn wait_for_queued(scheduler: &RequestScheduler, expected: usize) {
        for _ in 0..1_000 {
            if scheduler.status().queued_sequences == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "scheduler queue count did not become {expected}; status: {:?}",
            scheduler.status()
        );
    }

    #[test]
    fn scheduler_config_rejects_zero_active_and_stream_capacity() {
        assert!(SchedulerConfig::new(0, 1, 1).is_err());
        assert!(SchedulerConfig::new(1, 1, 0).is_err());
        assert_eq!(
            SchedulerConfig::new(2, 0, 8).unwrap(),
            SchedulerConfig {
                max_active_sequences: 2,
                max_queued_sequences: 0,
                stream_buffer_capacity: 8,
            }
        );
    }

    #[tokio::test]
    async fn scheduler_bounds_active_and_queued_requests() {
        let scheduler = Arc::new(RequestScheduler::new(
            SchedulerConfig::new(1, 1, 4).unwrap(),
        ));
        let first = scheduler.acquire().await.unwrap();
        assert_eq!(scheduler.status().active_sequences, 1);

        let queued_scheduler = scheduler.clone();
        let queued = tokio::spawn(async move { queued_scheduler.acquire().await.unwrap() });
        wait_for_queued(&scheduler, 1).await;

        assert!(matches!(
            scheduler.acquire().await,
            Err(SchedulerAcquireError::QueueFull)
        ));
        assert_eq!(scheduler.status().rejected_total, 1);

        drop(first);
        let second = queued.await.unwrap();
        let status = scheduler.status();
        assert_eq!(status.active_sequences, 1);
        assert_eq!(status.queued_sequences, 0);
        assert_eq!(status.admitted_total, 2);
        drop(second);
        assert_eq!(scheduler.status().active_sequences, 0);
    }

    #[tokio::test]
    async fn cancelled_waiter_releases_queue_capacity() {
        let scheduler = Arc::new(RequestScheduler::new(
            SchedulerConfig::new(1, 1, 4).unwrap(),
        ));
        let active = scheduler.acquire().await.unwrap();

        let queued_scheduler = scheduler.clone();
        let queued = tokio::spawn(async move { queued_scheduler.acquire().await });
        wait_for_queued(&scheduler, 1).await;

        queued.abort();
        match queued.await {
            Err(err) => assert!(err.is_cancelled()),
            Ok(_) => panic!("aborted scheduler waiter completed unexpectedly"),
        }
        wait_for_queued(&scheduler, 0).await;

        let replacement_scheduler = scheduler.clone();
        let replacement =
            tokio::spawn(async move { replacement_scheduler.acquire().await.unwrap() });
        wait_for_queued(&scheduler, 1).await;

        drop(active);
        let replacement = replacement.await.unwrap();
        assert_eq!(scheduler.status().queued_sequences, 0);
        drop(replacement);
        assert_eq!(scheduler.status().active_sequences, 0);
    }
}
