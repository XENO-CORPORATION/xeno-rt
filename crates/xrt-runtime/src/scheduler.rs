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

use parking_lot::{Condvar, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_active_sequences: usize,
    pub max_queued_sequences: usize,
    pub stream_buffer_capacity: usize,
    pub prefill_chunk_tokens: usize,
    pub max_decode_turns_before_prefill: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_active_sequences: 1,
            max_queued_sequences: 32,
            stream_buffer_capacity: 32,
            prefill_chunk_tokens: 128,
            max_decode_turns_before_prefill: 8,
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
            ..Self::default()
        })
    }

    pub fn with_execution_policy(
        mut self,
        prefill_chunk_tokens: usize,
        max_decode_turns_before_prefill: usize,
    ) -> Result<Self> {
        if prefill_chunk_tokens == 0 {
            return Err(XrtError::Runtime(
                "scheduler prefill_chunk_tokens must be at least 1".to_string(),
            ));
        }
        if max_decode_turns_before_prefill == 0 {
            return Err(XrtError::Runtime(
                "scheduler max_decode_turns_before_prefill must be at least 1".to_string(),
            ));
        }
        self.prefill_chunk_tokens = prefill_chunk_tokens;
        self.max_decode_turns_before_prefill = max_decode_turns_before_prefill;
        Ok(self)
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
        .and_then(|config| {
            config.with_execution_policy(
                parse_positive_usize("XRT_PREFILL_CHUNK_TOKENS")
                    .unwrap_or(default.prefill_chunk_tokens),
                parse_positive_usize("XRT_MAX_DECODE_TURNS_BEFORE_PREFILL")
                    .unwrap_or(default.max_decode_turns_before_prefill),
            )
        })
        .unwrap_or(default)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerExecutionPhase {
    Prefill,
    Decode,
    Exclusive,
}

impl SchedulerExecutionPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Exclusive => "exclusive",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SchedulerStatus {
    pub max_active_sequences: usize,
    pub max_queued_sequences: usize,
    pub stream_buffer_capacity: usize,
    pub prefill_chunk_tokens: usize,
    pub max_decode_turns_before_prefill: usize,
    pub active_sequences: usize,
    pub queued_sequences: usize,
    pub admitted_total: u64,
    pub rejected_total: u64,
    pub kv_budget_bytes: Option<u64>,
    pub kv_reserved_bytes: u64,
    pub active_execution_phase: Option<&'static str>,
    pub waiting_prefill_turns: usize,
    pub waiting_decode_turns: usize,
    pub waiting_exclusive_turns: usize,
    pub completed_prefill_turns: u64,
    pub completed_decode_turns: u64,
    pub completed_exclusive_turns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerAcquireError {
    QueueFull,
    Closed,
    KvBudgetExceeded {
        requested_bytes: u64,
        reserved_bytes: u64,
        budget_bytes: u64,
    },
}

impl fmt::Display for SchedulerAcquireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => f.write_str("inference scheduler queue is full"),
            Self::Closed => f.write_str("inference scheduler is closed"),
            Self::KvBudgetExceeded {
                requested_bytes,
                reserved_bytes,
                budget_bytes,
            } => write!(
                f,
                "inference KV reservation requires {requested_bytes} bytes with {reserved_bytes} bytes already reserved, exceeding the {budget_bytes}-byte scheduler budget"
            ),
        }
    }
}

impl std::error::Error for SchedulerAcquireError {}

pub struct RequestScheduler {
    config: SchedulerConfig,
    permits: Arc<Semaphore>,
    active_sequences: AtomicUsize,
    queued_sequences: AtomicUsize,
    admitted_total: AtomicU64,
    rejected_total: AtomicU64,
    kv_reservations: Mutex<KvReservationState>,
    execution: Mutex<ExecutionState>,
    execution_ready: Condvar,
}

#[derive(Debug, Default)]
struct ExecutionState {
    active: Option<SchedulerExecutionPhase>,
    waiting_prefill: usize,
    waiting_decode: usize,
    waiting_exclusive: usize,
    consecutive_decode_turns: usize,
    completed_prefill: u64,
    completed_decode: u64,
    completed_exclusive: u64,
}

#[derive(Debug, Default)]
struct KvReservationState {
    budget_bytes: Option<u64>,
    reserved_bytes: u64,
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
            kv_reservations: Mutex::new(KvReservationState::default()),
            execution: Mutex::new(ExecutionState::default()),
            execution_ready: Condvar::new(),
        }
    }

    pub fn config(&self) -> SchedulerConfig {
        self.config
    }

    pub fn configure_kv_budget(&self, budget_bytes: Option<u64>) {
        self.kv_reservations.lock().budget_bytes = budget_bytes;
    }

    pub fn status(&self) -> SchedulerStatus {
        let execution = self.execution.lock();
        let kv_reservations = self.kv_reservations.lock();
        SchedulerStatus {
            max_active_sequences: self.config.max_active_sequences,
            max_queued_sequences: self.config.max_queued_sequences,
            stream_buffer_capacity: self.config.stream_buffer_capacity,
            prefill_chunk_tokens: self.config.prefill_chunk_tokens,
            max_decode_turns_before_prefill: self.config.max_decode_turns_before_prefill,
            active_sequences: self.active_sequences.load(Ordering::Acquire),
            queued_sequences: self.queued_sequences.load(Ordering::Acquire),
            admitted_total: self.admitted_total.load(Ordering::Relaxed),
            rejected_total: self.rejected_total.load(Ordering::Relaxed),
            kv_budget_bytes: kv_reservations.budget_bytes,
            kv_reserved_bytes: kv_reservations.reserved_bytes,
            active_execution_phase: execution.active.map(SchedulerExecutionPhase::as_str),
            waiting_prefill_turns: execution.waiting_prefill,
            waiting_decode_turns: execution.waiting_decode,
            waiting_exclusive_turns: execution.waiting_exclusive,
            completed_prefill_turns: execution.completed_prefill,
            completed_decode_turns: execution.completed_decode,
            completed_exclusive_turns: execution.completed_exclusive,
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

    pub fn acquire_execution_turn(
        self: &Arc<Self>,
        phase: SchedulerExecutionPhase,
    ) -> SchedulerExecutionPermit {
        let mut execution = self.execution.lock();
        increment_waiting(&mut execution, phase);
        while !execution_turn_is_ready(
            &execution,
            phase,
            self.config.max_decode_turns_before_prefill,
        ) {
            self.execution_ready.wait(&mut execution);
        }
        decrement_waiting(&mut execution, phase);
        execution.active = Some(phase);
        if phase == SchedulerExecutionPhase::Decode {
            execution.consecutive_decode_turns =
                execution.consecutive_decode_turns.saturating_add(1);
        } else {
            execution.consecutive_decode_turns = 0;
        }
        drop(execution);

        SchedulerExecutionPermit {
            scheduler: self.clone(),
            phase,
        }
    }

    pub fn reserve_kv_bytes(
        self: &Arc<Self>,
        requested_bytes: u64,
    ) -> std::result::Result<SchedulerKvReservation, SchedulerAcquireError> {
        let mut reservations = self.kv_reservations.lock();
        let Some(next_reserved) = reservations.reserved_bytes.checked_add(requested_bytes) else {
            self.rejected_total.fetch_add(1, Ordering::Relaxed);
            return Err(SchedulerAcquireError::KvBudgetExceeded {
                requested_bytes,
                reserved_bytes: reservations.reserved_bytes,
                budget_bytes: reservations.budget_bytes.unwrap_or(u64::MAX),
            });
        };
        if let Some(budget_bytes) = reservations.budget_bytes {
            if next_reserved > budget_bytes {
                self.rejected_total.fetch_add(1, Ordering::Relaxed);
                return Err(SchedulerAcquireError::KvBudgetExceeded {
                    requested_bytes,
                    reserved_bytes: reservations.reserved_bytes,
                    budget_bytes,
                });
            }
        }
        reservations.reserved_bytes = next_reserved;
        drop(reservations);
        Ok(SchedulerKvReservation {
            scheduler: self.clone(),
            reserved_bytes: requested_bytes,
        })
    }

    fn activate(self: &Arc<Self>, permit: OwnedSemaphorePermit) -> SchedulerPermit {
        self.active_sequences.fetch_add(1, Ordering::AcqRel);
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
        SchedulerPermit {
            scheduler: self.clone(),
            _permit: permit,
        }
    }

    fn release_execution_turn(&self, phase: SchedulerExecutionPhase) {
        let mut execution = self.execution.lock();
        debug_assert_eq!(execution.active, Some(phase));
        execution.active = None;
        match phase {
            SchedulerExecutionPhase::Prefill => {
                execution.completed_prefill = execution.completed_prefill.saturating_add(1)
            }
            SchedulerExecutionPhase::Decode => {
                execution.completed_decode = execution.completed_decode.saturating_add(1)
            }
            SchedulerExecutionPhase::Exclusive => {
                execution.completed_exclusive = execution.completed_exclusive.saturating_add(1)
            }
        }
        drop(execution);
        self.execution_ready.notify_all();
    }
}

fn execution_turn_is_ready(
    execution: &ExecutionState,
    phase: SchedulerExecutionPhase,
    max_decode_turns_before_prefill: usize,
) -> bool {
    if execution.active.is_some() {
        return false;
    }
    if execution.waiting_exclusive > 0 {
        return phase == SchedulerExecutionPhase::Exclusive;
    }

    match phase {
        SchedulerExecutionPhase::Exclusive => true,
        SchedulerExecutionPhase::Decode => {
            execution.waiting_prefill == 0
                || execution.consecutive_decode_turns < max_decode_turns_before_prefill
        }
        SchedulerExecutionPhase::Prefill => {
            execution.waiting_decode == 0
                || execution.consecutive_decode_turns >= max_decode_turns_before_prefill
        }
    }
}

fn increment_waiting(execution: &mut ExecutionState, phase: SchedulerExecutionPhase) {
    let waiting = match phase {
        SchedulerExecutionPhase::Prefill => &mut execution.waiting_prefill,
        SchedulerExecutionPhase::Decode => &mut execution.waiting_decode,
        SchedulerExecutionPhase::Exclusive => &mut execution.waiting_exclusive,
    };
    *waiting = waiting.saturating_add(1);
}

fn decrement_waiting(execution: &mut ExecutionState, phase: SchedulerExecutionPhase) {
    let waiting = match phase {
        SchedulerExecutionPhase::Prefill => &mut execution.waiting_prefill,
        SchedulerExecutionPhase::Decode => &mut execution.waiting_decode,
        SchedulerExecutionPhase::Exclusive => &mut execution.waiting_exclusive,
    };
    *waiting = waiting.saturating_sub(1);
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

pub struct SchedulerExecutionPermit {
    scheduler: Arc<RequestScheduler>,
    phase: SchedulerExecutionPhase,
}

pub struct SchedulerKvReservation {
    scheduler: Arc<RequestScheduler>,
    reserved_bytes: u64,
}

impl Drop for SchedulerKvReservation {
    fn drop(&mut self) {
        let mut reservations = self.scheduler.kv_reservations.lock();
        debug_assert!(reservations.reserved_bytes >= self.reserved_bytes);
        reservations.reserved_bytes = reservations
            .reserved_bytes
            .saturating_sub(self.reserved_bytes);
    }
}

impl Drop for SchedulerExecutionPermit {
    fn drop(&mut self) {
        self.scheduler.release_execution_turn(self.phase);
    }
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
        assert!(SchedulerConfig::new(1, 1, 1)
            .unwrap()
            .with_execution_policy(0, 1)
            .is_err());
        assert!(SchedulerConfig::new(1, 1, 1)
            .unwrap()
            .with_execution_policy(1, 0)
            .is_err());
        assert_eq!(
            SchedulerConfig::new(2, 0, 8).unwrap(),
            SchedulerConfig {
                max_active_sequences: 2,
                max_queued_sequences: 0,
                stream_buffer_capacity: 8,
                ..SchedulerConfig::default()
            }
        );
    }

    #[test]
    fn execution_turns_prioritize_decode_and_bound_prefill_wait() {
        use std::{
            sync::mpsc::{self, TryRecvError},
            thread,
            time::{Duration, Instant},
        };

        let scheduler = Arc::new(RequestScheduler::new(
            SchedulerConfig::new(2, 2, 4)
                .unwrap()
                .with_execution_policy(4, 2)
                .unwrap(),
        ));
        let active = scheduler.acquire_execution_turn(SchedulerExecutionPhase::Prefill);

        let (started_tx, started_rx) = mpsc::channel();
        let (prefill_release_tx, prefill_release_rx) = mpsc::channel();
        let prefill_scheduler = scheduler.clone();
        let prefill_started = started_tx.clone();
        let prefill = thread::spawn(move || {
            let _turn = prefill_scheduler.acquire_execution_turn(SchedulerExecutionPhase::Prefill);
            prefill_started.send("prefill").unwrap();
            prefill_release_rx.recv().unwrap();
        });

        let (decode_release_tx, decode_release_rx) = mpsc::channel();
        let decode_scheduler = scheduler.clone();
        let decode = thread::spawn(move || {
            let _turn = decode_scheduler.acquire_execution_turn(SchedulerExecutionPhase::Decode);
            started_tx.send("decode").unwrap();
            decode_release_rx.recv().unwrap();
        });

        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let status = scheduler.status();
            if status.waiting_prefill_turns == 1 && status.waiting_decode_turns == 1 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "execution waiters did not queue; status: {status:?}"
            );
            thread::yield_now();
        }
        let status = scheduler.status();
        assert_eq!(status.waiting_prefill_turns, 1);
        assert_eq!(status.waiting_decode_turns, 1);

        drop(active);
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "decode"
        );
        assert!(matches!(started_rx.try_recv(), Err(TryRecvError::Empty)));
        decode_release_tx.send(()).unwrap();
        assert_eq!(
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            "prefill"
        );
        prefill_release_tx.send(()).unwrap();

        decode.join().unwrap();
        prefill.join().unwrap();
        let status = scheduler.status();
        assert_eq!(status.active_execution_phase, None);
        assert_eq!(status.completed_decode_turns, 1);
        assert_eq!(status.completed_prefill_turns, 2);
    }

    #[test]
    fn kv_reservations_enforce_aggregate_budget_and_release_on_drop() {
        let scheduler = Arc::new(RequestScheduler::new(SchedulerConfig::default()));
        scheduler.configure_kv_budget(Some(100));

        let first = scheduler.reserve_kv_bytes(60).unwrap();
        assert_eq!(scheduler.status().kv_reserved_bytes, 60);
        assert!(matches!(
            scheduler.reserve_kv_bytes(41),
            Err(SchedulerAcquireError::KvBudgetExceeded {
                requested_bytes: 41,
                reserved_bytes: 60,
                budget_bytes: 100,
            })
        ));
        assert_eq!(scheduler.status().rejected_total, 1);

        drop(first);
        assert_eq!(scheduler.status().kv_reserved_bytes, 0);
        let full = scheduler.reserve_kv_bytes(100).unwrap();
        assert_eq!(scheduler.status().kv_reserved_bytes, 100);
        drop(full);
        assert_eq!(scheduler.status().kv_reserved_bytes, 0);
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
