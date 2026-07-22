use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use xrt_core::{Result, XrtError};

use super::topology::CpuThreadBudget;

thread_local! {
    /// Nested dense kernels invoked by an expert task execute serially rather
    /// than trying to dispatch recursively into the same bounded pool.
    static IN_SPIN_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Cache-line aligned to prevent false sharing (critical for performance).
/// llama.cpp had a 30% perf regression from false sharing in their barrier.
#[repr(C, align(128))]
struct CacheAligned<T>(T);

/// A fixed-size thread pool with spin-wait synchronization.
/// Threads spin on atomics waiting for work -- zero dispatch latency.
pub struct SpinPool {
    threads: Vec<thread::JoinHandle<()>>,
    /// Shared state between main thread and workers
    shared: Arc<SharedState>,
    n_workers: usize,
    /// Exactly one dispatch may use the shared job pointer and barrier.
    dispatch_lock: Mutex<()>,
}

/// MoE-facing handle over the runtime-wide bounded CPU pool.
///
/// It intentionally does not create another set of compute threads. Dense
/// kernels and expert jobs therefore consume the same budget, and nested dense
/// projections execute serially inside an expert job.
#[derive(Clone, Copy)]
pub struct ExpertWorkerPool {
    inner: &'static SpinPool,
}

impl ExpertWorkerPool {
    pub fn shared() -> Self {
        Self {
            inner: global_pool(),
        }
    }

    pub fn thread_budget(&self) -> usize {
        self.inner.n_threads()
    }

    pub fn queue_capacity(&self) -> usize {
        1
    }

    pub fn execute_scoped<F>(&self, total: usize, work: F) -> Result<()>
    where
        F: Fn(usize, usize) + Sync,
    {
        self.inner.try_par_for(total, work)
    }

    /// Create a scoped submit/join ticket without allocating or requiring a
    /// `'static` callback. Dispatch begins at `join`, which keeps borrowed model
    /// weights and scratch valid for the entire execution.
    pub fn submit_scoped<F>(&self, total: usize, work: F) -> ExpertJoin<'_, F>
    where
        F: Fn(usize, usize) + Sync,
    {
        ExpertJoin {
            pool: self,
            total,
            work: Some(work),
        }
    }
}

pub struct ExpertJoin<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    pool: &'a ExpertWorkerPool,
    total: usize,
    work: Option<F>,
}

impl<F> ExpertJoin<'_, F>
where
    F: Fn(usize, usize) + Sync,
{
    pub fn join(mut self) -> Result<()> {
        let work = self.work.take().ok_or_else(|| {
            XrtError::Runtime("expert work ticket was already consumed".to_string())
        })?;
        self.pool.execute_scoped(self.total, work)
    }
}

struct SharedState {
    /// Work function pointer + data (set by main thread before signaling)
    work: CacheAligned<AtomicPtr<()>>,
    /// Generation counter -- workers spin on this
    generation: CacheAligned<AtomicU64>,
    /// Barrier for completion
    barrier_count: CacheAligned<AtomicU64>,
    barrier_phase: CacheAligned<AtomicU64>,
    /// Total threads including main
    n_threads: usize,
    /// Shutdown flag
    shutdown: AtomicBool,
    /// Set when any callback panics. Workers survive and reach the barrier.
    worker_failed: AtomicBool,
}

// Safety: SharedState is only accessed through atomic operations and Arc.
unsafe impl Sync for SharedState {}

/// Describes a parallel-for-each job
struct ParForJob {
    /// Function pointer: fn(chunk_start: usize, chunk_end: usize, data: *const u8)
    func: unsafe fn(usize, usize, *const u8),
    data: *const u8,
    total: usize,
    n_threads: usize,
}

impl SpinPool {
    /// Create a pool with `n_workers` background threads.
    /// Total parallelism = n_workers + 1 (main thread also works).
    pub fn new(n_workers: usize) -> Self {
        let n_threads = n_workers + 1;
        let shared = Arc::new(SharedState {
            work: CacheAligned(AtomicPtr::new(std::ptr::null_mut())),
            generation: CacheAligned(AtomicU64::new(0)),
            barrier_count: CacheAligned(AtomicU64::new(0)),
            barrier_phase: CacheAligned(AtomicU64::new(0)),
            n_threads,
            shutdown: AtomicBool::new(false),
            worker_failed: AtomicBool::new(false),
        });

        let mut threads = Vec::with_capacity(n_workers);
        for tid in 0..n_workers {
            let shared = Arc::clone(&shared);
            threads.push(
                thread::Builder::new()
                    .name(format!("xrt-spin-{}", tid))
                    .spawn(move || worker_loop(tid, shared))
                    .expect("failed to spawn worker thread"),
            );
        }

        SpinPool {
            threads,
            shared,
            dispatch_lock: Mutex::new(()),
            n_workers,
        }
    }

    /// Total number of threads (workers + main).
    pub fn n_threads(&self) -> usize {
        self.n_workers + 1
    }

    /// Execute `f(start, end)` in parallel across all threads.
    /// `total` items are split evenly. Main thread participates.
    /// Blocks until all threads complete.
    pub fn par_for<F>(&self, total: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if let Err(error) = self.try_par_for(total, f) {
            panic!("bounded CPU worker execution failed: {error}");
        }
    }

    /// Fallible dispatch for request-facing expert execution.
    ///
    /// The direct queue is bounded to one active job. Concurrent callers wait
    /// for the dispatch lease, while nested kernel calls execute serially on
    /// their current worker instead of oversubscribing or deadlocking the pool.
    pub fn try_par_for<F>(&self, total: usize, f: F) -> Result<()>
    where
        F: Fn(usize, usize) + Sync,
    {
        if total == 0 {
            return Ok(());
        }

        if IN_SPIN_POOL.with(Cell::get) {
            return catch_unwind(AssertUnwindSafe(|| f(0, total)))
                .map_err(|_| XrtError::Runtime("nested CPU kernel callback panicked".to_string()));
        }

        // SharedState has one job slot. Serializing publication keeps a second
        // inference request from replacing a job while workers still read it.
        let _dispatch = self
            .dispatch_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let n_threads = self.n_workers + 1; // workers + main

        // Create a trampoline that calls f
        // We need to erase the type of F for the function pointer
        unsafe fn trampoline<F: Fn(usize, usize) + Sync>(
            start: usize,
            end: usize,
            data: *const u8,
        ) {
            let f = &*(data as *const F);
            f(start, end);
        }

        let job = ParForJob {
            func: trampoline::<F>,
            data: &f as *const F as *const u8,
            total,
            n_threads,
        };

        let shared = &self.shared;

        // Publish work and signal workers
        shared.worker_failed.store(false, Ordering::Release);
        shared
            .work
            .0
            .store(&job as *const ParForJob as *mut (), Ordering::Release);
        shared.generation.0.fetch_add(1, Ordering::Release);
        for worker in &self.threads {
            worker.thread().unpark();
        }

        // Main thread does its chunk (tid = n_workers, i.e. the last chunk)
        let (start, end) = partition(total, n_threads, n_threads - 1);
        if start < end {
            let succeeded = IN_SPIN_POOL.with(|active| {
                let previous = active.replace(true);
                let result = catch_unwind(AssertUnwindSafe(|| f(start, end))).is_ok();
                active.set(previous);
                result
            });
            if !succeeded {
                shared.worker_failed.store(true, Ordering::Release);
            }
        }

        // Wait for all workers via barrier
        barrier_wait(&shared.barrier_count.0, &shared.barrier_phase.0, n_threads);
        shared.work.0.store(std::ptr::null_mut(), Ordering::Release);
        if shared.worker_failed.load(Ordering::Acquire) {
            return Err(XrtError::Runtime(
                "bounded CPU worker callback panicked".to_string(),
            ));
        }
        Ok(())
    }

    /// Shutdown and join all threads.
    pub fn shutdown(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.0.fetch_add(1, Ordering::Release);
        for handle in &self.threads {
            handle.thread().unpark();
        }
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(tid: usize, shared: Arc<SharedState>) {
    let mut last_gen = 0u64;
    loop {
        // Spin briefly for back-to-back kernels, then park so an idle runtime
        // does not consume whole CPU cores or distort unrelated work.
        let mut idle_spins = 0usize;
        let gen = loop {
            let g = shared.generation.0.load(Ordering::Acquire);
            if g != last_gen {
                break g;
            }
            if idle_spins < 4_096 {
                idle_spins += 1;
                core::hint::spin_loop(); // PAUSE on x86
            } else {
                thread::park_timeout(Duration::from_micros(100));
            }
        };

        if shared.shutdown.load(Ordering::Relaxed) {
            break;
        }

        // Load the job
        let job_ptr = shared.work.0.load(Ordering::Acquire) as *const ParForJob;
        let job = unsafe { &*job_ptr };

        // Compute this thread's chunk
        let (start, end) = partition(job.total, job.n_threads, tid);
        if start < end {
            let succeeded = IN_SPIN_POOL.with(|active| {
                let previous = active.replace(true);
                let result = catch_unwind(AssertUnwindSafe(|| unsafe {
                    (job.func)(start, end, job.data)
                }))
                .is_ok();
                active.set(previous);
                result
            });
            if !succeeded {
                shared.worker_failed.store(true, Ordering::Release);
            }
        }

        last_gen = gen;

        // Signal completion via barrier
        barrier_wait(
            &shared.barrier_count.0,
            &shared.barrier_phase.0,
            shared.n_threads,
        );
    }
}

/// Fair partition: first `remainder` threads get one extra item
fn partition(total: usize, n_threads: usize, tid: usize) -> (usize, usize) {
    let chunk = total / n_threads;
    let remainder = total % n_threads;
    let start = tid * chunk + tid.min(remainder);
    let end = start + chunk + if tid < remainder { 1 } else { 0 };
    (start, end)
}

/// Two-variable barrier with separate cache lines to avoid false sharing.
fn barrier_wait(count: &AtomicU64, phase: &AtomicU64, n_threads: usize) {
    let current_phase = phase.load(Ordering::Acquire);
    let arrived = count.fetch_add(1, Ordering::AcqRel) + 1;
    if arrived as usize == n_threads {
        // Last to arrive -- reset counter and flip phase
        count.store(0, Ordering::Release);
        phase.fetch_add(1, Ordering::Release);
    } else {
        // Spin until phase changes
        while phase.load(Ordering::Acquire) == current_phase {
            core::hint::spin_loop();
        }
    }
}

use std::sync::OnceLock;

static GLOBAL_POOL: OnceLock<SpinPool> = OnceLock::new();

/// Get or create the global thread pool.
/// Respects the configured CPU thread budget, falling back to approximately
/// the number of physical cores on SMT systems. Using all logical cores with
/// spin-wait threads causes severe SMT contention during single-threaded
/// sections (RoPE, attention, softmax).
pub fn global_pool() -> &'static SpinPool {
    GLOBAL_POOL.get_or_init(|| {
        let budget = CpuThreadBudget::resolve_from_environment()
            .unwrap_or_else(|_| CpuThreadBudget::host_default());
        // Use half of logical cores, approximately physical cores on SMT systems.
        SpinPool::new(budget.worker_threads())
    })
}

pub fn global_expert_pool() -> ExpertWorkerPool {
    ExpertWorkerPool::shared()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_for_sum() {
        let pool = SpinPool::new(3); // 3 workers + main = 4 threads
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let results = std::sync::Mutex::new(vec![0.0f32; 1000]);

        pool.par_for(1000, |start, end| {
            let mut local: Vec<(usize, f32)> = Vec::new();
            for i in start..end {
                local.push((i, data[i] * 2.0));
            }
            let mut r = results.lock().unwrap();
            for (i, v) in local {
                r[i] = v;
            }
        });

        let r = results.lock().unwrap();
        for i in 0..1000 {
            assert_eq!(r[i], i as f32 * 2.0);
        }
    }

    #[test]
    fn test_partition_fairness() {
        // 10 items across 3 threads: sizes should be 4, 3, 3
        let (s0, e0) = partition(10, 3, 0);
        let (s1, e1) = partition(10, 3, 1);
        let (s2, e2) = partition(10, 3, 2);
        assert_eq!((s0, e0), (0, 4));
        assert_eq!((s1, e1), (4, 7));
        assert_eq!((s2, e2), (7, 10));
    }

    #[test]
    fn test_par_for_empty() {
        let pool = SpinPool::new(2);
        // Should not panic or deadlock
        pool.par_for(0, |_start, _end| {
            panic!("should not be called");
        });
    }

    #[test]
    fn test_concurrent_par_for_dispatches_are_serialized() {
        use std::sync::{atomic::AtomicUsize, Barrier};

        let pool = Arc::new(SpinPool::new(3));
        let start = Arc::new(Barrier::new(5));
        let mut callers = Vec::new();

        for caller in 0..4 {
            let pool = Arc::clone(&pool);
            let start = Arc::clone(&start);
            callers.push(thread::spawn(move || {
                let values = (0..256).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
                start.wait();
                pool.par_for(values.len(), |begin, end| {
                    for value in &values[begin..end] {
                        value.store(caller + 1, Ordering::Relaxed);
                    }
                });
                assert!(values
                    .iter()
                    .all(|value| value.load(Ordering::Relaxed) == caller + 1));
            }));
        }

        start.wait();
        for caller in callers {
            caller.join().expect("parallel caller should complete");
        }
    }

    #[test]
    fn nested_dispatch_runs_serially_without_deadlock() {
        let pool = SpinPool::new(2);
        pool.try_par_for(4, |start, end| {
            for _ in start..end {
                pool.try_par_for(3, |nested_start, nested_end| {
                    assert_eq!((nested_start, nested_end), (0, 3));
                })
                .unwrap();
            }
        })
        .unwrap();
    }

    #[test]
    fn worker_panics_become_errors_and_workers_remain_usable() {
        let pool = SpinPool::new(2);
        assert!(pool
            .try_par_for(9, |start, _end| {
                if start == 0 {
                    panic!("injected worker failure");
                }
            })
            .is_err());
        pool.try_par_for(9, |_start, _end| {}).unwrap();
    }
}
