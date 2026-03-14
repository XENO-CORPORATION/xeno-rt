use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

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
        if total == 0 {
            return;
        }

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
        shared
            .work
            .0
            .store(&job as *const ParForJob as *mut (), Ordering::Release);
        shared.generation.0.fetch_add(1, Ordering::Release);

        // Main thread does its chunk (tid = n_workers, i.e. the last chunk)
        let (start, end) = partition(total, n_threads, n_threads - 1);
        if start < end {
            f(start, end);
        }

        // Wait for all workers via barrier
        barrier_wait(
            &shared.barrier_count.0,
            &shared.barrier_phase.0,
            n_threads,
        );
    }

    /// Shutdown and join all threads.
    pub fn shutdown(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.generation.0.fetch_add(1, Ordering::Release);
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
        // Spin-wait for new work
        let gen = loop {
            let g = shared.generation.0.load(Ordering::Acquire);
            if g != last_gen {
                break g;
            }
            core::hint::spin_loop(); // PAUSE on x86, reduces power and pipeline stalls
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
            unsafe { (job.func)(start, end, job.data) };
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
/// Respects RAYON_NUM_THREADS env var for compatibility, then falls back to
/// half of logical cores (≈ physical cores on SMT systems). Using all logical
/// cores with spin-wait threads causes severe SMT contention during
/// single-threaded sections (RoPE, attention, softmax).
pub fn global_pool() -> &'static SpinPool {
    GLOBAL_POOL.get_or_init(|| {
        let n = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.saturating_sub(1)) // n total threads = n-1 workers + main
            .unwrap_or_else(|| {
                let logical = thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4);
                // Use half of logical cores ≈ physical cores on SMT systems.
                // Cap at 16 to avoid diminishing returns on high-core-count CPUs.
                let physical_approx = (logical / 2).max(2).min(16);
                physical_approx.saturating_sub(1)
            });
        SpinPool::new(n)
    })
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
}
