//! Micro-benchmark to measure raw memory streaming bandwidth.
//! Compares sequential read throughput with different strategies to establish
//! the theoretical ceiling for our matvec performance.

use std::time::Instant;
use memmap2::MmapOptions;
use std::io::Write;

fn main() {
    let size_mb = 512;
    let size = size_mb * 1024 * 1024;
    let iterations = 5;

    println!("Memory bandwidth benchmark ({size_mb} MB buffer, {iterations} iterations)");
    println!("=========================================================");

    // Allocate and initialize (force physical pages)
    let mut data: Vec<u8> = vec![0u8; size];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }

    // 1. Single-thread sequential read
    {
        let mut best = f64::MAX;
        for _ in 0..iterations {
            let start = Instant::now();
            let mut sum: u64 = 0;
            let chunks = data.as_ptr() as *const u64;
            let n = size / 8;
            for i in 0..n {
                sum = sum.wrapping_add(unsafe { *chunks.add(i) });
            }
            std::hint::black_box(sum);
            let elapsed = start.elapsed().as_secs_f64();
            let gb_s = (size as f64) / elapsed / 1e9;
            best = best.min(elapsed);
            println!("  1-thread sequential: {gb_s:.1} GB/s (sum={sum})");
        }
        let gb_s = (size as f64) / best / 1e9;
        println!("  BEST 1-thread: {gb_s:.1} GB/s\n");
    }

    // 2. Multi-thread sequential read (like our matvec)
    let thread_counts = [4, 8, 16];
    for &n_threads in &thread_counts {
        let mut best = f64::MAX;
        for _ in 0..iterations {
            let start = Instant::now();
            let chunk_size = size / n_threads;
            let handles: Vec<_> = (0..n_threads).map(|t| {
                let ptr = data.as_ptr() as usize;
                let offset = t * chunk_size;
                let len = chunk_size;
                std::thread::spawn(move || {
                    let base = (ptr + offset) as *const u64;
                    let n = len / 8;
                    let mut sum: u64 = 0;
                    for i in 0..n {
                        sum = sum.wrapping_add(unsafe { *base.add(i) });
                    }
                    sum
                })
            }).collect();

            let mut total_sum: u64 = 0;
            for h in handles {
                total_sum = total_sum.wrapping_add(h.join().unwrap());
            }
            std::hint::black_box(total_sum);
            let elapsed = start.elapsed().as_secs_f64();
            let gb_s = (size as f64) / elapsed / 1e9;
            best = best.min(elapsed);
            println!("  {n_threads}-thread sequential: {gb_s:.1} GB/s");
        }
        let gb_s = (size as f64) / best / 1e9;
        println!("  BEST {n_threads}-thread: {gb_s:.1} GB/s\n");
    }

    // 3. Multi-thread with AVX2 streaming loads
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        for &n_threads in &thread_counts {
            let mut best = f64::MAX;
            for _ in 0..iterations {
                let start = Instant::now();
                let chunk_size = size / n_threads;
                let handles: Vec<_> = (0..n_threads).map(|t| {
                    let ptr = data.as_ptr() as usize;
                    let offset = t * chunk_size;
                    let len = chunk_size;
                    std::thread::spawn(move || unsafe {
                        let base = (ptr + offset) as *const __m256i;
                        let n = len / 32;
                        let mut acc = _mm256_setzero_si256();
                        for i in 0..n {
                            let v = _mm256_load_si256(base.add(i));
                            acc = _mm256_add_epi64(acc, v);
                        }
                        // Extract sum
                        let arr: [u64; 4] = std::mem::transmute(acc);
                        arr[0].wrapping_add(arr[1]).wrapping_add(arr[2]).wrapping_add(arr[3])
                    })
                }).collect();

                let mut total: u64 = 0;
                for h in handles {
                    total = total.wrapping_add(h.join().unwrap());
                }
                std::hint::black_box(total);
                let elapsed = start.elapsed().as_secs_f64();
                let gb_s = (size as f64) / elapsed / 1e9;
                best = best.min(elapsed);
                println!("  {n_threads}-thread AVX2: {gb_s:.1} GB/s");
            }
            let gb_s = (size as f64) / best / 1e9;
            println!("  BEST {n_threads}-thread AVX2: {gb_s:.1} GB/s\n");
        }
    }

    // 4. MMAP test - read from memory-mapped file (like our inference)
    println!("=== MMAP comparison ===");
    {
        let tmp = std::env::temp_dir().join("bandwidth_test.bin");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(&data).unwrap();
            f.flush().unwrap();
        }
        drop(data); // Free heap memory

        let file = std::fs::File::open(&tmp).unwrap();
        let mmap = unsafe { MmapOptions::new().populate().map(&file).unwrap() };

        // Pre-fault all pages (like our inference does)
        for offset in (0..mmap.len()).step_by(4096) {
            std::hint::black_box(mmap[offset]);
        }

        // MMAP 8-thread sequential
        let n_threads = 8;
        let chunk_size = size / n_threads;
        let mut best = f64::MAX;
        for _ in 0..iterations {
            let start = Instant::now();
            let handles: Vec<_> = (0..n_threads).map(|t| {
                let ptr = mmap.as_ptr() as usize;
                let offset = t * chunk_size;
                let len = chunk_size;
                std::thread::spawn(move || {
                    let base = (ptr + offset) as *const u64;
                    let n = len / 8;
                    let mut sum: u64 = 0;
                    for i in 0..n {
                        sum = sum.wrapping_add(unsafe { *base.add(i) });
                    }
                    sum
                })
            }).collect();
            let mut total: u64 = 0;
            for h in handles {
                total = total.wrapping_add(h.join().unwrap());
            }
            std::hint::black_box(total);
            let elapsed = start.elapsed().as_secs_f64();
            let gb_s = (size as f64) / elapsed / 1e9;
            best = best.min(elapsed);
            println!("  MMAP 8-thread sequential: {gb_s:.1} GB/s");
        }
        let gb_s = (size as f64) / best / 1e9;
        println!("  BEST MMAP 8-thread: {gb_s:.1} GB/s\n");

        // Now copy mmap data to heap and test again
        let heap_copy: Vec<u8> = mmap[..].to_vec();
        drop(mmap);
        drop(file);
        let _ = std::fs::remove_file(&tmp);

        let mut best = f64::MAX;
        for _ in 0..iterations {
            let start = Instant::now();
            let handles: Vec<_> = (0..n_threads).map(|t| {
                let ptr = heap_copy.as_ptr() as usize;
                let offset = t * chunk_size;
                let len = chunk_size;
                std::thread::spawn(move || {
                    let base = (ptr + offset) as *const u64;
                    let n = len / 8;
                    let mut sum: u64 = 0;
                    for i in 0..n {
                        sum = sum.wrapping_add(unsafe { *base.add(i) });
                    }
                    sum
                })
            }).collect();
            let mut total: u64 = 0;
            for h in handles {
                total = total.wrapping_add(h.join().unwrap());
            }
            std::hint::black_box(total);
            let elapsed = start.elapsed().as_secs_f64();
            let gb_s = (size as f64) / elapsed / 1e9;
            best = best.min(elapsed);
            println!("  Heap-copy 8-thread sequential: {gb_s:.1} GB/s");
        }
        let gb_s = (size as f64) / best / 1e9;
        println!("  BEST Heap-copy 8-thread: {gb_s:.1} GB/s\n");
    }
}
