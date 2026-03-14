//! Micro-benchmark: measure raw matvec_quantized throughput to determine
//! effective memory bandwidth of the kernel vs raw streaming bandwidth.

use std::io::Write;
use std::time::Instant;
use memmap2::MmapOptions;
use xrt_core::DType;
use xrt_kernels::cpu::matmul::matvec_quantized;

fn main() {
    // Simulate a typical Q4_K linear layer: 1024 cols (input dim), 2816 rows (FFN up)
    let cols = 1024usize;
    let rows = 2816usize;
    let dtype = DType::Q4_K;
    let block_size = dtype.block_size(); // 256
    let block_bytes = dtype.block_bytes(); // 144

    let blocks_per_row = cols / block_size;
    let row_bytes = blocks_per_row * block_bytes;
    let total_bytes = row_bytes * rows;

    println!("Matvec benchmark: {rows}x{cols} Q4_K_M");
    println!("  Weight data: {} KB", total_bytes / 1024);
    println!("  Blocks/row: {blocks_per_row}, Bytes/row: {row_bytes}");

    // Create weight data with random-ish bytes (NOT zeros — zero pages are copy-on-write!)
    let mut matrix = vec![0u8; total_bytes];
    for (i, b) in matrix.iter_mut().enumerate() {
        *b = ((i * 7 + 13) & 0xFF) as u8;
    }
    let input = vec![0.5f32; cols];
    let mut output = vec![0.0f32; rows];

    // Warm up
    for _ in 0..10 {
        matvec_quantized(&matrix, rows, cols, dtype, &input, &mut output).unwrap();
    }

    // Benchmark
    let iterations = 500;
    let start = Instant::now();
    for _ in 0..iterations {
        matvec_quantized(&matrix, rows, cols, dtype, &input, &mut output).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_call = elapsed / iterations as f64;
    let gb_s = (total_bytes as f64) / per_call / 1e9;

    println!("\n  {iterations} iterations in {elapsed:.3}s");
    println!("  Per matvec: {:.1} μs", per_call * 1e6);
    println!("  Effective bandwidth: {gb_s:.1} GB/s");
    println!("  (Raw HW bandwidth: ~55 GB/s)");

    // Also test the big output projection: 151936 rows × 1024 cols
    println!("\n--- Output projection: 151936x1024 Q4_K_M ---");
    let rows2 = 151936usize;
    let total_bytes2 = row_bytes * rows2;
    println!("  Weight data: {} MB", total_bytes2 / (1024 * 1024));

    let mut matrix2 = vec![0u8; total_bytes2];
    for (i, b) in matrix2.iter_mut().enumerate() {
        *b = ((i * 7 + 13) & 0xFF) as u8;
    }
    let mut output2 = vec![0.0f32; rows2];

    // Warm up
    for _ in 0..3 {
        matvec_quantized(&matrix2, rows2, cols, dtype, &input, &mut output2).unwrap();
    }

    let iterations2 = 30;
    let start = Instant::now();
    for _ in 0..iterations2 {
        matvec_quantized(&matrix2, rows2, cols, dtype, &input, &mut output2).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let per_call = elapsed / iterations2 as f64;
    let gb_s = (total_bytes2 as f64) / per_call / 1e9;

    println!("  {iterations2} iterations in {elapsed:.3}s");
    println!("  Per matvec: {:.1} ms", per_call * 1e3);
    println!("  Effective bandwidth: {gb_s:.1} GB/s");

    drop(matrix2);
    drop(output2);

    // === REALISTIC TEST: simulate one full forward pass ===
    // Allocate ~300MB of weight data as one big buffer, split into 197 matvecs
    // This prevents L3 caching between matvecs (total >> 32MB L3)
    println!("\n=== Simulated full forward pass (197 matvecs, cold cache) ===");

    // Per-layer: Q(1024×1024) + K(512×1024) + V(512×1024) + O(1024×1024)
    //          + gate(2816×1024) + up(2816×1024) + down(1024×2816)
    let layer_specs: Vec<(usize, usize)> = vec![
        (1024, 1024), // attn_q
        (512, 1024),  // attn_k
        (512, 1024),  // attn_v
        (1024, 1024), // attn_o
        (2816, 1024), // ffn_gate
        (2816, 1024), // ffn_up
        (1024, 2816), // ffn_down
    ];
    let n_layers = 28;

    // Calculate total weight bytes
    let mut total_weight_bytes = 0usize;
    let mut matvec_specs: Vec<(usize, usize, usize)> = Vec::new(); // (offset, rows, cols)
    for _ in 0..n_layers {
        for &(r, c) in &layer_specs {
            let bpr = (c / block_size) * block_bytes;
            matvec_specs.push((total_weight_bytes, r, c));
            total_weight_bytes += bpr * r;
        }
    }
    // Output projection
    {
        let r = 151936;
        let c = 1024;
        let bpr = (c / block_size) * block_bytes;
        matvec_specs.push((total_weight_bytes, r, c));
        total_weight_bytes += bpr * r;
    }

    println!("  Total weight data: {} MB ({} matvecs)", total_weight_bytes / (1024*1024), matvec_specs.len());

    let mut all_weights = vec![0u8; total_weight_bytes];
    for (i, b) in all_weights.iter_mut().enumerate() {
        *b = ((i * 7 + 13) & 0xFF) as u8;
    }
    let input_1024 = vec![0.5f32; 1024];
    let input_2816 = vec![0.5f32; 2816];
    let mut output_buf = vec![0.0f32; 151936]; // big enough for any output

    // Warm up
    let start = Instant::now();
    for &(offset, r, c) in &matvec_specs {
        let bpr = (c / block_size) * block_bytes;
        let mat = &all_weights[offset..offset + bpr * r];
        let inp = if c == 1024 { &input_1024[..] } else { &input_2816[..] };
        matvec_quantized(mat, r, c, dtype, inp, &mut output_buf[..r]).unwrap();
    }
    let warmup = start.elapsed().as_secs_f64();
    let warmup_bw = total_weight_bytes as f64 / warmup / 1e9;
    println!("  Warmup pass: {:.1} ms ({:.1} GB/s)", warmup * 1e3, warmup_bw);

    // Timed passes
    let passes = 5;
    let mut best = f64::MAX;
    for _ in 0..passes {
        let start = Instant::now();
        for &(offset, r, c) in &matvec_specs {
            let bpr = (c / block_size) * block_bytes;
            let mat = &all_weights[offset..offset + bpr * r];
            let inp = if c == 1024 { &input_1024[..] } else { &input_2816[..] };
            matvec_quantized(mat, r, c, dtype, inp, &mut output_buf[..r]).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64();
        let gb_s = total_weight_bytes as f64 / elapsed / 1e9;
        best = best.min(elapsed);
        println!("  Full pass: {:.1} ms ({:.1} GB/s)", elapsed * 1e3, gb_s);
    }
    let gb_s = total_weight_bytes as f64 / best / 1e9;
    let tok_s = 1.0 / best;
    println!("  BEST: {:.1} ms ({:.1} GB/s, matvec-only ~{:.0} tok/s)", best * 1e3, gb_s, tok_s);

    // === MMAP TEST: same pass but through memory-mapped file ===
    println!("\n=== Same pass via MMAP (like actual inference) ===");
    {
        let tmp = std::env::temp_dir().join("matvec_weights.bin");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(&all_weights).unwrap();
            f.flush().unwrap();
        }
        drop(all_weights);

        let file = std::fs::File::open(&tmp).unwrap();
        let mmap = unsafe { MmapOptions::new().populate().map(&file).unwrap() };
        // Pre-fault all pages
        for off in (0..mmap.len()).step_by(4096) {
            std::hint::black_box(mmap[off]);
        }

        let mmap_data: &[u8] = &mmap;
        // Warm up
        for &(offset, r, c) in &matvec_specs {
            let bpr = (c / block_size) * block_bytes;
            let mat = &mmap_data[offset..offset + bpr * r];
            let inp = if c == 1024 { &input_1024[..] } else { &input_2816[..] };
            matvec_quantized(mat, r, c, dtype, inp, &mut output_buf[..r]).unwrap();
        }

        let mut best = f64::MAX;
        for _ in 0..passes {
            let start = Instant::now();
            for &(offset, r, c) in &matvec_specs {
                let bpr = (c / block_size) * block_bytes;
                let mat = &mmap_data[offset..offset + bpr * r];
                let inp = if c == 1024 { &input_1024[..] } else { &input_2816[..] };
                matvec_quantized(mat, r, c, dtype, inp, &mut output_buf[..r]).unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64();
            let gb_s = total_weight_bytes as f64 / elapsed / 1e9;
            best = best.min(elapsed);
            println!("  MMAP pass: {:.1} ms ({:.1} GB/s)", elapsed * 1e3, gb_s);
        }
        let gb_s = total_weight_bytes as f64 / best / 1e9;
        let tok_s = 1.0 / best;
        println!("  BEST MMAP: {:.1} ms ({:.1} GB/s, matvec-only ~{:.0} tok/s)", best * 1e3, gb_s, tok_s);

        drop(mmap);
        drop(file);
        let _ = std::fs::remove_file(&tmp);
    }
}
