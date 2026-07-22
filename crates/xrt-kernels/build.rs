use std::process::Command;

fn main() {
    // Rust stabilized these AVX-512 intrinsics in 1.89. Keep the declared 1.76
    // MSRV on the AVX2/scalar path while retaining AVX-512 on newer compilers.
    println!("cargo:rerun-if-env-changed=RUSTC");

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = match Command::new(rustc).arg("--version").output() {
        Ok(output) if output.status.success() => output,
        _ => return,
    };
    let version = String::from_utf8_lossy(&output.stdout);
    let release = match version.split_whitespace().nth(1) {
        Some(release) => release,
        None => return,
    };
    let mut parts = release.split('.');
    let major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let minor = parts.next().and_then(|part| part.parse::<u32>().ok());

    if matches!((major, minor), (Some(major), Some(minor)) if major > 1 || minor >= 80) {
        println!("cargo:rustc-check-cfg=cfg(xrt_stable_avx512)");
    }
    if matches!((major, minor), (Some(major), Some(minor)) if major > 1 || minor >= 89) {
        println!("cargo:rustc-cfg=xrt_stable_avx512");
    }
}
