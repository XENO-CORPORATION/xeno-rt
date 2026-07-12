use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    const SOURCE: &str = "src/kernels/awq_gemm4.cu";
    println!("cargo:rerun-if-changed={SOURCE}");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must set OUT_DIR"))
        .join("awq_gemm4.ptx");
    println!("cargo:rerun-if-env-changed=XRT_AWQ_GEMM4_PTX");
    if let Some(precompiled) = env::var_os("XRT_AWQ_GEMM4_PTX") {
        let precompiled = PathBuf::from(precompiled);
        if !precompiled.is_file() {
            panic!(
                "XRT_AWQ_GEMM4_PTX does not name a PTX file: {}",
                precompiled.display()
            );
        }
        fs::copy(&precompiled, &output).unwrap_or_else(|err| {
            panic!(
                "failed to copy precompiled AWQ GEMM4 PTX from {} to {}: {err}",
                precompiled.display(),
                output.display()
            )
        });
        return;
    }
    let nvcc = find_nvcc();
    let result = Command::new(&nvcc)
        .arg("--ptx")
        .arg("--std=c++14")
        .arg("--gpu-architecture=compute_70")
        .arg("--use_fast_math")
        .arg("-O3")
        .arg("-o")
        .arg(&output)
        .arg(SOURCE)
        .output()
        .unwrap_or_else(|err| panic!("failed to execute `{}`: {err}", nvcc.display()));

    if !result.status.success() {
        panic!(
            "NVCC failed while compiling {SOURCE}:\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

fn find_nvcc() -> PathBuf {
    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        let candidate = Path::new(&cuda_path).join("bin").join("nvcc.exe");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("nvcc")
}
