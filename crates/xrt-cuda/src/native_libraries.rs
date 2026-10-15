use super::*;
use std::path::PathBuf;

fn probe(names: &[PathBuf], label: &str) -> Result<libloading::Library> {
    for name in names {
        // SAFETY: only conventional CUDA driver/runtime names are loaded from
        // the platform's library search path. No function pointer is invoked.
        if let Ok(library) = unsafe { libloading::Library::new(name) } {
            return Ok(library);
        }
    }
    Err(XrtError::Cuda(format!("{label} native library is unavailable; install the NVIDIA driver/native CUDA payload and ensure its directory is on the library search path")))
}
pub(super) fn driver() -> Result<libloading::Library> {
    probe(
        &[
            libloading::library_filename("cuda").into(),
            libloading::library_filename("nvcuda").into(),
        ],
        "NVIDIA driver",
    )
}
pub(super) fn nvrtc() -> Result<libloading::Library> {
    let prefix = std::env::consts::DLL_PREFIX;
    let suffix = std::env::consts::DLL_SUFFIX;
    let width = std::mem::size_of::<usize>() * 8;
    // Mirrors the conventional CUDA-12.0 ABI names accepted by cudarc 0.12.1.
    let variants = [
        format!("{prefix}nvrtc{suffix}"),
        format!("{prefix}nvrtc{width}{suffix}"),
        format!("{prefix}nvrtc{width}_12{suffix}"),
        format!("{prefix}nvrtc{width}_120{suffix}"),
        format!("{prefix}nvrtc{width}_120_0{suffix}"),
        format!("{prefix}nvrtc{width}_10{suffix}"),
        format!("{prefix}nvrtc{width}_9{suffix}"),
        format!("{prefix}nvrtc{suffix}.12"),
        format!("{prefix}nvrtc{suffix}.11"),
        format!("{prefix}nvrtc{suffix}.10"),
    ];
    probe(
        &variants.into_iter().map(PathBuf::from).collect::<Vec<_>>(),
        "NVRTC",
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_native_library_is_a_typed_error() {
        let error = probe(
            &[PathBuf::from(
                "xrt-test-nonexistent-nvidia-library-20261014",
            )],
            "test GPU runtime",
        )
        .unwrap_err();
        assert!(matches!(error, XrtError::Cuda(_)));
        assert!(error.to_string().contains("native library is unavailable"));
    }
}
