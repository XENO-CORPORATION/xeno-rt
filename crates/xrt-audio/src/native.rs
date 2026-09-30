//! Load a specific ONNX Runtime and query its actual version, not its Git
//! build-info branch (release NuGet libraries often report `git-branch=HEAD`).
use crate::AudioError;
use std::{ffi::CStr, path::PathBuf, sync::OnceLock};

pub fn initialize() -> Result<String, AudioError> {
    static INITIALIZED: OnceLock<Result<String, String>> = OnceLock::new();
    INITIALIZED
        .get_or_init(initialize_once)
        .clone()
        .map_err(AudioError::RuntimeIncompatible)
}

fn initialize_once() -> Result<String, String> {
    let filename = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let path = if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        PathBuf::from(path)
    } else {
        let beside = std::env::current_exe()
            .map_err(|e| e.to_string())?
            .parent()
            .ok_or("executable has no parent")?
            .join(filename);
        if beside.is_file() {
            beside
        } else {
            return Err(format!("{filename} is missing beside the executable; install the native runtime package or set ORT_DYLIB_PATH"));
        }
    };
    let path = std::fs::canonicalize(&path)
        .map_err(|e| format!("cannot open ONNX Runtime {}: {e}", path.display()))?;
    // SAFETY: the selected library is the operator override or packaged native
    // runtime. OrtGetApiBase and GetVersionString use the public ONNX C ABI.
    // Check every pointer and copy the string while the library handle is live.
    let version = unsafe {
        let library = libloading::Library::new(&path)
            .map_err(|e| format!("ONNX Runtime load failed: {e}"))?;
        let getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase> =
            library
                .get(b"OrtGetApiBase\0")
                .map_err(|e| format!("ONNX API entrypoint missing: {e}"))?;
        let base = getter();
        if base.is_null() {
            return Err("ONNX API base is null".into());
        }
        let get_version = (*base)
            .GetVersionString
            .ok_or("ONNX version function missing")?;
        let version = get_version();
        if version.is_null() {
            return Err("ONNX version is null".into());
        }
        CStr::from_ptr(version)
            .to_str()
            .map_err(|e| e.to_string())?
            .to_string()
    };
    validate_version(&version)?;
    std::panic::catch_unwind(|| ort::init_from(path.to_string_lossy()).commit())
        .map_err(|_| "ONNX Runtime initialization panicked; check native dependencies".to_string())?
        .map_err(|e| format!("ONNX Runtime initialization failed: {e}"))?;
    Ok(version)
}
fn validate_version(version: &str) -> Result<(), String> {
    let numbers = version
        .split('.')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "malformed ONNX version")?;
    if numbers.len() < 2 || numbers[0] != 1 || numbers[1] < 23 {
        return Err(format!(
            "ONNX Runtime {version} cannot run Chatterbox v3; install 1.23 or newer compatible 1.x"
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn version_floor_is_checked_not_guessed_from_build_info() {
        for version in ["1.23.0", "1.26.0"] {
            assert!(validate_version(version).is_ok());
        }
        for version in ["1.20.0", "unknown", "HEAD", "2.1.0", ""] {
            assert!(validate_version(version).is_err());
        }
    }
}
