//! C FFI bindings for xrt-runtime.
//!
//! Provides a minimal C API for loading models, creating sessions, and generating text.
//! All functions return 0 on success or -1 on error. Use `xrt_last_error` to get the
//! error message after a failure.
//!
//! # Example (C)
//! ```c
//! XrtRuntime *rt = NULL;
//! if (xrt_runtime_load("model.gguf", &rt) != 0) {
//!     fprintf(stderr, "error: %s\n", xrt_last_error());
//!     return 1;
//! }
//! XrtSession *sess = NULL;
//! xrt_session_new(rt, &sess);
//!
//! char *output = NULL;
//! xrt_generate(sess, "Hello!", 128, 0.8, &output);
//! printf("%s\n", output);
//! xrt_string_free(output);
//!
//! xrt_session_free(sess);
//! xrt_runtime_free(rt);
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

use xrt_runtime::{GenerateRequest, Runtime, Session};

thread_local! {
    static LAST_ERROR: std::cell::RefCell<CString> = std::cell::RefCell::new(CString::new("").unwrap());
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() =
            CString::new(msg).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    });
}

/// Returns a pointer to the last error message (thread-local, valid until the next error).
/// Returns an empty string if no error has occurred.
#[no_mangle]
pub extern "C" fn xrt_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

// --- Runtime ---

/// Opaque runtime handle.
pub struct XrtRuntime {
    inner: Arc<Runtime>,
}

/// Load a GGUF model file and create a runtime.
/// On success, writes the runtime pointer to `*out` and returns 0.
/// On failure, returns -1 (call `xrt_last_error` for details).
#[no_mangle]
pub unsafe extern "C" fn xrt_runtime_load(
    model_path: *const c_char,
    out: *mut *mut XrtRuntime,
) -> i32 {
    if model_path.is_null() || out.is_null() {
        set_error("null pointer argument");
        return -1;
    }
    let path = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&format!("invalid UTF-8 path: {e}"));
            return -1;
        }
    };
    match Runtime::load(path) {
        Ok(runtime) => {
            *out = Box::into_raw(Box::new(XrtRuntime { inner: runtime }));
            0
        }
        Err(e) => {
            set_error(&format!("{e}"));
            -1
        }
    }
}

/// Get the model name. Returns a pointer valid for the lifetime of the runtime.
#[no_mangle]
pub unsafe extern "C" fn xrt_runtime_model_name(rt: *const XrtRuntime) -> *const c_char {
    if rt.is_null() {
        return ptr::null();
    }
    // Store in a leaked CString — lifetime tied to runtime (freed on runtime_free)
    let name = (*rt).inner.model_name();
    match CString::new(name) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null(),
    }
}

/// Free a runtime. Null-safe.
#[no_mangle]
pub unsafe extern "C" fn xrt_runtime_free(rt: *mut XrtRuntime) {
    if !rt.is_null() {
        drop(Box::from_raw(rt));
    }
}

// --- Session ---

/// Opaque session handle.
pub struct XrtSession {
    inner: Session,
}

/// Create a new inference session from a runtime.
#[no_mangle]
pub unsafe extern "C" fn xrt_session_new(rt: *const XrtRuntime, out: *mut *mut XrtSession) -> i32 {
    if rt.is_null() || out.is_null() {
        set_error("null pointer argument");
        return -1;
    }
    let session = (*rt).inner.new_session();
    *out = Box::into_raw(Box::new(XrtSession { inner: session }));
    0
}

/// Reset the session (clear KV cache and state).
#[no_mangle]
pub unsafe extern "C" fn xrt_session_reset(sess: *mut XrtSession) {
    if !sess.is_null() {
        (*sess).inner.reset();
    }
}

/// Free a session. Null-safe.
#[no_mangle]
pub unsafe extern "C" fn xrt_session_free(sess: *mut XrtSession) {
    if !sess.is_null() {
        drop(Box::from_raw(sess));
    }
}

// --- Generation ---

/// Generate text from a prompt. Returns the full output as a C string.
/// The caller must free the output string with `xrt_string_free`.
#[no_mangle]
pub unsafe extern "C" fn xrt_generate(
    sess: *mut XrtSession,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    out: *mut *mut c_char,
) -> i32 {
    if sess.is_null() || prompt.is_null() || out.is_null() {
        set_error("null pointer argument");
        return -1;
    }
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&format!("invalid UTF-8 prompt: {e}"));
            return -1;
        }
    };
    let request = GenerateRequest {
        prompt: prompt_str.to_string(),
        max_tokens: max_tokens as usize,
        temperature,
        ..Default::default()
    };
    match (*sess).inner.generate(&request) {
        Ok(output) => match CString::new(output) {
            Ok(cs) => {
                *out = cs.into_raw();
                0
            }
            Err(e) => {
                set_error(&format!("output contains null byte: {e}"));
                -1
            }
        },
        Err(e) => {
            set_error(&format!("{e}"));
            -1
        }
    }
}

/// Generate text with streaming. Calls `callback(user_data, piece)` for each token piece.
/// The piece pointer is only valid for the duration of the callback.
#[no_mangle]
pub unsafe extern "C" fn xrt_generate_stream(
    sess: *mut XrtSession,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    callback: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *const c_char)>,
    user_data: *mut std::ffi::c_void,
) -> i32 {
    if sess.is_null() || prompt.is_null() {
        set_error("null pointer argument");
        return -1;
    }
    let cb = match callback {
        Some(cb) => cb,
        None => {
            set_error("null callback");
            return -1;
        }
    };
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(&format!("invalid UTF-8 prompt: {e}"));
            return -1;
        }
    };
    let request = GenerateRequest {
        prompt: prompt_str.to_string(),
        max_tokens: max_tokens as usize,
        temperature,
        ..Default::default()
    };
    match (*sess).inner.generate_stream(&request, |piece| {
        if let Ok(cs) = CString::new(piece) {
            cb(user_data, cs.as_ptr());
        }
    }) {
        Ok(()) => 0,
        Err(e) => {
            set_error(&format!("{e}"));
            -1
        }
    }
}

/// Free a string returned by xrt_generate. Null-safe.
#[no_mangle]
pub unsafe extern "C" fn xrt_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
