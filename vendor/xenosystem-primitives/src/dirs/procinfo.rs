//! Process identity for lock staleness. A pid alone is not an identity (pids are reused); pid + start time is.
//! The start-time string formats are shared with the TypeScript implementation so either can judge the other's lock:
//! `linux:<boot_id>:<ticks>`, `darwin:<lstart>`, `win32:<creation FILETIME>`.

#[cfg(target_os = "macos")]
use std::process::Command;

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }
    extern "system" {
        pub fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut c_void;
        pub fn CloseHandle(h: *mut c_void) -> i32;
        pub fn GetLastError() -> u32;
        pub fn GetProcessTimes(h: *mut c_void, creation: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        pub fn GetExitCodeProcess(h: *mut c_void, code: *mut u32) -> i32;
    }
    pub const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    pub const STILL_ACTIVE: u32 = 259;
    pub const ERROR_ACCESS_DENIED: u32 = 5;
}

/// True when a process with this pid exists (one we may not open still exists).
pub fn is_process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(windows)]
    unsafe {
        let h = win::OpenProcess(win::PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return win::GetLastError() == win::ERROR_ACCESS_DENIED;
        }
        let mut code = 0u32;
        let ok = win::GetExitCodeProcess(h, &mut code);
        win::CloseHandle(h);
        ok == 0 || code == win::STILL_ACTIVE
    }
    #[cfg(unix)]
    {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        // SAFETY: signal 0 only checks existence.
        let r = unsafe { kill(pid as i32, 0) };
        r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(1) // EPERM
    }
}

/// An opaque, stable identifier for "this process, started at this moment", or None when it cannot be read.
pub fn process_start_time(pid: u32) -> Option<String> {
    #[cfg(windows)]
    unsafe {
        let h = win::OpenProcess(win::PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return None;
        }
        let (mut c, mut e, mut k, mut u) = (win::FileTime::default(), win::FileTime::default(), win::FileTime::default(), win::FileTime::default());
        let ok = win::GetProcessTimes(h, &mut c, &mut e, &mut k, &mut u);
        win::CloseHandle(h);
        if ok == 0 {
            return None;
        }
        let v = ((c.high as u64) << 32) | c.low as u64;
        Some(format!("win32:{v}"))
    }
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let rest: Vec<&str> = stat[stat.rfind(')')? + 2..].split(' ').collect();
        let ticks = rest.get(19)?;
        let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap_or_default();
        Some(format!("linux:{}:{ticks}", boot.trim()))
    }
    #[cfg(target_os = "macos")]
    {
        let out = Command::new("ps").args(["-o", "lstart=", "-p", &pid.to_string()]).output().ok()?;
        let t = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if t.is_empty() { None } else { Some(format!("darwin:{t}")) }
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = pid;
        None
    }
}

pub fn hostname() -> String {
    #[cfg(windows)]
    {
        std::env::var("COMPUTERNAME").unwrap_or_default()
    }
    #[cfg(unix)]
    {
        extern "C" {
            fn gethostname(name: *mut u8, len: usize) -> i32;
        }
        let mut buf = [0u8; 256];
        // SAFETY: buffer is valid for len bytes.
        let r = unsafe { gethostname(buf.as_mut_ptr(), buf.len()) };
        if r != 0 {
            return String::new();
        }
        let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).to_string()
    }
}
