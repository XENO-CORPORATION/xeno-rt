//! Live host headroom is distinct from a configured model byte budget.
use crate::{Result, XrtError};

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct HostMemorySnapshot {
    pub available_physical_bytes: u64,
    pub available_commit_bytes: u64,
}
impl HostMemorySnapshot {
    pub fn admit(&self, physical_bytes: u64, commit_bytes: u64) -> Result<()> {
        if physical_bytes > self.available_physical_bytes
            || commit_bytes > self.available_commit_bytes
        {
            return Err(XrtError::Runtime(format!(
                "host headroom refused: need {physical_bytes} physical / {commit_bytes} commit bytes; available {} physical / {} commit bytes",
                self.available_physical_bytes, self.available_commit_bytes
            )));
        }
        Ok(())
    }
}

/// Windows reports available system commit independently of free physical RAM.
/// This is a preflight observation, not an OS reservation against other processes.
#[cfg(windows)]
pub fn windows_snapshot() -> Result<HostMemorySnapshot> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_physical: 0,
        available_physical: 0,
        total_page_file: 0,
        available_page_file: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended_virtual: 0,
    };
    // SAFETY: repr(C) matches MEMORYSTATUSEX (two DWORDs, seven DWORDLONGs).
    // dwLength is initialized; the writable stack value lives through the call.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(HostMemorySnapshot {
        available_physical_bytes: status.available_physical,
        available_commit_bytes: status.available_page_file.min(status.available_virtual),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn free_ram_cannot_substitute_for_missing_commit() {
        let snapshot = HostMemorySnapshot {
            available_physical_bytes: 64 << 30,
            available_commit_bytes: 2 << 30,
        };
        assert!(snapshot.admit(1 << 30, 3 << 30).is_err());
        assert!(snapshot.admit(65 << 30, 1 << 30).is_err());
        assert!(snapshot.admit(1 << 30, 1 << 30).is_ok());
    }
    #[test]
    #[cfg(windows)]
    fn reads_live_windows_memory_without_allocating_a_model() {
        let snapshot = windows_snapshot().unwrap();
        assert!(snapshot.available_physical_bytes > 0);
        assert!(snapshot.available_commit_bytes > 0);
    }
}
