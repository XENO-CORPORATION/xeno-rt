use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ProcessMemoryStatus {
    pub resident_bytes: u64,
    pub process_peak_resident_bytes: u64,
}

pub(crate) fn process_memory_status() -> Option<ProcessMemoryStatus> {
    platform::process_memory_status()
}

#[cfg(windows)]
mod platform {
    use super::ProcessMemoryStatus;
    use std::{ffi::c_void, mem::size_of};

    #[allow(non_snake_case)]
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    pub(super) fn process_memory_status() -> Option<ProcessMemoryStatus> {
        let size = u32::try_from(size_of::<ProcessMemoryCounters>()).ok()?;
        let mut counters = ProcessMemoryCounters {
            cb: size,
            PageFaultCount: 0,
            PeakWorkingSetSize: 0,
            WorkingSetSize: 0,
            QuotaPeakPagedPoolUsage: 0,
            QuotaPagedPoolUsage: 0,
            QuotaPeakNonPagedPoolUsage: 0,
            QuotaNonPagedPoolUsage: 0,
            PagefileUsage: 0,
            PeakPagefileUsage: 0,
        };
        // Both APIs are read-only process telemetry calls for the current process.
        let success = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) };
        (success != 0).then(|| ProcessMemoryStatus {
            resident_bytes: counters.WorkingSetSize as u64,
            process_peak_resident_bytes: counters.PeakWorkingSetSize as u64,
        })
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::ProcessMemoryStatus;

    pub(super) fn process_memory_status() -> Option<ProcessMemoryStatus> {
        parse_proc_status(&std::fs::read_to_string("/proc/self/status").ok()?)
    }

    fn parse_proc_status(status: &str) -> Option<ProcessMemoryStatus> {
        let resident_bytes = parse_kib(status, "VmRSS:")?;
        let process_peak_resident_bytes = parse_kib(status, "VmHWM:")?;
        Some(ProcessMemoryStatus {
            resident_bytes,
            process_peak_resident_bytes,
        })
    }

    fn parse_kib(status: &str, key: &str) -> Option<u64> {
        let value = status
            .lines()
            .find_map(|line| line.strip_prefix(key))?
            .split_whitespace()
            .next()?
            .parse::<u64>()
            .ok()?;
        value.checked_mul(1024)
    }

    #[cfg(test)]
    mod tests {
        use super::parse_proc_status;

        #[test]
        fn parses_linux_resident_and_high_water_memory() {
            let status = "Name:\txrt\nVmHWM:\t2048 kB\nVmRSS:\t1024 kB\n";
            let memory = parse_proc_status(status).unwrap();
            assert_eq!(memory.resident_bytes, 1024 * 1024);
            assert_eq!(memory.process_peak_resident_bytes, 2048 * 1024);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    use super::ProcessMemoryStatus;

    pub(super) fn process_memory_status() -> Option<ProcessMemoryStatus> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::process_memory_status;

    #[test]
    fn current_process_memory_is_nonzero_when_supported() {
        if let Some(memory) = process_memory_status() {
            assert!(memory.resident_bytes > 0);
            assert!(memory.process_peak_resident_bytes >= memory.resident_bytes);
        }
    }
}
