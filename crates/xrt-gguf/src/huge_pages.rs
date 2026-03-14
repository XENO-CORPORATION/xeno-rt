//! Windows huge page (2MB) allocation for model weight data.
//!
//! Huge pages reduce TLB pressure from ~150K entries (4KB pages, 600MB model)
//! to ~300 entries (2MB pages), significantly improving memory streaming bandwidth
//! for the sequential access pattern of matrix-vector multiplication.
//!
//! Requires SeLockMemoryPrivilege, which typically needs admin elevation.

use std::fmt;
use std::ptr;

// Windows constants
const MEM_COMMIT: u32 = 0x00001000;
const MEM_RESERVE: u32 = 0x00002000;
const MEM_RELEASE: u32 = 0x00008000;
const MEM_LARGE_PAGES: u32 = 0x20000000;
const PAGE_READWRITE: u32 = 0x04;

// Token privileges constants
const TOKEN_ADJUST_PRIVILEGES: u32 = 0x0020;
const TOKEN_QUERY: u32 = 0x0008;
const SE_PRIVILEGE_ENABLED: u32 = 0x00000002;

#[repr(C)]
struct Luid {
    low_part: u32,
    high_part: i32,
}

#[repr(C)]
struct LuidAndAttributes {
    luid: Luid,
    attributes: u32,
}

#[repr(C)]
struct TokenPrivileges {
    privilege_count: u32,
    privileges: [LuidAndAttributes; 1],
}

unsafe extern "system" {
    // kernel32
    fn VirtualAlloc(
        lp_address: *mut u8,
        dw_size: usize,
        fl_allocation_type: u32,
        fl_protect: u32,
    ) -> *mut u8;
    fn VirtualFree(lp_address: *mut u8, dw_size: usize, dw_free_type: u32) -> i32;
    fn GetLargePageMinimum() -> usize;
    fn GetLastError() -> u32;
    fn GetCurrentProcess() -> isize;

    // advapi32
    fn OpenProcessToken(process_handle: isize, desired_access: u32, token_handle: *mut isize)
        -> i32;
    fn LookupPrivilegeValueA(
        lp_system_name: *const u8,
        lp_name: *const u8,
        lp_luid: *mut Luid,
    ) -> i32;
    fn AdjustTokenPrivileges(
        token_handle: isize,
        disable_all: i32,
        new_state: *const TokenPrivileges,
        buffer_length: u32,
        previous_state: *mut TokenPrivileges,
        return_length: *mut u32,
    ) -> i32;

    fn CloseHandle(handle: isize) -> i32;
}

/// A buffer backed by Windows huge pages (2MB).
pub struct HugePageBuffer {
    ptr: *mut u8,
    len: usize,
    alloc_size: usize,
}

// SAFETY: The buffer owns its allocation exclusively. Once created, it's immutable
// (we only read from it during inference). Multiple threads can safely read concurrently.
unsafe impl Send for HugePageBuffer {}
unsafe impl Sync for HugePageBuffer {}

impl fmt::Debug for HugePageBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HugePageBuffer")
            .field("len", &self.len)
            .field("alloc_size", &self.alloc_size)
            .finish()
    }
}

impl Drop for HugePageBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                VirtualFree(self.ptr, 0, MEM_RELEASE);
            }
        }
    }
}

impl HugePageBuffer {
    /// Try to allocate huge pages and copy `data` into them.
    /// Returns an error if huge pages are not available (missing privilege, etc).
    pub fn try_alloc_and_copy(data: &[u8]) -> std::result::Result<Self, HugePageError> {
        if data.is_empty() {
            return Err(HugePageError::EmptyData);
        }

        // Enable SeLockMemoryPrivilege for this process
        enable_lock_memory_privilege()?;

        let large_page_size = unsafe { GetLargePageMinimum() };
        if large_page_size == 0 {
            return Err(HugePageError::NotSupported);
        }

        // Round up to huge page boundary
        let alloc_size = (data.len() + large_page_size - 1) & !(large_page_size - 1);

        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                alloc_size,
                MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
                PAGE_READWRITE,
            )
        };

        if ptr.is_null() {
            let err = unsafe { GetLastError() };
            return Err(HugePageError::AllocFailed(err));
        }

        // Copy data into huge page memory
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        Ok(Self {
            ptr,
            len: data.len(),
            alloc_size,
        })
    }

    /// Get a slice of the data (up to the original data length, not the allocation size).
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Enable SeLockMemoryPrivilege for the current process.
fn enable_lock_memory_privilege() -> std::result::Result<(), HugePageError> {
    unsafe {
        let mut token: isize = 0;
        let process = GetCurrentProcess();

        if OpenProcessToken(process, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token) == 0 {
            return Err(HugePageError::PrivilegeError(GetLastError()));
        }

        let mut luid = Luid {
            low_part: 0,
            high_part: 0,
        };
        // "SeLockMemoryPrivilege\0"
        let priv_name = b"SeLockMemoryPrivilege\0";
        if LookupPrivilegeValueA(ptr::null(), priv_name.as_ptr(), &mut luid) == 0 {
            CloseHandle(token);
            return Err(HugePageError::PrivilegeError(GetLastError()));
        }

        let tp = TokenPrivileges {
            privilege_count: 1,
            privileges: [LuidAndAttributes {
                luid,
                attributes: SE_PRIVILEGE_ENABLED,
            }],
        };

        let result = AdjustTokenPrivileges(
            token,
            0,
            &tp,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        let last_err = GetLastError();
        CloseHandle(token);

        // AdjustTokenPrivileges returns success even if privilege wasn't assigned.
        // Check GetLastError for ERROR_NOT_ALL_ASSIGNED (1300).
        if result == 0 || last_err == 1300 {
            return Err(HugePageError::PrivilegeDenied);
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum HugePageError {
    EmptyData,
    NotSupported,
    AllocFailed(u32),
    PrivilegeError(u32),
    PrivilegeDenied,
}

impl fmt::Display for HugePageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "empty data"),
            Self::NotSupported => write!(f, "huge pages not supported on this system"),
            Self::AllocFailed(code) => write!(f, "VirtualAlloc failed with error {code}"),
            Self::PrivilegeError(code) => write!(f, "privilege lookup failed with error {code}"),
            Self::PrivilegeDenied => write!(
                f,
                "SeLockMemoryPrivilege not assigned — run as admin or assign in Local Security Policy"
            ),
        }
    }
}
