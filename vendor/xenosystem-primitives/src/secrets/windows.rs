//! Windows Credential Manager (generic credential, persisted to this machine only; the system encrypts it with DPAPI).
//! The same entry the TypeScript implementation reads and writes: target `<service>:<account>`, type GENERIC,
//! persist LOCAL_MACHINE, user name `<account>`, blob = the 32-byte key.

use super::error::{Result, SecretsError, SecretsErrorCode as C};
use super::keystore::{KeyStore, Protection};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FileTime {
    low: u32,
    high: u32,
}

#[repr(C)]
struct Credential {
    flags: u32,
    typ: u32,
    target_name: *mut u16,
    comment: *mut u16,
    last_written: FileTime,
    blob_size: u32,
    blob: *mut u8,
    persist: u32,
    attribute_count: u32,
    attributes: *mut c_void,
    target_alias: *mut u16,
    user_name: *mut u16,
}

#[link(name = "advapi32")]
extern "system" {
    fn CredReadW(target: *const u16, typ: u32, flags: u32, cred: *mut *mut Credential) -> i32;
    fn CredWriteW(cred: *const Credential, flags: u32) -> i32;
    fn CredDeleteW(target: *const u16, typ: u32, flags: u32) -> i32;
    fn CredFree(p: *mut c_void);
}
extern "system" {
    fn GetLastError() -> u32;
}

const CRED_TYPE_GENERIC: u32 = 1;
const CRED_PERSIST_LOCAL_MACHINE: u32 = 2;
const ERROR_NOT_FOUND: u32 = 1168;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub struct WindowsCredentialKeyStore {
    target: String,
    account: String,
}

impl WindowsCredentialKeyStore {
    pub fn new(service: String, account: String) -> Self {
        WindowsCredentialKeyStore { target: format!("{service}:{account}"), account }
    }
}

impl KeyStore for WindowsCredentialKeyStore {
    fn id(&self) -> &str {
        "windows-credential-manager"
    }
    fn protection(&self) -> Protection {
        Protection::OsProtected
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>> {
        let target = wide(&self.target);
        let mut p: *mut Credential = std::ptr::null_mut();
        // SAFETY: CredReadW allocates `p`; we copy the blob and release it with CredFree.
        unsafe {
            if CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut p) == 0 {
                let e = GetLastError();
                return if e == ERROR_NOT_FOUND { Ok(None) } else { Err(SecretsError::new(C::KeyUnavailable, format!("Windows Credential Manager read failed ({e})"))) };
            }
            let c = &*p;
            let blob = std::slice::from_raw_parts(c.blob, c.blob_size as usize).to_vec();
            CredFree(p as *mut c_void);
            Ok(Some(blob))
        }
    }
    fn put_key(&self, key: &[u8]) -> Result<()> {
        let mut target = wide(&self.target);
        let mut user = wide(&self.account);
        let mut blob = key.to_vec();
        let cred = Credential {
            flags: 0,
            typ: CRED_TYPE_GENERIC,
            target_name: target.as_mut_ptr(),
            comment: std::ptr::null_mut(),
            last_written: FileTime::default(),
            blob_size: blob.len() as u32,
            blob: blob.as_mut_ptr(),
            persist: CRED_PERSIST_LOCAL_MACHINE,
            attribute_count: 0,
            attributes: std::ptr::null_mut(),
            target_alias: std::ptr::null_mut(),
            user_name: user.as_mut_ptr(),
        };
        // SAFETY: all pointers outlive the call.
        let ok = unsafe { CredWriteW(&cred, 0) };
        let err = if ok == 0 { Some(unsafe { GetLastError() }) } else { None };
        blob.fill(0);
        match err {
            None => Ok(()),
            Some(e) => Err(SecretsError::new(C::KeyUnavailable, format!("Windows Credential Manager write failed ({e})"))),
        }
    }
    fn delete_key(&self) -> Result<()> {
        let target = wide(&self.target);
        // SAFETY: target is a valid NUL-terminated wide string.
        unsafe {
            if CredDeleteW(target.as_ptr(), CRED_TYPE_GENERIC, 0) == 0 {
                let e = GetLastError();
                if e != ERROR_NOT_FOUND {
                    return Err(SecretsError::new(C::KeyUnavailable, format!("Windows Credential Manager delete failed ({e})")));
                }
            }
        }
        Ok(())
    }
}
