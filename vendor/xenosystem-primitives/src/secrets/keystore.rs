use super::error::{Result, SecretsError, SecretsErrorCode as C};
use crate::dirs::codec::{hex_decode, hex_encode};
use crate::dirs::registry::inputs;
use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protection {
    OsProtected,
    Unprotected,
}

/// Where the per-machine master key rests. The OS store holds a KEY, never payloads (a Windows credential blob is
/// capped at 2,560 bytes). A store that only offers an unprotected backend declares `Unprotected`, and the secret
/// store then refuses (no plaintext fallback).
pub trait KeyStore: Send + Sync {
    fn id(&self) -> &str;
    fn protection(&self) -> Protection;
    /// Why a store is unprotected (shown to the person).
    fn reason(&self) -> Option<&str> {
        None
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>>;
    fn put_key(&self, key: &[u8]) -> Result<()>;
    /// Removes the key; every secret sealed with it becomes unreadable.
    fn delete_key(&self) -> Result<()>;
}

/// A store with no protected backend on this machine. Every operation refuses with `reason`.
pub struct UnprotectedKeyStore {
    id: String,
    reason: String,
}

pub fn unprotected_key_store(id: &str, reason: &str) -> Box<dyn KeyStore> {
    Box::new(UnprotectedKeyStore { id: id.into(), reason: reason.into() })
}

impl KeyStore for UnprotectedKeyStore {
    fn id(&self) -> &str {
        &self.id
    }
    fn protection(&self) -> Protection {
        Protection::Unprotected
    }
    fn reason(&self) -> Option<&str> {
        Some(&self.reason)
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>> {
        Err(SecretsError::new(C::UnprotectedStore, self.reason.clone()))
    }
    fn put_key(&self, _: &[u8]) -> Result<()> {
        Err(SecretsError::new(C::UnprotectedStore, self.reason.clone()))
    }
    fn delete_key(&self) -> Result<()> {
        Err(SecretsError::new(C::UnprotectedStore, self.reason.clone()))
    }
}

#[derive(Clone, Default)]
pub struct OsStoreOptions {
    /// Defaults to the contract's secretsKeyService / secretsKeyAccount. Tests pass unique names.
    pub service: Option<String>,
    pub account: Option<String>,
}

impl OsStoreOptions {
    pub(crate) fn names(&self) -> (String, String) {
        let i = inputs();
        (self.service.clone().unwrap_or_else(|| i.secrets_key_service.clone()), self.account.clone().unwrap_or_else(|| i.secrets_key_account.clone()))
    }
}

/// Runs a program with stdin; stderr is discarded (it can echo input).
fn run(file: &str, args: &[&str], stdin: Option<&str>) -> std::io::Result<(i32, String)> {
    let mut child = Command::new(file).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn()?;
    if let Some(mut si) = child.stdin.take() {
        if let Some(s) = stdin {
            si.write_all(s.as_bytes())?;
        }
    }
    let out = child.wait_with_output()?;
    Ok((out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).to_string()))
}

fn io_err(what: &str) -> impl Fn(std::io::Error) -> SecretsError + '_ {
    move |_| SecretsError::new(C::KeyUnavailable, format!("could not run {what}"))
}

// ---- macOS: Keychain via `security -i` (secret on stdin, never on argv). UNVERIFIED on a real Mac. -----------------

pub struct MacKeychainKeyStore {
    service: String,
    account: String,
}

impl KeyStore for MacKeychainKeyStore {
    fn id(&self) -> &str {
        "macos-keychain"
    }
    fn protection(&self) -> Protection {
        Protection::OsProtected
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>> {
        let (code, out) = run("/usr/bin/security", &["find-generic-password", "-s", &self.service, "-a", &self.account, "-w"], None).map_err(io_err("security"))?;
        match code {
            44 => Ok(None),
            0 => Ok(hex_decode(out.trim())),
            c => Err(SecretsError::new(C::KeyUnavailable, format!("Keychain read failed (exit {c})"))),
        }
    }
    fn put_key(&self, key: &[u8]) -> Result<()> {
        let cmd = format!("add-generic-password -U -s {} -a {} -w {}\n", self.service, self.account, hex_encode(key));
        let (code, _) = run("/usr/bin/security", &["-i"], Some(&cmd)).map_err(io_err("security"))?;
        if code == 0 { Ok(()) } else { Err(SecretsError::new(C::KeyUnavailable, format!("Keychain write failed (exit {code})"))) }
    }
    fn delete_key(&self) -> Result<()> {
        let (code, _) = run("/usr/bin/security", &["delete-generic-password", "-s", &self.service, "-a", &self.account], None).map_err(io_err("security"))?;
        if code == 0 || code == 44 { Ok(()) } else { Err(SecretsError::new(C::KeyUnavailable, format!("Keychain delete failed (exit {code})"))) }
    }
}

// ---- Linux: libsecret via `secret-tool` (secret on stdin). UNVERIFIED against a live Secret Service. ---------------

pub struct LibsecretKeyStore {
    service: String,
    account: String,
}

impl LibsecretKeyStore {
    fn attrs(&self) -> Vec<&str> {
        vec!["service", &self.service, "account", &self.account]
    }
}

impl KeyStore for LibsecretKeyStore {
    fn id(&self) -> &str {
        "libsecret"
    }
    fn protection(&self) -> Protection {
        Protection::OsProtected
    }
    fn get_key(&self) -> Result<Option<Vec<u8>>> {
        let mut args = vec!["lookup"];
        args.extend(self.attrs());
        let (code, out) = run("secret-tool", &args, None).map_err(io_err("secret-tool"))?;
        if code != 0 || out.trim().is_empty() { Ok(None) } else { Ok(hex_decode(out.trim())) }
    }
    fn put_key(&self, key: &[u8]) -> Result<()> {
        let mut args = vec!["store", "--label=XENO secrets key"];
        args.extend(self.attrs());
        let (code, _) = run("secret-tool", &args, Some(&hex_encode(key))).map_err(io_err("secret-tool"))?;
        if code == 0 { Ok(()) } else { Err(SecretsError::new(C::KeyUnavailable, format!("Secret Service write failed (exit {code})"))) }
    }
    fn delete_key(&self) -> Result<()> {
        let mut args = vec!["clear"];
        args.extend(self.attrs());
        let _ = run("secret-tool", &args, None).map_err(io_err("secret-tool"))?;
        Ok(())
    }
}

/// The key store for this machine's operating system. May be unprotected, in which case the secret store refuses.
pub fn create_os_key_store(options: OsStoreOptions) -> Box<dyn KeyStore> {
    let (service, account) = options.names();
    #[cfg(windows)]
    {
        Box::new(super::windows::WindowsCredentialKeyStore::new(service, account))
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(MacKeychainKeyStore { service, account })
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // Probe now: a missing tool or an unreachable Secret Service means this machine has no protected store.
        let probe = Command::new("secret-tool").args(["lookup", "service", &service, "account", &account]).stdin(Stdio::null()).output();
        match probe {
            Err(_) => unprotected_key_store("libsecret", "secret-tool (libsecret) is not installed; secrets are not saved"),
            Ok(o) if !o.status.success() && !o.stderr.iter().all(|b| b.is_ascii_whitespace()) => {
                unprotected_key_store("libsecret", "no Secret Service (GNOME Keyring, KWallet) is reachable; secrets are not saved")
            }
            Ok(_) => Box::new(LibsecretKeyStore { service, account }),
        }
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (service, account);
        unprotected_key_store("unsupported", "no protected key store is implemented for this platform")
    }
}
