use super::error::{DirsError, ErrorCode, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A process-unique, unpredictable-enough suffix (temp names and lock tokens need uniqueness, not secrecy).
pub(crate) fn unique_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.write_u32(std::process::id());
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", h.finish())
}

fn rename_with_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    // On Windows a rename onto a file another process has open (a scanner, an indexer) fails transiently.
    let mut attempt = 0u64;
    loop {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) => {
                let retryable = cfg!(windows) && e.kind() == std::io::ErrorKind::PermissionDenied;
                if !retryable || attempt >= 15 {
                    return Err(e);
                }
                std::thread::sleep(std::time::Duration::from_millis(20 + attempt * 10));
                attempt += 1;
            }
        }
    }
}

fn sync_directory(dir: &Path) -> std::io::Result<()> {
    // Windows cannot open a directory for fsync through std; NTFS journals the rename itself.
    if cfg!(windows) {
        return Ok(());
    }
    fs::File::open(dir)?.sync_all()
}

/// Atomic replace (spec section 8): write a temp file in the same directory, flush it, rename it over the target,
/// flush the directory. A reader sees the old bytes or the new bytes, never a prefix. `mode` is POSIX-only.
pub fn atomic_write_file(target: &Path, data: &[u8], mode: Option<u32>) -> Result<()> {
    let dir = target.parent().ok_or_else(|| DirsError::new(ErrorCode::InvalidArgument, "target has no parent directory"))?;
    fs::create_dir_all(dir)?;
    let name = target.file_name().and_then(|n| n.to_str()).ok_or_else(|| DirsError::new(ErrorCode::InvalidArgument, "target has no file name"))?;
    let tmp = dir.join(format!(".{name}.{}.{}.tmp", std::process::id(), unique_token()));
    let result = (|| -> std::io::Result<()> {
        let mut opts = OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(mode.unwrap_or(0o644));
        }
        #[cfg(not(unix))]
        let _ = mode;
        let mut f = opts.open(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        drop(f);
        rename_with_retry(&tmp, target)?;
        sync_directory(dir)
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}
