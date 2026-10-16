use super::atomic::unique_token;
use super::error::{DirsError, ErrorCode, Result};
use super::json::{self, Value};
use super::procinfo::{hostname, is_process_alive, process_start_time};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// What a lock file records (same JSON shape as the TypeScript implementation).
#[derive(Debug, Clone, PartialEq)]
pub struct LockRecord {
    pub pid: u32,
    pub start_time: Option<String>,
    pub host: String,
    pub token: String,
    pub owner: String,
    pub created_at: String,
}

impl LockRecord {
    fn to_json(&self) -> String {
        json::to_string(&Value::Obj(vec![
            ("v".into(), Value::Num(1.0)),
            ("pid".into(), Value::Num(self.pid as f64)),
            ("startTime".into(), self.start_time.clone().map(Value::Str).unwrap_or(Value::Null)),
            ("host".into(), Value::Str(self.host.clone())),
            ("token".into(), Value::Str(self.token.clone())),
            ("owner".into(), Value::Str(self.owner.clone())),
            ("createdAt".into(), Value::Str(self.created_at.clone())),
        ]))
    }
    fn parse(text: &str) -> Option<LockRecord> {
        let v = json::parse(text).ok()?;
        if v.get("v")?.as_i64()? != 1 {
            return None;
        }
        Some(LockRecord {
            pid: u32::try_from(v.get("pid")?.as_i64()?).ok()?,
            start_time: v.get("startTime").and_then(|s| s.as_str()).map(String::from),
            host: v.get("host")?.as_str()?.to_string(),
            token: v.get("token")?.as_str()?.to_string(),
            owner: v.get("owner").and_then(Value::as_str).unwrap_or("").to_string(),
            created_at: v.get("createdAt").and_then(Value::as_str).unwrap_or("").to_string(),
        })
    }
}

/// Injection points so staleness rules can be tested without real processes.
pub struct Probe {
    pub is_alive: Box<dyn Fn(u32) -> bool>,
    pub start_time: Box<dyn Fn(u32) -> Option<String>>,
    pub host: Box<dyn Fn() -> String>,
}

impl Default for Probe {
    fn default() -> Self {
        Probe { is_alive: Box::new(is_process_alive), start_time: Box::new(process_start_time), host: Box::new(hostname) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockVerdict {
    Held,
    Stale,
    ForeignHost,
}

/// Decides whether the holder of a lock is still running. A bare pid is never trusted: a live pid whose start time
/// differs from the recorded one is a reused pid (stale); a live pid whose start time cannot be read on either side
/// is held, because guessing wrong would break mutual exclusion.
pub fn judge_lock(rec: &LockRecord, probe: &Probe) -> LockVerdict {
    if !rec.host.eq_ignore_ascii_case(&(probe.host)()) {
        return LockVerdict::ForeignHost;
    }
    if !(probe.is_alive)(rec.pid) {
        return LockVerdict::Stale;
    }
    match ((probe.start_time)(rec.pid), &rec.start_time) {
        (Some(now), Some(then)) => if &now == then { LockVerdict::Held } else { LockVerdict::Stale },
        _ => LockVerdict::Held,
    }
}

#[derive(Default)]
pub struct AcquireOptions {
    pub owner: String,
    pub timeout: Option<Duration>,
    pub probe: Option<Probe>,
}

#[derive(Debug)]
pub struct LockHandle {
    pub path: PathBuf,
    pub record: LockRecord,
    released: bool,
}

impl LockHandle {
    /// Removes the lock only if it is still ours. Returns false when it was not (gone or taken over).
    pub fn release(mut self) -> Result<bool> {
        self.release_inner()
    }
    fn release_inner(&mut self) -> Result<bool> {
        if self.released {
            return Ok(false);
        }
        self.released = true;
        match read_record(&self.path)? {
            Some((Some(r), _)) if r.token == self.record.token => {
                fs::remove_file(&self.path)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

impl Drop for LockHandle {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

fn read_record(path: &Path) -> Result<Option<(Option<LockRecord>, String)>> {
    match fs::read_to_string(path) {
        Ok(t) => Ok(Some((LockRecord::parse(&t), t))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Creates the lock file atomically with its full content: write a temp file, then hard-link it into place.
fn try_create(path: &Path, rec: &LockRecord) -> Result<bool> {
    let dir = path.parent().unwrap();
    let name = path.file_name().unwrap().to_string_lossy();
    let tmp = dir.join(format!(".{name}.{}.tmp", rec.token));
    {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(rec.to_json().as_bytes())?;
        f.sync_all()?;
    }
    let linked = fs::hard_link(&tmp, path);
    let _ = fs::remove_file(&tmp);
    match linked {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Removes a lock judged stale, but only if the file still holds the record that was judged.
fn break_stale(path: &Path, judged: &LockRecord) -> Result<()> {
    let moved = PathBuf::from(format!("{}.stale.{}", path.display(), unique_token()));
    match fs::rename(path, &moved) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    }
    let taken = read_record(&moved)?;
    if !matches!(&taken, Some((Some(r), _)) if r.token == judged.token) {
        // We moved a lock another process had just created. Put it back.
        let _ = fs::hard_link(&moved, path);
    }
    let _ = fs::remove_file(&moved);
    Ok(())
}

fn iso_now() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem % 3600 / 60, rem % 60, d.subsec_millis())
}

/// Acquires a lock file at a path produced by `XenoDirs::lock_path`.
pub fn acquire_lock_file(path: &Path, options: AcquireOptions) -> Result<LockHandle> {
    let probe = options.probe.unwrap_or_default();
    fs::create_dir_all(path.parent().ok_or_else(|| DirsError::new(ErrorCode::InvalidArgument, "lock path has no parent"))?)?;
    let deadline = options.timeout.map(|t| Instant::now() + t);
    let me = std::process::id();
    let start = (probe.start_time)(me);
    loop {
        let rec = LockRecord { pid: me, start_time: start.clone(), host: (probe.host)(), token: unique_token(), owner: options.owner.clone(), created_at: iso_now() };
        if try_create(path, &rec)? {
            return Ok(LockHandle { path: path.to_path_buf(), record: rec, released: false });
        }
        let Some((cur, _)) = read_record(path)? else { continue };
        let Some(cur) = cur else {
            return Err(DirsError::new(ErrorCode::LockCorrupt, format!("a lock file exists but cannot be read; it is not removed automatically: {}", path.display())));
        };
        match judge_lock(&cur, &probe) {
            LockVerdict::Stale => {
                break_stale(path, &cur)?;
                continue;
            }
            verdict => {
                if deadline.map_or(true, |d| Instant::now() >= d) {
                    let code = if options.timeout.is_some() { ErrorCode::LockTimeout } else { ErrorCode::LockHeld };
                    let why = if verdict == LockVerdict::ForeignHost { "held by another machine; not judged" } else { "held by a running process" };
                    return Err(DirsError::new(code, format!("{why} (pid {}, owner {:?}, since {})", cur.pid, cur.owner, cur.created_at)));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}
