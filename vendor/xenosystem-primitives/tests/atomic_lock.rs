//! Atomic write, layout refusal and lock staleness. Fixtures live in a test-owned temp root, never the real home.
#![cfg(feature = "dirs")]

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use xenosystem_primitives::dirs::lock::LockRecord;
use xenosystem_primitives::dirs::*;

struct Fixture(PathBuf);
impl Fixture {
    fn new(label: &str) -> Fixture {
        let p = std::env::temp_dir().join(format!("xeno-prim-{label}-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(&p).unwrap();
        Fixture(p)
    }
    fn dirs(&self) -> XenoDirs {
        create_dirs_at(&self.0.join("home")).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Fixtures never contain links; refuse to delete if one appeared.
        fn has_link(d: &Path) -> bool {
            fs::read_dir(d).map(|it| it.flatten().any(|e| e.file_type().map(|t| t.is_symlink() || (t.is_dir() && has_link(&e.path()))).unwrap_or(false))).unwrap_or(false)
        }
        if !has_link(&self.0) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}

fn files(root: &Path) -> Vec<String> {
    fn walk(d: &Path, rel: &str, out: &mut Vec<String>) {
        if let Ok(it) = fs::read_dir(d) {
            for e in it.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                let r = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
                if e.file_type().unwrap().is_dir() { walk(&e.path(), &r, out) } else { out.push(r) }
            }
        }
    }
    let mut v = Vec::new();
    walk(root, "", &mut v);
    v.sort();
    v
}

fn code<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Ok(v) => format!("OK({v:?})"),
        Err(e) => e.code.as_str().to_string(),
    }
}

fn dead_pid() -> u32 {
    let mut c = if cfg!(windows) { Command::new("cmd").args(["/c", "exit", "0"]).spawn().unwrap() } else { Command::new("true").spawn().unwrap() };
    let pid = c.id();
    c.wait().unwrap();
    pid
}

fn rec(pid: u32, start: Option<&str>, host: &str, token: &str) -> LockRecord {
    LockRecord { pid, start_time: start.map(String::from), host: host.into(), token: token.into(), owner: String::new(), created_at: "2026-01-01T00:00:00.000Z".into() }
}

fn write_lock(path: &Path, r: &LockRecord) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let json = format!(
        "{{\"v\":1,\"pid\":{},\"startTime\":{},\"host\":\"{}\",\"token\":\"{}\",\"owner\":\"\",\"createdAt\":\"{}\"}}",
        r.pid,
        r.start_time.as_ref().map(|s| format!("\"{s}\"")).unwrap_or("null".into()),
        r.host,
        r.token,
        r.created_at
    );
    fs::write(path, json).unwrap();
}

fn token_in(path: &Path) -> String {
    let t = fs::read_to_string(path).unwrap();
    let i = t.find("\"token\":\"").unwrap() + 9;
    t[i..i + t[i..].find('"').unwrap()].to_string()
}

#[test]
fn atomic_write_creates_replaces_and_leaves_no_temp_files() {
    let f = Fixture::new("atomic");
    let target = f.0.join("a/b/file.txt");
    atomic_write_file(&target, b"one", None).unwrap();
    atomic_write_file(&target, b"two", None).unwrap();
    assert_eq!(fs::read_to_string(&target).unwrap(), "two");
    assert_eq!(files(&f.0.join("a")), vec!["b/file.txt"]);
}

#[test]
fn atomic_write_that_cannot_rename_removes_its_temp_file() {
    let f = Fixture::new("atomic-rename");
    let target = f.0.join("dir-in-the-way");
    fs::create_dir_all(&target).unwrap();
    fs::write(target.join("x"), "x").unwrap();
    assert!(atomic_write_file(&target, b"data", None).is_err());
    assert_eq!(files(&f.0), vec!["dir-in-the-way/x"]);
}

#[test]
fn resolver_write_is_registry_checked_and_creates_layout() {
    let f = Fixture::new("write");
    let d = f.dirs();
    let p = d.write_file(Kind::Data, "agents/environments", &["claude", "profile.json"], b"{\"id\":1}").unwrap();
    assert_eq!(fs::read_to_string(p).unwrap(), "{\"id\":1}");
    assert_eq!(fs::read_to_string(d.layout_path()).unwrap().trim(), "{\"layout\":1}");
    assert_eq!(code(d.write_file(Kind::Data, "scratch", &["x"], b"no")), "unregistered-area");
    assert_eq!(code(d.write_file(Kind::State, "agents/environments", &["x"], b"no")), "kind-not-used");
    assert_eq!(code(d.write_file(Kind::Data, "minds", &["..", "escape"], b"no")), "invalid-segment");
    assert!(!Path::new(d.root(Kind::Data)).join("scratch").exists());
}

#[test]
fn newer_or_unreadable_layout_refuses_to_write() {
    let f = Fixture::new("layout");
    let d = f.dirs();
    fs::create_dir_all(d.root(Kind::Data)).unwrap();
    fs::write(d.layout_path(), "{\"layout\":2}").unwrap();
    assert_eq!(code(d.write_file(Kind::Data, "minds", &["m.json"], b"x")), "layout-newer");
    assert!(!d.path(Kind::Data, "minds", &["m.json"]).unwrap().exists());
    fs::write(d.layout_path(), "garbage").unwrap();
    assert_eq!(code(d.write_file(Kind::Data, "minds", &["m.json"], b"x")), "layout-invalid");
    assert_eq!(fs::read_to_string(d.layout_path()).unwrap(), "garbage");
}

#[test]
fn spawn_area_keeps_data_cache_and_locks_apart() {
    let f = Fixture::new("spawn");
    let d = f.dirs();
    let data = d.path(Kind::Data, "agents/environments", &["claude", "profile.json"]).unwrap();
    let cache = d.path(Kind::Cache, "agents/environments", &["claude", "runtime"]).unwrap();
    let lock = d.lock_path("agents/environments", "claude").unwrap();
    assert!(data.starts_with(d.root(Kind::Data)) && cache.starts_with(d.root(Kind::Cache)) && lock.starts_with(d.root(Kind::Run)));
}

#[test]
fn acquire_release_and_second_acquire_refused() {
    let f = Fixture::new("lock-basic");
    let path = f.0.join("run/platform/x.lock");
    let a = acquire_lock_file(&path, AcquireOptions { owner: "first".into(), ..Default::default() }).unwrap();
    assert!(fs::read_to_string(&path).unwrap().contains("\"owner\":\"first\""));
    assert_eq!(code(acquire_lock_file(&path, AcquireOptions::default())), "lock-held");
    assert!(a.release().unwrap());
    assert!(!path.exists());
}

#[test]
fn dead_holder_is_stale_and_taken_over() {
    let f = Fixture::new("lock-dead");
    let path = f.0.join("run/platform/x.lock");
    write_lock(&path, &rec(dead_pid(), Some("x"), &hostname(), "dead-holder"));
    let h = acquire_lock_file(&path, AcquireOptions::default()).unwrap();
    assert_ne!(h.record.token, "dead-holder");
    assert_eq!(token_in(&path), h.record.token);
}

fn hostname() -> String {
    xenosystem_primitives::dirs::procinfo::hostname()
}

#[test]
fn bare_live_pid_is_not_trusted() {
    let f = Fixture::new("lock-reuse");
    let path = f.0.join("run/platform/x.lock");
    write_lock(&path, &rec(std::process::id(), Some("not-the-start-time-of-this-process"), &hostname(), "reused"));
    let h = acquire_lock_file(&path, AcquireOptions::default()).unwrap();
    assert_ne!(h.record.token, "reused");
}

#[test]
fn matching_start_time_is_held_and_unknown_start_time_is_held() {
    let f = Fixture::new("lock-held");
    let path = f.0.join("run/platform/x.lock");
    let mine = xenosystem_primitives::dirs::procinfo::process_start_time(std::process::id());
    if let Some(m) = &mine {
        write_lock(&path, &rec(std::process::id(), Some(m), &hostname(), "really-held"));
        assert_eq!(code(acquire_lock_file(&path, AcquireOptions::default())), "lock-held");
        assert_eq!(token_in(&path), "really-held");
    }
    let probe = Probe { is_alive: Box::new(|_| true), start_time: Box::new(|_| None), host: Box::new(hostname) };
    assert_eq!(judge_lock(&rec(1, Some("x"), &hostname(), "t"), &probe), LockVerdict::Held);
    assert_eq!(judge_lock(&rec(1, None, &hostname(), "t"), &probe), LockVerdict::Held);
}

#[test]
fn verdict_table() {
    let probe = |alive: bool, now: Option<&'static str>| Probe { is_alive: Box::new(move |_| alive), start_time: Box::new(move |_| now.map(String::from)), host: Box::new(|| "BOX".into()) };
    assert_eq!(judge_lock(&rec(1, Some("a"), "box", "t"), &probe(false, None)), LockVerdict::Stale);
    assert_eq!(judge_lock(&rec(1, Some("a"), "box", "t"), &probe(true, Some("b"))), LockVerdict::Stale);
    assert_eq!(judge_lock(&rec(1, Some("a"), "box", "t"), &probe(true, Some("a"))), LockVerdict::Held);
    assert_eq!(judge_lock(&rec(1, Some("a"), "other", "t"), &probe(false, None)), LockVerdict::ForeignHost);
}

#[test]
fn foreign_host_corrupt_and_takeover_cases() {
    let f = Fixture::new("lock-misc");
    let path = f.0.join("run/platform/x.lock");
    write_lock(&path, &rec(dead_pid(), None, "some-other-machine-entirely", "foreign"));
    assert_eq!(code(acquire_lock_file(&path, AcquireOptions::default())), "lock-held");
    assert_eq!(token_in(&path), "foreign");

    fs::write(&path, "{\"half\":").unwrap();
    assert_eq!(code(acquire_lock_file(&path, AcquireOptions::default())), "lock-corrupt");
    assert_eq!(fs::read_to_string(&path).unwrap(), "{\"half\":");
    fs::remove_file(&path).unwrap();

    let a = acquire_lock_file(&path, AcquireOptions::default()).unwrap();
    let mine = xenosystem_primitives::dirs::procinfo::process_start_time(std::process::id());
    write_lock(&path, &rec(std::process::id(), mine.as_deref(), &hostname(), "someone-else"));
    assert!(!a.release().unwrap());
    assert_eq!(token_in(&path), "someone-else");
}

#[test]
fn timeout_ends_in_lock_timeout() {
    let f = Fixture::new("lock-timeout");
    let path = f.0.join("run/platform/x.lock");
    let a = acquire_lock_file(&path, AcquireOptions::default()).unwrap();
    let t0 = std::time::Instant::now();
    let r = acquire_lock_file(&path, AcquireOptions { timeout: Some(Duration::from_millis(200)), ..Default::default() });
    assert_eq!(code(r), "lock-timeout");
    assert!(t0.elapsed() >= Duration::from_millis(150));
    drop(a);
}

#[test]
fn mutual_exclusion_across_threads() {
    let f = Fixture::new("lock-mutex");
    let path = std::sync::Arc::new(f.0.join("run/platform/x.lock"));
    let inside = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hs: Vec<_> = (0..8)
        .map(|_| {
            let (p, i, m) = (path.clone(), inside.clone(), max.clone());
            std::thread::spawn(move || {
                let h = acquire_lock_file(&p, AcquireOptions { timeout: Some(Duration::from_secs(30)), ..Default::default() }).unwrap();
                let now = i.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                m.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(15));
                i.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                h.release().unwrap();
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    assert_eq!(max.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(!path.exists());
}

// ---- interoperability with the TypeScript implementation (skipped when node is not installed) ----

fn node_available() -> bool {
    Command::new("node").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false)
}

fn ts_lock_module() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/primitives/src/dirs/lock.ts");
    let abs = p.canonicalize().unwrap();
    let s = abs.to_string_lossy().replace('\\', "/");
    let s = s.strip_prefix("//?/").unwrap_or(&s).to_string();
    format!("file:///{}", s.trim_start_matches('/'))
}

#[test]
fn interop_each_implementation_honours_the_others_lock() {
    if !node_available() {
        eprintln!("node not found: interop test skipped");
        return;
    }
    let f = Fixture::new("lock-interop");
    let path = f.0.join("run/platform/x.lock");
    let module = ts_lock_module();

    // 1. TypeScript holds; Rust must see it as held (start-time formats agree) and, once killed, as stale.
    let hold = f.0.join("hold.mjs");
    fs::write(&hold, format!("import {{ acquireLockFile }} from {module:?};\nconst h = await acquireLockFile({:?}, {{ owner: 'ts' }});\nconsole.log('ready');\nsetInterval(() => {{}}, 1000);\n", path.to_string_lossy())).unwrap();
    let mut child = Command::new("node").arg(&hold).stdout(Stdio::piped()).spawn().unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    assert!(line.contains("ready"), "ts holder did not start: {line:?}");
    let held = code(acquire_lock_file(&path, AcquireOptions::default()));
    child.kill().unwrap();
    child.wait().unwrap();
    assert_eq!(held, "lock-held", "Rust must treat a live TypeScript holder as held");
    let h = acquire_lock_file(&path, AcquireOptions::default()).expect("a killed TypeScript holder's lock is stale for Rust");

    // 2. Rust holds; TypeScript must see it as held.
    let probe = f.0.join("probe.mjs");
    fs::write(&probe, format!("import {{ acquireLockFile }} from {module:?};\ntry {{ await acquireLockFile({:?}); console.log('acquired'); }} catch (e) {{ console.log(e.code); }}\n", path.to_string_lossy())).unwrap();
    let out = Command::new("node").arg(&probe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "lock-held", "TypeScript must treat a live Rust holder as held");
    assert!(h.release().unwrap());
}
