//! The copy-only migrator (XENO LOCAL HOME - SPEC.md section 10). It copies known legacy files to their registered
//! locations, records source, destination and SHA-256 in a journal under state/platform/migrations/, verifies every
//! copy by hash, refuses a destination that already holds different bytes and refuses links and junctions, converts
//! plaintext credentials into encrypted secrets through a caller-supplied sink, leaves every legacy home in place,
//! and reports whatever it found but could not place. It never deletes or moves anything and never guesses.
//! Plan ids and journals are byte-compatible with the TypeScript implementation.

use super::atomic::{atomic_write_file, unique_token};
use super::codec::{hex_encode, sha256, Sha256};
use super::error::{DirsError, ErrorCode, Result};
use super::json::{self, Value};
use super::lock::AcquireOptions;
use super::registry::inputs;
use super::roots::{Kind, KINDS};
use super::segments::assert_segment;
use super::xdirs::XenoDirs;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const LEGACY_MAP_JSON: &str = include_str!("../../contract/legacy-map.json");
const SAMPLE: usize = 200;
const WALK_CAP: usize = 500_000;

#[derive(Debug, Clone)]
pub enum RuleAction {
    Copy { kind: Kind, area: String, segments: Vec<String> },
    Secret { area: String, name: String },
}

#[derive(Debug, Clone)]
pub struct MigrationRule {
    pub id: String,
    pub legacy_home: String,
    pub match_pattern: String,
    pub action: RuleAction,
}

/// Rules the contract itself knows (only those the ecosystem documents explicitly). Owners add their own.
pub fn builtin_rules() -> Vec<MigrationRule> {
    let v = json::parse(LEGACY_MAP_JSON).expect("contract/legacy-map.json must parse");
    let s = |o: &Value, k: &str| o.get(k).and_then(Value::as_str).unwrap_or_else(|| panic!("legacy-map: missing {k}")).to_string();
    v.get("rules")
        .and_then(Value::as_array)
        .expect("rules")
        .iter()
        .map(|r| {
            let t = r.get("target").expect("target");
            let action = match s(r, "action").as_str() {
                "copy" => RuleAction::Copy {
                    kind: Kind::parse(&s(t, "kind")).expect("kind"),
                    area: s(t, "area"),
                    segments: t.get("segments").and_then(Value::as_array).expect("segments").iter().map(|x| x.as_str().unwrap().to_string()).collect(),
                },
                "secret" => RuleAction::Secret { area: s(t, "area"), name: s(t, "name") },
                other => panic!("legacy-map: unknown action {other}"),
            };
            MigrationRule { id: s(r, "id"), legacy_home: s(r, "legacyHome"), match_pattern: s(r, "match"), action }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanEntry {
    pub rule: String,
    pub is_secret: bool,
    pub legacy_home: String,
    pub source_rel: String,
    pub source: PathBuf,
    pub bytes: u64,
    pub dest: Option<PathBuf>,
    /// Destination relative to its kind root (copy) or `area/name` (secret).
    pub dest_rel: String,
    pub dest_kind: Option<Kind>,
    pub secret: Option<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HomeState {
    Present,
    Absent,
    RefusedLink,
}

#[derive(Debug, Clone)]
pub struct LegacyHome {
    pub name: String,
    pub path: PathBuf,
    pub state: HomeState,
}

#[derive(Debug, Clone, Default)]
pub struct Unplaced {
    pub count: usize,
    pub sample: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub id: String,
    pub homes: Vec<LegacyHome>,
    pub entries: Vec<PlanEntry>,
    pub refused: Vec<Refusal>,
    pub unplaced: Unplaced,
}

fn glob_match(pattern: &str, name: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == name;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut rest = name;
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            if !rest.starts_with(part) {
                return false;
            }
            rest = &rest[part.len()..];
        } else if i == parts.len() - 1 {
            return rest.ends_with(part) && rest.len() >= part.len();
        } else {
            match rest.find(part) {
                Some(p) => rest = &rest[p + part.len()..],
                None => return false,
            }
        }
    }
    true
}

fn match_rule<'a>(rule: &MigrationRule, rel: &'a str) -> Option<&'a str> {
    let want: Vec<&str> = rule.match_pattern.split('/').collect();
    let got: Vec<&str> = rel.split('/').collect();
    if want.len() != got.len() {
        return None;
    }
    if want[..want.len() - 1] != got[..got.len() - 1] {
        return None;
    }
    let last = got[got.len() - 1];
    if glob_match(want[want.len() - 1], last) { Some(last) } else { None }
}

/// Deterministic id: the same plan (rules, sources, destinations) always has the same journal.
pub fn plan_id(entries: &[PlanEntry]) -> String {
    let mut lines: Vec<String> = entries.iter().map(|e| format!("{}\t{}\t{}\t{}", e.rule, e.legacy_home, e.source_rel, e.dest_rel)).collect();
    lines.sort();
    format!("m-{}", &hex_encode(&sha256(lines.join("\n").as_bytes()))[..12])
}

fn is_link(p: &Path) -> bool {
    fs::symlink_metadata(p).map(|m| m.file_type().is_symlink()).unwrap_or(false)
}

fn cmp_key(a: &PlanEntry) -> String {
    format!("{}{}", a.legacy_home, a.source_rel)
}

pub fn plan_migration(dirs: &XenoDirs, home: &str, extra_rules: &[MigrationRule]) -> Result<MigrationPlan> {
    let mut rules = builtin_rules();
    rules.extend(extra_rules.iter().cloned());
    for (i, r) in rules.iter().enumerate() {
        if rules[..i].iter().any(|o| o.id == r.id) {
            return Err(DirsError::new(ErrorCode::InvalidArgument, format!("duplicate migration rule id: {}", r.id)));
        }
    }
    let sep = dirs.platform().sep();
    let mut homes = Vec::new();
    let mut entries: Vec<PlanEntry> = Vec::new();
    let mut refused = Vec::new();
    let mut unplaced = Unplaced::default();
    let mut visited = 0usize;

    for name in &inputs().legacy_homes {
        let path = PathBuf::from(format!("{home}{sep}{name}"));
        let md = match fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => {
                homes.push(LegacyHome { name: name.clone(), path, state: HomeState::Absent });
                continue;
            }
        };
        if md.file_type().is_symlink() {
            refused.push(Refusal { path: path.display().to_string(), reason: "the legacy home is a link or junction".into() });
            homes.push(LegacyHome { name: name.clone(), path, state: HomeState::RefusedLink });
            continue;
        }
        if !md.is_dir() {
            homes.push(LegacyHome { name: name.clone(), path, state: HomeState::Absent });
            continue;
        }
        homes.push(LegacyHome { name: name.clone(), path: path.clone(), state: HomeState::Present });
        let mine: Vec<&MigrationRule> = rules.iter().filter(|r| &r.legacy_home == name).collect();
        let mut stack: Vec<(PathBuf, String)> = vec![(path.clone(), String::new())];
        while let Some((dir, rel)) = stack.pop() {
            let list = match fs::read_dir(&dir) {
                Ok(l) => l,
                Err(_) => {
                    refused.push(Refusal { path: dir.display().to_string(), reason: "unreadable directory".into() });
                    continue;
                }
            };
            let mut names: Vec<String> = list.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
            names.sort();
            for entry in names {
                visited += 1;
                if visited > WALK_CAP {
                    unplaced.truncated = true;
                    break;
                }
                let abs = dir.join(&entry);
                let r = if rel.is_empty() { entry.clone() } else { format!("{rel}/{entry}") };
                let Ok(md) = fs::symlink_metadata(&abs) else { continue };
                let hits: Vec<(&MigrationRule, &str)> = mine.iter().filter_map(|rule| match_rule(rule, &r).map(|f| (*rule, f))).collect();
                let hits: Vec<(&MigrationRule, String)> = hits.into_iter().map(|(a, b)| (a, b.to_string())).collect();
                if md.file_type().is_symlink() {
                    if !hits.is_empty() {
                        refused.push(Refusal { path: abs.display().to_string(), reason: "a link or junction is never followed".into() });
                    } else {
                        unplaced.count += 1;
                        if unplaced.sample.len() < SAMPLE {
                            unplaced.sample.push(format!("{name}/{r} (link)"));
                        }
                    }
                    continue;
                }
                if md.is_dir() {
                    stack.push((abs, r));
                    continue;
                }
                if hits.is_empty() {
                    unplaced.count += 1;
                    if unplaced.sample.len() < SAMPLE {
                        unplaced.sample.push(format!("{name}/{r}"));
                    }
                    continue;
                }
                if hits.len() > 1 {
                    refused.push(Refusal { path: abs.display().to_string(), reason: format!("ambiguous: rules {} all match", hits.iter().map(|h| h.0.id.as_str()).collect::<Vec<_>>().join(", ")) });
                    continue;
                }
                let (rule, file) = &hits[0];
                let fill = |s: &str| s.replace("{file}", file);
                let built: Result<PlanEntry> = match &rule.action {
                    RuleAction::Copy { kind, area, segments } => {
                        let segs: Vec<String> = segments.iter().map(|s| fill(s)).collect();
                        let refs: Vec<&str> = segs.iter().map(String::as_str).collect();
                        dirs.path(*kind, area, &refs).map(|dest| PlanEntry {
                            rule: rule.id.clone(),
                            is_secret: false,
                            legacy_home: name.clone(),
                            source_rel: r.clone(),
                            source: abs.clone(),
                            bytes: md.len(),
                            dest: Some(dest),
                            dest_rel: std::iter::once(area.as_str()).chain(refs.iter().copied()).collect::<Vec<_>>().join("/"),
                            dest_kind: Some(*kind),
                            secret: None,
                        })
                    }
                    RuleAction::Secret { area, name: sname } => {
                        let sn = fill(sname);
                        assert_segment(&sn, "secret name").and_then(|_| dirs.path(Kind::Secrets, area, &[&format!("{sn}.secret")])).map(|_| PlanEntry {
                            rule: rule.id.clone(),
                            is_secret: true,
                            legacy_home: name.clone(),
                            source_rel: r.clone(),
                            source: abs.clone(),
                            bytes: md.len(),
                            dest: None,
                            dest_rel: format!("{area}/{sn}"),
                            dest_kind: None,
                            secret: Some((area.clone(), sn.clone())),
                        })
                    }
                };
                match built {
                    Ok(e) => entries.push(e),
                    Err(e) => refused.push(Refusal { path: abs.display().to_string(), reason: format!("rule {}: {}", rule.id, e.message) }),
                }
            }
        }
    }

    // Two sources must never map to one destination.
    let key = |e: &PlanEntry| format!("{}:{}", e.dest_kind.map(|k| k.as_str()).unwrap_or("secret"), e.dest_rel);
    let mut clean: Vec<PlanEntry> = Vec::new();
    for e in &entries {
        let group = entries.iter().filter(|o| key(o) == key(e)).count();
        if group == 1 {
            clean.push(e.clone());
        } else {
            refused.push(Refusal { path: e.source.display().to_string(), reason: format!("destination collision: {group} sources map to {}", e.dest_rel) });
        }
    }
    clean.sort_by_key(cmp_key);
    Ok(MigrationPlan { id: plan_id(&clean), homes, entries: clean, refused, unplaced })
}

// ---- running ------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryStatus {
    Copied,
    Already,
    Secret,
    Conflict,
    Refused,
    Failed,
}

impl EntryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            EntryStatus::Copied => "copied",
            EntryStatus::Already => "already",
            EntryStatus::Secret => "secret",
            EntryStatus::Conflict => "conflict",
            EntryStatus::Refused => "refused",
            EntryStatus::Failed => "failed",
        }
    }
    fn good(self) -> bool {
        matches!(self, EntryStatus::Copied | EntryStatus::Already | EntryStatus::Secret)
    }
}

#[derive(Debug, Clone)]
pub struct EntryResult {
    pub rule: String,
    pub legacy_home: String,
    pub source_rel: String,
    pub dest: Option<PathBuf>,
    pub dest_rel: String,
    pub status: EntryStatus,
    pub bytes: u64,
    /// SHA-256 of the file's bytes. Never recorded for a secret.
    pub sha256: Option<String>,
    pub reason: Option<String>,
}

/// Where plaintext credentials are converted into encrypted secrets (`secrets::SecretStore` implements it).
/// Errors are messages only; they must never contain a value.
pub trait SecretSink {
    fn set(&self, area: &str, name: &str, value: &[u8]) -> std::result::Result<(), String>;
    fn get(&self, area: &str, name: &str) -> std::result::Result<Option<Vec<u8>>, String>;
}

#[derive(Default)]
pub struct RunOptions<'a> {
    /// Nothing is written unless this is true (dry run by default).
    pub confirm: bool,
    pub secrets: Option<&'a dyn SecretSink>,
    /// Write XENO-MIGRATED.txt into each legacy home. Default true; set `skip_note` to turn it off.
    pub skip_note: bool,
}

#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub id: String,
    pub confirmed: bool,
    /// True only when every planned entry is copied/already/secret-verified and nothing was refused.
    pub ok: bool,
    pub entries: Vec<EntryResult>,
    pub refused: Vec<Refusal>,
    pub unplaced: Unplaced,
    pub journal: Option<PathBuf>,
    pub plaintext_remains: Vec<PathBuf>,
}

fn hash_file(p: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex_encode(&h.finish()))
}

fn refuse(msg: &str) -> DirsError {
    DirsError::new(ErrorCode::MigrationRefused, msg)
}

fn assert_no_links_below(root: &Path, parts: &[&str]) -> Result<()> {
    let mut cur = root.to_path_buf();
    for p in parts {
        cur.push(p);
        match fs::symlink_metadata(&cur) {
            Ok(m) if m.file_type().is_symlink() => return Err(refuse("a link or junction on the path is never followed")),
            Ok(_) => {}
            Err(_) => return Ok(()),
        }
    }
    Ok(())
}

enum Copied {
    Done { sha256: String, bytes: u64 },
    Conflict,
}

fn copy_verified(source: &Path, dest: &Path) -> Result<Copied> {
    fs::create_dir_all(dest.parent().unwrap())?;
    let tmp = dest.parent().unwrap().join(format!(".{}.{}.{}.migrating.tmp", dest.file_name().unwrap().to_string_lossy(), std::process::id(), unique_token()));
    let result = (|| -> Result<Copied> {
        let mut h = Sha256::new();
        let mut bytes = 0u64;
        {
            let mut src = fs::File::open(source)?;
            let mut out = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = src.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
                out.write_all(&buf[..n])?;
                bytes += n as u64;
            }
            out.sync_all()?;
        }
        let sha = hex_encode(&h.finish());
        match fs::hard_link(&tmp, dest) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Ok(Copied::Conflict),
            Err(_) => {
                // Filesystem without hard links: a no-clobber rename after re-checking the destination is still absent.
                if fs::symlink_metadata(dest).is_ok() {
                    return Ok(Copied::Conflict);
                }
                fs::rename(&tmp, dest)?;
            }
        }
        if hash_file(dest)? != sha {
            let _ = fs::remove_file(dest);
            return Err(refuse("the copy did not verify by hash"));
        }
        if hash_file(source)? != sha {
            let _ = fs::remove_file(dest);
            return Err(refuse("the source changed while it was being copied"));
        }
        Ok(Copied::Done { sha256: sha, bytes })
    })();
    let _ = fs::remove_file(&tmp);
    result
}

pub fn journal_path(dirs: &XenoDirs, id: &str) -> Result<PathBuf> {
    dirs.path(Kind::State, "platform", &["migrations", &format!("{id}.jsonl")])
}

fn append_journal(path: &Path, record: Vec<(&str, Value)>) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = json::to_string(&Value::Obj(record.into_iter().map(|(k, v)| (k.to_string(), v)).collect())) + "\n";
    f.write_all(line.as_bytes())?;
    f.sync_all()?;
    Ok(())
}

pub fn read_journal(path: &Path) -> Result<Vec<Value>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e.into()),
    };
    let lines: Vec<&str> = text.split('\n').filter(|l| !l.is_empty()).collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        match json::parse(l) {
            Ok(v) => out.push(v),
            // A torn last line is tolerated.
            Err(_) if i == lines.len() - 1 => {}
            Err(_) => return Err(refuse("the migration journal is damaged before its last line")),
        }
    }
    Ok(out)
}

fn s(x: &str) -> Value {
    Value::Str(x.to_string())
}

fn now_iso() -> String {
    // Same civil-from-days conversion as the lock records.
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
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

pub fn run_migration(plan: &MigrationPlan, dirs: &XenoDirs, options: RunOptions<'_>) -> Result<MigrationResult> {
    let mut base = MigrationResult {
        id: plan.id.clone(),
        confirmed: options.confirm,
        ok: false,
        entries: vec![],
        refused: plan.refused.clone(),
        unplaced: plan.unplaced.clone(),
        journal: None,
        plaintext_remains: vec![],
    };
    if !options.confirm {
        base.entries = plan
            .entries
            .iter()
            .map(|e| EntryResult {
                rule: e.rule.clone(),
                legacy_home: e.legacy_home.clone(),
                source_rel: e.source_rel.clone(),
                dest: e.dest.clone(),
                dest_rel: e.dest_rel.clone(),
                status: EntryStatus::Refused,
                bytes: e.bytes,
                sha256: None,
                reason: Some("dry run: pass confirm to migrate".into()),
            })
            .collect();
        return Ok(base);
    }
    dirs.ensure_layout()?;
    let journal = journal_path(dirs, &plan.id)?;
    base.journal = Some(journal.clone());
    let lock = dirs.acquire_lock("platform", "migrate", AcquireOptions { owner: "migration".into(), timeout: Some(Duration::from_secs(60)), ..Default::default() })?;
    let outcome = (|| -> Result<()> {
        let prior = read_journal(&journal)?;
        append_journal(&journal, vec![
            ("op", s(if prior.is_empty() { "begin" } else { "resume" })),
            ("id", s(&plan.id)),
            ("at", s(&now_iso())),
            ("entries", Value::Num(plan.entries.len() as f64)),
            ("refused", Value::Num(plan.refused.len() as f64)),
        ])?;
        let mut results: Vec<EntryResult> = Vec::new();
        for e in &plan.entries {
            let mut r = EntryResult { rule: e.rule.clone(), legacy_home: e.legacy_home.clone(), source_rel: e.source_rel.clone(), dest: e.dest.clone(), dest_rel: e.dest_rel.clone(), status: EntryStatus::Failed, bytes: e.bytes, sha256: None, reason: None };
            let step = (|| -> Result<()> {
                let home = plan.homes.iter().find(|h| h.name == e.legacy_home).expect("home in plan");
                let rel_parts: Vec<&str> = e.source_rel.split('/').collect();
                assert_no_links_below(&home.path, &rel_parts)?;
                if is_link(&home.path) {
                    return Err(refuse("the legacy home is a link or junction"));
                }
                if !fs::symlink_metadata(&e.source)?.is_file() {
                    return Err(refuse("the source is not a regular file"));
                }
                if !e.is_secret {
                    let dest = e.dest.as_ref().unwrap();
                    let kind = e.dest_kind.unwrap();
                    let dparts: Vec<&str> = e.dest_rel.split('/').collect();
                    assert_no_links_below(Path::new(dirs.root(kind)), &dparts)?;
                    let src_hash = hash_file(&e.source)?;
                    let existing = match fs::symlink_metadata(dest) {
                        Ok(m) if m.is_file() => Some(hash_file(dest)?),
                        Ok(_) => return Err(refuse("the destination exists and is not a file")),
                        Err(_) => None,
                    };
                    match existing {
                        Some(h) if h == src_hash => {
                            r.status = EntryStatus::Already;
                            r.sha256 = Some(src_hash);
                        }
                        Some(_) => {
                            r.status = EntryStatus::Conflict;
                            r.reason = Some("the destination already holds different bytes; it is not overwritten".into());
                        }
                        None => match copy_verified(&e.source, dest)? {
                            Copied::Conflict => {
                                r.status = EntryStatus::Conflict;
                                r.reason = Some("the destination appeared with different bytes; it is not overwritten".into());
                            }
                            Copied::Done { sha256, bytes } => {
                                r.status = EntryStatus::Copied;
                                r.sha256 = Some(sha256);
                                r.bytes = bytes;
                            }
                        },
                    }
                } else {
                    match options.secrets {
                        None => {
                            r.status = EntryStatus::Refused;
                            r.reason = Some("no secret sink supplied; the plaintext file is left as it is".into());
                        }
                        Some(sink) => {
                            let (area, name) = e.secret.as_ref().unwrap();
                            let mut bytes = fs::read(&e.source)?;
                            let map = |m: String| DirsError::new(ErrorCode::MigrationRefused, m);
                            let existing = sink.get(area, name).map_err(map)?;
                            match &existing {
                                Some(x) if x != &bytes => {
                                    r.status = EntryStatus::Conflict;
                                    r.reason = Some("a different secret already exists at the destination; it is not overwritten".into());
                                }
                                _ => {
                                    if existing.is_none() {
                                        sink.set(area, name, &bytes).map_err(map)?;
                                    }
                                    let back = sink.get(area, name).map_err(map)?;
                                    if back.as_ref() != Some(&bytes) {
                                        return Err(refuse("the secret did not verify after it was stored"));
                                    }
                                    r.status = if existing.is_none() { EntryStatus::Secret } else { EntryStatus::Already };
                                }
                            }
                            bytes.fill(0);
                        }
                    }
                }
                Ok(())
            })();
            if let Err(err) = step {
                r.status = if err.code == ErrorCode::MigrationRefused { EntryStatus::Refused } else { EntryStatus::Failed };
                r.reason = Some(err.message);
            }
            let mut rec: Vec<(&str, Value)> = vec![
                ("op", s(r.status.as_str())),
                ("rule", s(&r.rule)),
                ("home", s(&r.legacy_home)),
                ("source", s(&r.source_rel)),
                ("dest", s(&r.dest_rel)),
                ("destPath", r.dest.as_ref().map(|d| s(&d.to_string_lossy())).unwrap_or(Value::Null)),
                ("bytes", Value::Num(r.bytes as f64)),
            ];
            if let Some(h) = &r.sha256 {
                rec.push(("sha256", s(h)));
            }
            if let Some(why) = &r.reason {
                rec.push(("reason", s(why)));
            }
            append_journal(&journal, rec)?;
            results.push(r);
        }
        let ok = results.iter().all(|r| r.status.good()) && plan.refused.is_empty();
        let mut counts: Vec<(String, Value)> = Vec::new();
        for r in &results {
            match counts.iter_mut().find(|(k, _)| k == r.status.as_str()) {
                Some((_, Value::Num(n))) => *n += 1.0,
                _ => counts.push((r.status.as_str().to_string(), Value::Num(1.0))),
            }
        }
        append_journal(&journal, vec![
            ("op", s("end")),
            ("ok", Value::Bool(ok)),
            ("at", s(&now_iso())),
            ("counts", Value::Obj(counts)),
            ("unplaced", Value::Num(plan.unplaced.count as f64)),
        ])?;
        if !options.skip_note {
            for h in plan.homes.iter().filter(|h| h.state == HomeState::Present) {
                let mine: Vec<&EntryResult> = results.iter().filter(|r| r.legacy_home == h.name && r.status.good()).collect();
                let roots = KINDS.iter().map(|k| format!("{}={}", k.as_str(), dirs.root(*k))).collect::<Vec<_>>().join("; ");
                let mut note = vec![
                    "XENO has copied what it recognised from this folder to the new XENO layout.".to_string(),
                    "Nothing here was deleted or moved. Removing this folder is a step only you confirm.".to_string(),
                    String::new(),
                    format!("Journal: {}", journal.display()),
                    format!("Layout roots: {roots}"),
                    String::new(),
                ];
                if mine.is_empty() {
                    note.push("Nothing from this folder was copied.".into());
                } else {
                    note.push("Copied or already present:".into());
                    note.extend(mine.iter().map(|r| format!("  {} -> {}", r.source_rel, r.dest_rel)));
                }
                note.push(String::new());
                let _ = atomic_write_file(&h.path.join("XENO-MIGRATED.txt"), note.join("\n").as_bytes(), None);
            }
        }
        base.plaintext_remains = plan
            .entries
            .iter()
            .filter(|e| e.is_secret && results.iter().any(|r| r.source_rel == e.source_rel && r.legacy_home == e.legacy_home && matches!(r.status, EntryStatus::Secret | EntryStatus::Already)))
            .map(|e| e.source.clone())
            .collect();
        base.ok = ok;
        base.entries = results;
        Ok(())
    })();
    let _ = lock.release();
    outcome?;
    Ok(base)
}

// ---- verification -------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub id: String,
    pub complete: bool,
    pub checked: usize,
    pub problems: Vec<String>,
}

/// Re-checks a finished migration against its journal: every copied destination still exists and still hashes to the
/// recorded SHA-256. A product switches to the new location only when `complete` is true.
pub fn verify_migration(dirs: &XenoDirs, id: &str) -> Result<VerifyResult> {
    let records = read_journal(&journal_path(dirs, id)?)?;
    let mut problems = Vec::new();
    let mut checked = 0;
    if records.is_empty() {
        return Ok(VerifyResult { id: id.into(), complete: false, checked, problems: vec!["no journal".into()] });
    }
    let last_end = records.iter().rev().find(|r| r.get("op").and_then(Value::as_str) == Some("end"));
    match last_end {
        None => problems.push("the migration never finished".into()),
        Some(e) if e.get("ok").and_then(Value::as_bool) != Some(true) => problems.push("the last run did not finish cleanly".into()),
        _ => {}
    }
    let mut latest: Vec<(String, &Value)> = Vec::new();
    for r in &records {
        if let (Some(_), Some(src)) = (r.get("dest").and_then(Value::as_str), r.get("source").and_then(Value::as_str)) {
            let key = format!("{}:{}", r.get("home").and_then(Value::as_str).unwrap_or(""), src);
            match latest.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = r,
                None => latest.push((key, r)),
            }
        }
    }
    for (_, r) in latest {
        let op = r.get("op").and_then(Value::as_str).unwrap_or("");
        let dest_path = r.get("destPath");
        let dest_rel = r.get("dest").and_then(Value::as_str).unwrap_or("");
        if (op == "copied" || op == "already") && !matches!(dest_path, Some(Value::Null)) {
            checked += 1;
            match dest_path.and_then(Value::as_str) {
                None => problems.push(format!("{dest_rel}: journal has no destination path")),
                Some(p) => match hash_file(Path::new(p)) {
                    Ok(h) if Some(h.as_str()) == r.get("sha256").and_then(Value::as_str) => {}
                    Ok(_) => problems.push(format!("{dest_rel}: bytes differ from the journal")),
                    Err(_) => problems.push(format!("{dest_rel}: missing")),
                },
            }
        } else if op != "secret" && op != "already" {
            problems.push(format!("{}: {op}", r.get("source").and_then(Value::as_str).unwrap_or("")));
        }
    }
    Ok(VerifyResult { id: id.into(), complete: problems.is_empty(), checked, problems })
}

