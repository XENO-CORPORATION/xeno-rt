//! The migrator and `doctor`, plus interoperability with the TypeScript implementation (plan ids, journals).
//! Fixtures live in a test-owned temp root. Link tests create a junction/symlink there; the fixture's Drop lists
//! reparse points and unlinks each with `rmdir` BEFORE any recursive delete.
#![cfg(feature = "dirs")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use xenosystem_primitives::dirs::codec::{hex_encode, sha256};
use xenosystem_primitives::dirs::migrate::HomeState;
use xenosystem_primitives::dirs::*;

struct World {
    root: PathBuf,
    home: String,
    dirs: XenoDirs,
}

fn unlink_links(dir: &Path) {
    if let Ok(it) = fs::read_dir(dir) {
        for e in it.flatten() {
            let p = e.path();
            let Ok(m) = fs::symlink_metadata(&p) else { continue };
            if m.file_type().is_symlink() {
                if cfg!(windows) {
                    let _ = Command::new("cmd").args(["/c", "rmdir"]).arg(&p).output();
                } else {
                    let _ = fs::remove_file(&p);
                }
            } else if m.is_dir() {
                unlink_links(&p);
            }
        }
    }
}

impl Drop for World {
    fn drop(&mut self) {
        unlink_links(&self.root);
        let mut left = false;
        fn scan(d: &Path, left: &mut bool) {
            if let Ok(it) = fs::read_dir(d) {
                for e in it.flatten() {
                    if let Ok(m) = fs::symlink_metadata(e.path()) {
                        if m.file_type().is_symlink() {
                            *left = true;
                        } else if m.is_dir() {
                            scan(&e.path(), left);
                        }
                    }
                }
            }
        }
        scan(&self.root, &mut left);
        if !left {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

fn put(home: &str, rel: &str, body: &str) {
    let p = Path::new(home).join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

fn world(label: &str) -> World {
    let root = std::env::temp_dir().join(format!("xeno-prim-{label}-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir_all(&root).unwrap();
    let home = root.join("person").to_string_lossy().to_string();
    put(&home, ".xeno/shell/state.json", "{\"window\":\"main\"}");
    put(&home, ".xeno/random-note.txt", "not placed by any rule");
    put(&home, ".xeno-agent/sessions/one.txt", "session one");
    put(&home, ".xeno-agent/sessions/two.txt", "session two");
    put(&home, ".xeno-anima/secret.key", "fixture-plaintext-credential");
    let dirs = create_dirs_at(&root.join("xeno")).unwrap();
    World { root, home, dirs }
}

fn rules() -> Vec<MigrationRule> {
    vec![
        MigrationRule { id: "agent-sessions".into(), legacy_home: ".xeno-agent".into(), match_pattern: "sessions/*.txt".into(), action: RuleAction::Copy { kind: Kind::State, area: "agents/sessions".into(), segments: vec!["{file}".into()] } },
        MigrationRule { id: "soul-key".into(), legacy_home: ".xeno-anima".into(), match_pattern: "secret.key".into(), action: RuleAction::Secret { area: "minds".into(), name: "soul-key".into() } },
    ]
}

fn sha(s: &[u8]) -> String {
    hex_encode(&sha256(s))
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

fn confirm() -> RunOptions<'static> {
    RunOptions { confirm: true, skip_note: true, secrets: None }
}

struct MemSink(std::sync::Mutex<HashMap<String, Vec<u8>>>);
impl SecretSink for MemSink {
    fn set(&self, a: &str, n: &str, v: &[u8]) -> std::result::Result<(), String> {
        self.0.lock().unwrap().insert(format!("{a}/{n}"), v.to_vec());
        Ok(())
    }
    fn get(&self, a: &str, n: &str) -> std::result::Result<Option<Vec<u8>>, String> {
        Ok(self.0.lock().unwrap().get(&format!("{a}/{n}")).cloned())
    }
}

#[test]
fn plan_matches_rules_reports_the_rest_and_is_deterministic() {
    let w = world("m-plan");
    let plan = plan_migration(&w.dirs, &w.home, &rules()).unwrap();
    let mut got: Vec<String> = plan.entries.iter().map(|e| format!("{}:{}", e.rule, e.source_rel)).collect();
    got.sort();
    assert_eq!(got, ["agent-sessions:sessions/one.txt", "agent-sessions:sessions/two.txt", "shell-window-state:shell/state.json", "soul-key:secret.key"]);
    assert_eq!(plan.unplaced.count, 1);
    assert_eq!(plan.unplaced.sample, [".xeno/random-note.txt"]);
    assert_eq!(plan_migration(&w.dirs, &w.home, &rules()).unwrap().id, plan.id);
    put(&w.home, ".xeno-agent/sessions/three.txt", "x");
    assert_ne!(plan_migration(&w.dirs, &w.home, &rules()).unwrap().id, plan.id);
}

#[test]
fn dry_run_writes_nothing() {
    let w = world("m-dry");
    let plan = plan_migration(&w.dirs, &w.home, &rules()).unwrap();
    let before = files(&w.root);
    let r = run_migration(&plan, &w.dirs, RunOptions::default()).unwrap();
    assert!(!r.confirmed && !r.ok);
    assert_eq!(files(&w.root), before);
    assert!(!w.root.join("xeno").exists());
}

#[test]
fn confirmed_run_copies_journals_verifies_and_leaves_legacy_byte_identical() {
    let w = world("m-run");
    let before: Vec<(String, String)> = files(Path::new(&w.home)).into_iter().map(|f| (f.clone(), sha(&fs::read(Path::new(&w.home).join(&f)).unwrap()))).collect();
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    let r = run_migration(&plan, &w.dirs, RunOptions { confirm: true, skip_note: false, secrets: None }).unwrap();
    assert!(r.ok, "{:?}", r.entries);
    assert!(r.entries.iter().all(|e| e.status == EntryStatus::Copied));
    assert_eq!(fs::read_to_string(w.dirs.path(Kind::State, "agents/sessions", &["one.txt"]).unwrap()).unwrap(), "session one");
    let j = read_journal(&journal_path(&w.dirs, &plan.id).unwrap()).unwrap();
    assert_eq!(j.first().unwrap().get("op").and_then(|v| v.as_str()), Some("begin"));
    assert_eq!(j.last().unwrap().get("op").and_then(|v| v.as_str()), Some("end"));
    let one = j.iter().find(|r| r.get("source").and_then(|v| v.as_str()) == Some("sessions/one.txt")).unwrap();
    assert_eq!(one.get("sha256").and_then(|v| v.as_str()), Some(sha(b"session one").as_str()));
    assert!(verify_migration(&w.dirs, &plan.id).unwrap().complete);
    for (f, h) in &before {
        assert_eq!(&sha(&fs::read(Path::new(&w.home).join(f)).unwrap()), h, "{f}");
    }
    let now = files(Path::new(&w.home));
    let added: Vec<&String> = now.iter().filter(|f| !before.iter().any(|(b, _)| b == *f)).collect();
    assert_eq!(added, [".xeno-agent/XENO-MIGRATED.txt", ".xeno-anima/XENO-MIGRATED.txt", ".xeno/XENO-MIGRATED.txt"]);
    assert_eq!(r.unplaced.count, 2);
}

#[test]
fn second_run_is_idempotent_and_a_lost_destination_is_recopied() {
    let w = world("m-resume");
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    run_migration(&plan, &w.dirs, confirm()).unwrap();
    let again = run_migration(&plan, &w.dirs, confirm()).unwrap();
    assert!(again.ok && again.entries.iter().all(|e| e.status == EntryStatus::Already));
    fs::remove_dir_all(w.dirs.path(Kind::State, "agents/sessions", &[]).unwrap()).unwrap();
    let healed = run_migration(&plan, &w.dirs, confirm()).unwrap();
    let mut st: Vec<&str> = healed.entries.iter().map(|e| e.status.as_str()).collect();
    st.sort();
    assert_eq!(st, ["already", "copied", "copied"]);
    let ops: Vec<String> = read_journal(&journal_path(&w.dirs, &plan.id).unwrap()).unwrap().iter().map(|r| r.get("op").and_then(|v| v.as_str()).unwrap().to_string()).collect();
    assert_eq!(ops.iter().filter(|o| *o == "resume").count(), 2);
}

#[test]
fn different_bytes_at_the_destination_are_never_overwritten() {
    let w = world("m-conflict");
    let dest = w.dirs.path(Kind::State, "agents/sessions", &["one.txt"]).unwrap();
    fs::create_dir_all(dest.parent().unwrap()).unwrap();
    fs::write(&dest, "someone else wrote this").unwrap();
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    let r = run_migration(&plan, &w.dirs, confirm()).unwrap();
    assert!(!r.ok);
    assert_eq!(r.entries.iter().find(|e| e.source_rel == "sessions/one.txt").unwrap().status, EntryStatus::Conflict);
    assert_eq!(fs::read_to_string(&dest).unwrap(), "someone else wrote this");
    assert_eq!(r.entries.iter().find(|e| e.source_rel == "sessions/two.txt").unwrap().status, EntryStatus::Copied);
    assert!(!verify_migration(&w.dirs, &plan.id).unwrap().complete);
}

#[test]
fn verification_catches_a_destination_changed_afterwards() {
    let w = world("m-tamper");
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    run_migration(&plan, &w.dirs, confirm()).unwrap();
    fs::write(w.dirs.path(Kind::State, "agents/sessions", &["two.txt"]).unwrap(), "changed afterwards").unwrap();
    let v = verify_migration(&w.dirs, &plan.id).unwrap();
    assert!(!v.complete && v.problems.join("\n").contains("two.txt: bytes differ"));
}

#[test]
fn secrets_go_through_the_sink_and_never_into_the_journal() {
    let w = world("m-secret");
    let plan = plan_migration(&w.dirs, &w.home, &rules()).unwrap();
    let sink = MemSink(Default::default());
    let r = run_migration(&plan, &w.dirs, RunOptions { confirm: true, skip_note: true, secrets: Some(&sink) }).unwrap();
    assert!(r.ok);
    assert_eq!(sink.0.lock().unwrap().get("minds/soul-key").unwrap(), b"fixture-plaintext-credential");
    let text = fs::read_to_string(journal_path(&w.dirs, &plan.id).unwrap()).unwrap();
    assert!(!text.contains("fixture-plaintext-credential"));
    assert!(!text.contains(&sha(b"fixture-plaintext-credential")), "a hash of a low-entropy secret is itself a leak");
    assert_eq!(r.plaintext_remains, [Path::new(&w.home).join(".xeno-anima").join("secret.key")]);
    assert_eq!(fs::read_to_string(Path::new(&w.home).join(".xeno-anima/secret.key")).unwrap(), "fixture-plaintext-credential");
}

#[test]
fn secrets_without_a_sink_or_with_a_different_existing_value_are_not_migrated() {
    let w = world("m-secret2");
    let plan = plan_migration(&w.dirs, &w.home, &rules()[1..]).unwrap();
    let r = run_migration(&plan, &w.dirs, confirm()).unwrap();
    assert_eq!(r.entries.iter().find(|e| e.rule == "soul-key").unwrap().status, EntryStatus::Refused);
    assert!(!r.ok);
    let sink = MemSink(Default::default());
    sink.set("minds", "soul-key", b"already-there").unwrap();
    let r2 = run_migration(&plan, &w.dirs, RunOptions { confirm: true, skip_note: true, secrets: Some(&sink) }).unwrap();
    assert_eq!(r2.entries.iter().find(|e| e.rule == "soul-key").unwrap().status, EntryStatus::Conflict);
    assert_eq!(sink.get("minds", "soul-key").unwrap().unwrap(), b"already-there");
}

#[test]
fn ambiguous_rules_collisions_and_bad_targets_are_refused() {
    let w = world("m-refuse");
    let mut dup = rules()[..1].to_vec();
    dup.push(MigrationRule { id: "also".into(), legacy_home: ".xeno-agent".into(), match_pattern: "sessions/one.txt".into(), action: RuleAction::Copy { kind: Kind::State, area: "agents/sessions".into(), segments: vec!["other.txt".into()] } });
    let plan = plan_migration(&w.dirs, &w.home, &dup).unwrap();
    assert!(plan.refused.iter().any(|r| r.reason.contains("ambiguous") && r.path.ends_with("one.txt")));
    let collide = vec![MigrationRule { id: "all".into(), legacy_home: ".xeno-agent".into(), match_pattern: "sessions/*.txt".into(), action: RuleAction::Copy { kind: Kind::State, area: "agents/sessions".into(), segments: vec!["same.txt".into()] } }];
    let p2 = plan_migration(&w.dirs, &w.home, &collide).unwrap();
    assert_eq!(p2.refused.iter().filter(|r| r.reason.contains("collision")).count(), 2);
    let bad = vec![MigrationRule { id: "bad".into(), legacy_home: ".xeno-agent".into(), match_pattern: "sessions/*.txt".into(), action: RuleAction::Copy { kind: Kind::Data, area: "scratch".into(), segments: vec!["{file}".into()] } }];
    let p3 = plan_migration(&w.dirs, &w.home, &bad).unwrap();
    assert_eq!(p3.refused.len(), 2);
    assert!(p3.refused[0].reason.contains("not registered"));
    assert!(plan_migration(&w.dirs, &w.home, &[rules()[0].clone(), rules()[0].clone()]).is_err());
}

#[test]
fn torn_last_journal_line_is_tolerated_damage_earlier_is_not() {
    let w = world("m-journal");
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    run_migration(&plan, &w.dirs, confirm()).unwrap();
    let jp = journal_path(&w.dirs, &plan.id).unwrap();
    let mut f = fs::OpenOptions::new().append(true).open(&jp).unwrap();
    std::io::Write::write_all(&mut f, b"{\"op\":\"copied\",\"rul").unwrap();
    assert!(read_journal(&jp).unwrap().len() > 3);
    let text = fs::read_to_string(&jp).unwrap();
    let mut lines: Vec<&str> = text.split('\n').collect();
    lines.insert(1, "garbage in the middle");
    fs::write(&jp, lines.join("\n")).unwrap();
    assert!(read_journal(&jp).is_err());
}

// ---- links and junctions ----

#[cfg(windows)]
fn mk_link(target: &Path, link: &Path) {
    let o = Command::new("cmd").args(["/c", "mklink", "/J"]).arg(link).arg(target).output().unwrap();
    assert!(o.status.success(), "mklink failed");
}

#[cfg(unix)]
fn mk_link(target: &Path, link: &Path) {
    std::os::unix::fs::symlink(target, link).unwrap();
}

#[test]
fn a_legacy_home_that_is_a_link_is_refused_and_its_target_untouched() {
    let w = world("m-homelink");
    let real = w.root.join("elsewhere");
    fs::create_dir_all(real.join("sessions")).unwrap();
    fs::write(real.join("sessions/x.txt"), "sentinel").unwrap();
    let person2 = w.root.join("person2");
    fs::create_dir_all(&person2).unwrap();
    mk_link(&real, &person2.join(".xeno-agent"));
    let plan = plan_migration(&w.dirs, &person2.to_string_lossy(), &rules()).unwrap();
    assert_eq!(plan.homes.iter().find(|h| h.name == ".xeno-agent").unwrap().state, HomeState::RefusedLink);
    assert!(plan.entries.is_empty());
    let r = run_migration(&plan, &w.dirs, confirm()).unwrap();
    assert!(!r.ok);
    assert!(!w.dirs.path(Kind::State, "agents/sessions", &[]).unwrap().exists(), "nothing was copied through the link");
    assert_eq!(fs::read_to_string(real.join("sessions/x.txt")).unwrap(), "sentinel");
}

#[test]
fn a_link_inside_a_legacy_home_is_never_followed() {
    let w = world("m-innerlink");
    let real = w.root.join("elsewhere");
    fs::create_dir_all(&real).unwrap();
    fs::write(real.join("secret.txt"), "sentinel").unwrap();
    mk_link(&real, &Path::new(&w.home).join(".xeno-agent").join("sessions-linked"));
    let rule = MigrationRule { id: "linked".into(), legacy_home: ".xeno-agent".into(), match_pattern: "sessions-linked".into(), action: RuleAction::Copy { kind: Kind::State, area: "agents/sessions".into(), segments: vec!["linked".into()] } };
    let plan = plan_migration(&w.dirs, &w.home, &[rule]).unwrap();
    assert!(plan.refused.iter().any(|r| r.reason.contains("link or junction")));
    assert!(!plan.entries.iter().any(|e| e.source_rel.starts_with("sessions-linked")));
}

#[test]
fn a_link_on_the_destination_path_is_refused_and_nothing_is_written_through_it() {
    let w = world("m-destlink");
    let sentinel = w.root.join("sentinel-dir");
    fs::create_dir_all(&sentinel).unwrap();
    fs::create_dir_all(w.dirs.root(Kind::State)).unwrap();
    mk_link(&sentinel, &Path::new(w.dirs.root(Kind::State)).join("agents"));
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    let r = run_migration(&plan, &w.dirs, confirm()).unwrap();
    assert_eq!(r.entries.iter().find(|e| e.source_rel == "sessions/one.txt").unwrap().status, EntryStatus::Refused);
    assert!(files(&sentinel).is_empty(), "the link target stayed empty");
}

#[test]
fn a_link_swapped_in_after_planning_is_refused_at_run_time() {
    let w = world("m-swap");
    let plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    let real = w.root.join("elsewhere");
    fs::create_dir_all(&real).unwrap();
    fs::write(real.join("one.txt"), "sentinel one").unwrap();
    fs::write(real.join("two.txt"), "sentinel two").unwrap();
    fs::remove_dir_all(Path::new(&w.home).join(".xeno-agent").join("sessions")).unwrap();
    mk_link(&real, &Path::new(&w.home).join(".xeno-agent").join("sessions"));
    let r = run_migration(&plan, &w.dirs, confirm()).unwrap();
    let agent: Vec<_> = r.entries.iter().filter(|e| e.rule == "agent-sessions").map(|e| e.status).collect();
    assert_eq!(agent, [EntryStatus::Refused, EntryStatus::Refused]);
    assert!(!w.dirs.path(Kind::State, "agents/sessions", &["one.txt"]).unwrap().exists());
    assert!(!r.ok);
}

// ---- doctor ----

fn dput(root: &Path, rel: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, "x").unwrap();
}

#[test]
fn doctor_clean_layout_and_dirty_layout() {
    let w = world("m-doctor");
    let root = w.root.join("xeno");
    for rel in ["data/layout.json", "data/minds/m.json", "data/agents/environments/c/p.json", "state/apps/pixel/w.json", "state/platform/migrations/m.jsonl", "cache/models/w", "run/platform/p.json", "run/agents/environments/c.lock", "secrets/identity/a.secret"] {
        dput(&root, rel);
    }
    let r = doctor(&w.dirs, None);
    assert!(r.unregistered.is_empty() && r.ok);
    for rel in ["config/stray.json", "config/scratch/x.json", "data/models/weights", "data/agents/legacy/x", "config/apps/Bad_Slug/x", "data/undefined/x", "data/.layout.json.1234.tmp", "config/apps/good-slug/x", "data/agents/definitions/ok.json", "config/identity/ok.json"] {
        dput(&root, rel);
    }
    let r = doctor(&w.dirs, Some(&w.home));
    let got: Vec<String> = r.unregistered.iter().map(|u| format!("{}/{}", u.kind.as_str(), u.path)).collect();
    assert_eq!(got, ["config/apps/Bad_Slug", "config/scratch", "config/stray.json", "data/.layout.json.1234.tmp", "data/agents/legacy", "data/models", "data/undefined"]);
    assert!(r.unregistered.iter().find(|u| u.path == "models").unwrap().reason.contains("not registered for kind data"));
    let present: Vec<(&str, bool)> = r.legacy.iter().map(|l| (l.name.as_str(), l.present)).collect();
    assert_eq!(present, [(".xeno", true), (".xeno-agent", true), (".xeno-anima", true), (".xeno-code", false)]);
}

// ---- interoperability with the TypeScript implementation ----

fn node_available() -> bool {
    Command::new("node").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn ts_module(rel: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/primitives/src").join(rel).canonicalize().unwrap();
    let s = p.to_string_lossy().replace('\\', "/");
    format!("file:///{}", s.strip_prefix("//?/").unwrap_or(&s).trim_start_matches('/'))
}

fn run_node(dir: &Path, script: &str) -> String {
    let file = dir.join("script.mjs");
    fs::write(&file, script).unwrap();
    let out = Command::new("node").arg(&file).output().unwrap();
    assert!(out.status.success(), "node failed: {}", String::from_utf8_lossy(&out.stderr).lines().take(5).collect::<Vec<_>>().join(" | "));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

const TS_RULES: &str = r#"[
  { id: 'agent-sessions', legacyHome: '.xeno-agent', match: 'sessions/*.txt', action: 'copy', target: { kind: 'state', area: 'agents/sessions', segments: ['{file}'] } }
]"#;

#[test]
fn interop_plan_ids_and_journals_are_shared_between_the_implementations() {
    if !node_available() {
        eprintln!("node not found: interop test skipped");
        return;
    }
    let w = world("m-interop");
    let header = format!(
        "import {{ createDirsAt, planMigration, runMigration, verifyMigration }} from {:?};\nconst dirs = createDirsAt({:?});\nconst plan = planMigration({{ dirs, home: {:?}, rules: {TS_RULES} }});\n",
        ts_module("dirs/index.ts"), w.root.join("xeno").to_string_lossy(), w.home
    );
    // 1. Same plan, same id.
    let ts_id = run_node(&w.root, &format!("{header}console.log(plan.id);\n"));
    let rust_plan = plan_migration(&w.dirs, &w.home, &rules()[..1]).unwrap();
    assert_eq!(rust_plan.id, ts_id, "the plan id is a contract");
    // 2. TypeScript migrates; Rust verifies its journal and finds everything already in place.
    let out = run_node(&w.root, &format!("{header}const r = await runMigration(plan, {{ dirs, confirm: true, writeNote: false }});\nconsole.log(r.ok);\n"));
    assert_eq!(out, "true");
    assert!(verify_migration(&w.dirs, &rust_plan.id).unwrap().complete, "Rust reads the TypeScript journal");
    let again = run_migration(&rust_plan, &w.dirs, confirm()).unwrap();
    assert!(again.ok && again.entries.iter().all(|e| e.status == EntryStatus::Already));
    // 3. A destination changed afterwards is caught by TypeScript verifying a journal that Rust resumed.
    fs::write(w.dirs.path(Kind::State, "agents/sessions", &["two.txt"]).unwrap(), "changed afterwards").unwrap();
    let v = run_node(&w.root, &format!("{header}const v = await verifyMigration(dirs, plan.id);\nconsole.log(v.complete);\n"));
    assert_eq!(v, "false");
}
