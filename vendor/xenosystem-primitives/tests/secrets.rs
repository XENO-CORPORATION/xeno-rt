//! Secrets: the shared known-answer vector, the store's refusals, and interoperability with the TypeScript
//! implementation (envelope files, and the Windows Credential Manager entry). Every value here is a fixture.
#![cfg(feature = "testing")]

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use xenosystem_primitives::dirs::codec::{hex_decode, sha256, hex_encode};
use xenosystem_primitives::dirs::{create_dirs_at, XenoDirs};
use xenosystem_primitives::secrets::keystore::OsStoreOptions;
use xenosystem_primitives::secrets::testing::MemoryKeyStore;
use xenosystem_primitives::secrets::*;

fn vector() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../contract/vectors/secrets-envelope.json");
    serde_json::from_str(&fs::read_to_string(p).unwrap()).unwrap()
}

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
        let _ = fs::remove_dir_all(&self.0); // fixtures never contain links
    }
}

fn s(v: &Value, k: &str) -> String {
    v[k].as_str().unwrap().to_string()
}

fn code<T>(r: Result<T>) -> String {
    match r {
        Ok(_) => "no-error".into(),
        Err(e) => e.code_str().to_string(),
    }
}

fn message<T>(r: Result<T>) -> String {
    r.err().map(|e| e.to_string()).unwrap_or_default()
}

#[test]
fn sha256_matches_the_node_oracle() {
    for c in vector()["sha256Vectors"].as_array().unwrap() {
        assert_eq!(hex_encode(&sha256(c["input"].as_str().unwrap().as_bytes())), c["hex"].as_str().unwrap());
    }
}

#[test]
fn envelope_reproduces_and_opens_the_shared_vector() {
    let v = vector();
    let mk = hex_decode(&s(&v, "masterKeyHex")).unwrap();
    let arr = |k: &str| -> Vec<u8> { hex_decode(&s(&v, k)).unwrap() };
    let file = seal_secret(
        &mk,
        &s(&v, "area"),
        &s(&v, "name"),
        s(&v, "plaintextUtf8").as_bytes(),
        SealOptions {
            dek: Some(arr("dekHex").try_into().unwrap()),
            wrap_nonce: Some(arr("wrapNonceHex").try_into().unwrap()),
            data_nonce: Some(arr("dataNonceHex").try_into().unwrap()),
            created_at: Some(s(&v, "createdAt")),
        },
    )
    .unwrap();
    let e = &v["expectedFile"];
    assert_eq!(file.key_id, e["keyId"].as_str().unwrap());
    assert_eq!(file.wrap_nonce, e["wrap"]["nonce"].as_str().unwrap());
    assert_eq!(file.wrap_ct, e["wrap"]["ct"].as_str().unwrap());
    assert_eq!(file.data_nonce, e["data"]["nonce"].as_str().unwrap());
    assert_eq!(file.data_ct, e["data"]["ct"].as_str().unwrap());
    // And Rust opens the file Node/OpenSSL produced, via the parser.
    let parsed = EnvelopeFile::parse(&e.to_string()).unwrap();
    assert_eq!(open_secret(&mk, &parsed, &s(&v, "area"), &s(&v, "name")).unwrap(), s(&v, "plaintextUtf8").as_bytes());
    // The writer's JSON parses back to the same file.
    assert_eq!(EnvelopeFile::parse(&file.to_json()).unwrap(), file);
}

#[test]
fn tampering_relocation_and_wrong_key_are_refused_and_told_apart() {
    let v = vector();
    let mk = hex_decode(&s(&v, "masterKeyHex")).unwrap();
    let (area, name) = (s(&v, "area"), s(&v, "name"));
    let good = EnvelopeFile::parse(&v["expectedFile"].to_string()).unwrap();
    let flip = |b64: &str| {
        let mut b = xenosystem_primitives::dirs::codec::base64_decode(b64).unwrap();
        b[0] ^= 1;
        xenosystem_primitives::dirs::codec::base64_encode(&b)
    };
    let t = |f: &EnvelopeFile, key: &[u8], a: &str, n: &str| match open_secret(key, f, a, n) {
        Ok(_) => "opened".to_string(),
        Err(e) => e.code.as_str().to_string(),
    };
    assert_eq!(t(&EnvelopeFile { data_ct: flip(&good.data_ct), ..good.clone() }, &mk, &area, &name), "decrypt-failed");
    assert_eq!(t(&EnvelopeFile { wrap_ct: flip(&good.wrap_ct), ..good.clone() }, &mk, &area, &name), "decrypt-failed");
    assert_eq!(t(&good, &mk, &area, "another-name"), "decrypt-failed");
    assert_eq!(t(&EnvelopeFile { name: "another-name".into(), ..good.clone() }, &mk, &area, "another-name"), "decrypt-failed");
    assert_eq!(t(&EnvelopeFile { area: "minds".into(), ..good.clone() }, &mk, "minds", &name), "decrypt-failed");
    assert_eq!(t(&good, &[7u8; 32], &area, &name), "key-mismatch");
    assert!(EnvelopeFile::parse("{\"v\":2}").is_err() && EnvelopeFile::parse("nope").is_err());
}

#[test]
fn fresh_seals_use_fresh_randomness() {
    let mk = [5u8; 32];
    let a = seal_secret(&mk, "identity", "x", b"v", SealOptions::default()).unwrap();
    let b = seal_secret(&mk, "identity", "x", b"v", SealOptions::default()).unwrap();
    assert_ne!(a.data_nonce, b.data_nonce);
    assert_ne!(a.wrap_ct, b.wrap_ct);
}

#[test]
fn store_round_trip_has_no_plaintext_on_disk() {
    let f = Fixture::new("sec-roundtrip");
    let dirs = f.dirs();
    let store = SecretStore::new(dirs.clone(), Box::new(MemoryKeyStore::new()));
    store.set("identity", "refresh-token", b"value-4f1c9a-fixture").unwrap();
    assert_eq!(store.get("identity", "refresh-token").unwrap().unwrap(), b"value-4f1c9a-fixture");
    assert_eq!(store.get("identity", "missing").unwrap(), None);
    assert_eq!(store.list("identity").unwrap(), vec!["refresh-token"]);
    assert!(store.has("identity", "refresh-token").unwrap());
    let on_disk = fs::read_to_string(dirs.path(xenosystem_primitives::dirs::Kind::Secrets, "identity", &["refresh-token.secret"]).unwrap()).unwrap();
    assert!(!on_disk.contains("value-4f1c9a-fixture"));
    assert!(store.delete("identity", "refresh-token").unwrap());
    assert!(!store.delete("identity", "refresh-token").unwrap());
}

#[test]
fn missing_master_key_never_creates_a_new_one_over_existing_secrets() {
    let f = Fixture::new("sec-orphan");
    let a = SecretStore::new(f.dirs(), Box::new(MemoryKeyStore::new()));
    a.set("identity", "a", b"one").unwrap();
    let lost = MemoryKeyStore::new();
    let b = SecretStore::new(f.dirs(), Box::new(lost.clone()));
    assert_eq!(code(b.get("identity", "a")), "key-unavailable");
    assert_eq!(code(b.set("identity", "b", b"two")), "key-unavailable");
    assert!(lost.snapshot().is_none(), "no replacement key was created");
    let wrong = SecretStore::new(f.dirs(), Box::new(MemoryKeyStore::with_key(&[9u8; 32])));
    assert_eq!(code(wrong.get("identity", "a")), "key-mismatch");
}

#[test]
fn unprotected_store_refuses_everything() {
    let f = Fixture::new("sec-unprotected");
    let store = SecretStore::new(f.dirs(), unprotected_key_store("basic", "only an unprotected store is available"));
    let st = store.status();
    assert!(!st.available && st.reason.as_deref() == Some("only an unprotected store is available"));
    assert_eq!(code(store.set("identity", "a", b"v")), "unprotected-store");
    assert_eq!(code(store.get("identity", "a")), "unprotected-store");
    assert!(!f.0.join("home").join("secrets").exists(), "nothing was written");
}

#[test]
fn registry_and_layout_rules_apply_and_nothing_is_written_on_refusal() {
    let f = Fixture::new("sec-rules");
    let ks = MemoryKeyStore::new();
    let store = SecretStore::new(f.dirs(), Box::new(ks.clone()));
    assert_eq!(code(store.set("models", "a", b"v")), "kind-not-used");
    assert_eq!(code(store.set("scratch", "a", b"v")), "unregistered-area");
    assert_eq!(code(store.set("identity", "../a", b"v")), "invalid-segment");
    assert_eq!(code(store.set("identity", "CON", b"v")), "invalid-segment");
    assert!(!f.0.join("home").exists());
    let dirs = f.dirs();
    fs::create_dir_all(dirs.root(xenosystem_primitives::dirs::Kind::Data)).unwrap();
    fs::write(dirs.layout_path(), "{\"layout\":9}").unwrap();
    assert_eq!(code(store.set("identity", "a", b"v")), "layout-newer");
    assert!(ks.snapshot().is_none(), "a refused write must not create a master key");
}

#[test]
fn no_error_message_contains_the_value_or_the_key() {
    let f = Fixture::new("sec-leaks");
    let ks = MemoryKeyStore::new();
    let dirs = f.dirs();
    let store = SecretStore::new(dirs.clone(), Box::new(ks.clone()));
    let value = "leak-canary-9c2e41";
    store.set("identity", "a", value.as_bytes()).unwrap();
    let key = ks.snapshot().unwrap();
    let file = dirs.path(xenosystem_primitives::dirs::Kind::Secrets, "identity", &["a.secret"]).unwrap();
    let good = fs::read_to_string(&file).unwrap();
    let mut msgs = vec![];
    fs::write(&file, good.replacen("\"ct\":\"", "\"ct\":\"AAAA", 1)).unwrap();
    msgs.push(message(store.get("identity", "a")));
    fs::write(&file, format!("not json {value}")).unwrap();
    msgs.push(message(store.get("identity", "a")));
    fs::write(&file, &good).unwrap();
    msgs.push(message(SecretStore::new(dirs, Box::new(MemoryKeyStore::with_key(&[3u8; 32]))).get("identity", "a")));
    msgs.push(message(store.set("identity", &format!("bad/{value}"), b"x")));
    for m in msgs {
        assert!(!m.is_empty());
        assert!(!m.contains(value) && !m.contains(&hex_encode(&key)));
    }
}

// ---- interoperability with the TypeScript implementation (skipped when node is not installed) ----

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

#[test]
fn interop_secrets_sealed_by_typescript_open_in_rust_and_back() {
    if !node_available() {
        eprintln!("node not found: interop test skipped");
        return;
    }
    let f = Fixture::new("sec-interop");
    let key_hex = "11".repeat(32);
    let home = f.0.join("home");
    let header = format!(
        "import {{ createDirsAt }} from {:?};\nimport {{ createSecretStore }} from {:?};\nimport {{ createMemoryKeyStore }} from {:?};\nconst dirs = createDirsAt({:?});\nconst store = createSecretStore({{ dirs, keyStore: createMemoryKeyStore(Buffer.from({:?}, 'hex')) }});\n",
        ts_module("dirs/index.ts"), ts_module("secrets/index.ts"), ts_module("secrets/testing.ts"), home.to_string_lossy(), key_hex
    );
    // TypeScript writes, Rust reads.
    run_node(&f.0, &format!("{header}await store.set('identity', 'from-ts', 'ts-wrote-this-fixture');\nconsole.log('ok');\n"));
    let rust = SecretStore::new(create_dirs_at(&home).unwrap(), Box::new(MemoryKeyStore::with_key(&hex_decode(&key_hex).unwrap())));
    assert_eq!(rust.get("identity", "from-ts").unwrap().unwrap(), b"ts-wrote-this-fixture");
    // Rust writes, TypeScript reads.
    rust.set("minds", "from-rust", b"rust-wrote-this-fixture").unwrap();
    let out = run_node(&f.0, &format!("{header}console.log(await store.getString('minds', 'from-rust'));\n"));
    assert_eq!(out, "rust-wrote-this-fixture");
}

#[cfg(windows)]
#[test]
fn windows_credential_manager_is_shared_between_the_implementations() {
    use xenosystem_primitives::secrets::keystore::create_os_key_store;
    if !node_available() {
        eprintln!("node not found: interop test skipped");
        return;
    }
    let f = Fixture::new("sec-wincred");
    let service = format!("ai.xenosystem.test.rs{}", std::process::id());
    let rust = create_os_key_store(OsStoreOptions { service: Some(service.clone()), account: None });
    assert_eq!(rust.protection(), Protection::OsProtected);
    let cleanup = || {
        let _ = rust.delete_key();
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_eq!(rust.get_key().unwrap(), None);
        // TypeScript puts a key; Rust reads the same bytes.
        let ts_key = "ab".repeat(32);
        let header = format!("import {{ createWindowsCredentialKeyStore }} from {:?};\nconst ks = createWindowsCredentialKeyStore({{ service: {service:?} }});\n", ts_module("secrets/index.ts"));
        run_node(&f.0, &format!("{header}await ks.putKey(Buffer.from({ts_key:?}, 'hex'));\nconsole.log('ok');\n"));
        assert_eq!(hex_encode(&rust.get_key().unwrap().unwrap()), ts_key);
        // Rust replaces it; TypeScript reads the new bytes.
        let rs_key = vec![0xcdu8; 32];
        rust.put_key(&rs_key).unwrap();
        let out = run_node(&f.0, &format!("{header}console.log(Buffer.from(await ks.getKey()).toString('hex'));\n"));
        assert_eq!(out, hex_encode(&rs_key));
        // A full secret through the real store, Rust side.
        let store = SecretStore::new(f.dirs(), create_os_key_store(OsStoreOptions { service: Some(service.clone()), account: None }));
        store.set("identity", "token", b"windows-fixture").unwrap();
        assert_eq!(store.get("identity", "token").unwrap().unwrap(), b"windows-fixture");
    }));
    cleanup();
    assert_eq!(rust.get_key().unwrap(), None, "the test key was removed");
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
