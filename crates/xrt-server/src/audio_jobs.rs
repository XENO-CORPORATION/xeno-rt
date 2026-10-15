//! Local durable audio jobs. SQLite owns terminal state and result publication
//! in one transaction; no success row can name a missing result file.
use crate::{audio_api, AppState};
use axum::{
    body::{to_bytes, Body},
    extract::{Path as RoutePath, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};
use xrt_audio::control::InferenceControl;

const MAX_JOBS: i64 = 100;
const MAX_PENDING: i64 = 8;
const MAX_STORE_BYTES: i64 = 2 * 1024 * 1024 * 1024;
type StoredResult = Option<(String, Option<Vec<u8>>, String)>;
const MAX_RESULT: usize = 128 * 1024 * 1024;

fn submission_slot() -> &'static Arc<tokio::sync::Semaphore> {
    static SLOT: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SLOT.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1)))
}

pub(crate) struct JobStore {
    db: Mutex<Connection>,
    controls: Mutex<HashMap<String, Arc<InferenceControl>>>,
    _lock: File,
}
#[derive(Default)]
pub(crate) struct Jobs(Mutex<Option<Arc<JobStore>>>);
impl Jobs {
    #[cfg(test)]
    pub(crate) fn isolated(root: &Path) -> Self {
        let jobs = Self::default();
        jobs.get_at(root).unwrap();
        jobs
    }
    #[cfg(feature = "agent-capabilities")]
    pub(crate) fn audit(
        &self,
        capability: &str,
        class: &str,
        input_sha256: &str,
        status: u16,
    ) -> Result<(), String> {
        let store = self.get()?;
        let mut db = store.db.lock().map_err(|e| e.to_string())?;
        let tx = db.transaction().map_err(|e| e.to_string())?;
        tx.execute("INSERT INTO capability_audit(capability,class,input_sha256,status) VALUES(?1,?2,?3,?4)",
            params![capability,class,input_sha256,status]).map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM capability_audit WHERE id <= (SELECT coalesce(max(id),0)-10000 FROM capability_audit)",[]).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    }
    fn get(&self) -> Result<Arc<JobStore>, String> {
        self.get_at(&default_root())
    }

    fn get_at(&self, root: &Path) -> Result<Arc<JobStore>, String> {
        let mut slot = self.0.lock().map_err(|e| e.to_string())?;
        if let Some(store) = slot.as_ref() {
            return Ok(store.clone());
        }
        // Only cache success: a transient lock/disk/permission error is retryable.
        let store = Arc::new(JobStore::open(root)?);
        *slot = Some(store.clone());
        Ok(store)
    }
}
fn default_root() -> PathBuf {
    std::env::var_os("XRT_AUDIO_JOBS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            xrt_audio::voices::VoiceLibrary::default_root()
                .join("audio-jobs")
                .join("v1")
        })
}
fn failure(code: StatusCode, message: impl Into<String>) -> Response {
    (
        code,
        Json(serde_json::json!({"error":{"type":"audio_job_error","message":message.into()}})),
    )
        .into_response()
}
fn valid_id(id: &str) -> bool {
    id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Serialize)]
#[cfg_attr(feature = "agent-capabilities", derive(schemars::JsonSchema))]
struct Job {
    id: String,
    status: String,
    error: Option<String>,
    created: i64,
    updated: i64,
    result_url: Option<String>,
    completed_chunks: u64,
    kind: String,
    progress_stage: Option<String>,
    progress_units: u64,
    progress_attempt: u64,
    result_metadata: Option<serde_json::Value>,
}
#[cfg(feature = "agent-capabilities")]
pub(crate) fn job_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(Job)).expect("static job schema is serializable")
}

impl JobStore {
    fn open(root: &Path) -> Result<Self, String> {
        // Refuse redirected state directories before creating or opening files.
        for p in root.ancestors() {
            if let Ok(m) = std::fs::symlink_metadata(p) {
                #[cfg(windows)]
                {
                    use std::os::windows::fs::MetadataExt;
                    if m.file_attributes() & 0x400 != 0 {
                        return Err("job root contains a reparse point".into());
                    }
                }
                if m.file_type().is_symlink() {
                    return Err("job root contains a symlink".into());
                }
            }
        }
        std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join("store.lock"))
            .map_err(|e| e.to_string())?;
        fs2::FileExt::try_lock_exclusive(&lock)
            .map_err(|_| "audio job store already belongs to another server".to_string())?;
        let db = Connection::open(root.join("jobs.sqlite3")).map_err(|e| e.to_string())?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA auto_vacuum=INCREMENTAL;
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL,
            request BLOB NOT NULL, identity TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
            result BLOB, created INTEGER NOT NULL DEFAULT(unixepoch()), updated INTEGER NOT NULL DEFAULT(unixepoch()));
            UPDATE jobs SET status='interrupted', error='server stopped before job completion', updated=unixepoch()
            WHERE status IN ('queued','running','cancelling');").map_err(|e|e.to_string())?;
        let has_progress = {
            let mut statement = db
                .prepare("PRAGMA table_info(jobs)")
                .map_err(|e| e.to_string())?;
            let names = statement
                .query_map([], |r| r.get::<_, String>(1))
                .map_err(|e| e.to_string())?;
            names
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
                .iter()
                .any(|n| n == "completed_chunks")
        };
        if !has_progress {
            db.execute_batch(
                "ALTER TABLE jobs ADD COLUMN completed_chunks INTEGER NOT NULL DEFAULT 0;",
            )
            .map_err(|e| e.to_string())?;
        }
        for (name, declaration) in [
            ("kind", "TEXT NOT NULL DEFAULT 'audio.speech'"),
            ("content_type", "TEXT NOT NULL DEFAULT 'application/json'"),
            ("progress_stage", "TEXT"),
            ("progress_units", "INTEGER NOT NULL DEFAULT 0"),
            ("progress_attempt", "INTEGER NOT NULL DEFAULT 0"),
            ("result_metadata", "TEXT"),
        ] {
            let exists: bool = db
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM pragma_table_info('jobs') WHERE name=?1)",
                    [name],
                    |r| r.get(0),
                )
                .map_err(|e| e.to_string())?;
            if !exists {
                db.execute_batch(&format!(
                    "ALTER TABLE jobs ADD COLUMN {name} {declaration};"
                ))
                .map_err(|e| e.to_string())?;
            }
        }
        db.execute_batch("CREATE TABLE IF NOT EXISTS capability_audit(id INTEGER PRIMARY KEY AUTOINCREMENT,
            capability TEXT NOT NULL,class TEXT NOT NULL,input_sha256 TEXT NOT NULL,status INTEGER NOT NULL,
            created INTEGER NOT NULL DEFAULT(unixepoch()));").map_err(|e| e.to_string())?;
        Ok(Self {
            db: Mutex::new(db),
            controls: Mutex::new(HashMap::new()),
            _lock: lock,
        })
    }
    fn get_job(&self, id: &str) -> Result<Option<Job>, String> {
        self.db
            .lock()
            .map_err(|e| e.to_string())?
            .query_row(
                "SELECT id,status,error,created,updated,completed_chunks,kind,progress_stage,progress_units,result_metadata,progress_attempt FROM jobs WHERE id=?1",
                [id],
                |r| {
                    let status: String = r.get(1)?;
                    Ok(Job {
                        id: r.get(0)?,
                        result_url: if status == "succeeded" {
                            Some(format!("/v1/audio/jobs/{id}/result"))
                        } else {
                            None
                        },
                        status,
                        error: r.get(2)?,
                        created: r.get(3)?,
                        updated: r.get(4)?,
                        completed_chunks: r.get(5)?,
                        kind: r.get(6)?,
                        progress_stage: r.get(7)?,
                        progress_units: r.get(8)?,
                        progress_attempt: r.get(10)?,
                        result_metadata: r.get::<_,Option<String>>(9)?.map(|value|serde_json::from_str(&value))
                            .transpose().map_err(|e|rusqlite::Error::FromSqlConversionFailure(9,rusqlite::types::Type::Text,Box::new(e)))?,
                    })
                },
            )
            .optional()
            .map_err(|e| e.to_string())
    }
    fn create(&self, idem: &str, request: &[u8], identity: &str) -> Result<(String, bool), String> {
        self.create_typed(idem, request, identity, "audio.speech", "application/json")
    }
    fn create_typed(
        &self,
        idem: &str,
        request: &[u8],
        identity: &str,
        kind: &str,
        content_type: &str,
    ) -> Result<(String, bool), String> {
        // Preserve existing legacy fingerprints while separating native work.
        let fingerprint = if kind == "audio.speech" {
            hash(&[request, identity.as_bytes()].concat())
        } else {
            hash(
                &serde_json::to_vec(&(kind, content_type, request, identity))
                    .map_err(|e| e.to_string())?,
            )
        };
        let mut db = self.db.lock().map_err(|e| e.to_string())?;
        let tx = db.transaction().map_err(|e| e.to_string())?;
        let existing: Option<(String, String)> = tx
            .query_row(
                "SELECT id,fingerprint FROM jobs WHERE idem=?1",
                [idem],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        if let Some((id, previous)) = existing {
            if previous != fingerprint {
                return Err(
                    "idempotency key belongs to different request/model/reference bytes".into(),
                );
            }
            return Ok((id, false));
        }
        let (count,pending,size):(i64,i64,i64)=tx.query_row("SELECT count(*),coalesce(sum(status IN ('queued','running','cancelling')),0),coalesce(sum(length(request)+coalesce(length(result),0)),0) FROM jobs",[],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).map_err(|e|e.to_string())?;
        if count >= MAX_JOBS
            || pending >= MAX_PENDING
            || size + request.len() as i64 + (pending + 1) * MAX_RESULT as i64 > MAX_STORE_BYTES
        {
            return Err(
                "job store capacity reached; delete completed jobs before submitting more".into(),
            );
        }
        let id = format!("{:032x}", rand::random::<u128>());
        tx.execute("INSERT INTO jobs(id,idem,fingerprint,request,identity,status,kind,content_type) VALUES(?1,?2,?3,?4,?5,'queued',?6,?7)",params![id,idem,fingerprint,request,identity,kind,content_type]).map_err(|e|e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok((id, true))
    }
    fn start(&self, id: &str) -> Result<(), String> {
        let count=self.db.lock().map_err(|e|e.to_string())?.execute(
            "UPDATE jobs SET status='running',updated=unixepoch() WHERE id=?1 AND status='queued'",[id]).map_err(|e|e.to_string())?;
        if count != 1 {
            return Err("job cancelled before admission".into());
        }
        Ok(())
    }
    fn terminal(
        &self,
        id: &str,
        status: &str,
        result: Option<&[u8]>,
        error: Option<&str>,
    ) -> Result<(), String> {
        self.terminal_with_metadata(id, status, result, error, None)
    }
    fn terminal_with_metadata(
        &self,
        id: &str,
        status: &str,
        result: Option<&[u8]>,
        error: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<(), String> {
        let db = self.db.lock().map_err(|e| e.to_string())?;
        // Cancellation requested before this transaction wins over success.
        db.execute("UPDATE jobs SET status=CASE WHEN status='cancelling' THEN 'cancelled' ELSE ?2 END,
            result=CASE WHEN status='cancelling' THEN NULL ELSE ?3 END,error=?4,
            result_metadata=CASE WHEN status='cancelling' THEN NULL ELSE ?5 END,updated=unixepoch() WHERE id=?1",
            params![id,status,result,error,metadata.map(serde_json::Value::to_string)]).map_err(|e|e.to_string())?;
        self.controls.lock().map_err(|e| e.to_string())?.remove(id);
        Ok(())
    }
}

/// Bind idempotency to actual graph/tokenizer bytes, including ONNX external
/// tensors. Per-process caching can be added only with immutable bundle identity.
fn model_identity(needs_recognizer: bool) -> Result<String, String> {
    let mut roots = vec![xrt_audio::speech::default_model_dir()];
    if needs_recognizer {
        roots.push(xrt_audio::whisper::Recognizer::default_dir());
    }
    model_identity_at(&roots)
}

fn model_identity_at(roots: &[PathBuf]) -> Result<String, String> {
    let mut digest = Sha256::new();
    for root in roots {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(root).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_file() {
                files.push(path);
            }
        }
        for entry in std::fs::read_dir(root.join("onnx")).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.is_file() {
                files.push(path);
            }
        }
        files.sort();
        for path in files {
            use std::io::Read;
            digest.update(
                path.strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .as_bytes(),
            );
            let mut file = File::open(path).map_err(|e| e.to_string())?;
            let mut buf = vec![0; 1024 * 1024];
            loop {
                let n = file.read(&mut buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                digest.update(&buf[..n]);
            }
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) async fn submit(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<audio_api::SpeechRequest>,
) -> Response {
    if !audio_api::accepts_jobs(&state) {
        return failure(StatusCode::SERVICE_UNAVAILABLE, "audio is draining");
    }
    let key = match headers.get("idempotency-key").and_then(|v| v.to_str().ok()) {
        Some(key)
            if !key.is_empty() && key.len() <= 128 && key.bytes().all(|b| b.is_ascii_graphic()) =>
        {
            key.to_string()
        }
        _ => {
            return failure(
                StatusCode::BAD_REQUEST,
                "Idempotency-Key of 1..128 ASCII characters is required",
            )
        }
    };
    let Ok(admission) = submission_slot().clone().try_acquire_owned() else {
        return failure(
            StatusCode::TOO_MANY_REQUESTS,
            "another job submission is validating; retry shortly",
        );
    };
    let jobs = state.audio_jobs.clone();
    let executor = tokio::runtime::Handle::current();
    let prepared = tokio::task::spawn_blocking(move || {
        let _admission = admission;
        let store = jobs
            .get()
            .map_err(|e| Box::new(failure(StatusCode::SERVICE_UNAVAILABLE, e)))?;
        let req = audio_api::freeze_job(req)?;
        let identity = model_identity(req.needs_recognizer())
            .map_err(|e| Box::new(failure(StatusCode::PRECONDITION_REQUIRED, e)))?;
        let bytes = serde_json::to_vec(&req)
            .map_err(|e| Box::new(failure(StatusCode::BAD_REQUEST, e.to_string())))?;
        let (id, new) = store
            .create(&key, &bytes, &identity)
            .map_err(|e| Box::new(failure(StatusCode::CONFLICT, e)))?;
        // Schedule before returning to the HTTP future: a submitter that
        // disconnects after the durable INSERT must not strand a queued job.
        if new {
            let control = Arc::new(InferenceControl::default());
            store
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id.clone(), control.clone());
            let worker_store = store.clone();
            let worker_id = id.clone();
            executor.spawn(async move {
                execute(
                    state,
                    worker_store,
                    worker_id,
                    Work::Legacy(req),
                    identity,
                    control,
                )
                .await;
            });
        }
        Ok::<_, Box<Response>>((store, id, new))
    })
    .await;
    let (store, id, new) = match prepared {
        Ok(Ok(v)) => v,
        Ok(Err(r)) => return *r,
        Err(e) => return failure(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match store.get_job(&id) {
        Ok(Some(job)) => (
            if new {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            },
            Json(job),
        )
            .into_response(),
        _ => failure(StatusCode::INTERNAL_SERVER_ERROR, "job record unavailable"),
    }
}

enum Work {
    Legacy(audio_api::SpeechRequest),
    #[cfg(feature = "drama3-native")]
    Native(crate::drama3_api::SpeechRequest),
}

#[cfg(feature = "drama3-native")]
pub(crate) async fn submit_native(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<crate::drama3_api::SpeechRequest>,
) -> Response {
    if !audio_api::accepts_jobs(&state) {
        return failure(StatusCode::SERVICE_UNAVAILABLE, "audio is draining");
    }
    let key = match headers.get("idempotency-key").and_then(|v| v.to_str().ok()) {
        Some(key)
            if !key.is_empty() && key.len() <= 128 && key.bytes().all(|b| b.is_ascii_graphic()) =>
        {
            key.to_owned()
        }
        _ => {
            return failure(
                StatusCode::BAD_REQUEST,
                "Idempotency-Key of 1..128 ASCII characters is required",
            )
        }
    };
    let Ok(admission) = submission_slot().clone().try_acquire_owned() else {
        return failure(
            StatusCode::TOO_MANY_REQUESTS,
            "another job submission is validating; retry shortly",
        );
    };
    let jobs = state.audio_jobs.clone();
    let executor = tokio::runtime::Handle::current();
    let prepared = tokio::task::spawn_blocking(move || {
        let _admission = admission;
        let identity = req
            .freeze_job(&state)
            .map_err(|e| Box::new(failure(StatusCode::BAD_REQUEST, e)))?;
        let store = jobs
            .get()
            .map_err(|e| Box::new(failure(StatusCode::SERVICE_UNAVAILABLE, e)))?;
        let bytes = serde_json::to_vec(&req)
            .map_err(|e| Box::new(failure(StatusCode::BAD_REQUEST, e.to_string())))?;
        let (id, new) = store
            .create_typed(&key, &bytes, &identity, "audio.native.speech", "audio/wav")
            .map_err(|e| Box::new(failure(StatusCode::CONFLICT, e)))?;
        if new {
            let control = Arc::new(InferenceControl::default());
            store
                .controls
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id.clone(), control.clone());
            let worker_store = store.clone();
            let worker_id = id.clone();
            executor.spawn(async move {
                execute(
                    state,
                    worker_store,
                    worker_id,
                    Work::Native(req),
                    identity,
                    control,
                )
                .await;
            });
        }
        Ok::<_, Box<Response>>((store, id, new))
    })
    .await;
    let (store, id, new) = match prepared {
        Ok(Ok(v)) => v,
        Ok(Err(r)) => return *r,
        Err(e) => return failure(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match store.get_job(&id) {
        Ok(Some(job)) => (
            if new {
                StatusCode::ACCEPTED
            } else {
                StatusCode::OK
            },
            Json(job),
        )
            .into_response(),
        _ => failure(StatusCode::INTERNAL_SERVER_ERROR, "job record unavailable"),
    }
}

async fn execute(
    state: AppState,
    store: Arc<JobStore>,
    id: String,
    req: Work,
    identity: String,
    control: Arc<InferenceControl>,
) {
    let identity_check = match &req {
        Work::Legacy(req) => {
            let needs_recognizer = req.needs_recognizer();
            tokio::task::spawn_blocking(move || model_identity(needs_recognizer)).await
        }
        #[cfg(feature = "drama3-native")]
        Work::Native(req) => {
            let mut req = req.clone();
            let state = state.clone();
            tokio::task::spawn_blocking(move || req.freeze_job(&state)).await
        }
    };
    if !matches!(identity_check,Ok(Ok(ref current)) if current==&identity) {
        let _ = store.terminal(
            &id,
            "failed",
            None,
            Some("model bytes changed after submission"),
        );
        return;
    }
    loop {
        if control.is_cancelled() || !audio_api::accepts_jobs(&state) {
            let _ = store.terminal(
                &id,
                "cancelled",
                None,
                Some("job cancelled or server draining"),
            );
            return;
        }
        let started_store = store.clone();
        let started_id = id.clone();
        let progress_store = store.clone();
        let progress_id = id.clone();
        let progress_control = control.clone();
        let response = match &req {
            Work::Legacy(req) => audio_api::speech_with_control(state.clone(),req.clone(),control.clone(),
            move || started_store.start(&started_id), move |chunk| {
                let update = progress_store.db.lock().unwrap_or_else(|e|e.into_inner()).execute(
                    "UPDATE jobs SET completed_chunks=?2,updated=unixepoch() WHERE id=?1 AND status='running'",
                    params![progress_id,chunk.index as u64+1]);
                if let Err(e)=update {
                    tracing::error!(job=%progress_id,error=%e,"cannot persist job progress");
                    progress_control.cancel();
                }
            }).await,
            #[cfg(feature = "drama3-native")]
            Work::Native(req) => crate::drama3_api::speech_with_control(state.clone(), req.clone(), control.clone(),
                move || started_store.start(&started_id), move |stage, units, attempt| {
                    let update = progress_store.db.lock().unwrap_or_else(|e|e.into_inner()).execute(
                        "UPDATE jobs SET progress_stage=?2,progress_units=?3,progress_attempt=?4,updated=unixepoch() WHERE id=?1 AND status='running'",
                        params![progress_id, stage, units as u64, attempt as u64]);
                    if let Err(e) = update {
                        tracing::error!(job=%progress_id,error=%e,"cannot persist native job progress");
                        progress_control.cancel();
                    }
                }).await,
        };
        let status = response.status();
        if status == StatusCode::TOO_MANY_REQUESTS {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            continue;
        }
        let metadata = serde_json::Value::Object(
            response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    if name.as_str().starts_with("x-xrt-") {
                        value.to_str().ok().map(|value| {
                            (name.to_string(), serde_json::Value::String(value.into()))
                        })
                    } else {
                        None
                    }
                })
                .collect(),
        );
        let body = match to_bytes(response.into_body(), MAX_RESULT).await {
            Ok(b) => b,
            Err(e) => {
                let _ = store.terminal(&id, "failed", None, Some(&e.to_string()));
                return;
            }
        };
        let terminal = if status.is_success() {
            "succeeded"
        } else if status == StatusCode::REQUEST_TIMEOUT {
            "cancelled"
        } else {
            "failed"
        };
        let error = if status.is_success() {
            None
        } else {
            Some(
                String::from_utf8_lossy(&body)
                    .chars()
                    .take(2048)
                    .collect::<String>(),
            )
        };
        if let Err(e) = store.terminal_with_metadata(
            &id,
            terminal,
            if status.is_success() {
                Some(&body)
            } else {
                None
            },
            error.as_deref(),
            if status.is_success() {
                Some(&metadata)
            } else {
                None
            },
        ) {
            tracing::error!(job=%id,error=%e,"could not persist audio job terminal state");
        }
        return;
    }
}

pub(crate) async fn get(
    State(state): State<AppState>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    if !valid_id(&id) {
        return failure(StatusCode::BAD_REQUEST, "invalid job id");
    }
    match state.audio_jobs.get().and_then(|s| s.get_job(&id)) {
        Ok(Some(job)) => Json(job).into_response(),
        Ok(None) => failure(StatusCode::NOT_FOUND, "job not found"),
        Err(e) => failure(StatusCode::SERVICE_UNAVAILABLE, e),
    }
}
pub(crate) async fn result(
    State(state): State<AppState>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    if !valid_id(&id) {
        return failure(StatusCode::BAD_REQUEST, "invalid job id");
    }
    let jobs = state.audio_jobs.clone();
    match tokio::task::spawn_blocking(move || -> Result<StoredResult, String> {
        let store = jobs.get()?;
        let db = store.db.lock().map_err(|e| e.to_string())?;
        db.query_row(
            "SELECT status,result,content_type FROM jobs WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .map_err(|e| e.to_string())
    })
    .await
    {
        Ok(Ok(Some((status, Some(bytes), content_type)))) if status == "succeeded" => (
            [
                (header::CONTENT_TYPE, content_type),
                (header::CACHE_CONTROL, "no-store".into()),
            ],
            Body::from(bytes),
        )
            .into_response(),
        Ok(Ok(Some(_))) => failure(StatusCode::CONFLICT, "job has no successful result"),
        Ok(Ok(None)) => failure(StatusCode::NOT_FOUND, "job not found"),
        _ => failure(StatusCode::INTERNAL_SERVER_ERROR, "cannot read job result"),
    }
}
pub(crate) async fn cancel(
    State(state): State<AppState>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    if !valid_id(&id) {
        return failure(StatusCode::BAD_REQUEST, "invalid job id");
    }
    let store = match state.audio_jobs.get() {
        Ok(s) => s,
        Err(e) => return failure(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let changed=store.db.lock().unwrap_or_else(|e|e.into_inner()).execute("UPDATE jobs SET status='cancelling',updated=unixepoch() WHERE id=?1 AND status IN ('queued','running')",[&id]);
    if changed.is_err() {
        return failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot persist cancellation",
        );
    }
    if let Some(c) = store
        .controls
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&id)
    {
        c.cancel();
    }
    get(State(state), RoutePath(id)).await
}
pub(crate) async fn delete(
    State(state): State<AppState>,
    RoutePath(id): RoutePath<String>,
) -> Response {
    if !valid_id(&id) {
        return failure(StatusCode::BAD_REQUEST, "invalid job id");
    }
    let store = match state.audio_jobs.get() {
        Ok(s) => s,
        Err(e) => return failure(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let db = store.db.lock().unwrap_or_else(|e| e.into_inner());
    match db.execute("DELETE FROM jobs WHERE id=?1 AND status IN ('succeeded','failed','cancelled','interrupted')",[&id]) {
        Ok(1)=>{
            let absent: Result<bool,_> = db.query_row("SELECT NOT EXISTS(SELECT 1 FROM jobs WHERE id=?1)", [&id], |r|r.get(0));
            if !matches!(absent,Ok(true)) { return failure(StatusCode::INTERNAL_SERVER_ERROR,"job deletion readback failed"); }
            let _=db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); PRAGMA incremental_vacuum;");Json(serde_json::json!({"id":id,"deleted":true})).into_response()},
        Ok(_)=>failure(StatusCode::CONFLICT,"job missing or still active"),Err(e)=>failure(StatusCode::INTERNAL_SERVER_ERROR,e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transient_open_failure_can_recover_without_restart() {
        let root =
            std::env::temp_dir().join(format!("xrt-jobs-retry-{:032x}", rand::random::<u128>()));
        let other = JobStore::open(&root).unwrap();
        let jobs = Jobs::default();
        assert!(jobs.get_at(&root).is_err());
        drop(other);
        let store = jobs.get_at(&root).unwrap();
        assert!(Arc::ptr_eq(&store, &jobs.get_at(&root).unwrap()));
        assert!(store.create("recovered", b"request", "model").is_ok());
    }

    #[test]
    fn identity_covers_only_the_models_used_by_the_request() {
        let root =
            std::env::temp_dir().join(format!("xrt-job-identity-{:032x}", rand::random::<u128>()));
        let speech = root.join("speech");
        let whisper = root.join("absent-whisper");
        std::fs::create_dir_all(speech.join("onnx")).unwrap();
        std::fs::write(speech.join("onnx/graph.onnx"), b"speech graph").unwrap();
        let identity = model_identity_at(std::slice::from_ref(&speech)).unwrap();
        assert!(model_identity_at(&[speech.clone(), whisper]).is_err());
        std::fs::write(speech.join("onnx/graph.onnx"), b"modified graph").unwrap();
        assert_ne!(identity, model_identity_at(&[speech]).unwrap());
    }
    #[test]
    fn idempotency_crash_recovery_and_atomic_results() {
        let root =
            std::env::temp_dir().join(format!("xrt-jobs-test-{:032x}", rand::random::<u128>()));
        let store = JobStore::open(&root).unwrap();
        let (id, new) = store.create("one", b"request", "model-hash").unwrap();
        assert!(new);
        assert_eq!(
            store.create("one", b"request", "model-hash").unwrap(),
            (id.clone(), false)
        );
        assert!(store.create("one", b"different", "model-hash").is_err());
        assert!(store.create("one", b"request", "changed-model").is_err());
        assert!(JobStore::open(&root).is_err());
        drop(store);
        let store = JobStore::open(&root).unwrap();
        assert_eq!(store.get_job(&id).unwrap().unwrap().status, "interrupted");
        let (done, _) = store.create("two", b"request", "model-hash").unwrap();
        store
            .terminal(&done, "succeeded", Some(b"result"), None)
            .unwrap();
        drop(store);
        let store = JobStore::open(&root).unwrap();
        assert_eq!(store.get_job(&done).unwrap().unwrap().status, "succeeded");
    }

    #[test]
    fn native_and_legacy_jobs_share_capacity_but_not_identity_or_result_format() {
        let root =
            std::env::temp_dir().join(format!("xrt-jobs-native-{:032x}", rand::random::<u128>()));
        let store = JobStore::open(&root).unwrap();
        let (native, _) = store
            .create_typed(
                "native",
                b"request",
                "identity",
                "audio.native.speech",
                "audio/wav",
            )
            .unwrap();
        assert!(store.create("native", b"request", "identity").is_err());
        assert_eq!(
            store.get_job(&native).unwrap().unwrap().kind,
            "audio.native.speech"
        );
        store.start(&native).unwrap();
        store
            .terminal(&native, "succeeded", Some(b"RIFFverified"), None)
            .unwrap();
        let job = store.get_job(&native).unwrap().unwrap();
        assert_eq!(
            job.result_url.unwrap(),
            format!("/v1/audio/jobs/{native}/result")
        );
        let db = store.db.lock().unwrap();
        let (content_type, bytes): (String, Vec<u8>) = db
            .query_row(
                "SELECT content_type,result FROM jobs WHERE id=?1",
                [&native],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(content_type, "audio/wav");
        assert_eq!(bytes, b"RIFFverified");
        drop(db);
        let (cancelled, _) = store
            .create_typed(
                "cancel",
                b"request",
                "identity",
                "audio.native.speech",
                "audio/wav",
            )
            .unwrap();
        store
            .db
            .lock()
            .unwrap()
            .execute(
                "UPDATE jobs SET status='cancelling' WHERE id=?1",
                [&cancelled],
            )
            .unwrap();
        assert!(store.start(&cancelled).is_err());
        store
            .terminal(&cancelled, "succeeded", Some(b"must not publish"), None)
            .unwrap();
        assert_eq!(
            store.get_job(&cancelled).unwrap().unwrap().status,
            "cancelled"
        );
        let result: Option<Vec<u8>> = store
            .db
            .lock()
            .unwrap()
            .query_row("SELECT result FROM jobs WHERE id=?1", [&cancelled], |r| {
                r.get(0)
            })
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn migration_keeps_existing_json_results_and_adds_native_columns() {
        let root = std::env::temp_dir().join(format!(
            "xrt-jobs-migration-{:032x}",
            rand::random::<u128>()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let old = Connection::open(root.join("jobs.sqlite3")).unwrap();
        old.execute_batch("CREATE TABLE jobs(id TEXT PRIMARY KEY,idem TEXT UNIQUE NOT NULL,fingerprint TEXT NOT NULL,
            request BLOB NOT NULL,identity TEXT NOT NULL,status TEXT NOT NULL,error TEXT,result BLOB,
            created INTEGER NOT NULL DEFAULT(unixepoch()),updated INTEGER NOT NULL DEFAULT(unixepoch()));
            INSERT INTO jobs(id,idem,fingerprint,request,identity,status,result) VALUES('old','old','hash',X'7B7D','model','succeeded',X'7B7D');").unwrap();
        drop(old);
        let store = JobStore::open(&root).unwrap();
        let job = store.get_job("old").unwrap().unwrap();
        assert_eq!(job.status, "succeeded");
        assert_eq!(job.kind, "audio.speech");
        let content: String = store
            .db
            .lock()
            .unwrap()
            .query_row("SELECT content_type FROM jobs WHERE id='old'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(content, "application/json");
    }
}
