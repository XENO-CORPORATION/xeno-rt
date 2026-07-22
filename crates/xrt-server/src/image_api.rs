use axum::{
    extract::{
        multipart::{Field, MultipartRejection},
        rejection::JsonRejection,
        FromRequest, Multipart, Request, State,
    },
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    convert::Infallible,
    env,
    net::IpAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{mpsc, OwnedSemaphorePermit, RwLock, Semaphore},
    task,
};
use tokio_stream::{wrappers::ReceiverStream, Stream};
use xrt_hub::ModelHub;
use xrt_image::{
    decode_image, DecodedImage, ImageBackendKind, ImageBatchResult, ImageCancellation,
    ImageCapability, ImageEditRequest, ImageError, ImageErrorKind, ImageGenerationRequest,
    ImageIoLimits, ImageModelBundle, ImageOffloadPolicy, ImageOutputFormat, ImagePreviewEvent,
    ImageProgressEvent, ImageProgressSink, ImageQuality, ImageResizePolicy, ImageRuntime,
};
use xrt_openai::{
    OpenAiErrorBody, OpenAiErrorEnvelope, OpenAiImageBackground, OpenAiImageData,
    OpenAiImageEditFields, OpenAiImageEditJsonRequest, OpenAiImageFormat,
    OpenAiImageGenerationRequest, OpenAiImageQuality, OpenAiImageReference, OpenAiImageResponse,
    OpenAiImageResponseBackground, OpenAiImageResponseFormat, OpenAiImageResponseQuality,
    OpenAiImageResponseSize, OpenAiImageStreamEvent, OpenAiImageStreamEventType,
    OpenAiXenoImageBackend, OpenAiXenoImageOffload, OpenAiXenoResizePolicy,
};
use xrt_runtime::GpuResourceManager;

use crate::{AppState, ModelInfo};

const DEFAULT_MAX_ACTIVE_IMAGE_JOBS: usize = 1;
const DEFAULT_MAX_QUEUED_IMAGE_JOBS: usize = 4;
const MAX_ACTIVE_IMAGE_JOBS: usize = 64;
const MAX_QUEUED_IMAGE_JOBS: usize = 1_024;
const MAX_API_KEY_BYTES: usize = 4_096;
const MAX_BASE64_RESPONSE_BYTES: usize = 128 * 1024 * 1024;
pub(crate) const MAX_EDIT_REQUEST_BYTES: usize = 128 * 1024 * 1024;
const MAX_EDIT_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_EDIT_IMAGE_URL_BYTES: usize = 20 * 1024 * 1024;
const MAX_EDIT_SCALAR_BYTES: usize = 64 * 1024;
const MAX_EDIT_MULTIPART_FIELDS: usize = 64;
const MAX_EDIT_SOURCE_IMAGES: usize = 3;
const MAX_PARTIAL_IMAGES: u8 = 3;
const MAX_IMAGE_STREAM_BUFFER_CAPACITY: usize = 64;
const IMAGE_STREAM_USAGE_METERING_AVAILABLE: bool = false;

#[derive(Debug)]
struct ImageApiFailure {
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: &'static str,
}

impl ImageApiFailure {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        param: Option<&str>,
        code: &'static str,
    ) -> Self {
        Self {
            status,
            message: message.into(),
            param: param.map(ToOwned::to_owned),
            code,
        }
    }

    fn into_response(self) -> Response {
        api_error(self.status, self.message, self.param.as_deref(), self.code)
    }
}

#[derive(Clone)]
pub(crate) struct ImageServerState {
    inner: Arc<ImageServerInner>,
}

struct ImageServerInner {
    runtimes: RwLock<BTreeMap<String, Arc<LoadedImageRuntime>>>,
    resources: Arc<GpuResourceManager>,
    queue: ImageExecutionQueue,
    load_lock: tokio::sync::Mutex<()>,
    next_generation: AtomicU64,
    auth: ImageApiAuth,
}

struct LoadedImageRuntime {
    id: String,
    bundle_digest: String,
    quantization: String,
    capabilities: Vec<ImageCapability>,
    requested_backend: ImageBackendKind,
    active_backend: ImageBackendKind,
    generation: u64,
    created: u64,
    runtime: Arc<ImageRuntime>,
    accepting: AtomicBool,
    next_job_id: AtomicU64,
    jobs: Mutex<BTreeMap<u64, ImageCancellation>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImageRuntimeSummary {
    pub success: bool,
    pub modality: &'static str,
    pub loaded_model: String,
    pub bundle_digest: String,
    pub quantization: String,
    pub capabilities: Vec<String>,
    pub requested_backend: String,
    pub active_backend: String,
    pub generation: u64,
    pub state: &'static str,
    pub active_jobs: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ImageRuntimeUnloadResponse {
    pub success: bool,
    pub modality: &'static str,
    pub unloaded_model: String,
    pub generation: u64,
    pub state: &'static str,
    pub active_jobs: usize,
    pub cancelled_jobs: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct RuntimeModelsResponse {
    object: &'static str,
    data: Vec<RuntimeModelInfo>,
    image_queue: ImageQueueStatus,
}

#[derive(Debug, Serialize)]
struct RuntimeModelInfo {
    id: String,
    modality: &'static str,
    capabilities: Vec<String>,
    state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    requested_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation: Option<u64>,
    active_jobs: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ImageQueueStatus {
    max_active_jobs: usize,
    max_queued_jobs: usize,
    active_jobs: usize,
    queued_jobs: usize,
}

#[derive(Debug)]
struct ParsedEditRequest {
    fields: OpenAiImageEditFields,
    images: Vec<Vec<u8>>,
    mask: Option<Vec<u8>>,
}

enum ServerImageRequest {
    Generation(ImageGenerationRequest),
    Edit(ImageEditRequest),
}

struct ServerPreviewSink {
    sender: mpsc::Sender<Result<Event, Infallible>>,
    cancellation: ImageCancellation,
    fields: PreparedResponseFields,
    event_type: OpenAiImageStreamEventType,
    max_previews: usize,
    next_preview: AtomicUsize,
}

impl ImageProgressSink for ServerPreviewSink {
    fn on_progress(&self, _event: &ImageProgressEvent) {
        if self.sender.is_closed() {
            self.cancellation.cancel();
        }
    }

    fn wants_previews(&self) -> bool {
        self.max_previews > 0 && !self.sender.is_closed()
    }

    fn on_preview(&self, preview: &ImagePreviewEvent) {
        if preview.output_index != 0 || self.sender.is_closed() {
            if self.sender.is_closed() {
                self.cancellation.cancel();
            }
            return;
        }
        let index =
            match self
                .next_preview
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |index| {
                    (index < self.max_previews).then_some(index + 1)
                }) {
                Ok(index) => index,
                Err(_) => return,
            };
        let Some(size) = self.fields.size else {
            self.cancellation.cancel();
            return;
        };
        let event = OpenAiImageStreamEvent {
            event_type: self.event_type,
            b64_json: BASE64_STANDARD.encode(preview.bytes.as_ref()),
            created_at: unix_timestamp(),
            output_format: self.fields.output_format,
            quality: self.fields.quality,
            size,
            background: self.fields.background,
            partial_image_index: u8::try_from(index).ok(),
            usage: None,
        };
        let item = serialize_image_sse_event(&event);
        match self.sender.try_send(item) {
            Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => self.cancellation.cancel(),
        }
    }
}

struct CancelOnDropImageStream {
    inner: ReceiverStream<Result<Event, Infallible>>,
    cancellation: ImageCancellation,
}

impl Stream for CancelOnDropImageStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(context)
    }
}

impl Drop for CancelOnDropImageStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

#[derive(Clone)]
struct ImageApiAuth {
    api_key: Option<Arc<[u8]>>,
    loopback_bind: bool,
    allow_unauthenticated_generation: bool,
}

impl ImageServerState {
    pub(crate) fn from_env(
        resources: Arc<GpuResourceManager>,
        bind_host: &str,
    ) -> Result<Self, String> {
        let max_active = env_usize(
            "XRT_MAX_ACTIVE_IMAGE_JOBS",
            DEFAULT_MAX_ACTIVE_IMAGE_JOBS,
            1,
            MAX_ACTIVE_IMAGE_JOBS,
        )?;
        let max_queued = env_usize(
            "XRT_MAX_QUEUED_IMAGE_JOBS",
            DEFAULT_MAX_QUEUED_IMAGE_JOBS,
            0,
            MAX_QUEUED_IMAGE_JOBS,
        )?;
        let auth = ImageApiAuth::from_env(bind_host)?;
        Ok(Self::new(resources, max_active, max_queued, auth))
    }

    fn new(
        resources: Arc<GpuResourceManager>,
        max_active: usize,
        max_queued: usize,
        auth: ImageApiAuth,
    ) -> Self {
        Self {
            inner: Arc::new(ImageServerInner {
                runtimes: RwLock::new(BTreeMap::new()),
                resources,
                queue: ImageExecutionQueue::new(max_active, max_queued),
                load_lock: tokio::sync::Mutex::new(()),
                next_generation: AtomicU64::new(1),
                auth,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_tests(resources: Arc<GpuResourceManager>) -> Self {
        Self::new(resources, 1, 1, ImageApiAuth::loopback_without_key())
    }

    pub(crate) fn authorize_generation(&self, headers: &HeaderMap) -> Result<(), String> {
        self.inner.auth.authorize_generation(headers)
    }

    pub(crate) fn authorize_admin(&self, headers: &HeaderMap) -> Result<(), String> {
        self.inner.auth.authorize_admin(headers)
    }

    pub(crate) async fn load_installed(
        &self,
        model: String,
        requested_backend: ImageBackendKind,
    ) -> Result<ImageRuntimeSummary, (StatusCode, String)> {
        let model = model.trim().to_string();
        if model.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "image runtime load requires a non-empty catalog model ID".to_string(),
            ));
        }

        let _load_guard = self.inner.load_lock.lock().await;
        let resources = Arc::clone(&self.inner.resources);
        let expected_model = model.clone();
        let loaded = task::spawn_blocking(move || {
            let hub = ModelHub::new().map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to open the model cache: {error}"),
                )
            })?;
            let path = hub
                .resolve_installed_bundle(&expected_model, None)
                .map_err(|_| {
                    (
                        StatusCode::NOT_FOUND,
                        format!("image model `{expected_model}` is not installed"),
                    )
                })?;
            let bundle = ImageModelBundle::open(path)
                .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
            if bundle.manifest().id != expected_model {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "installed bundle ID `{}` does not match requested model `{expected_model}`",
                        bundle.manifest().id
                    ),
                ));
            }
            let id = bundle.manifest().id.clone();
            let bundle_digest = bundle.digest().to_string();
            let quantization = bundle.manifest().quantization.clone();
            let capabilities = bundle.manifest().capabilities.clone();
            let runtime = ImageRuntime::load(bundle, requested_backend, resources)
                .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
            let active_backend = runtime.backend();
            Ok::<_, (StatusCode, String)>((
                id,
                bundle_digest,
                quantization,
                capabilities,
                active_backend,
                runtime,
            ))
        })
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "image runtime loader worker failed".to_string(),
            )
        })??;

        let generation = self.inner.next_generation.fetch_add(1, Ordering::AcqRel);
        let runtime = Arc::new(LoadedImageRuntime {
            id: loaded.0,
            bundle_digest: loaded.1,
            quantization: loaded.2,
            capabilities: loaded.3,
            requested_backend,
            active_backend: loaded.4,
            generation,
            created: unix_timestamp(),
            runtime: Arc::new(loaded.5),
            accepting: AtomicBool::new(true),
            next_job_id: AtomicU64::new(1),
            jobs: Mutex::new(BTreeMap::new()),
        });

        let previous = self
            .inner
            .runtimes
            .write()
            .await
            .insert(runtime.id.clone(), Arc::clone(&runtime));
        if let Some(previous) = previous {
            previous.begin_draining(false);
        }
        Ok(runtime.summary("ready"))
    }

    pub(crate) async fn unload(
        &self,
        model: Option<&str>,
        force: bool,
    ) -> Result<ImageRuntimeUnloadResponse, (StatusCode, String)> {
        let mut runtimes = self.inner.runtimes.write().await;
        let id = match model.map(str::trim).filter(|model| !model.is_empty()) {
            Some(model) => model.to_string(),
            None if runtimes.len() == 1 => runtimes
                .keys()
                .next()
                .expect("one runtime has one key")
                .clone(),
            None if runtimes.is_empty() => {
                return Err((
                    StatusCode::NOT_FOUND,
                    "no image model is loaded".to_string(),
                ))
            }
            None => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "model is required when more than one image runtime is loaded".to_string(),
                ))
            }
        };
        let runtime = runtimes.remove(&id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("image model `{id}` is not loaded"),
            )
        })?;
        drop(runtimes);
        let active_jobs = runtime.begin_draining(force);
        Ok(ImageRuntimeUnloadResponse {
            success: true,
            modality: "image",
            unloaded_model: id,
            generation: runtime.generation,
            state: "draining",
            active_jobs,
            cancelled_jobs: if force { active_jobs } else { 0 },
        })
    }

    pub(crate) async fn openai_models(&self) -> Vec<ModelInfo> {
        self.inner
            .runtimes
            .read()
            .await
            .values()
            .map(|runtime| ModelInfo {
                id: runtime.id.clone(),
                object: "model",
                created: runtime.created,
                owned_by: "xeno-rt",
            })
            .collect()
    }

    async fn select_runtime(
        &self,
        requested_model: Option<&str>,
        capability: ImageCapability,
    ) -> Result<Arc<LoadedImageRuntime>, ImageApiFailure> {
        let runtimes = self.inner.runtimes.read().await;
        let runtime = match requested_model
            .map(str::trim)
            .filter(|model| !model.is_empty())
        {
            Some(model) => runtimes.get(model).cloned().ok_or_else(|| {
                ImageApiFailure::new(
                    StatusCode::NOT_FOUND,
                    format!("image model `{model}` is not loaded"),
                    Some("model"),
                    "model_not_loaded",
                )
            })?,
            None => {
                let mut candidates = runtimes
                    .values()
                    .filter(|runtime| runtime.capabilities.contains(&capability));
                let first = candidates.next().cloned().ok_or_else(|| {
                    ImageApiFailure::new(
                        StatusCode::NOT_FOUND,
                        "no image generation model is loaded",
                        Some("model"),
                        "model_not_loaded",
                    )
                })?;
                if candidates.next().is_some() {
                    return Err(ImageApiFailure::new(
                        StatusCode::BAD_REQUEST,
                        "model is required when more than one compatible image model is loaded",
                        Some("model"),
                        "model_required",
                    ));
                }
                first
            }
        };
        if !runtime.capabilities.contains(&capability) {
            return Err(ImageApiFailure::new(
                StatusCode::BAD_REQUEST,
                format!(
                    "image model `{}` does not advertise `{}`",
                    runtime.id,
                    capability.id()
                ),
                Some("model"),
                "unsupported_capability",
            ));
        }
        Ok(runtime)
    }

    #[cfg(test)]
    async fn install_synthetic_for_test(&self) -> ImageRuntimeSummary {
        let bundle = xrt_image::synthetic_bundle_for_tests();
        let id = bundle.manifest().id.clone();
        let bundle_digest = bundle.digest().to_string();
        let quantization = bundle.manifest().quantization.clone();
        let capabilities = bundle.manifest().capabilities.clone();
        let runtime = ImageRuntime::load(
            bundle,
            ImageBackendKind::Cpu,
            Arc::clone(&self.inner.resources),
        )
        .expect("synthetic runtime should load");
        let generation = self.inner.next_generation.fetch_add(1, Ordering::AcqRel);
        let runtime = Arc::new(LoadedImageRuntime {
            id,
            bundle_digest,
            quantization,
            capabilities,
            requested_backend: ImageBackendKind::Cpu,
            active_backend: runtime.backend(),
            generation,
            created: unix_timestamp(),
            runtime: Arc::new(runtime),
            accepting: AtomicBool::new(true),
            next_job_id: AtomicU64::new(1),
            jobs: Mutex::new(BTreeMap::new()),
        });
        self.inner
            .runtimes
            .write()
            .await
            .insert(runtime.id.clone(), Arc::clone(&runtime));
        runtime.summary("ready")
    }
}

impl LoadedImageRuntime {
    fn summary(&self, state: &'static str) -> ImageRuntimeSummary {
        ImageRuntimeSummary {
            success: true,
            modality: "image",
            loaded_model: self.id.clone(),
            bundle_digest: self.bundle_digest.clone(),
            quantization: self.quantization.clone(),
            capabilities: self
                .capabilities
                .iter()
                .map(|capability| capability.id().to_string())
                .collect(),
            requested_backend: self.requested_backend.as_str().to_string(),
            active_backend: self.active_backend.as_str().to_string(),
            generation: self.generation,
            state,
            active_jobs: self.jobs().len(),
        }
    }

    fn pin_job(self: &Arc<Self>) -> Result<ImageJobLease, ImageApiFailure> {
        let cancellation = ImageCancellation::new();
        let mut jobs = self.jobs();
        if !self.accepting.load(Ordering::Acquire) {
            return Err(ImageApiFailure::new(
                StatusCode::CONFLICT,
                format!("image model `{}` is draining", self.id),
                Some("model"),
                "model_draining",
            ));
        }
        let job_id = self.next_job_id.fetch_add(1, Ordering::AcqRel);
        jobs.insert(job_id, cancellation.clone());
        drop(jobs);
        Ok(ImageJobLease {
            runtime: Arc::clone(self),
            job_id,
            cancellation,
        })
    }

    fn begin_draining(&self, force: bool) -> usize {
        let jobs = self.jobs();
        self.accepting.store(false, Ordering::Release);
        let active = jobs.len();
        if force {
            for cancellation in jobs.values() {
                cancellation.cancel();
            }
        }
        active
    }

    fn jobs(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, ImageCancellation>> {
        self.jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

struct ImageJobLease {
    runtime: Arc<LoadedImageRuntime>,
    job_id: u64,
    cancellation: ImageCancellation,
}

impl ImageJobLease {
    fn cancellation(&self) -> ImageCancellation {
        self.cancellation.clone()
    }
}

impl Drop for ImageJobLease {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.runtime.jobs().remove(&self.job_id);
    }
}

struct ImageExecutionQueue {
    permits: Arc<Semaphore>,
    active: Arc<AtomicUsize>,
    queued: Arc<AtomicUsize>,
    max_active: usize,
    max_queued: usize,
}

impl ImageExecutionQueue {
    fn new(max_active: usize, max_queued: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max_active)),
            active: Arc::new(AtomicUsize::new(0)),
            queued: Arc::new(AtomicUsize::new(0)),
            max_active,
            max_queued,
        }
    }

    async fn acquire(&self) -> Result<ImageQueuePermit, ImageApiFailure> {
        if let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() {
            self.active.fetch_add(1, Ordering::AcqRel);
            return Ok(ImageQueuePermit {
                _permit: permit,
                active: Arc::clone(&self.active),
            });
        }

        self.queued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < self.max_queued).then_some(queued + 1)
            })
            .map_err(|_| {
                ImageApiFailure::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "image execution queue is full",
                    None,
                    "image_queue_full",
                )
            })?;
        let mut registration = QueuedRegistration {
            queued: Arc::clone(&self.queued),
            active: true,
        };
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| {
                ImageApiFailure::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "image execution queue is closed",
                    None,
                    "image_queue_closed",
                )
            })?;
        registration.release();
        self.active.fetch_add(1, Ordering::AcqRel);
        Ok(ImageQueuePermit {
            _permit: permit,
            active: Arc::clone(&self.active),
        })
    }

    fn status(&self) -> ImageQueueStatus {
        ImageQueueStatus {
            max_active_jobs: self.max_active,
            max_queued_jobs: self.max_queued,
            active_jobs: self.active.load(Ordering::Acquire),
            queued_jobs: self.queued.load(Ordering::Acquire),
        }
    }
}

struct QueuedRegistration {
    queued: Arc<AtomicUsize>,
    active: bool,
}

impl QueuedRegistration {
    fn release(&mut self) {
        if self.active {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            self.active = false;
        }
    }
}

impl Drop for QueuedRegistration {
    fn drop(&mut self) {
        self.release();
    }
}

struct ImageQueuePermit {
    _permit: OwnedSemaphorePermit,
    active: Arc<AtomicUsize>,
}

impl Drop for ImageQueuePermit {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl ImageApiAuth {
    fn from_env(bind_host: &str) -> Result<Self, String> {
        let loopback_bind = is_loopback_bind_host(bind_host);
        let api_key = env::var("XRT_API_KEY")
            .ok()
            .map(|value| value.trim().as_bytes().to_vec())
            .filter(|value| !value.is_empty());
        if api_key
            .as_ref()
            .is_some_and(|value| value.len() > MAX_API_KEY_BYTES)
        {
            return Err(format!(
                "XRT_API_KEY exceeds the {MAX_API_KEY_BYTES}-byte limit"
            ));
        }
        let allow_unauthenticated_generation = env_truthy("XRT_ALLOW_UNAUTHENTICATED_IMAGE_API");
        if !loopback_bind && api_key.is_none() && !allow_unauthenticated_generation {
            return Err(
                "image routes on a non-loopback bind require XRT_API_KEY or explicit XRT_ALLOW_UNAUTHENTICATED_IMAGE_API=1"
                    .to_string(),
            );
        }
        Ok(Self {
            api_key: api_key.map(Arc::from),
            loopback_bind,
            allow_unauthenticated_generation,
        })
    }

    fn authorize_generation(&self, headers: &HeaderMap) -> Result<(), String> {
        if let Some(expected) = &self.api_key {
            return authorize_bearer(headers, expected);
        }
        if self.loopback_bind || self.allow_unauthenticated_generation {
            Ok(())
        } else {
            Err("image API authentication is required".to_string())
        }
    }

    fn authorize_admin(&self, headers: &HeaderMap) -> Result<(), String> {
        if let Some(expected) = &self.api_key {
            return authorize_bearer(headers, expected);
        }
        if self.loopback_bind {
            Ok(())
        } else {
            Err(
                "an API key is required for runtime administration on a non-loopback bind"
                    .to_string(),
            )
        }
    }

    #[cfg(test)]
    fn loopback_without_key() -> Self {
        Self {
            api_key: None,
            loopback_bind: true,
            allow_unauthenticated_generation: false,
        }
    }
}

pub(crate) fn parse_image_backend(value: Option<&str>) -> Result<ImageBackendKind, String> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("auto") => Ok(ImageBackendKind::Auto),
        Some("cpu") => Ok(ImageBackendKind::Cpu),
        Some("cuda") => Ok(ImageBackendKind::Cuda),
        Some(other) => Err(format!("unsupported image backend value: {other}")),
    }
}

pub(crate) async fn image_generations(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<OpenAiImageGenerationRequest>, JsonRejection>,
) -> Response {
    if let Err(message) = state.image.authorize_generation(&headers) {
        return api_error(StatusCode::UNAUTHORIZED, message, None, "invalid_api_key");
    }
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("invalid image generation request: {error}"),
                None,
                "invalid_json",
            )
        }
    };
    let runtime = match state
        .image
        .select_runtime(request.model.as_deref(), ImageCapability::Generate)
        .await
    {
        Ok(runtime) => runtime,
        Err(error) => return error.into_response(),
    };
    let (request, response_fields) = match prepare_generation_request(request, &runtime.id) {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    let job = match runtime.pin_job() {
        Ok(job) => job,
        Err(error) => return error.into_response(),
    };
    let permit = match state.image.inner.queue.acquire().await {
        Ok(permit) => permit,
        Err(error) => return error.into_response(),
    };
    if response_fields.stream {
        return image_stream_response(
            ServerImageRequest::Generation(request),
            Arc::clone(&runtime.runtime),
            job,
            permit,
            response_fields,
            state.stream_buffer_capacity,
        );
    }
    let cancellation = job.cancellation();
    let image_runtime = Arc::clone(&runtime.runtime);
    let result =
        task::spawn_blocking(move || image_runtime.generate(request, cancellation, None)).await;
    drop(permit);
    drop(job);
    let result = match result {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => return map_image_error(error),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "image generation worker failed",
                None,
                "image_worker_failed",
            )
        }
    };

    image_batch_response(result, response_fields)
}

pub(crate) async fn image_edits(State(state): State<AppState>, request: Request) -> Response {
    if let Err(message) = state.image.authorize_generation(request.headers()) {
        return api_error(StatusCode::UNAUTHORIZED, message, None, "invalid_api_key");
    }
    let (is_multipart, is_json) = {
        let media_type = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .unwrap_or_default();
        (
            media_type.eq_ignore_ascii_case("multipart/form-data"),
            media_type.eq_ignore_ascii_case("application/json"),
        )
    };
    let parsed = if is_multipart {
        let payload: Result<Multipart, MultipartRejection> =
            Multipart::from_request(request, &state).await;
        let multipart = match payload {
            Ok(multipart) => multipart,
            Err(error) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    format!("invalid image edit multipart request: {error}"),
                    None,
                    "invalid_multipart",
                )
            }
        };
        match parse_edit_multipart(multipart).await {
            Ok(parsed) => parsed,
            Err(error) => return error.into_response(),
        }
    } else if is_json {
        let payload: Result<Json<OpenAiImageEditJsonRequest>, JsonRejection> =
            Json::from_request(request, &state).await;
        let Json(payload) = match payload {
            Ok(payload) => payload,
            Err(error) => {
                return api_error(
                    StatusCode::BAD_REQUEST,
                    format!("invalid image edit JSON request: {error}"),
                    None,
                    "invalid_json",
                )
            }
        };
        match task::spawn_blocking(move || parse_edit_json(payload)).await {
            Ok(Ok(parsed)) => parsed,
            Ok(Err(error)) => return error.into_response(),
            Err(_) => {
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "image edit JSON decoder worker failed",
                    None,
                    "image_worker_failed",
                )
            }
        }
    } else {
        return api_error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "image edits require multipart/form-data or application/json",
            Some("content_type"),
            "unsupported_media_type",
        );
    };
    if parsed.fields.input_fidelity.is_some() {
        return unsupported(
            "input_fidelity",
            "input_fidelity has no admitted Qwen Image mapping yet",
        )
        .into_response();
    }
    let runtime = match state
        .image
        .select_runtime(parsed.fields.model.as_deref(), ImageCapability::Edit)
        .await
    {
        Ok(runtime) => runtime,
        Err(error) => return error.into_response(),
    };
    if parsed.mask.is_some() && !runtime.capabilities.contains(&ImageCapability::Inpaint) {
        return unsupported(
            "mask",
            "mask requires an image.inpaint model; image.edit does not imply mask support",
        )
        .into_response();
    }

    let image_bytes = parsed.images;
    let mask_bytes = parsed.mask;
    let decoded = task::spawn_blocking(move || {
        let limits = ImageIoLimits::default();
        let images = image_bytes
            .iter()
            .map(|bytes| decode_image(bytes, limits))
            .collect::<Result<Vec<_>, _>>()?;
        let mask = mask_bytes
            .as_deref()
            .map(|bytes| decode_image(bytes, limits))
            .transpose()?;
        Ok::<_, ImageError>((images, mask))
    })
    .await;
    let (images, mask) = match decoded {
        Ok(Ok(decoded)) => decoded,
        Ok(Err(error)) => return map_image_input_error(error),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "image input decoder worker failed",
                None,
                "image_worker_failed",
            )
        }
    };
    let default_dimensions = match runtime.runtime.default_edit_dimensions(&images) {
        Ok(dimensions) => dimensions,
        Err(error) => return map_image_input_error(error),
    };
    let (request, response_fields) =
        match prepare_edit_request(parsed.fields, &runtime.id, images, mask, default_dimensions) {
            Ok(prepared) => prepared,
            Err(error) => return error.into_response(),
        };
    let job = match runtime.pin_job() {
        Ok(job) => job,
        Err(error) => return error.into_response(),
    };
    let permit = match state.image.inner.queue.acquire().await {
        Ok(permit) => permit,
        Err(error) => return error.into_response(),
    };
    if response_fields.stream {
        return image_stream_response(
            ServerImageRequest::Edit(request),
            Arc::clone(&runtime.runtime),
            job,
            permit,
            response_fields,
            state.stream_buffer_capacity,
        );
    }
    let cancellation = job.cancellation();
    let image_runtime = Arc::clone(&runtime.runtime);
    let result =
        task::spawn_blocking(move || image_runtime.edit(request, cancellation, None)).await;
    drop(permit);
    drop(job);
    let result = match result {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => return map_image_error(error),
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "image edit worker failed",
                None,
                "image_worker_failed",
            )
        }
    };
    image_batch_response(result, response_fields)
}

fn image_stream_response(
    request: ServerImageRequest,
    runtime: Arc<ImageRuntime>,
    job: ImageJobLease,
    permit: ImageQueuePermit,
    fields: PreparedResponseFields,
    configured_capacity: usize,
) -> Response {
    let is_edit = matches!(request, ServerImageRequest::Edit(_));
    let required_capacity = usize::from(fields.partial_images).saturating_add(1);
    let capacity = configured_capacity
        .clamp(2, MAX_IMAGE_STREAM_BUFFER_CAPACITY)
        .max(required_capacity)
        .min(MAX_IMAGE_STREAM_BUFFER_CAPACITY);
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(capacity);
    let cancellation = job.cancellation();
    let sink: Arc<dyn ImageProgressSink> = Arc::new(ServerPreviewSink {
        sender: sender.clone(),
        cancellation: cancellation.clone(),
        fields: fields.clone(),
        event_type: if is_edit {
            OpenAiImageStreamEventType::EditPartialImage
        } else {
            OpenAiImageStreamEventType::GenerationPartialImage
        },
        max_previews: usize::from(fields.partial_images),
        next_preview: AtomicUsize::new(0),
    });
    let worker_cancellation = cancellation.clone();
    let worker = task::spawn_blocking(move || {
        let _permit = permit;
        let _job = job;
        match request {
            ServerImageRequest::Generation(request) => {
                runtime.generate(request, worker_cancellation, Some(sink))
            }
            ServerImageRequest::Edit(request) => {
                runtime.edit(request, worker_cancellation, Some(sink))
            }
        }
    });
    task::spawn(async move {
        let terminal = match worker.await {
            Ok(Ok(result)) => completed_stream_event(result, &fields, is_edit)
                .map(|event| serialize_image_sse_event(&event))
                .unwrap_or_else(serialize_stream_error),
            Ok(Err(error)) => serialize_stream_error(stream_image_error(error)),
            Err(_) => serialize_stream_error(OpenAiErrorEnvelope {
                error: OpenAiErrorBody {
                    message: "image execution worker failed".to_string(),
                    error_type: "server_error".to_string(),
                    param: None,
                    code: Some("image_worker_failed".to_string()),
                },
            }),
        };
        let _ = sender.send(terminal).await;
    });

    let stream = CancelOnDropImageStream {
        inner: ReceiverStream::new(receiver),
        cancellation,
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn completed_stream_event(
    _result: ImageBatchResult,
    _fields: &PreparedResponseFields,
    _is_edit: bool,
) -> Result<OpenAiImageStreamEvent, OpenAiErrorEnvelope> {
    Err(stream_protocol_error(
        "image streaming is unavailable until versioned token-equivalent usage metering is implemented",
        "image_stream_usage_unavailable",
    ))
}

fn serialize_sse_event<T: Serialize>(value: &T) -> Result<Event, Infallible> {
    let data = serde_json::to_string(value).unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"failed to serialize image stream event\",\"type\":\"server_error\",\"code\":\"image_stream_serialization_failed\"}}".to_string()
    });
    Ok(Event::default().data(data))
}

fn serialize_image_sse_event(value: &OpenAiImageStreamEvent) -> Result<Event, Infallible> {
    let data = serde_json::to_string(value).unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"failed to serialize image stream event\",\"type\":\"server_error\",\"code\":\"image_stream_serialization_failed\"}}".to_string()
    });
    Ok(Event::default().event(value.event_type.as_str()).data(data))
}

fn serialize_stream_error(error: OpenAiErrorEnvelope) -> Result<Event, Infallible> {
    serialize_sse_event(&error)
}

fn stream_protocol_error(message: &str, code: &str) -> OpenAiErrorEnvelope {
    OpenAiErrorEnvelope {
        error: OpenAiErrorBody {
            message: message.to_string(),
            error_type: "server_error".to_string(),
            param: None,
            code: Some(code.to_string()),
        },
    }
}

fn stream_image_error(error: ImageError) -> OpenAiErrorEnvelope {
    let kind = error.kind();
    let (error_type, code, expose) = match kind {
        ImageErrorKind::InvalidRequest => ("invalid_request_error", "invalid_image_request", true),
        ImageErrorKind::UnsupportedCapability => {
            ("invalid_request_error", "unsupported_capability", true)
        }
        ImageErrorKind::UnsupportedQuantization => {
            ("invalid_request_error", "unsupported_quantization", true)
        }
        ImageErrorKind::UnsupportedTensor => ("invalid_request_error", "unsupported_tensor", false),
        ImageErrorKind::UnsupportedShape => (
            "invalid_request_error",
            "image_dimensions_unsupported",
            true,
        ),
        ImageErrorKind::UnsupportedBackend => {
            ("invalid_request_error", "unsupported_backend", true)
        }
        ImageErrorKind::Admission | ImageErrorKind::InsufficientMemory => {
            ("invalid_request_error", "image_admission_failed", true)
        }
        ImageErrorKind::Cancelled => ("invalid_request_error", "image_cancelled", true),
        ImageErrorKind::InputLimit => ("invalid_request_error", "image_input_limit", true),
        ImageErrorKind::Codec => ("server_error", "image_codec_failed", false),
        ImageErrorKind::MissingComponent
        | ImageErrorKind::CorruptComponent
        | ImageErrorKind::Checksum
        | ImageErrorKind::Manifest
        | ImageErrorKind::Execution
        | ImageErrorKind::Numerical
        | ImageErrorKind::Internal => ("server_error", "image_execution_failed", false),
    };
    OpenAiErrorEnvelope {
        error: OpenAiErrorBody {
            message: if expose {
                error.to_string()
            } else {
                "image generation failed".to_string()
            },
            error_type: error_type.to_string(),
            param: None,
            code: Some(code.to_string()),
        },
    }
}

pub(crate) async fn runtime_models(State(state): State<AppState>) -> Json<RuntimeModelsResponse> {
    let mut data = Vec::new();
    if let Some(id) = state.loaded_model_name.read().await.clone() {
        let runtime = state.runtime.read().await.clone();
        let external = state.external_openai.read().await.clone();
        data.push(RuntimeModelInfo {
            id,
            modality: "text",
            capabilities: vec!["text.generate".to_string()],
            state: "ready",
            bundle_digest: None,
            quantization: None,
            requested_backend: Some(state.requested_backend.read().await.as_str().to_string()),
            active_backend: external
                .as_ref()
                .map(|_| "external-openai".to_string())
                .or_else(|| {
                    runtime
                        .as_ref()
                        .map(|runtime| runtime.active_backend().as_str().to_string())
                }),
            generation: None,
            active_jobs: runtime
                .as_ref()
                .map(|runtime| runtime.gpu_resource_status().active_sessions)
                .unwrap_or(0),
        });
    }
    data.extend(
        state
            .image
            .inner
            .runtimes
            .read()
            .await
            .values()
            .map(|runtime| RuntimeModelInfo {
                id: runtime.id.clone(),
                modality: "image",
                capabilities: runtime
                    .capabilities
                    .iter()
                    .map(|capability| capability.id().to_string())
                    .collect(),
                state: "ready",
                bundle_digest: Some(runtime.bundle_digest.clone()),
                quantization: Some(runtime.quantization.clone()),
                requested_backend: Some(runtime.requested_backend.as_str().to_string()),
                active_backend: Some(runtime.active_backend.as_str().to_string()),
                generation: Some(runtime.generation),
                active_jobs: runtime.jobs().len(),
            }),
    );
    Json(RuntimeModelsResponse {
        object: "runtime.model.list",
        data,
        image_queue: state.image.inner.queue.status(),
    })
}

fn parse_edit_json(
    request: OpenAiImageEditJsonRequest,
) -> Result<ParsedEditRequest, ImageApiFailure> {
    if request.images.is_empty() {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image edit requires at least one `images` reference",
            Some("images"),
            "missing_parameter",
        ));
    }
    if request.images.len() > MAX_EDIT_SOURCE_IMAGES {
        return Err(ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("image edit accepts at most {MAX_EDIT_SOURCE_IMAGES} ordered source images"),
            Some("images"),
            "image_count_limit",
        ));
    }

    let mut aggregate_bytes = 0usize;
    let mut images = Vec::with_capacity(request.images.len());
    for (index, reference) in request.images.into_iter().enumerate() {
        let parameter = format!("images[{index}]");
        images.push(resolve_edit_image_reference(
            reference,
            &parameter,
            &mut aggregate_bytes,
        )?);
    }
    let mask = request
        .mask
        .map(|reference| resolve_edit_image_reference(reference, "mask", &mut aggregate_bytes))
        .transpose()?;

    Ok(ParsedEditRequest {
        fields: OpenAiImageEditFields {
            prompt: request.prompt,
            model: request.model,
            n: request.n,
            size: request.size,
            quality: request.quality,
            output_format: request.output_format,
            output_compression: request.output_compression,
            background: request.background,
            input_fidelity: request.input_fidelity,
            response_format: None,
            stream: request.stream,
            partial_images: request.partial_images,
            moderation: request.moderation,
            user: request.user,
            x_xeno: request.x_xeno,
        },
        images,
        mask,
    })
}

fn resolve_edit_image_reference(
    reference: OpenAiImageReference,
    parameter: &str,
    aggregate_bytes: &mut usize,
) -> Result<Vec<u8>, ImageApiFailure> {
    let bytes = match (reference.image_url, reference.file_id) {
        (Some(_), Some(_)) | (None, None) => {
            return Err(ImageApiFailure::new(
                StatusCode::BAD_REQUEST,
                format!(
                    "image edit reference `{parameter}` must contain exactly one of `image_url` or `file_id`"
                ),
                Some(parameter),
                "invalid_parameter",
            ))
        }
        (None, Some(file_id)) => {
            if file_id.trim().is_empty() {
                return Err(ImageApiFailure::new(
                    StatusCode::BAD_REQUEST,
                    format!("image edit reference `{parameter}.file_id` must not be empty"),
                    Some(parameter),
                    "invalid_parameter",
                ));
            }
            return Err(ImageApiFailure::new(
                StatusCode::BAD_REQUEST,
                format!(
                    "image edit reference `{parameter}.file_id` requires a configured bounded file resolver"
                ),
                Some(parameter),
                "unsupported_parameter",
            ));
        }
        (Some(image_url), None) => decode_edit_image_url(&image_url, parameter)?,
    };
    add_json_edit_bytes(aggregate_bytes, bytes.len(), parameter)?;
    Ok(bytes)
}

fn decode_edit_image_url(image_url: &str, parameter: &str) -> Result<Vec<u8>, ImageApiFailure> {
    if image_url.len() > MAX_EDIT_IMAGE_URL_BYTES {
        return Err(ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "image edit reference `{parameter}.image_url` exceeds the {MAX_EDIT_IMAGE_URL_BYTES}-byte limit"
            ),
            Some(parameter),
            "image_url_too_large",
        ));
    }
    let (scheme, _) = image_url.split_once(':').ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!(
                "image edit reference `{parameter}.image_url` must be a base64 data URL or a configured HTTPS URL"
            ),
            Some(parameter),
            "invalid_parameter",
        )
    })?;
    if scheme.eq_ignore_ascii_case("https") {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!(
                "image edit reference `{parameter}.image_url` requires a configured bounded HTTPS resolver"
            ),
            Some(parameter),
            "unsupported_parameter",
        ));
    }
    if !scheme.eq_ignore_ascii_case("data") {
        let message = if scheme.eq_ignore_ascii_case("file") {
            format!("local file URLs are never accepted for image edit reference `{parameter}`")
        } else {
            format!(
                "image edit reference `{parameter}.image_url` must be a base64 data URL or a configured HTTPS URL"
            )
        };
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            message,
            Some(parameter),
            "invalid_parameter",
        ));
    }

    let (metadata, encoded) = image_url.split_once(',').ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` is not a valid data URL"),
            Some(parameter),
            "invalid_parameter",
        )
    })?;
    let metadata = metadata.get(5..).ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` is not a valid data URL"),
            Some(parameter),
            "invalid_parameter",
        )
    })?;
    let (mime, encoding) = metadata.rsplit_once(';').ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` must use base64 encoding"),
            Some(parameter),
            "invalid_parameter",
        )
    })?;
    if !encoding.eq_ignore_ascii_case("base64") || mime.contains(';') {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` must use base64 encoding"),
            Some(parameter),
            "invalid_parameter",
        ));
    }
    let declared_format = if mime.eq_ignore_ascii_case("image/png") {
        image::ImageFormat::Png
    } else if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        image::ImageFormat::Jpeg
    } else if mime.eq_ignore_ascii_case("image/webp") {
        image::ImageFormat::WebP
    } else {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!(
                "image edit reference `{parameter}.image_url` must declare image/png, image/jpeg, or image/webp"
            ),
            Some(parameter),
            "invalid_parameter",
        ));
    };
    let bytes = BASE64_STANDARD.decode(encoded).map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` contains invalid base64"),
            Some(parameter),
            "invalid_parameter",
        )
    })?;
    if bytes.is_empty() {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` decodes to an empty image"),
            Some(parameter),
            "invalid_parameter",
        ));
    }
    if bytes.len() > MAX_EDIT_IMAGE_BYTES {
        return Err(ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "image edit reference `{parameter}.image_url` exceeds the {MAX_EDIT_IMAGE_BYTES}-byte decoded limit"
            ),
            Some(parameter),
            "image_too_large",
        ));
    }
    let actual_format = image::guess_format(&bytes).map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit reference `{parameter}.image_url` is not a recognized image"),
            Some(parameter),
            "invalid_image",
        )
    })?;
    if actual_format != declared_format {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!(
                "image edit reference `{parameter}.image_url` MIME type does not match its image bytes"
            ),
            Some(parameter),
            "invalid_image",
        ));
    }
    Ok(bytes)
}

fn add_json_edit_bytes(
    total: &mut usize,
    additional: usize,
    parameter: &str,
) -> Result<(), ImageApiFailure> {
    *total = total.checked_add(additional).ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image edit decoded input byte length overflowed",
            Some(parameter),
            "image_input_limit",
        )
    })?;
    if *total > MAX_EDIT_REQUEST_BYTES {
        return Err(ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image edit decoded inputs exceed the configured aggregate limit",
            Some(parameter),
            "image_input_limit",
        ));
    }
    Ok(())
}

async fn parse_edit_multipart(
    mut multipart: Multipart,
) -> Result<ParsedEditRequest, ImageApiFailure> {
    let mut images = Vec::new();
    let mut mask = None;
    let mut scalars = BTreeMap::<String, String>::new();
    let mut aggregate_bytes = 0usize;
    let mut field_count = 0usize;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("invalid image edit multipart body: {error}"),
            None,
            "invalid_multipart",
        )
    })? {
        field_count = field_count.saturating_add(1);
        if field_count > MAX_EDIT_MULTIPART_FIELDS {
            return Err(ImageApiFailure::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "image edit multipart field count exceeds the configured limit",
                None,
                "multipart_field_limit",
            ));
        }
        let name = field.name().map(str::to_string).ok_or_else(|| {
            ImageApiFailure::new(
                StatusCode::BAD_REQUEST,
                "every image edit multipart part requires a field name",
                None,
                "invalid_multipart",
            )
        })?;
        match name.as_str() {
            "image" | "image[]" => {
                if images.len() >= MAX_EDIT_SOURCE_IMAGES {
                    return Err(ImageApiFailure::new(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!(
                            "image edit accepts at most {MAX_EDIT_SOURCE_IMAGES} ordered source images"
                        ),
                        Some("image"),
                        "image_count_limit",
                    ));
                }
                let bytes = read_multipart_field(field, MAX_EDIT_IMAGE_BYTES, "image").await?;
                add_aggregate_bytes(&mut aggregate_bytes, bytes.len())?;
                images.push(bytes);
            }
            "mask" => {
                if mask.is_some() {
                    return Err(ImageApiFailure::new(
                        StatusCode::BAD_REQUEST,
                        "image edit accepts at most one mask part",
                        Some("mask"),
                        "duplicate_parameter",
                    ));
                }
                let bytes = read_multipart_field(field, MAX_EDIT_IMAGE_BYTES, "mask").await?;
                add_aggregate_bytes(&mut aggregate_bytes, bytes.len())?;
                mask = Some(bytes);
            }
            name if is_edit_scalar(name) => {
                if scalars.contains_key(name) {
                    return Err(ImageApiFailure::new(
                        StatusCode::BAD_REQUEST,
                        format!("duplicate image edit field `{name}`"),
                        Some("multipart"),
                        "duplicate_parameter",
                    ));
                }
                let bytes = read_multipart_field(field, MAX_EDIT_SCALAR_BYTES, name).await?;
                add_aggregate_bytes(&mut aggregate_bytes, bytes.len())?;
                let value = String::from_utf8(bytes).map_err(|_| {
                    ImageApiFailure::new(
                        StatusCode::BAD_REQUEST,
                        format!("image edit field `{name}` must be UTF-8 text"),
                        Some("multipart"),
                        "invalid_parameter",
                    )
                })?;
                scalars.insert(name.to_string(), value);
            }
            _ => {
                return Err(ImageApiFailure::new(
                    StatusCode::BAD_REQUEST,
                    format!("unknown image edit multipart field `{name}`"),
                    Some("multipart"),
                    "unknown_parameter",
                ))
            }
        }
    }

    if images.is_empty() {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image edit requires at least one `image` file part",
            Some("image"),
            "missing_parameter",
        ));
    }
    let fields = edit_fields_from_scalars(scalars)?;
    Ok(ParsedEditRequest {
        fields,
        images,
        mask,
    })
}

async fn read_multipart_field(
    mut field: Field<'_>,
    limit: usize,
    parameter: &str,
) -> Result<Vec<u8>, ImageApiFailure> {
    let mut bytes = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(|error| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("failed to read image edit multipart field: {error}"),
            Some(parameter),
            "invalid_multipart",
        )
    })? {
        let next_len = bytes.len().checked_add(chunk.len()).ok_or_else(|| {
            ImageApiFailure::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "multipart field byte length overflowed",
                Some(parameter),
                "multipart_field_limit",
            )
        })?;
        if next_len > limit {
            return Err(ImageApiFailure::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("image edit field `{parameter}` exceeds the {limit}-byte limit"),
                Some(parameter),
                "multipart_field_limit",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit field `{parameter}` must not be empty"),
            Some(parameter),
            "invalid_parameter",
        ));
    }
    Ok(bytes)
}

fn add_aggregate_bytes(total: &mut usize, additional: usize) -> Result<(), ImageApiFailure> {
    *total = total.checked_add(additional).ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "multipart request byte length overflowed",
            None,
            "multipart_body_limit",
        )
    })?;
    if *total > MAX_EDIT_REQUEST_BYTES {
        return Err(ImageApiFailure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image edit multipart body exceeds the configured aggregate limit",
            None,
            "multipart_body_limit",
        ));
    }
    Ok(())
}

fn is_edit_scalar(name: &str) -> bool {
    matches!(
        name,
        "prompt"
            | "model"
            | "n"
            | "size"
            | "quality"
            | "output_format"
            | "output_compression"
            | "background"
            | "input_fidelity"
            | "response_format"
            | "stream"
            | "partial_images"
            | "moderation"
            | "user"
            | "x_xeno"
    )
}

fn edit_fields_from_scalars(
    scalars: BTreeMap<String, String>,
) -> Result<OpenAiImageEditFields, ImageApiFailure> {
    let mut object = serde_json::Map::new();
    for (name, value) in scalars {
        let value = match name.as_str() {
            "n" => serde_json::json!(parse_multipart_usize(&name, &value)?),
            "output_compression" | "partial_images" => {
                serde_json::json!(parse_multipart_u8(&name, &value)?)
            }
            "stream" => serde_json::Value::Bool(match value.as_str() {
                "true" => true,
                "false" => false,
                _ => {
                    return Err(ImageApiFailure::new(
                        StatusCode::BAD_REQUEST,
                        "image edit field `stream` must be `true` or `false`",
                        Some("stream"),
                        "invalid_parameter",
                    ))
                }
            }),
            "x_xeno" => serde_json::from_str(&value).map_err(|error| {
                ImageApiFailure::new(
                    StatusCode::BAD_REQUEST,
                    format!("invalid x_xeno JSON: {error}"),
                    Some("x_xeno"),
                    "invalid_parameter",
                )
            })?,
            _ => serde_json::Value::String(value),
        };
        object.insert(name, value);
    }
    serde_json::from_value(serde_json::Value::Object(object)).map_err(|error| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("invalid image edit field: {error}"),
            Some("multipart"),
            "invalid_parameter",
        )
    })
}

fn parse_multipart_usize(name: &str, value: &str) -> Result<usize, ImageApiFailure> {
    value.parse::<usize>().map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit field `{name}` must be an unsigned integer"),
            Some("multipart"),
            "invalid_parameter",
        )
    })
}

fn parse_multipart_u8(name: &str, value: &str) -> Result<u8, ImageApiFailure> {
    value.parse::<u8>().map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("image edit field `{name}` must be an integer between 0 and 255"),
            Some("multipart"),
            "invalid_parameter",
        )
    })
}

#[derive(Debug, Clone)]
struct PreparedResponseFields {
    output_format: OpenAiImageFormat,
    quality: OpenAiImageResponseQuality,
    size: Option<OpenAiImageResponseSize>,
    background: OpenAiImageResponseBackground,
    stream: bool,
    partial_images: u8,
}

fn prepare_generation_request(
    request: OpenAiImageGenerationRequest,
    model: &str,
) -> Result<(ImageGenerationRequest, PreparedResponseFields), ImageApiFailure> {
    let stream = request.stream.unwrap_or(false);
    let partial_images = request.partial_images.unwrap_or(0);
    if partial_images > MAX_PARTIAL_IMAGES {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            format!("partial_images must be between 0 and {MAX_PARTIAL_IMAGES}"),
            Some("partial_images"),
            "invalid_parameter",
        ));
    }
    if request.partial_images.is_some() && !stream {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "partial_images requires stream=true",
            Some("partial_images"),
            "invalid_parameter",
        ));
    }
    if stream && !IMAGE_STREAM_USAGE_METERING_AVAILABLE {
        return Err(unsupported(
            "stream",
            "image streaming requires versioned token-equivalent usage metering, which is not available for this runtime profile",
        ));
    }
    if stream && request.n.unwrap_or(1) != 1 {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "streaming image requests require n=1 because the pinned event schema has no output index",
            Some("n"),
            "unsupported_value",
        ));
    }
    if request.moderation.is_some() {
        return Err(unsupported(
            "moderation",
            "no local image moderation implementation is configured",
        ));
    }
    if request.style.is_some() {
        return Err(unsupported(
            "style",
            "style is not supported by the Qwen Image profile",
        ));
    }
    if request.response_format == Some(OpenAiImageResponseFormat::Url) {
        return Err(unsupported(
            "response_format",
            "URL image output requires a configured bounded output store",
        ));
    }
    if request.output_compression.is_some() {
        return Err(unsupported(
            "output_compression",
            "configurable image compression is not admitted yet",
        ));
    }
    let background = match request.background.unwrap_or(OpenAiImageBackground::Auto) {
        OpenAiImageBackground::Auto | OpenAiImageBackground::Opaque => {
            OpenAiImageResponseBackground::Opaque
        }
        OpenAiImageBackground::Transparent => {
            return Err(unsupported(
                "background",
                "transparent image generation is not supported by the Qwen Image profile",
            ))
        }
    };
    if request
        .x_xeno
        .as_ref()
        .and_then(|options| options.allow_noop)
        .is_some()
    {
        return Err(unsupported(
            "x_xeno.allow_noop",
            "allow_noop is valid only for image edits",
        ));
    }

    let (width, height) = parse_size(request.size.as_deref().unwrap_or("auto"))?;
    let output_format = request.output_format.unwrap_or(OpenAiImageFormat::Png);
    let requested_quality = request.quality.unwrap_or(OpenAiImageQuality::Auto);
    let default_steps = match requested_quality {
        OpenAiImageQuality::Low => 20,
        OpenAiImageQuality::Medium | OpenAiImageQuality::Standard => 35,
        OpenAiImageQuality::Auto | OpenAiImageQuality::High | OpenAiImageQuality::Hd => 50,
    };
    let options = request.x_xeno.as_ref();
    let steps = options
        .and_then(|options| options.steps)
        .unwrap_or(default_steps);
    let requested_preview_interval = options.and_then(|options| options.preview_interval_steps);
    if requested_preview_interval.is_some() && (!stream || partial_images == 0) {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "x_xeno.preview_interval_steps requires stream=true and partial_images greater than zero",
            Some("x_xeno.preview_interval_steps"),
            "invalid_parameter",
        ));
    }
    if partial_images > 0 && steps <= usize::from(partial_images) {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "the denoising step count must leave one distinct checkpoint per requested partial image",
            Some("partial_images"),
            "invalid_parameter",
        ));
    }
    let preview_interval = if partial_images == 0 {
        None
    } else {
        let interval = requested_preview_interval
            .unwrap_or_else(|| steps.div_ceil(usize::from(partial_images) + 1));
        if interval == 0 || (steps - 1) / interval < usize::from(partial_images) {
            return Err(ImageApiFailure::new(
                StatusCode::BAD_REQUEST,
                "x_xeno.preview_interval_steps cannot produce the requested partial image count before completion",
                Some("x_xeno.preview_interval_steps"),
                "invalid_parameter",
            ));
        }
        Some(interval)
    };
    let backend = match options.and_then(|options| options.backend) {
        None | Some(OpenAiXenoImageBackend::Auto) => ImageBackendKind::Auto,
        Some(OpenAiXenoImageBackend::Cpu) => ImageBackendKind::Cpu,
        Some(OpenAiXenoImageBackend::Cuda) => ImageBackendKind::Cuda,
    };
    let offload = match options.and_then(|options| options.offload) {
        None | Some(OpenAiXenoImageOffload::Sequential) => ImageOffloadPolicy::Sequential,
        Some(OpenAiXenoImageOffload::None) => ImageOffloadPolicy::None,
        Some(OpenAiXenoImageOffload::Balanced) => ImageOffloadPolicy::Balanced,
        Some(OpenAiXenoImageOffload::Cpu) => ImageOffloadPolicy::Cpu,
    };
    let resize_policy = match options.and_then(|options| options.resize_policy) {
        None | Some(OpenAiXenoResizePolicy::Reject) => ImageResizePolicy::Reject,
        Some(OpenAiXenoResizePolicy::RoundDown) => ImageResizePolicy::RoundDown,
    };
    let quality = match requested_quality {
        OpenAiImageQuality::Low => OpenAiImageResponseQuality::Low,
        OpenAiImageQuality::Medium | OpenAiImageQuality::Standard => {
            OpenAiImageResponseQuality::Medium
        }
        OpenAiImageQuality::Auto | OpenAiImageQuality::High | OpenAiImageQuality::Hd => {
            OpenAiImageResponseQuality::High
        }
    };
    let internal_quality = match quality {
        OpenAiImageResponseQuality::High => ImageQuality::Hd,
        OpenAiImageResponseQuality::Low | OpenAiImageResponseQuality::Medium => {
            ImageQuality::Standard
        }
    };
    let internal_format = match output_format {
        OpenAiImageFormat::Png => ImageOutputFormat::Png,
        OpenAiImageFormat::Jpeg => ImageOutputFormat::Jpeg,
        OpenAiImageFormat::Webp => ImageOutputFormat::Webp,
    };
    let size = OpenAiImageResponseSize::from_dimensions(width, height);
    Ok((
        ImageGenerationRequest {
            model: model.to_string(),
            prompt: request.prompt,
            negative_prompt: options.and_then(|options| options.negative_prompt.clone()),
            width,
            height,
            n: request.n.unwrap_or(1),
            steps,
            true_cfg_scale: options
                .and_then(|options| options.true_cfg_scale)
                .unwrap_or(4.0),
            seed: options
                .and_then(|options| options.seed)
                .unwrap_or_else(rand::random),
            output_format: internal_format,
            quality: internal_quality,
            backend,
            offload,
            resize_policy,
            preview_interval,
        },
        PreparedResponseFields {
            output_format,
            quality,
            size,
            background,
            stream,
            partial_images,
        },
    ))
}

fn prepare_edit_request(
    fields: OpenAiImageEditFields,
    model: &str,
    images: Vec<DecodedImage>,
    mask: Option<DecodedImage>,
    default_dimensions: (u32, u32),
) -> Result<(ImageEditRequest, PreparedResponseFields), ImageApiFailure> {
    images.first().ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image edit requires at least one decoded source image",
            Some("image"),
            "missing_parameter",
        )
    })?;
    let size = match fields.size.as_deref() {
        None | Some("auto") => Some(format!("{}x{}", default_dimensions.0, default_dimensions.1)),
        Some(size) => Some(size.to_string()),
    };
    let mut options = fields.x_xeno.clone();
    if let Some(options) = &mut options {
        options.allow_noop = None;
    }
    let generation = OpenAiImageGenerationRequest {
        prompt: fields.prompt,
        model: fields.model,
        n: fields.n,
        size,
        quality: fields.quality,
        output_format: fields.output_format,
        output_compression: fields.output_compression,
        background: fields.background,
        response_format: fields.response_format,
        stream: fields.stream,
        partial_images: fields.partial_images,
        moderation: fields.moderation,
        style: None,
        user: fields.user,
        x_xeno: options,
    };
    let (generation, response_fields) = prepare_generation_request(generation, model)?;
    Ok((
        ImageEditRequest {
            generation,
            images,
            mask,
            strength: 1.0,
        },
        response_fields,
    ))
}

fn image_batch_response(
    result: ImageBatchResult,
    response_fields: PreparedResponseFields,
) -> Response {
    let response_size = result
        .images
        .first()
        .and_then(|image| OpenAiImageResponseSize::from_dimensions(image.width, image.height));
    let mut aggregate_bytes = 0usize;
    let mut data = Vec::with_capacity(result.images.len());
    for image in result.images {
        let encoded_len = image
            .bytes
            .len()
            .checked_add(2)
            .and_then(|length| length.checked_div(3))
            .and_then(|groups| groups.checked_mul(4));
        aggregate_bytes = match encoded_len.and_then(|length| aggregate_bytes.checked_add(length)) {
            Some(length) if length <= MAX_BASE64_RESPONSE_BYTES => length,
            _ => {
                return api_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "encoded image response exceeds the configured aggregate limit",
                    None,
                    "image_response_too_large",
                )
            }
        };
        data.push(OpenAiImageData {
            b64_json: Some(BASE64_STANDARD.encode(image.bytes)),
            url: None,
            revised_prompt: None,
        });
    }
    Json(OpenAiImageResponse {
        created: unix_timestamp(),
        data,
        output_format: Some(response_fields.output_format),
        quality: Some(response_fields.quality),
        size: response_size,
        background: Some(response_fields.background),
        usage: None,
    })
    .into_response()
}

fn parse_size(value: &str) -> Result<(u32, u32), ImageApiFailure> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok((1024, 1024));
    }
    let (width, height) = value.split_once(['x', 'X']).ok_or_else(|| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "size must be `auto` or WIDTHxHEIGHT",
            Some("size"),
            "image_dimensions_unsupported",
        )
    })?;
    let width = width.parse::<u32>().map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image width must be an unsigned integer",
            Some("size"),
            "image_dimensions_unsupported",
        )
    })?;
    let height = height.parse::<u32>().map_err(|_| {
        ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image height must be an unsigned integer",
            Some("size"),
            "image_dimensions_unsupported",
        )
    })?;
    if width == 0 || height == 0 {
        return Err(ImageApiFailure::new(
            StatusCode::BAD_REQUEST,
            "image dimensions must be non-zero",
            Some("size"),
            "image_dimensions_unsupported",
        ));
    }
    Ok((width, height))
}

fn map_image_input_error(error: ImageError) -> Response {
    match error.kind() {
        ImageErrorKind::Codec => api_error(
            StatusCode::BAD_REQUEST,
            error.to_string(),
            Some("image"),
            "invalid_image",
        ),
        ImageErrorKind::InputLimit => api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            error.to_string(),
            Some("image"),
            "image_input_limit",
        ),
        _ => map_image_error(error),
    }
}

fn map_image_error(error: ImageError) -> Response {
    let kind = error.kind();
    let (status, code, param, expose) = match kind {
        ImageErrorKind::InvalidRequest => {
            (StatusCode::BAD_REQUEST, "invalid_image_request", None, true)
        }
        ImageErrorKind::UnsupportedCapability => (
            StatusCode::BAD_REQUEST,
            "unsupported_capability",
            None,
            true,
        ),
        ImageErrorKind::UnsupportedQuantization => (
            StatusCode::BAD_REQUEST,
            "unsupported_quantization",
            Some("model"),
            true,
        ),
        ImageErrorKind::UnsupportedTensor => (
            StatusCode::BAD_REQUEST,
            "unsupported_tensor",
            Some("model"),
            false,
        ),
        ImageErrorKind::UnsupportedShape => (
            StatusCode::BAD_REQUEST,
            "image_dimensions_unsupported",
            Some("size"),
            true,
        ),
        ImageErrorKind::UnsupportedBackend => (
            StatusCode::BAD_REQUEST,
            "unsupported_backend",
            Some("x_xeno.backend"),
            true,
        ),
        ImageErrorKind::Admission | ImageErrorKind::InsufficientMemory => {
            (StatusCode::CONFLICT, "image_admission_failed", None, true)
        }
        ImageErrorKind::Cancelled => (
            StatusCode::from_u16(499).expect("499 is a valid extension status"),
            "image_cancelled",
            None,
            true,
        ),
        ImageErrorKind::Codec => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_codec_failed",
            None,
            false,
        ),
        ImageErrorKind::InputLimit => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "image_input_limit",
            None,
            true,
        ),
        ImageErrorKind::MissingComponent
        | ImageErrorKind::CorruptComponent
        | ImageErrorKind::Checksum
        | ImageErrorKind::Manifest
        | ImageErrorKind::Execution
        | ImageErrorKind::Numerical
        | ImageErrorKind::Internal => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_execution_failed",
            None,
            false,
        ),
    };
    let message = if expose {
        error.to_string()
    } else {
        "image generation failed".to_string()
    };
    api_error(status, message, param, code)
}

fn unsupported(param: &'static str, message: &'static str) -> ImageApiFailure {
    ImageApiFailure::new(
        StatusCode::BAD_REQUEST,
        message,
        Some(param),
        "unsupported_parameter",
    )
}

fn api_error(
    status: StatusCode,
    message: impl Into<String>,
    param: Option<&str>,
    code: &str,
) -> Response {
    let error_type = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: message.into(),
                error_type: error_type.to_string(),
                param: param.map(ToOwned::to_owned),
                code: Some(code.to_string()),
            },
        }),
    )
        .into_response()
}

fn authorize_bearer(headers: &HeaderMap, expected: &[u8]) -> Result<(), String> {
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| {
            let (scheme, token) = value.split_once(' ')?;
            scheme
                .eq_ignore_ascii_case("bearer")
                .then_some(token.trim())
        })
        .map(str::as_bytes)
        .unwrap_or_default();
    if constant_time_eq(supplied, expected) {
        Ok(())
    } else {
        Err("missing or invalid bearer token".to_string())
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..MAX_API_KEY_BYTES {
        difference |= usize::from(*left.get(index).unwrap_or(&0) ^ *right.get(index).unwrap_or(&0));
    }
    difference == 0
}

fn is_loopback_bind_host(host: &str) -> bool {
    let host = host.trim().trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn env_truthy(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_usize(name: &str, default: usize, minimum: usize, maximum: usize) -> Result<usize, String> {
    let value = match env::var(name) {
        Ok(value) if !value.trim().is_empty() => value
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("{name} must be an integer between {minimum} and {maximum}"))?,
        _ => default,
    };
    if !(minimum..=maximum).contains(&value) {
        return Err(format!(
            "{name} must be between {minimum} and {maximum}, found {value}"
        ));
    }
    Ok(value)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        extract::{DefaultBodyLimit, State},
        http::{HeaderValue, Request},
        routing::post,
        Router,
    };
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use xrt_runtime::{BackendKind, RequestScheduler, SchedulerConfig};

    fn test_state(image: ImageServerState) -> AppState {
        let resources = Arc::clone(&image.inner.resources);
        AppState {
            runtime: Arc::new(RwLock::new(None)),
            external_openai: Arc::new(RwLock::new(None)),
            requested_backend: Arc::new(RwLock::new(BackendKind::Auto)),
            loaded_model_name: Arc::new(RwLock::new(None)),
            loaded_model_path: Arc::new(RwLock::new(None)),
            loaded_mmproj_path: Arc::new(RwLock::new(None)),
            gpu_resources: resources,
            scheduler: Arc::new(RequestScheduler::new(
                SchedulerConfig::new(1, 1, 2).unwrap(),
            )),
            stream_buffer_capacity: 2,
            image,
        }
    }

    fn test_image_state() -> ImageServerState {
        ImageServerState::new(
            Arc::new(GpuResourceManager::from_env()),
            1,
            1,
            ImageApiAuth::loopback_without_key(),
        )
    }

    fn fixture_png(red: u8, green: u8, blue: u8) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(32 * 32 * 4);
        for _ in 0..32 * 32 {
            rgba.extend_from_slice(&[red, green, blue, 255]);
        }
        let image = DecodedImage::new_rgba8(32, 32, rgba).unwrap();
        xrt_image::encode_image(&image, ImageOutputFormat::Png, 90, 1024 * 1024).unwrap()
    }

    fn fixture_data_url(bytes: &[u8]) -> String {
        format!("data:image/png;base64,{}", BASE64_STANDARD.encode(bytes))
    }

    fn append_multipart_text(body: &mut Vec<u8>, boundary: &str, name: &str, value: &str) {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    fn append_multipart_file(
        body: &mut Vec<u8>,
        boundary: &str,
        name: &str,
        filename: &str,
        bytes: &[u8],
    ) {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }

    fn edit_multipart_body(model: &str, include_mask: bool, stream: bool) -> (String, Vec<u8>) {
        let boundary = "xrt-image-edit-boundary".to_string();
        let mut body = Vec::new();
        append_multipart_text(&mut body, &boundary, "prompt", "make it cobalt");
        append_multipart_text(&mut body, &boundary, "model", model);
        append_multipart_text(&mut body, &boundary, "output_format", "png");
        append_multipart_text(&mut body, &boundary, "response_format", "b64_json");
        append_multipart_text(
            &mut body,
            &boundary,
            "x_xeno",
            r#"{"seed":17,"steps":2,"backend":"cpu"}"#,
        );
        if stream {
            append_multipart_text(&mut body, &boundary, "stream", "true");
            append_multipart_text(&mut body, &boundary, "partial_images", "1");
        }
        append_multipart_file(
            &mut body,
            &boundary,
            "image",
            "source.png",
            &fixture_png(255, 0, 0),
        );
        if include_mask {
            append_multipart_file(
                &mut body,
                &boundary,
                "mask",
                "mask.png",
                &fixture_png(255, 255, 255),
            );
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (boundary, body)
    }

    async fn edit_request(model: &str, include_mask: bool) -> Response {
        let image = test_image_state();
        image.install_synthetic_for_test().await;
        let state = test_state(image);
        let app = Router::new()
            .route(
                "/v1/images/edits",
                post(image_edits).layer(DefaultBodyLimit::max(MAX_EDIT_REQUEST_BYTES)),
            )
            .with_state(state);
        let (boundary, body) = edit_multipart_body(model, include_mask, false);
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn edit_json_request(body: serde_json::Value) -> Response {
        let image = test_image_state();
        image.install_synthetic_for_test().await;
        let state = test_state(image);
        let app = Router::new()
            .route(
                "/v1/images/edits",
                post(image_edits).layer(DefaultBodyLimit::max(MAX_EDIT_REQUEST_BYTES)),
            )
            .with_state(state);
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header(header::CONTENT_TYPE, "application/json; charset=utf-8")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[test]
    fn auth_requires_exact_bearer_token_when_configured() {
        let auth = ImageApiAuth {
            api_key: Some(Arc::from(b"secret".as_slice())),
            loopback_bind: true,
            allow_unauthenticated_generation: false,
        };
        let mut headers = HeaderMap::new();
        assert!(auth.authorize_generation(&headers).is_err());
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong"),
        );
        assert!(auth.authorize_generation(&headers).is_err());
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        assert!(auth.authorize_generation(&headers).is_ok());
    }

    #[test]
    fn generation_conversion_rejects_unimplemented_controls() {
        let request: OpenAiImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "prompt": "test",
            "response_format": "url"
        }))
        .unwrap();
        let error = prepare_generation_request(request, "fixture").unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn generation_conversion_normalizes_response_domains() {
        for (requested, expected) in [
            ("auto", OpenAiImageResponseQuality::High),
            ("low", OpenAiImageResponseQuality::Low),
            ("medium", OpenAiImageResponseQuality::Medium),
            ("standard", OpenAiImageResponseQuality::Medium),
            ("high", OpenAiImageResponseQuality::High),
            ("hd", OpenAiImageResponseQuality::High),
        ] {
            let request: OpenAiImageGenerationRequest = serde_json::from_value(serde_json::json!({
                "model": "fixture",
                "prompt": "test",
                "quality": requested,
                "background": "auto",
                "size": "32x32"
            }))
            .unwrap();
            let (_, fields) = prepare_generation_request(request, "fixture").unwrap();
            assert_eq!(fields.quality, expected, "quality={requested}");
            assert_eq!(fields.background, OpenAiImageResponseBackground::Opaque);
            assert_eq!(fields.size, None);
        }

        let request: OpenAiImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "prompt": "test",
            "size": "1024x1536"
        }))
        .unwrap();
        let (_, fields) = prepare_generation_request(request, "fixture").unwrap();
        assert_eq!(
            fields.size,
            Some(OpenAiImageResponseSize::Portrait1024x1536)
        );
    }

    #[tokio::test]
    async fn synchronous_generation_returns_ordered_openai_base64_images() {
        let image = test_image_state();
        let summary = image.install_synthetic_for_test().await;
        let state = test_state(image);
        let request: OpenAiImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": summary.loaded_model,
            "prompt": "a cobalt keyboard",
            "n": 2,
            "size": "32x32",
            "quality": "low",
            "output_format": "png",
            "response_format": "b64_json",
            "x_xeno": {"seed": 41, "steps": 2, "backend": "cpu"}
        }))
        .unwrap();
        let response = image_generations(State(state), HeaderMap::new(), Ok(Json(request))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), MAX_BASE64_RESPONSE_BYTES)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["data"].as_array().unwrap().len(), 2);
        assert_eq!(value["quality"], "low");
        assert_eq!(value["background"], "opaque");
        assert!(value.get("size").is_none());
        for entry in value["data"].as_array().unwrap() {
            let bytes = BASE64_STANDARD
                .decode(entry["b64_json"].as_str().unwrap())
                .unwrap();
            let decoded = image::load_from_memory(&bytes).unwrap();
            assert_eq!((decoded.width(), decoded.height()), (32, 32));
        }
    }

    #[tokio::test]
    async fn streaming_generation_rejects_without_usage_metering() {
        let image = test_image_state();
        let summary = image.install_synthetic_for_test().await;
        let state = test_state(image);
        let request: OpenAiImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": summary.loaded_model,
            "prompt": "a cobalt keyboard",
            "size": "32x32",
            "quality": "low",
            "output_format": "png",
            "stream": true,
            "partial_images": 1,
            "x_xeno": {"seed": 41, "steps": 2, "backend": "cpu"}
        }))
        .unwrap();
        let response = image_generations(State(state), HeaderMap::new(), Ok(Json(request))).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["param"], "stream");
        assert_eq!(value["error"]["code"], "unsupported_parameter");
    }

    #[tokio::test]
    async fn multipart_edit_executes_the_ordered_synthetic_edit_contract() {
        let response = edit_request("xrt-image-synthetic-v1", false).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), MAX_BASE64_RESPONSE_BYTES)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["data"].as_array().unwrap().len(), 1);
        assert_eq!(value["quality"], "high");
        assert_eq!(value["background"], "opaque");
        assert!(value.get("size").is_none());
        let bytes = BASE64_STANDARD
            .decode(value["data"][0]["b64_json"].as_str().unwrap())
            .unwrap();
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (32, 32));
    }

    #[tokio::test]
    async fn json_edit_data_urls_preserve_order_and_execute() {
        let first = fixture_png(255, 0, 0);
        let second = fixture_png(0, 0, 255);
        let body = serde_json::json!({
            "images": [
                {"image_url": fixture_data_url(&first)},
                {"image_url": fixture_data_url(&second)}
            ],
            "prompt": "make it cobalt",
            "model": "xrt-image-synthetic-v1",
            "output_format": "png",
            "x_xeno": {"seed": 17, "steps": 2, "backend": "cpu"}
        });
        let parsed = parse_edit_json(serde_json::from_value(body.clone()).unwrap()).unwrap();
        assert_eq!(parsed.images, vec![first, second]);

        let response = edit_json_request(body).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), MAX_BASE64_RESPONSE_BYTES)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["data"].as_array().unwrap().len(), 1);
        assert_eq!(value["quality"], "high");
        assert_eq!(value["background"], "opaque");
        assert!(value.get("size").is_none());
    }

    #[test]
    fn json_edit_data_urls_validate_reference_shape_base64_and_mime() {
        let png = fixture_png(255, 0, 0);
        let both: OpenAiImageEditJsonRequest = serde_json::from_value(serde_json::json!({
            "images": [{
                "image_url": fixture_data_url(&png),
                "file_id": "file_fixture"
            }],
            "prompt": "fixture"
        }))
        .unwrap();
        let error = parse_edit_json(both).unwrap_err();
        assert_eq!(error.code, "invalid_parameter");
        assert_eq!(error.param.as_deref(), Some("images[0]"));

        let mismatched: OpenAiImageEditJsonRequest = serde_json::from_value(serde_json::json!({
            "images": [{
                "image_url": format!(
                    "data:image/jpeg;base64,{}",
                    BASE64_STANDARD.encode(&png)
                )
            }],
            "prompt": "fixture"
        }))
        .unwrap();
        let error = parse_edit_json(mismatched).unwrap_err();
        assert_eq!(error.code, "invalid_image");

        let invalid_base64: OpenAiImageEditJsonRequest =
            serde_json::from_value(serde_json::json!({
                "images": [{"image_url": "data:image/png;base64,not-base64"}],
                "prompt": "fixture"
            }))
            .unwrap();
        let error = parse_edit_json(invalid_base64).unwrap_err();
        assert_eq!(error.code, "invalid_parameter");
    }

    #[tokio::test]
    async fn json_edit_references_fail_closed_without_external_resolvers() {
        for (reference, code) in [
            (
                serde_json::json!({"image_url": "https://example.invalid/source.png"}),
                "unsupported_parameter",
            ),
            (
                serde_json::json!({"file_id": "file_fixture"}),
                "unsupported_parameter",
            ),
            (
                serde_json::json!({"image_url": "file:///tmp/source.png"}),
                "invalid_parameter",
            ),
        ] {
            let response = edit_json_request(serde_json::json!({
                "images": [reference],
                "prompt": "fixture",
                "model": "xrt-image-synthetic-v1"
            }))
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["error"]["param"], "images[0]");
            assert_eq!(value["error"]["code"], code);
        }
    }

    #[tokio::test]
    async fn json_edit_recognizes_unsupported_moderation_and_mask() {
        let image_url = fixture_data_url(&fixture_png(255, 0, 0));
        let moderation = edit_json_request(serde_json::json!({
            "images": [{"image_url": image_url}],
            "prompt": "fixture",
            "model": "xrt-image-synthetic-v1",
            "moderation": "auto"
        }))
        .await;
        assert_eq!(moderation.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(moderation.into_body(), 4096).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["param"], "moderation");
        assert_eq!(value["error"]["code"], "unsupported_parameter");

        let mask = edit_json_request(serde_json::json!({
            "images": [{"image_url": image_url}],
            "mask": {"image_url": fixture_data_url(&fixture_png(255, 255, 255))},
            "prompt": "fixture",
            "model": "xrt-image-synthetic-v1"
        }))
        .await;
        assert_eq!(mask.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(mask.into_body(), 4096).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["param"], "mask");
        assert_eq!(value["error"]["code"], "unsupported_parameter");
    }

    #[tokio::test]
    async fn multipart_edit_stream_rejects_without_usage_metering() {
        let image = test_image_state();
        image.install_synthetic_for_test().await;
        let state = test_state(image);
        let app = Router::new()
            .route(
                "/v1/images/edits",
                post(image_edits).layer(DefaultBodyLimit::max(MAX_EDIT_REQUEST_BYTES)),
            )
            .with_state(state);
        let (boundary, body) = edit_multipart_body("xrt-image-synthetic-v1", false, true);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/edits")
                    .header(
                        header::CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["param"], "stream");
        assert_eq!(value["error"]["code"], "unsupported_parameter");
    }

    #[tokio::test]
    async fn multipart_edit_rejects_masks_without_inpaint_capability() {
        let response = edit_request("xrt-image-synthetic-v1", true).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["param"], "mask");
        assert_eq!(value["error"]["code"], "unsupported_parameter");
    }

    #[tokio::test]
    async fn unload_removes_new_routing_and_preserves_generation_identity() {
        let image = test_image_state();
        let summary = image.install_synthetic_for_test().await;
        let unloaded = image
            .unload(Some(&summary.loaded_model), false)
            .await
            .unwrap();
        assert_eq!(unloaded.generation, summary.generation);
        assert_eq!(unloaded.state, "draining");
        assert!(image.openai_models().await.is_empty());
    }

    #[tokio::test]
    async fn force_unload_cancels_pinned_jobs() {
        let image = test_image_state();
        let summary = image.install_synthetic_for_test().await;
        let runtime = image
            .select_runtime(Some(&summary.loaded_model), ImageCapability::Generate)
            .await
            .unwrap();
        let job = runtime.pin_job().unwrap();
        let cancellation = job.cancellation();
        let unloaded = image
            .unload(Some(&summary.loaded_model), true)
            .await
            .unwrap();
        assert_eq!(unloaded.active_jobs, 1);
        assert_eq!(unloaded.cancelled_jobs, 1);
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn queue_rejects_work_beyond_its_active_and_waiting_bounds() {
        let queue = Arc::new(ImageExecutionQueue::new(1, 1));
        let first = queue.acquire().await.unwrap();
        let queued = tokio::spawn({
            let queue = Arc::clone(&queue);
            async move { queue.acquire().await }
        });
        for _ in 0..100 {
            if queue.status().queued_jobs == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            queue.status(),
            ImageQueueStatus {
                max_active_jobs: 1,
                max_queued_jobs: 1,
                active_jobs: 1,
                queued_jobs: 1,
            }
        );
        let rejected = match queue.acquire().await {
            Ok(_) => panic!("work beyond the queue bound must be rejected"),
            Err(error) => error,
        };
        assert_eq!(rejected.status, StatusCode::TOO_MANY_REQUESTS);
        drop(first);
        let second = queued.await.unwrap().unwrap();
        assert_eq!(queue.status().active_jobs, 1);
        assert_eq!(queue.status().queued_jobs, 0);
        drop(second);
        assert_eq!(queue.status().active_jobs, 0);
        assert_eq!(queue.status().queued_jobs, 0);
    }
}
