use axum::{
    body::{Body, Bytes},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde_json::Value;
use std::{
    env,
    io::{self, Read, Take},
    net::IpAddr,
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

const DEFAULT_TIMEOUT_SECONDS: u64 = 300;
const MAX_TIMEOUT_SECONDS: u64 = 3_600;
const MAX_BUFFERED_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
const STREAM_CHUNK_BYTES: usize = 16 * 1024;

pub(crate) type HandlerError = (StatusCode, String);

#[derive(Clone)]
pub(crate) struct ExternalOpenAiConfig {
    base_url: String,
    api_key: Option<String>,
    default_model: Option<String>,
    timeout: Duration,
}

impl std::fmt::Debug for ExternalOpenAiConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalOpenAiConfig")
            .field("base_url", &self.base_url)
            .field("api_key_configured", &self.api_key.is_some())
            .field("default_model", &self.default_model)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl ExternalOpenAiConfig {
    pub(crate) fn from_env_with_overrides(
        base_url: Option<&str>,
        api_key: Option<&str>,
        default_model: Option<&str>,
    ) -> Result<Self, String> {
        let base_url = nonempty(base_url)
            .map(ToOwned::to_owned)
            .or_else(|| env_nonempty("XRT_EXTERNAL_BASE_URL"))
            .ok_or_else(|| {
                "external-openai requires XRT_EXTERNAL_BASE_URL or external_base_url".to_string()
            })?;
        let api_key = nonempty(api_key)
            .map(ToOwned::to_owned)
            .or_else(|| env_nonempty("XRT_EXTERNAL_API_KEY"));
        let default_model = nonempty(default_model)
            .map(ToOwned::to_owned)
            .or_else(|| env_nonempty("XRT_EXTERNAL_MODEL"));
        let allow_remote = env_truthy("XRT_EXTERNAL_ALLOW_REMOTE");
        let timeout_seconds = match env_nonempty("XRT_EXTERNAL_TIMEOUT_SECONDS") {
            Some(raw) => raw.parse::<u64>().map_err(|_| {
                format!("XRT_EXTERNAL_TIMEOUT_SECONDS must be an integer, found `{raw}`")
            })?,
            None => DEFAULT_TIMEOUT_SECONDS,
        };
        Self::new(
            base_url,
            api_key,
            default_model,
            allow_remote,
            timeout_seconds,
        )
    }

    pub(crate) fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        default_model: Option<String>,
        allow_remote: bool,
        timeout_seconds: u64,
    ) -> Result<Self, String> {
        let base_url = normalize_base_url(&base_url.into(), allow_remote)?;
        if !(1..=MAX_TIMEOUT_SECONDS).contains(&timeout_seconds) {
            return Err(format!(
                "external OpenAI timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds, found {timeout_seconds}"
            ));
        }
        Ok(Self {
            base_url,
            api_key: api_key.and_then(normalize_optional),
            default_model: default_model.and_then(normalize_optional),
            timeout: Duration::from_secs(timeout_seconds),
        })
    }

    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    pub(crate) fn display_model(&self) -> &str {
        self.default_model().unwrap_or("external-openai")
    }

    pub(crate) fn prepare_payload(&self, mut payload: Value) -> Result<Value, HandlerError> {
        let object = payload.as_object_mut().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "OpenAI request body must be a JSON object".to_string(),
            )
        })?;
        let has_model = object
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(|model| !model.trim().is_empty());
        if !has_model {
            let model = self.default_model().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "external-openai request requires `model` or XRT_EXTERNAL_MODEL".to_string(),
                )
            })?;
            object.insert("model".to_string(), Value::String(model.to_string()));
        }
        Ok(payload)
    }

    fn endpoint(&self, relative_path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url,
            relative_path.trim_start_matches('/')
        )
    }

    fn agent(&self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .redirects(0)
            .timeout_connect(self.timeout.min(Duration::from_secs(30)))
            .timeout_read(self.timeout)
            .timeout_write(self.timeout)
            .build()
    }

    fn authorize(&self, request: ureq::Request) -> ureq::Request {
        match &self.api_key {
            Some(api_key) => request.set("Authorization", &format!("Bearer {api_key}")),
            None => request,
        }
    }
}

pub(crate) async fn proxy_json(
    config: ExternalOpenAiConfig,
    relative_path: &'static str,
    payload: Value,
) -> Result<Response, HandlerError> {
    let payload = config.prepare_payload(payload)?;
    tokio::task::spawn_blocking(move || {
        let request = config.authorize(
            config
                .agent()
                .post(&config.endpoint(relative_path))
                .set("Accept", "application/json")
                .set("Content-Type", "application/json"),
        );
        let response = response_from_ureq(request.send_json(payload))?;
        buffer_response(response)
    })
    .await
    .map_err(join_error)?
}

pub(crate) async fn proxy_get(
    config: ExternalOpenAiConfig,
    relative_path: &'static str,
) -> Result<Response, HandlerError> {
    tokio::task::spawn_blocking(move || {
        let request = config.authorize(
            config
                .agent()
                .get(&config.endpoint(relative_path))
                .set("Accept", "application/json"),
        );
        let response = response_from_ureq(request.call())?;
        buffer_response(response)
    })
    .await
    .map_err(join_error)?
}

pub(crate) async fn proxy_sse(
    config: ExternalOpenAiConfig,
    relative_path: &'static str,
    payload: Value,
    channel_capacity: usize,
) -> Result<Response, HandlerError> {
    let payload = config.prepare_payload(payload)?;
    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, io::Error>>(channel_capacity.max(1));
    let (ready_tx, ready_rx) = oneshot::channel::<Result<StreamReady, HandlerError>>();

    tokio::task::spawn_blocking(move || {
        let request = config.authorize(
            config
                .agent()
                .post(&config.endpoint(relative_path))
                .set("Accept", "text/event-stream")
                .set("Content-Type", "application/json"),
        );
        let response = match response_from_ureq(request.send_json(payload)) {
            Ok(response) => response,
            Err(error) => {
                let _ = ready_tx.send(Err(error));
                return;
            }
        };
        let status = match status_code(response.status()) {
            Ok(status) => status,
            Err(error) => {
                let _ = ready_tx.send(Err(error));
                return;
            }
        };
        let content_type = response
            .header("content-type")
            .unwrap_or("application/octet-stream")
            .to_string();

        if !status.is_success() {
            let result = read_bounded(response.into_reader())
                .and_then(|body| build_buffered_response(status, &content_type, body));
            let _ = ready_tx.send(result.map(StreamReady::Buffered));
            return;
        }
        if !content_type
            .to_ascii_lowercase()
            .starts_with("text/event-stream")
        {
            let _ = ready_tx.send(Err((
                StatusCode::BAD_GATEWAY,
                format!(
                    "external OpenAI stream returned content type `{content_type}` instead of text/event-stream"
                ),
            )));
            return;
        }

        if ready_tx
            .send(Ok(StreamReady::Streaming {
                status,
                content_type,
            }))
            .is_err()
        {
            return;
        }

        let mut reader = response.into_reader();
        let mut buffer = vec![0u8; STREAM_CHUNK_BYTES];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    if body_tx
                        .blocking_send(Ok(Bytes::copy_from_slice(&buffer[..read])))
                        .is_err()
                    {
                        break;
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "external OpenAI SSE body read failed");
                    let _ = body_tx.blocking_send(Err(error));
                    break;
                }
            }
        }
    });

    match ready_rx.await.map_err(join_error)?? {
        StreamReady::Buffered(response) => Ok(response),
        StreamReady::Streaming {
            status,
            content_type,
        } => build_streaming_response(status, &content_type, body_rx),
    }
}

enum StreamReady {
    Buffered(Response),
    Streaming {
        status: StatusCode,
        content_type: String,
    },
}

fn response_from_ureq(
    result: Result<ureq::Response, ureq::Error>,
) -> Result<ureq::Response, HandlerError> {
    match result {
        Ok(response) => Ok(response),
        Err(ureq::Error::Status(_, response)) => Ok(response),
        Err(ureq::Error::Transport(error)) => Err((
            StatusCode::BAD_GATEWAY,
            format!("external OpenAI request failed: {error}"),
        )),
    }
}

fn buffer_response(response: ureq::Response) -> Result<Response, HandlerError> {
    let status = status_code(response.status())?;
    let content_type = response
        .header("content-type")
        .unwrap_or("application/json")
        .to_string();
    let body = read_bounded(response.into_reader())?;
    build_buffered_response(status, &content_type, body)
}

fn read_bounded(reader: impl Read) -> Result<Vec<u8>, HandlerError> {
    let mut reader: Take<_> = reader.take(MAX_BUFFERED_RESPONSE_BYTES + 1);
    let mut body = Vec::new();
    reader.read_to_end(&mut body).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("failed to read external OpenAI response: {error}"),
        )
    })?;
    if body.len() as u64 > MAX_BUFFERED_RESPONSE_BYTES {
        return Err((
            StatusCode::BAD_GATEWAY,
            format!(
                "external OpenAI response exceeds {} buffered bytes",
                MAX_BUFFERED_RESPONSE_BYTES
            ),
        ));
    }
    Ok(body)
}

fn build_buffered_response(
    status: StatusCode,
    content_type: &str,
    body: Vec<u8>,
) -> Result<Response, HandlerError> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, safe_header_value(content_type)?)
        .body(Body::from(body))
        .map_err(response_build_error)
}

fn build_streaming_response(
    status: StatusCode,
    content_type: &str,
    body_rx: mpsc::Receiver<Result<Bytes, io::Error>>,
) -> Result<Response, HandlerError> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, safe_header_value(content_type)?)
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(body_rx)))
        .map_err(response_build_error)
}

fn normalize_base_url(value: &str, allow_remote: bool) -> Result<String, String> {
    let value = value.trim().trim_end_matches('/');
    if value.is_empty() {
        return Err("external OpenAI base URL must not be empty".to_string());
    }
    if value.contains('?') || value.contains('#') {
        return Err("external OpenAI base URL must not contain a query or fragment".to_string());
    }
    let authority = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"))
        .ok_or_else(|| "external OpenAI base URL must use http:// or https://".to_string())?
        .split('/')
        .next()
        .unwrap_or_default();
    if authority.is_empty() || authority.contains('@') {
        return Err("external OpenAI base URL has an invalid authority".to_string());
    }
    let host = authority_host(authority)?;
    if !allow_remote && !is_loopback_host(host) {
        return Err(format!(
            "external OpenAI base URL host `{host}` is not loopback; set XRT_EXTERNAL_ALLOW_REMOTE=1 to opt in"
        ));
    }
    Ok(value.to_string())
}

fn authority_host(authority: &str) -> Result<&str, String> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, suffix) = rest
            .split_once(']')
            .filter(|(host, _)| !host.is_empty())
            .ok_or_else(|| "external OpenAI base URL has an invalid IPv6 host".to_string())?;
        if !suffix.is_empty() && (!suffix.starts_with(':') || validate_port(&suffix[1..]).is_err())
        {
            return Err("external OpenAI base URL has an invalid IPv6 port".to_string());
        }
        return Ok(host);
    }
    if authority.matches(':').count() > 1 {
        return Err("external OpenAI IPv6 hosts must use brackets".to_string());
    }
    if let Some((host, port)) = authority.rsplit_once(':') {
        if host.is_empty() || validate_port(port).is_err() {
            return Err("external OpenAI base URL has an invalid port".to_string());
        }
        return Ok(host);
    }
    Ok(authority)
}

fn validate_port(value: &str) -> Result<u16, ()> {
    value
        .parse::<u16>()
        .ok()
        .filter(|port| *port != 0)
        .ok_or(())
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") || host.to_ascii_lowercase().ends_with(".localhost") {
        return true;
    }
    host.parse::<IpAddr>().is_ok_and(|address| match address {
        IpAddr::V4(address) => address.is_loopback(),
        IpAddr::V6(address) => address.is_loopback(),
    })
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn normalize_optional(value: String) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn env_nonempty(name: &str) -> Option<String> {
    env::var(name).ok().and_then(normalize_optional)
}

fn env_truthy(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn status_code(value: u16) -> Result<StatusCode, HandlerError> {
    StatusCode::from_u16(value).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("external OpenAI returned invalid HTTP status {value}: {error}"),
        )
    })
}

fn safe_header_value(value: &str) -> Result<HeaderValue, HandlerError> {
    HeaderValue::from_str(value).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("external OpenAI returned an invalid content type: {error}"),
        )
    })
}

fn response_build_error(error: axum::http::Error) -> HandlerError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("failed to build proxy response: {error}"),
    )
}

fn join_error(error: impl std::fmt::Display) -> HandlerError {
    (
        StatusCode::BAD_GATEWAY,
        format!("external OpenAI proxy worker failed: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::{proxy_json, proxy_sse, ExternalOpenAiConfig};
    use axum::{body::to_bytes, http::StatusCode};
    use serde_json::json;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::mpsc,
        thread,
        time::Duration,
    };

    struct MockServer {
        base_url: String,
        request_rx: mpsc::Receiver<Vec<u8>>,
        worker: thread::JoinHandle<()>,
    }

    impl MockServer {
        fn start(status: &str, content_type: &str, body: &[u8]) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("mock listener should bind");
            let address = listener.local_addr().unwrap();
            let (request_tx, request_rx) = mpsc::channel();
            let response_head = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let body = body.to_vec();
            let worker = thread::spawn(move || {
                let (mut stream, _) = listener.accept().expect("proxy should connect");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request = read_http_request(&mut stream);
                request_tx.send(request).unwrap();
                stream.write_all(response_head.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
                stream.flush().unwrap();
            });
            Self {
                base_url: format!("http://{address}/v1"),
                request_rx,
                worker,
            }
        }

        fn finish(self) -> Vec<u8> {
            let request = self
                .request_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("mock server should capture request");
            self.worker.join().expect("mock server should exit");
            request
        }
    }

    fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let read = stream.read(&mut chunk).expect("request should be readable");
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
            let Some(header_end) = find_bytes(&request, b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        request
    }

    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn request_json(request: &[u8]) -> serde_json::Value {
        let header_end = find_bytes(request, b"\r\n\r\n").expect("request should have headers");
        serde_json::from_slice(&request[header_end + 4..]).expect("request should contain JSON")
    }

    #[test]
    fn external_openai_config_rejects_remote_hosts_by_default() {
        let error = ExternalOpenAiConfig::new(
            "https://api.example.com/v1",
            None,
            Some("model".to_string()),
            false,
            30,
        )
        .err()
        .expect("remote host should require explicit opt-in");
        assert!(error.contains("not loopback"), "{error}");

        let config = ExternalOpenAiConfig::new(
            "http://127.0.0.1:8000/v1/",
            None,
            Some("model".to_string()),
            false,
            30,
        )
        .expect("loopback URL should be accepted");
        assert_eq!(config.base_url(), "http://127.0.0.1:8000/v1");
    }

    #[test]
    fn external_openai_config_redacts_api_keys_from_debug_output() {
        let config = ExternalOpenAiConfig::new(
            "http://localhost:8000/v1",
            Some("top-secret".to_string()),
            Some("model".to_string()),
            false,
            30,
        )
        .unwrap();
        let debug = format!("{config:?}");
        assert!(debug.contains("api_key_configured: true"));
        assert!(!debug.contains("top-secret"));
    }

    #[tokio::test]
    async fn external_proxy_preserves_json_fields_and_authorization() {
        let upstream_body = br#"{"id":"cmpl-upstream","object":"text_completion"}"#;
        let server = MockServer::start("200 OK", "application/json", upstream_body);
        let config = ExternalOpenAiConfig::new(
            &server.base_url,
            Some("top-secret".to_string()),
            Some("upstream-model".to_string()),
            false,
            30,
        )
        .unwrap();
        let response = proxy_json(
            config,
            "completions",
            json!({
                "prompt": "Hello",
                "temperature": 0.25,
                "vendor_extension": {"keep": true}
            }),
        )
        .await
        .expect("proxy request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
        let response_body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(response_body.as_ref(), upstream_body);

        let request = server.finish();
        let request_text = String::from_utf8_lossy(&request).to_ascii_lowercase();
        assert!(request_text.starts_with("post /v1/completions http/1.1"));
        assert!(request_text.contains("authorization: bearer top-secret"));
        let request_body = request_json(&request);
        assert_eq!(request_body["model"], "upstream-model");
        assert_eq!(request_body["vendor_extension"]["keep"], true);
    }

    #[tokio::test]
    async fn external_proxy_preserves_sse_bytes_and_done_marker() {
        let upstream_body = b"data: {\"id\":\"chunk-1\"}\n\ndata: [DONE]\n\n";
        let server = MockServer::start("200 OK", "text/event-stream; charset=utf-8", upstream_body);
        let config = ExternalOpenAiConfig::new(
            &server.base_url,
            None,
            Some("stream-model".to_string()),
            false,
            30,
        )
        .unwrap();
        let response = proxy_sse(
            config,
            "chat/completions",
            json!({
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
            2,
        )
        .await
        .expect("SSE proxy should connect");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"],
            "text/event-stream; charset=utf-8"
        );
        let response_body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(response_body.as_ref(), upstream_body);

        let request = server.finish();
        let request_body = request_json(&request);
        assert_eq!(request_body["model"], "stream-model");
        assert_eq!(request_body["stream_options"]["include_usage"], true);
    }

    #[tokio::test]
    async fn external_proxy_preserves_upstream_error_status_and_body() {
        let upstream_body = br#"{"error":{"message":"busy"}}"#;
        let server = MockServer::start("429 Too Many Requests", "application/json", upstream_body);
        let config =
            ExternalOpenAiConfig::new(&server.base_url, None, Some("model".to_string()), false, 30)
                .unwrap();
        let response = proxy_json(config, "chat/completions", json!({"messages": []}))
            .await
            .expect("upstream HTTP errors should remain responses");

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let response_body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(response_body.as_ref(), upstream_body);
        server.finish();
    }
}
