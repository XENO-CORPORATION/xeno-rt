use axum::{
    body::{Body, Bytes},
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde_json::Value;
use std::io::{self, Read, Take};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use xrt_openai::{ExternalOpenAiError, ExternalOpenAiResponse};

const MAX_BUFFERED_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
const STREAM_CHUNK_BYTES: usize = 16 * 1024;

pub(crate) type HandlerError = (StatusCode, String);
pub(crate) use xrt_openai::{ExternalOpenAiClient, ExternalOpenAiConfig};

pub(crate) async fn proxy_json(
    client: ExternalOpenAiClient,
    relative_path: &'static str,
    payload: Value,
) -> Result<Response, HandlerError> {
    tokio::task::spawn_blocking(move || {
        let response = client
            .post_json(relative_path, payload, "application/json")
            .map_err(proxy_error)?;
        buffer_response(response)
    })
    .await
    .map_err(join_error)?
}

pub(crate) async fn proxy_get(
    client: ExternalOpenAiClient,
    relative_path: &'static str,
) -> Result<Response, HandlerError> {
    tokio::task::spawn_blocking(move || {
        let response = client
            .get(relative_path, "application/json")
            .map_err(proxy_error)?;
        buffer_response(response)
    })
    .await
    .map_err(join_error)?
}

pub(crate) async fn proxy_sse(
    client: ExternalOpenAiClient,
    relative_path: &'static str,
    payload: Value,
    channel_capacity: usize,
) -> Result<Response, HandlerError> {
    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, io::Error>>(channel_capacity.max(1));
    let (ready_tx, ready_rx) = oneshot::channel::<Result<StreamReady, HandlerError>>();

    tokio::task::spawn_blocking(move || {
        let response = match client.post_json(relative_path, payload, "text/event-stream") {
            Ok(response) => response,
            Err(error) => {
                let _ = ready_tx.send(Err(proxy_error(error)));
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
        let content_type = response.content_type().to_string();

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

fn buffer_response(response: ExternalOpenAiResponse) -> Result<Response, HandlerError> {
    let status = status_code(response.status())?;
    let content_type = response.content_type().to_string();
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

fn status_code(value: u16) -> Result<StatusCode, HandlerError> {
    StatusCode::from_u16(value).map_err(|error| {
        (
            StatusCode::BAD_GATEWAY,
            format!("external OpenAI returned invalid HTTP status {value}: {error}"),
        )
    })
}

fn proxy_error(error: ExternalOpenAiError) -> HandlerError {
    let status = if error.is_invalid_request() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::BAD_GATEWAY
    };
    (status, error.to_string())
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
    use super::{proxy_json, proxy_sse, ExternalOpenAiClient, ExternalOpenAiConfig};
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
            ExternalOpenAiClient::new(config),
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
            ExternalOpenAiClient::new(config),
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
        let response = proxy_json(
            ExternalOpenAiClient::new(config),
            "chat/completions",
            json!({"messages": []}),
        )
        .await
        .expect("upstream HTTP errors should remain responses");

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let response_body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(response_body.as_ref(), upstream_body);
        server.finish();
    }
}
