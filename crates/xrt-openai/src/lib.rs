use serde_json::Value;
use std::{env, io::Read, net::IpAddr, time::Duration};

mod images;

pub use images::*;

const DEFAULT_TIMEOUT_SECONDS: u64 = 300;
const MAX_TIMEOUT_SECONDS: u64 = 3_600;

#[derive(Clone)]
pub struct ExternalOpenAiConfig {
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
    pub fn from_env_with_overrides(
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

    pub fn new(
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

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn default_model(&self) -> Option<&str> {
        self.default_model.as_deref()
    }

    pub fn display_model(&self) -> &str {
        self.default_model().unwrap_or("external-openai")
    }

    pub fn prepare_payload(&self, mut payload: Value) -> Result<Value, ExternalOpenAiError> {
        let object = payload.as_object_mut().ok_or_else(|| {
            ExternalOpenAiError::InvalidRequest(
                "OpenAI request body must be a JSON object".to_string(),
            )
        })?;
        let has_model = object
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(|model| !model.trim().is_empty());
        if !has_model {
            let model = self.default_model().ok_or_else(|| {
                ExternalOpenAiError::InvalidRequest(
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

    fn build_agent(&self) -> ureq::Agent {
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

#[derive(Clone)]
pub struct ExternalOpenAiClient {
    config: ExternalOpenAiConfig,
    agent: ureq::Agent,
}

impl ExternalOpenAiClient {
    pub fn new(config: ExternalOpenAiConfig) -> Self {
        let agent = config.build_agent();
        Self { config, agent }
    }

    pub fn config(&self) -> &ExternalOpenAiConfig {
        &self.config
    }

    pub fn post_json(
        &self,
        relative_path: &str,
        payload: Value,
        accept: &str,
    ) -> Result<ExternalOpenAiResponse, ExternalOpenAiError> {
        let payload = self.config.prepare_payload(payload)?;
        let request = self.config.authorize(
            self.agent
                .post(&self.config.endpoint(relative_path))
                .set("Accept", accept)
                .set("Content-Type", "application/json"),
        );
        map_response(request.send_json(payload))
    }

    pub fn get(
        &self,
        relative_path: &str,
        accept: &str,
    ) -> Result<ExternalOpenAiResponse, ExternalOpenAiError> {
        let request = self.config.authorize(
            self.agent
                .get(&self.config.endpoint(relative_path))
                .set("Accept", accept),
        );
        map_response(request.call())
    }
}

pub struct ExternalOpenAiResponse {
    status: u16,
    content_type: String,
    reader: Box<dyn Read + Send + Sync + 'static>,
}

impl ExternalOpenAiResponse {
    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn content_type(&self) -> &str {
        &self.content_type
    }

    pub fn into_reader(self) -> Box<dyn Read + Send + Sync + 'static> {
        self.reader
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalOpenAiError {
    InvalidRequest(String),
    Transport(String),
}

impl ExternalOpenAiError {
    pub fn is_invalid_request(&self) -> bool {
        matches!(self, Self::InvalidRequest(_))
    }
}

impl std::fmt::Display for ExternalOpenAiError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Transport(message) => {
                formatter.write_str(message)
            }
        }
    }
}

impl std::error::Error for ExternalOpenAiError {}

fn map_response(
    result: Result<ureq::Response, ureq::Error>,
) -> Result<ExternalOpenAiResponse, ExternalOpenAiError> {
    let response = match result {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => response,
        Err(ureq::Error::Transport(error)) => {
            return Err(ExternalOpenAiError::Transport(format!(
                "external OpenAI request failed: {error}"
            )))
        }
    };
    let status = response.status();
    let content_type = response
        .header("content-type")
        .unwrap_or("application/octet-stream")
        .to_string();
    Ok(ExternalOpenAiResponse {
        status,
        content_type,
        reader: response.into_reader(),
    })
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

#[cfg(test)]
mod tests {
    use super::{ExternalOpenAiConfig, ExternalOpenAiError};
    use serde_json::json;

    #[test]
    fn config_rejects_remote_hosts_by_default_and_redacts_keys() {
        let error = ExternalOpenAiConfig::new(
            "https://api.example.com/v1",
            None,
            Some("model".to_string()),
            false,
            30,
        )
        .unwrap_err();
        assert!(error.contains("not loopback"), "{error}");

        let config = ExternalOpenAiConfig::new(
            "http://127.0.0.1:8000/v1/",
            Some("top-secret".to_string()),
            Some("model".to_string()),
            false,
            30,
        )
        .unwrap();
        assert_eq!(config.base_url(), "http://127.0.0.1:8000/v1");
        let debug = format!("{config:?}");
        assert!(debug.contains("api_key_configured: true"));
        assert!(!debug.contains("top-secret"));
    }

    #[test]
    fn payload_requires_an_object_and_injects_the_default_model() {
        let config = ExternalOpenAiConfig::new(
            "http://localhost:8000/v1",
            None,
            Some("model".to_string()),
            false,
            30,
        )
        .unwrap();
        let payload = config
            .prepare_payload(json!({"messages": [], "vendor": true}))
            .unwrap();
        assert_eq!(payload["model"], "model");
        assert_eq!(payload["vendor"], true);

        let error = config.prepare_payload(json!([])).unwrap_err();
        assert!(matches!(error, ExternalOpenAiError::InvalidRequest(_)));
    }

    #[test]
    fn config_rejects_credentials_queries_fragments_and_invalid_ports() {
        for value in [
            "http://user:pass@localhost:8000/v1",
            "http://localhost:8000/v1?key=value",
            "http://localhost:8000/v1#fragment",
            "http://localhost:0/v1",
            "http://localhost:99999/v1",
            "http://[::1]:invalid/v1",
        ] {
            assert!(
                ExternalOpenAiConfig::new(value, None, None, false, 30).is_err(),
                "{value} should be rejected"
            );
        }
    }
}
