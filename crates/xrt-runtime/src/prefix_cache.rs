use std::{
    collections::{HashMap, VecDeque},
    env,
    sync::Arc,
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    backend::{BackendKind, BackendPrefixSnapshot},
    kv_cache::KvCacheMode,
    policy::{PromptSpan, PromptSpanKind, SessionPolicy},
};

const DEFAULT_MAX_ENTRIES: usize = 32;
const DEFAULT_MAX_BYTES: u64 = 256 * 1024 * 1024;
const DEFAULT_MIN_TOKENS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_bytes: u64,
    pub min_tokens: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
            min_tokens: DEFAULT_MIN_TOKENS,
        }
    }
}

impl PrefixCacheConfig {
    pub fn from_env() -> Self {
        Self::from_values(
            env::var("XRT_PREFIX_CACHE").ok().as_deref(),
            env::var("XRT_PREFIX_CACHE_MAX_ENTRIES").ok().as_deref(),
            env::var("XRT_PREFIX_CACHE_MAX_BYTES").ok().as_deref(),
            env::var("XRT_PREFIX_CACHE_MIN_TOKENS").ok().as_deref(),
        )
    }

    fn from_values(
        enabled: Option<&str>,
        max_entries: Option<&str>,
        max_bytes: Option<&str>,
        min_tokens: Option<&str>,
    ) -> Self {
        let default = Self::default();
        let enabled = enabled.and_then(parse_bool).unwrap_or(default.enabled);
        let max_entries = parse_usize(max_entries).unwrap_or(default.max_entries);
        let max_bytes = parse_u64(max_bytes).unwrap_or(default.max_bytes);
        let min_tokens = parse_usize(min_tokens).unwrap_or(default.min_tokens);
        Self {
            enabled: enabled && max_entries > 0 && max_bytes > 0,
            max_entries,
            max_bytes,
            min_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrefixCacheStatus {
    pub enabled: bool,
    pub namespace: String,
    pub entries: usize,
    pub max_entries: usize,
    pub resident_bytes: u64,
    pub max_bytes: u64,
    pub min_tokens: usize,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub prefill_tokens_saved: u64,
    pub inserts: u64,
    pub evictions: u64,
    pub rejected_entries: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrefixCacheKey {
    namespace: Arc<str>,
    backend: BackendKind,
    cache_mode: KvCacheMode,
    policy: SessionPolicy,
    tokens: Arc<[u32]>,
    spans: Arc<[PromptSpan]>,
}

#[derive(Debug, Clone)]
pub(crate) struct PrefixCacheRequest {
    key: PrefixCacheKey,
    prefix_len: usize,
}

impl PrefixCacheRequest {
    pub(crate) fn prefix_len(&self) -> usize {
        self.prefix_len
    }
}

#[derive(Debug)]
struct PrefixCacheEntry {
    snapshot: Arc<BackendPrefixSnapshot>,
    allocated_bytes: u64,
}

#[derive(Debug, Default)]
struct PrefixCacheState {
    entries: HashMap<PrefixCacheKey, PrefixCacheEntry>,
    lru: VecDeque<PrefixCacheKey>,
    resident_bytes: u64,
    lookups: u64,
    hits: u64,
    misses: u64,
    prefill_tokens_saved: u64,
    inserts: u64,
    evictions: u64,
    rejected_entries: u64,
}

#[derive(Debug)]
pub struct PrefixCacheManager {
    namespace: Arc<str>,
    config: PrefixCacheConfig,
    state: Mutex<PrefixCacheState>,
}

impl PrefixCacheManager {
    pub fn from_env(namespace: impl Into<String>) -> Self {
        Self::new(namespace, PrefixCacheConfig::from_env())
    }

    pub fn new(namespace: impl Into<String>, config: PrefixCacheConfig) -> Self {
        Self {
            namespace: Arc::from(namespace.into()),
            config,
            state: Mutex::new(PrefixCacheState::default()),
        }
    }

    pub(crate) fn request(
        &self,
        backend: BackendKind,
        cache_mode: KvCacheMode,
        policy: &SessionPolicy,
        tokens: &[u32],
        spans: &[PromptSpan],
    ) -> Option<PrefixCacheRequest> {
        if !self.config.enabled || tokens.len() < 2 {
            return None;
        }
        let prefix_len = reusable_prefix_len(tokens.len(), policy, spans);
        if prefix_len < self.config.min_tokens {
            return None;
        }
        let clipped_spans = spans
            .iter()
            .filter_map(|span| {
                let token_start = span.token_start.min(prefix_len);
                let token_end = span.token_end.min(prefix_len);
                (token_start < token_end).then_some(PromptSpan {
                    kind: span.kind,
                    token_start,
                    token_end,
                })
            })
            .collect::<Vec<_>>();
        Some(PrefixCacheRequest {
            key: PrefixCacheKey {
                namespace: self.namespace.clone(),
                backend,
                cache_mode,
                policy: policy.clone(),
                tokens: Arc::from(&tokens[..prefix_len]),
                spans: Arc::from(clipped_spans),
            },
            prefix_len,
        })
    }

    pub(crate) fn lookup(
        &self,
        request: &PrefixCacheRequest,
    ) -> Option<Arc<BackendPrefixSnapshot>> {
        let mut state = self.state.lock();
        state.lookups = state.lookups.saturating_add(1);
        let snapshot = state
            .entries
            .get(&request.key)
            .map(|entry| entry.snapshot.clone());
        if let Some(snapshot) = snapshot {
            state.hits = state.hits.saturating_add(1);
            state.prefill_tokens_saved = state
                .prefill_tokens_saved
                .saturating_add(request.prefix_len as u64);
            touch_lru(&mut state.lru, &request.key);
            Some(snapshot)
        } else {
            state.misses = state.misses.saturating_add(1);
            None
        }
    }

    pub(crate) fn insert(&self, request: PrefixCacheRequest, snapshot: BackendPrefixSnapshot) {
        if !self.config.enabled || snapshot.prefix_len() != request.prefix_len {
            return;
        }
        let allocated_bytes = snapshot.allocated_bytes();
        let mut state = self.state.lock();
        if state.entries.contains_key(&request.key) {
            touch_lru(&mut state.lru, &request.key);
            return;
        }
        if allocated_bytes > self.config.max_bytes {
            state.rejected_entries = state.rejected_entries.saturating_add(1);
            return;
        }
        while !state.entries.is_empty()
            && (state.entries.len() >= self.config.max_entries
                || state.resident_bytes.saturating_add(allocated_bytes) > self.config.max_bytes)
        {
            let Some(oldest) = state.lru.pop_front() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&oldest) {
                state.resident_bytes = state.resident_bytes.saturating_sub(evicted.allocated_bytes);
                state.evictions = state.evictions.saturating_add(1);
            }
        }
        state.resident_bytes = state.resident_bytes.saturating_add(allocated_bytes);
        state.inserts = state.inserts.saturating_add(1);
        state.lru.push_back(request.key.clone());
        state.entries.insert(
            request.key,
            PrefixCacheEntry {
                snapshot: Arc::new(snapshot),
                allocated_bytes,
            },
        );
    }

    pub fn status(&self) -> PrefixCacheStatus {
        let state = self.state.lock();
        let hit_rate = if state.lookups == 0 {
            0.0
        } else {
            state.hits as f64 / state.lookups as f64
        };
        PrefixCacheStatus {
            enabled: self.config.enabled,
            namespace: self.namespace.to_string(),
            entries: state.entries.len(),
            max_entries: self.config.max_entries,
            resident_bytes: state.resident_bytes,
            max_bytes: self.config.max_bytes,
            min_tokens: self.config.min_tokens,
            lookups: state.lookups,
            hits: state.hits,
            misses: state.misses,
            hit_rate,
            prefill_tokens_saved: state.prefill_tokens_saved,
            inserts: state.inserts,
            evictions: state.evictions,
            rejected_entries: state.rejected_entries,
        }
    }
}

fn reusable_prefix_len(token_count: usize, policy: &SessionPolicy, spans: &[PromptSpan]) -> usize {
    let maximum = token_count.saturating_sub(1);
    let mut prefix_end = 0usize;
    let mut found_structural_prefix = false;
    for span in spans {
        if span.token_start > prefix_end && found_structural_prefix {
            break;
        }
        let cacheable = matches!(
            span.kind,
            PromptSpanKind::System | PromptSpanKind::Developer | PromptSpanKind::ToolSchema
        ) || policy.is_span_pinned(span.kind);
        if !cacheable {
            break;
        }
        found_structural_prefix = true;
        prefix_end = prefix_end.max(span.token_end.min(maximum));
        if prefix_end == maximum {
            break;
        }
    }
    if found_structural_prefix {
        prefix_end
    } else {
        maximum
    }
}

fn touch_lru(lru: &mut VecDeque<PrefixCacheKey>, key: &PrefixCacheKey) {
    if let Some(index) = lru.iter().position(|candidate| candidate == key) {
        lru.remove(index);
    }
    lru.push_back(key.clone());
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" | "enabled" => Some(true),
        "0" | "false" | "no" | "off" | "disabled" => Some(false),
        _ => None,
    }
}

fn parse_usize(value: Option<&str>) -> Option<usize> {
    value?.trim().parse().ok()
}

fn parse_u64(value: Option<&str>) -> Option<u64> {
    value?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::SessionKvCache;

    fn snapshot(prefix_len: usize, value: f32) -> BackendPrefixSnapshot {
        use xrt_core::KvCache;

        let mut cache = SessionKvCache::new(KvCacheMode::F32, 1, 1, 2);
        for _ in 0..prefix_len {
            cache.append(0, &[value], &[value]).unwrap();
        }
        let cache = cache.snapshot_prefix(prefix_len).unwrap();
        BackendPrefixSnapshot::Cpu {
            allocated_bytes: cache.allocated_bytes(),
            cache,
            prefix_len,
        }
    }

    fn config(max_entries: usize) -> PrefixCacheConfig {
        PrefixCacheConfig {
            enabled: true,
            max_entries,
            max_bytes: 1024 * 1024,
            min_tokens: 1,
        }
    }

    #[test]
    fn structural_prefix_stops_before_the_first_user_span() {
        let policy = SessionPolicy::agent_adaptive();
        let spans = vec![
            PromptSpan {
                kind: PromptSpanKind::System,
                token_start: 0,
                token_end: 3,
            },
            PromptSpan {
                kind: PromptSpanKind::ToolSchema,
                token_start: 3,
                token_end: 5,
            },
            PromptSpan {
                kind: PromptSpanKind::User,
                token_start: 5,
                token_end: 8,
            },
        ];
        assert_eq!(reusable_prefix_len(8, &policy, &spans), 5);
        assert_eq!(reusable_prefix_len(8, &policy, &[]), 7);
    }

    #[test]
    fn exact_key_dimensions_control_hits() {
        let manager = PrefixCacheManager::new("model-a:tokenizer-a", config(4));
        let policy = SessionPolicy::default_chat();
        let tokens = [1, 2, 3];
        let request = manager
            .request(BackendKind::Cpu, KvCacheMode::F32, &policy, &tokens, &[])
            .unwrap();
        assert!(manager.lookup(&request).is_none());
        manager.insert(request.clone(), snapshot(request.prefix_len(), 1.0));
        assert!(manager.lookup(&request).is_some());

        let different_mode = manager
            .request(BackendKind::Cpu, KvCacheMode::Q8, &policy, &tokens, &[])
            .unwrap();
        assert!(manager.lookup(&different_mode).is_none());
        let different_tokens = manager
            .request(BackendKind::Cpu, KvCacheMode::F32, &policy, &[1, 2, 4], &[])
            .unwrap();
        assert!(manager.lookup(&different_tokens).is_none());

        let status = manager.status();
        assert_eq!(status.lookups, 4);
        assert_eq!(status.hits, 1);
        assert_eq!(status.misses, 3);
        assert_eq!(status.prefill_tokens_saved, 2);
    }

    #[test]
    fn lru_eviction_keeps_the_recent_entry() {
        let manager = PrefixCacheManager::new("model-a:tokenizer-a", config(1));
        let policy = SessionPolicy::default_chat();
        let first = manager
            .request(BackendKind::Cpu, KvCacheMode::F32, &policy, &[1, 2], &[])
            .unwrap();
        let second = manager
            .request(BackendKind::Cpu, KvCacheMode::F32, &policy, &[3, 4], &[])
            .unwrap();
        manager.insert(first.clone(), snapshot(first.prefix_len(), 1.0));
        manager.insert(second.clone(), snapshot(second.prefix_len(), 2.0));

        assert!(manager.lookup(&first).is_none());
        assert!(manager.lookup(&second).is_some());
        let status = manager.status();
        assert_eq!(status.entries, 1);
        assert_eq!(status.evictions, 1);
    }

    #[test]
    fn invalid_config_values_use_bounded_defaults() {
        assert_eq!(
            PrefixCacheConfig::from_values(Some("bad"), Some("bad"), Some("bad"), Some("bad")),
            PrefixCacheConfig::default()
        );
        assert!(!PrefixCacheConfig::from_values(Some("off"), None, None, None).enabled);
    }
}
