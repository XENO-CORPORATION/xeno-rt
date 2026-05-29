use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicyKind {
    DefaultChat,
    AgentAdaptive,
    LongContext,
    MemorySaver,
}

impl CachePolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DefaultChat => "default_chat",
            Self::AgentAdaptive => "agent_adaptive",
            Self::LongContext => "long_context",
            Self::MemorySaver => "memory_saver",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default_chat" | "default-chat" | "default" | "chat" => Some(Self::DefaultChat),
            "agent_adaptive" | "agent-adaptive" | "adaptive" | "agent" => Some(Self::AgentAdaptive),
            "long_context" | "long-context" | "longcontext" => Some(Self::LongContext),
            "memory_saver" | "memory-saver" | "memory" => Some(Self::MemorySaver),
            _ => None,
        }
    }

    pub fn requires_adaptive_cache(self) -> bool {
        !matches!(self, Self::DefaultChat)
    }
}

impl Default for CachePolicyKind {
    fn default() -> Self {
        Self::DefaultChat
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptSpanKind {
    System,
    Developer,
    ToolSchema,
    ToolResult,
    Retrieval,
    User,
    Assistant,
    Scratch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptSpan {
    pub kind: PromptSpanKind,
    pub token_start: usize,
    pub token_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPolicy {
    pub cache_policy: CachePolicyKind,
    pub recent_window_tokens: usize,
    pub pin_system_prompt: bool,
    pub pin_tool_schema: bool,
    pub protect_tool_results: bool,
    pub retrieval_compression_delay: usize,
}

impl SessionPolicy {
    pub fn default_chat() -> Self {
        Self {
            cache_policy: CachePolicyKind::DefaultChat,
            recent_window_tokens: 256,
            pin_system_prompt: false,
            pin_tool_schema: false,
            protect_tool_results: false,
            retrieval_compression_delay: 0,
        }
    }

    pub fn agent_adaptive() -> Self {
        Self {
            cache_policy: CachePolicyKind::AgentAdaptive,
            recent_window_tokens: 256,
            pin_system_prompt: true,
            pin_tool_schema: true,
            protect_tool_results: true,
            retrieval_compression_delay: 256,
        }
    }

    pub fn long_context() -> Self {
        Self {
            cache_policy: CachePolicyKind::LongContext,
            recent_window_tokens: 384,
            pin_system_prompt: true,
            pin_tool_schema: true,
            protect_tool_results: true,
            retrieval_compression_delay: 128,
        }
    }

    pub fn memory_saver() -> Self {
        Self {
            cache_policy: CachePolicyKind::MemorySaver,
            recent_window_tokens: 192,
            pin_system_prompt: true,
            pin_tool_schema: false,
            protect_tool_results: true,
            retrieval_compression_delay: 96,
        }
    }

    pub fn from_request(
        policy_name: Option<&str>,
        recent_window_tokens: Option<usize>,
        default_policy: CachePolicyKind,
    ) -> Self {
        let base = match policy_name
            .and_then(CachePolicyKind::parse)
            .unwrap_or(default_policy)
        {
            CachePolicyKind::DefaultChat => Self::default_chat(),
            CachePolicyKind::AgentAdaptive => Self::agent_adaptive(),
            CachePolicyKind::LongContext => Self::long_context(),
            CachePolicyKind::MemorySaver => Self::memory_saver(),
        };

        Self {
            recent_window_tokens: recent_window_tokens.unwrap_or(base.recent_window_tokens),
            ..base
        }
    }

    pub fn is_span_pinned(&self, kind: PromptSpanKind) -> bool {
        match kind {
            PromptSpanKind::System | PromptSpanKind::Developer => self.pin_system_prompt,
            PromptSpanKind::ToolSchema => self.pin_tool_schema,
            PromptSpanKind::ToolResult => self.protect_tool_results,
            PromptSpanKind::Retrieval => false,
            PromptSpanKind::User | PromptSpanKind::Assistant | PromptSpanKind::Scratch => false,
        }
    }
}

impl Default for SessionPolicy {
    fn default() -> Self {
        Self::default_chat()
    }
}
