use minijinja::{context, Environment, Value};
use xrt_core::{Result, XrtError};

/// A chat message with role and content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Preprocess a HuggingFace chat template to work with MiniJinja.
/// Converts Python string methods to MiniJinja filter equivalents.
fn preprocess_template(template: &str) -> String {
    let mut result = template.to_string();
    // .strip() → |trim
    result = result.replace(".strip()", "|trim");

    // .startswith("x") → |startswith("x")
    // .endswith("x") → |endswith("x")
    for method in &["startswith", "endswith"] {
        loop {
            let needle = format!(".{method}(");
            if let Some(pos) = result.find(&needle) {
                let after = pos + needle.len();
                if let Some(close) = result[after..].find(')') {
                    let arg = &result[after..after + close];
                    let replacement = format!("|{method}({arg})");
                    result = format!(
                        "{}{}{}",
                        &result[..pos],
                        replacement,
                        &result[after + close + 1..]
                    );
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    // Python/Jinja mappings expose .get(key, default), while MiniJinja maps do not.
    result = result.replace(".get(", "|dict_get(");

    // HuggingFace's Qwen3.8 template scans messages newest-first with Python's
    // negative-step slice. MiniJinja accepts the syntax but produces an empty
    // sequence, which makes the template incorrectly report that no user query
    // exists. Its built-in reverse filter preserves the intended ordering.
    result = result.replace("messages[::-1]", "messages|reverse");

    // XRT renders non-generating message prefixes to derive cache-policy spans.
    // Qwen3.8 rejects a system-only prefix even though rendering that prefix is
    // valid bookkeeping and no assistant generation is requested. Preserve the
    // upstream guard for real generation while allowing these prefix renders.
    if result.contains("No user query found in messages.") {
        for (needle, replacement) in [
            (
                "{%- if ns.multi_step_tool %}",
                "{%- if ns.multi_step_tool and add_generation_prompt %}",
            ),
            (
                "{%- if ns.multi_step_tool -%}",
                "{%- if ns.multi_step_tool and add_generation_prompt -%}",
            ),
        ] {
            if result.contains(needle) {
                result = result.replacen(needle, replacement, 1);
                break;
            }
        }
    }

    result
}

/// Renders a chat template (Jinja2 format from GGUF metadata) with the given messages.
///
/// Variables available in the template:
/// - `messages`: array of {role, content} objects
/// - `bos_token`, `eos_token`: special token strings (or empty)
/// - `add_generation_prompt`: whether to add the assistant turn prefix
pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
) -> Result<String> {
    apply_chat_template_with_thinking(
        template,
        messages,
        bos_token,
        eos_token,
        add_generation_prompt,
        None,
    )
}

/// Renders a chat template with an optional model-native thinking-mode switch.
///
/// Leaving `enable_thinking` as `None` preserves the template's default. This
/// matters for compatibility: models that do not expose the variable behave
/// exactly as before, while Qwen templates can implement their documented
/// `enable_thinking=false` hard switch.
pub fn apply_chat_template_with_thinking(
    template: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
) -> Result<String> {
    let processed = preprocess_template(template);
    let mut env = Environment::new();

    // HuggingFace templates use raise_exception for unsupported features
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    // Python string methods used in HuggingFace templates
    env.add_filter("startswith", |s: String, prefix: String| -> bool {
        s.starts_with(&prefix)
    });
    env.add_filter("endswith", |s: String, suffix: String| -> bool {
        s.ends_with(&suffix)
    });
    env.add_filter(
        "dict_get",
        |value: Value,
         key: Value,
         default: Option<Value>|
         -> std::result::Result<Value, minijinja::Error> {
            let resolved = value.get_item(&key)?;
            Ok(if resolved.is_undefined() {
                default.unwrap_or(Value::UNDEFINED)
            } else {
                resolved
            })
        },
    );

    env.add_template("chat", &processed)
        .map_err(|e| XrtError::Runtime(format!("invalid chat template: {e}")))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| XrtError::Runtime(format!("chat template not found: {e}")))?;
    let rendered = tmpl
        .render(context! {
            messages => messages,
            bos_token => bos_token,
            eos_token => eos_token,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
        })
        .map_err(|e| XrtError::Runtime(format!("chat template render error: {e}")))?;
    Ok(rendered)
}

/// ChatML fallback template for models without a chat_template in GGUF.
pub const CHATML_TEMPLATE: &str = "\
{%- for message in messages %}\
<|im_start|>{{ message.role }}\n\
{{ message.content }}<|im_end|>\n\
{%- endfor %}\
{%- if add_generation_prompt %}\
<|im_start|>assistant\n\
{%- endif %}";

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(r, c)| ChatMessage {
                role: r.to_string(),
                content: c.to_string(),
            })
            .collect()
    }

    #[test]
    fn chatml_basic() {
        let messages = msgs(&[("user", "hello"), ("assistant", "hi")]);
        let out = apply_chat_template(CHATML_TEMPLATE, &messages, "", "", true).unwrap();
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\nhi<|im_end|>"));
        assert!(out.contains("<|im_start|>assistant\n") || out.contains("<|im_start|>assistant"));
    }

    #[test]
    fn namespace_support() {
        // Simplified Qwen-style template that uses namespace() for mutable state
        let template = r#"{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
{%- if message.role == "system" -%}
{%- set ns.found = true -%}
SYSTEM:{{ message.content }}
{% endif -%}
{%- endfor -%}
{%- if not ns.found -%}
NO_SYSTEM
{% endif -%}"#;
        let messages = msgs(&[("system", "You are helpful"), ("user", "hi")]);
        let out = apply_chat_template(template, &messages, "", "", false).unwrap();
        assert!(out.contains("SYSTEM:You are helpful"), "got: {out}");
        assert!(!out.contains("NO_SYSTEM"), "got: {out}");
    }

    #[test]
    fn namespace_no_system() {
        let template = r#"{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
{%- if message.role == "system" -%}
{%- set ns.found = true -%}
{%- endif -%}
{%- endfor -%}
{%- if not ns.found -%}
NO_SYSTEM
{%- endif -%}"#;
        let messages = msgs(&[("user", "hi")]);
        let out = apply_chat_template(template, &messages, "", "", false).unwrap();
        assert!(out.contains("NO_SYSTEM"), "got: {out}");
    }

    #[test]
    fn qwen38_reverse_message_scan_finds_latest_user_query() {
        let template = r#"{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
{%- set index = (messages|length - 1) - loop.index0 -%}
{%- if ns.multi_step_tool and message.role == "user" -%}
{%- set ns.multi_step_tool = false -%}
{%- set ns.last_query_index = index -%}
{%- endif -%}
{%- endfor -%}
{%- if ns.multi_step_tool -%}{{ raise_exception('No user query found in messages.') }}
{%- else -%}USER_AT={{ ns.last_query_index }}
{%- endif -%}"#;
        let messages = msgs(&[
            ("system", "You are helpful"),
            ("user", "first"),
            ("assistant", "answer"),
            ("user", "latest"),
        ]);
        let out = apply_chat_template(template, &messages, "", "", true).unwrap();
        assert_eq!(out.trim(), "USER_AT=3");
    }

    #[test]
    fn qwen38_system_only_span_prefix_does_not_request_generation() {
        let template = r#"{%- set ns = namespace(multi_step_tool=true) -%}
{%- for message in messages[::-1] -%}
{%- if ns.multi_step_tool and message.role == "user" -%}
{%- set ns.multi_step_tool = false -%}
{%- endif -%}
{%- endfor -%}
{%- if ns.multi_step_tool -%}
{{- raise_exception('No user query found in messages.') -}}
{%- endif -%}
{{- messages[0].content -}}"#;
        let messages = msgs(&[("system", "You are helpful")]);

        let prefix = apply_chat_template(template, &messages, "", "", false).unwrap();
        assert_eq!(prefix, "You are helpful");
        let error = apply_chat_template(template, &messages, "", "", true).unwrap_err();
        assert!(error.to_string().contains("No user query found"));
    }

    #[test]
    fn startswith_endswith_preprocessing() {
        let template = r#"{%- for message in messages -%}
{%- if message.role.startswith("use") -%}
USER:{{ message.content }}
{% endif -%}
{%- endfor -%}"#;
        let messages = msgs(&[("user", "hello")]);
        let out = apply_chat_template(template, &messages, "", "", false).unwrap();
        assert!(out.contains("USER:hello"), "got: {out}");
    }

    #[test]
    fn strip_preprocessing() {
        let template = r#"{{ "  hello  ".strip() }}"#;
        let out = apply_chat_template(template, &[], "", "", false).unwrap();
        assert_eq!(out.trim(), "hello");
    }

    #[test]
    fn mapping_get_preprocessing_supports_present_and_default_values() {
        let template = r#"{{ messages[0].get("role", "unknown") }}|{{ messages[0].get("missing", "fallback") }}"#;
        let messages = msgs(&[("user", "hello")]);
        let out = apply_chat_template(template, &messages, "", "", false).unwrap();
        assert_eq!(out, "user|fallback");
    }

    #[test]
    fn optional_thinking_switch_preserves_default_and_supports_hard_disable() {
        let template = r#"{%- if add_generation_prompt -%}
{%- if enable_thinking is defined and enable_thinking is false -%}DISABLED
{%- else -%}ENABLED
{%- endif -%}
{%- endif -%}"#;
        let messages = msgs(&[("user", "hello")]);

        let default = apply_chat_template(template, &messages, "", "", true).unwrap();
        let disabled =
            apply_chat_template_with_thinking(template, &messages, "", "", true, Some(false))
                .unwrap();
        let enabled =
            apply_chat_template_with_thinking(template, &messages, "", "", true, Some(true))
                .unwrap();

        assert_eq!(default, "ENABLED");
        assert_eq!(disabled, "DISABLED");
        assert_eq!(enabled, "ENABLED");
    }
}
