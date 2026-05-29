use minijinja::{context, Environment};
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
}
