//! One constrained response envelope for model-selected tools and ordinary text.

use serde_json::{json, Value};
use std::collections::HashSet;
use xrt_runtime::Grammar;

use crate::{ChatToolCall, ChatToolCustom, ChatToolFunction, GrammarCompletion};

const MAX_TOOLS: usize = 128;
const TEXT_PREFIX: &str = "<xeno_text>\n";
const FUNCTIONS_PREFIX: &str = "<xeno_functions>\n";
const FUNCTIONS_SUFFIX: &str = "\n</xeno_functions>";
const CALLS_PREFIX: &str = "<xeno_calls>\n";
const CALLS_SUFFIX: &str = "\n</xeno_calls>";

#[derive(Clone)]
struct Tool {
    name: String,
    custom: bool,
    prefix: String,
    suffix: String,
}

pub(crate) struct ToolPlan {
    pub grammar: Option<Grammar>,
    tools: Vec<Tool>,
    allow_text: bool,
    allow_functions: bool,
    allow_batch: bool,
}

impl ToolPlan {
    #[cfg(test)]
    pub fn prepare(definitions: Option<&[Value]>, choice: Option<&Value>) -> Result<Self, String> {
        Self::prepare_with_parallel(definitions, choice, true)
    }

    #[cfg(test)]
    pub fn prepare_with_parallel(
        definitions: Option<&[Value]>,
        choice: Option<&Value>,
        parallel: bool,
    ) -> Result<Self, String> {
        Self::prepare_bounded(definitions, choice, parallel, None, None)
    }

    pub fn prepare_bounded(
        definitions: Option<&[Value]>,
        choice: Option<&Value>,
        parallel: bool,
        minimum: Option<usize>,
        maximum: Option<usize>,
    ) -> Result<Self, String> {
        let min_calls = minimum.unwrap_or(1);
        let max_calls = maximum.unwrap_or(if parallel { MAX_TOOLS } else { 1 });
        if min_calls == 0
            || min_calls > max_calls
            || max_calls > MAX_TOOLS
            || (!parallel && max_calls > 1)
        {
            return Err(
                "tool call bounds must satisfy 1 <= min <= max <= 128 and parallel-call policy"
                    .to_string(),
            );
        }
        let definitions = definitions.unwrap_or_default();
        if definitions.len() > MAX_TOOLS {
            return Err(format!("at most {MAX_TOOLS} tools may be declared"));
        }
        let mut names = HashSet::new();
        let mut tools = Vec::new();
        let mut rules = Vec::new();
        let mut schemas = Vec::new();
        for (index, definition) in definitions.iter().enumerate() {
            let kind = definition
                .get("type")
                .and_then(Value::as_str)
                .ok_or("tool type is required")?;
            if kind != "function" && kind != "custom" {
                return Err(format!("unsupported tool type: {kind}"));
            }
            let declaration = definition
                .get(kind)
                .and_then(Value::as_object)
                .ok_or("tool declaration is missing")?;
            let name = declaration
                .get("name")
                .and_then(Value::as_str)
                .ok_or("tool name is required")?;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
            {
                return Err("tool names must match [A-Za-z0-9_-]{1,64}".to_string());
            }
            if !names.insert(name.to_string()) {
                return Err(format!("duplicate tool name: {name}"));
            }
            let prefix = format!("<xeno_tool_{index}>\n");
            let suffix = format!("\n</xeno_tool_{index}>");
            let custom = kind == "custom";
            let body = if custom {
                let format = declaration
                    .get("format")
                    .ok_or("custom tools require a grammar format")?;
                if format.get("type").and_then(Value::as_str) != Some("grammar")
                    || format.get("syntax").and_then(Value::as_str) != Some("gbnf")
                {
                    return Err("custom tool format must be grammar/gbnf".to_string());
                }
                let grammar = format
                    .get("definition")
                    .and_then(Value::as_str)
                    .ok_or("custom grammar definition must be text")?;
                let namespace = format!("g{index}");
                rules.push(Grammar::gbnf_rules(grammar, &namespace)?);
                format!("{namespace}start")
            } else {
                let schema = declaration
                    .get("parameters")
                    .cloned()
                    .ok_or("function parameters must be an object schema")?;
                if schema.get("type").and_then(Value::as_str) != Some("object")
                    || !schema.get("properties").is_some_and(Value::is_object)
                {
                    return Err(
                        "function parameters must preserve the object/properties schema"
                            .to_string(),
                    );
                }
                let schema_name = format!("args{index}");
                schemas.push((schema_name.clone(), schema));
                format!("@{schema_name}")
            };
            rules.push(format!(
                "tool{index}[capture=\"__LIST_APPEND:xeno_tool{index}\"]: {} {body} {}\n",
                literal(&prefix),
                literal(&suffix)
            ));
            rules.push(format!(
                "next_tool{index}[capture=\"__LIST_APPEND:xeno_tool{index}\"]: {} {body} {}\n",
                literal(&format!("\n{prefix}")),
                literal(&suffix)
            ));
            if !custom {
                let header = format!("{{\"name\":{},\"arguments\":", literal(name));
                rules.push(format!(
                    "function{index}: {} {body} \"}}\"\n",
                    literal(&header)
                ));
            }
            tools.push(Tool {
                name: name.to_string(),
                custom,
                prefix,
                suffix,
            });
        }

        let strategy = choice.and_then(Value::as_str).unwrap_or("auto");
        let selected = if choice.is_some_and(Value::is_object) {
            let choice = choice.expect("choice object exists");
            let kind = choice
                .get("type")
                .and_then(Value::as_str)
                .ok_or("tool_choice type is required")?;
            if kind != "function" && kind != "custom" {
                return Err("unsupported tool_choice type".to_string());
            }
            let name = choice
                .get(kind)
                .and_then(|value| value.get("name"))
                .and_then(Value::as_str)
                .ok_or("named tool_choice must include a name")?;
            Some(
                tools
                    .iter()
                    .position(|tool| tool.name == name && tool.custom == (kind == "custom"))
                    .ok_or("tool_choice does not name a declared tool of the same type")?,
            )
        } else {
            if choice.is_some_and(|value| !value.is_string())
                || !["auto", "none", "required"].contains(&strategy)
            {
                return Err(
                    "tool_choice must be auto, none, required, or a named declaration".to_string(),
                );
            }
            None
        };
        if tools.is_empty()
            && (strategy == "required"
                || selected.is_some()
                || minimum.is_some()
                || maximum.is_some())
        {
            return Err("required tool_choice needs at least one declared tool".to_string());
        }
        if tools.is_empty() || strategy == "none" {
            if strategy == "none" && (minimum.is_some() || maximum.is_some()) {
                return Err("tool call bounds conflict with tool_choice none".to_string());
            }
            return Ok(Self {
                grammar: None,
                tools: Vec::new(),
                allow_text: true,
                allow_functions: false,
                allow_batch: false,
            });
        }
        let allow_text = selected.is_none() && strategy == "auto" && minimum.is_none();
        let singles = selected
            .map(|index| vec![format!("tool{index}")])
            .unwrap_or_else(|| {
                (0..tools.len())
                    .map(|index| format!("tool{index}"))
                    .collect()
            });
        let mut alternatives = if min_calls == 1 {
            singles.clone()
        } else {
            Vec::new()
        };
        let allow_batch = max_calls > 1;
        if allow_batch {
            rules.push(format!("tool_call: {}\n", singles.join(" | ")));
            let next = selected
                .map(|index| vec![format!("next_tool{index}")])
                .unwrap_or_else(|| {
                    (0..tools.len())
                        .map(|index| format!("next_tool{index}"))
                        .collect()
                });
            rules.push(format!("next_tool_call: {}\n", next.join(" | ")));
            rules.push(format!(
                "batch_reply: {} tool_call next_tool_call{{{},{}}} {}\n",
                literal(CALLS_PREFIX),
                min_calls - 1,
                max_calls - 1,
                literal(CALLS_SUFFIX)
            ));
            alternatives.push("batch_reply".to_string());
        }
        let functions = (0..tools.len())
            .filter(|index| {
                !tools[*index].custom && selected.map_or(true, |selected| selected == *index)
            })
            .map(|index| format!("function{index}"))
            .collect::<Vec<_>>();
        let allow_functions = !functions.is_empty();
        if allow_functions {
            rules.push(format!("function_call: {}\n", functions.join(" | ")));
            let tail = if max_calls == 1 {
                String::new()
            } else {
                format!(
                    "(\",\" function_call){{{},{}}}",
                    min_calls - 1,
                    max_calls - 1
                )
            };
            rules.push(format!(
                "function_reply: {} \"[\" function_call {tail} \"]\" {}\n",
                literal(FUNCTIONS_PREFIX),
                literal(FUNCTIONS_SUFFIX)
            ));
            alternatives.push("function_reply".to_string());
        }
        if allow_text {
            alternatives.push("text_reply".to_string());
            rules.push(format!(
                "text_reply: {} /[\\s\\S]*/\n",
                literal(TEXT_PREFIX)
            ));
        }
        let lark = format!("start: {}\n{}", alternatives.join(" | "), rules.join("\n"));
        let grammar = Grammar::from_lark(lark, schemas)?;
        Ok(Self {
            grammar: Some(grammar),
            tools,
            allow_text,
            allow_functions,
            allow_batch,
        })
    }

    pub fn instructions(&self, definitions: Option<&[Value]>) -> Option<String> {
        self.grammar.as_ref()?;
        let envelopes = self
            .tools
            .iter()
            .map(|tool| {
                format!(
                    "{}: {}{}{}",
                    tool.name,
                    tool.prefix,
                    if tool.custom {
                        "raw input matching its GBNF"
                    } else {
                        "JSON object matching parameters"
                    },
                    tool.suffix
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Some(format!(
            "Return exactly one response envelope with no reasoning, tags, or prose outside it.\n\
             Tool envelopes (use the exact markers):\n{envelopes}\n{}\n{}\n{}\nAvailable tool contracts:\n{}",
            if self.allow_batch { format!("For multiple custom/function calls use {CALLS_PREFIX}the tool envelopes above separated by one newline{CALLS_SUFFIX}. Do not JSON-escape custom input.") } else { String::new() },
            if self.allow_functions { format!("For one or more function calls use {FUNCTIONS_PREFIX}[{{\"name\":\"exact function name\",\"arguments\":{{}}}}]{FUNCTIONS_SUFFIX}. Arguments must match the function schema.") } else { String::new() },
            if self.allow_text { format!("For an ordinary answer use {TEXT_PREFIX} followed by the answer.") }
                else { "A tool call is required; an ordinary answer is not permitted.".to_string() },
            serde_json::to_string(definitions.unwrap_or_default()).expect("tool definitions serialize")
        ))
    }

    pub fn response(
        &self,
        text: String,
        complete: bool,
        call_id: &str,
    ) -> Result<(String, Option<Vec<ChatToolCall>>), String> {
        if self.grammar.is_none() {
            return Ok((text, None));
        }
        if self.allow_text {
            if let Some(body) = text.strip_prefix(TEXT_PREFIX) {
                return Ok((body.to_string(), None));
            }
        }
        if self.allow_functions && text.starts_with(FUNCTIONS_PREFIX) {
            if !complete {
                return Ok((String::new(), None));
            }
            return Ok((
                String::new(),
                Some(function_batch(&self.tools, &text, call_id)?),
            ));
        }
        if self.allow_batch && text.starts_with(CALLS_PREFIX) {
            if !complete {
                return Ok((String::new(), None));
            }
            return Ok((
                String::new(),
                Some(captured_batch(
                    self.grammar.as_ref().expect("batch grammar"),
                    &self.tools,
                    &text,
                    call_id,
                )?),
            ));
        }
        for tool in &self.tools {
            if let Some(body) = text.strip_prefix(&tool.prefix) {
                if !complete {
                    return Ok((String::new(), None));
                }
                let input = body
                    .strip_suffix(&tool.suffix)
                    .ok_or("completed tool is missing its closing marker")?;
                let call = if tool.custom {
                    ChatToolCall {
                        id: Some(call_id.to_string()),
                        kind: Some("custom".to_string()),
                        function: None,
                        custom: Some(ChatToolCustom {
                            name: tool.name.clone(),
                            input: input.to_string(),
                        }),
                        xeno_grammar: Some(GrammarCompletion { complete }),
                    }
                } else {
                    let _: serde_json::Map<String, Value> = serde_json::from_str(input)
                        .map_err(|_| "completed function arguments are not a JSON object")?;
                    ChatToolCall {
                        id: Some(call_id.to_string()),
                        kind: Some("function".to_string()),
                        function: Some(ChatToolFunction {
                            name: tool.name.clone(),
                            arguments: input.to_string(),
                        }),
                        custom: None,
                        xeno_grammar: Some(GrammarCompletion { complete }),
                    }
                };
                return Ok((String::new(), Some(vec![call])));
            }
        }
        if complete {
            Err("completed output has no negotiated response envelope".to_string())
        } else {
            Ok((String::new(), None))
        }
    }

    pub fn decoder(&self, call_id: String) -> ToolStreamDecoder {
        ToolStreamDecoder {
            tools: self.tools.clone(),
            allow_text: self.allow_text,
            call_id,
            selected: None,
            text_reply: false,
            pending: String::new(),
            identified: false,
            function_reply: false,
            allow_functions: self.allow_functions,
            batch_reply: false,
            grammar: self.grammar.clone(),
            allow_batch: self.allow_batch,
        }
    }
}

fn literal(text: &str) -> String {
    serde_json::to_string(text).expect("grammar literal serializes")
}

fn function_batch(tools: &[Tool], text: &str, call_id: &str) -> Result<Vec<ChatToolCall>, String> {
    let body = text
        .strip_prefix(FUNCTIONS_PREFIX)
        .and_then(|body| body.strip_suffix(FUNCTIONS_SUFFIX))
        .ok_or("function batch is missing its envelope")?;
    let calls: Vec<Value> =
        serde_json::from_str(body).map_err(|_| "function batch must be a JSON array")?;
    if calls.is_empty() || calls.len() > MAX_TOOLS {
        return Err("function batch count is outside its bound".to_string());
    }
    calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .ok_or("function batch name is missing")?;
            if !tools.iter().any(|tool| !tool.custom && tool.name == name) {
                return Err("function batch names an undeclared function".to_string());
            }
            let arguments = call
                .get("arguments")
                .filter(|value| value.is_object())
                .ok_or("function batch arguments must be an object")?;
            Ok(ChatToolCall {
                id: Some(format!("{call_id}_{index}")),
                kind: Some("function".to_string()),
                function: Some(ChatToolFunction {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                }),
                custom: None,
                xeno_grammar: Some(GrammarCompletion { complete: true }),
            })
        })
        .collect()
}

fn captured_batch(
    grammar: &Grammar,
    tools: &[Tool],
    text: &str,
    call_id: &str,
) -> Result<Vec<ChatToolCall>, String> {
    if !text.starts_with(CALLS_PREFIX) || !text.ends_with(CALLS_SUFFIX) {
        return Err("captured batch is missing its envelope".to_string());
    }
    let captures = grammar.captures(text)?;
    let mut calls = Vec::new();
    for (name, captured) in captures {
        let Some(index) = name
            .strip_prefix("__LIST_APPEND:xeno_tool")
            .and_then(|index| index.parse::<usize>().ok())
        else {
            continue;
        };
        let tool = tools
            .get(index)
            .ok_or("captured batch names an unknown tool")?;
        let captured = captured.strip_prefix('\n').unwrap_or(&captured);
        let input = captured
            .strip_prefix(&tool.prefix)
            .and_then(|input| input.strip_suffix(&tool.suffix))
            .ok_or("captured tool boundary is invalid")?;
        let mut call = ChatToolCall {
            id: Some(format!("{call_id}_{}", calls.len())),
            kind: Some(if tool.custom { "custom" } else { "function" }.to_string()),
            function: None,
            custom: None,
            xeno_grammar: Some(GrammarCompletion { complete: true }),
        };
        if tool.custom {
            call.custom = Some(ChatToolCustom {
                name: tool.name.clone(),
                input: input.to_string(),
            });
        } else {
            let _: serde_json::Map<String, Value> = serde_json::from_str(input)
                .map_err(|_| "captured function arguments are not a JSON object")?;
            call.function = Some(ChatToolFunction {
                name: tool.name.clone(),
                arguments: input.to_string(),
            });
        }
        calls.push(call);
    }
    if calls.is_empty() || calls.len() > MAX_TOOLS {
        return Err("captured batch count is outside its bound".to_string());
    }
    Ok(calls)
}

pub(crate) struct ToolStreamDecoder {
    tools: Vec<Tool>,
    allow_text: bool,
    call_id: String,
    selected: Option<usize>,
    text_reply: bool,
    pending: String,
    identified: bool,
    function_reply: bool,
    allow_functions: bool,
    batch_reply: bool,
    grammar: Option<Grammar>,
    allow_batch: bool,
}

impl ToolStreamDecoder {
    pub fn push(&mut self, piece: &str) -> Result<Vec<Value>, String> {
        if self.pending.len().saturating_add(piece.len()) > 4 * 1024 * 1024 {
            return Err("tool stream buffer exceeds its 4 MiB bound".to_string());
        }
        self.pending.push_str(piece);
        if self.function_reply || self.batch_reply {
            return Ok(Vec::new());
        }
        if !self.identified {
            if self.allow_batch && self.pending.starts_with(CALLS_PREFIX) {
                self.batch_reply = true;
                self.identified = true;
                return Ok(Vec::new());
            } else if self.allow_functions && self.pending.starts_with(FUNCTIONS_PREFIX) {
                self.function_reply = true;
                self.identified = true;
                return Ok(Vec::new());
            } else if self.allow_text && self.pending.starts_with(TEXT_PREFIX) {
                self.pending.drain(..TEXT_PREFIX.len());
                self.text_reply = true;
                self.identified = true;
            } else if let Some(index) = self
                .tools
                .iter()
                .position(|tool| self.pending.starts_with(&tool.prefix))
            {
                self.pending.drain(..self.tools[index].prefix.len());
                self.selected = Some(index);
                self.identified = true;
            } else if !(self.allow_text && TEXT_PREFIX.starts_with(&self.pending))
                && !(self.allow_functions && FUNCTIONS_PREFIX.starts_with(&self.pending))
                && !(self.allow_batch && CALLS_PREFIX.starts_with(&self.pending))
                && !self
                    .tools
                    .iter()
                    .any(|tool| tool.prefix.starts_with(&self.pending))
            {
                return Err("streamed response is outside its negotiated envelope".to_string());
            } else {
                return Ok(Vec::new());
            }
            let mut deltas = vec![self.delta(String::new(), true)];
            deltas.extend(self.flush(false)?);
            return Ok(deltas);
        }
        self.flush(false)
    }

    pub fn finish(&mut self, complete: bool) -> Result<Vec<Value>, String> {
        if self.batch_reply {
            if !complete {
                return Ok(Vec::new());
            }
            let calls = captured_batch(
                self.grammar.as_ref().ok_or("missing batch grammar")?,
                &self.tools,
                &self.pending,
                &self.call_id,
            )?;
            self.pending.clear();
            return Ok(calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    json!({ "tool_calls": [{
                "index": index, "id": call.id, "type": call.kind, "custom": call.custom,
                "function": call.function, "xeno_grammar": { "complete": true } }] })
                })
                .collect());
        }
        if self.function_reply {
            if !complete {
                return Ok(Vec::new());
            }
            let calls = function_batch(&self.tools, &self.pending, &self.call_id)?;
            self.pending.clear();
            return Ok(calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    json!({ "tool_calls": [{
                "index": index, "id": call.id, "type": "function", "function": call.function,
                "xeno_grammar": { "complete": true } }] })
                })
                .collect());
        }
        let mut deltas = self.flush(complete)?;
        if let Some(index) = self.selected {
            deltas.push(json!({ "tool_calls": [{ "index": 0,
                "type": if self.tools[index].custom { "custom" } else { "function" },
                "xeno_grammar": { "complete": complete } }] }));
        }
        Ok(deltas)
    }

    fn flush(&mut self, complete: bool) -> Result<Vec<Value>, String> {
        if self.text_reply {
            let text = std::mem::take(&mut self.pending);
            return Ok(if text.is_empty() {
                Vec::new()
            } else {
                vec![json!({ "content": text })]
            });
        }
        let Some(index) = self.selected else {
            return Ok(Vec::new());
        };
        let suffix = &self.tools[index].suffix;
        let length = if complete {
            if !self.pending.ends_with(suffix) {
                return Err("completed tool stream is missing its closing marker".to_string());
            }
            self.pending.len() - suffix.len()
        } else {
            self.pending.len().saturating_sub(suffix.len())
        };
        let mut boundary = length;
        while !self.pending.is_char_boundary(boundary) {
            boundary -= 1;
        }
        let text: String = self.pending.drain(..boundary).collect();
        if complete {
            self.pending.clear();
        }
        Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![self.delta(text, false)]
        })
    }

    fn delta(&self, text: String, identity: bool) -> Value {
        let Some(index) = self.selected else {
            return json!({ "content": text });
        };
        let tool = &self.tools[index];
        let mut call =
            json!({ "index": 0, "type": if tool.custom { "custom" } else { "function" } });
        if identity {
            call["id"] = json!(self.call_id);
        }
        let kind = if tool.custom { "custom" } else { "function" };
        call[kind] = if tool.custom {
            json!({ "input": text })
        } else {
            json!({ "arguments": text })
        };
        if identity {
            call[kind]["name"] = json!(tool.name);
        }
        json!({ "tool_calls": [call] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definitions() -> Vec<Value> {
        vec![
            json!({ "type": "custom", "custom": { "name": "Patch", "format": {
            "type": "grammar", "syntax": "gbnf", "definition": "root ::= \"hello\"" } } }),
            json!({ "type": "function", "function": { "name": "Read", "parameters": {
                "type": "object", "properties": { "path": { "type": "string", "const": "allowed" } },
                "required": ["path"], "additionalProperties": false } } }),
        ]
    }

    #[test]
    fn constrained_tool_selection_and_arguments_are_one_grammar() {
        let plan = ToolPlan::prepare(Some(&definitions()), None).unwrap();
        let grammar = plan.grammar.as_ref().unwrap();
        for output in [
            "<xeno_tool_0>\nhello\n</xeno_tool_0>",
            "<xeno_tool_1>\n{\"path\":\"allowed\"}\n</xeno_tool_1>",
            "<xeno_text>\nanswer",
        ] {
            let state = grammar.advance(&grammar.start(), output).unwrap();
            assert!(grammar.is_complete(&state));
        }
        assert!(grammar
            .advance(&grammar.start(), "<xeno_tool_0>\nwrong")
            .is_none());
        assert!(grammar
            .advance(&grammar.start(), "<xeno_tool_1>\n{\"path\":\"wrong\"}")
            .is_none());
    }

    #[test]
    fn explicit_tool_choice_is_enforced_and_invalid_choices_are_refused() {
        let definitions = definitions();
        let plan = ToolPlan::prepare(
            Some(&definitions),
            Some(&json!({ "type": "custom", "custom": { "name": "Patch" } })),
        )
        .unwrap();
        let grammar = plan.grammar.as_ref().unwrap();
        assert!(grammar.advance(&grammar.start(), "<xeno_text>\n").is_none());
        assert!(grammar
            .advance(&grammar.start(), "<xeno_tool_1>\n")
            .is_none());
        assert!(ToolPlan::prepare(Some(&definitions), Some(&json!("none")))
            .unwrap()
            .grammar
            .is_none());
        assert!(ToolPlan::prepare(None, Some(&json!("required"))).is_err());
        assert!(ToolPlan::prepare(
            Some(&definitions),
            Some(&json!({ "type": "function", "function": { "name": "Patch" } }))
        )
        .is_err());
    }

    #[test]
    fn custom_stream_and_response_preserve_exact_input_and_refuse_truncation() {
        let plan = ToolPlan::prepare(Some(&definitions()), None).unwrap();
        let mut decoder = plan.decoder("call-proof".to_string());
        let mut deltas = Vec::new();
        for piece in ["<xeno_", "tool_0>\nhe", "llo\n</xeno_tool_", "0>"] {
            deltas.extend(decoder.push(piece).unwrap());
        }
        deltas.extend(decoder.finish(true).unwrap());
        let input = deltas
            .iter()
            .filter_map(|delta| delta["tool_calls"][0]["custom"]["input"].as_str())
            .collect::<String>();
        assert_eq!(input, "hello");
        let (_, calls) = plan
            .response(
                "<xeno_tool_0>\nhello\n</xeno_tool_0>".to_string(),
                true,
                "call-proof",
            )
            .unwrap();
        assert_eq!(calls.unwrap()[0].custom.as_ref().unwrap().input, "hello");
        assert!(plan
            .response("<xeno_tool_0>\nhel".to_string(), false, "call-proof")
            .unwrap()
            .1
            .is_none());
        assert!(
            deltas.last().unwrap()["tool_calls"][0]["xeno_grammar"]["complete"]
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn ordinary_text_and_unicode_tool_inputs_do_not_leak_envelope_markers() {
        let plan = ToolPlan::prepare(Some(&definitions()), None).unwrap();
        let mut decoder = plan.decoder("text".to_string());
        let mut deltas = decoder.push("<xeno_text>\n\u{03b3} hello").unwrap();
        deltas.extend(decoder.push(" world").unwrap());
        deltas.extend(decoder.finish(true).unwrap());
        let text = deltas
            .iter()
            .filter_map(|delta| delta["content"].as_str())
            .collect::<String>();
        assert_eq!(text, "\u{03b3} hello world");
        assert!(deltas.iter().all(|delta| delta.get("tool_calls").is_none()));
        let mut definitions = definitions();
        definitions[0]["custom"]["format"]["definition"] = json!("root ::= [^\\x00]*");
        let plan = ToolPlan::prepare(Some(&definitions), Some(&json!("required"))).unwrap();
        let mut decoder = plan.decoder("unicode".to_string());
        let input = "\u{03b3} \n \u{1f642}";
        let raw = format!("<xeno_tool_0>\n{input}\n</xeno_tool_0>");
        let mut deltas = Vec::new();
        for piece in raw.chars() {
            deltas.extend(decoder.push(&piece.to_string()).unwrap());
        }
        deltas.extend(decoder.finish(true).unwrap());
        let actual = deltas
            .iter()
            .filter_map(|delta| delta["tool_calls"][0]["custom"]["input"].as_str())
            .collect::<String>();
        assert_eq!(actual, input);
    }

    #[test]
    fn schema_constraints_are_never_silently_weakened() {
        let mut definitions = definitions();
        definitions[1]["function"]["parameters"] = json!({ "type": "object", "properties": {
            "path": { "oneOf": [{"type":"string"}, {"type":"string", "minLength":1}] } } });
        assert!(ToolPlan::prepare(Some(&definitions), None).is_err());
    }

    #[test]
    fn multiple_function_calls_keep_distinct_ids_and_stream_indices() {
        let plan = ToolPlan::prepare(Some(&definitions()), Some(&json!("required"))).unwrap();
        let output = concat!(
            "<xeno_functions>\n[",
            "{\"name\":\"Read\",\"arguments\":{\"path\":\"allowed\"}},",
            "{\"name\":\"Read\",\"arguments\":{\"path\":\"allowed\"}}]",
            "\n</xeno_functions>"
        );
        let grammar = plan.grammar.as_ref().unwrap();
        let state = grammar.advance(&grammar.start(), output).unwrap();
        assert!(grammar.is_complete(&state));
        let (_, calls) = plan.response(output.to_string(), true, "batch").unwrap();
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        let mut decoder = plan.decoder("batch".to_string());
        assert!(decoder.push(output).unwrap().is_empty());
        let deltas = decoder.finish(true).unwrap();
        assert_eq!(deltas[0]["tool_calls"][0]["index"], 0);
        assert_eq!(deltas[1]["tool_calls"][0]["index"], 1);
        let single =
            ToolPlan::prepare_with_parallel(Some(&definitions()), Some(&json!("required")), false)
                .unwrap();
        assert!(single
            .grammar
            .as_ref()
            .unwrap()
            .advance(&single.grammar.as_ref().unwrap().start(), output)
            .is_none());
    }

    #[test]
    fn mixed_and_repeated_custom_batches_use_captures_not_payload_delimiters() {
        let mut definitions = definitions();
        let raw = "hello\n</xeno_tool_0>\n<xeno_tool_1>\nnot a call";
        definitions[0]["custom"]["format"]["definition"] =
            json!(format!("root ::= {}", literal(raw)));
        let plan = ToolPlan::prepare(Some(&definitions), Some(&json!("required"))).unwrap();
        let custom = format!("<xeno_tool_0>\n{raw}\n</xeno_tool_0>");
        let output = format!("{CALLS_PREFIX}{custom}\n<xeno_tool_1>\n{{\"path\":\"allowed\"}}\n</xeno_tool_1>\n{custom}{CALLS_SUFFIX}");
        let grammar = plan.grammar.as_ref().unwrap();
        let mut state = grammar.start();
        for (index, ch) in output.char_indices() {
            state = grammar
                .advance(&state, &ch.to_string())
                .unwrap_or_else(|| panic!("batch rejected byte {index} ({ch:?})"));
        }
        assert!(grammar.is_complete(&state));
        let (_, calls) = plan.response(output.clone(), true, "mixed").unwrap();
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].custom.as_ref().unwrap().input, raw);
        assert_eq!(calls[1].function.as_ref().unwrap().name, "Read");
        assert_eq!(calls[2].custom.as_ref().unwrap().input, raw);
        assert_ne!(calls[0].id, calls[2].id);
        let mut decoder = plan.decoder("mixed".to_string());
        for ch in output.chars() {
            assert!(decoder.push(&ch.to_string()).unwrap().is_empty());
        }
        let deltas = decoder.finish(true).unwrap();
        assert_eq!(deltas.len(), 3);
        assert_eq!(deltas[2]["tool_calls"][0]["custom"]["input"], raw);
        let mut partial = plan.decoder("partial".to_string());
        partial
            .push(&output[..output.len() - CALLS_SUFFIX.len()])
            .unwrap();
        assert!(partial.finish(false).unwrap().is_empty());
    }

    #[test]
    fn explicit_call_count_bounds_are_enforced_and_conflicts_refused() {
        let definitions = definitions();
        let plan = ToolPlan::prepare_bounded(
            Some(&definitions),
            Some(&json!("required")),
            true,
            Some(2),
            Some(2),
        )
        .unwrap();
        let grammar = plan.grammar.as_ref().unwrap();
        assert!(grammar
            .advance(&grammar.start(), "<xeno_tool_0>\nhello\n</xeno_tool_0>")
            .is_none());
        assert!(
            ToolPlan::prepare_bounded(Some(&definitions), None, false, Some(2), Some(2)).is_err()
        );
        assert!(
            ToolPlan::prepare_bounded(Some(&definitions), None, true, Some(3), Some(2)).is_err()
        );
        assert!(
            ToolPlan::prepare_bounded(Some(&definitions), None, true, Some(0), Some(2)).is_err()
        );
        assert!(ToolPlan::prepare_bounded(
            Some(&definitions),
            Some(&json!("none")),
            true,
            Some(1),
            None
        )
        .is_err());
    }
}
