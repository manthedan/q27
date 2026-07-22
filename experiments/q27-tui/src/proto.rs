//! Frontend Protocol v1 + legacy jsonl (v0) event types.
//! See `docs/metal/plans/2026-07-21-frontend-protocol-v1.md`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Wire event from `q27-agent` stdout (v0 jsonl or FP1).
#[derive(Debug, Clone, Deserialize)]
pub struct ServerEvent {
    #[serde(default)]
    pub v: Option<u32>,
    #[serde(default)]
    pub seq: u64,
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(default)]
    pub ts_ms: Option<u64>,
    #[serde(default)]
    pub command_id: u64,
    #[serde(default)]
    pub client_req_id: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub data_b64: Option<String>,
    #[serde(default)]
    pub stream: Option<String>,
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub cached_tokens: Option<u32>,
    #[serde(default)]
    pub prefill_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub tool_call_complete: Option<bool>,
    #[serde(default)]
    pub eos_reached: Option<bool>,
    #[serde(default)]
    pub tool_kind: Option<String>,
    #[serde(default)]
    pub tool_exit_code: Option<i32>,
    #[serde(default)]
    pub tool_flags: Option<u32>,
    #[serde(default)]
    pub tool_output_bytes: Option<u32>,
    #[serde(default)]
    pub ctx_used: Option<u32>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub queue_len: Option<u32>,
    #[serde(default)]
    pub protocol: Option<u32>,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(default)]
    pub context: Option<u32>,
    #[serde(default)]
    pub workspace: Option<String>,
    #[serde(default)]
    pub session_path: Option<Value>,
    #[serde(default)]
    pub auto_tools: Option<bool>,
    #[serde(default)]
    pub thinking: Option<bool>,
    #[serde(default)]
    pub max_tool_rounds: Option<u32>,
    #[serde(default)]
    pub features: Option<Vec<String>>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub preflight: Option<bool>,
    #[serde(default)]
    pub items: Option<Vec<HistoryItem>>,
}

/// One replayed transcript message in a `history` event (r11 codex P2).
#[derive(Debug, Clone, Deserialize)]
pub struct HistoryItem {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

impl ServerEvent {
    pub fn parse_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line.trim())
    }

    /// Prefer UTF-8 `text`; fall back to base64 `data_b64`.
    pub fn payload_text(&self) -> Option<String> {
        if let Some(t) = &self.text {
            return Some(t.clone());
        }
        let b64 = self.data_b64.as_deref()?;
        if b64.is_empty() {
            return Some(String::new());
        }
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
    }

    /// Decoded payload bytes regardless of UTF-8 validity — binary tool
    /// chunks must be accounted for, not silently dropped (r11 codex P2).
    pub fn payload_bytes(&self) -> Option<Vec<u8>> {
        if let Some(t) = &self.text {
            return Some(t.clone().into_bytes());
        }
        let b64 = self.data_b64.as_deref()?;
        if b64.is_empty() {
            return Some(Vec::new());
        }
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(b64).ok()
    }

    pub fn has_feature(&self, name: &str) -> bool {
        self.features
            .as_ref()
            .map(|f| f.iter().any(|x| x == name))
            .unwrap_or(false)
    }

    pub fn is_fp1(&self) -> bool {
        self.v == Some(1) || self.type_name == "hello"
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClientMessage {
    pub v: u32,
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_req_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

impl ClientMessage {
    fn bare(op: impl Into<String>, req_id: Option<String>) -> Self {
        Self {
            v: 1,
            op: op.into(),
            client_req_id: req_id,
            text: None,
            kind: None,
            path: None,
            needle: None,
            command: None,
        }
    }

    pub fn prompt(req_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::bare("prompt", Some(req_id.into()))
        }
    }

    pub fn cancel(req_id: impl Into<String>) -> Self {
        Self::bare("cancel", Some(req_id.into()))
    }

    pub fn quit() -> Self {
        Self::bare("quit", None)
    }

    pub fn save(req_id: impl Into<String>) -> Self {
        Self::bare("save", Some(req_id.into()))
    }

    pub fn compact(req_id: impl Into<String>) -> Self {
        Self::bare("compact", Some(req_id.into()))
    }

    pub fn new_session(req_id: impl Into<String>) -> Self {
        Self::bare("new", Some(req_id.into()))
    }

    pub fn session(req_id: impl Into<String>) -> Self {
        Self::bare("session", Some(req_id.into()))
    }

    pub fn help(req_id: impl Into<String>) -> Self {
        Self::bare("help", Some(req_id.into()))
    }

    pub fn queue_clear(req_id: impl Into<String>) -> Self {
        Self::bare("queue_clear", Some(req_id.into()))
    }

    pub fn tool_read(req_id: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            kind: Some("read".into()),
            path: Some(path.into()),
            ..Self::bare("tool", Some(req_id.into()))
        }
    }

    pub fn tool_search(
        req_id: impl Into<String>,
        path: impl Into<String>,
        needle: impl Into<String>,
    ) -> Self {
        Self {
            kind: Some("search".into()),
            path: Some(path.into()),
            needle: Some(needle.into()),
            ..Self::bare("tool", Some(req_id.into()))
        }
    }

    pub fn tool_shell(req_id: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            kind: Some("shell".into()),
            command: Some(command.into()),
            ..Self::bare("tool", Some(req_id.into()))
        }
    }

    pub fn to_line(&self) -> Result<String, serde_json::Error> {
        let mut s = serde_json::to_string(self)?;
        s.push('\n');
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_v0_text_delta() {
        let line = r#"{"seq":12,"command_id":3,"type":"text_delta","state":"generating","status":"ok","data_b64":"SGk=","prompt_tokens":0,"cached_tokens":0,"prefill_tokens":0,"output_tokens":0,"tool_call_complete":false,"eos_reached":false,"tool_kind":"none","tool_exit_code":0,"tool_flags":0,"tool_output_bytes":0}"#;
        let ev = ServerEvent::parse_line(line).unwrap();
        assert_eq!(ev.type_name, "text_delta");
        assert_eq!(ev.seq, 12);
        assert_eq!(ev.payload_text().as_deref(), Some("Hi"));
        assert!(!ev.is_fp1());
    }

    #[test]
    fn parse_v1_hello() {
        let line = r#"{"v":1,"seq":1,"type":"hello","ts_ms":1,"command_id":0,"client_req_id":null,"state":"idle","status":"ok","code":null,"protocol":1,"model_path":"m.q27","tokenizer_path":"m.tok","context":8192,"workspace":".","session_path":null,"auto_tools":true,"thinking":true,"max_tool_rounds":8,"features":["tools","selections","compact","session"]}"#;
        let ev = ServerEvent::parse_line(line).unwrap();
        assert_eq!(ev.type_name, "hello");
        assert!(ev.is_fp1());
        assert!(ev.has_feature("tools"));
        assert!(!ev.has_feature("queue"));
    }

    #[test]
    fn client_prompt_line() {
        let m = ClientMessage::prompt("ui-1", "hello");
        let line = m.to_line().unwrap();
        assert!(line.contains(r#""op":"prompt""#));
        assert!(line.ends_with('\n'));
    }
}
