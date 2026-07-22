//! UI model derived only from ServerEvents (FP1 principle 1).

use crate::md::split_thinking;
use crate::proto::ServerEvent;
use crate::theme::ThemeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Starting,
    Idle,
    Prefill,
    Generating,
    Tool,
    Compacting,
    Error,
    Stopped,
}

/// Strip terminal control bytes before content reaches the scrollback:
/// tool output and replayed history can contain ESC/OSC/C1 sequences that
/// would spoof or alter the terminal when rendered (r11 codex P2).
/// Keeps \n and \t.
fn sanitize_terminal_text(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\n' | '\t' => c,
            c if (c as u32) < 0x20 || (0x7f..=0x9f).contains(&(c as u32)) => {
                '\u{fffd}'
            }
            c => c,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum Block {
    User(String),
    Assistant {
        thinking: Option<String>,
        body: String,
    },
    Tool {
        kind: String,
        detail: String,
        body: String,
        exit: Option<i32>,
        open: bool,
    },
    Notice {
        severity: String,
        text: String,
    },
    System(String),
}

#[derive(Debug)]
pub struct Model {
    pub phase: Phase,
    pub features: Vec<String>,
    pub model_path: String,
    pub ctx_used: u32,
    pub ctx_size: u32,
    pub prefill_done: u32,
    pub prefill_total: u32,
    pub gen_tokens: u32,
    pub queue_len: u32,
    pub scrollback: Vec<Block>,
    pub assistant_buf: String,
    pub last_error: Option<String>,
    pub status_line: String,
    pub input_enabled: bool,
    pub bye_reason: Option<String>,
    pub saw_hello: bool,
    /// When false, thinking spans render as a one-line summary.
    pub show_thinking: bool,
    pub theme: ThemeId,
    pub markdown: bool,
}

impl Default for Phase {
    fn default() -> Self {
        Phase::Starting
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            phase: Phase::Starting,
            features: Vec::new(),
            model_path: String::new(),
            ctx_used: 0,
            ctx_size: 0,
            prefill_done: 0,
            prefill_total: 0,
            gen_tokens: 0,
            queue_len: 0,
            scrollback: Vec::new(),
            assistant_buf: String::new(),
            last_error: None,
            status_line: String::new(),
            input_enabled: false,
            bye_reason: None,
            saw_hello: false,
            show_thinking: false,
            theme: ThemeId::Dark,
            markdown: true,
        }
    }
}

impl Model {
    pub fn apply(&mut self, ev: &ServerEvent) {
        if let Some(s) = ev.state.as_deref() {
            self.phase = match s {
                "starting" => Phase::Starting,
                "idle" => Phase::Idle,
                "generating" => {
                    if self.phase == Phase::Tool {
                        Phase::Tool
                    } else {
                        Phase::Generating
                    }
                }
                "tool_running" => Phase::Tool,
                "compacting" => Phase::Compacting,
                "error" => Phase::Error,
                "stopped" => Phase::Stopped,
                "session_io" => Phase::Idle,
                "stopping" => Phase::Stopped,
                _ => self.phase,
            };
        }

        match ev.type_name.as_str() {
            "hello" => {
                self.saw_hello = true;
                self.features = ev.features.clone().unwrap_or_default();
                self.model_path = ev.model_path.clone().unwrap_or_default();
                if let Some(c) = ev.context {
                    self.ctx_size = c;
                }
                self.status_line = format!(
                    "FP1 hello · features=[{}]",
                    self.features.join(", ")
                );
                self.input_enabled = false;
            }
            "idle" => {
                self.phase = Phase::Idle;
                self.input_enabled = true;
                if let Some(u) = ev.ctx_used {
                    self.ctx_used = u;
                }
                if let Some(s) = ev.ctx_size {
                    self.ctx_size = s;
                }
                if let Some(q) = ev.queue_len {
                    self.queue_len = q;
                }
                self.status_line = "idle".into();
                self.finish_assistant_if_any();
            }
            "prefill_progress" => {
                self.phase = Phase::Prefill;
                self.input_enabled = self.has_queue_feature();
                let prompt = ev.prompt_tokens.unwrap_or(0);
                let cached = ev.cached_tokens.unwrap_or(0);
                let pref = ev.prefill_tokens.unwrap_or(0);
                self.prefill_total = prompt;
                self.prefill_done = cached.saturating_add(pref).min(prompt);
                self.ctx_used = self.prefill_done;
                self.status_line = format!(
                    "prefill {}/{}",
                    self.prefill_done, self.prefill_total
                );
            }
            "text_delta" => {
                self.phase = Phase::Generating;
                self.input_enabled = self.has_queue_feature();
                if let Some(t) = ev.payload_text() {
                    self.assistant_buf.push_str(&t);
                }
                if let Some(o) = ev.output_tokens {
                    self.gen_tokens = o;
                }
                self.status_line = format!("generating · {} tok", self.gen_tokens);
            }
            "turn_done" => {
                let toolish = ev.tool_call_complete.unwrap_or(false);
                if !toolish {
                    self.finish_assistant_if_any();
                }
                if let Some(o) = ev.output_tokens {
                    self.gen_tokens = o;
                }
                let st = ev.status.as_deref().unwrap_or("ok");
                let note = ev
                    .payload_text()
                    .or_else(|| ev.text.clone())
                    .filter(|t| !t.is_empty());
                self.status_line = if toolish {
                    "calling tools…".into()
                } else if let Some(n) = note {
                    // e.g. "thinking token budget reached" on length-style stop
                    format!("turn {st} · {n}")
                } else {
                    format!("turn {st}")
                };
                if st == "cancelled" {
                    self.scrollback.push(Block::Notice {
                        severity: "info".into(),
                        text: "interrupted".into(),
                    });
                }
            }
            "tool_start" => {
                self.phase = Phase::Tool;
                self.input_enabled = self.has_queue_feature();
                let kind = ev.tool_kind.clone().unwrap_or_else(|| "tool".into());
                let detail = ev.detail.clone().unwrap_or_default();
                self.scrollback.push(Block::Tool {
                    kind: kind.clone(),
                    detail,
                    body: String::new(),
                    exit: None,
                    open: true,
                });
                self.status_line = format!("tool {kind}");
            }
            "tool_output" => {
                // Binary chunks arrive base64 with invalid UTF-8: show a
                // visible placeholder instead of dropping them (r11 P2),
                // and sanitize control bytes before render (r11 P2).
                let t = match ev.payload_text() {
                    Some(t) => sanitize_terminal_text(&t),
                    None => {
                        let n = ev.payload_bytes().map(|b| b.len()).unwrap_or(0);
                        format!("\u{27e8}binary chunk: {n} bytes\u{27e9}")
                    }
                };
                if let Some(Block::Tool { body, open: true, .. }) =
                    self.scrollback.iter_mut().rev().find(|b| {
                        matches!(b, Block::Tool { open: true, .. })
                    })
                {
                    body.push_str(&t);
                } else {
                    let kind = ev.tool_kind.clone().unwrap_or_else(|| "tool".into());
                    self.scrollback.push(Block::Tool {
                        kind,
                        detail: String::new(),
                        body: t,
                        exit: None,
                        open: true,
                    });
                }
            }
            "history" => {
                // Restored-session replay (r11 codex P2): hydrate scrollback
                // from the agent's loaded transcript.
                if let Some(items) = &ev.items {
                    let count = items.len();
                    for item in items {
                        let text = sanitize_terminal_text(
                            item.text.as_deref().unwrap_or(""),
                        );
                        match item.role.as_deref() {
                            Some("user") => self.push_user(&text),
                            Some("assistant") => {
                                self.scrollback.push(Block::Assistant {
                                    thinking: None,
                                    body: text,
                                })
                            }
                            _ => {}
                        }
                    }
                    if count > 0 {
                        self.status_line = format!("restored {count} messages");
                    }
                }
            }
            "tool_done" => {
                let exit = ev.tool_exit_code;
                if let Some(Block::Tool {
                    exit: e, open, ..
                }) = self.scrollback.iter_mut().rev().find(|b| {
                    matches!(b, Block::Tool { open: true, .. })
                }) {
                    *e = exit;
                    *open = false;
                }
                let kind = ev.tool_kind.as_deref().unwrap_or("tool");
                self.status_line = format!(
                    "tool {kind} done ({})",
                    exit.map(|c| c.to_string()).unwrap_or_else(|| "?".into())
                );
            }
            "rejected" => {
                let code = ev.code.as_deref().unwrap_or("");
                let text = ev
                    .payload_text()
                    .or_else(|| ev.text.clone())
                    .unwrap_or_default();
                let msg = if code.is_empty() {
                    text
                } else {
                    format!("[{code}] {text}")
                };
                self.scrollback.push(Block::Notice {
                    severity: "error".into(),
                    text: msg.clone(),
                });
                self.last_error = Some(msg);
                if matches!(ev.state.as_deref(), Some("idle") | None) {
                    self.input_enabled = true;
                    self.phase = Phase::Idle;
                } else {
                    // Busy-state rejection of a slash op: the op completed,
                    // so re-open input for queueing when advertised (codex
                    // P2); the active turn keeps the phase.
                    self.input_enabled = self.has_queue_feature();
                }
            }
            "notice" => {
                let sev = ev.severity.clone().unwrap_or_else(|| "info".into());
                let text = ev
                    .payload_text()
                    .or_else(|| ev.text.clone())
                    .unwrap_or_default();
                self.scrollback.push(Block::Notice {
                    severity: sev,
                    text,
                });
                if matches!(ev.state.as_deref(), Some("idle") | None) {
                    self.input_enabled = true;
                    self.phase = Phase::Idle;
                } else {
                    self.input_enabled = self.has_queue_feature();
                }
            }
            "error" | "generation_stalled" => {
                let text = ev
                    .payload_text()
                    .or_else(|| ev.text.clone())
                    .unwrap_or_else(|| ev.type_name.clone());
                self.phase = Phase::Error;
                self.last_error = Some(text.clone());
                self.scrollback.push(Block::Notice {
                    severity: "error".into(),
                    text,
                });
                self.input_enabled = true;
            }
            "session_done" => {
                let st = ev.status.as_deref().unwrap_or("ok");
                // Clear the visible transcript only after a SUCCESSFUL reset:
                // a failed durable discard keeps the agent-side transcript,
                // so wiping the UI here would lie about the state (codex P2).
                if ev.action.as_deref() == Some("new") && st == "ok" {
                    self.scrollback.clear();
                    self.assistant_buf.clear();
                }
                let act = ev.action.as_deref().unwrap_or("session");
                self.status_line = format!("{act} {st}");
                if let Some(t) = ev.payload_text().or_else(|| ev.text.clone()) {
                    if !t.is_empty() {
                        let sev = if st == "ok" { "info" } else { "error" };
                        self.scrollback.push(Block::Notice {
                            severity: sev.into(),
                            text: format!("{act}: {t}"),
                        });
                    }
                }
                if matches!(ev.state.as_deref(), Some("idle") | None) {
                    self.input_enabled = true;
                    self.phase = Phase::Idle;
                }
            }
            "bye" => {
                self.phase = Phase::Stopped;
                self.input_enabled = false;
                self.bye_reason = ev.reason.clone();
                self.status_line = format!(
                    "bye ({})",
                    self.bye_reason.as_deref().unwrap_or("?")
                );
            }
            "state" => {
                if ev.state.as_deref() == Some("compacting") {
                    self.phase = Phase::Compacting;
                    self.status_line = "compacting…".into();
                }
            }
            "queue" => {
                if let Some(q) = ev.queue_len {
                    self.queue_len = q;
                }
                self.status_line = format!("queue {}", self.queue_len);
                if self.has_queue_feature() && self.phase != Phase::Stopped {
                    self.input_enabled = true;
                }
            }
            _ => {}
        }
    }

    fn finish_assistant_if_any(&mut self) {
        if !self.assistant_buf.is_empty() {
            let t = std::mem::take(&mut self.assistant_buf);
            let (thinking, body) = split_thinking(&t);
            self.scrollback.push(Block::Assistant { thinking, body });
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.scrollback.push(Block::User(text.to_string()));
        if self.has_queue_feature() {
            self.input_enabled = true;
            self.status_line = if self.phase == Phase::Idle {
                "submitting…".into()
            } else {
                format!("queued · depth {}", self.queue_len.saturating_add(1))
            };
            if self.phase == Phase::Idle {
                self.phase = Phase::Generating;
            }
        } else {
            self.input_enabled = false;
            self.phase = Phase::Generating;
            self.status_line = "submitting…".into();
        }
    }

    pub fn has_queue_feature(&self) -> bool {
        self.features.iter().any(|f| f == "queue")
    }

    pub fn toggle_thinking(&mut self) {
        self.show_thinking = !self.show_thinking;
        self.status_line = if self.show_thinking {
            "thinking: shown".into()
        } else {
            "thinking: collapsed".into()
        };
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.status_line = format!("theme: {}", self.theme.name());
    }

    pub fn toggle_markdown(&mut self) {
        self.markdown = !self.markdown;
        self.status_line = if self.markdown {
            "markdown: on".into()
        } else {
            "markdown: off".into()
        };
    }
}
