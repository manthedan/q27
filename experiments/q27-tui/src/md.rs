//! Lightweight markdown → ratatui Lines (subset good enough for chat).

use crate::theme::Theme;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

/// Render markdown-ish text into display lines with basic styling.
/// Supports: `#` headings, fenced ``` blocks, inline `code`, **bold**, *italic*,
/// `-` / `*` lists, and plain paragraphs. Unknown markup is shown raw.
pub fn render_markdown(text: &str, theme: &Theme, indent: &str) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut in_fence = false;
    let mut fence_lang = String::new();

    for raw in text.lines() {
        let line = raw;
        if let Some(rest) = line.strip_prefix("```") {
            if in_fence {
                in_fence = false;
                fence_lang.clear();
                out.push(Line::from(Span::styled(
                    format!("{indent}```"),
                    Style::default().fg(theme.dim),
                )));
            } else {
                in_fence = true;
                fence_lang = rest.trim().to_string();
                let label = if fence_lang.is_empty() {
                    format!("{indent}```")
                } else {
                    format!("{indent}```{fence_lang}")
                };
                out.push(Line::from(Span::styled(
                    label,
                    Style::default().fg(theme.dim),
                )));
            }
            continue;
        }
        if in_fence {
            out.push(Line::from(Span::styled(
                format!("{indent}{line}"),
                Style::default().fg(theme.code),
            )));
            continue;
        }

        // Headings
        if let Some(body) = strip_heading(line) {
            out.push(Line::from(Span::styled(
                format!("{indent}{body}"),
                Style::default()
                    .fg(theme.md_heading)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        // List items
        let list_body = line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .or_else(|| {
                if line.len() >= 3
                    && line.as_bytes()[0].is_ascii_digit()
                    && line.as_bytes().get(1) == Some(&b'.')
                    && line.as_bytes().get(2) == Some(&b' ')
                {
                    Some(&line[3..])
                } else {
                    None
                }
            });
        if let Some(body) = list_body {
            let mut spans = vec![Span::styled(
                format!("{indent}• "),
                Style::default().fg(theme.dim),
            )];
            spans.extend(inline_spans(body, theme));
            out.push(Line::from(spans));
            continue;
        }

        if line.trim().is_empty() {
            out.push(Line::from(""));
            continue;
        }

        let mut spans = vec![Span::raw(indent.to_string())];
        spans.extend(inline_spans(line, theme));
        out.push(Line::from(spans));
    }
    out
}

fn strip_heading(line: &str) -> Option<&str> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let mut i = 0;
    for c in t.chars() {
        if c == '#' {
            i += 1;
            if i > 6 {
                return None;
            }
        } else {
            break;
        }
    }
    if i == 0 {
        return None;
    }
    let rest = t[i..].trim_start();
    if rest.is_empty() {
        return None;
    }
    Some(rest)
}

fn inline_spans(text: &str, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut buf = String::new();

    let flush_plain = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::raw(std::mem::take(buf)));
        }
    };

    while i < chars.len() {
        // **bold**
        if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
            if let Some(end) = find_close(&chars, i + 2, "**") {
                flush_plain(&mut buf, &mut spans);
                let inner: String = chars[i + 2..end].iter().collect();
                spans.push(Span::styled(
                    inner,
                    Style::default()
                        .fg(theme.bold)
                        .add_modifier(Modifier::BOLD),
                ));
                i = end + 2;
                continue;
            }
        }
        // *italic*
        if chars[i] == '*' {
            if let Some(end) = find_close_single(&chars, i + 1, '*') {
                flush_plain(&mut buf, &mut spans);
                let inner: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(
                    inner,
                    Style::default().add_modifier(Modifier::ITALIC),
                ));
                i = end + 1;
                continue;
            }
        }
        // `code`
        if chars[i] == '`' {
            if let Some(end) = find_close_single(&chars, i + 1, '`') {
                flush_plain(&mut buf, &mut spans);
                let inner: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(
                    format!("`{inner}`"),
                    Style::default().fg(theme.code),
                ));
                i = end + 1;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }
    flush_plain(&mut buf, &mut spans);
    spans
}

fn find_close(chars: &[char], start: usize, delim: &str) -> Option<usize> {
    let d: Vec<char> = delim.chars().collect();
    let mut i = start;
    while i + d.len() <= chars.len() {
        if chars[i..i + d.len()] == d[..] {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn find_close_single(chars: &[char], start: usize, delim: char) -> Option<usize> {
    let mut i = start;
    while i < chars.len() {
        if chars[i] == delim {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Split assistant text into optional thinking + visible body.
/// Recognizes `<think>…</think>` (Qwen / ChatML reconstruction).
/// Unclosed open tags still count as thinking (rare at turn end).
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    let live = split_thinking_live(text);
    let thinking = match live.thinking {
        Some(s) if s.trim().is_empty() => None,
        other => other,
    };
    (thinking, live.body)
}

/// Result of a (possibly still-streaming) thinking split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveThinking {
    /// Thinking content so far (present once `<think>` has been seen).
    pub thinking: Option<String>,
    /// True while `<think>` is open and `</think>` has not arrived yet.
    pub open: bool,
    /// Visible assistant body (prefix before open + text after close).
    pub body: String,
}

/// Streaming-aware split: open `<think>` without a close still counts as thinking.
pub fn split_thinking_live(text: &str) -> LiveThinking {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if let Some(start) = text.find(OPEN) {
        let after_open = start + OPEN.len();
        let prefix = text[..start].trim_end();
        if let Some(rel) = text[after_open..].find(CLOSE) {
            let think = text[after_open..after_open + rel].trim().to_string();
            let rest = text[after_open + rel + CLOSE.len()..].trim_start();
            let body = if prefix.is_empty() {
                rest.to_string()
            } else if rest.is_empty() {
                prefix.to_string()
            } else {
                format!("{prefix}\n{rest}")
            };
            let thinking = if think.is_empty() {
                None
            } else {
                Some(think)
            };
            return LiveThinking {
                thinking,
                open: false,
                body,
            };
        }
        // Open tag, no close yet — entire remainder is streaming thinking.
        let think = text[after_open..].to_string();
        return LiveThinking {
            thinking: Some(think),
            open: true,
            body: prefix.to_string(),
        };
    }
    LiveThinking {
        thinking: None,
        open: false,
        body: text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemeId;

    #[test]
    fn split_think_block() {
        let (t, b) = split_thinking("<think>\nhmm\n</think>\nHello **world**");
        assert_eq!(t.as_deref(), Some("hmm"));
        assert_eq!(b, "Hello **world**");
    }

    #[test]
    fn split_think_streaming_open() {
        let live = split_thinking_live("<think>\nstill thinking");
        assert!(live.open);
        assert_eq!(live.thinking.as_deref(), Some("\nstill thinking"));
        assert_eq!(live.body, "");
    }

    #[test]
    fn split_think_streaming_closed() {
        let live = split_thinking_live("<think>\nhmm\n</think>\nHi");
        assert!(!live.open);
        assert_eq!(live.thinking.as_deref(), Some("hmm"));
        assert_eq!(live.body, "Hi");
    }

    #[test]
    fn split_think_no_tags() {
        let live = split_thinking_live("just text");
        assert!(!live.open);
        assert!(live.thinking.is_none());
        assert_eq!(live.body, "just text");
    }

    #[test]
    fn render_has_heading() {
        let th = ThemeId::Dark.palette();
        let lines = render_markdown("# Title\n\nplain", &th, "  ");
        assert!(!lines.is_empty());
    }
}
