//! Ratatui layout: scrollback / editor / sticky footer.

use crate::app::{Block, Model, Phase};
use crate::md::{render_markdown, split_thinking_live};
use crate::theme::Theme;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block as WBlock, Borders, Paragraph, Wrap};
use ratatui::Frame;
use std::time::Instant;

/// Spinner frames for busy phases.
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Draw the full UI. Returns max scroll offset (0 = top / oldest; max = bottom /
/// current LLM output). Caller clamps `scroll` to this value.
pub fn draw(
    frame: &mut Frame,
    model: &Model,
    input: &str,
    scroll: u16,
    tick: Instant,
) -> u16 {
    let theme = model.theme.palette();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let max_scroll = draw_scrollback(frame, chunks[0], model, scroll, &theme, tick);
    draw_input(frame, chunks[1], model, input, &theme);
    draw_footer(frame, chunks[2], model, &theme, tick);
    max_scroll
}

fn draw_scrollback(
    frame: &mut Frame,
    area: Rect,
    model: &Model,
    scroll: u16,
    theme: &Theme,
    tick: Instant,
) -> u16 {
    let lines = build_scrollback_lines(model, theme, tick);
    let max_scroll = max_scroll_for(&lines, area);

    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((scroll.min(max_scroll), 0));
    frame.render_widget(para, area);
    max_scroll
}

fn max_scroll_for(lines: &[Line<'_>], area: Rect) -> u16 {
    if area.height == 0 {
        return 0;
    }
    let width = area.width.max(1) as usize;
    // Approximate post-wrap line count (Paragraph uses the same wrap).
    let mut wrapped = 0usize;
    for line in lines {
        let w = line.width().max(1);
        wrapped += w.div_ceil(width).max(1);
    }
    wrapped
        .saturating_sub(area.height as usize)
        .min(u16::MAX as usize) as u16
}

fn build_scrollback_lines(model: &Model, theme: &Theme, tick: Instant) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let spin = spinner_char(tick);

    if !model.model_path.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                "q27-tui · {} · theme={} · think={} · md={}",
                short_path(&model.model_path),
                model.theme.name(),
                if model.show_thinking { "on" } else { "off" },
                if model.markdown { "on" } else { "off" },
            ),
            Style::default()
                .fg(theme.header)
                .add_modifier(Modifier::BOLD),
        )));
    }

    for block in &model.scrollback {
        match block {
            Block::User(t) => {
                lines.push(Line::from(Span::styled("you", theme.role("you"))));
                for l in t.lines() {
                    lines.push(Line::from(format!("  {l}")));
                }
                lines.push(Line::from(""));
            }
            Block::Assistant { thinking, body } => {
                lines.push(Line::from(Span::styled(
                    "assistant",
                    theme.role("assistant"),
                )));
                push_thinking_lines(&mut lines, model, theme, thinking.as_deref(), false, spin);
                if model.markdown {
                    lines.extend(render_markdown(body, theme, "  "));
                } else {
                    for l in body.lines() {
                        lines.push(Line::from(format!("  {l}")));
                    }
                }
                lines.push(Line::from(""));
            }
            Block::Tool {
                kind,
                detail,
                body,
                exit,
                open,
            } => {
                let head = if *open {
                    format!("{spin} ┌─ {kind} {detail}")
                } else {
                    format!(
                        "┌─ {kind} {}  exit={}",
                        detail,
                        exit.map(|e| e.to_string()).unwrap_or_else(|| "?".into())
                    )
                };
                lines.push(Line::from(Span::styled(
                    head,
                    Style::default().fg(theme.tool),
                )));
                let collapsed = collapse(body, 12);
                for l in collapsed.lines() {
                    lines.push(Line::from(format!("│ {l}")));
                }
                lines.push(Line::from(Span::styled(
                    "└─",
                    Style::default().fg(theme.tool),
                )));
                lines.push(Line::from(""));
            }
            Block::Notice { severity, text } => {
                let mut first = true;
                for l in text.lines() {
                    if first {
                        lines.push(Line::from(Span::styled(
                            format!("[{severity}] {l}"),
                            theme.notice(severity),
                        )));
                        first = false;
                    } else {
                        lines.push(Line::from(Span::styled(
                            format!("  {l}"),
                            theme.notice(severity),
                        )));
                    }
                }
            }
            Block::System(t) => {
                lines.push(Line::from(Span::styled(
                    t.clone(),
                    Style::default().fg(theme.dim),
                )));
            }
        }
    }

    // Live assistant stream (with streaming thinking).
    if !model.assistant_buf.is_empty() {
        lines.push(Line::from(Span::styled(
            "assistant",
            theme.role("assistant"),
        )));
        let live = split_thinking_live(&model.assistant_buf);
        push_thinking_lines(
            &mut lines,
            model,
            theme,
            live.thinking.as_deref(),
            live.open,
            spin,
        );
        // Live body: plain text while streaming (markdown after finalize).
        if !live.body.is_empty() {
            for l in live.body.lines() {
                lines.push(Line::from(format!("  {l}")));
            }
            // Cursor only when not inside an open thinking span, or after body
            // started (model left think and is emitting answer).
            if !live.open {
                lines.push(Line::from(Span::styled(
                    "  ▌",
                    Style::default().fg(theme.assistant),
                )));
            }
        } else if live.open {
            // Thinking in progress, no body yet — cursor inside think block
            // is already shown via streaming branch.
        } else {
            lines.push(Line::from(Span::styled(
                "  ▌",
                Style::default().fg(theme.assistant),
            )));
        }
    } else if matches!(
        model.phase,
        Phase::Prefill | Phase::Generating | Phase::Compacting | Phase::Starting
    ) && model.assistant_buf.is_empty()
    {
        // No tokens yet — still show a live status card so the bottom is the
        // "current" activity the user can scroll to.
        let label = match model.phase {
            Phase::Prefill => format!("{spin} prefill…"),
            Phase::Compacting => format!("{spin} compacting…"),
            Phase::Starting => format!("{spin} starting…"),
            _ => format!("{spin} generating…"),
        };
        lines.push(Line::from(Span::styled(
            label,
            Style::default().fg(theme.footer_busy),
        )));
    }

    lines
}

fn push_thinking_lines(
    lines: &mut Vec<Line<'static>>,
    model: &Model,
    theme: &Theme,
    thinking: Option<&str>,
    open: bool,
    spin: char,
) {
    let Some(th) = thinking else {
        return;
    };
    // Even empty-but-open thinking should show a streaming header.
    if th.is_empty() && !open {
        return;
    }
    // Live stream always shows the full think body (no mid-sentence ellipsis).
    // Completed turns honor show_thinking; [t] collapses them to a one-liner.
    let expand = model.show_thinking || open;
    if !expand {
        let n = th.lines().count().max(1);
        lines.push(Line::from(Span::styled(
            format!("  ▸ thinking collapsed ({n} lines)  [t]"),
            Style::default().fg(theme.thinking),
        )));
        return;
    }

    let header = if open {
        format!("  {spin} thinking…")
    } else {
        "  thinking".to_string()
    };
    lines.push(Line::from(Span::styled(
        header,
        theme.role("thinking"),
    )));
    for l in th.lines() {
        lines.push(Line::from(Span::styled(
            format!("  │ {l}"),
            Style::default().fg(theme.thinking),
        )));
    }
    if open {
        lines.push(Line::from(Span::styled(
            "  │ ▌",
            Style::default().fg(theme.thinking),
        )));
    }
}

fn draw_input(frame: &mut Frame, area: Rect, model: &Model, input: &str, theme: &Theme) {
    let busy = !model.input_enabled
        && !model.has_queue_feature()
        && model.phase != Phase::Idle
        && model.phase != Phase::Stopped;
    let border = if busy {
        theme.input_border_busy
    } else {
        theme.input_border
    };
    let title = if model.phase == Phase::Stopped {
        " stopped "
    } else if model.has_queue_feature() && model.phase != Phase::Idle {
        " prompt (queue on Enter) "
    } else if model.input_enabled {
        " prompt "
    } else {
        " busy… "
    };
    let block = WBlock::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let shown = if input.is_empty() {
        Span::styled("…", Style::default().fg(theme.dim))
    } else {
        Span::raw(input.to_string())
    };
    frame.render_widget(Paragraph::new(Line::from(shown)), inner);
}

fn draw_footer(frame: &mut Frame, area: Rect, model: &Model, theme: &Theme, tick: Instant) {
    let spin = spinner_char(tick);
    let phase = match model.phase {
        Phase::Starting => "starting",
        Phase::Idle => "idle",
        Phase::Prefill => "prefill",
        Phase::Generating => "gen",
        Phase::Tool => "tool",
        Phase::Compacting => "compact",
        Phase::Error => "error",
        Phase::Stopped => "stopped",
    };
    let ctx = if model.ctx_size > 0 {
        format!("{}/{}", model.ctx_used, model.ctx_size)
    } else {
        format!("{}", model.ctx_used)
    };
    let ctx_bar = if model.ctx_size > 0 {
        format!(
            " {}",
            bar(model.ctx_used as f64 / model.ctx_size as f64, 8)
        )
    } else {
        String::new()
    };
    let q = if model.queue_len > 0 {
        format!(" · queue {}", model.queue_len)
    } else {
        String::new()
    };

    let progress = match model.phase {
        Phase::Prefill if model.prefill_total > 0 => {
            let frac = model.prefill_done as f64 / model.prefill_total as f64;
            format!(
                " {spin} {} {}/{}",
                bar(frac, 12),
                model.prefill_done,
                model.prefill_total
            )
        }
        Phase::Prefill => format!(" {spin}"),
        Phase::Generating => {
            format!(" {spin} {} tok", model.gen_tokens)
        }
        Phase::Tool | Phase::Compacting | Phase::Starting => {
            format!(" {spin}")
        }
        _ => String::new(),
    };

    let keys = "  ↑↓ scroll  Esc/^C cancel  empty: t/T/m  ^Q quit";
    let text = format!(
        " ctx {ctx}{ctx_bar} · {phase}{progress} · {}{q}{keys}",
        model.status_line
    );
    let color = if matches!(
        model.phase,
        Phase::Generating | Phase::Prefill | Phase::Tool | Phase::Compacting | Phase::Starting
    ) {
        theme.footer_busy
    } else {
        theme.footer
    };
    frame.render_widget(
        Paragraph::new(Span::styled(text, Style::default().fg(color))),
        area,
    );
}

fn spinner_char(session_start: Instant) -> char {
    // ~12.5 fps spin (80ms/frame), driven by session wall time.
    let ms = session_start.elapsed().as_millis();
    let idx = (ms / 80) as usize % SPINNER.len();
    SPINNER[idx]
}

/// ASCII progress bar, `frac` in 0..=1.
fn bar(frac: f64, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let f = frac.clamp(0.0, 1.0);
    let filled = ((f * width as f64).round() as usize).min(width);
    let empty = width - filled;
    format!(
        "[{}{}]",
        "█".repeat(filled),
        "░".repeat(empty)
    )
}

fn short_path(p: &str) -> String {
    let bytes = p.as_bytes();
    if bytes.len() <= 48 {
        return p.to_string();
    }
    let start = p
        .char_indices()
        .rev()
        .nth(40)
        .map(|(i, _)| i)
        .unwrap_or(0);
    format!("…{}", &p[start..])
}

fn collapse(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= max_lines {
        return s.to_string();
    }
    let head = max_lines.saturating_sub(1);
    let mut out = lines[..head].join("\n");
    out.push_str(&format!("\n… ({} more lines)", lines.len() - head));
    out
}
