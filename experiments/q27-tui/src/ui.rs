//! Ratatui layout: scrollback / editor / sticky footer.

use crate::app::{Block, Model, Phase};
use crate::md::{render_markdown, split_thinking_live};
use crate::theme::{Theme, ThemeId};
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
    cache: &mut ScrollbackCache,
    input: &str,
    scroll: u32,
    tick: Instant,
) -> u32 {
    let theme = model.theme.palette();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let max_scroll = draw_scrollback(frame, chunks[0], model, cache, scroll, &theme, tick);
    draw_input(frame, chunks[1], model, input, &theme);
    draw_footer(frame, chunks[2], model, &theme, tick);
    max_scroll
}

/// Per-block render cache. Finished blocks keep their styled lines and wrapped
/// height, so a frame wraps only the rows on screen plus whatever changed.
/// Before this, every frame rebuilt and word-wrapped the whole transcript
/// twice (height, then render): O(transcript) per frame, ~94 ms at 400 turns.
#[derive(Default)]
pub struct ScrollbackCache {
    /// Everything besides the block itself that changes its rendering.
    key: Option<(ThemeId, bool, bool, u16)>,
    entries: Vec<CachedBlock>,
    /// Lines word-wrapped by the last frame (cost probe for the tests).
    pub wrapped_last_frame: usize,
}

struct CachedBlock {
    block: Block,
    lines: Vec<Line<'static>>,
    /// Wrapped rows per line: lets a frame slice the visible lines out of a
    /// block of any size instead of re-wrapping the whole block.
    heights: Vec<usize>,
    height: usize,
}

fn draw_scrollback(
    frame: &mut Frame,
    area: Rect,
    model: &Model,
    cache: &mut ScrollbackCache,
    scroll: u32,
    theme: &Theme,
    tick: Instant,
) -> u32 {
    let width = area.width.max(1);
    let spin = spinner_char(tick);
    let key = (model.theme, model.show_thinking, model.markdown, width);
    if cache.key != Some(key) {
        cache.key = Some(key);
        cache.entries.clear();
    }
    cache.wrapped_last_frame = 0;

    // Segments in display order: header, blocks, live tail. Blocks are
    // re-rendered only when they changed or are an open tool card (its
    // spinner animates).
    let header = header_lines(model, theme);
    let header_h = line_heights(&header, width);
    let tail = live_lines(model, theme, spin);
    let tail_h = line_heights(&tail, width);
    let mut wrapped = header.len() + tail.len();

    cache.entries.truncate(model.scrollback.len());
    for (i, block) in model.scrollback.iter().enumerate() {
        let live_card = matches!(block, Block::Tool { open: true, .. });
        if !live_card && cache.entries.get(i).is_some_and(|e| e.block == *block) {
            continue;
        }
        let lines = block_lines(block, model, theme, spin);
        wrapped += lines.len();
        let heights = line_heights(&lines, width);
        let entry = CachedBlock { block: block.clone(), height: heights.iter().sum(), heights, lines };
        if i < cache.entries.len() {
            cache.entries[i] = entry;
        } else {
            cache.entries.push(entry);
        }
    }

    let mut segs: Vec<(&[Line<'static>], &[usize], usize)> = Vec::with_capacity(cache.entries.len() + 2);
    segs.push((&header, &header_h, header_h.iter().sum()));
    segs.extend(cache.entries.iter().map(|e| (e.lines.as_slice(), e.heights.as_slice(), e.height)));
    segs.push((&tail, &tail_h, tail_h.iter().sum()));

    let total: usize = segs.iter().map(|s| s.2).sum();
    let max_scroll = total.saturating_sub(area.height as usize).min(u32::MAX as usize) as u32;
    let top = scroll.min(max_scroll) as usize;
    let bottom = top + area.height as usize;

    // Gather only the lines overlapping rows [top, bottom). `offset` is the
    // rows of the first gathered line above the viewport (< one line's height).
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut offset = 0usize;
    let mut row = 0usize;
    'segments: for (seg_lines, seg_heights, seg_total) in &segs {
        if row + seg_total <= top {
            row += seg_total;
            continue;
        }
        for (line, h) in seg_lines.iter().zip(seg_heights.iter()) {
            if row >= bottom {
                break 'segments;
            }
            if row + h > top {
                if lines.is_empty() {
                    offset = top - row;
                }
                lines.push(line.clone());
            }
            row += h;
        }
    }
    wrapped += lines.len();
    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .scroll((offset.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(para, area);
    cache.wrapped_last_frame = wrapped;
    max_scroll
}

/// Wrapped rows of each line at `width` (ratatui wraps every Line
/// independently, so these sum to the whole-transcript height).
fn line_heights(lines: &[Line<'static>], width: u16) -> Vec<usize> {
    lines
        .iter()
        .map(|l| Paragraph::new(l.clone()).wrap(Wrap { trim: false }).line_count(width))
        .collect()
}

#[cfg(test)]
fn wrapped_height(lines: &[Line<'static>], width: u16) -> usize {
    Paragraph::new(lines.to_vec())
        .wrap(Wrap { trim: false })
        .line_count(width)
}

#[cfg(test)]
mod scroll_tests {
    use super::*;

    #[test]
    fn word_wrap_rows_are_counted() {
        // 20 six-letter words in a 10-column viewport: word wrap puts one
        // word per row (20 rows); the width estimate said ceil(139/10)=14.
        let text = vec!["abcdef"; 20].join(" ");
        assert_eq!(wrapped_height(&[Line::from(text)], 10), 20);
    }
}

fn header_lines(model: &Model, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

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
    lines
}

fn block_lines(block: &Block, model: &Model, theme: &Theme, spin: char) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
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
    lines
}

fn live_lines(model: &Model, theme: &Theme, spin: char) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
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
    // Completed turns honor show_thinking; Ctrl-T collapses them to a one-liner.
    let expand = model.show_thinking || open;
    if !expand {
        let n = th.lines().count().max(1);
        lines.push(Line::from(Span::styled(
            format!("  ▸ thinking collapsed ({n} lines)  [Ctrl-T]"),
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

#[cfg(test)]
mod draw_cost {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui::Terminal;

    /// A long agent session: `turns` rounds of prompt, thinking + markdown
    /// answer, and a tool card, with an answer streaming.
    pub(crate) fn long_session(turns: usize) -> Model {
        let mut m = Model::default();
        m.model_path = "models/bonsai2/bonsai2-27b-t2-slim.q27".into();
        m.show_thinking = true;
        let para = "The quick brown fox **jumps** over the `lazy` dog, then reads src/main.rs and edits it. ";
        for i in 0..turns {
            m.scrollback.push(Block::User(format!("turn {i}: fix the parser bug in module {i}")));
            let mut body = String::from("## Plan\n\n");
            for j in 0..6 {
                body.push_str(&format!("- step {j}: {}\n", para));
            }
            body.push_str("\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\n");
            body.push_str(&para.repeat(8));
            m.scrollback.push(Block::Assistant {
                thinking: Some(para.repeat(12)),
                body,
            });
            m.scrollback.push(Block::Tool {
                kind: "shell".into(),
                detail: "cargo test".into(),
                body: (0..40).map(|k| format!("test case_{k} ... ok\n")).collect(),
                exit: Some(0),
                open: false,
            });
        }
        m.assistant_buf = format!("<think>{}</think>{}", para.repeat(4), para.repeat(6));
        m.phase = Phase::Generating;
        m
    }

    fn frame(term: &mut Terminal<TestBackend>, m: &Model, cache: &mut ScrollbackCache, scroll: u32, tick: Instant) -> (u32, Buffer) {
        let mut max = 0;
        term.draw(|f| max = draw(f, m, cache, "", scroll, tick)).unwrap();
        (max, term.backend().buffer().clone())
    }

    /// The pre-cache renderer: whole transcript in one wrapped Paragraph.
    fn reference(m: &Model, w: u16, h: u16, scroll: u32, tick: Instant) -> (u32, Buffer) {
        let theme = m.theme.palette();
        let spin = spinner_char(tick);
        let mut lines = header_lines(m, &theme);
        for b in &m.scrollback {
            lines.extend(block_lines(b, m, &theme, spin));
        }
        lines.extend(live_lines(m, &theme, spin));
        let para = Paragraph::new(lines).wrap(Wrap { trim: false });
        let rows = h - 4; // input box (3) + footer (1)
        let max = para.line_count(w).saturating_sub(rows as usize) as u32;
        let para = para.scroll((scroll.min(max) as u16, 0));
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| {
            let area = Rect::new(0, 0, w, rows);
            f.render_widget(para, area);
        })
        .unwrap();
        (max, term.backend().buffer().clone())
    }

    fn scrollback_rows(buf: &Buffer, w: u16, rows: u16) -> Vec<String> {
        (0..rows)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>())
            .collect()
    }

    fn assert_matches_reference(m: &Model, cache: &mut ScrollbackCache, w: u16, h: u16, scroll: u32) {
        let tick = Instant::now();
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let (max, got) = frame(&mut term, m, cache, scroll, tick);
        let (want_max, want) = reference(m, w, h, scroll, tick);
        assert_eq!(max, want_max, "max scroll (w={w} scroll={scroll})");
        assert_eq!(
            scrollback_rows(&got, w, h - 4),
            scrollback_rows(&want, w, h - 4),
            "rows (w={w} scroll={scroll})"
        );
    }

    #[test]
    fn windowed_render_matches_whole_transcript_render() {
        let mut m = long_session(12);
        let mut cache = ScrollbackCache::default();
        for (w, h) in [(120u16, 40u16), (37, 17), (200, 60)] {
            for scroll in [0u32, 1, 7, 50, 333, 1000, u32::MAX] {
                assert_matches_reference(&m, &mut cache, w, h, scroll);
            }
        }
        // Mutations must invalidate: an open card streaming output, then
        // closing; a toggle; a theme switch; a cleared stream.
        m.scrollback.push(Block::Tool {
            kind: "shell".into(),
            detail: "make".into(),
            body: "line 1\n".into(),
            exit: None,
            open: true,
        });
        assert_matches_reference(&m, &mut cache, 120, 40, u32::MAX);
        if let Some(Block::Tool { body, open, exit, .. }) = m.scrollback.last_mut() {
            body.push_str("line 2\nline 3\n");
            *open = false;
            *exit = Some(2);
        }
        assert_matches_reference(&m, &mut cache, 120, 40, u32::MAX);
        if let Some(Block::User(t)) = m.scrollback.get_mut(3) {
            t.push_str(" (edited)");
        }
        assert_matches_reference(&m, &mut cache, 120, 40, 0);
        m.show_thinking = false;
        assert_matches_reference(&m, &mut cache, 120, 40, 200);
        m.markdown = false;
        assert_matches_reference(&m, &mut cache, 120, 40, 200);
        m.theme = m.theme.next();
        assert_matches_reference(&m, &mut cache, 120, 40, 200);
        m.scrollback.truncate(5);
        m.assistant_buf.clear();
        assert_matches_reference(&m, &mut cache, 120, 40, u32::MAX);
    }

    /// Ratchet: a steady-state frame (nothing changed but the stream) wraps
    /// the same number of lines at 10 turns and at 400 — frame cost no longer
    /// grows with the transcript. Before the cache, 400 turns wrapped ~30k.
    #[test]
    fn steady_frame_cost_is_independent_of_session_length() {
        let tick = Instant::now();
        let mut cost = Vec::new();
        for turns in [10, 400] {
            let m = long_session(turns);
            let mut cache = ScrollbackCache::default();
            let mut term = Terminal::new(TestBackend::new(120, 40)).unwrap();
            frame(&mut term, &m, &mut cache, u32::MAX, tick);
            frame(&mut term, &m, &mut cache, u32::MAX, tick);
            cost.push(cache.wrapped_last_frame);
        }
        assert_eq!(cost[0], cost[1], "steady-state wrapped lines: {cost:?}");
        assert!(cost[1] <= 200, "steady-state frame wraps {} lines", cost[1]);
    }

    /// One block taller than u16::MAX rows: scrolling anywhere in it (and to
    /// the bottom) shows the right lines, and a frame wraps only what is on
    /// screen, not the block (Sol 6.1 review: offset clamped at 65,535 and
    /// the whole visible block re-wrapped each frame).
    #[test]
    fn giant_block_scrolls_and_stays_cheap() {
        let n = 70_000;
        let mut m = Model::default();
        m.phase = Phase::Idle;
        m.scrollback.push(Block::User(
            (0..n).map(|k| format!("line {k}")).collect::<Vec<_>>().join("\n"),
        ));
        let (w, h) = (80u16, 24u16);
        let rows = h - 4;
        let mut cache = ScrollbackCache::default();
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let tick = Instant::now();
        // Rows: "you", "  line 0" .. "  line 69999", "".
        let total = n + 2;
        let (max, _) = frame(&mut term, &m, &mut cache, u32::MAX, tick);
        assert_eq!(max as usize, total - rows as usize);
        for top in [0usize, 1, 65_534, 65_535, 65_536, 69_000, max as usize] {
            let (_, buf) = frame(&mut term, &m, &mut cache, top as u32, tick);
            let got = scrollback_rows(&buf, w, rows);
            let expect = |row: usize| match row {
                0 => "you".to_string(),
                r if r <= n => format!("  line {}", r - 1),
                _ => String::new(),
            };
            for (i, text) in got.iter().enumerate() {
                assert_eq!(text.trim_end(), expect(top + i), "scroll {top}, row {i}");
            }
            assert!(
                cache.wrapped_last_frame <= rows as usize + 2,
                "scroll {top}: wrapped {} lines",
                cache.wrapped_last_frame
            );
        }
    }

    /// Wall-clock probe: `cargo test --release draw_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn print_draw_time_by_session_length() {
        for turns in [10, 100, 400] {
            let m = long_session(turns);
            let mut cache = ScrollbackCache::default();
            let mut term = Terminal::new(TestBackend::new(120, 40)).unwrap();
            let tick = Instant::now();
            frame(&mut term, &m, &mut cache, u32::MAX, tick);
            let frames = 40;
            let start = Instant::now();
            for _ in 0..frames {
                frame(&mut term, &m, &mut cache, u32::MAX, tick);
            }
            eprintln!(
                "turns={turns:4} per-frame {:8.3} ms, wrapped lines {}",
                start.elapsed().as_secs_f64() * 1e3 / frames as f64,
                cache.wrapped_last_frame
            );
        }
    }
}
