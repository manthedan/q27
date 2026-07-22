//! q27-tui — Rust Ratatui frontend for q27-agent (Frontend Protocol v1).
//!
//! ```text
//! q27-tui -- MODEL.q27 MODEL.tok [--auto-tools] [--session FILE] …
//! ```
//!
//! Spawns `q27-agent … --frontend-proto 1` and owns its stdio.

mod app;
mod backend;
mod md;
mod proto;
mod theme;
mod ui;

use app::Model;
use backend::{Backend, BackendEvent};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::env;
use std::io::{self, stdout, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn main() {
    // Always try to restore the terminal on panic.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal();
        default_hook(info);
    }));

    if let Err(e) = run() {
        let _ = restore_terminal();
        eprintln!("q27-tui: {e}");
        std::process::exit(1);
    }
}

fn restore_terminal() -> io::Result<()> {
    let _ = disable_raw_mode();
    let mut out = stdout();
    let _ = execute!(out, LeaveAlternateScreen);
    let _ = out.flush();
    Ok(())
}

/// RAII: raw mode + alternate screen; restored on drop.
struct TermGuard {
    active: bool,
}

impl TermGuard {
    fn enter() -> io::Result<(Self, Terminal<CrosstermBackend<io::Stdout>>)> {
        enable_raw_mode()?;
        let mut out = stdout();
        execute!(out, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(out))?;
        Ok((Self { active: true }, terminal))
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = restore_terminal();
            self.active = false;
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    // Optional `--` separator before agent args.
    if args.first().map(|s| s.as_str()) == Some("--") {
        args.remove(0);
    }

    let agent_bin = resolve_agent_bin()?;
    if args.len() < 2 {
        print_usage();
        return Err("MODEL and TOKENIZER are required".into());
    }
    let model = PathBuf::from(&args[0]);
    let tokenizer = PathBuf::from(&args[1]);
    let extra: Vec<String> = args[2..].to_vec();

    // Fail fast with a clear message before touching the terminal.
    if !agent_bin.exists() {
        return Err(format!("agent binary not found: {}", agent_bin.display()).into());
    }
    if !model.exists() {
        return Err(format!("model not found: {}", model.display()).into());
    }
    if !tokenizer.exists() {
        return Err(format!("tokenizer not found: {}", tokenizer.display()).into());
    }
    if !std::io::IsTerminal::is_terminal(&io::stdin())
        || !std::io::IsTerminal::is_terminal(&io::stdout())
    {
        return Err(
            "stdout/stdin must be a TTY (run in a real terminal, not a pipe)".into(),
        );
    }

    let log_path = std::env::temp_dir().join(format!(
        "q27-tui-stderr-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    eprintln!(
        "q27-tui: starting agent {} …\n  model: {}\n  tok:   {}\n  log:   {}",
        agent_bin.display(),
        model.display(),
        tokenizer.display(),
        log_path.display()
    );

    let mut backend = Backend::spawn_fp1(
        &agent_bin,
        &model,
        &tokenizer,
        &extra,
        &log_path,
    )
    .map_err(|e| {
        format!(
            "failed to spawn agent: {e}\n  bin: {}\n  see also: {}",
            agent_bin.display(),
            log_path.display()
        )
    })?;

    // Wait for hello *before* entering the alternate screen so load failures
    // surface as normal terminal text, not a flash-and-die TUI.
    let mut model_state = Model::default();
    model_state.status_line = format!("waiting for hello · log {}", log_path.display());
    wait_for_hello(&mut backend, &mut model_state, &log_path, Duration::from_secs(120))?;

    let (_guard, mut terminal) = TermGuard::enter()?;

    let mut input = String::new();
    // scroll = lines from top of content. 0 = oldest; max = current LLM output.
    let mut scroll: u16 = 0;
    let mut max_scroll: u16 = 0;
    // Stick to bottom (current stream) until the user scrolls up into history.
    let mut follow_bottom = true;
    let session_start = Instant::now();
    let mut running = true;
    let mut fatal: Option<String> = None;

    while running {
        if follow_bottom {
            scroll = max_scroll;
        } else {
            scroll = scroll.min(max_scroll);
        }

        max_scroll = 0;
        terminal.draw(|f| {
            max_scroll = ui::draw(f, &model_state, &input, scroll, session_start);
        })?;
        if follow_bottom {
            scroll = max_scroll;
        } else {
            scroll = scroll.min(max_scroll);
        }

        // Drain backend events.
        let mut saw_content = false;
        loop {
            match backend.try_recv() {
                Ok(BackendEvent::Server(ev)) => {
                    // Content-bearing events should keep a following viewport
                    // glued to the live stream.
                    if matches!(
                        ev.type_name.as_str(),
                        "text_delta"
                            | "prefill_progress"
                            | "tool_start"
                            | "tool_output"
                            | "tool_done"
                            | "turn_done"
                            | "idle"
                            | "notice"
                            | "rejected"
                            | "session_done"
                            | "state"
                    ) {
                        saw_content = true;
                    }
                    model_state.apply(&ev);
                }
                Ok(BackendEvent::BadLine(l)) => {
                    model_state.scrollback.push(app::Block::Notice {
                        severity: "warning".into(),
                        text: format!("bad line: {l}"),
                    });
                    saw_content = true;
                }
                Ok(BackendEvent::Stderr(_)) => {}
                Ok(BackendEvent::Exited(code)) => {
                    if model_state.bye_reason.is_none() {
                        let tail = read_log_tail(&log_path, 40);
                        let msg = format!(
                            "agent exited unexpectedly (code={code:?})\n{tail}\nfull log: {}",
                            log_path.display()
                        );
                        model_state.scrollback.push(app::Block::Notice {
                            severity: "error".into(),
                            text: msg.clone(),
                        });
                        model_state.phase = app::Phase::Stopped;
                        model_state.input_enabled = false;
                        fatal = Some(msg);
                        running = false;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if model_state.bye_reason.is_none() {
                        let tail = read_log_tail(&log_path, 40);
                        fatal = Some(format!(
                            "agent event stream closed\n{tail}\nfull log: {}",
                            log_path.display()
                        ));
                    }
                    running = false;
                    break;
                }
            }
        }
        if saw_content && follow_bottom {
            // Next draw will recompute max_scroll and snap.
        }

        if model_state.phase == app::Phase::Stopped && model_state.bye_reason.is_some() {
            terminal.draw(|f| {
                let _ = ui::draw(f, &model_state, &input, scroll, session_start);
            })?;
            std::thread::sleep(Duration::from_millis(250));
            break;
        }

        if !running {
            break;
        }

        // Short poll so spinners keep animating while idle/busy.
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key_binding(&key) {
                    KeyBind::Cancel => {
                        // Busy → interrupt generation/tool. Idle with draft →
                        // clear the line. Idle empty → no-op (still report).
                        let busy = !matches!(
                            model_state.phase,
                            app::Phase::Idle
                                | app::Phase::Stopped
                                | app::Phase::Starting
                        );
                        if busy {
                            model_state.status_line = "cancelling…".into();
                            if let Err(e) = backend.cancel() {
                                model_state.scrollback.push(app::Block::Notice {
                                    severity: "error".into(),
                                    text: format!("cancel failed: {e}"),
                                });
                            }
                            // Clear any half-typed queue prompt so the next
                            // keystroke isn't mixed with the interrupt.
                            input.clear();
                        } else if !input.is_empty() {
                            input.clear();
                            model_state.status_line = "cleared".into();
                        } else {
                            model_state.status_line = "nothing to cancel".into();
                        }
                    }
                    KeyBind::Quit => {
                        let _ = backend.quit();
                        // Give the agent a moment to emit bye; then leave.
                        let until = Instant::now() + Duration::from_millis(800);
                        while Instant::now() < until {
                            match backend.try_recv() {
                                Ok(BackendEvent::Server(ev)) => model_state.apply(&ev),
                                Ok(BackendEvent::Exited(_)) => break,
                                Err(std::sync::mpsc::TryRecvError::Empty) => {
                                    std::thread::sleep(Duration::from_millis(20));
                                }
                                _ => break,
                            }
                            if model_state.bye_reason.is_some() {
                                break;
                            }
                        }
                        running = false;
                    }
                    KeyBind::Other => match (key.code, key.modifiers) {
                        (KeyCode::Enter, _) => {
                            // Accept prompts while busy when hello.features has queue.
                            let can_type = model_state.input_enabled
                                || (model_state.has_queue_feature()
                                    && model_state.phase != app::Phase::Stopped
                                    && model_state.phase != app::Phase::Starting);
                            if can_type && !input.trim().is_empty() {
                                let text = input.clone();
                                input.clear();
                                if let Err(e) =
                                    dispatch_input(&mut backend, &mut model_state, &text)
                                {
                                    model_state.scrollback.push(app::Block::Notice {
                                        severity: "error".into(),
                                        text: format!("send failed: {e}"),
                                    });
                                    model_state.input_enabled = true;
                                } else {
                                    // New turn: jump to live output.
                                    follow_bottom = true;
                                }
                            }
                        }
                        (KeyCode::Backspace, _) => {
                            input.pop();
                        }
                        // Chat-native: Up/PgUp = older history; Down/PgDn = toward
                        // current LLM output. Max scroll = bottom of live stream.
                        (KeyCode::Up, _) => {
                            if scroll > 0 {
                                scroll = scroll.saturating_sub(1);
                                follow_bottom = false;
                            }
                        }
                        (KeyCode::Down, _) => {
                            if scroll < max_scroll {
                                scroll = scroll.saturating_add(1);
                                follow_bottom = scroll >= max_scroll;
                            } else {
                                follow_bottom = true;
                            }
                        }
                        (KeyCode::PageUp, _) => {
                            if scroll > 0 {
                                scroll = scroll.saturating_sub(10);
                                follow_bottom = false;
                            }
                        }
                        (KeyCode::PageDown, _) => {
                            let next = scroll.saturating_add(10).min(max_scroll);
                            scroll = next;
                            follow_bottom = scroll >= max_scroll;
                        }
                        (KeyCode::Home, _) => {
                            scroll = 0;
                            follow_bottom = false;
                        }
                        (KeyCode::End, _) => {
                            scroll = max_scroll;
                            follow_bottom = true;
                        }
                        // Presentation toggles only when the input line is empty
                        // so typing "the" / "make" still works.
                        (KeyCode::Char('t'), KeyModifiers::NONE) if input.is_empty() => {
                            model_state.toggle_thinking();
                        }
                        (KeyCode::Char('T'), KeyModifiers::SHIFT) if input.is_empty() => {
                            model_state.cycle_theme();
                        }
                        (KeyCode::Char('m'), KeyModifiers::NONE) if input.is_empty() => {
                            model_state.toggle_markdown();
                        }
                        (KeyCode::Char(c), KeyModifiers::NONE)
                        | (KeyCode::Char(c), KeyModifiers::SHIFT) => {
                            let can_type = model_state.input_enabled
                                || (model_state.has_queue_feature()
                                    && model_state.phase != app::Phase::Stopped
                                    && model_state.phase != app::Phase::Starting);
                            if can_type {
                                input.push(c);
                            }
                        }
                        _ => {}
                    },
                }
            }
        }
    }

    // TermGuard drops here and restores the terminal.
    drop(terminal);
    backend.kill();

    if let Some(msg) = fatal {
        return Err(msg.into());
    }
    Ok(())
}

/// Map Enter input to an FP1 ClientMessage. Slash forms become structured ops
/// (no raw_line op — FP1 §10). Plain text is a prompt.
fn dispatch_input(
    backend: &mut Backend,
    model: &mut Model,
    text: &str,
) -> Result<(), backend::BackendError> {
    let trimmed = text.trim();
    let (cmd, args) = split_slash(trimmed);
    match cmd {
        Some("help") | Some("hotkeys") => {
            // Local chrome help + authoritative server op list.
            model.scrollback.push(app::Block::Notice {
                severity: "info".into(),
                text: local_help().into(),
            });
            model.status_line = "help…".into();
            model.input_enabled = false;
            backend.help().map(|_| ())
        }
        Some("save") => {
            model.status_line = "saving…".into();
            model.input_enabled = false;
            backend.save().map(|_| ())
        }
        Some("compact") => {
            model.status_line = "compacting…".into();
            model.input_enabled = false;
            backend.compact().map(|_| ())
        }
        Some("session") => {
            model.status_line = "session…".into();
            model.input_enabled = false;
            backend.session().map(|_| ())
        }
        Some("new") => {
            model.status_line = "new transcript…".into();
            model.input_enabled = false;
            backend.new_session().map(|_| ())
        }
        Some("read") => {
            let path = args.trim();
            if path.is_empty() {
                model.scrollback.push(app::Block::Notice {
                    severity: "error".into(),
                    text: "usage: /read PATH".into(),
                });
                return Ok(());
            }
            model.status_line = format!("tool read {path}");
            model.input_enabled = false;
            backend.tool_read(path).map(|_| ())
        }
        Some("search") => {
            let mut parts = args.splitn(2, char::is_whitespace);
            let path = parts.next().unwrap_or("").trim();
            let needle = parts.next().unwrap_or("").trim();
            if path.is_empty() || needle.is_empty() {
                model.scrollback.push(app::Block::Notice {
                    severity: "error".into(),
                    text: "usage: /search PATH NEEDLE".into(),
                });
                return Ok(());
            }
            model.status_line = format!("tool search {path}");
            model.input_enabled = false;
            backend.tool_search(path, needle).map(|_| ())
        }
        Some("shell") => {
            let command = args.trim();
            if command.is_empty() {
                model.scrollback.push(app::Block::Notice {
                    severity: "error".into(),
                    text: "usage: /shell COMMAND".into(),
                });
                return Ok(());
            }
            model.status_line = "tool shell…".into();
            model.input_enabled = false;
            backend.tool_shell(command).map(|_| ())
        }
        Some("queue_clear") | Some("clear-queue") => {
            if !model.has_queue_feature() {
                model.scrollback.push(app::Block::Notice {
                    severity: "error".into(),
                    text: "queue not advertised in hello.features".into(),
                });
                return Ok(());
            }
            model.status_line = "clearing queue…".into();
            backend.queue_clear().map(|_| ())
        }
        Some("cancel") | Some("interrupt") | Some("stop") => {
            // Guard like the Esc/^C key path: an idle agent treats SIGINT as
            // the global interrupt and exits, so idle /cancel must be a
            // no-op (codex P1).
            let busy = !matches!(
                model.phase,
                app::Phase::Idle | app::Phase::Stopped | app::Phase::Starting
            );
            if busy {
                model.status_line = "cancelling…".into();
                backend.cancel().map(|_| ())
            } else {
                model.status_line = "nothing to cancel".into();
                Ok(())
            }
        }
        Some("quit") | Some("exit") | Some("q") => backend.quit(),
        Some(_) => {
            model.scrollback.push(app::Block::Notice {
                severity: "error".into(),
                text: format!("unknown command (try /help): {trimmed}"),
            });
            Ok(())
        }
        None => {
            model.push_user(text);
            backend.prompt(text).map(|_| ())
        }
    }
}

fn local_help() -> &'static str {
    "keys: Enter send · Esc/^C cancel · ^Q quit\n\
     ↑/PgUp older · ↓/PgDn/End current output · Home top\n\
     (empty input) t thinking · T theme · m markdown\n\
     /cancel /help /save /compact /session /new /read /search /shell /queue_clear"
}

/// Control bindings that must tolerate extra modifier bits (some terminals
/// report Ctrl+C as CONTROL|SHIFT, etc.). Exact `== CONTROL` misses those.
enum KeyBind {
    Cancel,
    Quit,
    Other,
}

fn key_binding(key: &KeyEvent) -> KeyBind {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        // Esc always interrupts when the UI is live (no draft conflict).
        KeyCode::Esc => KeyBind::Cancel,
        KeyCode::Char('c') | KeyCode::Char('C') if ctrl => KeyBind::Cancel,
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Char('d') | KeyCode::Char('D')
            if ctrl =>
        {
            KeyBind::Quit
        }
        _ => KeyBind::Other,
    }
}

/// Returns (command_name, args) when line starts with `/` or `:`.
fn split_slash(line: &str) -> (Option<&str>, &str) {
    let rest = if let Some(r) = line.strip_prefix('/') {
        r
    } else if let Some(r) = line.strip_prefix(':') {
        r
    } else {
        return (None, line);
    };
    let rest = rest.trim_start();
    if rest.is_empty() {
        return (Some(""), "");
    }
    let mut it = rest.splitn(2, char::is_whitespace);
    let name = it.next().unwrap_or("");
    let args = it.next().unwrap_or("").trim_start();
    (Some(name), args)
}

/// Block until `hello` (and preferably the first `idle`), without alternate screen.
fn wait_for_hello(
    backend: &mut Backend,
    model: &mut Model,
    log_path: &Path,
    timeout: Duration,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut saw_hello = false;
    eprint!("q27-tui: loading model");
    let _ = io::stderr().flush();

    while start.elapsed() < timeout {
        match backend.recv_timeout(Duration::from_millis(200)) {
            Ok(BackendEvent::Server(ev)) => {
                let ty = ev.type_name.clone();
                model.apply(&ev);
                if ty == "hello" {
                    saw_hello = true;
                    eprintln!(" ok");
                    eprintln!(
                        "q27-tui: hello · features=[{}] · ctx={}",
                        model.features.join(", "),
                        model.ctx_size
                    );
                }
                if ty == "idle" && saw_hello {
                    return Ok(());
                }
                if ty == "bye" {
                    eprintln!();
                    let reason = model.bye_reason.as_deref().unwrap_or("?");
                    let tail = read_log_tail(log_path, 50);
                    return Err(format!(
                        "agent bye during startup (reason={reason})\n{tail}\nfull log: {}",
                        log_path.display()
                    )
                    .into());
                }
            }
            Ok(BackendEvent::BadLine(l)) => {
                eprintln!("\nq27-tui: non-JSON from agent: {l}");
            }
            Ok(BackendEvent::Exited(code)) => {
                eprintln!();
                let tail = read_log_tail(log_path, 50);
                return Err(format!(
                    "agent exited during startup (code={code:?}) before hello\n{tail}\nfull log: {}",
                    log_path.display()
                )
                .into());
            }
            Ok(BackendEvent::Stderr(_)) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                eprint!(".");
                let _ = io::stderr().flush();
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!();
                let tail = read_log_tail(log_path, 50);
                return Err(format!(
                    "agent stream closed during startup\n{tail}\nfull log: {}",
                    log_path.display()
                )
                .into());
            }
        }
    }

    eprintln!();
    let tail = read_log_tail(log_path, 50);
    Err(format!(
        "timed out waiting for hello after {timeout:?}\n{tail}\nfull log: {}",
        log_path.display()
    )
    .into())
}

fn read_log_tail(path: &Path, max_lines: usize) -> String {
    match std::fs::read_to_string(path) {
        Ok(s) if s.trim().is_empty() => "(agent stderr log empty)".into(),
        Ok(s) => {
            let lines: Vec<&str> = s.lines().collect();
            let start = lines.len().saturating_sub(max_lines);
            let body = lines[start..].join("\n");
            format!("--- agent stderr ---\n{body}\n--- end ---")
        }
        Err(e) => format!("(could not read {}: {e})", path.display()),
    }
}

fn resolve_agent_bin() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(p) = env::var("Q27_AGENT") {
        let pb = PathBuf::from(&p);
        if !pb.exists() {
            return Err(format!("Q27_AGENT set but not found: {p}").into());
        }
        return Ok(pb);
    }
    // Walk up from CWD looking for build/q27-agent.
    let mut dir = env::current_dir()?;
    for _ in 0..8 {
        let candidate = dir.join("build/q27-agent");
        if candidate.exists() {
            return Ok(candidate);
        }
        // Also accept target next to this crate when developed in-tree.
        let alt = dir.join("experiments/../build/q27-agent");
        if alt.exists() {
            return Ok(alt.canonicalize().unwrap_or(alt));
        }
        if !dir.pop() {
            break;
        }
    }
    // PATH
    if let Ok(path) = env::var("PATH") {
        for entry in path.split(':') {
            let p = PathBuf::from(entry).join("q27-agent");
            if p.exists() {
                return Ok(p);
            }
        }
    }
    Err("could not find q27-agent (set Q27_AGENT or build with `make build/q27-agent`)".into())
}

fn print_usage() {
    eprintln!(
        "\
usage: q27-tui [--] MODEL.q27 MODEL.tok [agent-options…]

Rust TUI client for q27-agent (Frontend Protocol v1).
Spawns: q27-agent MODEL TOKENIZER --frontend-proto 1 [agent-options…]

Environment:
  Q27_AGENT   path to q27-agent binary (default: walk up to build/q27-agent)

Keys:
  Enter       send prompt
  Esc / Ctrl-C  cancel active turn (or clear draft when idle)
  Ctrl-D/Q    quit
  ↑/PgUp      older history
  ↓/PgDn/End  toward current LLM output (cannot scroll past it)
  Home        top of transcript
  t / T / m   thinking / theme / markdown (empty input)
  /cancel     same as Esc while a turn is running

Notes:
  - Must run in a real TTY (not piped).
  - Model load failures print the agent stderr log and exit *before*
    entering the full-screen UI.
"
    );
}
