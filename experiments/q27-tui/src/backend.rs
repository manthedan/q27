//! Spawn / attach to `q27-agent` and shuttle NDJSON lines.

use crate::proto::{ClientMessage, ServerEvent};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("agent binary not found: {0}")]
    MissingAgent(PathBuf),
    #[error("failed to spawn agent")]
    Spawn,
    #[error("stdin closed")]
    StdinClosed,
}

pub enum BackendEvent {
    Server(ServerEvent),
    /// Non-JSON line or parse error (usually should not happen in FP1).
    BadLine(String),
    /// Child stderr line (diagnostics).
    Stderr(String),
    /// Child exited.
    Exited(Option<i32>),
}

pub struct Backend {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<BackendEvent>,
    pub stderr_log: PathBuf,
    req_counter: u64,
}

impl Backend {
    /// Spawn `q27-agent` with FP1. Extra args after model/tokenizer.
    pub fn spawn_fp1(
        agent_bin: impl AsRef<Path>,
        model: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
        extra_args: &[String],
        stderr_log: impl AsRef<Path>,
    ) -> Result<Self, BackendError> {
        let agent_bin = agent_bin.as_ref();
        if !agent_bin.exists() {
            return Err(BackendError::MissingAgent(agent_bin.to_path_buf()));
        }
        let stderr_log = stderr_log.as_ref().to_path_buf();
        if let Some(parent) = stderr_log.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let stderr_file = std::fs::File::create(&stderr_log)?;

        let mut cmd = Command::new(agent_bin);
        cmd.arg(model.as_ref())
            .arg(tokenizer.as_ref())
            .arg("--frontend-proto")
            .arg("1")
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr_file));

        // Metal loads q27_kernels.metal at runtime from cwd-relative paths
        // (or Q27_METAL_SOURCE). The TUI is often launched from
        // experiments/q27-tui/, so inject the repo shader when unset.
        if std::env::var_os("Q27_METAL_SOURCE").is_none() {
            if let Some(shader) = discover_metal_source(agent_bin) {
                cmd.env("Q27_METAL_SOURCE", shader);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            BackendError::Io(std::io::Error::new(
                e.kind(),
                format!("spawn {}: {e}", agent_bin.display()),
            ))
        })?;
        let stdin = child.stdin.take().ok_or(BackendError::Spawn)?;
        let stdout = child.stdout.take().ok_or(BackendError::Spawn)?;

        let (tx, rx) = mpsc::channel();
        spawn_stdout_reader(stdout, tx);

        Ok(Self {
            child,
            stdin,
            rx,
            stderr_log,
            req_counter: 0,
        })
    }

    /// Read-only attach: treat an existing process's stdout as event stream.
    /// (Used for v0 `--output-format jsonl --prompt` pipes via stdin not owned here.)
    pub fn try_recv(&self) -> Result<BackendEvent, mpsc::TryRecvError> {
        self.rx.try_recv()
    }

    pub fn recv_timeout(
        &self,
        timeout: std::time::Duration,
    ) -> Result<BackendEvent, mpsc::RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }

    pub fn send(&mut self, msg: &ClientMessage) -> Result<(), BackendError> {
        let line = msg
            .to_line()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.flush()?;
        Ok(())
    }

    pub fn next_req_id(&mut self) -> String {
        self.req_counter += 1;
        format!("ui-{}", self.req_counter)
    }

    pub fn prompt(&mut self, text: &str) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::prompt(&id, text))?;
        Ok(id)
    }

    pub fn cancel(&mut self) -> Result<(), BackendError> {
        let id = self.next_req_id();
        let send_result = self.send(&ClientMessage::cancel(id));
        // Dual-path: FP1 cancel flag *and* SIGINT. The agent alive-check
        // consults both (`cancel_requested` and `interrupted`). Classic
        // linenoise only had SIGINT; if the NDJSON cancel is delayed or
        // missed, the signal still aborts the active Metal quantum.
        #[cfg(unix)]
        {
            let pid = self.child.id();
            if pid > 0 {
                unsafe {
                    // libc not required as a crate dep — raw syscall.
                    extern "C" {
                        fn kill(pid: i32, sig: i32) -> i32;
                    }
                    const SIGINT: i32 = 2;
                    let _ = kill(pid as i32, SIGINT);
                }
            }
        }
        send_result
    }

    pub fn quit(&mut self) -> Result<(), BackendError> {
        self.send(&ClientMessage::quit())
    }

    pub fn save(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::save(&id))?;
        Ok(id)
    }

    pub fn compact(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::compact(&id))?;
        Ok(id)
    }

    pub fn new_session(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::new_session(&id))?;
        Ok(id)
    }

    pub fn session(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::session(&id))?;
        Ok(id)
    }

    pub fn help(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::help(&id))?;
        Ok(id)
    }

    pub fn queue_clear(&mut self) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::queue_clear(&id))?;
        Ok(id)
    }

    pub fn tool_read(&mut self, path: &str) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::tool_read(&id, path))?;
        Ok(id)
    }

    pub fn tool_search(&mut self, path: &str, needle: &str) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::tool_search(&id, path, needle))?;
        Ok(id)
    }

    pub fn tool_shell(&mut self, command: &str) -> Result<String, BackendError> {
        let id = self.next_req_id();
        self.send(&ClientMessage::tool_shell(&id, command))?;
        Ok(id)
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.quit();
        // Give the child a moment to emit bye; then force-kill.
        let _ = self.child.try_wait();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Walk from the agent binary and CWD looking for `src/metal/q27_kernels.metal`.
fn discover_metal_source(agent_bin: &Path) -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Some(parent) = agent_bin.parent() {
        // build/q27-agent → repo root is parent of build/
        roots.push(parent.to_path_buf());
        if let Some(grand) = parent.parent() {
            roots.push(grand.to_path_buf());
        }
    }
    for root in roots {
        let mut dir = root;
        for _ in 0..8 {
            let candidate = dir.join("src/metal/q27_kernels.metal");
            if candidate.is_file() {
                return Some(candidate);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    None
}

fn spawn_stdout_reader<R: std::io::Read + Send + 'static>(stdout: R, tx: Sender<BackendEvent>) {
    thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    if l.trim().is_empty() {
                        continue;
                    }
                    match ServerEvent::parse_line(&l) {
                        Ok(ev) => {
                            if tx.send(BackendEvent::Server(ev)).is_err() {
                                break;
                            }
                        }
                        Err(_) => {
                            if tx.send(BackendEvent::BadLine(l)).is_err() {
                                break;
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(BackendEvent::Exited(None));
    });
}
