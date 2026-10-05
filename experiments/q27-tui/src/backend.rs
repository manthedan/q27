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
    #[allow(dead_code)] // retained for explicit detached-backend reporting
    #[error("stdin closed")]
    StdinClosed,
}

pub enum BackendEvent {
    Server(ServerEvent),
    /// Non-JSON line or parse error (usually should not happen in FP1).
    BadLine(String),
    /// Child stderr line (diagnostics).
    #[allow(dead_code)] // the reader persists stderr today; event routing is reserved
    Stderr(String),
    /// Child exited.
    Exited(Option<i32>),
}

pub struct Backend {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<BackendEvent>,
    #[allow(dead_code)] // exposed for diagnostics and future attachment UIs
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
        // Exclusive + no-follow, owner-only: a predictable temp path opened
        // with File::create would let a local process pre-create or race a
        // symlink (codex P2 r1), and default permissions would leak agent
        // stderr (paths, tool diagnostics) to other local users (r15 P2).
        let stderr_file = {
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            opts.open(&stderr_log)?
        };

        let mut cmd = Command::new(agent_bin);
        cmd.arg(model.as_ref())
            .arg(tokenizer.as_ref())
            .arg("--frontend-proto")
            .arg("1")
            .args(extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr_file));

        // Metal finds q27_kernels.metal next to its executable in release and
        // Homebrew layouts. A dev build (build/q27-agent) gets its checkout's
        // shader injected here, keyed off the agent binary, never the cwd.
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
        // FP1 cancel only: the agent checks the flag in every alive() and
        // treats an idle cancel as a no-op. The dual-path SIGINT raced the
        // turn boundary — arriving after the turn finished but before we saw
        // the terminal, it latched the global interrupt and killed the
        // session with bye:error (r15 codex P1). A wedged backend is what
        // quit/kill is for.
        self.send(&ClientMessage::cancel(id))
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

/// Walk up from the agent binary looking for `src/metal/q27_kernels.metal`.
/// The cwd is deliberately not searched: an installed agent launched inside
/// some other q27 checkout must keep its own packaged shader.
fn discover_metal_source(agent_bin: &Path) -> Option<PathBuf> {
    let mut roots = Vec::new();
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

#[cfg(test)]
mod tests {
    use super::discover_metal_source;
    use std::path::Path;

    #[test]
    fn packaged_agent_ignores_a_checkout_in_cwd() {
        // cargo runs tests inside this repository, so a cwd walk would find
        // its src/metal shader; an agent outside any checkout must not.
        let dir = std::env::temp_dir().join(format!("q27-tui-pkg-{}", std::process::id()));
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let found = discover_metal_source(&bin.join("q27-agent"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(found, None);
    }

    #[test]
    fn dev_agent_finds_its_checkout_shader() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let found = discover_metal_source(&repo.join("build/q27-agent")).unwrap();
        assert!(found.ends_with("src/metal/q27_kernels.metal"));
    }
}
