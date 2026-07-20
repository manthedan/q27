#!/usr/bin/env python3
"""Experimental local installer/proxy for q27 Metal harness-prefix snapshots."""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import sys
import tempfile
import threading
import urllib.parse
import urllib.request

ELIGIBLE = {
    "/v1/chat/completions": "chat_completions",
    "/v1/responses": "responses",
    "/v1/messages": "messages",
}
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


def target_url(value: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise argparse.ArgumentTypeError("target must be a loopback http:// URL")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("target must not contain a query or fragment")
    return parsed


def listen_address(value: str) -> str:
    if value not in {"127.0.0.1", "localhost"}:
        raise argparse.ArgumentTypeError("listen address must be loopback (127.0.0.1 or localhost)")
    return value


def secret_from_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"set {name}")
    if "\r" in value or "\n" in value:
        raise SystemExit(f"{name} must not contain newlines")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req, _fp, _code, _msg, _headers, _newurl):
        return None


def prewarm(target: urllib.parse.ParseResult, token: str, api: str, request: dict) -> dict:
    base = target.path.rstrip("/")
    url = urllib.parse.urlunparse((
        target.scheme, target.netloc,
        base + "/experimental/prefix-cache/prewarm", "", "", "",
    ))
    data = json.dumps({"api": api, "request": request}, separators=(",", ":")).encode()
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Q27-Admin-Token": token,
        },
    )
    # Never honor HTTP_PROXY/macOS proxy settings for a credential-bearing
    # loopback admin request. target_url() already rejects non-loopback URLs;
    # this keeps transport routing aligned with that security boundary.
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=3600) as response:
            result = json.load(response)
    except Exception as exc:
        if hasattr(exc, "read"):
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"prewarm failed: {detail}") from exc
        raise RuntimeError(f"prewarm failed: {exc}") from exc
    if result.get("status") != "ok":
        raise RuntimeError(f"prewarm failed: {result}")
    return result


def load_request(path: str) -> dict:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise SystemExit("request JSON must be an object")
    return value


def command_prewarm(args: argparse.Namespace) -> int:
    payload = load_request(args.request)
    # A --dump-request file is an envelope {"api": ..., "request": ...}; consume
    # it directly instead of re-wrapping it as a bare request body. --api may be
    # omitted when the envelope supplies the api; an explicit --api overrides it.
    api = args.api
    if "request" in payload and isinstance(payload.get("request"), dict) and "api" in payload:
        if api is None:
            api = payload["api"]
        payload = payload["request"]
    if api is None:
        raise SystemExit("--api is required when the request file is a bare request body "
                         "(not a --dump-request envelope)")
    result = prewarm(args.target, secret_from_env("Q27_METAL_ADMIN_TOKEN"), api, payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def sse_error_payload(message: str, api: str | None) -> bytes:
    if api == "responses":
        error = {"type": "error",
                 "error": {"type": "api_error", "message": message}}
        failed = {"type": "response.failed",
                  "response": {"id": "resp_q27_prefix_proxy",
                               "object": "response", "status": "failed",
                               "last_error": {"code": "server_error",
                                              "message": message},
                               "output": []}}
        return (f"data: {json.dumps(error, separators=(',', ':'))}\n\n"
                f"data: {json.dumps(failed, separators=(',', ':'))}\n\n").encode()
    if api in ("messages", "anthropic"):
        # Anthropic /v1/messages streams named events: the error rides an
        # "event: error" frame whose body is the real API's error envelope.
        body = {"type": "error",
                "error": {"type": "api_error", "message": message}}
        return (f"event: error\ndata: {json.dumps(body, separators=(',', ':'))}\n\n").encode()
    error = {"error": {"message": message, "type": "api_error"}}
    return (f"data: {json.dumps(error, separators=(',', ':'))}\n\n"
            "data: [DONE]\n\n").encode()


class ProxyState:
    def __init__(self, target: urllib.parse.ParseResult, admin_token: str,
                 capture_token: str, dump_request: str | None = None):
        self.target = target
        self.admin_token = admin_token
        self.capture_token = capture_token
        self.dump_request = dump_request
        self.lock = threading.Lock()
        self.warmed = False

    def maybe_dump(self, api: str, request: dict) -> None:
        # Write the exact captured request (with its api tag) so it can be
        # re-installed out-of-band via the direct `prewarm` subcommand — the
        # robust path for large prompts whose cold prewarm outlives a harness
        # HTTP timeout. Local-only file; it is the harness's own request.
        if not self.dump_request:
            return
        tmp_path = None
        try:
            # The dump holds the full harness prompt, tool schemas, and
            # project/user content, so it must never be readable by another
            # local user. Write a FRESHLY created 0600 temp file in the target's
            # directory and atomically rename it over the destination: this
            # refuses to write through a pre-existing symlink and, crucially,
            # means no other process can hold an already-open readable fd to the
            # destination while new private content lands (the failure mode of
            # open(O_TRUNC)+chmod on an existing permissive file).
            directory = os.path.dirname(os.path.abspath(self.dump_request)) or "."
            fd, tmp_path = tempfile.mkstemp(prefix=".q27-dump-", dir=directory)
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"api": api, "request": request}, stream, indent=2, sort_keys=True)
            os.replace(tmp_path, self.dump_request)
            tmp_path = None
            print(f"q27 experimental prefix: captured request dumped to {self.dump_request}",
                  file=sys.stderr, flush=True)
        except OSError as exc:
            print(f"q27 experimental prefix: dump-request write failed: {exc}",
                  file=sys.stderr, flush=True)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


class PrefixProxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "q27-experimental-prefix-proxy/1"

    @property
    def state(self) -> ProxyState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        self._forward(b"")

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("Transfer-Encoding"):
            self._error(411, "chunked request bodies are not supported by the experimental proxy")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        if length < 0 or length > 64 * 1024 * 1024:
            self._error(413, "request body is too large")
            return
        body = self.rfile.read(length)
        path = urllib.parse.urlsplit(self.path).path
        api = ELIGIBLE.get(path)
        downstream_started = False
        if api:
            bearer = self.headers.get("Authorization", "")
            capture = self.headers.get("X-Q27-Prefix-Capture-Token", "")
            # Anthropic clients (Claude Code) authenticate with x-api-key, not
            # Authorization: Bearer. Accept it as an equivalent capture credential.
            api_key = self.headers.get("x-api-key", "")
            if capture != self.state.capture_token and \
               bearer != "Bearer " + self.state.capture_token and \
               api_key != self.state.capture_token:
                self._error(403, "invalid experimental prefix capture credential")
                return
            try:
                request = json.loads(body)
                if not isinstance(request, dict):
                    raise ValueError("request JSON is not an object")
            except Exception as exc:
                self._error(400, str(exc))
                return
            with self.state.lock:
                if not self.state.warmed:
                    self.state.maybe_dump(api, request)
                    if request.get("stream") is True:
                        downstream_started = True
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        result_box: list[object] = []
                        done = threading.Event()

                        def install() -> None:
                            try:
                                result_box.append(prewarm(
                                    self.state.target, self.state.admin_token, api, request))
                            except Exception as exc:  # reported on the SSE stream below
                                result_box.append(exc)
                            finally:
                                done.set()

                        threading.Thread(target=install, daemon=True).start()
                        client_alive = True
                        while not done.wait(1.0):
                            if client_alive:
                                try:
                                    self.wfile.write(b": q27 prefix prewarm in progress\n\n")
                                    self.wfile.flush()
                                except (BrokenPipeError, ConnectionResetError):
                                    # Finish the explicitly requested install even
                                    # if the harness gives up; do not forward later.
                                    client_alive = False
                        result = result_box[0]
                        if isinstance(result, Exception):
                            if client_alive:
                                self._sse_error(str(result), api)
                            self.close_connection = True
                            return
                        if not client_alive:
                            self.state.warmed = True
                            self._report_installed(result)
                            self.close_connection = True
                            return
                    else:
                        try:
                            result = prewarm(self.state.target, self.state.admin_token, api, request)
                        except Exception as exc:
                            self._error(502, str(exc))
                            return
                    self.state.warmed = True
                    self._report_installed(result)
        self._forward(body, downstream_started=downstream_started, api=api)

    def _report_installed(self, result: object) -> None:
        print("q27 experimental prefix installed: " +
              json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)

    def _sse_error(self, message: str, api: str | None) -> None:
        self.wfile.write(sse_error_payload(message, api))
        self.wfile.flush()

    def _forward(self, body: bytes, downstream_started: bool = False,
                 api: str | None = None) -> None:
        target = self.state.target
        base = target.path.rstrip("/")
        path = base + self.path
        connection = http.client.HTTPConnection(target.hostname, target.port or 80, timeout=3600)
        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS | {
                "host", "content-length", "authorization", "x-q27-admin-token",
                "x-q27-prefix-capture-token", "x-api-key",
            }
        }
        headers["Host"] = target.netloc
        headers["Connection"] = "close"
        if body:
            headers["Content-Length"] = str(len(body))
        response_started = downstream_started
        try:
            connection.request(self.command, path, body=body or None, headers=headers)
            upstream = connection.getresponse()
            if downstream_started:
                if upstream.status < 200 or upstream.status >= 300:
                    detail = upstream.read().decode("utf-8", "replace")
                    self._sse_error(
                        f"upstream returned HTTP {upstream.status}: {detail}", api)
                    return
            else:
                self.send_response(upstream.status, upstream.reason)
                response_started = True
                for key, value in upstream.getheaders():
                    if key.lower() not in HOP_HEADERS | {"content-length"}:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
            while True:
                # HTTPResponse.read(n) waits for n bytes or EOF and therefore
                # destroys SSE latency. read1() returns currently available
                # decoded data (at most one underlying/chunked read).
                chunk = upstream.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if not response_started and not self.wfile.closed:
                self._error(502, f"upstream request failed: {exc}")
            else:
                print(f"upstream response interrupted: {exc}", file=sys.stderr, flush=True)
        finally:
            self.close_connection = True
            connection.close()

    def _error(self, status: int, message: str) -> None:
        data = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True


def command_proxy(args: argparse.Namespace) -> int:
    state = ProxyState(args.target, secret_from_env("Q27_METAL_ADMIN_TOKEN"),
                       secret_from_env("Q27_PREFIX_CAPTURE_TOKEN"),
                       dump_request=args.dump_request)
    server = http.server.ThreadingHTTPServer((args.listen, args.port), PrefixProxy)
    server.state = state  # type: ignore[attr-defined]
    print(f"q27 experimental prefix proxy: http://{args.listen}:{args.port}", file=sys.stderr)
    print(f"forwarding to {urllib.parse.urlunparse(args.target)}", file=sys.stderr)
    print("Run one fresh Pi or Codex session through this URL, then stop the proxy.", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    direct = sub.add_parser("prewarm", help="prewarm from a captured request JSON object")
    direct.add_argument("--target", type=target_url, default=target_url("http://127.0.0.1:8080"))
    direct.add_argument("--api", choices=("chat_completions", "responses", "messages", "anthropic"),
                        default=None,
                        help="api of the request; optional when the file is a "
                             "--dump-request envelope that carries its own api")
    direct.add_argument("request", help="request JSON file, or - for stdin")
    direct.set_defaults(func=command_prewarm)

    proxy = sub.add_parser("proxy", help="capture and prewarm the first live harness request")
    proxy.add_argument("--target", type=target_url, default=target_url("http://127.0.0.1:8080"))
    proxy.add_argument("--listen", type=listen_address, default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8081)
    proxy.add_argument("--dump-request", metavar="PATH",
                       help="also write the first captured request JSON to PATH "
                            "(for out-of-band `prewarm` re-install)")
    proxy.set_defaults(func=command_proxy)
    return result


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "port", 1) < 1 or getattr(args, "port", 1) > 65535:
        raise SystemExit("--port must be 1..65535")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
